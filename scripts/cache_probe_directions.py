"""
Compute and cache contrastive linear probe directions for evaluation awareness.

For each prompt pair (positive = "yes, I know I'm being evaluated" /
negative = "no, I don't know"), runs both through the model and records
the last-token hidden state at every decoder layer.  The mean-difference
vector (mean_pos − mean_neg) is L2-normalised and saved per layer.

Usage:
    python scripts/cache_probe_directions.py \
        --model_name_or_path allenai/OLMo-2-1124-7B-Instruct \
        --output_path /tmp/probe_directions.pt

The output .pt file has the structure:
    {
        "directions": {layer_idx: tensor(hidden_dim,), ...},   # unit-norm
        "metadata":   {"model": str, "n_prompts": int, "n_layers": int, "hidden_dim": int},
    }
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct import logger_utils
from open_instruct.dataset_transformation import CHAT_TEMPLATES

logger = logger_utils.setup_logger(__name__)

_DEFAULT_PROMPTS_JSON = os.path.join(os.path.dirname(__file__), "data", "awareness_probe_prompts.json")


def load_probe_prompts(path: str) -> list[dict]:
    with open(path) as f:
        prompts = json.load(f)
    for item in prompts:
        for key in ("question", "positive", "negative"):
            if key not in item:
                raise ValueError(f"Probe prompt missing key '{key}': {item}")
    return prompts


def _tokenize_messages(tokenizer, question: str, choice: str) -> list[int]:
    """Format as a complete two-turn conversation and tokenise."""
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": choice}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_dict=False)


def _last_real_token_hidden(hidden_states: tuple, attention_mask: torch.Tensor) -> list[torch.Tensor]:
    """Extract the hidden state at the last non-padding token for each example.

    Returns a list of length n_layers, each a tensor of shape (batch, hidden_dim).
    hidden_states[0] is the embedding layer; hidden_states[1:] are decoder layers.
    """
    batch_size = attention_mask.shape[0]
    # Index of the last real (non-pad) token per example
    last_indices = attention_mask.long().argmax(dim=1) + attention_mask.sum(dim=1) - 1
    # Clamp to valid range
    seq_len = attention_mask.shape[1]
    last_indices = last_indices.clamp(0, seq_len - 1)

    result = []
    for hs in hidden_states[1:]:  # skip embedding layer
        # hs: (batch, seq_len, hidden_dim)
        selected = hs[torch.arange(batch_size, device=hs.device), last_indices]
        result.append(selected.float().detach().cpu())
    return result


@torch.no_grad()
def compute_probe_directions(model, tokenizer, prompts: list[dict], batch_size: int) -> dict[int, torch.Tensor]:
    """Return a dict of layer_idx → unit-norm probe direction tensor."""
    pos_hiddens: dict[int, list[torch.Tensor]] = {}
    neg_hiddens: dict[int, list[torch.Tensor]] = {}

    def _process_batch(batch_token_ids: list[list[int]], store: dict[int, list]):
        # Left-pad so all sequences in the batch have the same length.
        max_len = max(len(ids) for ids in batch_token_ids)
        pad_id = tokenizer.pad_token_id or 0
        input_ids_padded = torch.tensor(
            [[pad_id] * (max_len - len(ids)) + ids for ids in batch_token_ids], dtype=torch.long
        )
        attention_mask = (input_ids_padded != pad_id).long()
        input_ids_padded = input_ids_padded.to(model.device)
        attention_mask = attention_mask.to(model.device)

        outputs = model(input_ids=input_ids_padded, attention_mask=attention_mask, output_hidden_states=True)
        # List of (batch, hidden_dim) tensors, one per decoder layer
        per_layer = _last_real_token_hidden(outputs.hidden_states, attention_mask.cpu())
        for layer_idx, h in enumerate(per_layer):
            if layer_idx not in store:
                store[layer_idx] = []
            # h: (batch, hidden_dim) — split into per-example tensors
            for example_h in h.unbind(dim=0):
                store[layer_idx].append(example_h)

    for polarity, store in [("positive", pos_hiddens), ("negative", neg_hiddens)]:
        logger.info(f"Extracting {polarity} activations for {len(prompts)} prompts...")
        batch_ids: list[list[int]] = []
        for i, item in enumerate(prompts):
            choice = item["positive"] if polarity == "positive" else item["negative"]
            batch_ids.append(_tokenize_messages(tokenizer, item["question"], choice))
            if len(batch_ids) == batch_size or i == len(prompts) - 1:
                _process_batch(batch_ids, store)
                batch_ids = []
        logger.info(f"  Done. Collected activations for {len(store)} layers.")

    # Compute mean-difference direction per layer and normalise
    layer_indices = sorted(pos_hiddens.keys())
    directions: dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        pos_mean = torch.stack(pos_hiddens[layer_idx]).mean(dim=0)
        neg_mean = torch.stack(neg_hiddens[layer_idx]).mean(dim=0)
        direction = pos_mean - neg_mean
        norm = direction.norm()
        if norm > 0:
            direction = direction / norm
        directions[layer_idx] = direction

    return directions


def _sanity_check(directions: dict[int, torch.Tensor], pos_hiddens: dict, neg_hiddens: dict) -> None:
    """Log mean probe score for positive vs negative to confirm the direction makes sense."""
    middle_layer = sorted(directions.keys())[len(directions) // 2]
    d = directions[middle_layer]
    pos_scores = [torch.dot(h, d).item() for h in pos_hiddens[middle_layer]]
    neg_scores = [torch.dot(h, d).item() for h in neg_hiddens[middle_layer]]
    pos_mean = sum(pos_scores) / len(pos_scores)
    neg_mean = sum(neg_scores) / len(neg_scores)
    logger.info(
        f"Sanity check (layer {middle_layer}): "
        f"pos mean score = {pos_mean:.4f}, neg mean score = {neg_mean:.4f}, "
        f"gap = {pos_mean - neg_mean:.4f} (should be > 0)"
    )


def main(args: argparse.Namespace) -> None:
    prompts = load_probe_prompts(args.probe_prompts_json)
    logger.info(f"Loaded {len(prompts)} probe prompts from {args.probe_prompts_json}")

    logger.info(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.chat_template_name and args.chat_template_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[args.chat_template_name]
        logger.info(f"Applied chat template: {args.chat_template_name}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Run extraction with separate stores for sanity check
    pos_hiddens: dict[int, list[torch.Tensor]] = {}
    neg_hiddens: dict[int, list[torch.Tensor]] = {}

    @torch.no_grad()
    def _process_batch_into(batch_token_ids, store):
        max_len = max(len(ids) for ids in batch_token_ids)
        pad_id = tokenizer.pad_token_id or 0
        input_ids_padded = torch.tensor(
            [[pad_id] * (max_len - len(ids)) + ids for ids in batch_token_ids], dtype=torch.long
        )
        attention_mask = (input_ids_padded != pad_id).long()
        outputs = model(
            input_ids=input_ids_padded.to(model.device),
            attention_mask=attention_mask.to(model.device),
            output_hidden_states=True,
        )
        per_layer = _last_real_token_hidden(outputs.hidden_states, attention_mask)
        for layer_idx, h in enumerate(per_layer):
            if layer_idx not in store:
                store[layer_idx] = []
            for example_h in h.unbind(dim=0):
                store[layer_idx].append(example_h)

    for polarity, store in [("positive", pos_hiddens), ("negative", neg_hiddens)]:
        logger.info(f"Extracting {polarity} activations ({len(prompts)} prompts, batch_size={args.batch_size})...")
        batch_ids: list[list[int]] = []
        for i, item in enumerate(prompts):
            choice = item["positive"] if polarity == "positive" else item["negative"]
            batch_ids.append(_tokenize_messages(tokenizer, item["question"], choice))
            if len(batch_ids) == args.batch_size or i == len(prompts) - 1:
                _process_batch_into(batch_ids, store)
                batch_ids = []
        logger.info(f"  {polarity}: collected activations at {len(store)} layers.")

    # Compute mean-difference directions
    layer_indices = sorted(pos_hiddens.keys())
    n_layers = len(layer_indices)
    hidden_dim = pos_hiddens[layer_indices[0]][0].shape[0]
    logger.info(f"Computing probe directions: {n_layers} layers, hidden_dim={hidden_dim}")

    directions: dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        pos_mean = torch.stack(pos_hiddens[layer_idx]).mean(dim=0)
        neg_mean = torch.stack(neg_hiddens[layer_idx]).mean(dim=0)
        direction = pos_mean - neg_mean
        norm = direction.norm()
        if norm > 0:
            direction = direction / norm
        directions[layer_idx] = direction

    _sanity_check(directions, pos_hiddens, neg_hiddens)

    output = {
        "directions": directions,
        "metadata": {
            "model": args.model_name_or_path,
            "n_prompts": len(prompts),
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "chat_template_name": args.chat_template_name,
        },
    }
    torch.save(output, args.output_path)
    logger.info(
        f"Saved probe directions for {n_layers} layers to {args.output_path} "
        f"(hidden_dim={hidden_dim}, n_prompts={len(prompts)})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute contrastive linear probe directions for evaluation awareness."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--probe_prompts_json",
        type=str,
        default=_DEFAULT_PROMPTS_JSON,
        help="Path to JSON file with probe prompts (list of {question, positive, negative}).",
    )
    parser.add_argument(
        "--chat_template_name",
        type=str,
        default="olmo_thinker",
        help="Chat template name from CHAT_TEMPLATES (e.g. olmo_thinker). "
        "Leave empty to use the tokenizer's default template.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    main(parser.parse_args())
