"""
Evaluate the evaluation-awareness linear probe on the jjpn2/eval_awareness dataset.

The dataset contains real transcripts labelled eval (metadata.eval_category=true)
vs. organic/deployment (false).  For each transcript we prefill the model, take
the mean hidden state over all token positions, project onto the cached probe
direction, and compute AUC-ROC per layer.

This is the proper out-of-distribution test: the probe is fit on synthetic
positive/negative prompts but evaluated on real transcripts where eval awareness
is not explicitly stated in the text.

Dataset setup:
    git clone https://huggingface.co/datasets/jjpn2/eval_awareness
    cd eval_awareness && ./scripts/decrypt.sh   # produces dataset.json

Usage:
    python scripts/eval_probe_on_eval_awareness.py \\
        --model_name_or_path allenai/OLMo-2-1124-7B-Instruct \\
        --probe_directions_path /tmp/probe_directions.pt \\
        --dataset_json /path/to/eval_awareness/dataset.json \\
        [--probe_layer_indices 15,31] \\
        [--max_examples 500] \\
        [--max_tokens 2048] \\
        [--pooling last|mean]
"""

import argparse
import json

import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct import logger_utils
from open_instruct.dataset_transformation import CHAT_TEMPLATES

logger = logger_utils.setup_logger(__name__)


def load_dataset(path: str, max_examples: int | None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    eval_count = sum(1 for x in data if x["metadata"]["eval_category"])
    organic_count = len(data) - eval_count
    logger.info(f"Loaded {len(data)} examples ({eval_count} eval, {organic_count} organic) from {path}")
    if max_examples is not None:
        # Subsample while preserving balance
        evals = [x for x in data if x["metadata"]["eval_category"]][:max_examples // 2]
        organics = [x for x in data if not x["metadata"]["eval_category"]][:max_examples // 2]
        data = evals + organics
        logger.info(f"Subsampled to {len(data)} ({len(evals)} eval, {len(organics)} organic)")
    return data


def _messages_to_token_ids(item: dict, tokenizer) -> list[int]:
    """Convert a dataset item's input to token IDs.

    item["input"] is either:
      - a list of {role, content} dicts  (chat format)
      - a plain string
    """
    raw = item["input"]
    if isinstance(raw, list):
        # Normalize role names: inspect_ai uses "user"/"assistant"/"system"
        messages = [{"role": m["role"], "content": m["content"]} for m in raw]
        try:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_dict=False)
        except Exception:
            # Fallback: concatenate as plain text
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return tokenizer.encode(text)
    else:
        return tokenizer.encode(str(raw))


@torch.no_grad()
def extract_hidden_states(
    token_ids: list[int], model, max_tokens: int, pooling: str
) -> list[torch.Tensor]:
    """Run a forward pass and return one vector per decoder layer.

    pooling="mean" → mean over all positions
    pooling="last" → last token position only
    """
    if len(token_ids) > max_tokens:
        token_ids = token_ids[-max_tokens:]  # Keep the tail (most recent context)
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    outputs = model(
        input_ids=input_tensor,
        attention_mask=torch.ones_like(input_tensor),
        output_hidden_states=True,
    )
    result = []
    for hs in outputs.hidden_states[1:]:  # skip embedding layer; hs: (1, seq_len, hidden)
        h = hs[0].float().detach().cpu()  # (seq_len, hidden)
        if pooling == "last":
            result.append(h[-2])
        else:
            result.append(h.mean(dim=0))
    return result


def main(args: argparse.Namespace) -> None:
    logger.info(f"Loading probe directions from {args.probe_directions_path}...")
    probe_data = torch.load(args.probe_directions_path, weights_only=True)
    all_directions: dict[int, torch.Tensor] = probe_data["directions"]
    meta = probe_data.get("metadata", {})
    logger.info(
        f"  Loaded directions for {len(all_directions)} layers "
        f"(model={meta.get('model', 'unknown')}, hidden_dim={meta.get('hidden_dim', '?')})"
    )

    if args.probe_layer_indices:
        layer_indices = [int(x) for x in args.probe_layer_indices.split(",")]
        directions = {k: all_directions[k] for k in layer_indices if k in all_directions}
    else:
        sorted_layers = sorted(all_directions.keys())
        cutoff = len(sorted_layers) * 2 // 3
        directions = {k: all_directions[k] for k in sorted_layers[cutoff:]}
    logger.info(f"  Using layers: {sorted(directions)}")

    data = load_dataset(args.dataset_json, args.max_examples)

    logger.info(f"Loading tokenizer + model from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.chat_template_name and args.chat_template_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[args.chat_template_name]
        logger.info(f"Applied chat template: {args.chat_template_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Collect probe scores and labels
    scores_by_layer: dict[int, list[float]] = {lk: [] for lk in directions}
    labels: list[int] = []

    for i, item in enumerate(data):
        label = int(item["metadata"]["eval_category"])
        try:
            token_ids = _messages_to_token_ids(item, tokenizer)
        except Exception as e:
            logger.warning(f"Skipping item {i}: tokenization failed — {e}")
            continue

        per_layer = extract_hidden_states(token_ids, model, args.max_tokens, args.pooling)
        for lk, direction in directions.items():
            if lk < len(per_layer):
                score = torch.dot(per_layer[lk], direction.float()).item()
                scores_by_layer[lk].append(score)

        labels.append(label)
        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(data)} examples...")

    logger.info(f"Scored {len(labels)} examples ({sum(labels)} eval, {len(labels) - sum(labels)} organic)")

    # Compute AUC-ROC per layer
    print("\n" + "=" * 60)
    print(f"AUC-ROC  (pooling={args.pooling}, n={len(labels)})")
    print(f"Label: eval_category=True → positive class")
    print("=" * 60)
    print(f"  {'Layer':>6}  {'AUC-ROC':>8}  {'Mean gap':>10}")
    print("-" * 60)

    for lk in sorted(directions):
        s = scores_by_layer[lk]
        if len(s) < len(labels):
            logger.warning(f"Layer {lk}: only {len(s)} scores (expected {len(labels)}), skipping")
            continue
        auc = roc_auc_score(labels, s)
        eval_scores = [sc for sc, lb in zip(s, labels) if lb == 1]
        organic_scores = [sc for sc, lb in zip(s, labels) if lb == 0]
        gap = sum(eval_scores) / len(eval_scores) - sum(organic_scores) / len(organic_scores)
        print(f"  {lk:>6}  {auc:>8.4f}  {gap:>+10.4f}")

    print("=" * 60)

    # Random probe control
    torch.manual_seed(99)
    random_directions = {
        lk: torch.nn.functional.normalize(torch.randn_like(v), dim=0)
        for lk, v in directions.items()
    }
    rand_scores_by_layer: dict[int, list[float]] = {lk: [] for lk in random_directions}

    logger.info("Scoring with random probe directions (control)...")
    for i, item in enumerate(data):
        try:
            token_ids = _messages_to_token_ids(item, tokenizer)
        except Exception:
            continue
        per_layer = extract_hidden_states(token_ids, model, args.max_tokens, args.pooling)
        for lk, direction in random_directions.items():
            if lk < len(per_layer):
                rand_scores_by_layer[lk].append(torch.dot(per_layer[lk], direction.float()).item())

    print("\n" + "=" * 60)
    print("RANDOM PROBE CONTROL (expected AUC ≈ 0.50)")
    print("=" * 60)
    print(f"  {'Layer':>6}  {'AUC-ROC':>8}")
    print("-" * 60)
    for lk in sorted(random_directions):
        s = rand_scores_by_layer[lk]
        if len(s) < len(labels):
            continue
        auc = roc_auc_score(labels, s)
        print(f"  {lk:>6}  {auc:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate eval-awareness probe via AUC-ROC on real eval vs. organic transcripts."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--probe_directions_path", type=str, required=True)
    parser.add_argument(
        "--dataset_json",
        type=str,
        required=True,
        help="Path to the decrypted dataset.json from jjpn2/eval_awareness.",
    )
    parser.add_argument(
        "--probe_layer_indices",
        type=str,
        default=None,
        help="Comma-separated layer indices (e.g. '15,31'). Default: last third of layers.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Cap total examples (balanced: half eval, half organic). Default: use all.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Truncate transcripts to this many tokens (tail-truncation). Default: 2048.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "last"],
        help="How to pool hidden states over token positions. Default: mean.",
    )
    parser.add_argument("--chat_template_name", type=str, default="olmo_thinker")
    main(parser.parse_args())
