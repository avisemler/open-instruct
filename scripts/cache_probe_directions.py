"""
Compute and cache contrastive linear probe directions for evaluation awareness.

For each prompt pair (positive = "yes, I know I'm being evaluated" /
negative = "no, I don't know"), runs both through the model and records
the last-token hidden state at every decoder layer.  The mean-difference
vector (mean_pos − mean_neg) is L2-normalised and saved per layer.

Default behaviour (no args): fetches all HuggingFace branches of
allenai/Olmo-3-32B-Think, computes probes for every revision, and writes
a cosine-similarity matrix to probe_cache/allenai_Olmo-3-32B-Think/similarity_matrix.png.

Single-model usage:
    python scripts/cache_probe_directions.py \
        --model_name_or_path allenai/OLMo-2-1124-7B-Instruct \
        --output_path /tmp/probe_directions.pt

Multi-revision usage (explicit revisions):
    python scripts/cache_probe_directions.py \
        --model_name_or_path allenai/Olmo-3-32B-Think \
        --model_revisions main step1000 step2000 step3000 \
        --output_dir /tmp/probe_revisions/ \
        --plot_output_path /tmp/probe_similarity_matrix.png

Multi-revision usage (auto-fetch all branches):
    python scripts/cache_probe_directions.py \
        --model_name_or_path allenai/Olmo-3-32B-Think

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

_DEFAULT_MODEL = "allenai/Olmo-3-32B-Think"
_HF_CACHE_DIR = "/workspace/.cache/huggingface"


def _fetch_all_revisions(model_name: str) -> list[str]:
    """Return all branch names for a HuggingFace repo, sorted lexicographically."""
    from huggingface_hub import list_repo_refs

    out = list_repo_refs(model_name)
    branches = sorted(b.name for b in out.branches)
    logger.info(f"Auto-fetched {len(branches)} branches from {model_name}: {branches}")
    return branches


def _default_output_dir(model_name: str, mode: str) -> str:
    slug = model_name.replace("/", "_") + f"_{mode}"
    return os.path.join("probe_cache", slug)


def _clear_hf_cache() -> None:
    """Delete the HuggingFace model cache to reclaim disk space between revisions."""
    import shutil

    if os.path.isdir(_HF_CACHE_DIR):
        shutil.rmtree(_HF_CACHE_DIR)
        logger.info(f"Cleared HuggingFace cache at {_HF_CACHE_DIR}")
    else:
        logger.info(f"HuggingFace cache directory not found, skipping: {_HF_CACHE_DIR}")


def load_probe_prompts(path: str) -> list[dict]:
    with open(path) as f:
        prompts = json.load(f)
    for item in prompts:
        for key in ("positive", "negative"):
            if key not in item:
                raise ValueError(f"Probe prompt missing key '{key}': {item}")
    return prompts


def _tokenize_messages(tokenizer, question: str, choice: str) -> list[int]:
    """Format as a complete two-turn conversation and tokenise."""
    if question:
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": choice}]
    else:
        messages = [{"role": "user", "content": choice,}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_dict=False)


@torch.no_grad()
def _extract_answer_hidden(token_ids: list[int], model, index: int) -> list[torch.Tensor]:
    """Hidden state at the answer token (position -2, just before EOS) per decoder layer.

    Returns a list of length n_layers, each shape (hidden_dim,).
    The chat template always ends with <answer><EOS>, so -2 is reliably the answer token.
    """
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    outputs = model(input_ids=input_tensor, attention_mask=torch.ones_like(input_tensor), output_hidden_states=True)
    # print the value of the token at index
    return [hs[0, index, :].float().detach().cpu() for hs in outputs.hidden_states[1:]]


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


def compute_probe_directions(
    model,
    tokenizer,
    prompts: list[dict],
    model_name: str,
    chat_template_name: str,
    mode: str,
) -> dict:
    """Run the probe extraction for a loaded model and return the output dict."""
    pos_hiddens: dict[int, list[torch.Tensor]] = {}
    neg_hiddens: dict[int, list[torch.Tensor]] = {}

    for polarity, store in [("positive", pos_hiddens), ("negative", neg_hiddens)]:
        logger.info(f"Extracting {polarity} activations ({len(prompts)} prompts)...")
        for item in prompts:
            distinct_input = item["positive"] if polarity == "positive" else item["negative"]
            if mode == "hua":
                token_ids = _tokenize_messages(tokenizer, "", distinct_input)
            elif mode == "nguyen":
                token_ids = _tokenize_messages(tokenizer, item["question"], distinct_input)

            index = -3 if mode=="hua" else -3
            print("Decoded token at index:", tokenizer.decode(token_ids[index]))
            per_layer = _extract_answer_hidden(token_ids, model, index)
            for layer_idx, h in enumerate(per_layer):
                if layer_idx not in store:
                    store[layer_idx] = []
                store[layer_idx].append(h)
        logger.info(f"  {polarity}: collected activations at {len(store)} layers.")

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

    return {
        "directions": directions,
        "metadata": {
            "model": model_name,
            "n_prompts": len(prompts),
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "chat_template_name": chat_template_name,
        },
    }


def plot_cosine_similarity_matrix(
    all_directions: dict[str, dict[int, torch.Tensor]],
    plot_output_path: str,
    n_plot_layers: int = 8,
) -> None:
    import math

    import matplotlib.pyplot as plt

    revisions = list(all_directions.keys())
    layer_indices = sorted(all_directions[revisions[0]].keys())

    # Pick ~n_plot_layers evenly-spaced layers across the network depth.
    step = max(1, len(layer_indices) // n_plot_layers)
    plot_layers = layer_indices[::step][:n_plot_layers]

    ncols = min(4, len(plot_layers))
    nrows = math.ceil(len(plot_layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    rev_labels = [rev[:12] for rev in revisions]

    # Pre-compute all sim matrices so we can set a shared adaptive scale.
    sims = {}
    for layer in plot_layers:
        V = torch.stack([all_directions[rev][layer] for rev in revisions])
        V_norm = V / V.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sims[layer] = (V_norm @ V_norm.T).numpy()

    all_vals = [sims[l] for l in plot_layers]
    import numpy as np
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))

    for ax, layer in zip(axes_flat, plot_layers):
        sim = sims[layer]
        im = ax.imshow(sim, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax.set_title(f"Layer {layer}")
        ax.set_xticks(range(len(revisions)))
        ax.set_yticks(range(len(revisions)))
        ax.set_xticklabels(rev_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(rev_labels, fontsize=7)
        plt.colorbar(im, ax=ax)

    for ax in axes_flat[len(plot_layers):]:
        ax.set_visible(False)

    plt.suptitle("Probe direction cosine similarity across revisions")
    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved similarity matrix plot to {plot_output_path}")


def main(args: argparse.Namespace) -> None:

    # Set default probe prompts file based on mode
    if args.probe_prompts_json is None:
        if args.mode == "hua":
            args.probe_prompts_json = os.path.join(os.path.dirname(__file__), "data", "awareness_hua_prompts.json")
        elif args.mode == "nguyen":
            args.probe_prompts_json = os.path.join(os.path.dirname(__file__), "data", "awareness_probe_prompts.json")
        else:
            raise ValueError(f"Unknown mode '{args.mode}' for default probe prompts")
    
    prompts = load_probe_prompts(args.probe_prompts_json)
    logger.info(f"Loaded {len(prompts)} probe prompts from {args.probe_prompts_json}")


    revisions = args.model_revisions or _fetch_all_revisions(args.model_name_or_path)
    output_dir = args.output_dir or _default_output_dir(args.model_name_or_path, args.mode)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    all_directions: dict[str, dict[int, torch.Tensor]] = {}
    model = None

    for revision in revisions:
        safe_rev = revision.replace("/", "_")
        cache_path = os.path.join(output_dir, f"probe_directions_{safe_rev}.pt")

        if os.path.exists(cache_path):
            logger.info(f"Loading cached directions for revision '{revision}' from {cache_path}")
            cached = torch.load(cache_path, weights_only=False)
            all_directions[revision] = cached["directions"]
            continue

        # Free GPU memory and HuggingFace disk cache from the previous revision.
        if model is not None:
            del model
            torch.cuda.empty_cache()
            _clear_hf_cache()

        logger.info(f"Loading tokenizer from {args.model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if args.chat_template_name and args.chat_template_name in CHAT_TEMPLATES:
            tokenizer.chat_template = CHAT_TEMPLATES[args.chat_template_name]

        logger.info(f"Loading model at revision '{revision}'...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, revision=revision, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        output = compute_probe_directions(
            model, tokenizer, prompts, args.model_name_or_path, args.chat_template_name, args.mode
        )
        torch.save(output, cache_path)
        logger.info(f"Saved probe directions for revision '{revision}' to {cache_path}")
        all_directions[revision] = output["directions"]

    if model is not None:
        del model
        torch.cuda.empty_cache()

    plot_path = args.plot_output_path or os.path.join(output_dir, "similarity_matrix.png")
    plot_cosine_similarity_matrix(all_directions, plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute contrastive linear probe directions for evaluation awareness."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hua", "nguyen"],
        default="hua",
        help="Which prompt set to use for the probe. Named after first author of the paper."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=_DEFAULT_MODEL,
        help=f"HuggingFace model name or local path. Defaults to {_DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to cache per-revision .pt files. "
        "Defaults to probe_cache/<model_slug>/.",
    )
    parser.add_argument(
        "--model_revisions",
        type=str,
        nargs="+",
        default=None,
        help="One or more HuggingFace revision strings (branch names, tags, commit SHAs). "
        "When omitted, all branches are fetched automatically "
        "via huggingface_hub.list_repo_refs.",
    )
    parser.add_argument(
        "--plot_output_path",
        type=str,
        default=None,
        help="Path to save the cosine-similarity matrix plot (PNG). "
        "Defaults to <output_dir>/similarity_matrix.png when using --model_revisions.",
    )
    parser.add_argument(
        "--probe_prompts_json",
        type=str,
        default=None,
        help="Path to JSON file with probe prompts (list of {question, positive, negative}).",
    )
    parser.add_argument(
        "--chat_template_name",
        type=str,
        default="olmo_thinker",
        help="Chat template name from CHAT_TEMPLATES (e.g. olmo_thinker). "
        "Leave empty to use the tokenizer's default template.",
    )
    main(parser.parse_args())
