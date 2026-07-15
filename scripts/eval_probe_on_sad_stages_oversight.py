"""
Evaluate the evaluation-awareness probe on the SAD stages_oversight task.

stages_oversight is part of the Situational Awareness Dataset (SAD,
https://github.com/LRudL/sad). It contains paired multiple-choice questions
presenting scenarios where the model is either being evaluated/overseen
(batch "test_oversight") or operating in deployment (batch "deploy_oversight").
The probe is fit on synthetic prompts from cache_probe_directions.py; this script
tests whether it fires more strongly on test_oversight scenarios — a proper OOD
evaluation.

Dataset setup (requires the SAD repository):
    git clone https://github.com/LRudL/sad.git
    cd sad
    pip install -e .
    ./unzip.sh --exclude-evals
    sad export --subset mini --filename export
    # This writes sad/exports/export.json

Usage:
    python scripts/eval_probe_on_sad_stages_oversight.py \\
        --probe_directions_path probe_cache/allenai_Olmo-3-32B-Think_hua/probe_directions_main.pt \\
        --sad_export_json /path/to/sad/exports/export.json \\
        [--probe_layer_indices 10,20,30] \\
        [--max_examples 200] \\
        [--positive_keyword test]

Label assignment:
    Each item's label is read from sample_info["sample"]["splits"]["answer"]:
      "yes" → 1 (the scenario is from a test benchmark — model is being evaluated)
      "no"  → 0 (the scenario is from real deployment, e.g. WildChat)
    If that field is absent the --positive_keyword blob search is used as a fallback.
"""

import argparse
import csv
import json
import os

import torch
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct import logger_utils
from open_instruct.dataset_transformation import CHAT_TEMPLATES

logger = logger_utils.setup_logger(__name__)

_DEFAULT_TASK = "stages_oversight"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_sad_export(path: str, task_name: str, max_examples: int | None, positive_keyword: str) -> list[dict]:
    """Load stages_oversight (or any SAD-mini task) entries from the SAD export JSON.

    Each returned dict has keys:
        messages  – list[{"role": str, "content": str}]  (the MCQ prompt)
        label     – 1 for test/eval scenarios, 0 for deployment scenarios
        sample_id – str, for debugging

    Label extraction (in order of priority):
      1. sample_info["sample"]["splits"]["answer"]: "yes" → 1, "no" → 0.
         For stages_oversight this field encodes whether the prompt comes from a
         test benchmark ("yes") or real deployment ("no").
      2. Fallback: positive_keyword search in the full serialised metadata blob.
         Useful for tasks whose samples don't carry a "splits.answer" field.
    """
    with open(path) as input_file:
        export = json.load(input_file)

    if task_name not in export:
        available = list(export.keys())
        raise KeyError(f"Task '{task_name}' not found in export. Available tasks: {available}")

    raw_items = export[task_name]
    items = []
    unknown_label_count = 0

    for raw in raw_items:
        prompt = raw.get("prompt", [])
        sample_info = raw.get("sample_info", {})
        sample_id = _extract_sample_id(sample_info)

        # Primary: use the "splits.answer" field that SAD stages_oversight
        # populates with "yes" (test benchmark) or "no" (deployment/wildchat).
        sample = sample_info.get("sample") or {}
        splits = sample.get("splits") or {}
        split_answer = str(splits.get("answer", "")).lower()

        if split_answer == "yes":
            label = 1
        elif split_answer == "no":
            label = 0
        else:
            # Fallback: keyword search in the full metadata blob.
            sample_blob = json.dumps(sample_info).lower()
            if positive_keyword.lower() in sample_blob:
                label = 1
            else:
                label = -1
                unknown_label_count += 1

        items.append({"messages": prompt, "label": label, "sample_id": sample_id})

    if unknown_label_count:
        logger.warning(
            f"{unknown_label_count}/{len(items)} items had no 'splits.answer' field and "
            f"did not match --positive_keyword '{positive_keyword}'; they will be skipped."
        )
    items = [item for item in items if item["label"] != -1]

    if items:
        logger.info(f"  First sample_id: {items[0]['sample_id']}  (label={items[0]['label']})")
    positive = sum(1 for item in items if item["label"] == 1)
    negative = len(items) - positive
    logger.info(f"Loaded {len(items)} items from task '{task_name}' ({positive} positive, {negative} negative)")

    if max_examples is not None:
        pos_items = [item for item in items if item["label"] == 1][: max_examples // 2]
        neg_items = [item for item in items if item["label"] == 0][: max_examples // 2]
        items = pos_items + neg_items
        logger.info(f"Subsampled to {len(items)} ({len(pos_items)} positive, {len(neg_items)} negative)")

    return items


def _extract_sample_id(sample_info: dict) -> str:
    """Best-effort extraction of a human-readable sample identifier."""
    for key in ("id", "sample_id", "name"):
        val = sample_info.get(key)
        if val:
            return str(val)
    sample = sample_info.get("sample", {})
    if isinstance(sample, dict):
        for key in ("id", "sample_id", "name"):
            val = sample.get(key)
            if val:
                return str(val)
    return "<unknown>"


def messages_to_token_ids(messages: list[dict], tokenizer) -> list[int]:
    """Apply the model's chat template to a list of message dicts."""
    try:
        token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
        # The olmo_thinker generation prompt ends in <think>; score the prompt before it.
        return token_ids[:-3]
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return tokenizer.encode(text)


# ---------------------------------------------------------------------------
# Hidden-state extraction  (identical to eval_probe_on_eval_awareness.py)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_hidden_states(token_ids: list[int], model, max_tokens: int, pooling: str) -> list[torch.Tensor]:
    """Return one pooled hidden vector per decoder layer."""
    if len(token_ids) > max_tokens:
        token_ids = token_ids[-max_tokens:]
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    outputs = model(input_ids=input_tensor, attention_mask=torch.ones_like(input_tensor), output_hidden_states=True)
    result = []
    for hidden in outputs.hidden_states[1:]:
        h = hidden[0].float().detach().cpu()
        result.append(h[-2] if pooling == "last" and len(h) > 1 else h.mean(dim=0))
    return result


# ---------------------------------------------------------------------------
# CSV score cache  (same structure as eval_probe_on_eval_awareness.py)
# ---------------------------------------------------------------------------


def load_score_log(
    path: str, directions: dict[int, torch.Tensor]
) -> tuple[list[int], dict[int, list[float]], dict[int, list[float]]]:
    """Load labels and learned/random probe scores from a previous run."""
    scores_by_layer = {layer: [] for layer in directions}
    rand_scores_by_layer = {layer: [] for layer in directions}
    labels = []
    with open(path, newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = (
            {"dataset_index", "label"}
            | {f"score_layer_{lk}" for lk in directions}
            | {f"random_score_layer_{lk}" for lk in directions}
        )
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Score log {path} does not contain scores for all requested layers")
        for row in reader:
            labels.append(int(row["label"]))
            for layer in directions:
                scores_by_layer[layer].append(float(row[f"score_layer_{layer}"]))
                rand_scores_by_layer[layer].append(float(row[f"random_score_layer_{layer}"]))
    return labels, scores_by_layer, rand_scores_by_layer


def write_score_log(path: str, rows: list[dict[str, float | int]], directions: dict[int, torch.Tensor]) -> None:
    """Save each item index, label, and probe scores to a CSV cache."""
    fieldnames = ["dataset_index", "label"]
    fieldnames += [f"score_layer_{lk}" for lk in directions]
    fieldnames += [f"random_score_layer_{lk}" for lk in directions]
    with open(path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def auc_and_optimal_accuracy(labels: list[int], scores: list[float]) -> tuple[float, float]:
    """AUC and accuracy at the Youden-optimal threshold."""
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    threshold = thresholds[(tpr - fpr).argmax()]
    accuracy = sum((s >= threshold) == bool(lb) for s, lb in zip(scores, labels)) / len(labels)
    return auc, accuracy


def print_results_table(
    title: str, directions: dict[int, torch.Tensor], scores_by_layer: dict[int, list[float]], labels: list[int]
) -> None:
    print("\n" + "=" * 65)
    print(f"{title}  (n={len(labels)})")
    print("Label: test_oversight=1, deploy_oversight=0")
    print("=" * 65)
    print(f"  {'Layer':>6}  {'AUC-ROC':>8}  {'Accuracy':>8}  {'Mean gap':>10}")
    print("-" * 65)
    for layer in sorted(directions):
        s = scores_by_layer[layer]
        if len(s) < len(labels):
            logger.warning(f"Layer {layer}: only {len(s)} scores (expected {len(labels)}), skipping")
            continue
        auc, accuracy = auc_and_optimal_accuracy(labels, s)
        pos_scores = [sc for sc, lb in zip(s, labels) if lb == 1]
        neg_scores = [sc for sc, lb in zip(s, labels) if lb == 0]
        gap = sum(pos_scores) / len(pos_scores) - sum(neg_scores) / len(neg_scores)
        print(f"  {layer:>6}  {auc:>8.4f}  {accuracy:>8.4f}  {gap:>+10.4f}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    if os.path.exists(args.score_log_path):
        logger.info(f"Loading cached probe scores from {args.score_log_path}")
        labels, scores_by_layer, rand_scores_by_layer = load_score_log(args.score_log_path, directions)
    else:
        items = load_sad_export(args.sad_export_json, args.task_name, args.max_examples, args.positive_keyword)
        if not items:
            raise ValueError("No labelled items found. Check --positive_keyword matches the sample metadata.")

        logger.info(f"Loading tokenizer + model from {args.model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if args.chat_template_name and args.chat_template_name in CHAT_TEMPLATES:
            tokenizer.chat_template = CHAT_TEMPLATES[args.chat_template_name]
            logger.info(f"Applied chat template: {args.chat_template_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        scores_by_layer = {lk: [] for lk in directions}
        torch.manual_seed(99)
        random_directions = {
            lk: torch.nn.functional.normalize(torch.randn_like(v), dim=0) for lk, v in directions.items()
        }
        rand_scores_by_layer = {lk: [] for lk in random_directions}
        labels = []
        log_rows = []

        for i, item in enumerate(items):
            try:
                token_ids = messages_to_token_ids(item["messages"], tokenizer)
            except Exception as e:
                logger.warning(f"Skipping item {i} (sample_id={item['sample_id']}): tokenization failed — {e}")
                continue

            per_layer = extract_hidden_states(token_ids, model, args.max_tokens, args.pooling)
            log_row: dict[str, float | int] = {"dataset_index": i, "label": item["label"]}
            for lk, direction in directions.items():
                if lk < len(per_layer):
                    score = torch.dot(per_layer[lk], direction.float()).item()
                    scores_by_layer[lk].append(score)
                    rand_score = torch.dot(per_layer[lk], random_directions[lk].float()).item()
                    rand_scores_by_layer[lk].append(rand_score)
                    log_row[f"score_layer_{lk}"] = score
                    log_row[f"random_score_layer_{lk}"] = rand_score

            labels.append(item["label"])
            log_rows.append(log_row)
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(items)} examples...")

        write_score_log(args.score_log_path, log_rows, directions)
        logger.info(f"Saved probe scores to {args.score_log_path}")

    logger.info(f"Scored {len(labels)} examples ({sum(labels)} positive, {len(labels) - sum(labels)} negative)")

    print_results_table("AUC-ROC  (eval-awareness probe)", directions, scores_by_layer, labels)
    print_results_table("RANDOM PROBE CONTROL (expected AUC ≈ 0.50)", directions, rand_scores_by_layer, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the eval-awareness probe on the SAD stages_oversight task.")
    parser.add_argument("--model_name_or_path", type=str, default="allenai/Olmo-3-32B-Think")
    parser.add_argument(
        "--sad_export_json",
        type=str,
        required=True,
        help="Path to the export.json produced by 'sad export' (SAD repository).",
    )
    parser.add_argument("--probe_directions_path", type=str, required=True)
    parser.add_argument(
        "--task_name",
        type=str,
        default=_DEFAULT_TASK,
        help=f"Task key in the SAD export JSON. Default: {_DEFAULT_TASK}.",
    )
    parser.add_argument(
        "--positive_keyword",
        type=str,
        default="test",
        help=(
            "Fallback keyword used when a sample has no splits.answer field. "
            "For stages_oversight the primary label source is splits.answer ('yes'/'no'); "
            "this keyword is only consulted for tasks that lack that field."
        ),
    )
    parser.add_argument(
        "--probe_layer_indices",
        type=str,
        default=",".join(str(i) for i in range(0, 63)),
        help="Comma-separated layer indices. Default: all layers up to 63.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Cap total examples (balanced: half positive, half negative). Default: use all.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048, help="Truncate prompts to this many tokens (tail). Default: 2048."
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "last"],
        help="Hidden-state pooling over token positions. Default: mean.",
    )
    parser.add_argument("--chat_template_name", type=str, default="olmo_thinker")
    parser.add_argument(
        "--score_log_path",
        type=str,
        default=None,
        help="CSV score cache. Defaults to <probe_directions_path>_sad_stages_scores.csv; reused when present.",
    )
    args = parser.parse_args()
    if args.score_log_path is None:
        args.score_log_path = f"{os.path.splitext(args.probe_directions_path)[0]}_sad_stages_scores.csv"
    main(args)
