# scripts/grade_rollouts.py
"""
Grade rollouts (completions) for RLVR / verifiable-reward tasks and emit rewards.

This is the second stage of the two-stage rewards-only pipeline. It scores the
completions written by ``scripts/generate_rollouts.py`` (or any compatible JSONL)
against the verifiable-reward functions, without GRPO training.

Usage:
   python scripts/grade_rollouts.py \
     --completions_jsonl /tmp/completions.jsonl \
     --output_jsonl /tmp/rewards.jsonl \
     [normal Open-Instruct dataset/tokenizer/config args...]

Expected completions JSONL format (as produced by generate_rollouts.py):
{"idx": 0, "completion": "...", "ground_truth": ..., "dataset": "..."}

When ``ground_truth`` and ``dataset`` are present on each row, no dataset load is
needed at all. For externally-produced completions that contain only {"idx", "completion"},
the dataset is loaded once and the ground truth is looked up by ``idx`` — in that case
pass the same dataset/cache args that were used for generation so the on-disk caches are
shared rather than duplicated.
"""

import argparse
import asyncio

import rlvr_rollouts_common as common

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_utils, logger_utils
from open_instruct.dataset_transformation import GROUND_TRUTHS_KEY, VERIFIER_SOURCE_KEY, TokenizerConfig
from open_instruct.environments.tools.utils import EnvsConfig
from open_instruct.ground_truth_utils import apply_verifiable_reward, build_all_verifiers

logger = logger_utils.setup_logger(__name__)


def _resolve_ground_truths(completions: list[dict], dataset) -> tuple[list, list]:
    """Return (ground_truths, dataset_names) aligned with ``completions``.

    Prefers the values embedded in each completion row (written by generate_rollouts.py);
    falls back to a dataset lookup by ``idx`` for externally-produced completions.
    """
    have_embedded = all("ground_truth" in c and "dataset" in c for c in completions)
    if have_embedded:
        logger.info("Using ground truth embedded in the completions file (no dataset load needed).")
        ground_truths = [c["ground_truth"] for c in completions]
        dataset_names = [c["dataset"] for c in completions]
        return ground_truths, dataset_names

    if dataset is None:
        raise ValueError(
            "Completions are missing 'ground_truth'/'dataset' columns and no dataset was loaded. "
            "These columns are written by generate_rollouts.py; for external completions, ensure the "
            "dataset args resolve the same dataset used for generation."
        )
    logger.info("Looking up ground truth from the dataset by idx...")
    indices = [c["idx"] for c in completions]
    ground_truths = [dataset[i][GROUND_TRUTHS_KEY] for i in indices]
    dataset_names = [dataset[i][VERIFIER_SOURCE_KEY] for i in indices]
    return ground_truths, dataset_names


def score_completions(
    completions: list[dict],
    ground_truths: list,
    dataset_names: list,
    verifier_functions: dict,
    verification_reward: float,
) -> list[dict]:
    """Score {idx, completion} dicts against the provided ground truths."""
    texts = [c["completion"] for c in completions]

    # responses (tokenized) are not used by most verifiers; pass empty lists
    responses = [[] for _ in texts]

    rewards, per_func = asyncio.run(
        apply_verifiable_reward(
            reward_fn_mapping=verifier_functions,
            responses=responses,
            decoded_responses=texts,
            ground_truths=ground_truths,
            datasets=dataset_names,
            reward_mult=verification_reward,
        )
    )

    results = []
    for i, (entry, reward, pf) in enumerate(zip(completions, rewards, per_func)):
        results.append(
            {
                "idx": entry["idx"],
                "completion": entry["completion"],
                "reward": reward,
                "per_func_rewards": pf,
                "ground_truth": ground_truths[i],
                "dataset": dataset_names[i],
            }
        )

    return results


def main(
    script_args: argparse.Namespace,
    args: grpo_utils.GRPOExperimentConfig,
    tc: TokenizerConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    tools_config: EnvsConfig,
) -> None:
    logger.info(f"Loading completions from {script_args.completions_jsonl}...")
    completions = common.load_completions_jsonl(script_args.completions_jsonl)
    logger.info(f"Loaded {len(completions)} completions")

    # Only load the dataset when the completions don't already carry their ground truth.
    have_embedded = all("ground_truth" in c and "dataset" in c for c in completions)
    train_dataset = None
    if not have_embedded:
        tokenizer = common.resolve_tokenizer(tc, script_args.model_name_or_path)
        train_dataset = common.load_rollout_dataset(
            script_args, args, tc, streaming_config, tokenizer, needs_tokenized_prompts=False
        )

    ground_truths, dataset_names = _resolve_ground_truths(completions, train_dataset)

    verifier_functions = build_all_verifiers(args, streaming_config)

    logger.info("Scoring completions...")
    results = score_completions(
        completions=completions,
        ground_truths=ground_truths,
        dataset_names=dataset_names,
        verifier_functions=verifier_functions,
        verification_reward=int(streaming_config.verification_reward),
    )

    logger.info(f"Writing {len(results)} results to {script_args.output_jsonl}...")
    common.write_jsonl(script_args.output_jsonl, results)

    rewards = [r["reward"] for r in results]
    nonzero = sum(1 for r in rewards if r != 0)
    logger.info(
        f"Done. Mean reward: {sum(rewards) / len(rewards):.4f}, "
        f"Non-zero: {nonzero}/{len(rewards)} ({100 * nonzero / len(rewards):.1f}%)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--completions_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    # Optional: only used (as tokenizer source) when the dataset must be loaded to look
    # up ground truth for completions that don't already carry it.
    parser.add_argument("--model_name_or_path", type=str, default=None)
    common.add_shared_dataset_args(parser)
    script_args, remaining = parser.parse_known_args()

    args, tc, streaming_config, vllm_config, tools_config = common.parse_oi_configs(remaining)

    main(script_args, args, tc, streaming_config, vllm_config, tools_config)  # type: ignore[arg-type]
