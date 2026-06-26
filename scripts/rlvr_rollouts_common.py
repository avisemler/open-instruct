# scripts/rlvr_rollouts_common.py
"""Shared helpers for the two-stage RLVR rewards-only pipeline.

The pipeline is split into two scripts:

  * ``scripts/generate_rollouts.py`` — load the model and produce completions.
  * ``scripts/grade_rollouts.py``    — score completions against verifiers.

Both stages load the *same* dataset through the standard Open-Instruct caching
pipeline (and, when a small Dolci subset is requested, the *same* on-disk subset).
Keeping the cache-related arguments identical between the two scripts means the
model weights (Hugging Face hub cache) and the dataset (local dataset cache +
optional Dolci subset) are downloaded once and reused, rather than duplicated to
separate disk locations. This module centralises that shared logic so the two
scripts cannot drift apart.
"""

import argparse
import json
import os
import random

from datasets import Dataset, concatenate_datasets, load_dataset

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_fast, grpo_utils, logger_utils, utils
from open_instruct.dataset_transformation import INPUT_IDS_PROMPT_KEY, TokenizerConfig, get_cached_dataset_tulu
from open_instruct.environments.tools.utils import EnvsConfig

logger = logger_utils.setup_logger(__name__)


# Dolci-Think-RL-7B has 6 raw category labels that map to 4 logical groups.
DOLCI_CATEGORY_GROUPS: dict[str, list[str]] = {
    "math": ["math"],
    "ifeval": ["ifeval"],
    "code": ["code", "code_stdio"],
    "general-quality": ["general-quality", "general-quality_ref"],
}


def get_or_create_dolci_subset(path: str, n_per_category: int = 200, seed: int = 42) -> Dataset:
    """Load a small balanced Dolci subset from disk, creating it first if absent.

    Samples ``n_per_category`` examples from each of the four merged categories
    (math, ifeval, code, general-quality) and saves the result to ``path`` so
    subsequent runs (including the other pipeline stage) skip the full dataset
    download and read the identical subset from the same location.
    """
    if path and os.path.exists(path):
        logger.info(f"Loading Dolci subset from {path}")
        return Dataset.load_from_disk(path).shuffle(seed=seed)

    logger.info(f"Dolci subset not found at {path!r} — creating from full dataset...")
    full = load_dataset("allenai/Dolci-Think-RL-7B", split="train")

    # Build a flat list of (index, group_name) pairs for each row.
    raw_categories = full["dataset"]  # each element is a list like ['math']
    group_indices: dict[str, list[int]] = {g: [] for g in DOLCI_CATEGORY_GROUPS}
    for i, cats in enumerate(raw_categories):
        cat = cats[0] if isinstance(cats, list) else cats
        for group, members in DOLCI_CATEGORY_GROUPS.items():
            if cat in members:
                group_indices[group].append(i)
                break

    rng = random.Random(seed)
    subsets = []
    for group, indices in group_indices.items():
        n = min(n_per_category, len(indices))
        sampled = rng.sample(indices, n)
        logger.info(f"  {group}: {len(indices)} available → sampled {n}")
        subsets.append(full.select(sampled))

    subset = concatenate_datasets(subsets)
    subset.save_to_disk(path)
    logger.info(f"Saved {len(subset)}-example Dolci subset to {path}")
    return subset


def _load_full_dataset(
    args: grpo_utils.GRPOExperimentConfig,
    tc: TokenizerConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    tokenizer,
    needs_tokenized_prompts: bool,
) -> Dataset:
    """Load (and optionally tokenize) the full dataset via the standard caching pipeline."""
    # Load raw first to detect whether the dataset is already preprocessed.
    # Datasets like Dolci-Think-RL-7B ship with input_ids_prompt/ground_truth/dataset
    # already present and have no `messages` column for the tokenization transform.
    raw_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=streaming_config.dataset_mixer_list,
        dataset_mixer_list_splits=streaming_config.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=[],
        transform_fn_args=[],
        dataset_cache_mode=streaming_config.dataset_cache_mode,
        dataset_config_hash=streaming_config.dataset_config_hash,
        hf_entity=args.hf_entity,
        dataset_local_cache_dir=streaming_config.dataset_local_cache_dir,
        dataset_skip_cache=streaming_config.dataset_skip_cache,
    )

    if INPUT_IDS_PROMPT_KEY in raw_dataset.column_names:
        logger.info("Dataset already has tokenized prompts -- skipping tokenization.")
        return raw_dataset
    elif needs_tokenized_prompts:
        logger.info("Tokenizing dataset for generation mode...")
        train_dataset, _ = grpo_fast.setup_datasets(
            args, tc, tokenizer, streaming_config, tool_definitions=[], pass_tools_to_chat_template=False
        )
        return train_dataset
    else:
        # Scoring pre-computed completions only needs ground_truth + dataset columns.
        return raw_dataset


def load_rollout_dataset(
    script_args: argparse.Namespace,
    args: grpo_utils.GRPOExperimentConfig,
    tc: TokenizerConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    tokenizer,
    needs_tokenized_prompts: bool,
) -> Dataset:
    """Load the dataset shared by both pipeline stages and apply the final shuffle.

    ``needs_tokenized_prompts`` should be True for the generation stage (which needs
    ``input_ids_prompt``) and False for the grading stage. When the dataset already
    ships tokenized prompts (e.g. Dolci) the row set and order are identical either
    way, so an ``idx`` produced by the generation stage indexes the same row here.
    """
    logger.info("Loading dataset...")
    if script_args.dolci_subset_path is not None:
        train_dataset = get_or_create_dolci_subset(
            path=script_args.dolci_subset_path, n_per_category=script_args.dolci_subset_n_per_category, seed=args.seed
        )
    else:
        train_dataset = _load_full_dataset(args, tc, streaming_config, tokenizer, needs_tokenized_prompts)
    train_dataset = train_dataset.shuffle(seed=args.seed)
    logger.info(f"Dataset columns: {train_dataset.column_names}")
    logger.info(f"Dataset loaded: {len(train_dataset)} examples")
    return train_dataset


def add_shared_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add the dataset/cache arguments common to both pipeline stages.

    Pass the *same* values to both scripts so they hit the same on-disk caches.
    """
    parser.add_argument(
        "--dolci_subset_path",
        type=str,
        default=None,
        help=(
            "Path to save/load a small balanced Dolci subset "
            "(200 samples per merged category by default). "
            "If the path exists on disk it is loaded directly, skipping the full dataset download. "
            "If it does not exist, the subset is created from allenai/Dolci-Think-RL-7B and saved there. "
            "Use the same path for generate_rollouts.py and grade_rollouts.py to share the subset."
        ),
    )
    parser.add_argument(
        "--dolci_subset_n_per_category",
        type=int,
        default=200,
        help="Number of examples to sample per merged category when creating the Dolci subset.",
    )


def parse_oi_configs(remaining_argv: list[str]):
    """Parse the standard Open-Instruct config dataclasses with the rewards-only defaults.

    Returns ``(args, tc, streaming_config, vllm_config, tools_config)``. The defaults
    are shared between both pipeline stages so dataset selection, tokenizer, and cache
    locations stay identical.
    """
    import sys  # noqa: PLC0415

    sys.argv = [sys.argv[0]] + remaining_argv

    oi_parser = utils.ArgumentParserPlus(
        [  # ty: ignore[invalid-argument-type]
            grpo_utils.GRPOExperimentConfig,
            TokenizerConfig,
            data_loader_lib.StreamingDataLoaderConfig,
            data_loader_lib.VLLMConfig,
            EnvsConfig,
        ]
    )
    oi_parser.set_defaults(
        exp_name="rlvr_rewards_only",
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        per_device_train_batch_size=1,
        fused_optimizer=False,
        # pack_length is unused here; set high enough to pass StreamingDataLoaderConfig's
        # assertion (pack_length >= max_prompt_token_length + response_length).
        pack_length=10_000_000,
        dataset_mixer_list=["allenai/Dolci-Think-RL-7B", "1.0"],
        dataset_mixer_list_splits=["train"],
        # For Olmo 3 7B think models at RL/eval stages, use olmo_thinker which adds
        # <think> to add_generation_prompt so the model starts and closes thinking correctly.
        # See docs/olmo3.md: think evaluation and post-SFT stages use olmo-3.2-tokenizer-think-dev.
        chat_template_name="olmo_thinker",
        filter_zero_std_samples=False,
        system_prompt_override_file="scripts/train/qwen/math_system_prompt.txt",
        response_length=32768,
        num_samples_per_prompt_rollout=1,
    )
    return oi_parser.parse_args_into_dataclasses()


def resolve_tokenizer(tc: TokenizerConfig, model_name_or_path: str | None):
    """Resolve the tokenizer, falling back to ``model_name_or_path`` for its source."""
    if tc.tokenizer_name_or_path is None:
        if model_name_or_path is None:
            raise ValueError("Provide --tokenizer_name_or_path or --model_name_or_path.")
        tc.tokenizer_name_or_path = model_name_or_path
    return tc.tokenizer


def load_completions_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
