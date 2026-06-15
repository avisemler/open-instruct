# scripts/run_rlvr_rewards_only.py
"""
Run RLVR / verifiable-reward tasks and emit rewards, without GRPO training.

Two modes:

1. Score precomputed completions:
   python scripts/run_rlvr_rewards_only.py \
     --completions_jsonl /path/to/completions.jsonl \
     --output_jsonl /tmp/rewards.jsonl \
     [normal Open-Instruct dataset/tokenizer/config args...]

2. Generate completions locally with transformers, then score:
   python scripts/run_rlvr_rewards_only.py \
     --model_name_or_path allenai/OLMo-2-1124-7B-Instruct \
     --generate_with_transformers \
     --max_examples 100 \
     --output_jsonl /tmp/rewards.jsonl \
     [normal Open-Instruct dataset/tokenizer/config args...]

Expected completions JSONL format:
{"idx": 0, "completion": "..."}
{"idx": 1, "completion": "..."}

The idx is interpreted as an index into the selected dataset.
"""

import argparse
import asyncio
import json
import sys

import torch
from transformers import AutoModelForCausalLM

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_fast, grpo_utils, logger_utils, utils
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
    TokenizerConfig,
    get_cached_dataset_tulu,
)
from open_instruct.environments.tools.utils import EnvsConfig
from open_instruct.ground_truth_utils import apply_verifiable_reward, build_all_verifiers

logger = logger_utils.setup_logger(__name__)


def load_completions_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def score_completions(
    completions: list[dict],
    dataset,
    verifier_functions: dict,
    verification_reward: float,
) -> list[dict]:
    """Score a list of {idx, completion} dicts against the dataset ground truths."""
    indices = [c["idx"] for c in completions]
    texts = [c["completion"] for c in completions]

    ground_truths = [dataset[i][GROUND_TRUTHS_KEY] for i in indices]
    dataset_names = [dataset[i][VERIFIER_SOURCE_KEY] for i in indices]

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


def generate_with_transformers(
    dataset,
    model_name_or_path: str,
    tokenizer,
    max_examples: int,
    max_new_tokens: int = 512,
) -> list[dict]:
    """Generate completions using a local HF transformers model."""
    logger.info(f"Loading model {model_name_or_path} for local generation...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    n = min(max_examples, len(dataset))
    completions = []

    logger.info(f"Generating {n} completions...")
    for i in range(n):
        input_ids = dataset[i][INPUT_IDS_PROMPT_KEY]
        input_tensor = torch.tensor([input_ids], device=model.device)

        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output[0][len(input_ids):]
        text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        completions.append({"idx": i, "completion": text})

        if (i + 1) % 10 == 0:
            logger.info(f"  {i + 1}/{n} generated")

    return completions


def main(
    script_args: argparse.Namespace,
    args: grpo_utils.GRPOExperimentConfig,
    tc: TokenizerConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    tools_config: EnvsConfig,
) -> None:
    # --model_name_or_path doubles as the tokenizer source when --tokenizer_name_or_path
    # is not supplied separately.
    if tc.tokenizer_name_or_path is None:
        if script_args.model_name_or_path is None:
            raise ValueError("Provide --tokenizer_name_or_path or --model_name_or_path.")
        tc.tokenizer_name_or_path = script_args.model_name_or_path
    tokenizer = tc.tokenizer

    logger.info("Loading dataset...")
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
        logger.info("Dataset already has tokenized prompts — skipping tokenization.")
        train_dataset = raw_dataset
    elif script_args.generate_with_transformers:
        # Generation needs tokenized prompts: run the full transform pipeline.
        logger.info("Tokenizing dataset for generation mode...")
        train_dataset, _ = grpo_fast.setup_datasets(
            args, tc, tokenizer, streaming_config,
            tool_definitions=[], pass_tools_to_chat_template=False,
        )
    else:
        # Scoring pre-computed completions only needs ground_truth + dataset columns.
        train_dataset = raw_dataset

    logger.info(f"Dataset loaded: {len(train_dataset)} examples")

    verifier_functions = build_all_verifiers(args, streaming_config)

    if script_args.generate_with_transformers:
        if script_args.model_name_or_path is None:
            raise ValueError("--model_name_or_path is required with --generate_with_transformers")
        completions = generate_with_transformers(
            train_dataset,
            script_args.model_name_or_path,
            tokenizer,
            max_examples=script_args.max_examples,
            max_new_tokens=script_args.max_new_tokens,
        )
    else:
        logger.info(f"Loading completions from {script_args.completions_jsonl}...")
        completions = load_completions_jsonl(script_args.completions_jsonl)
        logger.info(f"Loaded {len(completions)} completions")

    logger.info("Scoring completions...")
    results = score_completions(
        completions=completions,
        dataset=train_dataset,
        verifier_functions=verifier_functions,
        verification_reward=int(streaming_config.verification_reward),
    )

    logger.info(f"Writing {len(results)} results to {script_args.output_jsonl}...")
    with open(script_args.output_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    rewards = [r["reward"] for r in results]
    nonzero = sum(1 for r in rewards if r != 0)
    logger.info(
        f"Done. Mean reward: {sum(rewards)/len(rewards):.4f}, "
        f"Non-zero: {nonzero}/{len(rewards)} ({100*nonzero/len(rewards):.1f}%)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--completions_jsonl", type=str, default=None)
    parser.add_argument("--generate_with_transformers", action="store_true")
    # Optional: doubles as tokenizer source when --tokenizer_name_or_path is absent
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_jsonl", type=str, required=True)
    script_args, remaining = parser.parse_known_args()

    if not script_args.generate_with_transformers and script_args.completions_jsonl is None:
        parser.error("Either --completions_jsonl or --generate_with_transformers must be specified.")

    sys.argv = [sys.argv[0]] + remaining

    oi_parser = utils.ArgumentParserPlus(
        [  # ty: ignore[invalid-argument-type]
            grpo_utils.GRPOExperimentConfig,
            TokenizerConfig,
            data_loader_lib.StreamingDataLoaderConfig,
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
    )
    args, tc, streaming_config, tools_config = oi_parser.parse_args_into_dataclasses()

    main(script_args, args, tc, streaming_config, tools_config)  # type: ignore[arg-type]
