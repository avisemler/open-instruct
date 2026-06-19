# scripts/run_rlvr_rewards_only.py
"""
Run RLVR / verifiable-reward tasks and emit rewards, without GRPO training.

Three modes:

1. Score precomputed completions:
   python scripts/run_rlvr_rewards_only.py \
     --completions_jsonl /path/to/completions.jsonl \
     --output_jsonl /tmp/rewards.jsonl \
     [normal Open-Instruct dataset/tokenizer/config args...]

2. Generate completions with vLLM, then score (recommended):
   python scripts/run_rlvr_rewards_only.py \
     --model_name_or_path allenai/OLMo-2-1124-7B-Instruct \
     --generate_with_vllm \
     --max_examples 100 \
     --output_jsonl /tmp/rewards.jsonl \
     [normal Open-Instruct dataset/tokenizer/config args...]

3. Generate completions locally with transformers, then score:
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
import os
import random
import sys

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
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


# ---------------------------------------------------------------------------
# Activation-hook helpers (used by the linear-probe code path)
# ---------------------------------------------------------------------------

def _probe_register_hooks(model) -> dict:
    """Register forward hooks on decoder layers; runs inside the vLLM engine-core process.

    vLLM 0.19+ executes the model in a separate subprocess. apply_model() serialises
    this function and runs it there. We persist handles and activations in sys attributes
    so subsequent apply_model calls can read / remove them.
    """
    import sys

    sys._probe_store: dict = {}
    sys._probe_handles: list = []

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = list(model.model.layers)
    elif hasattr(model, "layers"):
        layers = list(model.layers)
    else:
        return {"n_layers": 0, "model_type": type(model).__name__, "error": "no decoder layers found"}

    def _make_hook(idx: int):
        def _hook(_mod, _inp, out):
            hidden = out[0] if isinstance(out, (tuple, list)) else out
            # vLLM packs sequences as [num_tokens, hidden_dim] (no batch dim).
            sys._probe_store[idx] = hidden[0, :].detach().cpu()
        return _hook

    for i, layer in enumerate(layers):
        sys._probe_handles.append(layer.register_forward_hook(_make_hook(i)))

    return {"n_layers": len(layers), "model_type": type(model).__name__}


def _probe_read_activations(model) -> dict:
    """Return stored activations; runs inside the vLLM engine-core process."""
    import sys
    return dict(getattr(sys, "_probe_store", {}))


def _probe_remove_hooks(model) -> None:
    """Remove hooks and clear state; runs inside the vLLM engine-core process."""
    import sys
    for h in getattr(sys, "_probe_handles", []):
        h.remove()
    sys._probe_handles = []
    sys._probe_store = {}


def _print_activation_summary(store: dict) -> None:
    logger.info(f"Linear-probe activations — {len(store)} layers (token[0] of packed batch, last forward pass):")
    for idx in sorted(store):
        act = store[idx]
        logger.info(
            f"  layer {idx:3d} | shape {list(act.shape)} | "
            f"mean {act.mean().item():+.4f} | std {act.std().item():.4f} | "
            f"first_5: {[round(v, 4) for v in act[:5].tolist()]}"
        )


# Dolci-Think-RL-7B has 6 raw category labels that map to 4 logical groups.
DOLCI_CATEGORY_GROUPS: dict[str, list[str]] = {
    "math": ["math"],
    "ifeval": ["ifeval"],
    "code": ["code", "code_stdio"],
    "general-quality": ["general-quality", "general-quality_ref"],
}


def get_or_create_dolci_subset(
    path: str,
    n_per_category: int = 200,
    seed: int = 42,
) -> Dataset:
    """Load a small balanced Dolci subset from disk, creating it first if absent.

    Samples ``n_per_category`` examples from each of the four merged categories
    (math, ifeval, code, general-quality) and saves the result to ``path`` so
    subsequent runs skip the full dataset download.
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


def generate_with_vllm(
    dataset,
    model_name_or_path: str,
    tokenizer,
    max_examples: int,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    seed: int,
    linear_probe: bool = False,
) -> list[dict]:
    """Generate completions using vLLM (single GPU, no Ray).

    Mirrors the sampling setup in grpo_fast.create_generation_configs():
      - max_tokens = streaming_config.response_length
      - temperature = streaming_config.temperature
      - top_p = vllm_config.vllm_top_p
      - stop = streaming_config.stop_strings
      - n = streaming_config.num_samples_per_prompt_rollout
    """
    import vllm

    max_model_len = streaming_config.max_prompt_token_length + streaming_config.response_length

    if linear_probe:
        # Must be set before the engine-core subprocess is spawned so it inherits the flag.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    logger.info(f"Loading {model_name_or_path} with vLLM (max_model_len={max_model_len})...")
    llm = vllm.LLM(
        model=model_name_or_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=vllm_config.vllm_gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=vllm_config.vllm_enforce_eager,
        enable_prefix_caching=vllm_config.vllm_enable_prefix_caching,
        seed=seed,
    )

    num_samples = streaming_config.num_samples_per_prompt_rollout
    sampling_params = vllm.SamplingParams(
        temperature=streaming_config.temperature,
        top_p=vllm_config.vllm_top_p,
        max_tokens=streaming_config.response_length,
        n=num_samples,
        stop=streaming_config.stop_strings or None,
        seed=seed,
        logprobs=1,
        include_stop_str_in_output=True,
    )

    n = min(max_examples, len(dataset))
    inputs = [{"prompt_token_ids": list(dataset[i][INPUT_IDS_PROMPT_KEY])} for i in range(n)]

    logger.info(
        f"Generating {n} prompts x {num_samples} sample(s) "
        f"(temperature={streaming_config.temperature}, "
        f"max_tokens={streaming_config.response_length}, "
        f"stop={streaming_config.stop_strings})..."
    )

    if linear_probe:
        if not vllm_config.vllm_enforce_eager:
            logger.warning(
                "Linear probe hooks require eager execution (CUDA graphs bypass Python hooks). "
                "Pass --vllm_enforce_eager; activations may be absent without it."
            )
        # collective_rpc returns a list (one result per worker); unpack the single TP=1 result.
        info = llm.llm_engine.apply_model(_probe_register_hooks)[0]
        if "error" in info:
            logger.warning(f"Linear probe setup failed: {info['error']}")
        else:
            logger.info(
                f"Linear probe: hooks registered on {info['n_layers']} decoder layers "
                f"of {info['model_type']} (inside engine-core process)."
            )

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    if linear_probe:
        activation_store = llm.llm_engine.apply_model(_probe_read_activations)[0]
        if activation_store:
            _print_activation_summary(activation_store)
        else:
            logger.warning(
                "Linear probe: no activations captured — try --vllm_enforce_eager to disable CUDA graphs."
            )
        llm.llm_engine.apply_model(_probe_remove_hooks)

    completions = []
    for i, request_output in enumerate(outputs):
        for completion_output in request_output.outputs:
            text = tokenizer.decode(completion_output.token_ids, skip_special_tokens=False)
            completions.append({"idx": i, "completion": text})

    logger.info(f"Generated {len(completions)} completions for {n} prompts.")
    logger.info("Here are the first 5 completions:")
    for j, completion in enumerate(completions[:5]):
        logger.info(f"  {j + 1}: {completion['completion'][:300]}")
    return completions


def generate_with_transformers(
    dataset,
    model_name_or_path: str,
    tokenizer,
    max_examples: int,
    max_new_tokens: int,
    temperature: float,
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

    logger.info(f"Generating {n} completions with temperature={temperature}, max_new_tokens={max_new_tokens}...")
    for i in range(n):
        input_ids = dataset[i][INPUT_IDS_PROMPT_KEY]
        input_tensor = torch.tensor([input_ids], device=model.device)

        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                #pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output[0]#[len(input_ids):]
        text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        completions.append({"idx": i, "completion": text})

        if (i + 1) % 10 == 0:
            logger.info(f"  {i + 1}/{n} generated")
    logger.info("Generation complete.")
    logger.info("Here are the first 5 completions:")
    for j, completion in enumerate(completions[:5]):
        logger.info(f"  {j + 1}: {completion['completion']}")
    return completions


def _load_full_dataset(
    script_args: argparse.Namespace,
    args: grpo_utils.GRPOExperimentConfig,
    tc: TokenizerConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    tokenizer,
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

    needs_tokenized_prompts = script_args.generate_with_transformers or script_args.generate_with_vllm
    if INPUT_IDS_PROMPT_KEY in raw_dataset.column_names:
        logger.info("Dataset already has tokenized prompts -- skipping tokenization.")
        return raw_dataset
    elif needs_tokenized_prompts:
        logger.info("Tokenizing dataset for generation mode...")
        train_dataset, _ = grpo_fast.setup_datasets(
            args, tc, tokenizer, streaming_config,
            tool_definitions=[], pass_tools_to_chat_template=False,
        )
        return train_dataset
    else:
        # Scoring pre-computed completions only needs ground_truth + dataset columns.
        return raw_dataset


def main(
    script_args: argparse.Namespace,
    args: grpo_utils.GRPOExperimentConfig,
    tc: TokenizerConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
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
    if script_args.dolci_subset_path is not None:
        train_dataset = get_or_create_dolci_subset(
            path=script_args.dolci_subset_path,
            n_per_category=script_args.dolci_subset_n_per_category,
            seed=args.seed,
        )
    else:
        train_dataset = _load_full_dataset(script_args, args, tc, streaming_config, tokenizer)
    logger.info(f"Dataset loaded: {len(train_dataset)} examples")

    verifier_functions = build_all_verifiers(args, streaming_config)

    if script_args.generate_with_vllm:
        if script_args.model_name_or_path is None:
            raise ValueError("--model_name_or_path is required with --generate_with_vllm")
        completions = generate_with_vllm(
            dataset=train_dataset,
            model_name_or_path=script_args.model_name_or_path,
            tokenizer=tokenizer,
            max_examples=script_args.max_examples,
            streaming_config=streaming_config,
            vllm_config=vllm_config,
            seed=args.seed,
            linear_probe=script_args.linear_probe,
        )
    elif script_args.generate_with_transformers:
        if script_args.model_name_or_path is None:
            raise ValueError("--model_name_or_path is required with --generate_with_transformers")
        completions = generate_with_transformers(
            train_dataset,
            script_args.model_name_or_path,
            tokenizer,
            max_examples=script_args.max_examples,
            max_new_tokens=streaming_config.response_length * 4,
            temperature=streaming_config.temperature,
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
    parser.add_argument("--generate_with_vllm", action="store_true")
    parser.add_argument("--generate_with_transformers", action="store_true")
    parser.add_argument(
        "--linear_probe",
        action="store_true",
        help=(
            "Attach forward hooks to every decoder layer during vLLM generation and print "
            "the first-token hidden-state summary (mean, std, first 5 values) for each layer. "
            "Requires --generate_with_vllm. Use --vllm_enforce_eager to ensure hooks fire."
        ),
    )
    # Optional: doubles as tokenizer source when --tokenizer_name_or_path is absent
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument(
        "--dolci_subset_path",
        type=str,
        default=None,
        help=(
            "Path to save/load a small balanced Dolci subset "
            "(200 samples per merged category by default). "
            "If the path exists on disk it is loaded directly, skipping the full dataset download. "
            "If it does not exist, the subset is created from allenai/Dolci-Think-RL-7B and saved there."
        ),
    )
    parser.add_argument(
        "--dolci_subset_n_per_category",
        type=int,
        default=200,
        help="Number of examples to sample per merged category when creating the Dolci subset.",
    )
    script_args, remaining = parser.parse_known_args()

    if (
        not script_args.generate_with_vllm
        and not script_args.generate_with_transformers
        and script_args.completions_jsonl is None
    ):
        parser.error(
            "One of --completions_jsonl, --generate_with_vllm, or --generate_with_transformers must be specified."
        )

    sys.argv = [sys.argv[0]] + remaining

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
        system_prompt_override_file="scripts/train/qwen/math_system_prompt.txt"
    )
    args, tc, streaming_config, vllm_config, tools_config = oi_parser.parse_args_into_dataclasses()

    main(script_args, args, tc, streaming_config, vllm_config, tools_config)  # type: ignore[arg-type]
