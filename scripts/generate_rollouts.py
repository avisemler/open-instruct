# scripts/generate_rollouts.py
"""
Generate rollouts (completions) for RLVR / verifiable-reward tasks.

This is the first stage of the two-stage rewards-only pipeline; score the
completions it writes with ``scripts/grade_rollouts.py``.

Two generation backends:

1. vLLM (recommended):
   python scripts/generate_rollouts.py \
     --model_name_or_path allenai/OLMo-2-1124-7B-Instruct \
     --generate_with_vllm \
     --max_examples 100 \
     --output_dir /tmp/generate_rollouts \
     [normal Open-Instruct dataset/tokenizer/config args...]

2. Local transformers:
   python scripts/generate_rollouts.py \
     --model_name_or_path allenai/OLMo-2-1124-7B-Instruct \
     --generate_with_transformers \
     --max_examples 100 \
     --output_dir /tmp/generate_rollouts \
     [normal Open-Instruct dataset/tokenizer/config args...]

Each run creates a fresh timestamped subdirectory under ``--output_dir`` (e.g.
``/tmp/generate_rollouts/20260702_153000_482913/``) containing:
  * ``args.json``      -- every argument the run was invoked with
  * ``rollouts.jsonl``  -- one row per completion:
    {"idx": 0, "prompt": "...", "completion": "...", "ground_truth": ..., "dataset": "..."}

The ``prompt`` (flattened task prompt), ``ground_truth``, and ``dataset`` (verifier source)
columns are carried forward so downstream stages (grading, judging) are self-contained and
do not depend on the dataset row order matching. Score completions against verifiers with
``scripts/grade_rollouts.py``; judge them for evaluation awareness with
``scripts/judge_eval_awareness.py``.

Examples are generated one at a time and their completions are appended (and flushed)
to the output file immediately, so partial progress survives an interrupted run. To resume
an interrupted run, pass ``--run_subdir <name>`` matching the timestamped subdir it was
writing to (printed as "Run directory: ..." at startup); generation picks up from the first
example not yet completed instead of starting over. ``--run_subdir`` also works on a fresh
name to control where output goes.

--system_prompt_suffix appends text to the system prompt at inference time. Because the
dataset ships pre-tokenized, this re-renders the chat template per example: the conversation
is parsed back from the dataset's flattened `prompt` column (multi-turn) and the suffix is
appended to the base system prompt (--system_prompt_override_file, else empty).

Disk efficiency: this stage and grade_rollouts.py share the Hugging Face model/dataset
cache and the optional on-disk Dolci subset. Only this stage loads the model weights.
"""

import argparse
import dataclasses
import json
import os
import random
import time
from collections.abc import Iterator

import rlvr_rollouts_common as common
import torch
from transformers import AutoModelForCausalLM

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_utils, logger_utils
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
    TokenizerConfig,
)
from open_instruct.environments.tools.utils import EnvsConfig

logger = logger_utils.setup_logger(__name__)


def build_effective_system(
    streaming_config: data_loader_lib.StreamingDataLoaderConfig, system_prompt_suffix: str | None
) -> str | None:
    """Resolve the system-prompt content to render with when a suffix is requested.

    Returns ``None`` (meaning: use the dataset's baked prompt as-is) when no suffix is set.
    Otherwise the base system prompt is the contents of ``system_prompt_override_file`` (if
    configured; the scripts default it to the math system prompt) and the suffix is appended.
    """
    if system_prompt_suffix is None:
        return None
    base = ""
    if streaming_config.system_prompt_override_file is not None:
        with open(streaming_config.system_prompt_override_file) as f:
            base = f.read().strip()
    return f"{base}\n\n{system_prompt_suffix}" if base else system_prompt_suffix


# Roles that can begin a turn in the flattened RAW_PROMPT_KEY string.
_FLATTENED_PROMPT_ROLES = ("system", "user", "assistant")


def messages_from_flattened_prompt(text: str) -> list[dict]:
    """Invert the dataset's flattened ``prompt`` string back into chat messages.

    The dataset stores the prompt as ``"\\n".join(f"{role}: {content}")`` over the
    conversation turns (see ``rlvr_tokenize_*`` in dataset_transformation.py), so a multi-turn
    prompt looks like ``"user: ...\\nassistant: ...\\nuser: ..."``. We split on lines that begin
    with a known ``"role: "`` prefix; any other line is a continuation of the current message's
    content (which may span multiple lines).
    """
    messages: list[dict] = []
    for line in text.split("\n"):
        role = next((r for r in _FLATTENED_PROMPT_ROLES if line.startswith(f"{r}: ")), None)
        if role is not None:
            messages.append({"role": role, "content": line[len(role) + 2 :]})
        elif messages:
            messages[-1]["content"] += "\n" + line
        else:
            # No leading role prefix at all -> treat the whole thing as a single user turn.
            messages.append({"role": "user", "content": line})
    return messages


def prompt_token_ids_for(row: dict, tokenizer, effective_system: str | None) -> list[int]:
    """Return the prompt token ids for one example.

    With ``effective_system is None`` the dataset's pre-tokenized ``input_ids_prompt`` is used
    unchanged. Otherwise the chat template is re-rendered from the dataset's flattened ``prompt``
    column (parsed back into multi-turn messages) with ``effective_system`` replacing the system
    prompt — this is how a system-prompt suffix takes effect on a pre-tokenized dataset.
    """
    if effective_system is None:
        return list(row[INPUT_IDS_PROMPT_KEY])
    messages = messages_from_flattened_prompt(row[RAW_PROMPT_KEY])
    if messages and messages[0]["role"] == "system":
        messages = messages[1:]
    messages = [{"role": "system", "content": effective_system}, *messages]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)


def generate_with_vllm(
    dataset,
    model_name_or_path: str,
    tokenizer,
    max_examples: int,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    seed: int,
    effective_system: str | None = None,
    batch_size: int = 64,
    start_idx: int = 0,
    model_revision: str | None = None,
) -> "Iterator[tuple[int, list[str]]]":
    """Yield ``(idx, [completion_text, ...])``, ``batch_size`` examples at a time.

    The vLLM engine is loaded once. Prompts are submitted to ``llm.generate()`` in batches
    of ``batch_size`` so vLLM's continuous batching scheduler can run many sequences
    concurrently on the GPU -- submitting one prompt per call serializes everything and
    leaves the GPU idle most of the time. Results are still yielded (and can be persisted)
    one example at a time as each batch completes, so at most one batch of progress is lost
    on a crash. Each example yields ``num_samples_per_prompt_rollout`` completion strings.

    ``start_idx`` skips ahead to resume a partially-completed run (examples before it were
    already generated and persisted in a previous invocation).

    Mirrors the sampling setup in grpo_fast.create_generation_configs():
      - max_tokens = streaming_config.response_length
      - temperature = streaming_config.temperature
      - top_p = vllm_config.vllm_top_p
      - stop = streaming_config.stop_strings
      - n = streaming_config.num_samples_per_prompt_rollout
    """
    import vllm  # noqa: PLC0415

    max_model_len = streaming_config.max_prompt_token_length + streaming_config.response_length

    logger.info(f"Loading {model_name_or_path} with vLLM (max_model_len={max_model_len})...")
    llm = vllm.LLM(
        model=model_name_or_path,
        revision=model_revision,
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
        include_stop_str_in_output=True,
    )

    n = min(max_examples, len(dataset))
    logger.info(
        f"Generating {n} prompts x {num_samples} sample(s) in batches of {batch_size} "
        f"(temperature={streaming_config.temperature}, "
        f"max_tokens={streaming_config.response_length}, "
        f"stop={streaming_config.stop_strings})" + (f", resuming from example {start_idx}..." if start_idx else "...")
    )

    for batch_start in range(start_idx, n, batch_size):
        batch_indices = range(batch_start, min(batch_start + batch_size, n))
        prompt_token_ids = [prompt_token_ids_for(dataset[i], tokenizer, effective_system) for i in batch_indices]
        if batch_start == 0 and effective_system is not None:
            logger.info(f"Reconstructed prompt (example 0):\n{tokenizer.decode(prompt_token_ids[0])}")
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for ids in prompt_token_ids], sampling_params=sampling_params
        )
        for i, output in zip(batch_indices, outputs):
            texts = [tokenizer.decode(o.token_ids, skip_special_tokens=False) for o in output.outputs]
            yield i, texts


def generate_with_transformers(
    dataset,
    model_name_or_path: str,
    tokenizer,
    max_examples: int,
    max_new_tokens: int,
    temperature: float,
    effective_system: str | None = None,
    start_idx: int = 0,
    model_revision: str | None = None,
) -> "Iterator[tuple[int, list[str]]]":
    """Yield ``(idx, [completion_text])`` one dataset example at a time (local HF model).

    ``start_idx`` skips ahead to resume a partially-completed run.
    """
    logger.info(f"Loading model {model_name_or_path} for local generation...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, revision=model_revision, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    n = min(max_examples, len(dataset))

    logger.info(
        f"Generating {n} completions, one at a time (temperature={temperature}, max_new_tokens={max_new_tokens})"
        + (f", resuming from example {start_idx}..." if start_idx else "...")
    )
    for i in range(start_idx, n):
        input_ids = prompt_token_ids_for(dataset[i], tokenizer, effective_system)
        if i == 0 and effective_system is not None:
            logger.info(f"Reconstructed prompt (example 0):\n{tokenizer.decode(input_ids)}")
        input_tensor = torch.tensor([input_ids], device=model.device)

        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                # pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output[0]  # [len(input_ids):]
        text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        yield i, [text]


def _rows_for_example(idx: int, texts: list[str], dataset) -> list[dict]:
    """Build the output rows for one dataset example.

    Embeds the task prompt and the verifier ground truth + source so downstream stages
    (grading, judging) are self-contained: they don't need to reload the dataset or rely
    on row ordering matching.
    """
    row = dataset[idx]
    prompt = row[RAW_PROMPT_KEY]
    ground_truth = row[GROUND_TRUTHS_KEY]
    verifier_source = row[VERIFIER_SOURCE_KEY]
    return [
        {"idx": idx, "prompt": prompt, "completion": text, "ground_truth": ground_truth, "dataset": verifier_source}
        for text in texts
    ]


def make_run_dir(output_dir: str, run_subdir: str | None = None) -> str:
    """Create and return the run subdirectory under ``output_dir``.

    With ``run_subdir`` unset, a fresh timestamped name is generated (the normal case). With
    ``run_subdir`` set, that exact name is used instead -- this is what makes the directory
    reusable across invocations so a run can be resumed (see ``determine_resume_state``).
    """
    name = run_subdir or f"{time.strftime('%Y%m%d_%H%M%S')}_{random.randint(0, 999_999):06d}"
    run_dir = os.path.join(output_dir, name)
    os.makedirs(run_dir, exist_ok=run_subdir is not None)
    return run_dir


def determine_resume_state(rollouts_path: str, expected_samples_per_idx: int) -> tuple[int, list[dict]]:
    """Inspect an existing ``rollouts.jsonl`` and determine where to resume generation.

    Rows are written to disk in increasing ``idx`` order, flushed after each example's rows are
    all written, so a crash can only leave the *last* idx's rows incomplete. Returns
    ``(resume_idx, rows_to_keep)``: rows for the last idx are dropped (and generation restarts
    at that idx) if there are fewer of them than ``expected_samples_per_idx``.
    """
    if not os.path.exists(rollouts_path):
        return 0, []
    rows = []
    with open(rollouts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return 0, []
    last_idx = rows[-1]["idx"]
    last_idx_rows = [r for r in rows if r["idx"] == last_idx]
    if len(last_idx_rows) < expected_samples_per_idx:
        rows = [r for r in rows if r["idx"] != last_idx]
        return last_idx, rows
    return last_idx + 1, rows


def write_args(
    run_dir: str,
    script_args: argparse.Namespace,
    args: grpo_utils.GRPOExperimentConfig,
    tc: TokenizerConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    tools_config: EnvsConfig,
) -> None:
    """Dump every argument this run was invoked with to ``run_dir/args.json``.

    No-ops if ``args.json`` already exists (a resumed run), so the file always reflects the
    arguments of the original invocation that created the run directory.
    """
    args_path = os.path.join(run_dir, "args.json")
    if os.path.exists(args_path):
        return
    payload = {
        "script_args": vars(script_args),
        "args": dataclasses.asdict(args),
        "tokenizer_config": dataclasses.asdict(tc),
        "streaming_config": dataclasses.asdict(streaming_config),
        "vllm_config": dataclasses.asdict(vllm_config),
        "tools_config": dataclasses.asdict(tools_config),
    }
    with open(args_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def main(
    script_args: argparse.Namespace,
    args: grpo_utils.GRPOExperimentConfig,
    tc: TokenizerConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    tools_config: EnvsConfig,
) -> None:
    run_dir = make_run_dir(script_args.output_dir, script_args.run_subdir)
    write_args(run_dir, script_args, args, tc, streaming_config, vllm_config, tools_config)
    logger.info(f"Run directory: {run_dir}")

    # --model_name_or_path doubles as the tokenizer source when --tokenizer_name_or_path
    # is not supplied separately.
    tokenizer = common.resolve_tokenizer(tc, script_args.model_name_or_path)

    train_dataset = common.load_rollout_dataset(
        script_args, args, tc, streaming_config, tokenizer, needs_tokenized_prompts=True
    )

    # When a suffix is set, prompts are re-rendered with this system content (else None =
    # use the dataset's baked input_ids_prompt unchanged).
    effective_system = build_effective_system(streaming_config, script_args.system_prompt_suffix)
    if effective_system is not None:
        logger.info(f"Re-tokenizing prompts with system prompt:\n#####\n{effective_system}\n#####")

    # If --run_subdir points at a directory left over from a previous (possibly interrupted)
    # invocation, pick up where it left off instead of regenerating already-completed examples.
    expected_samples_per_idx = streaming_config.num_samples_per_prompt_rollout if script_args.generate_with_vllm else 1
    rollouts_path = os.path.join(run_dir, "rollouts.jsonl")
    start_idx, kept_rows = determine_resume_state(rollouts_path, expected_samples_per_idx)
    if start_idx:
        logger.info(
            f"Resuming from {rollouts_path}: {len(kept_rows)} completions already present, "
            f"continuing from example {start_idx}."
        )

    if script_args.generate_with_vllm:
        if script_args.model_name_or_path is None:
            raise ValueError("--model_name_or_path is required with --generate_with_vllm")
        row_stream = generate_with_vllm(
            dataset=train_dataset,
            model_name_or_path=script_args.model_name_or_path,
            tokenizer=tokenizer,
            max_examples=script_args.max_examples,
            streaming_config=streaming_config,
            vllm_config=vllm_config,
            seed=args.seed,
            effective_system=effective_system,
            batch_size=script_args.generation_batch_size,
            start_idx=start_idx,
            model_revision=script_args.model_revision,
        )
    else:
        if script_args.model_name_or_path is None:
            raise ValueError("--model_name_or_path is required with --generate_with_transformers")
        row_stream = generate_with_transformers(
            train_dataset,
            script_args.model_name_or_path,
            tokenizer,
            max_examples=script_args.max_examples,
            max_new_tokens=streaming_config.response_length * 4,
            temperature=streaming_config.temperature,
            effective_system=effective_system,
            start_idx=start_idx,
            model_revision=script_args.model_revision,
        )

    # Generate one dataset example at a time and append its completions to the output
    # file immediately, flushing after each example so partial progress survives a crash.
    logger.info(f"Writing completions incrementally to {rollouts_path}...")
    n_written = 0
    with open(rollouts_path, "w") as f:
        for row in kept_rows:
            f.write(json.dumps(row) + "\n")
        for idx, texts in row_stream:
            for row in _rows_for_example(idx, texts, train_dataset):
                f.write(json.dumps(row) + "\n")
                n_written += 1
            f.flush()
            if idx < 5:
                logger.info(f"  example {idx}: {texts[0][:300]}")

    logger.info(
        f"Done. Wrote {n_written} new completions ({len(kept_rows) + n_written} total) to {rollouts_path}. "
        "Score them with scripts/grade_rollouts.py."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--generate_with_vllm", action="store_true")
    parser.add_argument("--generate_with_transformers", action="store_true")
    # Optional: doubles as tokenizer source when --tokenizer_name_or_path is absent
    parser.add_argument("--model_name_or_path", type=str, default="allenai/Olmo-3.1-32B-Think")
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help=(
            "Revision of the model to load from the Hugging Face Hub (branch, tag, or commit hash). "
            "For example, intermediate RLVR checkpoints of OLMo-2-0325-32B-Instruct are accessible "
            "as revisions such as 'step_200'. Defaults to the latest revision on the main branch."
        ),
    )
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=64,
        help=(
            "Number of prompts submitted to vLLM's generate() call at once (vLLM backend only). "
            "Larger values let vLLM's continuous batching scheduler keep more sequences in flight "
            "concurrently, which is much faster than one prompt per call. Only affects crash-recovery "
            "granularity (progress is flushed as each batch completes), not output correctness. Tune "
            "down if you hit GPU OOM with a long --response_length."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generate_rollouts",
        help=(
            "Directory under which a fresh timestamped subdirectory is created for this run, "
            "containing args.json (the run's arguments) and rollouts.jsonl (the completions)."
        ),
    )
    parser.add_argument(
        "--run_subdir",
        type=str,
        default=None,
        help=(
            "Name of the run subdirectory under --output_dir, overriding the default fresh "
            "timestamped name. If this subdirectory already exists and contains a rollouts.jsonl "
            "from a previous invocation, generation resumes from the first example index not yet "
            "completed instead of starting over."
        ),
    )
    parser.add_argument(
        "--system_prompt_suffix",
        type=str,
        default=None,
        help=(
            "If set, re-tokenize each prompt with this text appended to the system prompt "
            "(base = --system_prompt_override_file, else empty). Required because the dataset ships "
            "pre-tokenized; the user turn is reconstructed from the dataset's `prompt` column."
        ),
    )
    common.add_shared_dataset_args(parser)
    script_args, remaining = parser.parse_known_args()

    if not script_args.generate_with_transformers:
        # Default to vLLM if neither flag is specified, since it's the recommended backend.
        script_args.generate_with_vllm = True

    args, tc, streaming_config, vllm_config, tools_config = common.parse_oi_configs(remaining)

    main(script_args, args, tc, streaming_config, vllm_config, tools_config)  # type: ignore[arg-type]
