"""
Plot mean reward (+/- standard error) as a bar chart across all run subdirs in generate_rollouts/.

For any subdir missing a rewards.jsonl, this shells out to scripts/grade_rollouts.py to produce
one first (passing through any extra args, e.g. dataset config).

Usage:
    python scripts/plot_rollout_rewards.py [--rollouts_dir generate_rollouts] [--output rewards.png]
"""

import argparse
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

GRADE_ROLLOUTS_SCRIPT = os.path.join(os.path.dirname(__file__), "grade_rollouts.py")


def load_rewards(rewards_jsonl: str) -> list[float]:
    rewards = []
    with open(rewards_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rewards.append(json.loads(line)["reward"])
    return rewards


def main(args: argparse.Namespace, extra_args: list[str]) -> None:
    run_dirs = sorted(
        d
        for d in os.listdir(args.rollouts_dir)
        if os.path.isdir(os.path.join(args.rollouts_dir, d))
        and os.path.exists(os.path.join(args.rollouts_dir, d, "rollouts.jsonl"))
    )
    if not run_dirs:
        logger.error(f"No run subdirs with rollouts.jsonl found under {args.rollouts_dir}")
        sys.exit(1)

    labels, means, sems, rewards_by_run = [], [], [], []
    for run_dir in run_dirs:
        run_path = os.path.join(args.rollouts_dir, run_dir)
        rewards_path = os.path.join(run_path, "rewards.jsonl")
        if not os.path.exists(rewards_path):
            logger.info(f"No rewards.jsonl for {run_dir}, running grade_rollouts.py...")
            subprocess.run([sys.executable, GRADE_ROLLOUTS_SCRIPT, "--run_dir", run_path, *extra_args], check=True)

        rewards = load_rewards(rewards_path)
        labels.append(run_dir)
        means.append(np.mean(rewards))
        sems.append(np.std(rewards, ddof=1) / np.sqrt(len(rewards)))
        rewards_by_run.append(rewards)

    # First run subdir (alphabetically) is treated as the baseline; every other run is compared
    # against it with a two-tailed Welch's t-test (does not assume equal population variances).
    baseline_rewards = rewards_by_run[0]
    p_values = [None] + [
        stats.ttest_ind(baseline_rewards, rewards, equal_var=False).pvalue for rewards in rewards_by_run[1:]
    ]
    for label, p in zip(labels[1:], p_values[1:]):
        logger.info(f"{label} vs baseline ({labels[0]}): p={p:.4g}")

    logger.info("Plotting bar chart...")
    plt.figure(figsize=(max(8, len(labels) * 1.5), 6))
    plt.bar(labels, means, yerr=sems, capsize=5, color="skyblue", edgecolor="black")
    plt.ylabel("Mean reward")
    plt.title(f"Mean reward (+/- SEM) by rollout run\n(p-values vs baseline: {labels[0]})")
    plt.xticks(rotation=30, ha="right")

    y_max = max(m + s for m, s in zip(means, sems))
    for i, (mean, sem, p) in enumerate(zip(means, sems, p_values)):
        if p is not None:
            plt.text(i, mean + sem + 0.02 * y_max, f"p={p:.3g}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(args.output)
    logger.info(f"Saved plot to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollouts_dir", type=str, default="generate_rollouts", help="Directory containing run subdirs."
    )
    parser.add_argument("--output", type=str, default="rewards.png", help="Path to save the bar chart to.")
    args, extra_args = parser.parse_known_args()

    main(args, extra_args)
