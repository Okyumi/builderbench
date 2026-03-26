"""Generates the final JSON object linking results to environment metadata."""

import re
import json
import numpy as np
from pathlib import Path
from typing import Any

from dataclasses import dataclass
from simple_parsing import ArgumentParser
from builderbench.create_task_data import *

@dataclass
class Args:
    results_dir: str
    level_id: str
    model_id: str = "naive"
    agent_name: str = "Qwen3-235B-A22B-Instruct-2507"
    website_url: str = "N/A"

RUN_DIR_PATTERN = re.compile(r"^(?P<task>.+)-seed-(?P<seed>\d+).*$")

def verify_consistency(runs: list[dict[str, Any]]) -> None:
    """
    Sanity Check: Ensures all seeds ran on the exact same task version,
    code commit, and model configuration.
    """
    if not runs:
        return

    first_run = runs[0]
    first_meta = first_run.get("run_metadata")
    if not first_meta:
        raise ValueError(f"Missing metadata for Run: {first_run}")

    ref_sha = first_meta.get("task_version", {}).get("task_file_sha256")
    ref_commit = first_meta.get("builderbench_git_commit")
    ref_model = first_meta.get("model", {}).get("model_id")

    for run in runs[1:]:
        meta = run.get("run_metadata")
        if not meta:
            raise ValueError(f"Missing metadata for Run: {run}")

        # Check Task Version
        curr_sha = meta.get("task_version", {}).get("task_file_sha256")
        if curr_sha != ref_sha:
            raise ValueError(
                f"Consistency Error! Seed {run['seed']} used task SHA {curr_sha}, "
                f"but seed {first_run['seed']} used {ref_sha}. "
                "Cannot aggregate results from different task versions."
            )

        # Check Git Commit
        curr_commit = meta.get("builderbench_git_commit")
        if curr_commit != ref_commit:
            raise ValueError(
                f"Consistency Error! Seed {run['seed']} used commit {curr_commit}, "
                f"but seed {first_run['seed']} used {ref_commit}."
             )
        # Check Model ID
        curr_model = meta.get("model", {}).get("model_id")
        if curr_model != ref_model:
            raise ValueError(
                f"Consistency Error! Seed {run['seed']} used model {curr_model}, "
                f"but seed {first_run['seed']} used {ref_model}."
            )

def load_eval_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    chunks = [chunk for chunk in content.split("\n\n") if chunk.strip()]
    return [json.loads(chunk) for chunk in chunks]

def collect_task_runs(results_dir: Path, task_name: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(results_dir.glob(f"{task_name}-seed-*/")):
        if not run_dir.is_dir():
            continue
        match = RUN_DIR_PATTERN.match(run_dir.name)
        if not match:
            continue
        seed = int(match.group("seed"))

        eval_summary = load_eval_summary(run_dir / "eval_summary.jsonl")
        run_config_path = run_dir / "run_config.json"
        run_metadata_path = run_dir / "run_metadata.json"

        run_config = (
            json.loads(run_config_path.read_text(encoding="utf-8"))
            if run_config_path.exists()
            else None
        )
        run_metadata = (
            json.loads(run_metadata_path.read_text(encoding="utf-8"))
            if run_metadata_path.exists()
            else None
        )

        runs.append(
            {
                "task": task_name,
                "seed": seed,
                "path": str(run_dir),
                "eval_summary": eval_summary,
                "run_config": run_config,
                "run_metadata": run_metadata,
            }
        )
    return runs


def compute_task_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def compute_progress(per_cube_success, per_cube_mask):
        if per_cube_mask:
            filtered = [s for s, m in zip(per_cube_success, per_cube_mask) if m]
            total = sum(1 for m in per_cube_mask if m)
        else:
            filtered = per_cube_success
            total = len(per_cube_success)
        return (sum(filtered) / total) if total else 0.0

    per_seed = []
    for run in runs:
        episodes = run["eval_summary"]
        if not episodes:
            continue
                
        top1_success = any(bool(ep.get("success", False)) for ep in episodes)

        max_progress_in_run = 0.0
        num_input_tokens = 0
        num_output_tokens = 0
    
        for episode in episodes:
            num_input_tokens += int(episode.get("tokens_in", 0) or 0)
            num_output_tokens += int(episode.get("tokens_out", 0) or 0)

            per_cube_success = episode.get("per_cube_success", [])
            per_cube_mask = episode.get("per_cube_mask")
            progress = compute_progress(per_cube_success, per_cube_mask)
            max_progress_in_run = max(max_progress_in_run, progress)

        final_ep = episodes[-1]
        final_success = bool(final_ep.get("success", False))
        final_pcs = final_ep.get("per_cube_success", [])
        final_masks = final_ep.get("per_cube_mask")
        final_progress = compute_progress(final_pcs, final_masks)

        per_seed.append(
            {
                "seed": run.get("seed"),
                "top1_success": top1_success,
                "top1_progress": max_progress_in_run,
                "final_success": final_success,
                "final_progress": final_progress,
                "input_tokens": num_input_tokens,
                "output_tokens": num_output_tokens,
                "num_episodes": len(episodes),
            }
        )

    num_seeds = len(per_seed)
    if num_seeds == 0:
        return {"num_seeds": 0}

    # Aggregate
    mean_final_success = np.mean( [x["final_success"] for x in per_seed] )
    mean_top1_success = np.mean( [x["top1_success"] for x in per_seed] )
    mean_final_progress = np.mean( [x["final_progress"] for x in per_seed] )
    mean_top1_progress = np.mean( [x["top1_progress"] for x in per_seed] )
    mean_input_tokens_per_run = np.mean( [x["input_tokens"] for x in per_seed] )
    mean_output_tokens_per_run = np.mean( [x["output_tokens"] for x in per_seed] )
    mean_num_episodes = np.mean( [x["num_episodes"] for x in per_seed] )

    std_final_success = np.std( [x["final_success"] for x in per_seed] )
    std_top1_success = np.std( [x["top1_success"] for x in per_seed] )
    std_final_progress = np.std( [x["final_progress"] for x in per_seed] )
    std_top1_progress = np.std( [x["top1_progress"] for x in per_seed] )
    std_input_tokens_per_run = np.std( [x["input_tokens"] for x in per_seed] )
    std_output_tokens_per_run = np.std( [x["output_tokens"] for x in per_seed] )
    std_num_episodes = np.std( [x["num_episodes"] for x in per_seed] )

    return {
        "num_seeds": num_seeds,
        "mean_final_success_rate": mean_final_success,
        "mean_top1_success_rate": mean_top1_success,
        "mean_final_progress": mean_final_progress,
        "mean_top1_progress": mean_top1_progress,
        "mean_num_episodes": mean_num_episodes,
        "mean_input_tokens_per_run": mean_input_tokens_per_run,
        "mean_output_tokens_per_run": mean_output_tokens_per_run,
        "std_final_success_rate": std_final_success,
        "std_top1_success_rate": std_top1_success,
        "std_final_progress": std_final_progress,
        "std_top1_progress": std_top1_progress,
        "std_num_episodes": std_num_episodes,
        "std_input_tokens_per_run": std_input_tokens_per_run,
        "std_output_tokens_per_run": std_output_tokens_per_run,
    }

def main(config: Args) -> int:
    results_dir = ( Path(config.results_dir) / config.agent_name / config.model_id ).expanduser().resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")
        
    runs = collect_task_runs(results_dir, config.level_id)
    if not runs:
        raise SystemExit(f"No runs found for task '{config.level_id}' in {results_dir}")
    print(f"Found {len(runs)} run(s) for {config.level_id} in {results_dir}")

    try:
        verify_consistency(runs)
        print("Consistency check passed (Task SHA, Commit, Model ID match).")
    except ValueError as e:
        print(f"Consistency Check Failed: {e}")
        return 1

    metrics = compute_task_metrics(runs)
    if metrics["num_seeds"] == 0:
        print("No valid episodes found in runs.")
        return 0

    meta = runs[0].get("run_metadata", {})
    
    # Safely extract nested keys
    task_ver = meta.get("task_version", {})
    deps = meta.get("dependency_versions", {})
    model_info = meta.get("model", {})
    
    leaderboard = {
        "leaderboard_results": {
            "level_id": task_ver.get("level_id", "unknown"),
            "agent_name": config.agent_name,
            "model_id": config.model_id,
            "website_url": config.website_url,
            "timestamp": meta.get("run_timestamp"),
            **metrics
        },
        "reproducibility": {
            "builderbench_git_commit": meta.get("builderbench_git_commit"),
            "task_version": {
                **task_ver,
            },
            "dependencies": {
                **deps
            },
        },
    }
    
    Path("tmp").mkdir(exist_ok=True)
    output_path = Path("tmp") / f"{config.level_id}-leaderboard.json"
    output_path.write_text(json.dumps(leaderboard, indent=4), encoding="utf-8")
    print(f"\nLeaderboard results saved to: {output_path}")

    return 0

if __name__ == "__main__":
    parser = ArgumentParser(description="Collect results for a task.")
    parser.add_arguments(Args, dest="config")
    args = parser.parse_args()

    raise SystemExit(main(args.config))