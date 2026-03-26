import os
import re
import sys
import json
import random
import hashlib
import platform
import subprocess
import numpy as np

import openai

from tabulate import tabulate
from pathlib import Path
from importlib import metadata
from dataclasses import asdict

from agents.client import resolve_sampling_params
from agents.base import resolve_agent_config

REPO_ROOT = Path(__file__).resolve().parent

def print_summary_table(output_path):
    episodes_path = os.path.join(output_path, "eval_summary.jsonl")
    results = []
    
    with open(episodes_path, "r") as f:
        # Split by double newline as used in write logic
        content = f.read().strip().split("\n\n")
        for entry in content:
            data = json.loads(entry)

            per_cube_success = data["per_cube_success"]
            per_cube_easy_success = data["per_cube_easy_success"]
            per_cube_mask = data.get("per_cube_mask")
            if per_cube_mask:
                masked_success = [s for s, m in zip(per_cube_success, per_cube_mask) if m]
                masked_easy_success = [s for s, m in zip(per_cube_easy_success, per_cube_mask) if m]
                total_cubes = sum(1 for m in per_cube_mask if m)
            else:
                masked_success = per_cube_success
                masked_easy_success = per_cube_easy_success
                total_cubes = len(per_cube_success)

            num_success = sum(masked_success)
            num_easy_success = sum(masked_easy_success)
            progress = f"{num_success}/{total_cubes}"
            easy_progress = f"{num_easy_success}/{total_cubes}"
            
            results.append([
                data["episode_idx"],
                "\U00002705" if data["success"] else "\U0000274C",
                progress,
                "\U00002705" if data["easy_success"] else "\U0000274C",
                easy_progress,
                data["steps"],
                data["num_actions"],
                data["tokens_in"],
                data["tokens_out"],
                data["episode_time"]
            ])

    headers = ["Episode", "Success", "Progress", "Success Easy ", "Progress Easy", "Length", "Actions", "In Tokens", "Out Tokens", "Time"]
    print("\n"*2)
    print("="*30 + " EVALUATION SUMMARY " + "="*30)
    print(tabulate(results, headers=headers, tablefmt="rounded_grid"))
    print("="*80)
    
def seed_everything(seed: int):
    """Seed Python, NumPy, and available ML frameworks for reproducibility."""
    seed_info = {
        "seed": seed,
        "python_random_seed": seed,
        "numpy_seed": seed,
        "torch_seed": None,
        "torch_cuda_seed": None,
    }

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seed_info["torch_cuda_seed"] = seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seed_info["torch_seed"] = seed
    except Exception:
        pass

    return seed_info

def get_experiment_data(config, timestamp, global_seed_info):
    run_metadata = {
        "run_timestamp": timestamp,
        "builderbench_git_commit": get_git_commit_hash(REPO_ROOT),
        "task_version": resolve_task_version(config.level_id),
        "dependency_versions": get_dependency_versions(),
        "model": {
            "model_id": config.llm_config.model_id,
            "client_name": config.llm_config.client_name,
            "base_url": config.llm_config.base_url,
        },
        "global_seed": global_seed_info,
    }

    run_config = {
        "cli_args": sys.argv[1:],
        "config": asdict(config),
        "resolved_generate_kwargs": resolve_sampling_params(
            model_id=config.llm_config.model_id,
            generate_kwargs=config.llm_config.generate_kwargs,
        ),
        "resolved_agent_config": asdict(resolve_agent_config(config)),
    }

    return run_metadata, run_config
    
def file_sha256(path: Path):
    if not path.exists():
        return "missing"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_task_version(level_id: str):
    """Return task version metadata based on the task file and level id."""
    task_info = {
        "level_id": level_id,
        "task_file": None,
        "task_file_sha256": None,
    }
    try:
        cube_count = int( re.search(r"cube-(\d+)", level_id).group(1))
        task_file = REPO_ROOT / "builderbench" / "tasks" / f"cube-{cube_count}.npz"
        task_info.update(
            {
                "task_file": str(task_file),
                "task_file_sha256": file_sha256(task_file),
            }
        )
    except Exception:
        task_info["task_file_sha256"] = "unknown"
    return task_info

def get_git_commit_hash(repo_path: Path):
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return "unknown"

def get_dependency_versions():
    """Return a dict of Python version, platform info, and key package versions."""
    packages = [
        "numpy",
        "mujoco",
        "gymnasium",
        "dm-control",
        "scipy",
        "openai",
        "anthropic",
        "simple-parsing",
        "mediapy",
        "tabulate",
        "tqdm",
        "lxml",
    ]
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
        except Exception:
            versions[pkg] = "unknown"
    return versions

def load_secrets(file_path):
    """Load secrets from a file with key-value pairs.

    Args:
        file_path (str): Path to the secrets file.

    Returns:
        dict: A dictionary of secrets with keys and values.
    """
    secrets = {}
    if not os.path.exists(file_path):
        return secrets
    with open(file_path) as f:
        for line in f:
            if not line.strip() or line.strip().startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            secrets[key] = value
    return secrets


def setup_environment_variables(
    openai_tag: str = "OPENAI_API_KEY",
    gemini_tag: str = "GEMINI_API_KEY",
    anthropic_tag: str = "ANTHROPIC_API_KEY",
    organization: str = None,
    original_cwd: str = "",
):
    """Set up environment variables for API keys.

    Args:
        openai_tag (str): Environment variable tag for OpenAI API key.
        gemini_tag (str): Environment variable tag for Gemini API key.
        anthropic_tag (str): Environment variable tag for Anthropic API key.
        organization (str, optional): Organization name for OpenAI. Defaults to None.
        original_cwd (str, optional): Original working directory. Defaults to "".
    """
    secrets = load_secrets(os.path.join(original_cwd, "SECRETS"))
    if secrets.get(openai_tag):
        os.environ["OPENAI_API_KEY"] = secrets[openai_tag]
    if secrets.get(gemini_tag):
        os.environ["GEMINI_API_KEY"] = secrets[gemini_tag]
    if secrets.get(anthropic_tag):
        os.environ["ANTHROPIC_API_KEY"] = secrets[anthropic_tag]
    if organization is not None and secrets.get(organization):
        openai.organization = secrets[organization]