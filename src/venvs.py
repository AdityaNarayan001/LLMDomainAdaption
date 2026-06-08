"""Venv router — auto-switch to the right interpreter per stage.

The three stacks have mutually-exclusive pins (torch/transformers), so they live in
separate venvs and talk over subprocess/HTTP (never in-process imports across stacks):

  .venv        pipeline + training (Axolotl CPT/SFT) + data/eval/orchestration
  .venv-serve  vLLM (inference / rollout server, OpenAI-compatible endpoint)
  .venv-rl     veRL (RL trainer; GRPO/DAPO with our verifiable cargo reward) [-> .venv-verl]

`run_pipeline.sh` resolves the interpreter per stage via this module; the flywheel
orchestrator (running in .venv) shells out to .venv-rl for RL training and launches
the vLLM server from .venv-serve. Missing venvs fail loudly with a clear message.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from src import config

# venv directory names (under repo root)
VENV_DIRS = {"default": ".venv", "serve": ".venv-serve", "rl": ".venv-rl"}

# which venv each pipeline stage runs in
STAGE_VENV = {
    "ingest": "default", "census": "default", "data": "default",
    "cpt": "default", "sft": "default", "eval": "default",
    "flywheel": "default", "smoke": "default",
    "serve": "serve",
    "rl": "rl",
}


def interpreter(venv_key: str = "default") -> Path:
    """Absolute path to a venv's python (resolves an alias or a literal dir name)."""
    name = VENV_DIRS.get(venv_key, venv_key)
    return config.ROOT / name / "bin" / "python"


def python_for(stage: str) -> Path:
    return interpreter(STAGE_VENV.get(stage, "default"))


def exists(venv_key: str) -> bool:
    return interpreter(venv_key).exists()


def require(venv_key: str) -> Path:
    py = interpreter(venv_key)
    if not py.exists():
        raise FileNotFoundError(
            f"venv '{VENV_DIRS.get(venv_key, venv_key)}' not found at {py}. "
            f"Create it on the training box (see scripts/setup_env.sh)."
        )
    return py


def run_module(venv_key: str, module: str, *args: str, **popen_kw) -> subprocess.CompletedProcess:
    """Run `python -m <module> <args>` in the given venv (blocking). Auto-switch entrypoint."""
    return subprocess.run([str(require(venv_key)), "-m", module, *args],
                          cwd=str(config.ROOT), check=True, **popen_kw)


def launch_vllm(model: str, port: int = 8000, quantization: str | None = "modelopt_fp4",
                extra: tuple[str, ...] = ()) -> subprocess.Popen:
    """Start a vLLM OpenAI-compatible server from .venv-serve (NVFP4 by default). Non-blocking."""
    cmd = [str(require("serve")), "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--port", str(port)]
    if quantization:
        cmd += ["--quantization", quantization]
    cmd += list(extra)
    return subprocess.Popen(cmd, cwd=str(config.ROOT))


def run_rl(*args: str) -> subprocess.CompletedProcess:
    """Run the SkyRL RL trainer in .venv-rl (auto-switch from the .venv orchestrator)."""
    return run_module("rl", "src.train.rl", *args)
