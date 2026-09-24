"""`traceml record`: run any CLI agent under the TraceML recorder."""
from __future__ import annotations

import os
import shutil
from importlib.resources import as_file, files
from pathlib import Path

from traceml_toolkit import config

DEFAULT_CONT = (
    "Continue the ML competition run in this directory. Re-read task.md for the rules and the "
    "time budget, inspect the code and results you already have, and keep improving "
    "submission.csv.\n"
)


def script_path() -> Path:
    with as_file(files("traceml_toolkit") / "record" / "traceml_record.sh") as p:
        return Path(p)


def record(run_dir: str | Path, slug: str, minutes: int, harness: str, agent_cmd: list[str], *,
           mlebench: str | None = None, mlebench_cache: str | Path | None = None,
           poll: int | None = None) -> None:
    """Replace this process with the recorder shell loop (does not return)."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory {run_dir} does not exist")
    if not (run_dir / "task.md").is_file():
        raise FileNotFoundError(f"{run_dir}/task.md is missing: write the task prompt there first")
    if not any("{PROMPT}" in tok for tok in agent_cmd):
        raise ValueError("the agent command must contain the literal token {PROMPT}")
    if not (run_dir / "cont.md").is_file():
        (run_dir / "cont.md").write_text(DEFAULT_CONT)
        print(f"[traceml] wrote a default continuation prompt to {run_dir / 'cont.md'}")
    if shutil.which("bash") is None:
        raise RuntimeError("bash is required for `traceml record`")
    env = os.environ.copy()
    env["TRACEML_MLEBENCH"] = config.mlebench_bin(mlebench)
    cache = config.mlebench_cache(mlebench_cache)
    if cache is None:
        raise ValueError("pass --mlebench-cache (or set TRACEML_MLEBENCH_CACHE) to a prepared MLE-bench data dir")
    env["TRACEML_MLEBENCH_CACHE"] = str(cache)
    if poll:
        env["TRACEML_POLL"] = str(poll)
    args = ["bash", str(script_path()), str(run_dir), slug, str(int(minutes)), harness, "--", *agent_cmd]
    os.execvpe("bash", args, env)
