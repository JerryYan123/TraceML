"""Auto-detect the shape of an agent run directory.

Supported shapes:

1. ``aide``: an AIDE/MLEvolve tree-search journal (``logs/{filtered_,}journal.json``)
   anywhere under the run directory.
2. ``codex-tags``: a git sidecar with grading tags ``v<N>-true-<score>`` /
   ``v<N>-na``, as written by ``traceml record`` or the Codex grading shim.
   Scores come from the tags, so no re-grading is needed.
3. ``codex-regrade``: a git sidecar that tracks ``submission.csv`` but has no
   grading tags. Every submission is re-graded with ``mlebench grade-sample``.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from traceml_toolkit.extract import parse_tag

SKIP_DIRS = {".git", "cache", "__pycache__", "_bin", "extracted", "workspace"}


class UnsupportedRunLayout(ValueError):
    """The directory is not one of the supported run shapes."""


def _find_journal(run_dir: Path) -> Path | None:
    stack = [run_dir]
    while stack:
        d = stack.pop()
        try:
            children = sorted(d.iterdir())
        except OSError:
            continue
        for c in children:
            if c.is_dir():
                if c.name not in SKIP_DIRS:
                    stack.append(c)
            elif c.name in ("filtered_journal.json", "journal.json"):
                return c
    return None


def _git_tags(run_dir: Path) -> list[str]:
    r = subprocess.run(["git", "-C", str(run_dir), "tag"],
                       capture_output=True, text=True, timeout=120)
    return r.stdout.splitlines() if r.returncode == 0 else []


def infer_slug(run_dir: Path) -> str | None:
    """Competition slug from run_meta.json, else from the run directory name."""
    meta = Path(run_dir) / "run_meta.json"
    if meta.exists():
        try:
            slug = json.loads(meta.read_text()).get("competition")
            if slug:
                return slug
        except Exception:
            pass
    # Run directory names end with the competition slug after the tag segment.
    m = re.search(r"_(?:[a-z0-9]+_)*?([a-z0-9][a-z0-9-]{8,})$", Path(run_dir).name)
    return m.group(1) if m else None


def detect(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise UnsupportedRunLayout(f"not a directory: {run_dir}")
    jp = _find_journal(run_dir)
    if jp is not None:
        return {"layout": "aide", "journal": str(jp), "slug": infer_slug(run_dir)}
    if (run_dir / ".git").exists():
        if shutil.which("git") is None:
            raise UnsupportedRunLayout("this run has a git sidecar but the `git` executable is not on PATH")
        parsed = [p for p in (parse_tag(t) for t in _git_tags(run_dir)) if p is not None]
        n_scored = sum(1 for _, score in parsed if score is not None)
        if n_scored:
            return {"layout": "codex-tags", "n_tags": len(parsed), "n_scored_tags": n_scored,
                    "slug": infer_slug(run_dir)}
        # No scored grading tag at all (none written, or every grading failed):
        # fall back to re-grading the committed submissions.
        return {"layout": "codex-regrade", "n_tags": len(parsed), "slug": infer_slug(run_dir)}
    raise UnsupportedRunLayout(
        f"{run_dir}: unsupported run layout. Supported shapes:\n"
        "  1. AIDE/MLEvolve journal (logs/journal.json) anywhere under the run dir\n"
        "  2. git sidecar with v<N>-true-<score> grading tags (traceml record, Codex shim)\n"
        "  3. git sidecar tracking submission.csv (re-graded with mlebench; needs --slug)")
