"""Build the labeler input records from an extracted trajectory.

State records: one per version, holding the version's main file.
Action records: one per adjacent pair of versions, holding a unified diff and
the two state labels, so state labeling must run first.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

MAX_DIFF_CHARS = 8000


def run_key_id(traj: dict) -> str:
    """Trajectory id used in label rows and parquet tables: ``<harness>_<run name>``."""
    name = traj.get("run_name") or Path(traj["run_dir"]).name
    return f"{traj.get('harness', 'agent')}_{name}"


def make_diff(a: str, b: str) -> tuple[str, int, int]:
    """(unified diff truncated to MAX_DIFF_CHARS, lines added, lines removed)."""
    lines = list(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                      fromfile="old.py", tofile="new.py", n=2))
    n_added = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    n_removed = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
    return "".join(lines)[:MAX_DIFF_CHARS], n_added, n_removed


def code_text(extract_dir: Path, version: dict, full: bool = False) -> str:
    """Main file of a version, or the whole-codebase concatenation when ``full``.

    State labeling uses the single main file (the released pipeline's input,
    which fits the state model's context); action diffs default to the whole
    codebase so edits outside the main file are visible.
    """
    keys = ("code_all_path", "code_path") if full else ("code_path",)
    for key in keys:
        if version.get(key):
            p = Path(extract_dir) / version[key]
            if p.exists():
                return p.read_text(errors="ignore")
    return ""


def build_state_records(extract_dir: Path, traj: dict) -> list[dict]:
    kid = run_key_id(traj)
    comp = traj.get("competition") or "unknown"
    out = []
    for v in sorted(traj["versions"], key=lambda v: v["version_number"]):
        code = code_text(extract_dir, v)
        if not code.strip():
            continue
        out.append({
            "key_id": kid,
            "comp": comp,
            "group": traj.get("harness", "agent"),
            "version_number": int(v["version_number"]),
            "code_text": code,
            "code_lines": code.count("\n") + 1,
            "node_id": f"{kid}_v{v['version_number']}",
        })
    return out


def build_action_records(extract_dir: Path, traj: dict, state_rows: list[dict],
                         diff_scope: str = "full") -> list[dict]:
    """One record per adjacent version pair.

    ``diff_scope='full'`` diffs the whole codebase (default); ``'main'`` diffs
    only the main file, reproducing the input the released labels were built on.
    """
    if diff_scope not in ("full", "main"):
        raise ValueError(f"diff_scope must be 'full' or 'main', got {diff_scope!r}")
    state_lookup = {}
    for j in state_rows:
        if j.get("key_id") is not None and j.get("version_number") is not None:
            state_lookup[(j["key_id"], int(j["version_number"]))] = j
    kid = run_key_id(traj)
    comp = traj.get("competition") or "unknown"
    versions = sorted(traj["versions"], key=lambda v: v["version_number"])
    total = len(versions)
    out = []
    prev, prev_code = None, ""
    for idx, v in enumerate(versions):
        cur_code = code_text(extract_dir, v, full=(diff_scope == "full"))
        if prev is not None:
            score_old = prev.get("linked_score")
            score_new = v.get("linked_score")
            delta = (score_new - score_old) if (score_new is not None and score_old is not None) else None
            diff, n_added, n_removed = make_diff(prev_code, cur_code)
            out.append({
                "key_id": kid, "comp": comp, "group": traj.get("harness", "agent"),
                "kind": "agent",
                "v_old": int(prev["version_number"]), "v_new": int(v["version_number"]),
                "version_old": int(prev["version_number"]), "version_new": int(v["version_number"]),
                "code_diff": diff,
                "n_added": n_added,
                "n_removed": n_removed,
                "atoms": [],
                "state_old": state_lookup.get((kid, int(prev["version_number"])), {}),
                "state_new": state_lookup.get((kid, int(v["version_number"])), {}),
                "score_old": score_old, "score_new": score_new, "score_delta": delta,
                "position_in_trajectory": idx / max(total - 1, 1),
                "total_versions": total,
            })
        prev, prev_code = v, cur_code
    return out


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for ln in Path(path).read_text().splitlines():
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
