"""Extraction: agent run directory -> ``<out>/extracted/{trajectory.json, versions/}``.

Three input shapes are supported (see :mod:`traceml_toolkit.detect`). Every
path is read-only on the run directory.

Code-state identity (git paths): a version is a distinct *filtered tree
fingerprint*, the md5 over sorted ``(path, blob_sha)`` of the ``.py`` and
``.ipynb`` files in a commit, excluding submissions, binaries, prompts and
logs. Config and metrics files are excluded on purpose: agents regenerate
metrics JSONs on every training run, which would inflate version counts
(2,505 grader calls collapse to 16 states on the paper's skill-rep1 commonlit
run, matching the released split exactly, versus 409 with configs included).
Consecutive grader calls on an unchanged tree merge into one version; their
scores are kept in ``agent_metadata`` and the best one is linked,
direction-aware. This mirrors a Kaggle "save version".
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

from traceml_toolkit import __version__, config

# Grading-shim tags written by the recorder: v<N>-true-<score> or v<N>-na.
TAG_RE = re.compile(r"^v(\d+)-(?:true-([-+0-9.eE]+)|na)$")

CODE_EXTS = {".py", ".ipynb"}
IGNORE_FILES = {"_cont_prompt.md", "prompt.source.md", "task.md", "cont.md", "native_session.log",
                "run_meta.json", "_skill_block.md"}
IGNORE_DIRS = {"cache", "_bin", "__pycache__", ".git", "extracted"}
IGNORE_PATH_PATTERNS = [
    re.compile(r"^submission.*\.csv$"),
    re.compile(r".*\.(pt|pth|bin|h5|npy|npz|parquet|feather)$"),
    re.compile(r"^candidate_.*\.csv$"),
    re.compile(r"^oof_.*\.csv$"),
]


def parse_tag(name: str) -> tuple[int, float | None] | None:
    """(n, score) for a grading tag, score None for ``-na`` or unparsable scores."""
    m = TAG_RE.match(name)
    if not m:
        return None
    score = None
    if m.group(2) is not None:
        try:
            score = float(m.group(2))
        except ValueError:
            return None
        if not math.isfinite(score):
            score = None
    return int(m.group(1)), score


def git(d: Path, *args: str, timeout: int = 300) -> str:
    return subprocess.run(["git", "-C", str(d), *args],
                          capture_output=True, text=True, timeout=timeout).stdout


def _git_show(run_dir: Path, sha: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(run_dir), "show", f"{sha}:{path}"],
                          capture_output=True).stdout


def _better(a: float, b: float, direction: str) -> bool:
    return a < b if direction == "lower" else a > b


def _resolve_direction(slug: str | None, direction: str | None) -> str:
    direction = direction or config.score_direction(slug)
    if direction is None:
        print(f"[traceml] WARNING: unknown score direction for competition {slug!r}; "
              "assuming higher is better (pass --direction to override)")
        direction = "higher"
    return direction


def _run_meta(run_dir: Path) -> dict:
    meta = run_dir / "run_meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text())
        except Exception:
            pass
    return {}


def _harness_of(run_dir: Path, default: str = "codex") -> str:
    return _run_meta(run_dir).get("harness", default)


def _header(run_dir: Path) -> dict:
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "toolkit_version": __version__,
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# ---------------------------------------------------------------------------
# git sidecar helpers
# ---------------------------------------------------------------------------
def _tags_with_commits(run_dir: Path):
    """[(tag_n, tag_name, score|None, commit_sha, commit_ts)] in grader-call order."""
    out = []
    lines = git(run_dir, "for-each-ref",
                "--format=%(refname:short)|%(objectname)|%(creatordate:unix)",
                "refs/tags").splitlines()
    for ln in lines:
        try:
            name, sha, ts = ln.split("|")
        except ValueError:
            continue
        parsed = parse_tag(name)
        if parsed is None:
            continue
        n, score = parsed
        out.append((n, name, score, sha, int(ts or 0)))
    out.sort()
    return out


def _tree_info(run_dir: Path, sha: str):
    """(fingerprint, code_files, largest_py) for a commit, filtered.

    fingerprint = md5 over sorted (path, blob_sha); largest_py is the path of the
    largest .py file, used as the version's "main file".
    """
    lines = git(run_dir, "ls-tree", "-r", "-l", sha).splitlines()
    keep = []
    largest = (None, -1)
    for ln in lines:
        try:
            meta, path = ln.split("\t", 1)
            parts = meta.split()
            obj_sha, size = parts[2], int(parts[3]) if parts[3] != "-" else 0
        except (ValueError, IndexError):
            continue
        pp = path.split("/")
        if pp[0] in IGNORE_DIRS or path in IGNORE_FILES:
            continue
        base = pp[-1]
        if any(p.match(base) for p in IGNORE_PATH_PATTERNS):
            continue
        ext = "." + base.rsplit(".", 1)[-1] if "." in base else ""
        if ext not in CODE_EXTS:
            continue
        keep.append((path, obj_sha))
        if ext == ".py" and size > largest[1]:
            largest = (path, size)
    h = hashlib.md5()
    for p, s in sorted(keep):
        h.update(p.encode())
        h.update(b"\0")
        h.update(s.encode())
        h.update(b"\0")
    return h.hexdigest(), [p for p, _ in keep], largest[0]


def _write_version_files(run_dir: Path, sha: str, code_files: list[str], largest_py: str | None,
                         out_dir: Path, k: int) -> tuple[str, str]:
    """Write versions/vNNN.py (main file) and versions/vNNN_all.txt (whole codebase)."""
    code_path = f"versions/v{k:03d}.py"
    if largest_py:
        (out_dir / code_path).write_bytes(_git_show(run_dir, sha, largest_py))
    else:
        (out_dir / code_path).write_text("")
    # Full-codebase view: every kept code file at this commit, so action
    # labeling sees edits outside the main file too.
    parts = [f"# === {cf} ===\n".encode() + _git_show(run_dir, sha, cf) + b"\n" for cf in code_files]
    full_path = f"versions/v{k:03d}_all.txt"
    (out_dir / full_path).write_bytes(b"".join(parts))
    return code_path, full_path


def _fmt_local(ts: int | None) -> str | None:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)) if ts else None


# ---------------------------------------------------------------------------
# 1. git sidecar with grading tags (recorder / Codex shim)
# ---------------------------------------------------------------------------
def extract_codex_tags(run_dir: Path, out_dir: Path, slug: str | None,
                       direction: str | None = None) -> dict:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    harness = _harness_of(run_dir)
    direction = _resolve_direction(slug, direction)
    (out_dir / "versions").mkdir(parents=True, exist_ok=True)
    tags = _tags_with_commits(run_dir)
    versions, last_fp = [], None
    n_merged = 0
    for tag_n, name, score, sha, ts in tags:
        fp, code_files, largest_py = _tree_info(run_dir, sha)
        if not code_files:
            continue
        if fp == last_fp and versions:
            prev = versions[-1]
            md = prev["agent_metadata"]
            md["tag_names"].append(name)
            if score is not None:
                md["dedup_scores"].append(score)
                if prev["linked_score"] is None or _better(score, prev["linked_score"], direction):
                    prev["linked_score"] = score
            n_merged += 1
            continue
        last_fp = fp
        k = len(versions) + 1
        code_path, full_path = _write_version_files(run_dir, sha, code_files, largest_py, out_dir, k)
        versions.append({
            "version_number": k,
            "commit": sha[:8],
            "ctime": ts,
            "date": _fmt_local(ts),
            "code_path": code_path,
            "code_all_path": full_path,
            "linked_score": score,
            "score_type": "held-out",
            "agent_metadata": {
                "harness": harness,
                "tree_fp": fp,
                "main_file": largest_py,
                "code_files": code_files,
                "first_tag_n": tag_n,
                "tag_names": [name],
                "dedup_scores": [score] if score is not None else [],
            },
        })
    traj = {
        **_header(run_dir),
        "harness": harness,
        "competition": slug,
        "score_direction": direction,
        "score_type": "held-out",
        "n_tags_total": len(tags),
        "n_dedup_merged": n_merged,
        "version_count": len(versions),
        "versions": versions,
    }
    (out_dir / "trajectory.json").write_text(json.dumps(traj, indent=2))
    return traj


# ---------------------------------------------------------------------------
# 2. AIDE / MLEvolve tree-search journal
# ---------------------------------------------------------------------------
def find_journal(run_dir: Path) -> Path | None:
    for pat in ("filtered_journal.json", "journal.json"):
        hits = sorted(Path(run_dir).rglob(pat))
        if hits:
            return hits[0]
    return None


def extract_aide_journal(run_dir: Path, out_dir: Path, journal: str | Path | None = None,
                         slug: str | None = None, direction: str | None = None) -> dict:
    """Nodes ordered by creation time; scores are the agent's internal CV metric."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    jp = Path(journal) if journal else find_journal(run_dir)
    if jp is None:
        raise FileNotFoundError(f"no journal.json / filtered_journal.json under {run_dir}")
    j = json.loads(jp.read_text())
    nodes = sorted(j.get("nodes", []), key=lambda n: n.get("ctime") or 0)
    if direction is None and config.score_direction(slug) is None:
        # AIDE journals record the metric direction on each node.
        flags = {n["metric"].get("maximize") for n in nodes
                 if isinstance(n.get("metric"), dict) and n["metric"].get("maximize") is not None}
        if len(flags) == 1:
            direction = "higher" if flags.pop() else "lower"
    direction = _resolve_direction(slug, direction)
    (out_dir / "versions").mkdir(parents=True, exist_ok=True)
    versions = []
    for n in nodes:
        if n.get("parent") is None and not n.get("code"):
            continue
        k = len(versions) + 1
        code_path = f"versions/v{k:03d}.py"
        (out_dir / code_path).write_text(n.get("code") or "")
        metric = n.get("metric")
        if isinstance(metric, dict):
            metric = metric.get("value")
        versions.append({
            "version_number": k,
            "node_id": n.get("id"),
            "parent_id": n.get("parent"),
            "stage": n.get("stage"),
            "ctime": n.get("ctime"),
            "code_path": code_path,
            "linked_score": metric,
            "score_type": "cv",
            "is_buggy": n.get("is_buggy"),
        })
    traj = {
        **_header(run_dir),
        "harness": _harness_of(run_dir, default="mlevolve"),
        "competition": slug,
        "score_type": "cv",
        "score_direction": direction,
        "journal": str(jp),
        "version_count": len(versions),
        "versions": versions,
    }
    (out_dir / "trajectory.json").write_text(json.dumps(traj, indent=2))
    return traj


# ---------------------------------------------------------------------------
# 3. git sidecar without tags: re-grade submission.csv commits with mlebench
# ---------------------------------------------------------------------------
def extract_codex_regrade(run_dir: Path, out_dir: Path, slug: str | None,
                          direction: str | None = None, max_grades: int = 50,
                          mlebench: str | None = None, cache: str | Path | None = None) -> dict:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if not slug:
        raise ValueError("the re-grade path needs the competition slug (--slug)")
    cache_dir = config.mlebench_cache(cache)
    if cache_dir is None:
        raise ValueError("the re-grade path needs an MLE-bench data cache "
                         "(--mlebench-cache or TRACEML_MLEBENCH_CACHE)")
    mlebench = config.mlebench_bin(mlebench)
    direction = _resolve_direction(slug, direction)
    harness = _harness_of(run_dir)
    log = git(run_dir, "log", "--all", "--reverse", "--pretty=%H|%cI|%ct",
              "--", "submission.csv", "workspace/local/submission.csv")
    commits = [ln.split("|") for ln in log.splitlines() if ln.count("|") == 2]
    n_total = len(commits)
    if n_total > max_grades:
        step = n_total / max_grades
        commits = [commits[int(i * step)] for i in range(max_grades)]
        print(f"[traceml] WARNING: re-grading {max_grades} of {n_total} submission commits (--max-grades)")
    (out_dir / "versions").mkdir(parents=True, exist_ok=True)
    versions, last_fp = [], None
    for sha, iso, ts in commits:
        fp, code_files, largest_py = _tree_info(run_dir, sha)
        if fp == last_fp:
            continue
        last_fp = fp
        k = len(versions) + 1
        code_path, full_path = _write_version_files(run_dir, sha, code_files, largest_py, out_dir, k)
        score = None
        for sub_path in ("submission.csv", "workspace/local/submission.csv"):
            raw = _git_show(run_dir, sha, sub_path)
            if not raw:
                continue
            tmp = out_dir / "_tmp_submission.csv"
            tmp.write_bytes(raw)
            try:
                r = subprocess.run([mlebench, "grade-sample", str(tmp), slug, "--data-dir", str(cache_dir)],
                                   capture_output=True, text=True, timeout=180)
                m = re.search(r'"score":\s*([0-9.eE+-]+)', r.stdout + r.stderr)
                if m:
                    score = float(m.group(1))
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                pass
            tmp.unlink(missing_ok=True)
            break
        versions.append({
            "version_number": k, "commit": sha[:8], "ctime": int(ts), "date": iso,
            "code_path": code_path, "code_all_path": full_path,
            "linked_score": score, "score_type": "held-out",
            "agent_metadata": {"harness": harness, "tree_fp": fp,
                               "main_file": largest_py, "code_files": code_files},
        })
    traj = {
        **_header(run_dir),
        "harness": harness,
        "competition": slug,
        "score_direction": direction,
        "score_type": "held-out",
        "n_submission_commits": n_total,
        "version_count": len(versions),
        "versions": versions,
    }
    (out_dir / "trajectory.json").write_text(json.dumps(traj, indent=2))
    return traj
