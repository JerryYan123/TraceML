"""One command: run directory -> trajectory -> labels -> parquet tables -> report."""
from __future__ import annotations

import json
import time
from pathlib import Path

from traceml_toolkit import detect as detect_mod
from traceml_toolkit import extract as ex
from traceml_toolkit import report as report_mod
from traceml_toolkit import schema as schema_mod


def default_out_dir(run_dir: Path) -> Path:
    return Path("traceml_out") / Path(run_dir).resolve().name


def check_out_dir(run_dir: Path, out_dir: Path) -> None:
    """The run directory is read-only: refuse to write inside it (or inside the package)."""
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir).resolve()
    if out_dir == run_dir or run_dir in out_dir.parents:
        raise ValueError(f"--out {out_dir} is inside the run directory {run_dir}; "
                         "choose a location outside it (the toolkit never writes into a run)")
    pkg = Path(__file__).resolve().parent
    if out_dir == pkg or pkg in out_dir.parents:
        raise ValueError(f"--out {out_dir} is inside the installed package; choose another location")


def extract_run(run_dir: str | Path, out_dir: str | Path, *, slug: str | None = None,
                direction: str | None = None, max_grades: int = 50, mlebench: str | None = None,
                mlebench_cache: str | Path | None = None) -> dict:
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir)
    check_out_dir(run_dir, out_dir)
    info = detect_mod.detect(run_dir)
    slug = slug or info.get("slug")
    edir = out_dir / "extracted"
    if edir.exists():  # a previous extraction may have more versions; start clean
        for p in sorted(edir.glob("versions/*")):
            p.unlink()
    if info["layout"] == "codex-tags":
        traj = ex.extract_codex_tags(run_dir, edir, slug, direction)
    elif info["layout"] == "aide":
        traj = ex.extract_aide_journal(run_dir, edir, info.get("journal"), slug, direction)
    else:
        traj = ex.extract_codex_regrade(run_dir, edir, slug, direction, max_grades, mlebench, mlebench_cache)
    return {"layout": info["layout"], "slug": slug, "version_count": traj["version_count"],
            "n_tags_total": traj.get("n_tags_total"), "n_dedup_merged": traj.get("n_dedup_merged")}


def analyze(run_dir: str | Path, out_dir: str | Path | None = None, *, slug: str | None = None,
            direction: str | None = None, gpu: int | None = None, backend: str = "auto",
            skip_label: bool = False, max_grades: int = 50, diff_scope: str = "full",
            dataset_dir: str | Path | None = None, models_dir: str | Path | None = None,
            mlebench: str | None = None, mlebench_cache: str | Path | None = None,
            gpu_mem_util: float = 0.5) -> dict:
    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir) if out_dir else default_out_dir(run_dir)
    check_out_dir(run_dir, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"run_dir": str(run_dir), "out_dir": str(out_dir), "timings_s": {}}
    t = summary["timings_s"]

    t0 = time.time()
    summary["extract"] = extract_run(run_dir, out_dir, slug=slug, direction=direction,
                                     max_grades=max_grades, mlebench=mlebench, mlebench_cache=mlebench_cache)
    t["extract"] = round(time.time() - t0, 2)
    print(f"[traceml] extracted {summary['extract']['version_count']} versions "
          f"({summary['extract']['layout']}, {summary['extract']['slug']})", flush=True)

    labels_exist = (out_dir / "labels" / "action_output.jsonl").exists()
    if not skip_label:
        from traceml_toolkit.labeling.runner import label_run
        t0 = time.time()
        summary["label"] = label_run(out_dir, backend=backend, gpu=gpu, diff_scope=diff_scope,
                                     models_dir=models_dir, dataset_dir=dataset_dir,
                                     gpu_mem_util=gpu_mem_util)
        t["label"] = round(time.time() - t0, 2)
        labels_exist = True
    if labels_exist:
        t0 = time.time()
        summary["schema"] = schema_mod.write(out_dir)
        t["schema"] = round(time.time() - t0, 2)
    t0 = time.time()
    rep = report_mod.build_report(out_dir, dataset_dir=dataset_dir)
    t["report"] = round(time.time() - t0, 2)
    summary["report"] = {k: rep[k] for k in ("best_score", "human_percentile", "nearest_cohorts",
                                             "n_labeled_transitions")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
