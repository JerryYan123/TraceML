"""Command-line interface: ``traceml <command> ...``."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from traceml_toolkit import __version__

EXIT_USAGE, EXIT_MISSING, EXIT_UNLABELED = 2, 3, 4


def _add_common(p: argparse.ArgumentParser, labeling: bool = False) -> None:
    p.add_argument("--dataset-dir", help="an existing clone of the TraceML dataset (default: download "
                                         "what is needed into $TRACEML_HOME)")
    if labeling:
        p.add_argument("--models-dir", help="directory holding qwen3-1.7b-{state,action}/final")
        p.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"],
                       help="labeling backend (auto: vLLM when available, else transformers)")
        p.add_argument("--gpu", type=int, default=None, help="GPU index for labeling")
        p.add_argument("--gpu-mem-util", type=float, default=0.5,
                       help="vLLM gpu_memory_utilization (default 0.5)")
        p.add_argument("--diff-scope", default="full", choices=["full", "main"],
                       help="action diffs over the whole codebase (full) or the main file only")


def _add_extract_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--slug", help="competition slug (default: from run_meta.json or the directory name)")
    p.add_argument("--direction", choices=["higher", "lower"], help="score direction if the slug is unknown")
    p.add_argument("--max-grades", type=int, default=50, help="re-grade path: max submissions to grade")
    p.add_argument("--mlebench", help="mlebench executable (re-grade path)")
    p.add_argument("--mlebench-cache", help="prepared MLE-bench data directory (re-grade path)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="traceml",
        description="Turn an ML-engineering agent run into a TraceML trajectory and a behavior "
                    "report against human Kaggle cohorts (arXiv:2608.26086).")
    ap.add_argument("--version", action="version", version=f"traceml-toolkit {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<command>")

    p = sub.add_parser("analyze", help="extract, label and report on a run in one command")
    p.add_argument("run_dir")
    p.add_argument("--out", help="output directory (default: ./traceml_out/<run name>)")
    p.add_argument("--skip-label", action="store_true",
                   help="no labeling (no GPU needed); report on existing labels if any")
    _add_extract_opts(p)
    _add_common(p, labeling=True)

    p = sub.add_parser("extract", help="run directory -> <out>/extracted/trajectory.json")
    p.add_argument("run_dir")
    p.add_argument("--out", help="output directory (default: ./traceml_out/<run name>)")
    _add_extract_opts(p)

    p = sub.add_parser("label", help="label an extracted run with the released labelers")
    p.add_argument("out_dir")
    p.add_argument("--task", default="both", choices=["state", "action", "both"])
    p.add_argument("--limit", type=int, help="label only the first N rows (smoke test)")
    _add_common(p, labeling=True)

    p = sub.add_parser("schema", help="write TraceML-schema data/{state,action}.parquet")
    p.add_argument("out_dir")

    p = sub.add_parser("report", help="write report.md, report.json and figs/")
    p.add_argument("out_dir")
    p.add_argument("--comp", help="competition slug (default: from the trajectory)")
    p.add_argument("--run-dir", help="original run directory (for native_session.log)")
    _add_common(p)

    p = sub.add_parser("record", help="run a CLI agent under the recorder",
                       description="traceml record <run_dir> --slug S --minutes M --harness H -- "
                                   "<agent command containing {PROMPT}>")
    p.add_argument("run_dir")
    p.add_argument("--slug", required=True)
    p.add_argument("--minutes", type=int, required=True)
    p.add_argument("--harness", required=True, help="a name for the agent, e.g. claude-code")
    p.add_argument("--mlebench")
    p.add_argument("--mlebench-cache")
    p.add_argument("--poll", type=int, help="seconds between recorder polls (default 60)")

    p = sub.add_parser("download-models", help="download reference data and labelers from HuggingFace")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--data-only", action="store_true", help="reference tables only (~11 MB)")
    g.add_argument("--models-only", action="store_true", help="labelers only (~6.9 GB)")

    p = sub.add_parser("from-released", help="materialize a released trajectory for `traceml report`")
    p.add_argument("key_id")
    p.add_argument("--split", default="paired", choices=["paired", "humans_only", "experiment_run"])
    p.add_argument("--out", help="output directory (default: ./traceml_out/released/<key_id>)")
    _add_common(p)

    sub.add_parser("info", help="show resolved paths, assets and available backends")
    return ap


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_info() -> int:
    import importlib.util
    import platform

    from traceml_toolkit import assets, config
    from traceml_toolkit.labeling.backends import transformers_available, vllm_available
    info = {
        "traceml-toolkit": __version__,
        "python": platform.python_version(),
        "TRACEML_HOME": str(config.traceml_home()),
        "hf_dataset": config.hf_repo(),
        "reference_data": {rel: str(assets.find_file(rel) or "missing") for rel in assets.REFERENCE_FILES},
        "labelers": {t: str(assets.find_labeler(t) or "missing") for t in ("state", "action")},
        "backends": {"vllm": vllm_available(), "transformers": transformers_available()},
        "vllm_importable": importlib.util.find_spec("vllm") is not None,
        "git": shutil.which("git") is not None,
        "mlebench": shutil.which(config.mlebench_bin()) is not None,
    }
    _print_json(info)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    agent_cmd: list[str] = []
    if argv[:1] == ["record"] and "--" in argv:
        cut = argv.index("--")
        argv, agent_cmd = argv[:cut], argv[cut + 1:]
    args = build_parser().parse_args(argv)

    from traceml_toolkit.assets import MissingAsset
    from traceml_toolkit.detect import UnsupportedRunLayout
    try:
        return _dispatch(args, agent_cmd)
    except UnsupportedRunLayout as e:
        print(f"traceml: {e}", file=sys.stderr)
        return EXIT_USAGE
    except MissingAsset as e:
        print(f"traceml: {e}", file=sys.stderr)
        return EXIT_MISSING
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
        print(f"traceml: {e}", file=sys.stderr)
        return EXIT_USAGE


def _dispatch(args, agent_cmd: list[str]) -> int:
    if args.cmd == "info":
        return cmd_info()

    if args.cmd == "analyze":
        from traceml_toolkit.pipeline import analyze
        s = analyze(args.run_dir, args.out, slug=args.slug, direction=args.direction, gpu=args.gpu,
                    backend=args.backend, skip_label=args.skip_label, max_grades=args.max_grades,
                    diff_scope=args.diff_scope, dataset_dir=args.dataset_dir, models_dir=args.models_dir,
                    mlebench=args.mlebench, mlebench_cache=args.mlebench_cache,
                    gpu_mem_util=args.gpu_mem_util)
        _print_json(s)
        print(f"[traceml] report: {Path(s['out_dir']) / 'report.md'}")
        if args.skip_label and not (Path(s["out_dir"]) / "labels").exists():
            print(f"[traceml] next: traceml label {s['out_dir']}  (needs a GPU), then traceml report {s['out_dir']}")
        return 0

    if args.cmd == "extract":
        from traceml_toolkit.pipeline import default_out_dir, extract_run
        out = Path(args.out) if args.out else default_out_dir(Path(args.run_dir))
        s = extract_run(args.run_dir, out, slug=args.slug, direction=args.direction,
                        max_grades=args.max_grades, mlebench=args.mlebench, mlebench_cache=args.mlebench_cache)
        _print_json({**s, "out_dir": str(out)})
        return 0

    if args.cmd == "label":
        from traceml_toolkit.labeling.runner import label_run
        tasks = ("state", "action") if args.task == "both" else (args.task,)
        s = label_run(args.out_dir, backend=args.backend, gpu=args.gpu, diff_scope=args.diff_scope,
                      tasks=tasks, models_dir=args.models_dir, dataset_dir=args.dataset_dir,
                      gpu_mem_util=args.gpu_mem_util, limit=args.limit)
        _print_json(s)
        labeled = [t.get("n_labeled", 0) for t in s["tasks"].values() if t.get("n_in")]
        return EXIT_UNLABELED if labeled and not any(labeled) else 0

    if args.cmd == "schema":
        from traceml_toolkit.schema import write
        _print_json(write(args.out_dir))
        return 0

    if args.cmd == "report":
        from traceml_toolkit.report import build_report
        rep = build_report(args.out_dir, comp=args.comp, dataset_dir=args.dataset_dir, run_dir=args.run_dir)
        print(f"[traceml] wrote {Path(args.out_dir) / 'report.md'}")
        _print_json({k: rep[k] for k in ("best_score", "human_percentile", "nearest_cohorts",
                                         "n_labeled_transitions")})
        return 0

    if args.cmd == "record":
        from traceml_toolkit.record import record
        if not agent_cmd:
            print("traceml: give the agent command after `--`", file=sys.stderr)
            return EXIT_USAGE
        record(args.run_dir, args.slug, args.minutes, args.harness, agent_cmd,
               mlebench=args.mlebench, mlebench_cache=args.mlebench_cache, poll=args.poll)
        return 0  # not reached: record() execs the recorder

    if args.cmd == "download-models":
        from traceml_toolkit import assets
        out = {}
        if not args.models_only:
            out["data"] = [str(assets.resolve_file(rel)) for rel in assets.DATA_FILES]
        if not args.data_only:
            out["labelers"] = {t: str(assets.resolve_labeler(t)) for t in ("state", "action")}
        _print_json(out)
        return 0

    if args.cmd == "from-released":
        from traceml_toolkit.from_released import materialize
        out = materialize(args.key_id, args.split, args.out, args.dataset_dir)
        print(f"[traceml] materialized {args.key_id} -> {out}")
        print(f"[traceml] next: traceml report {out}")
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
