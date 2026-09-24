"""Label stage: ``extracted/trajectory.json`` -> ``labels/{state,action}_{input,output}.jsonl``.

State labels come first because every action prompt embeds the two state
labels it connects. Rows the labeler fails to produce as valid JSON are retried
once in small fresh batches with a larger output cap (greedy decoding can
degenerate on batches of near-identical prompts). Rows that still fail are kept
as PARSE_ERROR rows and counted in ``labels/label_log.json``; nothing is
dropped silently.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from traceml_toolkit import assets, config
from traceml_toolkit.labeling import inputs, parse, render
from traceml_toolkit.labeling.backends import get_backend, resolve_backend_name

DEFAULTS = {
    # The state labeler reads whole files; 16384 (the old default) skipped
    # versions above ~650 lines. The checkpoints support 40960 positions.
    "state": {"max_model_len": 32768, "max_new_tokens": 2000},
    # 8192 (the paper-time setting) silently dropped long diffs.
    "action": {"max_model_len": 16384, "max_new_tokens": 2000},
}
RETRY_MAX_NEW_TOKENS = 6000
RETRY_CHUNK = 3


def model_name(model_dir: Path) -> str:
    return f"{Path(model_dir).parent.name}/{Path(model_dir).name}"


def label_task(task: str, records: list[dict], *, backend, tokenizer, name: str,
               max_model_len: int, max_new_tokens: int) -> tuple[list[dict], dict]:
    """Label `records` with one backend; returns (rows aligned with records, stats)."""
    t0 = time.time()
    budget = max_model_len - max_new_tokens - render.SAFETY_MARGIN
    fitted = [render.fit_prompt(task, r, tokenizer, budget) for r in records]
    todo = [i for i, f in enumerate(fitted) if f.text is not None]
    texts = backend.generate([fitted[i].text for i in todo],
                             max_new_tokens=max_new_tokens, max_model_len=max_model_len)
    generated = dict(zip(todo, texts))
    rows = [parse.make_row(task, rec, generated.get(i, ""), name) for i, rec in enumerate(records)]

    failed = [i for i, r in enumerate(rows) if not parse.is_parsed(task, r)]
    n_first_pass_failed = len(failed)
    recovered = 0
    oversize = set()
    if failed:
        print(f"[traceml] {task}: retrying {len(failed)} unparsed rows in batches of {RETRY_CHUNK}", flush=True)
        retry_budget = config.LABELER_MAX_POSITIONS - RETRY_MAX_NEW_TOKENS - render.SAFETY_MARGIN
        for c in range(0, len(failed), RETRY_CHUNK):
            chunk = failed[c:c + RETRY_CHUNK]
            refits = {i: render.fit_prompt(task, records[i], tokenizer, retry_budget) for i in chunk}
            ok = [i for i in chunk if refits[i].text is not None]
            oversize.update(i for i in chunk if refits[i].text is None)
            if not ok:
                continue
            mml = min(config.LABELER_MAX_POSITIONS,
                      max(refits[i].n_tokens for i in ok) + RETRY_MAX_NEW_TOKENS + 64)
            retry_texts = backend.generate([refits[i].text for i in ok],
                                           max_new_tokens=RETRY_MAX_NEW_TOKENS, max_model_len=mml)
            for i, text in zip(ok, retry_texts):
                row = parse.make_row(task, records[i], text, name)
                if parse.is_parsed(task, row):
                    rows[i] = row
                    recovered += 1
    for i in oversize:
        rows[i]["skipped"] = "oversize"
    n_failed = sum(1 for r in rows if not parse.is_parsed(task, r))
    if n_failed:
        print(f"[traceml] WARNING: {n_failed} {task} rows still unparsed after retry "
              f"({len(oversize)} too long for the labeler)", flush=True)
    stats = {
        "n_in": len(records),
        "n_labeled": len(records) - n_failed,
        "n_refit_truncated": sum(1 for f in fitted if f.refit),
        "n_first_pass_failed": n_first_pass_failed,
        "n_retry_recovered": recovered,
        "n_failed": n_failed,
        "n_oversize": len(oversize),
        "max_model_len": max_model_len,
        "max_new_tokens": max_new_tokens,
        "seconds": round(time.time() - t0, 1),
    }
    return rows, stats


def label_run(out_dir: str | Path, *, backend: str = "auto", gpu: int | None = None,
              diff_scope: str = "full", tasks: tuple[str, ...] = ("state", "action"),
              models_dir: str | Path | None = None, dataset_dir: str | Path | None = None,
              gpu_mem_util: float = 0.5, limit: int | None = None, download: bool = True) -> dict:
    """Label an extracted run in place (writes ``<out_dir>/labels/``)."""
    out_dir = Path(out_dir)
    extract_dir = out_dir / "extracted"
    traj = json.loads((extract_dir / "trajectory.json").read_text())
    ldir = out_dir / "labels"
    ldir.mkdir(parents=True, exist_ok=True)
    backend_name = resolve_backend_name(backend)
    summary: dict = {"backend": backend_name, "diff_scope": diff_scope, "tasks": {}}

    for task in ("state", "action"):
        if task not in tasks:
            continue
        if task == "state":
            records = inputs.build_state_records(extract_dir, traj)
        else:
            state_out = ldir / "state_output.jsonl"
            if not state_out.exists():
                raise FileNotFoundError(f"{state_out} is missing; label states first")
            records = inputs.build_action_records(extract_dir, traj, inputs.read_jsonl(state_out),
                                                  diff_scope=diff_scope)
        if limit is not None:
            records = records[:limit]
        inputs.write_jsonl(ldir / f"{task}_input.jsonl", records)
        if not records:
            inputs.write_jsonl(ldir / f"{task}_output.jsonl", [])
            summary["tasks"][task] = {"n_in": 0, "n_labeled": 0}
            continue
        model_dir = assets.resolve_labeler(task, models_dir, dataset_dir, download_missing=download)
        print(f"[traceml] labeling {len(records)} {task} rows with {backend_name} ({model_name(model_dir)})",
              flush=True)
        tokenizer = render.load_tokenizer(model_dir)
        be = get_backend(backend_name, model_dir, gpu=gpu, gpu_mem_util=gpu_mem_util, log_dir=ldir)
        try:
            rows, stats = label_task(task, records, backend=be, tokenizer=tokenizer,
                                     name=model_name(model_dir), **DEFAULTS[task])
        finally:
            be.close()
        inputs.write_jsonl(ldir / f"{task}_output.jsonl", rows)
        stats["labeler"] = model_name(model_dir)
        stats["labeler_revision"] = assets.local_revision(f"{config.LABELER_DIRS[task]}/model.safetensors")
        summary["tasks"][task] = stats

    (ldir / "label_log.json").write_text(json.dumps(summary, indent=2))
    return summary
