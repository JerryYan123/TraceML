"""Materialize a trajectory of the released dataset in the toolkit's layout.

Any human or agent trajectory of the TraceML dataset can then be reported on
with ``traceml report`` exactly like a freshly analyzed run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traceml_toolkit import assets, config
from traceml_toolkit.labeling.inputs import write_jsonl


def _parse(x, default):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return default
    return x if x is not None else default


def _num(x):
    return float(x) if x is not None and pd.notna(x) else None


def materialize(key_id: str, split: str = "paired", out_dir: str | Path | None = None,
                dataset_dir: str | Path | None = None, download: bool = True) -> Path:
    st = pd.read_parquet(assets.resolve_file(assets.split_file(split, "state"), dataset_dir, download))
    ac = pd.read_parquet(assets.resolve_file(assets.split_file(split, "action"), dataset_dir, download))
    st = st[st["key_id"].astype(str) == key_id].sort_values("version_number")
    ac = ac[ac["key_id"].astype(str) == key_id].sort_values(["v_old", "v_new"])
    if st.empty:
        raise KeyError(f"key_id {key_id!r} not found in the {split} split")

    comp = str(st["comp"].iloc[0])
    direction = config.score_direction(comp) or "higher"
    is_agent = bool(st["is_agent"].iloc[0])
    group = st["group"].iloc[0]
    harness = (group if group in ("codex", "mlevolve") else "agent") if is_agent else "human"
    out = Path(out_dir) if out_dir else Path("traceml_out") / "released" / key_id
    (out / "extracted").mkdir(parents=True, exist_ok=True)

    versions = []
    for _, r in st.iterrows():
        versions.append({
            "version_number": int(r["version_number"]),
            "ctime": _num(r.get("ctime")),
            "date": None if pd.isna(r.get("submission_date")) else str(r.get("submission_date")),
            "code_path": None,
            "linked_score": _num(r.get("score")),
            "score_type": "held-out",
        })
    traj = {
        "run_dir": f"released:{split}:{key_id}",
        "run_name": key_id,
        "harness": harness,
        "group": None if pd.isna(group) else str(group),
        "competition": comp,
        "score_direction": direction,
        "score_type": "held-out",
        "version_count": len(versions),
        "versions": versions,
    }
    (out / "extracted" / "trajectory.json").write_text(json.dumps(traj, indent=2))
    write_jsonl(out / "labels" / "state_output.jsonl", [{
        "key_id": key_id, "comp": comp, "group": traj["group"],
        "version_number": int(r["version_number"]),
        "coarse_tags": _parse(r["coarse_tags"], []), "fine_tags": _parse(r["fine_tags"], []),
        "summary": r.get("summary") if pd.notna(r.get("summary")) else "",
        "keywords": _parse(r.get("keywords"), []),
    } for _, r in st.iterrows()])
    write_jsonl(out / "labels" / "action_output.jsonl", [{
        "key_id": key_id, "v_old": int(r["v_old"]), "v_new": int(r["v_new"]),
        "coarse_actions": _parse(r["coarse_actions"], []),
        "fine_actions": _parse(r["fine_actions"], []),
        "intents": _parse(r["intents"], []),
        "magnitude": r.get("magnitude"),
    } for _, r in ac.iterrows()])
    return out
