"""Materialize a labeled run as TraceML-schema ``data/{state,action}.parquet``.

The column lists are exactly those of the released ``data/*/state.parquet`` and
``action.parquet``, so a toolkit run can be concatenated with the dataset.
Two deliberate differences from the released experiment_run split:

* ``track``/``model`` name the labeler that produced the rows;
* ``score_effect`` is derived from the measured scores and the competition's
  score direction instead of trusting the labeler's field.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from traceml_toolkit.labeling.inputs import read_jsonl, run_key_id

V3_STATE_COLS = ["key_id", "version_number", "comp", "group", "track", "model", "coarse_tags",
                 "fine_tags", "summary", "keywords", "score", "branch_id", "depth", "stage",
                 "orig_version_number", "is_best_branch", "is_agent", "node_id", "parent_id",
                 "edge_kind", "tree_id", "ctime", "score_public", "score_private",
                 "submission_id", "submission_date", "is_valid_submission", "kernel_id",
                 "version_id", "raw_code_path", "alt_parents_json"]
V3_ACTION_COLS = ["key_id", "comp", "group", "v_old", "v_new", "model", "coarse_actions",
                  "fine_actions", "intents", "magnitude", "score_effect", "goal_nl",
                  "diff_summary", "score_old", "score_new", "orig_v_old", "orig_v_new",
                  "depth_old", "depth_new", "stage_old", "stage_new", "is_best_branch",
                  "is_agent", "edge_kind", "edge_kind_label", "parent_node_id",
                  "child_node_id", "parent_kernel_id", "child_kernel_id", "tree_id",
                  "ctime_old", "ctime_new"]

GROUP_BY_HARNESS = {"codex": "codex", "mlevolve": "mlevolve"}


def _parse(x, default):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return default
    return x if x is not None else default


def derive_score_effect(old, new, direction: str) -> str:
    if old is None or new is None or (isinstance(old, float) and math.isnan(old)) \
            or (isinstance(new, float) and math.isnan(new)):
        return "unknown"
    delta = new - old
    if abs(delta) < 1e-9:
        return "plateau"
    improving = delta < 0 if direction == "lower" else delta > 0
    return "improving" if improving else "regressing"


def build(out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    traj = json.loads((out_dir / "extracted/trajectory.json").read_text())
    labels = out_dir / "labels"
    if not (labels / "state_output.jsonl").exists():
        raise FileNotFoundError(f"{labels / 'state_output.jsonl'} is missing; run `traceml label` first")
    harness = traj.get("harness", "codex")
    group = GROUP_BY_HARNESS.get(harness, "external")
    comp = traj.get("competition")
    direction = traj.get("score_direction") or "higher"
    kid = run_key_id(traj)

    state_out = {int(r["version_number"]): r for r in read_jsonl(labels / "state_output.jsonl")
                 if r.get("version_number") is not None}
    v_by_num = {int(v["version_number"]): v for v in traj["versions"]}
    order = sorted(v_by_num)

    srows, prev_node = [], None
    for depth, vno in enumerate(order):
        v, s = v_by_num[vno], state_out.get(vno, {})
        score = v.get("linked_score")
        node_id = f"a:{kid}:{vno}"
        srows.append({
            "key_id": kid, "version_number": vno, "comp": comp, "group": group,
            "track": "qwen3_1.7b_distill", "model": "qwen3-1.7b-state/final",
            "coarse_tags": json.dumps(_parse(s.get("coarse_tags"), [])),
            "fine_tags": json.dumps(_parse(s.get("fine_tags"), [])),
            "summary": s.get("summary", ""),
            "keywords": json.dumps(_parse(s.get("keywords"), [])),
            "score": score, "branch_id": 0, "depth": depth, "stage": v.get("stage"),
            "orig_version_number": float("nan"), "is_best_branch": True, "is_agent": True,
            "node_id": node_id, "parent_id": prev_node, "edge_kind": "version",
            "tree_id": f"agent:{harness}:{kid}", "ctime": v.get("ctime"),
            "score_public": score, "score_private": None, "submission_id": None,
            "submission_date": v.get("date"),
            "is_valid_submission": v.get("valid_submission", True),
            "kernel_id": kid, "version_id": vno,
            "raw_code_path": f"extracted/{v['code_path']}",
            "alt_parents_json": "[]",
        })
        prev_node = node_id

    ain_path = labels / "action_input.jsonl"
    ain = {(int(j["v_old"]), int(j["v_new"])): j for j in read_jsonl(ain_path)} if ain_path.exists() else {}
    aout_path = labels / "action_output.jsonl"
    arows = []
    for j in (read_jsonl(aout_path) if aout_path.exists() else []):
        if j.get("v_old") is None or j.get("v_new") is None:
            continue
        vo, vn = int(j["v_old"]), int(j["v_new"])
        meta = ain.get((vo, vn), {})
        so = meta.get("score_old", v_by_num.get(vo, {}).get("linked_score"))
        sn = meta.get("score_new", v_by_num.get(vn, {}).get("linked_score"))
        arows.append({
            "key_id": kid, "comp": comp, "group": group, "v_old": vo, "v_new": vn,
            "model": "qwen3-1.7b-action/final",
            "coarse_actions": json.dumps(_parse(j.get("coarse_actions"), [])),
            "fine_actions": json.dumps(_parse(j.get("fine_actions"), [])),
            "intents": json.dumps(_parse(j.get("intents"), [])),
            "magnitude": j.get("magnitude") or "unknown",
            "score_effect": derive_score_effect(so, sn, direction),
            "goal_nl": j.get("goal_nl", ""), "diff_summary": j.get("diff_summary", ""),
            "score_old": so, "score_new": sn,
            "orig_v_old": float("nan"), "orig_v_new": float("nan"),
            "depth_old": float("nan"), "depth_new": float("nan"),
            "stage_old": None, "stage_new": None,
            "is_best_branch": True, "is_agent": True,
            "edge_kind": "version", "edge_kind_label": "version",
            "parent_node_id": f"a:{kid}:{vo}", "child_node_id": f"a:{kid}:{vn}",
            "parent_kernel_id": kid, "child_kernel_id": kid,
            "tree_id": f"agent:{harness}:{kid}",
            "ctime_old": v_by_num.get(vo, {}).get("ctime"),
            "ctime_new": v_by_num.get(vn, {}).get("ctime"),
        })
    return (pd.DataFrame(srows, columns=V3_STATE_COLS),
            pd.DataFrame(arows, columns=V3_ACTION_COLS))


def write(out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    st, ac = build(out_dir)
    d = out_dir / "data"
    d.mkdir(parents=True, exist_ok=True)
    st.to_parquet(d / "state.parquet", index=False)
    ac.to_parquet(d / "action.parquet", index=False)
    return {"state_rows": len(st), "action_rows": len(ac),
            "score_effect": ac["score_effect"].value_counts().to_dict() if len(ac) else {}}
