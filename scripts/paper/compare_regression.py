#!/usr/bin/env python3
"""Compare toolkit labels on a run with the labels released in the TraceML dataset.

This is the regression check behind the toolkit's validation claim: the paper's
skill-rep1 commonlit run was processed by both the dataset pipeline and the
toolkit. Extraction must reproduce the released version count; state coarse
tags should match; action and intent labels are reported as measured (they
are not expected to match exactly, see the README).

    python scripts/paper/compare_regression.py \
        --out-dir traceml_out/run_20260506_041940_gpt_5_4_mini_run29_12h_skillv3_rep1_commonlitreadabilityprize \
        --key-id codex_run_20260506_041940_gpt_5_4_mini_run29_12h_skill_rep1_commonlitreadabilityprize \
        [--split experiment_run] [--dataset-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from traceml_toolkit import assets
from traceml_toolkit.labeling.inputs import read_jsonl


def _parse(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return x


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path, help="toolkit output dir of the run")
    ap.add_argument("--key-id", required=True, help="key_id of the same run in the released split")
    ap.add_argument("--split", default="experiment_run")
    ap.add_argument("--dataset-dir")
    a = ap.parse_args()

    st = pd.read_parquet(assets.resolve_file(assets.split_file(a.split, "state"), a.dataset_dir))
    ac = pd.read_parquet(assets.resolve_file(assets.split_file(a.split, "action"), a.dataset_dir))
    st = st[st["key_id"] == a.key_id].sort_values("version_number")
    ac = ac[ac["key_id"] == a.key_id].sort_values("v_new")
    if st.empty:
        raise SystemExit(f"{a.key_id} not in the {a.split} split")

    labels = a.out_dir / "labels"
    tk_state = sorted((r for r in read_jsonl(labels / "state_output.jsonl") if r.get("coarse_tags")),
                      key=lambda r: int(r["version_number"]))
    tk_act = sorted((r for r in read_jsonl(labels / "action_output.jsonl") if r.get("coarse_actions")),
                    key=lambda r: int(r["v_new"]))
    traj = json.loads((a.out_dir / "extracted" / "trajectory.json").read_text())

    # Released versions keep raw extraction indices while the toolkit renumbers
    # 1..N, so both sides are aligned by rank.
    rel_coarse = [set(_parse(x) or []) for x in st["coarse_tags"]]
    rel_fine = [{d["tag"] for d in (_parse(x) or []) if isinstance(d, dict)} for x in st["fine_tags"]]
    coarse_j = [_jaccard(r, set(t["coarse_tags"])) for r, t in zip(rel_coarse, tk_state)]
    fine_j = [_jaccard(r, {d["tag"] for d in t.get("fine_tags", [])}) for r, t in zip(rel_fine, tk_state)]
    rel_act = [(set(_parse(c) or []), (_parse(i) or [{}])[0].get("intent"))
               for c, i in zip(ac["coarse_actions"], ac["intents"])]
    act_j = [_jaccard(r, set(t["coarse_actions"])) for (r, _), t in zip(rel_act, tk_act)]
    intent_m = [ri == (t.get("intents") or [{}])[0].get("intent") for (_, ri), t in zip(rel_act, tk_act)]

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    res = {
        "key_id": a.key_id,
        "grader_calls": traj.get("n_tags_total"),
        "released_versions": len(st),
        "toolkit_versions": traj["version_count"],
        "version_count_match": len(st) == traj["version_count"],
        "state_rows_compared": len(coarse_j),
        "state_coarse_mean_jaccard": mean(coarse_j),
        "state_coarse_exact_share": mean([float(j == 1.0) for j in coarse_j]),
        "state_fine_mean_jaccard": mean(fine_j),
        "action_rows_compared": len(act_j),
        "action_coarse_mean_jaccard": mean(act_j),
        "intent_primary_match_share": mean([float(m) for m in intent_m]),
    }
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
