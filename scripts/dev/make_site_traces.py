#!/usr/bin/env python3
"""Export the real example trajectories shown at the top of the project page.

Writes docs/data/traces.json: four trajectories from the paired split on one
competition (a Codex run, an MLEvolve branch and two top-10% humans), each edit
classed by the paper's fine-action clusters (Figure 3d), the best-so-far score
of every version as a percentile of the competition's human trajectories, and
the share of edits in each cluster for every cohort.

    python scripts/dev/make_site_traces.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from traceml_toolkit import cohorts, config
from traceml_toolkit.assets import resolve_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "data" / "traces.json"
COMP = "learning-agency-lab-automated-essay-scoring-2"
EXAMPLES = [
    ("codex", "Codex", "run_20260422_215440_gpt_5_4_mini_run29_12h_learning-agency-lab-automated-essay-scoring-2"),
    ("mlevolve", "MLEvolve", "run29_12h_learning-agency-lab-automated-essay-scoring-2__"
                             "learning-agency-lab-automated-essay-scoring-2__branch27"),
    ("human", "Top-10% human A", "55621328"),
    ("human", "Top-10% human B", "59253662"),
]
# The paper's fine-action clusters (Figure 3d, Table 17), extended with the closest tags.
SUBMISSION = {"change_weights", "postprocess_change", "threshold_tune", "submission_format", "add_member",
              "stacking", "remove_member", "tta_change"}
MODEL = {"layer_modification", "epoch_change", "seed_averaging", "lr_change", "hparam_single_knob",
         "hparam_multi_knob", "batch_size_change", "optimizer_swap", "scheduler_change", "regularization_change",
         "early_stopping_change", "custom_module_change", "add_head", "gradient_config"}
DIRECTION = {"swap_backbone", "pretrained_swap", "checkpoint_swap", "noop", "feature_selection", "add_feature",
             "cv_scheme", "split_strategy", "loss_change", "target_transform", "text_vectorization",
             "metric_change", "fold_count"}
PHRASE = {
    "change_weights": "re-weight the ensemble", "postprocess_change": "tweak post-processing",
    "threshold_tune": "tune a threshold", "submission_format": "fix the submission format",
    "add_member": "add an ensemble member", "stacking": "stack models", "remove_member": "drop a member",
    "tta_change": "change test-time augmentation", "layer_modification": "edit layers",
    "epoch_change": "change epochs", "seed_averaging": "average seeds", "lr_change": "change the learning rate",
    "hparam_single_knob": "turn one knob", "hparam_multi_knob": "turn several knobs",
    "swap_backbone": "swap the backbone", "pretrained_swap": "swap the pretrained model",
    "checkpoint_swap": "swap a checkpoint", "noop": "re-run to check", "feature_selection": "select features",
    "add_feature": "add a feature", "cv_scheme": "change the CV scheme", "loss_change": "change the loss",
    "error_fix": "fix an error", "data_loading": "change data loading", "code_restructure": "restructure code",
}


def parse(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    return x if isinstance(x, list) else []


def edit_class(fine: list[str]) -> str:
    c = {"direction": sum(a in DIRECTION for a in fine), "submission": sum(a in SUBMISSION for a in fine),
         "model": sum(a in MODEL for a in fine)}
    if not any(c.values()):
        return "other"
    order = {"direction": 2, "submission": 1, "model": 0}
    return max(c, key=lambda k: (c[k], order[k]))


def main() -> None:
    act = pd.read_parquet(resolve_file("data/paired/action.parquet"))
    st = pd.read_parquet(resolve_file("data/paired/state.parquet"))
    _, idx = cohorts.load_reference()
    hs = cohorts.human_scores(idx, COMP).to_numpy()
    lower = config.score_direction(COMP) == "lower"

    act["fine"] = act["fine_actions"].map(lambda x: [d.get("action") for d in parse(x) if isinstance(d, dict)])
    act = act[act["fine"].map(len) > 0].copy()
    act["cls"] = act["fine"].map(edit_class)
    kid2g = dict(zip(idx["key_id"].astype(str), idx["group7"]))
    g = act["key_id"].astype(str).map(kid2g)
    act["cohort"] = np.where(act["group"] == "codex", "Codex",
                    np.where(act["group"] == "mlevolve", "MLEvolve",
                    np.where(g == "top10", "Top-10% humans", np.where(g.notna(), "Other humans", None))))
    shares = {}
    for coh, sub in act.dropna(subset=["cohort"]).groupby("cohort"):
        vc = sub["cls"].value_counts(normalize=True)
        shares[coh] = {k: round(float(vc.get(k, 0.0)), 3) for k in ("submission", "model", "direction", "other")}
        shares[coh]["n_edits"] = int(len(sub))

    traces = []
    for kind, label, key in EXAMPLES:
        a = act[(act["comp"] == COMP) & act["key_id"].astype(str).str.endswith(key)].sort_values(["v_old", "v_new"])
        kid = a["key_id"].iloc[0]
        a = a[a["key_id"] == kid]
        s = st[st["key_id"] == kid].sort_values("version_number")
        best, pct = None, []
        for x in s["score"].astype(float):
            if not np.isnan(x):
                best = x if best is None else (min(best, x) if lower else max(best, x))
            pct.append(None if best is None else round(float(((hs > best) if lower else (hs < best)).mean() * 100), 1))
        edits = []
        members = {"submission": SUBMISSION, "model": MODEL, "direction": DIRECTION}
        for fine, intents, cls in zip(a["fine"], a["intents"].map(parse), a["cls"]):
            uniq = list(dict.fromkeys(fine))
            # the edits that decided the class come first
            uniq.sort(key=lambda f: f not in members.get(cls, set()))
            names = [PHRASE.get(f, f.replace("_", " ")) for f in uniq][:3]
            intent = intents[0].get("intent") if intents and isinstance(intents[0], dict) else None
            edits.append({"c": cls, "what": names, "why": intent})
        traces.append({"kind": kind, "label": label, "n_edits": len(edits), "edits": edits, "pct": pct})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "competition": COMP, "competition_label": "Automated Essay Scoring 2.0 (Kaggle)",
        "n_humans": int(len(hs)), "traces": traces, "shares": shares,
        "paper": {"pivot": {"Humans": 25, "Codex": 9, "MLEvolve": 58},
                  "returns": {"Top humans": [9.1, "9% of eligible versions"],
                              "Codex": [round(100 / 658, 2), "1 of 658"], "MLEvolve": [0.0, "0 of 344"]}},
    }, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
    for t in traces:
        print(t["label"], t["n_edits"], "edits; final percentile", t["pct"][-1])
    print(json.dumps(shares))


if __name__ == "__main__":
    main()
