import json

import numpy as np
import pandas as pd
import pytest
from conftest import CLAUDE, EXPECTED, GEMINI, load_json

from traceml_toolkit import cohorts, report, schema


def test_human_best_score_uses_min_for_lower_is_better():
    idx = pd.DataFrame({
        "key_id": ["a", "b"], "comp": ["commonlitreadabilityprize", "google-quest-challenge"],
        "min_score": [0.45, 0.30], "max_score": [9.0, 0.41]})
    assert cohorts.human_best_score(idx).tolist() == [0.45, 0.41]


def test_buckets_rank_by_best_not_worst_score():
    # h0 has the best RMSE but a terrible worst submission; the old ranking put it last.
    n = 10
    idx = pd.DataFrame({
        "key_id": [f"h{i}" for i in range(n)], "comp": "commonlitreadabilityprize",
        "group": "Expert", "is_agent": False,
        "min_score": [0.40 + 0.01 * i for i in range(n)],
        "max_score": [99.0] + [0.60 + 0.01 * i for i in range(1, n)]})
    b = cohorts.human_buckets(idx)
    assert b["h0"] == "top10" and b["h9"] == "top70-100"


def test_group7_agents():
    df = pd.DataFrame({"key_id": ["c", "m1", "m2", "x"], "group": ["codex", "mlevolve", "mlevolve", "other"],
                       "is_agent": [True, True, True, True], "is_best_branch": [False, True, False, False]})
    assert cohorts.assign_group7(df, {}).tolist()[:3] == ["codex", "mlev_best", "mlev_normal"]
    assert pd.isna(cohorts.assign_group7(df, {}).iloc[3])


def test_percentile_of_claude_demo_uses_best_scores(dataset_dir):
    _, idx = cohorts.load_reference(dataset_dir, download=False)
    hs = cohorts.human_scores(idx, "commonlitreadabilityprize")
    assert len(hs) == 103
    assert report.human_percentile(hs, 0.73304, "lower") == 20.4
    assert report.human_percentile(hs, 0.49086, "lower") == 72.8
    assert report.human_percentile([], 0.5, "lower") is None


def test_jsd_properties():
    p = np.array([0.5, 0.5, 0.0])
    assert report.jsd(p, p) == pytest.approx(0.0)
    assert report.jsd([1, 0], [0, 1]) == pytest.approx(1.0)
    assert report.jsd(p, [0.2, 0.3, 0.5]) == pytest.approx(report.jsd([0.2, 0.3, 0.5], p))


def _log(tmp_path, rcs, extra=""):
    lines = ["[loop start pid=1 end=2 reminder=0]"]
    for i, rc in enumerate(rcs, 1):
        lines += [f"[respawn iter={i} remain=100s prompt=task]", f"[respawn iter={i} exited rc={rc} tag=task]"]
    (tmp_path / "native_session.log").write_text("\n".join(lines) + "\n" + extra)
    return tmp_path


def test_run_health_treats_recorder_timeout_as_capped(tmp_path):
    h = report.run_health(_log(tmp_path, [0, 0, 124]))
    assert h["sessions"] == 3 and h["failed"] == 0 and h["capped_by_recorder"] == 1 and h["healthy"]


def test_run_health_flags_crash_loops(tmp_path):
    h = report.run_health(_log(tmp_path, [0, 1, 1, 1], '{"type":"error","message":"usage limit reached"}\n'))
    assert h["failed"] == 3 and h["fatal_provider_errors"] == 1 and not h["healthy"]
    assert report.run_health(tmp_path / "missing") is None


@pytest.mark.parametrize("name", [CLAUDE, GEMINI])
def test_report_matches_expected(name, example, dataset_dir, request):
    d = example(name)
    rep = report.build_report(d, dataset_dir=dataset_dir, download=False)
    got = {k: v for k, v in rep.items() if k != "provenance"}
    path = EXPECTED / f"{name}_report.json"
    if request.config.getoption("--regen-expected"):
        path.write_text(json.dumps(got, indent=2) + "\n")
    expected = load_json(path)
    assert json.loads(json.dumps(got)) == expected
    md = (d / "report.md").read_text()
    for section in ("## Behavioral fingerprint", "## Score progress", "## Memory profile"):
        assert section in md
    assert "Run truncated" not in md
    assert (d / "figs" / "fig1_actions.png").stat().st_size > 10_000
    assert (d / "figs" / "fig2_progress.png").stat().st_size > 10_000


def test_report_without_labels(example, dataset_dir):
    d = example(CLAUDE)
    (d / "labels" / "action_output.jsonl").unlink()
    rep = report.build_report(d, dataset_dir=dataset_dir, download=False)
    assert rep["n_labeled_transitions"] == 0 and rep["jsd_bits"] == {}
    assert "No action labels yet" in (d / "report.md").read_text()


def test_report_pooled_reference_for_unknown_comp(example, dataset_dir):
    d = example(CLAUDE)
    rep = report.build_report(d, comp="titanic", dataset_dir=dataset_dir, download=False)
    assert rep["reference"] == "pooled" and rep["human_percentile"] is None
    assert set(rep["jsd_bits"]) == {"top10", "top10-40", "top40-70", "top70-100", "codex",
                                    "mlev_normal", "mlev_best"}


def test_schema_columns_match_release(example, dataset_dir):
    d = example(CLAUDE)
    st, ac = schema.build(d)
    released = pd.read_parquet(dataset_dir / "data" / "paired" / "state.parquet")
    assert list(st.columns) == list(released.columns) == schema.V3_STATE_COLS
    assert len(ac.columns) == len(schema.V3_ACTION_COLS) == 32
    assert len(st) == 13 and len(ac) == 12
    assert st["raw_code_path"].iloc[0] == "extracted/versions/v001.py"
    # score_effect comes from measured scores (RMSE: lower is better)
    first = ac.iloc[0]
    assert (first["score_old"], first["score_new"], first["score_effect"]) == (0.88072, 0.88332, "regressing")


def test_derive_score_effect():
    assert schema.derive_score_effect(1.0, 0.5, "lower") == "improving"
    assert schema.derive_score_effect(1.0, 0.5, "higher") == "regressing"
    assert schema.derive_score_effect(1.0, 1.0, "higher") == "plateau"
    assert schema.derive_score_effect(None, 1.0, "higher") == "unknown"
    assert schema.derive_score_effect(float("nan"), 1.0, "higher") == "unknown"
