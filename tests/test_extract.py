import json

import pytest

from traceml_toolkit import detect, extract


@pytest.mark.parametrize("tag,expected", [
    ("v1-true-0.5", (1, 0.5)),
    ("v2-true--0.5", (2, -0.5)),
    ("v3-true-1e-05", (3, 1e-05)),
    ("v4-true-2.5E+2", (4, 250.0)),
    ("v12-true-0.73304", (12, 0.73304)),
    ("v5-na", (5, None)),
    ("v6-true-nan", None),
])
def test_parse_tag_accepts_recorder_scores(tag, expected):
    got = extract.parse_tag(tag)
    if expected is None:
        assert got is None or got[1] is None
    else:
        assert got == expected


@pytest.mark.parametrize("tag", ["v6-true-", "foo", "v7-true-abc", "1-true-0.5", "v1-false-0.5"])
def test_parse_tag_rejects_non_grading_tags(tag):
    assert extract.parse_tag(tag) is None


def test_detect_codex_tags(git_sidecar):
    run = git_sidecar()
    info = detect.detect(run)
    assert info["layout"] == "codex-tags"
    assert info["n_tags"] == 4 and info["n_scored_tags"] == 3
    assert info["slug"] == "commonlitreadabilityprize"


def test_detect_regrade_when_no_scored_tags(git_sidecar):
    run = git_sidecar()
    import subprocess
    for t in ("v1-true-0.9", "v2-true-0.95", "v3-true-0.97"):
        subprocess.run(["git", "-C", str(run), "tag", "-d", t], check=True, capture_output=True)
    assert detect.detect(run)["layout"] == "codex-regrade"


def test_detect_aide_journal(tmp_path):
    run = tmp_path / "run_aide_commonlitreadabilityprize"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "journal.json").write_text(json.dumps({"nodes": []}))
    assert detect.detect(run)["layout"] == "aide"


def test_detect_unsupported(tmp_path):
    with pytest.raises(detect.UnsupportedRunLayout):
        detect.detect(tmp_path)


def test_infer_slug_from_dir_name(tmp_path):
    d = tmp_path / "run_20260801_190617_haiku45_rec_1h_commonlitreadabilityprize"
    d.mkdir()
    assert detect.infer_slug(d) == "commonlitreadabilityprize"


def test_extract_codex_tags_lower_is_better(git_sidecar, tmp_path):
    run = git_sidecar()
    out = tmp_path / "out" / "extracted"
    traj = extract.extract_codex_tags(run, out, "commonlitreadabilityprize")
    assert traj["score_direction"] == "lower"  # from the bundled manifest
    assert traj["n_tags_total"] == 4
    assert traj["version_count"] == 3
    assert traj["n_dedup_merged"] == 1
    v1, v2, v3 = traj["versions"]
    assert v2["agent_metadata"]["dedup_scores"] == [0.95, 0.97]
    assert v2["linked_score"] == 0.95  # best of the merged calls, lower is better
    assert v3["linked_score"] is None
    assert v2["agent_metadata"]["code_files"] == ["train.py"]  # metrics/npy/submission ignored
    assert v3["agent_metadata"]["code_files"] == ["features.py", "train.py"]
    assert (out / v1["code_path"]).read_text().endswith("print('baseline')\n")
    assert "# === features.py ===" in (out / v3["code_all_path"]).read_text()
    assert traj["run_name"] == run.name
    assert json.loads((out / "trajectory.json").read_text()) == traj


def test_extract_codex_tags_higher_is_better(git_sidecar, tmp_path):
    run = git_sidecar(slug="google-quest-challenge", name="run_x_google-quest-challenge")
    traj = extract.extract_codex_tags(run, tmp_path / "e", "google-quest-challenge")
    assert traj["score_direction"] == "higher"
    assert traj["versions"][1]["linked_score"] == 0.97


def test_unknown_slug_defaults_to_higher(git_sidecar, tmp_path, capsys):
    run = git_sidecar(slug="not-a-competition", name="run_x_not-a-competition")
    traj = extract.extract_codex_tags(run, tmp_path / "e", "not-a-competition")
    assert traj["score_direction"] == "higher"
    assert "unknown score direction" in capsys.readouterr().out


def test_extract_aide_journal_uses_maximize_flag(tmp_path):
    run = tmp_path / "run_aide_x"
    (run / "logs").mkdir(parents=True)
    nodes = [
        {"id": "a", "parent": None, "ctime": 1, "code": "print(1)", "metric": {"value": 0.5, "maximize": False}},
        {"id": "b", "parent": "a", "ctime": 2, "code": "print(2)", "metric": {"value": 0.4, "maximize": False}},
        {"id": "c", "parent": None, "ctime": 0, "code": "", "metric": None},
    ]
    (run / "logs" / "journal.json").write_text(json.dumps({"nodes": nodes}))
    traj = extract.extract_aide_journal(run, tmp_path / "e", slug=None)
    assert traj["version_count"] == 2
    assert traj["score_direction"] == "lower"
    assert [v["linked_score"] for v in traj["versions"]] == [0.5, 0.4]
    assert traj["score_type"] == "cv"
