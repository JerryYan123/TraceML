import json
import shutil
import subprocess
import sys

import pytest
from conftest import DATASET_SUBSET, FIXTURES

from traceml_toolkit import assets, cli, config


def run_cli(*args):
    return cli.main([str(a) for a in args])


def test_version_and_help(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert "traceml-toolkit" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["analyze", "--help"])


def test_module_entry_point():
    r = subprocess.run([sys.executable, "-m", "traceml_toolkit", "--version"], capture_output=True, text=True)
    assert r.returncode == 0 and "traceml-toolkit" in r.stdout


def test_bundled_resources():
    comps = config.competitions()
    assert len(comps) == 141
    assert sum(v["score_direction"] == "lower" for v in comps.values()) == 38
    assert all(c in comps for c in config.PAIRED_COMPETITIONS)
    assert set(config.schema("schema_state")["coarse_tags"]) == set(config.COARSE_TAGS)


def test_analyze_skip_label_then_report(git_sidecar, tmp_path, capsys):
    run = git_sidecar()
    out = tmp_path / "out"
    assert run_cli("analyze", run, "--out", out, "--skip-label", "--dataset-dir", DATASET_SUBSET) == 0
    traj = json.loads((out / "extracted" / "trajectory.json").read_text())
    assert traj["version_count"] == 3
    summary = json.loads((out / "summary.json").read_text())
    assert summary["extract"]["layout"] == "codex-tags"
    rep = json.loads((out / "report.json").read_text())
    assert rep["best_score"] == 0.9 and rep["human_percentile"] is not None
    assert "No action labels yet" in (out / "report.md").read_text()


def test_analyze_refuses_output_inside_run(git_sidecar):
    run = git_sidecar()
    assert run_cli("analyze", run, "--out", run / "traceml_out", "--skip-label",
                   "--dataset-dir", DATASET_SUBSET) == cli.EXIT_USAGE


def test_unsupported_layout_exit_code(tmp_path):
    assert run_cli("extract", tmp_path, "--out", tmp_path.parent / "o") == cli.EXIT_USAGE


def test_missing_assets_exit_code(git_sidecar, tmp_path, monkeypatch):
    run = git_sidecar()

    def offline(*a, **k):
        raise assets.MissingAsset("offline")
    monkeypatch.setattr(assets, "download", offline)
    # isolated TRACEML_HOME and no network: the reference data cannot be found
    assert run_cli("analyze", run, "--out", tmp_path / "o", "--skip-label") == cli.EXIT_MISSING


def test_from_released_then_report(tmp_path):
    human = (FIXTURES / "dataset_subset" / "KEY_IDS.txt").read_text().split()[1]
    out = tmp_path / "rel"
    assert run_cli("from-released", human, "--out", out, "--dataset-dir", DATASET_SUBSET) == 0
    traj = json.loads((out / "extracted" / "trajectory.json").read_text())
    assert traj["harness"] == "human" and traj["competition"] == "commonlitreadabilityprize"
    assert run_cli("report", out, "--dataset-dir", DATASET_SUBSET) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["n_labeled_transitions"] > 0 and rep["reference"] == "paired"


def test_lfs_pointer_counts_as_missing(tmp_path):
    p = tmp_path / "model.safetensors"
    p.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3441185608\n")
    assert assets.is_lfs_pointer(p)
    real = tmp_path / "real.bin"
    real.write_bytes(b"\x00" * 2048)
    assert not assets.is_lfs_pointer(real)


def test_labeler_resolution_prefers_models_dir(tmp_path):
    d = tmp_path / "models" / "qwen3-1.7b-state" / "final"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00" * 4096)
    (d / "tokenizer.json").write_text("{}" + " " * 2048)
    assert assets.resolve_labeler("state", models_dir=tmp_path / "models", download_missing=False) == d
    with pytest.raises(assets.MissingAsset):
        assets.resolve_labeler("action", models_dir=tmp_path / "models", download_missing=False)


@pytest.mark.skipif(shutil.which("timeout") is None and shutil.which("gtimeout") is None,
                    reason="GNU timeout not installed")
def test_recorder_end_to_end(tmp_path):
    """Record a scripted 'agent' with a fake grader, then extract the run."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    grader = fake_bin / "mlebench"
    grader.write_text('#!/bin/bash\nn=$(wc -l < "$2"); echo "{\\"score\\": 0.$((n + 3))}"\n')
    grader.chmod(0o755)
    agent = fake_bin / "agent.sh"
    agent.write_text(
        "#!/bin/bash\n"
        "i=$(ls train_*.py 2>/dev/null | wc -l); i=$((i+1))\n"
        "echo \"print($i)\" > train_$i.py\n"
        "seq 1 $i > submission.csv\n"
        "sleep 2\n")
    agent.chmod(0o755)
    run = tmp_path / "run_rec_commonlitreadabilityprize"
    run.mkdir()
    (run / "task.md").write_text("do the task")
    env = {**__import__("os").environ, "TRACEML_MLEBENCH": str(grader),
           "TRACEML_MLEBENCH_CACHE": str(tmp_path), "TRACEML_POLL": "1",
           "TRACEML_BUDGET_SECONDS": "12"}
    from traceml_toolkit.record import script_path
    (run / "cont.md").write_text("continue")
    r = subprocess.run(["bash", str(script_path()), str(run), "commonlitreadabilityprize", "1",
                        "test-agent", "--", str(agent), "{PROMPT}"],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    tags = subprocess.run(["git", "-C", str(run), "tag"], capture_output=True, text=True).stdout.split()
    assert tags and all(t.startswith("v") and "-true-0." in t for t in tags)
    meta = json.loads((run / "run_meta.json").read_text())
    assert meta["harness"] == "test-agent" and meta["competition"] == "commonlitreadabilityprize"
    log = (run / "native_session.log").read_text()
    assert "[respawn iter=1 remain=" in log and "exited rc=0" in log
    out = tmp_path / "out"
    assert run_cli("extract", run, "--out", out) == 0
    traj = json.loads((out / "extracted" / "trajectory.json").read_text())
    assert traj["harness"] == "test-agent"
    assert traj["version_count"] >= 2
