from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
DATASET_SUBSET = FIXTURES / "dataset_subset"
EXAMPLES = ROOT / "examples"
EXPECTED = ROOT / "tests" / "expected"
CLAUDE = "claudecode_haiku45_1h_commonlit"
GEMINI = "geminicli_flash25_1h_commonlit"


def pytest_addoption(parser):
    parser.addoption("--regen-expected", action="store_true",
                     help="rewrite tests/expected/*.json from the current code")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never read or write the user's cache, never download."""
    monkeypatch.setenv("TRACEML_HOME", str(tmp_path / "traceml_home"))
    monkeypatch.delenv("TRACEML_DATASET_DIR", raising=False)
    monkeypatch.delenv("TRACEML_MODELS_DIR", raising=False)
    from traceml_toolkit import assets

    def no_download(*a, **k):
        raise AssertionError("tests must not download from HuggingFace")
    monkeypatch.setattr(assets, "download", no_download)


@pytest.fixture
def dataset_dir() -> Path:
    return DATASET_SUBSET


@pytest.fixture
def example(tmp_path):
    """A writable copy of a shipped example run."""
    def _copy(name: str) -> Path:
        dst = tmp_path / name
        shutil.copytree(EXAMPLES / name, dst, ignore=shutil.ignore_patterns("figs", "report.*", "data"))
        return dst
    return _copy


def _git(d: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@example.com",
                    "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
                   check=True, capture_output=True)


@pytest.fixture
def git_sidecar(tmp_path):
    """A recorded run: git sidecar with grading tags, four grader calls, three code states."""
    def _make(slug: str = "commonlitreadabilityprize", scores=("0.9", "0.95", "0.97", None),
              name: str = "run_x_commonlitreadabilityprize") -> Path:
        d = tmp_path / name
        d.mkdir()
        _git(d, "init", "-q")
        (d / "run_meta.json").write_text(json.dumps({"harness": "claude-code", "competition": slug}))
        (d / "task.md").write_text("task")
        # v1: baseline
        (d / "train.py").write_text("import numpy as np\nprint('baseline')\n")
        (d / "submission.csv").write_text("id,target\n1,0.1\n")
        _git(d, "add", "-A")
        _git(d, "commit", "-qm", "v1")
        _git(d, "tag", f"v1-true-{scores[0]}")
        # v2: code change
        (d / "train.py").write_text("import numpy as np\nprint('model two')\n")
        (d / "submission.csv").write_text("id,target\n1,0.2\n")
        _git(d, "add", "-A")
        _git(d, "commit", "-qm", "v2")
        _git(d, "tag", f"v2-true-{scores[1]}")
        # v3: submission and metrics only -> same code state as v2
        (d / "submission.csv").write_text("id,target\n1,0.3\n")
        (d / "metrics.json").write_text('{"cv": 0.5}')
        (d / "weights.npy").write_bytes(b"\x00" * 16)
        _git(d, "add", "-A")
        _git(d, "commit", "-qm", "v3")
        _git(d, "tag", f"v3-true-{scores[2]}")
        # v4: new helper file, grading failed
        (d / "features.py").write_text("def f(x):\n    return x * 2\n")
        (d / "submission.csv").write_text("id,target\n1,0.4\n")
        _git(d, "add", "-A")
        _git(d, "commit", "-qm", "v4")
        _git(d, "tag", "v4-na" if scores[3] is None else f"v4-true-{scores[3]}")
        return d
    return _make


def load_json(p: Path):
    return json.loads(Path(p).read_text())
