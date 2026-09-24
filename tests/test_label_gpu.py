"""GPU integration test: relabel the Claude Code example with the released labelers.

Skipped unless vLLM, a CUDA GPU and both labelers are available. Run with

    TRACEML_HOME=<cache with the labelers> pytest -m gpu
"""
import json
import shutil

import pytest
from conftest import CLAUDE, EXAMPLES

from traceml_toolkit import assets
from traceml_toolkit.labeling import inputs
from traceml_toolkit.labeling.backends import vllm_available

pytestmark = pytest.mark.gpu


@pytest.fixture
def real_home(monkeypatch):
    # undo the autouse isolation: this test needs the real cache with the labelers
    import os
    home = os.environ.get("TRACEML_GPU_TEST_HOME") or os.path.expanduser("~/.cache/traceml")
    monkeypatch.setenv("TRACEML_HOME", home)
    if not vllm_available() or assets.find_labeler("state") is None or assets.find_labeler("action") is None:
        pytest.skip("needs vLLM, a CUDA GPU and downloaded labelers")


def test_vllm_labels_reproduce_shipped_example(real_home, tmp_path):
    from traceml_toolkit.labeling.runner import label_run
    out = tmp_path / CLAUDE
    shutil.copytree(EXAMPLES / CLAUDE / "extracted", out / "extracted")
    summary = label_run(out, backend="vllm", download=False)
    assert summary["tasks"]["state"]["n_failed"] == 0
    assert summary["tasks"]["action"]["n_labeled"] == 12

    def coarse(p, key):
        return {r["version_number"] if key == "coarse_tags" else r["v_new"]: set(r[key])
                for r in inputs.read_jsonl(p)}
    new = coarse(out / "labels" / "state_output.jsonl", "coarse_tags")
    ref = coarse(EXAMPLES / CLAUDE / "labels" / "state_output.jsonl", "coarse_tags")
    jac = [len(new[v] & ref[v]) / len(new[v] | ref[v]) for v in ref]
    assert sum(jac) / len(jac) >= 0.9
    log = json.loads((out / "labels" / "label_log.json").read_text())
    assert log["backend"] == "vllm"
