import json

import pytest
from conftest import CLAUDE, EXAMPLES

from traceml_toolkit.labeling import inputs, parse, render, runner
from traceml_toolkit.labeling.prompts import action as action_prompt
from traceml_toolkit.labeling.prompts import state as state_prompt


def _example(name=CLAUDE):
    d = EXAMPLES / name
    return d, json.loads((d / "extracted" / "trajectory.json").read_text())


def test_state_inputs_match_shipped_example():
    d, traj = _example()
    rebuilt = inputs.build_state_records(d / "extracted", traj)
    assert rebuilt == inputs.read_jsonl(d / "labels" / "state_input.jsonl")


def test_action_inputs_match_shipped_example():
    d, traj = _example()
    states = inputs.read_jsonl(d / "labels" / "state_output.jsonl")
    rebuilt = inputs.build_action_records(d / "extracted", traj, states)
    shipped = inputs.read_jsonl(d / "labels" / "action_input.jsonl")
    assert [r["v_new"] for r in rebuilt] == [r["v_new"] for r in shipped]
    for a, b in zip(rebuilt, shipped):
        assert a["code_diff"] == b["code_diff"]
        assert a.get("n_added") == b.get("n_added") and a.get("n_removed") == b.get("n_removed")


def test_make_diff_counts_and_truncates():
    diff, add, rem = inputs.make_diff("a\nb\nc\n", "a\nB\nc\nd\n")
    assert (add, rem) == (2, 1)
    big, _, _ = inputs.make_diff("", "x = 1\n" * 5000)
    assert len(big) == inputs.MAX_DIFF_CHARS


def test_main_diff_scope_differs_from_full():
    d, traj = _example()
    states = inputs.read_jsonl(d / "labels" / "state_output.jsonl")
    full = inputs.build_action_records(d / "extracted", traj, states, diff_scope="full")
    main = inputs.build_action_records(d / "extracted", traj, states, diff_scope="main")
    assert any(a["code_diff"] != b["code_diff"] for a, b in zip(full, main))
    with pytest.raises(ValueError):
        inputs.build_action_records(d / "extracted", traj, states, diff_scope="all")


def test_system_prompts_use_bundled_schemas():
    s, a = state_prompt.build_system_prompt(), action_prompt.build_system_prompt()
    for tag in ("data_io", "feature_eng", "ensemble_blend", "infra_util"):
        assert f"  - {tag}:" in s
    for act in ("augmentation", "ensemble", "housekeeping"):
        assert f"  - {act}:" in a
    assert "INTENT DISAMBIGUATION RULES" in a


def test_parse_response_variants():
    ok = action_prompt.parse_response(json.dumps({
        "coarse_actions": ["model", "bogus"], "intent": "optimization",
        "fine_actions": [{"action": "other_model", "parent": "model", "confidence": "weird"}]}))
    assert ok["coarse_actions"] == ["model"]
    assert ok["intents"] == [{"intent": "optimization", "confidence": "mid"}]
    assert ok["fine_actions"][0]["confidence"] == "mid" and "proposed_tag" in ok["fine_actions"][0]
    bad = action_prompt.parse_response('{"coarse_actions": ["model"')
    assert bad["coarse_actions"] == [] and bad["intent"] == "PARSE_ERROR"
    assert action_prompt.parse_response("[1, 2]")["intent"] == "PARSE_ERROR"


def test_parse_state_output_variants():
    got = parse.parse_state_output(json.dumps({
        "coarse_tags": ["model_def", 3], "summary": "x" * 400, "keywords": list("abcdefghij"),
        "fine_tags": [{"tag": "lgbm", "parent": "model_def", "confidence": "HIGH"}, {"tag": "no_parent"}]}))
    assert got["coarse_tags"] == ["model_def"]
    assert len(got["summary"]) == 300 and len(got["keywords"]) == 7
    assert got["fine_tags"] == [{"tag": "lgbm", "parent": "model_def", "confidence": "mid"}]
    assert parse.parse_state_output("not json")["summary"] == "PARSE_ERROR"


# ---------------------------------------------------------------------------
# runner: fitting and retry logic, with a fake tokenizer and backend
# ---------------------------------------------------------------------------
class FakeTokenizer:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "".join(m["content"] for m in msgs)

    def encode(self, text, add_special_tokens=False):
        return [0] * (len(text) // 4)


class FakeBackend:
    """Fails every prompt of the first call when fail_first, succeeds afterwards."""
    name = "fake"

    def __init__(self, task, fail_first=True):
        self.task, self.fail_first, self.calls = task, fail_first, []

    def generate(self, prompts, *, max_new_tokens, max_model_len):
        self.calls.append({"n": len(prompts), "max_new_tokens": max_new_tokens, "max_model_len": max_model_len})
        if self.fail_first and len(self.calls) == 1:
            return ['{"coarse_actions": [' for _ in prompts]  # truncated JSON
        good = ({"coarse_tags": ["model_def"], "fine_tags": [], "summary": "s", "keywords": []}
                if self.task == "state" else {"coarse_actions": ["model"], "intents": [{"intent": "exploration"}]})
        return [json.dumps(good) for _ in prompts]

    def close(self):
        pass


def _action_records():
    d, traj = _example()
    return inputs.build_action_records(d / "extracted", traj,
                                       inputs.read_jsonl(d / "labels" / "state_output.jsonl"))


def test_retry_recovers_rows_in_small_batches():
    recs = _action_records()
    be = FakeBackend("action")
    rows, stats = runner.label_task("action", recs, backend=be, tokenizer=FakeTokenizer(), name="m",
                                    **runner.DEFAULTS["action"])
    assert stats["n_first_pass_failed"] == len(recs) and stats["n_retry_recovered"] == len(recs)
    assert stats["n_failed"] == 0
    assert all(parse.is_parsed("action", r) for r in rows)
    retries = be.calls[1:]
    assert all(c["n"] <= runner.RETRY_CHUNK for c in retries)
    # the retry window must hold the prompt plus the larger output cap
    assert all(c["max_model_len"] > runner.RETRY_MAX_NEW_TOKENS for c in retries)
    assert all(c["max_new_tokens"] == runner.RETRY_MAX_NEW_TOKENS for c in retries)


def test_oversize_prompt_is_refit_not_dropped():
    recs = _action_records()
    big = dict(recs[0], code_diff="+ x = 1\n" * 300)  # 300 diff lines at default truncation
    tok = FakeTokenizer()
    budget = len(render.render(tok, render.system_prompt("action"), render.user_prompt("action", big, 60))) // 4
    fitted = render.fit_prompt("action", big, tok, budget)
    assert fitted.text is not None and fitted.level is not None and fitted.refit
    too_small = render.fit_prompt("action", big, tok, budget=10)
    assert too_small.text is None


def test_unfittable_rows_are_marked_oversize(monkeypatch):
    recs = _action_records()[:2]
    monkeypatch.setattr(runner.config, "LABELER_MAX_POSITIONS", 7000)  # tiny model window
    be = FakeBackend("action", fail_first=False)
    rows, stats = runner.label_task("action", recs, backend=be, tokenizer=FakeTokenizer(), name="m",
                                    max_model_len=1000, max_new_tokens=100)
    assert stats["n_oversize"] == 2 and stats["n_failed"] == 2
    assert all(r.get("skipped") == "oversize" for r in rows)
