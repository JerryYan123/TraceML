"""Render labeler prompts with the model's chat template and fit them into a token budget.

The released ``infer_vllm.py`` silently skipped any prompt longer than
``max_model_len - max_output_tokens - 8`` tokens, which left rows unlabeled
with no trace (one transition of the Claude Code demo run). Here a prompt that
does not fit is re-rendered with progressively shorter code/diff excerpts, and
only a prompt that cannot fit even at the smallest excerpt is reported as
oversize.
"""
from __future__ import annotations

from dataclasses import dataclass

from traceml_toolkit.labeling.prompts import action as action_prompt
from traceml_toolkit.labeling.prompts import state as state_prompt

# Excerpt sizes tried in order; None = the released default truncation.
REFIT_LEVELS = {
    "state": (None, 800, 500, 300, 150),
    "action": (None, 200, 120, 60, 30),
}
SAFETY_MARGIN = 8


@dataclass
class FittedPrompt:
    text: str | None     # rendered prompt, None if it could not fit
    n_tokens: int        # tokens of the last rendering tried
    level: int | None    # excerpt size used (None = released default)

    @property
    def refit(self) -> bool:
        return self.text is not None and self.level is not None


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir))


def system_prompt(task: str) -> str:
    return state_prompt.build_system_prompt() if task == "state" else action_prompt.build_system_prompt()


def user_prompt(task: str, rec: dict, level: int | None = None) -> str:
    mod = state_prompt if task == "state" else action_prompt
    return mod.build_user_prompt(rec, max_lines=level)


def render(tokenizer, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def fit_prompt(task: str, rec: dict, tokenizer, budget: int, count=None) -> FittedPrompt:
    """Render `rec` so that it uses at most `budget` tokens.

    Records that already carry a rendered ``system``/``user`` pair are used as is.
    `count` overrides the token counter (used by tests).
    """
    count = count or (lambda t: count_tokens(tokenizer, t))
    if "system" in rec and "user" in rec:
        text = render(tokenizer, rec["system"], rec["user"])
        n = count(text)
        return FittedPrompt(text if n <= budget else None, n, None)
    sys_text = system_prompt(task)
    n = 0
    for level in REFIT_LEVELS[task]:
        text = render(tokenizer, sys_text, user_prompt(task, rec, level))
        n = count(text)
        if n <= budget:
            return FittedPrompt(text, n, level)
    return FittedPrompt(None, n, REFIT_LEVELS[task][-1])
