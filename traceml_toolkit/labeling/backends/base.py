"""Backend protocol: turn rendered prompts into raw generations (greedy decoding)."""
from __future__ import annotations

from typing import Protocol


class LabelBackend(Protocol):
    name: str

    def generate(self, prompts: list[str], *, max_new_tokens: int, max_model_len: int) -> list[str]:
        """One greedy generation per prompt, in order. Prompts are already chat-rendered."""
        ...

    def close(self) -> None:
        ...
