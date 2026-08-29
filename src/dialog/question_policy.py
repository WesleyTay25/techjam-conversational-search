"""Expected-information-gain clarification policy. OWNER: Branch D.

Scores every askable attribute by how evenly its answer would split the *live*
candidate pool, discounted by the probability the customer can answer it at all.
Never asks empty-handed: the caller always ships ten recommendations alongside
the question.
"""

from __future__ import annotations

from typing import Sequence

from src.contracts import DialogState, Product


class InfoGainQuestionPolicy:
    def choose(self, state: DialogState, pool: Sequence[Product]) -> tuple[str | None, str]:
        raise NotImplementedError("Branch D")

    def expected_gain(self, attribute: str, state: DialogState, pool: Sequence[Product]) -> float:
        raise NotImplementedError("Branch D")
