"""Buying / Browsing / Override / Boundary detection. OWNER: Branch C.

Two-layer by design: a fast deterministic layer keyed on the simulator's
message shapes, and a lexical//semantic fallback that still routes correctly if
the organizer paraphrases the private set. The fallback is not optional — it is
the insurance policy on 800 unseen sessions.
"""

from __future__ import annotations

from src.contracts import DialogState


class HybridIntentRouter:
    def route(self, message: str, state: DialogState) -> tuple[str, float]:
        raise NotImplementedError("Branch C")
