"""Message -> typed slots. OWNER: Branch C.

Must preserve the verbatim constraint substring alongside the typed value:
the raw string is what the exact-constraint index keys on, the typed value is
what the structured route and question policy reason over.
"""

from __future__ import annotations

from src.contracts import DialogState, Slot


class SlotExtractor:
    def extract(self, message: str, state: DialogState) -> list[Slot]:
        raise NotImplementedError("Branch C")
