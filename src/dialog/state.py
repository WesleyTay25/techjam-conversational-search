"""Conversational state machine: accumulation, decay, override erasure.
OWNER: Branch C.
"""

from __future__ import annotations

from typing import Sequence

from src.contracts import DialogState, Slot


class StateMachine:
    def start(self, session_id: str, user_profile: dict) -> DialogState:
        raise NotImplementedError("Branch C")

    def ingest(self, state: DialogState, message: str, turn: int) -> DialogState:
        """Route the message, extract slots, apply accumulation or erasure."""
        raise NotImplementedError("Branch C")

    def apply_override(self, state: DialogState, new_value: str) -> None:
        """Zero the weight of superseded slots and promote the new intent."""
        raise NotImplementedError("Branch C")

    def decay(self, state: DialogState) -> None:
        """Age soft slots so stale preferences stop dominating late turns."""
        raise NotImplementedError("Branch C")
