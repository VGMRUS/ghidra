from __future__ import annotations
from .paramid import PcodeLikeGraph
from .action import Action

class ActionVarnodeProps(Action):
    """Calculates boolean properties and non-zero masks for varnodes."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "varnodeprops")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionDirectWrite(Action):
    """Marks varnodes built out of legal parameters as direct writes."""

    def __init__(self, group: str, propagate_indirect: bool) -> None:
        super().__init__(group, f"directwrite_{propagate_indirect}")
        self.propagate_indirect = propagate_indirect

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0
