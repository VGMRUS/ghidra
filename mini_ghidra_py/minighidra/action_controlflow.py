from __future__ import annotations
from .paramid import PcodeLikeGraph
from .action import Action

class ActionUnreachable(Action):
    """Removes blocks that are unreachable from the entry point."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "unreachable")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionForceGoto(Action):
    """Enforces unstructured goto branches when structure blocks cannot cover flow."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "forcegoto")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionRestrictLocal(Action):
    """Restricts scopes of local variables prior to dead code removal."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "restrictlocal")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0
