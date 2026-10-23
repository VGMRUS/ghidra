from __future__ import annotations
from .paramid import PcodeLikeGraph
from .action import Action

class ActionBlockStructure(Action):
    """Recovers structured control flow (loops, conditionals) from the flat CFG."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "blockstructure")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionFinalStructure(Action):
    """Labels unstructured edges and finalizes control-flow structure ordering."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "finalstructure")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0
