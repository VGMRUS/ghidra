from __future__ import annotations
from .paramid import PcodeLikeGraph
from .action import Action
from .merge import Merge

class ActionNameVars(Action):
    """Assigns concrete names to HighVariables using database recommendations."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "namevars")

    def apply(self, data: PcodeLikeGraph) -> int:
        if hasattr(data, "merge"):
            merge: Merge = getattr(data, "merge")
            for high in merge.highs:
                if high.name is None:
                    high.name = f"var_{high.high_id}"
                    self.count += 1
        return 0

class ActionDynamicSymbols(Action):
    """Links explicitly named dynamic symbols to variables in the graph."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "dynamicsymbols")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0
