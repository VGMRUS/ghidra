from __future__ import annotations
from .paramid import PcodeLikeGraph
from .action import Action
from .merge import Merge

def _get_merge(data: PcodeLikeGraph) -> Merge:
    if not hasattr(data, "merge"):
        data.merge = Merge(data) # type: ignore
    return data.merge # type: ignore

class ActionAssignHigh(Action):
    def __init__(self, group: str) -> None:
        super().__init__(group, "assignhigh")

    def apply(self, data: PcodeLikeGraph) -> int:
        _get_merge(data).assign_highs()
        self.count = 1
        return 0

class ActionMergeRequired(Action):
    def __init__(self, group: str) -> None:
        super().__init__(group, "mergerequired")

    def apply(self, data: PcodeLikeGraph) -> int:
        _get_merge(data).merge_op("MULTIEQUAL")
        self.count = 1
        return 0

class ActionMarkExplicit(Action):
    def __init__(self, group: str) -> None:
        super().__init__(group, "markexplicit")

    def apply(self, data: PcodeLikeGraph) -> int:
        _get_merge(data).mark_explicit_types()
        self.count = 1
        return 0

class ActionMarkImplied(Action):
    def __init__(self, group: str) -> None:
        super().__init__(group, "markimplied")

    def apply(self, data: PcodeLikeGraph) -> int:
        _get_merge(data).mark_implied_types()
        self.count = 1
        return 0

class ActionMergeCopy(Action):
    def __init__(self, group: str) -> None:
        super().__init__(group, "mergecopy")

    def apply(self, data: PcodeLikeGraph) -> int:
        _get_merge(data).merge_op("COPY")
        self.count = 1
        return 0

class ActionMergeAdjacent(Action):
    def __init__(self, group: str) -> None:
        super().__init__(group, "mergeadjacent")

    def apply(self, data: PcodeLikeGraph) -> int:
        _get_merge(data).merge_adjacent()
        self.count = 1
        return 0

class ActionMergeType(Action):
    def __init__(self, group: str) -> None:
        super().__init__(group, "mergetype")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 1
        return 0


class ActionDominantCopy(Action):
    """This action operates on the SSA merge phase in Ghidra to trim copies.
    In Python minighidra, we currently defer SSA structural trims."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "dominantcopy")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0
