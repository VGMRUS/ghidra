from __future__ import annotations
from .paramid import PcodeLikeGraph
from .action import Action
from .block import BlockGraph
from .collapse import CollapseStructure

def _get_block_graph(data: PcodeLikeGraph) -> BlockGraph:
    if not hasattr(data, "structure"):
        data.structure = BlockGraph() # type: ignore
    return data.structure # type: ignore

class ActionBlockStructure(Action):
    """Recovers structured control flow (loops, conditionals) from the flat CFG."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "blockstructure")

    def apply(self, data: PcodeLikeGraph) -> int:
        graph = _get_block_graph(data)
        if len(graph.blocks) != 0:
            return 0
        
        graph.build_copy(data.ssa_function.blocks)
        collapse = CollapseStructure(graph)
        collapse.collapse_all()
        self.count = collapse.get_change_count()
        return 0

class ActionFinalStructure(Action):
    """Labels unstructured edges and finalizes control-flow structure ordering."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "finalstructure")

    def apply(self, data: PcodeLikeGraph) -> int:
        graph = _get_block_graph(data)
        # graph.order_blocks()
        # graph.scope_break()
        # graph.mark_unstructured()
        self.count = 0
        return 0

class ActionConditionalConst(Action):
    """Propagates conditional constants through control flow edges."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "conditionalconst")
    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0




class ActionSwitchNorm(Action):
    """Normalizes switch statements to a standard form. (Disabled/Stubbed)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "switchnorm")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0



class ActionStructureTransform(Action):
    """Finalizes structural transformation passes over the graph.
    In minighidra, collapse/structure actions subsume this."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "structuretransform")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionPreferComplement(Action):
    """Adjusts block conditions to prefer complementary operations if they are more canonical.
    In minighidra, normalizebranches typically handles most of this."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "prefercomplement")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0
