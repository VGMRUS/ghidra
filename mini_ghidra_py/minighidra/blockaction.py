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
        
        # graph.build_copy(data.get_basic_blocks())
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
