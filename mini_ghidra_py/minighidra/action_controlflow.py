from __future__ import annotations
from .paramid import PcodeLikeGraph
from .action import Action


class ActionUnreachable(Action):
    """Removes blocks that are unreachable from the entry point."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "unreachable")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if not hasattr(data, "ssa_function"): return 0
        blocks = data.ssa_function.blocks
        if not blocks: return 0
        
        visited = set()
        stack = [blocks[0]]
        while stack:
            b = stack.pop()
            if b.start not in visited:
                visited.add(b.start)
                for s in b.successors:
                    s_block = next((bx for bx in blocks if bx.start == s), None)
                    if s_block and s_block.start not in visited:
                        stack.append(s_block)
                        
        to_remove = [b for b in blocks if b.start not in visited]
        if not to_remove: return 0
        
        for b in to_remove:
            # removing instructions from data.ops
            ops_to_delete = [op for op in data.ops if op.addr >= b.start and op.addr < b.start + b.size]
            for op in ops_to_delete:
                data.op_destroy(op)
                
            blocks.remove(b)
            self.count += 1
            
        # Clean up any edges pointing to removed blocks
        removed_starts = {b.start for b in to_remove}
        for b in blocks:
            b.successors = [s for s in b.successors if s not in removed_starts]
            
        return self.count

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


class ActionNormalizeBranches(Action):
    """Flips conditional branches to canonical forms."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "normalizebranches")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if not hasattr(data, "structure"):
            return 0
            
        graph = data.structure
        for bb in graph.blocks:
            if bb.type != "basic" or len(bb.out_edges) != 2:
                continue
                
            cbranch = bb.pcode_ops[-1] if hasattr(bb, "pcode_ops") and bb.pcode_ops else None
            if not cbranch or cbranch.opcode != "CBRANCH":
                continue
                
            fliplist = []
            if data.op_flip_in_place_test(cbranch, fliplist, True) != 0:
                continue
                
            data.op_flip_in_place_execute(fliplist)
            
            # flip out edges
            bb.out_edges = [bb.out_edges[1], bb.out_edges[0]]
            
            self.count += 1
            
        # clear dead ops (ActionDeadCode will clean up usually, but we could do it here)
        return self.count
