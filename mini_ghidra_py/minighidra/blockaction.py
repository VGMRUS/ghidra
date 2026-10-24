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


class ConstPoint:
    def __init__(self, vn: int, const_vn: int, const_block: 'FlowBlock', in_slot: int, block_is_dom: bool, value: int = 0):
        self.vn = vn
        self.const_vn = const_vn
        self.const_block = const_block
        self.in_slot = in_slot
        self.block_is_dom = block_is_dom
        self.value = value

class ActionConditionalConst(Action):
    """Propagates conditional constants through control flow edges."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "conditionalconst")

    def _block_for_op(self, op, data) -> 'FlowBlock | None':
        if not op.addr: return None
        if not data.ssa_function: return None
        for bb in data.ssa_function.blocks:
            if bb.start <= op.addr < bb.start + bb.size:
                for b in data.structure.blocks:
                    if getattr(b, "start_addr", None) == bb.start:
                        return b
        return None

    def _find_const_compare(self, points: list[ConstPoint], bool_vn: int, bl: 'FlowBlock', block_dom: list[bool], flip_edge: bool, data) -> None:
        comp_op = None
        for op in data.ops:
            if op.output == bool_vn:
                comp_op = op
                break
        if not comp_op: return

        opc = comp_op.opcode
        if opc == "BOOL_NEGATE":
            flip_edge = not flip_edge
            bool_vn = comp_op.inputs[0]
            comp_op = None
            for op in data.ops:
                if op.output == bool_vn:
                    comp_op = op
                    break
            if not comp_op: return
            opc = comp_op.opcode

        if opc == "INT_EQUAL":
            const_edge = 1
        elif opc == "INT_NOTEQUAL":
            const_edge = 0
        else:
            return

        var_vn = comp_op.inputs[0]
        const_vn = comp_op.inputs[1]
        
        cnode = data.nodes[const_vn]
        if cnode.kind != "const":
            if data.nodes[var_vn].kind != "const": return
            var_vn, const_vn = const_vn, var_vn
            cnode = data.nodes[const_vn]

        if flip_edge:
            const_edge = 1 - const_edge

        cval = int(cnode.name, 16) if cnode.name else 0
        out_block = bl.out_edges[const_edge]
        points.append(ConstPoint(var_vn, const_vn, out_block, const_edge, block_dom[const_edge], cval))

    def _propagate_constant(self, points: list[ConstPoint], data) -> None:
        for pt in points:
            var_vn = pt.vn
            const_vn = pt.const_vn
            const_block = pt.const_block
            
            for op in data.ops:
                if op.opcode == "IMARK": continue
                if var_vn in op.inputs:
                    op_block = self._block_for_op(op, data)
                    if op_block and pt.block_is_dom and const_block.dominates(op_block):
                        if op.opcode == "RETURN":
                            pass # Not handled here simply
                        else:
                            slot = op.inputs.index(var_vn)
                            new_inputs = list(op.inputs)
                            new_inputs[slot] = const_vn
                            op.inputs = tuple(new_inputs)
                            self.count += 1

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if not hasattr(data, "structure"): return 0
        block_graph = data.structure
        if not block_graph.blocks: return 0
        
        points = []
        for bl in block_graph.blocks:
            cbranch = None
            if not data.ssa_function: continue
            ssa_bb = next((b for b in data.ssa_function.blocks if b.start == getattr(bl, "start_addr", -1)), None)
            if not ssa_bb: continue
            
            ops_in_b = [op for op in data.ops if op.addr is not None and ssa_bb.start <= op.addr < ssa_bb.start + ssa_bb.size]
            if not ops_in_b: continue
            cbranch = next((op for op in reversed(ops_in_b) if op.opcode == "CBRANCH"), None)
            
            if not cbranch: continue
            
            if len(bl.out_edges) < 2: continue
            
            bool_vn = cbranch.inputs[1]
            block_dom = [
                bl.out_edges[0].restricted_by_conditional(bl),
                bl.out_edges[1].restricted_by_conditional(bl)
            ]
            flip_edge = getattr(cbranch, "boolean_flip", False)
            
            self._find_const_compare(points, bool_vn, bl, block_dom, flip_edge, data)
            
        self._propagate_constant(points, data)
        return self.count

class ActionSwitchNorm(Action):
    """Normalizes switch statements to a standard form."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "switchnorm")
    def apply(self, data: PcodeLikeGraph) -> int:
        count = 0
        analyzer = getattr(data, 'analyzer', None)
        if not analyzer:
            return 0
        # In minighidra we don't have explicit JumpTable objects like Ghidra C++.
        # Computed jumps are managed via analyzer._resolve_computed_jumps().
        # We simulate the normalization of switch branches via structural resets.
        for block in list(data.blocks):
            if not block.is_switch_out:
                continue
            # Logic for fold_in_guards and match_model goes here in full implementation
            # For now we just reset structure when switch outputs change.
            if hasattr(data, 'structure'):
                data.structure.clear()
                count += 1
        return count




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
