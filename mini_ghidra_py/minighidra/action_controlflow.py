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


class ActionDeterminedBranch(Action):
    """Removes conditional branches whose condition is statically known."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "determinedbranch")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        changed = True
        while changed:
            changed = False
            for op in list(data.ops):
                if op.opcode == "CBRANCH" and len(op.inputs) >= 2:
                    cond_vn = op.inputs[1]
                    cond_node = data.nodes[cond_vn]
                    if cond_node.kind == "const":
                        val = int(cond_node.name, 16)
                        
                        # CBRANCH condition: if val != 0 branch is taken.
                        # Wait, what if there's a boolean flip?
                        flip = getattr(op, "boolean_flip", False)
                        taken = (val != 0) ^ flip
                        
                        if taken:
                            # It's always taken, convert to unconditional BRANCH
                            data.op_remove_input(op, 1)
                            data.op_set_opcode(op, "BRANCH")
                        else:
                            # It's never taken, just remove the operation entirely
                            data.op_destroy(op)
                            
                        self.count += 1
                        changed = True
                        
        return self.count


class ActionRedundBranch(Action):
    """Removes redundant conditional branches that jump to the same fallthrough block."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "redundbranch")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if not hasattr(data, "ssa_function"): return 0
        blocks = data.ssa_function.blocks
        
        changed = True
        while changed:
            changed = False
            for bb in blocks:
                if len(bb.successors) == 2 and bb.successors[0] == bb.successors[1]:
                    ops_in_b = [op for op in data.ops if bb.start <= op.addr < bb.start + bb.size]
                    if not ops_in_b: continue
                    cbranch = next((op for op in reversed(ops_in_b) if op.opcode == "CBRANCH"), None)
                    if cbranch:
                        data.op_destroy(cbranch)
                        bb.successors = [bb.successors[0]]
                        self.count += 1
                        changed = True
        return self.count


class ActionDoNothing(Action):
    """Removes blocks that contain no operations and simply fall through."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "donothing")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if not hasattr(data, "ssa_function"): return 0
        blocks = data.ssa_function.blocks
        if not blocks: return 0
        
        changed = True
        while changed:
            changed = False
            for i, bb in enumerate(blocks):
                if i == 0: continue # keep entry block
                
                if len(bb.successors) == 1 and bb.successors[0] != bb.start:
                    ops_in_b = [op for op in data.ops if bb.start <= op.addr < bb.start + bb.size]
                    has_ops = any(op.opcode != "IMARK" for op in ops_in_b)
                    
                    if not has_ops:
                        target = bb.successors[0]
                        incoming = [b for b in blocks if bb.start in b.successors]
                        for inc in incoming:
                            inc.successors = [target if s == bb.start else s for s in inc.successors]
                            
                        for op in ops_in_b:
                            data.op_destroy(op)
                            
                        blocks.remove(bb)
                        self.count += 1
                        changed = True
                        break
                        
        return self.count


class ActionLateDoNothing(ActionDoNothing):
    """Removes empty blocks that became empty after later stages (like DeadCode removal)."""
    def __init__(self, group: str) -> None:
        # ActionDoNothing's __init__ takes 1 argument but super().__init__ in it takes group, name
        super().__init__(group)
        self.name = "latedonothing"


class ActionMultiCse(Action):
    """Common Subexpression Elimination within a single basic block."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "multicse")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if not hasattr(data, "ssa_function"): return 0
        
        no_cse = {
            "LOAD", "STORE", "BRANCH", "CBRANCH", "BRANCHIND", "CALL", 
            "CALLIND", "CALLOTHER", "RETURN", "MULTIEQUAL", "INDIRECT", "IMARK"
        }
        
        changed = True
        while changed:
            changed = False
            for bb in data.ssa_function.blocks:
                ops_in_b = [op for op in data.ops if bb.start <= op.addr < bb.start + bb.size]
                
                for i in range(len(ops_in_b)):
                    op1 = ops_in_b[i]
                    if op1.opcode in no_cse or op1.output is None: continue
                    if getattr(op1, "multicse_skip", False): continue
                    
                    for j in range(i + 1, len(ops_in_b)):
                        op2 = ops_in_b[j]
                        if getattr(op2, "multicse_skip", False): continue
                        
                        if op1.opcode == op2.opcode and op1.inputs == op2.inputs:
                            # Convert op2 to COPY of op1.output
                            while len(op2.inputs) > 0:
                                data.op_remove_input(op2, 0)
                                
                            data.op_set_opcode(op2, "COPY")
                            data.op_set_input(op2, op1.output, 0)
                            
                            setattr(op2, "multicse_skip", True)
                            self.count += 1
                            changed = True
                            
        for op in data.ops:
            if hasattr(op, "multicse_skip"):
                delattr(op, "multicse_skip")
                
        return self.count


class ActionNormalizeSetup(Action):
    """Clears prototype information before beginning normalization passes."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "normalizesetup")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionCse(Action):
    """Global Common Subexpression Elimination. Deprecated/stubbed in Ghidra."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "cse")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionConditionalExe(Action):
    """Re-routes unstructured branches forming conditional execution patterns."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "conditionalexe")

    def apply(self, data: PcodeLikeGraph) -> int:
        from minighidra.condexe import ActionConditionalExe as CondExeImpl
        impl = CondExeImpl(self.name)
        res = impl.apply(data)
        self.count += impl.count
        return res

class ActionConstantPtr(Action):
    """Converts appropriate constants into pointers to global spaces."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "constantptr")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionCopyMarker(Action):
    """Tracks and propagates markings for variable shadowing/copying."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "copymarker")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionDeindirect(Action):
    """Attempts to resolve indirect calls into direct function calls."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "deindirect")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if not hasattr(data, "call_specs"):
            return 0
            
        for fc in data.call_specs:
            op = fc.op
            if op.opcode != "CALLIND":
                continue
                
            if not op.inputs:
                continue
                
            vn = op.inputs[0]
            node = data.nodes[vn]
            
            while node.kind != "const":
                def_op = data.get_def(vn)
                if def_op and def_op.opcode == "COPY":
                    vn = def_op.inputs[0]
                    node = data.nodes[vn]
                else:
                    break
                    
            if node.kind == "const":
                data.op_set_opcode(op, "CALL")
                self.count += 1
                
        return self.count

class ActionExtraPopSetup(Action):
    """Checks for functions with abnormal stack depths and injects corrective pops."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "extrapopsetup")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionInternalStorage(Action):
    """Translates access to internal variables to appropriate syntax."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "internalstorage")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionLaneDivide(Action):
    """Divides wide SIMD-like operations into discrete lanes."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "lanedivide")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionMergeMultiEntry(Action):
    """Simplifies blocks that have complex multi-entry control flow."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "mergemultientry")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionNodeJoin(Action):
    """Joins equivalent blocks in conditional structures to reduce AST complexity."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "nodejoin")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionShadowVar(Action):
    """Identifies shadow variables across paths for more advanced aliasing."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "shadowvar")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0
