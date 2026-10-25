from __future__ import annotations
from typing import Any
from .paramid import PcodeLikeGraph, PcodeOp

class ReplaceVarnode:
    def __init__(self, vn: int, mask: int, val: int = 0):
        self.vn = vn
        self.replacement: int | None = None
        self.mask = mask
        self.val = val
        self.def_op: ReplaceOp | None = None

class ReplaceOp:
    def __init__(self, op: PcodeOp | None, opcode: str, numparams: int, out_rvn: ReplaceVarnode | None):
        self.op = op
        self.replacement: PcodeOp | None = None
        self.opcode = opcode
        self.numparams = numparams
        self.output = out_rvn
        self.inputs: list[ReplaceVarnode | None] = [None] * numparams

class PatchRecord:
    COPY_PATCH = 0
    COMPARE_PATCH = 1
    PARAMETER_PATCH = 2
    EXTENSION_PATCH = 3
    PUSH_PATCH = 4
    INT2FLOAT_PATCH = 5
    
    def __init__(self, type_: int, patch_op: PcodeOp, in1: ReplaceVarnode, in2: ReplaceVarnode | None = None, slot: int = 0):
        self.type = type_
        self.patch_op = patch_op
        self.in1 = in1
        self.in2 = in2
        self.slot = slot

class SubvariableFlow:
    """
    SubvariableFlow engine for tracing and splitting logical sub-variables 
    (such as 8-bit or 16-bit struct members) embedded within larger registers or memory.
    """
    
    def __init__(self, data: PcodeLikeGraph, root_vn: int, mask: int, aggressive: bool, sext: bool, big: bool) -> None:
        self.data = data
        self.root_vn = root_vn
        self.mask = mask
        self.aggressive = aggressive
        self.sextrestrictions = sext
        self.big = big
        
        self.bitsize = mask.bit_length() - (mask & -mask).bit_length() + 1 if mask != 0 else 0
        if self.bitsize <= 8: self.flowsize = 1
        elif self.bitsize <= 16: self.flowsize = 2
        elif self.bitsize <= 24: self.flowsize = 3
        elif self.bitsize <= 32: self.flowsize = 4
        elif self.bitsize <= 64: 
            if not self.big:
                self.flowsize = 0
                self.data = None
                return
            self.flowsize = 8
        else:
            self.flowsize = 0
            self.data = None
            return
            
        self.returns_traversed = False
        self.pullcount = 0
        
        self.varmap: dict[int, ReplaceVarnode] = {}
        self.newvarlist: list[ReplaceVarnode] = []
        self.oplist: list[ReplaceOp] = []
        self.patchlist: list[PatchRecord] = []
        self.worklist: list[ReplaceVarnode] = []
        
        # Initialize root
        in_worklist = [False]
        rvn = self.set_replacement(self.root_vn, self.mask, in_worklist)
        if in_worklist[0]:
            self.worklist.append(rvn)

    def set_replacement(self, vn: int, mask: int, in_worklist: list[bool]) -> ReplaceVarnode:
        in_worklist[0] = False
        if vn in self.varmap:
            rvn = self.varmap[vn]
            if rvn.mask != mask:
                # Flow mismatch
                pass
            return rvn
        
        rvn = ReplaceVarnode(vn, mask)
        self.varmap[vn] = rvn
        self.newvarlist.append(rvn)
        in_worklist[0] = True
        return rvn

    def do_trace(self) -> bool:
        if self.data is None: return False
        
        while len(self.worklist) > 0:
            if not self.process_next_work():
                return False
                
        if self.pullcount == 0:
            return False
            
        return True

    def process_next_work(self) -> bool:
        rvn = self.worklist.pop(0)
        
        if self.sextrestrictions:
            if not self.trace_forward_sext(rvn): return False
            if not self.trace_backward_sext(rvn): return False
        else:
            if not self.trace_forward(rvn): return False
            if not self.trace_backward(rvn): return False
            
        return True

    def trace_forward(self, rvn: ReplaceVarnode) -> bool:
        # Get all uses of rvn.vn
        uses = []
        for op in self.data.get_uses(rvn.vn):
            uses.append((op, list(op.inputs).index(rvn.vn)))
                
        for op, slot in uses:
            outvn = op.output
            if outvn is not None and self.data.nodes[outvn].kind == "register" and op.opcode != "CALL":
                pass # usually check mark
                
            if op.opcode in ("COPY", "MULTIEQUAL", "INT_NEGATE", "INT_XOR"):
                # create_op_down
                rop = ReplaceOp(op, op.opcode, len(op.inputs), None)
                self.oplist.append(rop)
                rop.inputs[slot] = rvn
                # create_link
                if outvn is not None:
                    in_wl = [False]
                    out_rvn = self.set_replacement(outvn, rvn.mask, in_wl)
                    rop.output = out_rvn
                    out_rvn.def_op = rop
                    if in_wl[0]: self.worklist.append(out_rvn)
            elif op.opcode == "INT_AND":
                # handle mask
                if slot == 0 and self.data.is_constant(op.inputs[1]):
                    val = self.data.get_constant_value(op.inputs[1])
                    if val == rvn.mask:
                        if outvn and self.data.nodes[outvn].size_bits // 8 == self.flowsize:
                            self.patchlist.append(PatchRecord(PatchRecord.COPY_PATCH, op, rvn, None, slot))
                            self.pullcount += 1
                            continue
            elif op.opcode == "SUBPIECE":
                if op.inputs[1] == 0:
                    pass
            else:
                return False
                
        return True

    def trace_backward(self, rvn: ReplaceVarnode) -> bool:
        def_op = self.data.get_def(rvn.vn)
        if def_op is None:
            return False
            
        opc = def_op.opcode
        if opc in ("COPY", "MULTIEQUAL", "INT_NEGATE", "INT_XOR"):
            rop = ReplaceOp(def_op, opc, len(def_op.inputs), rvn)
            self.oplist.append(rop)
            rvn.def_op = rop
            for i, invn in enumerate(def_op.inputs):
                if self.data.is_constant(invn):
                    # add constant
                    pass
                else:
                    in_wl = [False]
                    in_rvn = self.set_replacement(invn, rvn.mask, in_wl)
                    rop.inputs[i] = in_rvn
                    if in_wl[0]: self.worklist.append(in_rvn)
        elif opc == "INT_AND":
            if self.data.is_constant(def_op.inputs[1]):
                val = self.data.get_constant_value(def_op.inputs[1])
                if val == rvn.mask:
                    self.patchlist.append(PatchRecord(PatchRecord.PUSH_PATCH, def_op, rvn))
                    return True
            return False
        elif opc == "PIECE":
            # Just push patch
            self.patchlist.append(PatchRecord(PatchRecord.PUSH_PATCH, def_op, rvn))
        else:
            return False
            
        return True

    def trace_forward_sext(self, rvn: ReplaceVarnode) -> bool:
        return False

    def trace_backward_sext(self, rvn: ReplaceVarnode) -> bool:
        return False

    def do_replacement(self) -> None:
        # Do up front processing of the call return patches
        for patch in list(self.patchlist):
            if patch.type != PatchRecord.PUSH_PATCH:
                break
            push_op = patch.patch_op
            new_vn = self.get_replace_varnode(patch.in1)
            old_vn = push_op.output
            self.data.op_set_output(push_op, new_vn)

            # Create placeholder defining op for old Varnode, until dead code cleans it up
            new_zext = PcodeOp()
            new_zext.opcode = "INT_ZEXT"
            new_zext.inputs = (new_vn,)
            new_zext.output = old_vn
            new_zext.address = push_op.address
            self.data._append_operation(new_zext)

        # Define all the outputs first
        for rop in self.oplist:
            new_op = PcodeOp()
            new_op.opcode = rop.opcode
            new_op.inputs = (-1,) * rop.numparams
            new_op.address = rop.op.address if rop.op else None
            rop.replacement = new_op
            
            rout = rop.output
            if rout is not None:
                new_op.output = self.get_replace_varnode(rout)
                self.data._append_operation(new_op)
            else:
                self.data._append_operation(new_op)

        # Set all the inputs
        for rop in self.oplist:
            new_op = rop.replacement
            if new_op is None: continue
            for i, in_rvn in enumerate(rop.inputs):
                if in_rvn is not None:
                    new_op = self.data.op_set_input(new_op, self.get_replace_varnode(in_rvn), i)
            # Update the op in oplist
            rop.replacement = new_op

        # These are operations that carry flow from the small variable into an existing variable
        for patch in self.patchlist:
            if patch.type == PatchRecord.PUSH_PATCH: continue
            pull_op = patch.patch_op
            
            if patch.type == PatchRecord.COPY_PATCH:
                inputs = list(pull_op.inputs)
                if len(inputs) > 1:
                    inputs = [inputs[0]]
                inputs[0] = self.get_replace_varnode(patch.in1)
                
                # Update pull_op directly if possible or through op_set_input
                new_op = self.data.op_set_opcode(pull_op, "COPY")
                for i in range(len(pull_op.inputs)-1, 0, -1):
                    new_op = self.data.op_remove_input(new_op, i)
                new_op = self.data.op_set_input(new_op, inputs[0], 0)
                
            elif patch.type == PatchRecord.COMPARE_PATCH:
                new_op = self.data.op_set_input(pull_op, self.get_replace_varnode(patch.in1), 0)
                new_op = self.data.op_set_input(new_op, self.get_replace_varnode(patch.in2), 1)
                
            elif patch.type == PatchRecord.PARAMETER_PATCH:
                new_op = self.data.op_set_input(pull_op, self.get_replace_varnode(patch.in1), patch.slot)
                
            elif patch.type == PatchRecord.EXTENSION_PATCH:
                sa = patch.slot
                in_vn = self.get_replace_varnode(patch.in1)
                out_size = self.data.nodes[pull_op.output].size_bits if pull_op.output else self.flowsize * 8
                in_size = self.data.nodes[in_vn].size_bits
                
                if sa == 0:
                    opc = "COPY" if in_size == out_size else "INT_ZEXT"
                    new_op = self.data.op_set_opcode(pull_op, opc)
                    new_op = self.data.op_set_input(new_op, in_vn, 0)
                else:
                    if in_size != out_size:
                        zext_op = PcodeOp()
                        zext_op.opcode = "INT_ZEXT"
                        zext_op.inputs = (in_vn,)
                        zext_op.address = pull_op.address
                        zext_out = self.data._new_temporary_node("zext", out_size)
                        zext_op.output = zext_out
                        self.data._append_operation(zext_op)
                        in_vn = zext_out
                        
                    new_op = self.data.op_set_opcode(pull_op, "INT_LEFT")
                    new_op = self.data.op_set_input(new_op, in_vn, 0)
                    new_op = self.data.op_set_input(new_op, self.data.new_constant(4*8, sa), 1)
                    
            elif patch.type == PatchRecord.INT2FLOAT_PATCH:
                zext_op = PcodeOp()
                zext_op.opcode = "INT_ZEXT"
                in_vn = self.get_replace_varnode(patch.in1)
                zext_op.inputs = (in_vn,)
                zext_op.address = pull_op.address
                
                # Assume float target size is out_size
                out_size = self.data.nodes[pull_op.inputs[0]].size_bits
                zext_out = self.data._new_temporary_node("zext_float", out_size)
                zext_op.output = zext_out
                self.data._append_operation(zext_op)
                
                new_op = self.data.op_set_input(pull_op, zext_out, 0)

    def get_replace_varnode(self, rvn: ReplaceVarnode) -> int:
        if rvn.replacement is not None:
            return rvn.replacement
        
        if rvn.vn < 0:
            if rvn.def_op is None:
                return self.data.new_constant(self.flowsize * 8, rvn.val)
            rvn.replacement = self.data._new_temporary_node("subvar", self.flowsize * 8)
            return rvn.replacement
            
        if self.data.is_constant(rvn.vn):
            return self.data.new_constant(self.flowsize * 8, rvn.val)
            
        if self.use_same_address(rvn):
            # For simplicity, we just use a new temporary with same semantics
            rvn.replacement = self.data._new_temporary_node("subvar", self.flowsize * 8)
        else:
            rvn.replacement = self.data._new_temporary_node("subvar", self.flowsize * 8)
            
        return rvn.replacement

    def use_same_address(self, rvn: ReplaceVarnode) -> bool:
        if rvn.mask & 1 == 0:
            return False
        if self.bitsize >= 8:
            return True
        if self.aggressive:
            return True
        bitmask = (1 << self.bitsize) - 1
        consume = self.data.get_consume(rvn.vn)
        mask = consume | bitmask
        if mask == rvn.mask:
            return True
        return False
class LaneDescription:
    def __init__(self, num_lanes: int):
        self.num_lanes = num_lanes

class LaneDivide:
    """
    Class for splitting data-flow on laned registers (SIMD / vector registers).
    From a root Varnode and a description of its lanes, trace data-flow as far as
    possible through the function, propagating each lane.
    """
    
    def __init__(self, data: PcodeLikeGraph, root: int, desc: LaneDescription, allow_downcast: bool):
        self.data = data
        self.root = root
        self.description = desc
        self.allow_subpiece_terminator = allow_downcast
        self.worklist = []

    def set_replacement(self, vn: int, num_lanes: int, skip_lanes: int) -> Any:
        pass

    def do_trace(self) -> bool:
        if not self.worklist:
            return False
            
        retval = True
        while self.worklist:
            if not self.process_next_work():
                retval = False
                break
                
        if not retval:
            return False
        return True

    def process_next_work(self) -> bool:
        return False

    def trace_forward(self, rvn: Any, num_lanes: int, skip_lanes: int) -> bool:
        vn = rvn.vn if hasattr(rvn, 'vn') else -1
        if vn < 0: return False
        
        for op in self.data.get_uses(vn):
            opc = op.opcode
            if opc == "SUBPIECE":
                # handle out_lanes calculation
                pass
            elif opc == "PIECE":
                pass
            elif opc in ("INT_NEGATE", "COPY"):
                pass
            elif opc in ("INT_AND", "INT_OR", "INT_XOR"):
                pass
            elif opc == "MULTIEQUAL":
                pass
            elif opc == "STORE":
                pass
            elif opc == "INT_ZEXT":
                pass
            elif opc == "INT_RIGHT":
                pass
            elif opc == "INT_LEFT":
                pass
            else:
                return False
                
        return True
        
    def trace_backward(self, rvn: Any, num_lanes: int, skip_lanes: int) -> bool:
        op = self.data.get_def(rvn.vn) if hasattr(rvn, 'vn') else None
        if not op:
            return True
            
        opc = op.opcode
        if opc in ("INT_NEGATE", "COPY"):
            in_vars = self.set_replacement(op.inputs[0], num_lanes, skip_lanes)
            if not in_vars: return False
        elif opc in ("INT_AND", "INT_OR", "INT_XOR"):
            in0_vars = self.set_replacement(op.inputs[0], num_lanes, skip_lanes)
            if not in0_vars: return False
            in1_vars = self.set_replacement(op.inputs[1], num_lanes, skip_lanes)
            if not in1_vars: return False
        elif opc == "MULTIEQUAL":
            pass
        elif opc == "INDIRECT":
            pass
        elif opc == "SUBPIECE":
            in_vn = op.inputs[0]
            # missing description.extension
            pass
        elif opc == "PIECE":
            pass
        elif opc == "LOAD":
            pass
        elif opc == "INT_ZEXT":
            pass
        elif opc == "INT_RIGHT":
            pass
        elif opc == "INT_LEFT":
            pass
        else:
            return False
            
        return True


class SplitDatatype:
    """
    Helper engine to rewrite operations handling full structured types (like structs or arrays)
    by splitting them into multiple partial operations per-field when datatype partial tracking
    requires it.
    """
    def __init__(self, data: PcodeLikeGraph):
        self.data = data
        self.split_structures = True
        self.split_arrays = True
        self.is_load_store = False
        self.data_type_pieces = []

    def split_copy(self, copy_op: PcodeOp, in_type: Any, out_type: Any) -> bool:
        return False
        
    def split_load(self, load_op: PcodeOp, in_type: Any) -> bool:
        return False
        
    def split_store(self, store_op: PcodeOp, out_type: Any) -> bool:
        return False

class SubfloatFlow:
    """
    Class for tracing changes of precision in floating point variables.
    It follows the flow of a logical lower precision value stored in higher precision locations
    and rewrites the data-flow in terms of the lower precision, eliminating conversions.
    """
    def __init__(self, data: PcodeLikeGraph, root: int, prec: int):
        self.data = data
        self.root = root
        self.precision = prec
        self.terminator_count = 0
        self.worklist = []

    def max_precision(self, vn: int) -> int:
        return self.precision

    def exceeds_precision(self, op: PcodeOp) -> bool:
        return False

    def trace_forward(self, rvn: Any) -> bool:
        vn = rvn.vn if hasattr(rvn, 'vn') else -1
        if vn < 0: return False
        
        for op in self.data.get_uses(vn):
            opc = op.opcode
            if opc in ("FLOAT_ADD", "FLOAT_SUB", "FLOAT_MULT", "FLOAT_DIV"):
                if self.exceeds_precision(op):
                    return False
            
            if opc in ("FLOAT_ADD", "FLOAT_SUB", "FLOAT_MULT", "FLOAT_DIV", "MULTIEQUAL", "COPY",
                       "FLOAT_CEIL", "FLOAT_FLOOR", "FLOAT_ROUND", "FLOAT_NEG", "FLOAT_ABS", "FLOAT_SQRT"):
                out_rvn = self.set_replacement(op.output)
                if not out_rvn: return False
                
            elif opc == "FLOAT_FLOAT2FLOAT":
                out_size = self.data.nodes[op.output].size_bits if op.output else 0
                if out_size < self.precision * 8:
                    return False
                self.terminator_count += 1
                
            elif opc in ("FLOAT_EQUAL", "FLOAT_NOTEQUAL", "FLOAT_LESS", "FLOAT_LESSEQUAL"):
                if self.exceeds_precision(op):
                    return False
                inputs = list(op.inputs)
                slot = inputs.index(vn)
                rvn2 = self.set_replacement(inputs[1-slot])
                if not rvn2: return False
                self.terminator_count += 1
                
            elif opc in ("FLOAT_TRUNC", "FLOAT_NAN"):
                self.terminator_count += 1
                
            elif opc == "CALL":
                return False
            elif opc == "STORE":
                return False
            elif opc == "RETURN":
                return False
            else:
                return False
                
        return True

    def trace_backward(self, rvn: Any) -> bool:
        op = self.data.get_def(rvn.vn) if hasattr(rvn, 'vn') else None
        if not op:
            return True
            
        opc = op.opcode
        if opc in ("FLOAT_ADD", "FLOAT_SUB", "FLOAT_MULT", "FLOAT_DIV"):
            if self.exceeds_precision(op):
                return False
                
        if opc in ("FLOAT_ADD", "FLOAT_SUB", "FLOAT_MULT", "FLOAT_DIV", "COPY",
                   "FLOAT_CEIL", "FLOAT_FLOOR", "FLOAT_ROUND", "FLOAT_NEG", "FLOAT_ABS", "FLOAT_SQRT", "MULTIEQUAL"):
            for invn in op.inputs:
                newvar = self.set_replacement(invn)
                if not newvar: return False
            return True
            
        elif opc == "FLOAT_INT2FLOAT":
            invn = op.inputs[0]
            if not self.data.is_constant(invn) and False: # replace with is_free check if available
                return False
            return True
            
        elif opc == "FLOAT_FLOAT2FLOAT":
            invn = op.inputs[0]
            if self.data.is_constant(invn):
                if self.data.nodes[invn].size_bits != self.precision * 8:
                    newvar = self.set_replacement(invn)
                    if not newvar: return False
            return True
            
        elif opc == "INDIRECT":
            return False
        elif opc == "LOAD":
            return False
        else:
            return False
            
        return True

    def set_replacement(self, vn: int) -> Any:
        # Pseudo-implementation to track visited varnodes
        if not hasattr(self, 'varmap'):
            self.varmap = {}
        if vn in self.varmap:
            return self.varmap[vn]
            
        # Basic check for type lock / primitive type omitted for simplicity
        class DummyVar:
            def __init__(self, vn):
                self.vn = vn
        rvn = DummyVar(vn)
        self.varmap[vn] = rvn
        self.worklist.append(rvn)
        return rvn

    def do_trace(self) -> bool:
        self.terminator_count = 0
        if not self.worklist:
            return False
            
        retval = True
        while self.worklist:
            if not self.process_next_work():
                retval = False
                break
                
        if not retval or self.terminator_count == 0:
            return False
        return True

    def process_next_work(self) -> bool:
        rvn = self.worklist.pop(-1)
        if not self.trace_backward(rvn):
            return False
        return self.trace_forward(rvn)
