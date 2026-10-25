from typing import Optional, List, Dict
from .paramid import PcodeLikeGraph

class LaneDescription:
    def __init__(self, whole_size: int, lane_size: int):
        self.whole_size = whole_size
        self.lane_sizes = []
        self.lane_positions = []
        
        pos = 0
        while pos < whole_size:
            sz = lane_size if pos + lane_size <= whole_size else whole_size - pos
            self.lane_sizes.append(sz)
            self.lane_positions.append(pos)
            pos += sz
            
    def get_num_lanes(self) -> int:
        return len(self.lane_sizes)
        
    def get_size(self, i: int) -> int:
        return self.lane_sizes[i]
        
    def get_position(self, i: int) -> int:
        return self.lane_positions[i]
        
    def get_boundary(self, byte_pos: int) -> int:
        for i, pos in enumerate(self.lane_positions):
            if pos == byte_pos: return i
        return -1


class TransformVar:
    PIECE = 1
    PREEXISTING = 2
    NORMAL_TEMP = 3
    PIECE_TEMP = 4
    CONSTANT = 5
    
    def __init__(self, tp: int, vn: int, bits: int, bytes_: int, val: int):
        self.type = tp
        self.vn = vn
        self.val = val
        self.bit_size = bits
        self.byte_size = bytes_
        self.flags = 0
        self.def_op = None
        self.replacement = -1


class TransformOp:
    OP_REPLACEMENT = 1
    OP_PREEXISTING = 2
    OP_INDIRECT = 4
    
    def __init__(self, op: Any, opc: str, num_params: int):
        self.op = op
        self.opc = opc
        self.special = 0
        self.output = None
        self.inputs = [None] * num_params
        self.replacement = None
        self.follow = None


class TransformManager:
    """
    Classes for building large scale transforms of function data-flow.
    Equivalent to Ghidra's TransformManager.
    """
    def __init__(self, data: PcodeLikeGraph):
        self.data = data
        self.piece_map: Dict[int, TransformVar] = {}
        self.new_varnodes: List[TransformVar] = []
        self.new_ops: List[TransformOp] = []

    def clear_varnode_marks(self):
        pass

    def preserve_address(self, vn: int, bit_size: int, lsb_offset: int) -> bool:
        if (lsb_offset & 7) != 0: return False
        return True

    def new_preexisting_varnode(self, vn: int) -> TransformVar:
        if vn in self.piece_map:
            return self.piece_map[vn]
        node = self.data.nodes[vn]
        res = TransformVar(TransformVar.PREEXISTING, vn, node.size_bits, node.size_bits // 8, 0)
        self.piece_map[vn] = res
        return res

    def new_unique(self, size: int) -> TransformVar:
        res = TransformVar(TransformVar.NORMAL_TEMP, -1, size * 8, size, 0)
        self.new_varnodes.append(res)
        return res

    def new_constant(self, size: int, lsb_offset: int, val: int) -> TransformVar:
        shifted_val = (val >> lsb_offset) & ((1 << (size * 8)) - 1)
        res = TransformVar(TransformVar.CONSTANT, -1, size * 8, size, shifted_val)
        self.new_varnodes.append(res)
        return res
        
    def new_piece(self, vn: int, bit_size: int, lsb_offset: int) -> TransformVar:
        res = TransformVar(TransformVar.PIECE, vn, bit_size, bit_size // 8, lsb_offset)
        self.new_varnodes.append(res)
        return res
        
    def new_split(self, vn: int, desc: LaneDescription) -> TransformVar:
        # returns the first piece, but realistically it creates an array of TransformVars
        # In minighidra we can return a list or just one
        pass
        
    def new_op_replace(self, num_params: int, opc: str, replace_op: Any) -> TransformOp:
        res = TransformOp(replace_op, opc, num_params)
        res.special = TransformOp.OP_REPLACEMENT
        self.new_ops.append(res)
        return res

    def new_op(self, num_params: int, opc: str, follow: TransformOp) -> TransformOp:
        res = TransformOp(None, opc, num_params)
        res.follow = follow
        self.new_ops.append(res)
        return res

    def new_preexisting_op(self, num_params: int, opc: str, original_op: Any) -> TransformOp:
        res = TransformOp(original_op, opc, num_params)
        res.special = TransformOp.OP_PREEXISTING
        self.new_ops.append(res)
        return res

    def get_preexisting_varnode(self, vn: int) -> TransformVar:
        if vn in self.piece_map:
            return self.piece_map[vn]
        return self.new_preexisting_varnode(vn)
        
    def get_piece(self, vn: int, bit_size: int, lsb_offset: int) -> TransformVar:
        # Search for existing piece first
        return self.new_piece(vn, bit_size, lsb_offset)

    def op_set_input(self, rop: TransformOp, rvn: TransformVar, slot: int):
        rop.inputs[slot] = rvn

    def op_set_output(self, rop: TransformOp, rvn: TransformVar):
        rop.output = rvn
        rvn.def_op = rop
        
    @staticmethod
    def preexisting_guard(slot: int, rvn: TransformVar) -> bool:
        if slot == 0: return True
        if rvn.type in (TransformVar.PIECE, TransformVar.PIECE_TEMP):
            return False
        return True
        
    def create_ops(self):
        for rop in self.new_ops:
            if rop.special == TransformOp.OP_PREEXISTING:
                rop.replacement = rop.op
                if rop.replacement:
                    self.data.op_set_opcode(rop.replacement, rop.opc)
            else:
                addr = rop.op.addr if rop.op else 0
                new_op = self.data.new_op(len(rop.inputs), addr)
                self.data.op_set_opcode(new_op, rop.opc)
                rop.replacement = new_op
                if rop.special == TransformOp.OP_REPLACEMENT and rop.op:
                    self.data.op_insert_before(new_op, rop.op)

    def create_varnodes(self, input_list: List[TransformVar]):
        for rvn in self.new_varnodes:
            if rvn.type == TransformVar.PREEXISTING:
                rvn.replacement = rvn.vn
            elif rvn.type == TransformVar.CONSTANT:
                rvn.replacement = self.data.new_constant(rvn.byte_size, rvn.val)
            else:
                # Normal or piece temp
                if rvn.def_op and rvn.def_op.replacement:
                    out_vn = self.data.new_unique_out(rvn.byte_size, rvn.def_op.replacement)
                    rvn.replacement = out_vn
                else:
                    rvn.replacement = self.data.new_unique(rvn.byte_size)
                    input_list.append(rvn)
                    
        for rvn in self.piece_map.values():
            if rvn.type == TransformVar.PREEXISTING:
                rvn.replacement = rvn.vn

    def remove_old(self):
        for rop in self.new_ops:
            if rop.special == TransformOp.OP_REPLACEMENT and rop.op:
                self.data.op_destroy(rop.op)

    def transform_input_varnodes(self, input_list: List[TransformVar]):
        for rvn in input_list:
            if rvn.vn >= 0:
                pass # Unlink old input? (Abstracted over in minighidra mostly via just removing old ops)

    def place_inputs(self):
        for rop in self.new_ops:
            if not rop.replacement: continue
            for i, rvn in enumerate(rop.inputs):
                if rvn:
                    self.data.op_set_input(rop.replacement, rvn.replacement, i)
                    
    def apply(self):
        input_list = []
        self.create_ops()
        self.create_varnodes(input_list)
        self.remove_old()
        self.transform_input_varnodes(input_list)
        self.place_inputs()

