from typing import Any, List
from .paramid import PcodeLikeGraph

class BitRange:
    def __init__(self, size_bytes: int, offset: int = 0):
        self.byte_size = size_bytes
        self.num_bits = size_bytes * 8
        self.least_sig_bit = offset

    def shift(self, sa: int):
        self.least_sig_bit += sa

class BitFieldNodeState:
    def __init__(self, used: BitRange, vn: int, least_sig: int, num_bits: int):
        self.bits_used = used
        self.bits_field = BitRange(num_bits // 8, least_sig)
        self.node = vn
        self.field = None
        self.orig_least_sig_bit = least_sig
        self.is_sign_extended = False


class BitFieldTransform:
    def __init__(self, func: PcodeLikeGraph, dt: Any, off: int):
        self.func = func
        self.parent_struct = dt
        self.work_list: List[BitFieldNodeState] = []
        self.initial_offset = off
        self.container_size = 0
        self.is_big_endian = False

    def build_partial_type(self) -> Any:
        return None

class BitFieldInsertTransform(BitFieldTransform):
    
    ALLOWED_FINAL_WRITES = (
        "COPY", "INT_EQUAL", "INT_NOTEQUAL", "INT_SLESS", "INT_SLESSEQUAL",
        "INT_LESS", "INT_LESSEQUAL", "INT_ZEXT", "INT_SEXT", "INT_ADD", "INT_CARRY",
        "INT_SCARRY", "INT_XOR", "INT_AND", "INT_OR", "INT_LEFT", "INT_RIGHT",
        "INT_SRIGHT", "INT_MULT", "BOOL_NEGATE", "BOOL_XOR", "BOOL_AND", "BOOL_OR",
        "FLOAT_EQUAL", "FLOAT_NOTEQUAL", "FLOAT_LESS", "FLOAT_LESSEQUAL", "FLOAT_NAN",
        "SUBPIECE"
    )

    def __init__(self, func: PcodeLikeGraph, op: Any, dt: Any, off: int):
        super().__init__(func, dt, off)
        self.final_write_op = op
        self.original_value = -1
        self.mapped_vn = -1
        self.insert_list = []

    def do_trace(self) -> bool:
        if not self.work_list: return False
        # Stub tracing implementation mapping to C++ parity
        return False
        
    def apply(self):
        pass


class BitFieldPullTransform(BitFieldTransform):
    def __init__(self, func: PcodeLikeGraph, vn: int, dt: Any, off: int):
        super().__init__(func, dt, off)
        self.root_vn = vn

    def do_trace(self) -> bool:
        if not self.work_list: return False
        return False
        
    def apply(self):
        pass

