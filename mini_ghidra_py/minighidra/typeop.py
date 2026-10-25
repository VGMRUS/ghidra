from __future__ import annotations
from typing import Optional
from .typesys import CType, TypeKind
from .paramid import PcodeOp

class TypeOp:
    """Base logic for pushing and popping data types across a P-code operation."""
    
    @staticmethod
    def push(op: PcodeOp, in_types: list[Optional[CType]]) -> Optional[CType]:
        """Infer output type from input types (forward propagation)."""
        return None
        
    @staticmethod
    def pop(op: PcodeOp, out_type: Optional[CType], in_types: list[Optional[CType]], slot: int) -> Optional[CType]:
        """Infer input type at 'slot' from output type and other inputs (backward propagation)."""
        return None

    @staticmethod
    def float_sign_manipulation(op: PcodeOp, data: PcodeLikeGraph) -> str | None:
        """Returns FLOAT_ABS or FLOAT_NEG if the operation manipulates float sign bit, else None."""
        opc = op.opcode
        if opc == "INT_AND" and len(op.inputs) == 2:
            in1 = op.inputs[1]
            if in1 >= 0:
                node = data.nodes[in1]
                if node.kind == "const":
                    val = int(node.name, 16) if node.name else 0
                    mask = (1 << node.size_bits) - 1
                    if mask >> 1 == val:
                        return "FLOAT_ABS"
        elif opc == "INT_XOR" and len(op.inputs) == 2:
            in1 = op.inputs[1]
            if in1 >= 0:
                node = data.nodes[in1]
                if node.kind == "const":
                    val = int(node.name, 16) if node.name else 0
                    mask = (1 << node.size_bits) - 1
                    test_val = mask ^ (mask >> 1)
                    if test_val == val:
                        return "FLOAT_NEG"
        return None

class TypeOpCopy(TypeOp):
    @staticmethod
    def push(op: PcodeOp, in_types: list[Optional[CType]]) -> Optional[CType]:
        return in_types[0]
        
    @staticmethod
    def pop(op: PcodeOp, out_type: Optional[CType], in_types: list[Optional[CType]], slot: int) -> Optional[CType]:
        return out_type

class TypeOpMultiEqual(TypeOp):
    @staticmethod
    def push(op: PcodeOp, in_types: list[Optional[CType]]) -> Optional[CType]:
        for t in in_types:
            if t is not None:
                return t
        return None
        
    @staticmethod
    def pop(op: PcodeOp, out_type: Optional[CType], in_types: list[Optional[CType]], slot: int) -> Optional[CType]:
        return out_type

class TypeOpIntAdd(TypeOp):
    @staticmethod
    def push(op: PcodeOp, in_types: list[Optional[CType]]) -> Optional[CType]:
        # If one side is a pointer, the result is a pointer
        for t in in_types[:2]:
            if t is not None and t.kind in (TypeKind.POINTER, TypeKind.ARRAY):
                return t
        return None
        
    @staticmethod
    def pop(op: PcodeOp, out_type: Optional[CType], in_types: list[Optional[CType]], slot: int) -> Optional[CType]:
        if out_type is not None and out_type.kind == TypeKind.POINTER:
            # If the other input is an integer, this slot must be the pointer
            other_slot = 1 - slot
            if other_slot < len(in_types):
                other_t = in_types[other_slot]
                if other_t is not None and other_t.kind == TypeKind.INTEGER:
                    return out_type
        return None

class TypeOpLoad(TypeOp):
    @staticmethod
    def push(op: PcodeOp, in_types: list[Optional[CType]]) -> Optional[CType]:
        if len(in_types) > 1 and in_types[1] is not None:
            if in_types[1].kind == TypeKind.POINTER:
                return in_types[1].target
        return None
        
    @staticmethod
    def pop(op: PcodeOp, out_type: Optional[CType], in_types: list[Optional[CType]], slot: int) -> Optional[CType]:
        if slot == 1 and out_type is not None:
            # Reconstruct pointer type? We can't easily instantiate here without TypeDatabase.
            pass
        return None

class TypeOpStore(TypeOp):
    @staticmethod
    def pop(op: PcodeOp, out_type: Optional[CType], in_types: list[Optional[CType]], slot: int) -> Optional[CType]:
        if slot == 2 and len(in_types) > 1 and in_types[1] is not None:
            if in_types[1].kind == TypeKind.POINTER:
                return in_types[1].target
        return None


class TypeOpMath(TypeOp):
    @staticmethod
    def push(op: PcodeOp, in_types: list[Optional[CType]]) -> Optional[CType]:
        # Return the first known type, favoring non-boolean
        for t in in_types:
            if t is not None:
                return t
        return None

    @staticmethod
    def pop(op: PcodeOp, out_type: Optional[CType], in_types: list[Optional[CType]], slot: int) -> Optional[CType]:
        if out_type is not None:
            return out_type
        # If output unknown, try the other input
        other = 1 - slot
        if other >= 0 and other < len(in_types) and in_types[other] is not None:
            return in_types[other]
        return None

class TypeOpCompare(TypeOp):
    @staticmethod
    def push(op: PcodeOp, in_types: list[Optional[CType]]) -> Optional[CType]:
        # Typically returns a boolean
        # We don't have TypeDatabase to create a BOOL CType on the fly easily,
        # but we can return None or construct a dummy if needed.
        return None
        
    @staticmethod
    def pop(op: PcodeOp, out_type: Optional[CType], in_types: list[Optional[CType]], slot: int) -> Optional[CType]:
        other_slot = 1 - slot
        if other_slot < len(in_types):
            return in_types[other_slot]
        return None

TYPE_OPS = {
    "SUBPIECE": TypeOpSubpiece(),
    "PIECE": TypeOpPiece(),
    "PTRADD": TypeOpPtradd(),
    "PTRSUB": TypeOpPtrsub(),
    "INDIRECT": TypeOpIndirect(),
    "CAST": TypeOpCast(),
    "CALL": TypeOpCall(),
    "CALLIND": TypeOpCall(),
    "CALLOTHER": TypeOpCall(),
    "RETURN": TypeOpReturn(),
    "BRANCH": TypeOpBranch(),
    "CBRANCH": TypeOpCbranch(),
    "BRANCHIND": TypeOpBranchind(),
    "INT_ZEXT": TypeOpMath(),
    "INT_SEXT": TypeOpMath(),
    "INT_2COMP": TypeOpMath(),
    "INT_NEGATE": TypeOpMath(),
    "COPY": TypeOpCopy(),
    "MULTIEQUAL": TypeOpMultiEqual(),
    "INT_ADD": TypeOpIntAdd(),
    "INT_SUB": TypeOpIntAdd(),

    "INT_MULT": TypeOpMath(),
    "INT_DIV": TypeOpMath(),
    "INT_SDIV": TypeOpMath(),
    "INT_REM": TypeOpMath(),
    "INT_SREM": TypeOpMath(),
    "INT_AND": TypeOpMath(),
    "INT_OR": TypeOpMath(),
    "INT_XOR": TypeOpMath(),
    "INT_LEFT": TypeOpMath(),
    "INT_RIGHT": TypeOpMath(),
    "INT_SRIGHT": TypeOpMath(),

    "LOAD": TypeOpLoad(),
    "STORE": TypeOpStore(),
    "INT_EQUAL": TypeOpCompare(),
    "INT_NOTEQUAL": TypeOpCompare(),
    "INT_LESS": TypeOpCompare(),
    "INT_SLESS": TypeOpCompare(),
    "INT_LESSEQUAL": TypeOpCompare(),
    "INT_SLESSEQUAL": TypeOpCompare(),
}

class TypeOpIntSub(TypeOpIntAdd):
    pass

class TypeOpIntXor(TypeOpMath):
    pass

class TypeOpPiece(TypeOp):
    pass

class TypeOpSubpiece(TypeOp):
    pass

class TypeOpPtradd(TypeOp):
    pass

class TypeOpPtrsub(TypeOp):
    pass

class TypeOpIndirect(TypeOp):
    pass

class TypeOpCall(TypeOp):
    pass

class TypeOpReturn(TypeOp):
    pass

class TypeOpCast(TypeOp):
    pass

class TypeOpCbranch(TypeOp):
    pass

class TypeOpBranch(TypeOp):
    pass

class TypeOpBranchind(TypeOp):
    pass

