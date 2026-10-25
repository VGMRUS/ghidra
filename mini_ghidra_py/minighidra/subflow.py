"""Small data-flow transform helpers used by p-code rules.

The original file contained several partially indented, non-executable drafts.
The aggregate datatype splitter now lives in :mod:`minighidra.datatype_split`;
this module retains the helper names imported by the existing rule catalog and
provides conservative analysis shells for the unrelated lane/subvariable
rewrites until their transform semantics are ported.
"""

from __future__ import annotations

from dataclasses import dataclass

from .paramid import PcodeLikeGraph, PcodeOp
from .transform import TransformManager


@dataclass(slots=True)
class ReplaceVarnode:
    vn: int
    mask: int
    val: int = 0
    replacement: int | None = None
    def_op: ReplaceOp | None = None


@dataclass(slots=True)
class ReplaceOp:
    op: PcodeOp | None
    opcode: str
    numparams: int
    output: ReplaceVarnode | None = None
    inputs: list[ReplaceVarnode | None] | None = None
    replacement: PcodeOp | None = None

    def __post_init__(self) -> None:
        if self.inputs is None:
            self.inputs = [None] * self.numparams


@dataclass(slots=True)
class PatchRecord:
    COPY_PATCH = 0
    COMPARE_PATCH = 1
    PARAMETER_PATCH = 2
    EXTENSION_PATCH = 3
    PUSH_PATCH = 4
    INT2FLOAT_PATCH = 5

    type: int
    patch_op: PcodeOp
    in1: ReplaceVarnode
    in2: ReplaceVarnode | None = None
    slot: int = 0


class SubvariableFlow:
    """Conservative placeholder for bit-range propagation transforms.

    Rules can still ask whether a flow was transformed; unsupported or
    ambiguous graphs are rejected without modifying their p-code.
    """

    def __init__(
        self,
        data: PcodeLikeGraph,
        root_vn: int,
        mask: int,
        aggressive: bool,
        sext: bool,
        big: bool,
    ) -> None:
        self.data = data
        self.root_vn = root_vn
        self.mask = mask
        self.aggressive = aggressive
        self.sextrestrictions = sext
        self.big = big
        self.flowsize = max(1, (mask.bit_length() + 7) // 8)
        self.worklist: list[ReplaceVarnode] = []
        self.patchlist: list[PatchRecord] = []
        self.oplist: list[ReplaceOp] = []

    def do_trace(self) -> bool:
        return False

    def do_replacement(self) -> None:
        return None


class LaneDescription:
    def __init__(self, whole_size: int, lane_size: int) -> None:
        if whole_size <= 0 or lane_size <= 0:
            raise ValueError("El tamaño de vector y lane debe ser positivo")
        self.whole_size = whole_size
        self.lane_size = lane_size
        self.lane_sizes = [
            min(lane_size, whole_size - offset)
            for offset in range(0, whole_size, lane_size)
        ]
        self.lane_positions = list(range(0, whole_size, lane_size))

    def get_num_lanes(self) -> int:
        return len(self.lane_sizes)

    def get_size(self, index: int) -> int:
        return self.lane_sizes[index]

    def get_position(self, index: int) -> int:
        return self.lane_positions[index]

    def get_boundary(self, byte_pos: int) -> int:
        try:
            return self.lane_positions.index(byte_pos)
        except ValueError:
            return -1


class LaneDivide:
    """Conservative SIMD lane split analysis; never mutates unsupported graphs."""

    def __init__(
        self,
        data: PcodeLikeGraph,
        root: int,
        desc: LaneDescription,
        allow_downcast: bool,
    ) -> None:
        self.data = data
        self.root = root
        self.description = desc
        self.allow_subpiece_terminator = allow_downcast
        self.terminators: list[PcodeOp] = []

    def do_trace(self) -> bool:
        return False

    def apply(self) -> None:
        return None


class SplitFlow:
    """Conservative interface for splitting artificial PIECE/SUBPIECE flows."""

    def __init__(self, data: PcodeLikeGraph, root: int, low_size: int) -> None:
        self.data = data
        self.root = root
        self.low_size = low_size

    def do_trace(self) -> bool:
        return False

    def apply(self) -> None:
        return None


class SubfloatFlow(TransformManager):
    """Conservative interface for narrowing floating-point data-flow."""

    def __init__(self, data: PcodeLikeGraph, root: int, prec: int) -> None:
        super().__init__(data)
        self.precision = prec
        self.root = root
        self.worklist = []

    def do_trace(self) -> bool:
        return False
