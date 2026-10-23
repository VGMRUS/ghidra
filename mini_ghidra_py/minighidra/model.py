"""Data structures shared by the loader and static analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .typesys import CType, StorageLocation


@dataclass(frozen=True)
class Section:
    name: str
    start: int
    virtual_size: int
    file_offset: Optional[int]
    data: bytes = field(repr=False)
    executable: bool = False
    readable: bool = True
    writable: bool = False

    @property
    def end(self) -> int:
        """End of the mapped section, excluding the address at ``end``."""
        return self.start + max(self.virtual_size, len(self.data))

    @property
    def data_end(self) -> int:
        """End of file-backed bytes in the section."""
        return self.start + len(self.data)

    def contains(self, address: int, *, file_backed: bool = False) -> bool:
        end = self.data_end if file_backed else self.end
        return self.start <= address < end


@dataclass(frozen=True)
class Symbol:
    name: str
    address: int
    kind: str = "symbol"
    size: int = 0
    imported: bool = False
    library: Optional[str] = None


@dataclass(frozen=True)
class StringEntry:
    address: int
    text: str
    encoding: str
    section: str


@dataclass(frozen=True)
class AddressTable:
    """A conservatively recognized run of absolute code pointers."""

    address: int
    section: str
    entry_size: int
    targets: tuple[int, ...]
    jump_sources: tuple[int, ...] = ()

    @property
    def end(self) -> int:
        return self.address + self.entry_size * len(self.targets)


@dataclass(frozen=True)
class Instruction:
    address: int
    size: int
    raw: bytes = field(repr=False)
    mnemonic: str
    op_str: str
    groups: tuple[str, ...] = ()
    branch_target: Optional[int] = None
    memory_references: tuple[int, ...] = ()
    indexed_memory_references: tuple[int, ...] = ()
    scalar_references: tuple[int, ...] = ()
    computed_targets: tuple[int, ...] = ()

    @property
    def next_address(self) -> int:
        return self.address + self.size

    @property
    def is_call(self) -> bool:
        return "call" in self.groups

    @property
    def is_jump(self) -> bool:
        return "jump" in self.groups

    @property
    def is_return(self) -> bool:
        return "ret" in self.groups

    @property
    def is_trap(self) -> bool:
        return self.mnemonic.lower() in {"int3", "ud2", "hlt"}

    @property
    def is_conditional_jump(self) -> bool:
        return self.is_jump and self.mnemonic.lower() not in {"jmp", "ljmp"}


@dataclass(frozen=True)
class Xref:
    source: int
    target: int
    kind: str


@dataclass(frozen=True)
class BasicBlock:
    start: int
    instructions: tuple[Instruction, ...]
    successors: tuple[int, ...] = ()

    @property
    def end(self) -> int:
        if not self.instructions:
            return self.start
        return self.instructions[-1].next_address


@dataclass(frozen=True)
class PEUnwindOperation:
    opcode: int
    name: str
    code_offset: int
    op_info: int
    operand: Optional[int] = None
    register: Optional[str] = None


@dataclass(frozen=True)
class PEUnwindInfo:
    unwind_rva: int
    version: int
    flags: int
    prolog_size: int
    frame_register: Optional[str]
    frame_offset: int
    code_slot_count: int
    operations: tuple[PEUnwindOperation, ...] = ()
    stack_allocation_bytes: Optional[int] = None
    saved_registers: tuple[str, ...] = ()
    handler_rva: Optional[int] = None


@dataclass(frozen=True)
class FunctionParameter:
    """A formal parameter, optionally annotated as an inferred candidate.

    Candidate confidence/evidence refer to presence and ABI location; the C type
    can remain a conservative fallback.
    """

    name: str
    type_name: str
    data_type: CType | None = None
    locations: tuple[StorageLocation, ...] = ()
    implicit: bool = False
    fixed: bool = True
    confidence: float | None = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class FunctionSignature:
    """Resolved prototype or unlocked candidate.

    Confidence scores are heuristic rankings, not calibrated probabilities.
    """

    return_type: str
    parameters: tuple[FunctionParameter, ...]
    calling_convention: str
    variadic: bool = False
    return_data_type: CType | None = None
    return_locations: tuple[StorageLocation, ...] = ()
    hidden_return: bool = False
    hidden_return_location: StorageLocation | None = None
    varargs_sse_count_register: str | None = None
    source: str = "inferred"
    locked: bool = False
    inference_confidence: float | None = None
    inference_evidence: tuple[str, ...] = ()


@dataclass
class Function:
    address: int
    name: str
    instructions: list[Instruction] = field(default_factory=list)
    blocks: list[BasicBlock] = field(default_factory=list)
    source: str = "heuristic"
    no_return: bool = False
    is_thunk: bool = False
    thunk_target: Optional[int] = None
    thunk_target_name: Optional[str] = None
    stack_purge_size: Optional[int] = None
    pe_unwind_info: Optional[PEUnwindInfo] = None
    calling_convention: Optional[str] = None
    signature: Optional[FunctionSignature] = None

    @property
    def start(self) -> int:
        return self.address

    @property
    def end(self) -> int:
        if not self.instructions:
            return self.address
        return max(insn.next_address for insn in self.instructions)

    @property
    def size(self) -> int:
        return max(0, self.end - self.address)
