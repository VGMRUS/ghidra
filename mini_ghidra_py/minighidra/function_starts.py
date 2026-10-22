"""Conservative x86 function-start signatures inspired by Ghidra's analyzer.

Ghidra's full implementation uses a language-specific PatternDecisionTree, XML
patterns, pre/post-patterns, code-boundary checks and a pseudo-disassembler.
This module ports a small, architecture-limited subset of that workflow: common
x86 prologues are matched only at decoded instruction boundaries and are gated
by preceding control-flow/padding and optional reachable-instruction checks.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class X86StartSignature:
    name: str
    pattern: bytes
    mask: bytes
    bits: int
    min_valid_instructions: int = 0
    label: Optional[str] = None
    require_boundary: bool = True

    def matches(self, data: bytes, offset: int) -> bool:
        if offset < 0 or offset + len(self.pattern) > len(data):
            return False
        return all(
            (data[offset + index] & mask) == (expected & mask)
            for index, (expected, mask) in enumerate(zip(self.pattern, self.mask))
        )


@dataclass(frozen=True)
class FunctionStartCandidate:
    address: int
    pattern: str
    bits: int
    confidence: str
    label: Optional[str] = None


def _signature(
    name: str,
    byte_spec: str,
    bits: int,
    *,
    min_valid_instructions: int = 0,
    label: Optional[str] = None,
    require_boundary: bool = True,
) -> X86StartSignature:
    values = []
    masks = []
    for token in byte_spec.split():
        if token in {"??", "..", "**"}:
            values.append(0)
            masks.append(0)
            continue
        try:
            value = int(token, 16)
        except ValueError as exc:
            raise ValueError(f"Byte de firma no válido: {token}") from exc
        if not 0 <= value <= 0xFF:
            raise ValueError(f"Byte de firma fuera de rango: {token}")
        values.append(value)
        masks.append(0xFF)
    if not values:
        raise ValueError("La firma no puede estar vacía.")
    return X86StartSignature(
        name=name,
        pattern=bytes(values),
        mask=bytes(masks),
        bits=bits,
        min_valid_instructions=min_valid_instructions,
        label=label,
        require_boundary=require_boundary,
    )


# Subset of Ghidra x86 pattern data. The names/byte sequences correspond to
# common entries in x86gcc_patterns.xml, x86-64gcc_patterns.xml and
# x86win_patterns.xml. Pattern-pair rules and the complete XML bit-mask syntax
# are intentionally not claimed to be ported.
_SIGNATURES = (
    # 32-bit GCC/MSVC frame and stack-allocation prologues.
    _signature("x86: push ebp; mov ebp, esp", "55 89 e5", 32),
    _signature("x86: push ebp; mov ebp, esp (alternate encoding)", "55 8b ec", 32),
    _signature("x86: frame + small stack allocation", "55 89 e5 83 ec ??", 32),
    _signature("x86: frame + large stack allocation", "55 89 e5 81 ec ?? ?? ?? ??", 32),
    _signature("x86: alternate frame + stack allocation", "55 8b ec 83 ec ??", 32),
    _signature("x86: hotpatch nop + frame", "8b ff 55 8b ec", 32),
    _signature("x86: push esi; mov esi, ecx", "56 8b f1", 32),
    _signature("x86: save ebx/esi/edi", "53 56 57", 32),
    _signature("x86: save ebx/ebp/esi", "53 55 56", 32),
    _signature("x86: save ebp/edi/esi/ebx", "55 57 56 53", 32),
    _signature("x86: stack allocation after boundary", "83 ec ??", 32, min_valid_instructions=2),
    _signature("x86: large stack allocation after boundary", "81 ec ?? ?? ?? ??", 32, min_valid_instructions=2),
    _signature("x86: get-pc thunk ax", "8b 04 24 c3", 32, label="__i686.get_pc_thunk.ax"),
    _signature("x86: get-pc thunk bx", "8b 1c 24 c3", 32, label="__i686.get_pc_thunk.bx"),
    _signature("x86: get-pc thunk cx", "8b 0c 24 c3", 32, label="__i686.get_pc_thunk.cx"),
    _signature("x86: get-pc thunk dx", "8b 14 24 c3", 32, label="__i686.get_pc_thunk.dx"),
    _signature("x86: get-pc thunk si", "8b 34 24 c3", 32, label="__i686.get_pc_thunk.si"),

    # x86-64 GCC/Windows patterns, including the common CET ENDBR64 prefix.
    _signature("x86-64: frame pointer", "55 48 89 e5", 64),
    _signature("x86-64: frame + small stack allocation", "55 48 89 e5 48 83 ec ??", 64),
    _signature("x86-64: frame + large stack allocation", "55 48 89 e5 48 81 ec ?? ?? ?? ??", 64),
    _signature("x86-64: push rbx; mov rbx, rdi", "53 48 89 fb", 64),
    _signature("x86-64: push rbp; mov rbp, rdi", "55 48 89 fd", 64),
    _signature("x86-64: push rbx; allocate stack", "53 48 83 ec ??", 64),
    _signature("x86-64: push rbx; allocate large stack", "53 48 81 ec ?? ?? ?? ??", 64),
    _signature("x86-64: save r15/r14/r13", "41 57 41 56 41 55", 64),
    _signature("x86-64: save r14/r13", "41 56 41 55", 64),
    _signature("x86-64: save r13/r12", "41 55 41 54", 64),
    _signature("x86-64: frame + save rbx", "55 48 89 e5 53", 64),
    _signature("x86-64: save rbp/rbx + stack", "55 53 48 83 ec ??", 64),
    _signature("x86-64: ENDBR64 + frame", "f3 0f 1e fa 55 48 89 e5", 64),
    _signature("x86-64: ENDBR64 + rbx prologue", "f3 0f 1e fa 53 48 89 fb", 64),
    # Ghidra's stack-allocation-only signatures require longer valid-code runs;
    # mirror that precaution to suppress mid-function matches.
    _signature("x86-64: stack allocation + valid body", "48 83 ec ??", 64, min_valid_instructions=10),
    _signature("x86-64: large stack allocation + valid body", "48 81 ec ?? ?? ?? ??", 64, min_valid_instructions=10),
    _signature("x86: ENDBR32 + frame", "f3 0f 1e fb 55 89 e5", 32),
    _signature("x86: ENDBR32 + alternate frame", "f3 0f 1e fb 55 8b ec", 32),
)


def find_function_start_candidates(
    image,
    instructions: dict[int, object],
    *,
    known_starts: Iterable[int] = (),
) -> list[FunctionStartCandidate]:
    """Find likely, as-yet-unknown x86 function starts in executable sections.

    Ghidra's ``FunctionStartAnalyzer`` first filters by executable memory and
    code boundaries, then validates pattern-specific preconditions. Here the
    already-decoded instruction map provides the code-boundary check, and the
    small signature table replaces Ghidra's XML decision tree.
    """
    known = set(known_starts)
    signatures = [signature for signature in _SIGNATURES if signature.bits == image.bits]
    by_first_byte = defaultdict(list)
    for signature in signatures:
        by_first_byte[signature.pattern[0]].append(signature)

    found: dict[int, FunctionStartCandidate] = {}
    instruction_addresses = sorted(instructions)
    previous_by_end = {instruction.next_address: instruction for instruction in instructions.values()}
    for section in image.executable_sections:
        first_index = bisect_left(instruction_addresses, section.start)
        last_index = bisect_left(instruction_addresses, section.data_end)
        for address in instruction_addresses[first_index:last_index]:
            offset = address - section.start
            first = section.data[offset]
            for signature in by_first_byte.get(first, ()):
                if address in known or not signature.matches(section.data, offset):
                    continue
                if not _signature_is_instruction_aligned(address, signature, instructions):
                    continue
                if signature.require_boundary and not _has_code_boundary(section, address, previous_by_end):
                    continue
                if signature.min_valid_instructions and _reachable_instruction_count(
                    address,
                    instructions,
                    known,
                    minimum=signature.min_valid_instructions,
                ) < signature.min_valid_instructions:
                    continue

                confidence = "alta" if signature.min_valid_instructions == 0 else "media"
                candidate = FunctionStartCandidate(
                    address=address,
                    pattern=signature.name,
                    bits=image.bits,
                    confidence=confidence,
                    label=signature.label,
                )
                current = found.get(address)
                if current is None or (candidate.label and not current.label):
                    found[address] = candidate
    return [found[address] for address in sorted(found)]


def _signature_is_instruction_aligned(address: int, signature: X86StartSignature, instructions) -> bool:
    """Require the whole byte signature to cover complete decoded instructions."""
    cursor = address
    consumed = 0
    while consumed < len(signature.pattern):
        instruction = instructions.get(cursor)
        if instruction is None or consumed + instruction.size > len(signature.pattern):
            return False
        consumed += instruction.size
        cursor = instruction.next_address
    return consumed == len(signature.pattern)


def _has_code_boundary(section, address: int, previous_by_end: dict[int, object]) -> bool:
    if address == section.start:
        return True
    previous = previous_by_end.get(address)
    if previous is not None:
        mnemonic = previous.mnemonic.lower()
        if previous.is_return or (previous.is_jump and not previous.is_conditional_jump):
            return True
        if mnemonic in {"nop", "int3", "ud2", "hlt"}:
            return True
        return False

    offset = address - section.start
    # Ghidra's x86 prepatterns recognize code padding and trap-fill bytes.
    return offset > 0 and section.data[offset - 1] in {0x90, 0xCC, 0xC3}


def _reachable_instruction_count(
    start: int,
    instructions: dict[int, object],
    known_starts: set[int],
    *,
    minimum: int,
) -> int:
    """Count valid reachable instructions, following branches but not calls.

    This approximates the valid-code prerequisite used by Ghidra's pseudo
    disassembler. It is bounded so malformed control flow cannot cause an
    unbounded scan.
    """
    pending = deque([start])
    visited = set()
    limit = max(minimum * 4, 64)
    while pending and len(visited) < limit:
        address = pending.popleft()
        while address not in visited and len(visited) < limit:
            if address != start and address in known_starts:
                break
            instruction = instructions.get(address)
            if instruction is None:
                break
            visited.add(address)
            if len(visited) >= minimum:
                return len(visited)
            following = instruction.next_address
            if instruction.is_return:
                break
            if instruction.is_jump:
                target = instruction.branch_target
                if target is not None and target in instructions:
                    pending.append(target)
                if instruction.is_conditional_jump:
                    address = following
                    continue
                break
            address = following
    return len(visited)
