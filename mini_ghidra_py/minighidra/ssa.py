"""P-code SSA construction inspired by Ghidra's ``Heritage`` pass.

The pass consumes exact per-instruction SLEIGH p-code, builds a p-code-level
CFG (including internal p-code branches), computes dominators and dominance
frontiers, pruned phi placement, and performs dominance-tree renaming. Register
and temporary storage are split into byte cells so overlapping x86 varnodes
share definitions. RAM accesses use a conservative, whole-memory SSA token;
this preserves alias dependencies across LOAD/STORE, direct RAM varnodes, calls,
and unknown side-effecting userops.

This is a Python SSA/heritage core, not a translation of every Ghidra
``Heritage`` guard, stack-join, call-spec, or type-recovery subsystem.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

from .model import BasicBlock, Function, Instruction
from .sleigh import SleighInstruction, SleighPcodeOp, SleighVarnode

_MEMORY = "__memory__"
_KNOWN_PURE_USEROPS = frozenset({0x11, 0x12, 0xE7, 0xE8, 0xE9, 0xEA})
_GPR_FAMILIES = {
    alias: family
    for family, aliases in {
        "rax": ("rax", "eax", "ax", "al", "ah"),
        "rbx": ("rbx", "ebx", "bx", "bl", "bh"),
        "rcx": ("rcx", "ecx", "cx", "cl", "ch"),
        "rdx": ("rdx", "edx", "dx", "dl", "dh"),
        "rsi": ("rsi", "esi", "si", "sil"),
        "rdi": ("rdi", "edi", "di", "dil"),
        "rbp": ("rbp", "ebp", "bp", "bpl"),
        "rsp": ("rsp", "esp", "sp", "spl"),
        "rip": ("rip", "eip", "ip"),
    }.items()
    for alias in aliases
}
_FLAG_REGISTERS = frozenset(
    {"cf", "pf", "af", "zf", "sf", "tf", "if", "df", "of", "eflags", "rflags"}
)


def _register_family(display: str) -> str:
    """Canonicalize SLEIGH register aliases for ABI call-clobber handling."""
    name = display.strip().lower()
    if name in _GPR_FAMILIES:
        return _GPR_FAMILIES[name]
    if name in _FLAG_REGISTERS:
        return "__flags__"
    extended = re.fullmatch(r"r(8|9|1[0-5])(?:[dwb])?", name)
    if extended:
        return f"r{extended.group(1)}"
    vector = re.fullmatch(r"(?:xmm|ymm|zmm)(\d+)", name)
    if vector:
        return f"xmm{vector.group(1)}"
    return name


@dataclass(frozen=True, slots=True, order=True)
class SSALocation:
    """One byte of storage, or the single whole-memory version token."""

    space: str
    offset: int
    scope_address: int | None = None

    @property
    def label(self) -> str:
        if self.space == _MEMORY:
            return "memory"
        scope = f"@{self.scope_address:X}" if self.scope_address is not None else ""
        return f"{self.space}{scope}[{self.offset:X}]"


MEMORY_LOCATION = SSALocation(_MEMORY, 0)


@dataclass(frozen=True, slots=True)
class SSAOperand:
    """An operation input after heritage has resolved its reaching versions."""

    kind: str
    size_bits: int
    values: tuple[int, ...] = ()
    constant: int | None = None
    address: int | None = None
    memory_version: int | None = None


@dataclass(frozen=True, slots=True)
class SSAValue:
    """A single SSA definition, input, phi, or byte-slice value."""

    identifier: int
    kind: str
    name: str
    size_bits: int
    block_id: int | None = None
    instruction_address: int | None = None
    operation_sequence: int | None = None
    opcode: str | None = None
    storage: SSALocation | None = None
    source: int | None = None
    offset_bits: int = 0
    incoming: tuple[tuple[int | None, int], ...] = ()


@dataclass(frozen=True, slots=True)
class SSAPhi:
    """A heritage phi with an explicit value for each incoming CFG edge."""

    block_id: int
    location: SSALocation
    output_value: int
    incoming: tuple[tuple[int | None, int], ...]


@dataclass(frozen=True, slots=True)
class SSAOperation:
    """A SLEIGH operation with its SSA-resolved inputs and effects."""

    index: int
    block_id: int
    instruction_address: int
    instruction_mnemonic: str
    sequence: int
    opcode: str
    inputs: tuple[SSAOperand, ...]
    output_value: int | None = None
    output_parts: tuple[int, ...] = ()
    memory_inputs: tuple[int, ...] = ()
    memory_output: int | None = None
    memory_write_address: SSAOperand | None = None
    memory_write_value: SSAOperand | None = None
    memory_write_bits: int | None = None
    memory_clobber: bool = False
    call_effect_values: tuple[int, ...] = ()
    pcode: SleighPcodeOp | None = None


@dataclass(frozen=True, slots=True)
class SSAReturnSite:
    """Candidate ABI return-register versions reaching one p-code ``RETURN``."""

    block_id: int
    instruction_address: int
    operation_sequence: int
    register_values: tuple[tuple[str, SSAOperand], ...]


@dataclass(frozen=True, slots=True)
class SSAControlBlock:
    """A basic block in the p-code CFG, split at internal branches."""

    identifier: int
    machine_block_start: int
    instruction_address: int
    instruction_mnemonic: str
    operation_start: int
    operation_end: int
    predecessors: tuple[int, ...]
    successors: tuple[int, ...]
    immediate_dominator: int | None
    dominance_frontier: tuple[int, ...]
    reachable: bool
    phi_outputs: tuple[int, ...] = ()
    operations: tuple[SSAOperation, ...] = ()


@dataclass(frozen=True, slots=True)
class SSAFunction:
    """Immutable SSA/heritage result for one recovered function."""

    address: int
    blocks: tuple[SSAControlBlock, ...]
    values: tuple[SSAValue, ...]
    phis: tuple[SSAPhi, ...]
    operations: tuple[SSAOperation, ...]
    entry_blocks: tuple[int, ...]
    live_in: tuple[tuple[int, tuple[SSALocation, ...]], ...]
    live_out: tuple[tuple[int, tuple[SSALocation, ...]], ...]
    diagnostics: tuple[str, ...] = ()
    valid: bool = True
    complete: bool = True
    return_sites: tuple[SSAReturnSite, ...] = ()

    @property
    def phi_count(self) -> int:
        return len(self.phis)

    @property
    def memory_phi_count(self) -> int:
        return sum(phi.location.space == _MEMORY for phi in self.phis)

    @property
    def block_by_id(self) -> dict[int, SSAControlBlock]:
        return {block.identifier: block for block in self.blocks}

    @property
    def value_by_id(self) -> dict[int, SSAValue]:
        return {value.identifier: value for value in self.values}

    @property
    def phi_by_block(self) -> dict[int, tuple[SSAPhi, ...]]:
        grouped: dict[int, list[SSAPhi]] = defaultdict(list)
        for phi in self.phis:
            grouped[phi.block_id].append(phi)
        return {block: tuple(phis) for block, phis in grouped.items()}

    def value(self, identifier: int) -> SSAValue:
        """Look up a value id, raising ``KeyError`` for an invalid reference."""
        return self.value_by_id[identifier]


@dataclass(slots=True)
class _PcodeBlock:
    identifier: int
    machine_block: BasicBlock
    instruction: Instruction | None
    sleigh: SleighInstruction | None
    operation_start: int
    operation_end: int
    operations: tuple[SleighPcodeOp, ...]
    successors: set[int]
    predecessors: set[int]
    reachable: bool = False

    @property
    def instruction_address(self) -> int:
        if self.instruction is not None:
            return self.instruction.address
        return self.machine_block.start


class SSAHeritage:
    """Construct a pruned SSA form from a function's real SLEIGH translations.

    The algorithm follows Ghidra's heritage architecture: collect storage
    definitions/uses, compute a dominator tree and dominance frontiers, place
    only live phis at iterated frontiers, then rename along the dominator tree.
    Internal ``BRANCH``/``CBRANCH`` p-code is split into its own CFG nodes so
    operations such as x86 bit scans and locked compare/exchanges have accurate
    intra-instruction SSA joins.
    """

    def __init__(
        self,
        function: Function,
        translations: Iterable[SleighInstruction],
        *,
        caller_saved_registers: Iterable[str] = (),
        return_registers: Iterable[str] = (),
        no_return_calls: Iterable[int] = (),
        registers: Mapping[str, SleighVarnode] | None = None,
    ) -> None:
        self.function = function
        self.translations = {item.address: item for item in translations}
        self._abi_registers = {
            name.lower(): varnode for name, varnode in (registers or {}).items()
        }
        self._caller_saved_families = {
            _register_family(register) for register in caller_saved_registers
        } | {"__flags__"}
        self._return_register_names = tuple(
            dict.fromkeys(register.lower() for register in return_registers)
        )
        self._return_register_order = tuple(
            dict.fromkeys(
                _register_family(register) for register in self._return_register_names
            )
        )
        self._return_families = set(self._return_register_order)
        self._call_clobber_families = (
            self._caller_saved_families | self._return_families
        )
        self._no_return_calls = frozenset(no_return_calls)
        self._call_register_families: dict[SSALocation, str] = {}
        self._call_stack_locations: dict[int, set[SSALocation]] = defaultdict(set)
        self._call_entry_values: dict[int, dict[SSALocation, int]] = defaultdict(dict)
        self._collect_call_registers()
        self.diagnostics: list[str] = []
        self.complete = True
        self._blocks: list[_PcodeBlock] = []
        self._block_by_id: dict[int, _PcodeBlock] = {}
        self._instruction_entry: dict[int, int] = {}
        self._machine_entry: dict[int, int] = {}
        self._op_to_block: dict[tuple[int, int], int] = {}
        self._instruction_next: dict[int, Instruction | None] = {}
        self._entry_blocks: tuple[int, ...] = ()
        self._active_blocks: set[int] = set()
        self._virtual_root = -1
        self._dom_parent: dict[int, int] = {}
        self._dom_children: dict[int, list[int]] = defaultdict(list)
        self._dom_frontier: dict[int, set[int]] = defaultdict(set)
        self._rpo_index: dict[int, int] = {}
        self._predecessors: dict[int, set[int]] = defaultdict(set)
        self._successors: dict[int, set[int]] = defaultdict(set)
        self._uses: dict[int, set[SSALocation]] = defaultdict(set)
        self._defs: dict[int, set[SSALocation]] = defaultdict(set)
        self._live_in: dict[int, set[SSALocation]] = defaultdict(set)
        self._live_out: dict[int, set[SSALocation]] = defaultdict(set)
        self._definitions: dict[SSALocation, set[int]] = defaultdict(set)
        self._phi_locations: dict[int, set[SSALocation]] = defaultdict(set)
        self._values: list[SSAValue] = []
        self._input_values: dict[SSALocation, int] = {}
        self._phi_outputs: dict[tuple[int, SSALocation], int] = {}
        self._phi_incoming: dict[tuple[int, SSALocation], dict[int | None, int]] = (
            defaultdict(dict)
        )
        self._operation_results: dict[int, list[SSAOperation]] = defaultdict(list)
        self._operation_list: list[SSAOperation] = []
        self._return_sites: list[SSAReturnSite] = []
        self._stacks: dict[SSALocation, list[int]] = defaultdict(list)

    def build(self) -> SSAFunction:
        """Build and validate the p-code SSA graph."""
        self._build_pcode_cfg()
        if not self._blocks:
            return SSAFunction(
                self.function.address,
                (),
                (),
                (),
                (),
                (),
                (),
                (),
                tuple(self.diagnostics),
                valid=True,
                complete=self.complete,
            )
        self._compute_dominators()
        self._collect_use_def()
        self._compute_liveness()
        self._place_phis()
        self._rename()
        return self._freeze_and_validate()

    def _collect_call_registers(self) -> None:
        for name in self._return_register_names:
            varnode = self._abi_registers.get(name)
            if varnode is None or varnode.space != "register":
                continue
            family = _register_family(name)
            for location in self._locations_for_varnode(varnode, self.function.address):
                self._call_register_families[location] = family

        instructions = {
            instruction.address: instruction
            for instruction in self.function.instructions
        }
        call_addresses = {
            address
            for address, translation in self.translations.items()
            if (
                (instruction := instructions.get(address)) is not None
                and instruction.is_call
            )
            or any(
                operation.opcode in {"CALL", "CALLIND"} for operation in translation.ops
            )
        }
        for address, translation in self.translations.items():
            for operation in translation.ops:
                varnodes = (
                    *operation.inputs,
                    *((operation.output,) if operation.output is not None else ()),
                )
                for varnode in varnodes:
                    if varnode.space != "register":
                        continue
                    family = _register_family(varnode.display)
                    locations = self._locations_for_varnode(varnode, address)
                    if family in self._call_clobber_families:
                        for location in locations:
                            self._call_register_families[location] = family
                    if address in call_addresses and family == "rsp":
                        self._call_stack_locations[address].update(locations)

    def _machine_blocks(self) -> list[BasicBlock]:
        blocks = sorted(self.function.blocks, key=lambda item: item.start)
        if blocks:
            return blocks
        if self.function.instructions:
            return [
                BasicBlock(
                    self.function.instructions[0].address,
                    tuple(self.function.instructions),
                    (),
                )
            ]
        return []

    @staticmethod
    def _signed_offset(varnode: SleighVarnode) -> int:
        bits = max(varnode.size_bits, 1)
        value = int(varnode.offset) & ((1 << bits) - 1)
        if value & (1 << (bits - 1)):
            value -= 1 << bits
        return value

    def _build_pcode_cfg(self) -> None:
        machine_blocks = self._machine_blocks()
        if not machine_blocks:
            return
        machine_by_start = {block.start: block for block in machine_blocks}
        machine_predecessors: dict[int, set[int]] = defaultdict(set)
        for block in machine_blocks:
            for successor in block.successors:
                if successor in machine_by_start:
                    machine_predecessors[successor].add(block.start)
            for index, instruction in enumerate(block.instructions):
                next_instruction = (
                    block.instructions[index + 1]
                    if index + 1 < len(block.instructions)
                    else None
                )
                self._instruction_next[instruction.address] = next_instruction

        for instruction in self.function.instructions:
            translation = self.translations.get(instruction.address)
            if translation is None:
                self.complete = False
                self.diagnostics.append(
                    f"missing SLEIGH translation at {instruction.address:#x}"
                )
            elif translation.error or not translation.translated:
                self.complete = False
                self.diagnostics.append(
                    f"SLEIGH translation failed at {instruction.address:#x}: "
                    f"{translation.error or 'no p-code operations'}"
                )

        for machine_block in machine_blocks:
            if not machine_block.instructions:
                identifier = len(self._blocks)
                pseudo_instruction = Instruction(machine_block.start, 0, b"", "", "")
                item = _PcodeBlock(
                    identifier,
                    machine_block,
                    pseudo_instruction,
                    None,
                    0,
                    0,
                    (),
                    set(),
                    set(),
                )
                self._blocks.append(item)
                self._block_by_id[identifier] = item
                self._machine_entry.setdefault(machine_block.start, identifier)
                continue

            for instruction in machine_block.instructions:
                sleigh = self.translations.get(instruction.address)
                pcode = sleigh.ops if sleigh is not None and not sleigh.error else ()
                leaders = {0}
                for op_index, operation in enumerate(pcode):
                    if operation.opcode in {
                        "BRANCH",
                        "CBRANCH",
                        "BRANCHIND",
                        "CALL",
                        "CALLIND",
                        "RETURN",
                    } and op_index + 1 < len(pcode):
                        leaders.add(op_index + 1)
                    if (
                        operation.opcode in {"BRANCH", "CBRANCH"}
                        and operation.inputs
                        and operation.inputs[0].space == "const"
                    ):
                        target_index = op_index + self._signed_offset(
                            operation.inputs[0]
                        )
                        if 0 <= target_index < len(pcode):
                            leaders.add(target_index)
                        elif target_index != len(pcode):
                            self.complete = False
                            self.diagnostics.append(
                                "p-code branch target out of range at "
                                f"{instruction.address:#x}:{op_index}"
                            )
                starts = sorted(leaders)
                ranges = (
                    [
                        (
                            start,
                            starts[index + 1]
                            if index + 1 < len(starts)
                            else len(pcode),
                        )
                        for index, start in enumerate(starts)
                    ]
                    if pcode
                    else [(0, 0)]
                )
                first_identifier = len(self._blocks)
                self._instruction_entry[instruction.address] = first_identifier
                self._machine_entry.setdefault(machine_block.start, first_identifier)
                for operation_start, operation_end in ranges:
                    identifier = len(self._blocks)
                    item = _PcodeBlock(
                        identifier,
                        machine_block,
                        instruction,
                        sleigh,
                        operation_start,
                        operation_end,
                        tuple(pcode[operation_start:operation_end]),
                        set(),
                        set(),
                    )
                    self._blocks.append(item)
                    self._block_by_id[identifier] = item

                # Map each SLEIGH operation index to the micro-block containing it.
                for item in self._blocks[first_identifier:]:
                    if item.instruction is not instruction:
                        continue
                    for op_index in range(item.operation_start, item.operation_end):
                        self._op_to_block[(instruction.address, op_index)] = (
                            item.identifier
                        )

        machine_roots = [
            block.start
            for block in machine_blocks
            if not machine_predecessors.get(block.start)
        ]
        if self.function.address in self._machine_entry:
            machine_roots = [
                self.function.address,
                *[root for root in machine_roots if root != self.function.address],
            ]
        if not machine_roots:
            machine_roots = [min(machine_by_start)]
        roots = [
            self._machine_entry[root]
            for root in machine_roots
            if root in self._machine_entry
        ]

        for item in self._blocks:
            instruction = item.instruction
            pcode = (
                item.sleigh.ops
                if item.sleigh is not None and not item.sleigh.error
                else ()
            )
            last_index = item.operation_end - 1
            last = pcode[last_index] if 0 <= last_index < len(pcode) else None
            if last is not None and last.opcode == "BRANCH":
                self._add_branch_successors(item, last_index, last, roots)
            elif last is not None and last.opcode == "CBRANCH":
                self._add_branch_successors(item, last_index, last, roots)
                if last_index + 1 < len(pcode):
                    self._add_edge(
                        item.identifier,
                        self._op_to_block.get((instruction.address, last_index + 1)),
                    )
                else:
                    self._add_continuations(item)
            elif last is not None and last.opcode in {"RETURN", "BRANCHIND"}:
                if last.opcode == "BRANCHIND":
                    self._add_machine_successors(item)
                    if not item.machine_block.successors:
                        self.complete = False
                        self.diagnostics.append(
                            "unresolved indirect p-code branch at "
                            f"{item.instruction_address:#x}"
                        )
            else:
                self._add_continuations(item)

        # Keep disconnected recovered machine fragments as roots, but do not
        # promote dead p-code tails after internal unconditional branches.
        reachable: set[int] = set()
        pending = list(reversed(roots))
        while pending:
            identifier = pending.pop()
            if identifier in reachable or identifier not in self._block_by_id:
                continue
            reachable.add(identifier)
            pending.extend(
                sorted(self._block_by_id[identifier].successors, reverse=True)
            )
        for machine_block in machine_blocks:
            entry = self._machine_entry.get(machine_block.start)
            if entry is None or entry in reachable:
                continue
            if any(
                self._machine_entry.get(predecessor) in reachable
                for predecessor in machine_predecessors.get(machine_block.start, ())
            ):
                continue
            roots.append(entry)
            pending = [entry]
            while pending:
                identifier = pending.pop()
                if identifier in reachable or identifier not in self._block_by_id:
                    continue
                reachable.add(identifier)
                pending.extend(
                    sorted(self._block_by_id[identifier].successors, reverse=True)
                )
        self._entry_blocks = tuple(dict.fromkeys(roots))
        self._active_blocks = reachable
        for item in self._blocks:
            item.reachable = item.identifier in reachable
            for successor in item.successors:
                self._block_by_id[successor].predecessors.add(item.identifier)

    def _add_branch_successors(
        self,
        item: _PcodeBlock,
        operation_index: int,
        operation: SleighPcodeOp,
        roots: list[int],
    ) -> None:
        del roots  # Roots are derived from the completed CFG.
        if not operation.inputs:
            self.complete = False
            self.diagnostics.append(
                f"p-code {operation.opcode} without target at {item.instruction_address:#x}"
            )
            return
        target = operation.inputs[0]
        instruction = item.instruction
        if target.space == "const":
            target_index = operation_index + self._signed_offset(target)
            pcode = item.sleigh.ops if item.sleigh is not None else ()
            if target_index == len(pcode):
                self._add_continuations(item)
            elif 0 <= target_index < len(pcode):
                self._add_edge(
                    item.identifier,
                    self._op_to_block.get((instruction.address, target_index)),
                )
            else:
                self.complete = False
                self.diagnostics.append(
                    f"invalid internal branch at {instruction.address:#x}:"
                    f"{operation_index}->{target_index}"
                )
            return
        if target.space == "ram":
            if instruction.is_call:
                self._add_continuations(item)
                return
            if target.offset == instruction.next_address:
                self._add_continuations(item)
                return
            destination = self._instruction_entry.get(target.offset)
            if destination is None:
                destination = self._machine_entry.get(target.offset)
            if destination is not None:
                self._add_edge(item.identifier, destination)
            elif self.function.address <= target.offset < self.function.end:
                self.complete = False
                self.diagnostics.append(
                    f"unresolved in-function p-code branch target {target.offset:#x}"
                )
            # A branch to an address outside the recovered function is a valid
            # terminal CFG edge, not an SSA construction error.
            return
        self.complete = False
        self.diagnostics.append(
            f"unsupported p-code branch target at {instruction.address:#x}:"
            f"{target.space}"
        )

    def _add_continuations(self, item: _PcodeBlock) -> None:
        instruction = item.instruction
        if (
            instruction is not None
            and instruction.is_call
            and instruction.address in self._no_return_calls
        ):
            return
        if instruction is not None:
            next_instruction = self._instruction_next.get(instruction.address)
            if next_instruction is not None:
                self._add_edge(
                    item.identifier,
                    self._instruction_entry.get(next_instruction.address),
                )
                return
        self._add_machine_successors(item)

    def _add_machine_successors(self, item: _PcodeBlock) -> None:
        for successor in item.machine_block.successors:
            self._add_edge(item.identifier, self._machine_entry.get(successor))

    def _add_edge(self, source: int, target: int | None) -> None:
        if target is None or target == source:
            if target == source:
                self._block_by_id[source].successors.add(target)
            return
        self._block_by_id[source].successors.add(target)

    @staticmethod
    def _locations_for_varnode(
        varnode: SleighVarnode, instruction_address: int
    ) -> tuple[SSALocation, ...]:
        if varnode.space in {"const", "ram"}:
            return ()
        scope = instruction_address if varnode.space == "unique" else None
        return tuple(
            SSALocation(varnode.space, varnode.offset + byte, scope)
            for byte in range(varnode.size)
        )

    @staticmethod
    def _branch_target_input(operation: SleighPcodeOp, input_index: int) -> bool:
        return operation.opcode in {"BRANCH", "CBRANCH", "CALL"} and input_index == 0

    @staticmethod
    def _userop_id(operation: SleighPcodeOp) -> int | None:
        if (
            operation.opcode == "CALLOTHER"
            and operation.inputs
            and operation.inputs[0].space == "const"
        ):
            return operation.inputs[0].offset
        return None

    @staticmethod
    def _is_call_operation(item: _PcodeBlock, operation: SleighPcodeOp) -> bool:
        instruction = item.instruction
        direct_target = (
            operation.inputs[0]
            if operation.inputs and operation.opcode in {"CALL", "BRANCH"}
            else None
        )
        is_call_next = (
            instruction is not None
            and instruction.is_call
            and direct_target is not None
            and direct_target.space == "ram"
            and direct_target.offset == instruction.next_address
        )
        if is_call_next:
            return False
        return operation.opcode in {"CALL", "CALLIND"} or (
            instruction is not None
            and instruction.is_call
            and operation.opcode == "BRANCH"
        )

    def _memory_effects(
        self, operation: SleighPcodeOp, item: _PcodeBlock | None = None
    ) -> tuple[bool, bool, bool]:
        """Return (reads-memory, writes-memory, unknown-clobber)."""
        if operation.opcode == "IMARK":
            return False, False, False
        direct_read = any(
            varnode.space == "ram" and not self._branch_target_input(operation, index)
            for index, varnode in enumerate(operation.inputs)
        )
        direct_write = operation.output is not None and operation.output.space == "ram"
        call_like = operation.opcode in {"CALL", "CALLIND"} or (
            item is not None and self._is_call_operation(item, operation)
        )
        userop_id = self._userop_id(operation)
        unknown_userop = (
            operation.opcode == "CALLOTHER" and userop_id not in _KNOWN_PURE_USEROPS
        )
        clobber = call_like or unknown_userop
        reads = operation.opcode in {"LOAD", "STORE"} or direct_read or clobber
        writes = operation.opcode == "STORE" or direct_write or clobber
        return reads, writes, clobber

    def _collect_use_def(self) -> None:
        for item in self._blocks:
            if not item.reachable:
                continue
            seen_definitions: set[SSALocation] = set()
            for operation in item.operations:
                if operation.opcode == "IMARK":
                    continue
                for index, varnode in enumerate(operation.inputs):
                    if varnode.space == "const" or self._branch_target_input(
                        operation, index
                    ):
                        continue
                    if varnode.space == "ram":
                        if MEMORY_LOCATION not in seen_definitions:
                            self._uses[item.identifier].add(MEMORY_LOCATION)
                        continue
                    for location in self._locations_for_varnode(
                        varnode, item.instruction_address
                    ):
                        if location not in seen_definitions:
                            self._uses[item.identifier].add(location)

                reads_memory, writes_memory, _clobber = self._memory_effects(
                    operation, item
                )
                if reads_memory and MEMORY_LOCATION not in seen_definitions:
                    self._uses[item.identifier].add(MEMORY_LOCATION)

                if self._is_call_operation(item, operation):
                    call_locations = set(self._call_register_families)
                    call_locations.update(
                        self._call_stack_locations.get(item.instruction_address, ())
                    )
                    for location in call_locations:
                        seen_definitions.add(location)
                        self._defs[item.identifier].add(location)
                        self._definitions[location].add(item.identifier)

                output = operation.output
                if output is not None and output.space != "ram":
                    for location in self._locations_for_varnode(
                        output, item.instruction_address
                    ):
                        seen_definitions.add(location)
                        self._defs[item.identifier].add(location)
                        self._definitions[location].add(item.identifier)
                if writes_memory:
                    seen_definitions.add(MEMORY_LOCATION)
                    self._defs[item.identifier].add(MEMORY_LOCATION)
                    self._definitions[MEMORY_LOCATION].add(item.identifier)

    def _compute_liveness(self) -> None:
        order = [
            item.identifier
            for item in self._blocks
            if item.identifier in self._active_blocks
        ]
        changed = True
        while changed:
            changed = False
            for identifier in reversed(order):
                outgoing: set[SSALocation] = set()
                for successor in self._successors.get(identifier, ()):
                    outgoing.update(self._live_in[successor])
                incoming = self._uses[identifier] | (outgoing - self._defs[identifier])
                if (
                    outgoing != self._live_out[identifier]
                    or incoming != self._live_in[identifier]
                ):
                    self._live_out[identifier] = outgoing
                    self._live_in[identifier] = incoming
                    changed = True

    def _place_phis(self) -> None:
        for location in sorted(self._definitions):
            definitions = self._definitions[location]
            worklist = deque(sorted(definitions, key=self._rpo_index.get))
            queued = set(definitions)
            placed: set[int] = set()
            while worklist:
                block = worklist.popleft()
                for frontier in sorted(
                    self._dom_frontier.get(block, ()),
                    key=self._rpo_index.get,
                ):
                    if (
                        frontier not in self._active_blocks
                        or location not in self._live_in[frontier]
                        or frontier in placed
                    ):
                        continue
                    placed.add(frontier)
                    self._phi_locations[frontier].add(location)
                    if frontier not in definitions and frontier not in queued:
                        queued.add(frontier)
                        worklist.append(frontier)

        for block, locations in self._phi_locations.items():
            for location in sorted(locations):
                identifier = self._new_value(
                    kind="phi",
                    name=f"phi_{location.label}_b{block}",
                    size_bits=1 if location.space == _MEMORY else 8,
                    block_id=block,
                    storage=location,
                )
                self._phi_outputs[(block, location)] = identifier

    def _compute_dominators(self) -> None:
        self._successors = {
            item.identifier: set(item.successors) & self._active_blocks
            for item in self._blocks
            if item.identifier in self._active_blocks
        }
        self._predecessors = defaultdict(set)
        for block, successors in self._successors.items():
            for successor in successors:
                self._predecessors[successor].add(block)

        roots = [entry for entry in self._entry_blocks if entry in self._active_blocks]
        self._predecessors[self._virtual_root] = set()
        self._successors[self._virtual_root] = set(roots)
        for root in roots:
            self._predecessors[root].add(self._virtual_root)

        postorder: list[int] = []
        visited: set[int] = set()
        stack: list[tuple[int, bool]] = [(self._virtual_root, False)]
        while stack:
            node, exiting = stack.pop()
            if exiting:
                postorder.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for successor in sorted(self._successors.get(node, ()), reverse=True):
                if successor not in visited:
                    stack.append((successor, False))
        reverse_postorder = list(reversed(postorder))
        self._rpo_index = {
            block: index for index, block in enumerate(reverse_postorder)
        }
        self._dom_parent[self._virtual_root] = self._virtual_root

        def intersect(left: int, right: int) -> int:
            while left != right:
                while self._rpo_index[left] > self._rpo_index[right]:
                    left = self._dom_parent[left]
                while self._rpo_index[right] > self._rpo_index[left]:
                    right = self._dom_parent[right]
            return left

        changed = True
        while changed:
            changed = False
            for block in reverse_postorder[1:]:
                processed = sorted(
                    (
                        predecessor
                        for predecessor in self._predecessors.get(block, ())
                        if predecessor in self._dom_parent
                    ),
                    key=self._rpo_index.get,
                )
                if not processed:
                    continue
                new_parent = processed[0]
                for predecessor in processed[1:]:
                    new_parent = intersect(predecessor, new_parent)
                if self._dom_parent.get(block) != new_parent:
                    self._dom_parent[block] = new_parent
                    changed = True

        for block in self._active_blocks:
            parent = self._dom_parent.get(block, self._virtual_root)
            self._dom_parent[block] = parent
            self._dom_children[parent].append(block)
        for children in self._dom_children.values():
            children.sort(key=self._rpo_index.get)

        for block in self._active_blocks:
            predecessors = set(self._predecessors.get(block, ()))
            if len(predecessors) < 2:
                continue
            parent = self._dom_parent[block]
            for predecessor in predecessors:
                runner = predecessor
                while runner != parent:
                    self._dom_frontier[runner].add(block)
                    next_runner = self._dom_parent.get(runner)
                    if next_runner is None or next_runner == runner:
                        break
                    runner = next_runner

    def _current_value(self, location: SSALocation) -> int:
        stack = self._stacks.get(location)
        if stack:
            return stack[-1]
        value = self._input_values.get(location)
        if value is None:
            value = self._new_value(
                kind="memory_input" if location.space == _MEMORY else "input",
                name=f"input_{location.label}",
                size_bits=1 if location.space == _MEMORY else 8,
                storage=location,
            )
            self._input_values[location] = value
        return value

    def _return_register_values(self) -> tuple[tuple[str, SSAOperand], ...]:
        locations_by_family: dict[str, list[SSALocation]] = defaultdict(list)
        for location, family in self._call_register_families.items():
            if family in self._return_families:
                locations_by_family[family].append(location)
        return tuple(
            (
                family,
                SSAOperand(
                    "register",
                    len(locations) * 8,
                    values=tuple(
                        self._current_value(location) for location in locations
                    ),
                ),
            )
            for family in self._return_register_order
            if (locations := sorted(locations_by_family.get(family, ())))
        )

    def _read_operand(
        self,
        varnode: SleighVarnode,
        instruction_address: int,
        *,
        operation: SleighPcodeOp,
        input_index: int,
    ) -> SSAOperand:
        size_bits = max(varnode.size_bits, 1)
        if varnode.space == "const":
            return SSAOperand(
                "constant",
                size_bits,
                constant=varnode.offset & ((1 << size_bits) - 1),
            )
        if varnode.space == "ram":
            if self._branch_target_input(operation, input_index):
                return SSAOperand("address", size_bits, address=varnode.offset)
            return SSAOperand(
                "memory",
                size_bits,
                address=varnode.offset,
                memory_version=self._current_value(MEMORY_LOCATION),
            )
        locations = self._locations_for_varnode(varnode, instruction_address)
        values = tuple(self._current_value(location) for location in locations)
        return SSAOperand(
            "value" if len(values) == 1 else "composite",
            size_bits,
            values=values,
        )

    def _rename(self) -> None:
        for entry in self._entry_blocks:
            for location in self._phi_locations.get(entry, ()):
                key = (entry, location)
                self._phi_incoming[key][None] = self._current_value(location)

        events: list[tuple[str, int, tuple[SSALocation, ...]]] = []
        for child in reversed(self._dom_children.get(self._virtual_root, ())):
            events.append(("enter", child, ()))
        while events:
            action, block_id, pushed = events.pop()
            if action == "exit":
                for location in reversed(pushed):
                    self._stacks[location].pop()
                continue

            local_pushes: list[SSALocation] = []
            for location in sorted(self._phi_locations.get(block_id, ())):
                phi_value = self._phi_outputs[(block_id, location)]
                self._stacks[location].append(phi_value)
                local_pushes.append(location)

            item = self._block_by_id[block_id]
            if item.operation_start == 0:
                for location in self._call_stack_locations.get(
                    item.instruction_address, ()
                ):
                    self._call_entry_values[item.instruction_address][location] = (
                        self._current_value(location)
                    )

            for operation in item.operations:
                if operation.opcode == "IMARK":
                    operands: tuple[SSAOperand, ...] = ()
                else:
                    operands = tuple(
                        self._read_operand(
                            varnode,
                            item.instruction_address,
                            operation=operation,
                            input_index=input_index,
                        )
                        for input_index, varnode in enumerate(operation.inputs)
                    )
                if operation.opcode == "RETURN":
                    self._return_sites.append(
                        SSAReturnSite(
                            block_id,
                            item.instruction_address,
                            operation.sequence,
                            self._return_register_values(),
                        )
                    )
                is_call = self._is_call_operation(item, operation)
                reads_memory, writes_memory, clobbers_memory = self._memory_effects(
                    operation, item
                )
                memory_inputs = (
                    (self._current_value(MEMORY_LOCATION),)
                    if reads_memory or writes_memory
                    else ()
                )

                output = operation.output
                output_value = None
                output_parts: tuple[int, ...] = ()
                if output is not None:
                    locations = self._locations_for_varnode(
                        output, item.instruction_address
                    )
                    output_value = self._new_value(
                        kind="operation",
                        name=(
                            f"{operation.opcode.lower()}_"
                            f"{item.instruction_address:X}_{operation.sequence}"
                        ),
                        size_bits=max(output.size_bits, 1),
                        block_id=block_id,
                        instruction_address=item.instruction_address,
                        operation_sequence=operation.sequence,
                        opcode=operation.opcode,
                        storage=(
                            locations[0]
                            if output.size == 1 and len(locations) == 1
                            else None
                        ),
                    )
                    part_ids: list[int] = []
                    for byte_index, location in enumerate(locations):
                        if output.size == 1:
                            part_id = output_value
                        else:
                            part_id = self._new_value(
                                kind="slice",
                                name=f"slice_{output_value}_{byte_index}",
                                size_bits=8,
                                block_id=block_id,
                                instruction_address=item.instruction_address,
                                operation_sequence=operation.sequence,
                                opcode=operation.opcode,
                                storage=location,
                                source=output_value,
                                offset_bits=byte_index * 8,
                            )
                        self._stacks[location].append(part_id)
                        local_pushes.append(location)
                        part_ids.append(part_id)
                    output_parts = tuple(part_ids)

                call_effect_values: list[int] = []
                if is_call:
                    call_locations = set(self._call_register_families)
                    call_locations.update(
                        self._call_stack_locations.get(item.instruction_address, ())
                    )
                    for location in sorted(call_locations):
                        family = self._call_register_families.get(location)
                        stack_source = self._call_entry_values.get(
                            item.instruction_address, {}
                        ).get(location)
                        if stack_source is not None:
                            kind = "call_stack_restore"
                        elif family in self._return_families:
                            kind = "call_return"
                        else:
                            kind = "call_clobber"
                        value_id = self._new_value(
                            kind=kind,
                            name=f"{kind}_{location.label}_{item.instruction_address:X}",
                            size_bits=8,
                            block_id=block_id,
                            instruction_address=item.instruction_address,
                            operation_sequence=operation.sequence,
                            opcode=operation.opcode,
                            storage=location,
                            source=stack_source,
                        )
                        self._stacks[location].append(value_id)
                        local_pushes.append(location)
                        call_effect_values.append(value_id)

                memory_write_address = None
                memory_write_value = None
                memory_write_bits = None
                if operation.opcode == "STORE" and len(operands) >= 3:
                    memory_write_address = operands[1]
                    memory_write_value = operands[2]
                    memory_write_bits = operation.inputs[2].size_bits
                elif output is not None and output.space == "ram":
                    memory_write_address = SSAOperand(
                        "address", output.size_bits, address=output.offset
                    )
                    if output_value is not None:
                        memory_write_value = SSAOperand(
                            "value", output.size_bits, values=(output_value,)
                        )
                        memory_write_bits = output.size_bits
                elif clobbers_memory:
                    memory_write_address = None
                    memory_write_value = None
                    memory_write_bits = None

                memory_output = None
                if writes_memory:
                    previous_memory = memory_inputs[0]
                    memory_output = self._new_value(
                        kind=("memory_clobber" if clobbers_memory else "memory_write"),
                        name=f"memory_{item.instruction_address:X}_{operation.sequence}",
                        size_bits=1,
                        block_id=block_id,
                        instruction_address=item.instruction_address,
                        operation_sequence=operation.sequence,
                        opcode=operation.opcode,
                        storage=MEMORY_LOCATION,
                        source=previous_memory,
                    )
                    self._stacks[MEMORY_LOCATION].append(memory_output)
                    local_pushes.append(MEMORY_LOCATION)

                ssa_operation = SSAOperation(
                    index=len(self._operation_list),
                    block_id=block_id,
                    instruction_address=item.instruction_address,
                    instruction_mnemonic=(
                        item.sleigh.mnemonic
                        if item.sleigh is not None
                        else item.instruction.mnemonic
                        if item.instruction is not None
                        else ""
                    ),
                    sequence=operation.sequence,
                    opcode=operation.opcode,
                    inputs=operands,
                    output_value=output_value,
                    output_parts=output_parts,
                    memory_inputs=memory_inputs,
                    memory_output=memory_output,
                    memory_write_address=memory_write_address,
                    memory_write_value=memory_write_value,
                    memory_write_bits=memory_write_bits,
                    memory_clobber=clobbers_memory,
                    call_effect_values=tuple(call_effect_values),
                    pcode=operation,
                )
                self._operation_list.append(ssa_operation)
                self._operation_results[block_id].append(ssa_operation)

            for successor in self._successors.get(block_id, ()):
                for location in self._phi_locations.get(successor, ()):
                    self._phi_incoming[(successor, location)][block_id] = (
                        self._current_value(location)
                    )

            events.append(("exit", block_id, tuple(local_pushes)))
            for child in reversed(self._dom_children.get(block_id, ())):
                events.append(("enter", child, ()))

    def _new_value(
        self,
        *,
        kind: str,
        name: str,
        size_bits: int,
        block_id: int | None = None,
        instruction_address: int | None = None,
        operation_sequence: int | None = None,
        opcode: str | None = None,
        storage: SSALocation | None = None,
        source: int | None = None,
        offset_bits: int = 0,
    ) -> int:
        identifier = len(self._values)
        self._values.append(
            SSAValue(
                identifier,
                kind,
                name,
                size_bits,
                block_id,
                instruction_address,
                operation_sequence,
                opcode,
                storage,
                source,
                offset_bits,
            )
        )
        return identifier

    def _freeze_and_validate(self) -> SSAFunction:
        phis: list[SSAPhi] = []
        for (block, location), output in sorted(
            self._phi_outputs.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            incoming = tuple(
                sorted(
                    self._phi_incoming.get((block, location), {}).items(),
                    key=lambda item: -1 if item[0] is None else item[0],
                )
            )
            phis.append(SSAPhi(block, location, output, incoming))
            self._values[output] = replace(self._values[output], incoming=incoming)

        frozen_blocks: list[SSAControlBlock] = []
        for item in self._blocks:
            if item.identifier not in self._active_blocks:
                frozen_blocks.append(
                    SSAControlBlock(
                        item.identifier,
                        item.machine_block.start,
                        item.instruction_address,
                        item.sleigh.mnemonic
                        if item.sleigh is not None
                        else item.instruction.mnemonic
                        if item.instruction is not None
                        else "",
                        item.operation_start,
                        item.operation_end,
                        tuple(sorted(item.predecessors)),
                        tuple(sorted(item.successors)),
                        None,
                        (),
                        False,
                        (),
                        (),
                    )
                )
                continue
            parent = self._dom_parent.get(item.identifier, self._virtual_root)
            phi_values = tuple(
                self._phi_outputs[(item.identifier, location)]
                for location in sorted(self._phi_locations.get(item.identifier, ()))
            )
            frozen_blocks.append(
                SSAControlBlock(
                    item.identifier,
                    item.machine_block.start,
                    item.instruction_address,
                    item.sleigh.mnemonic
                    if item.sleigh is not None
                    else item.instruction.mnemonic
                    if item.instruction is not None
                    else "",
                    item.operation_start,
                    item.operation_end,
                    tuple(sorted(item.predecessors)),
                    tuple(sorted(item.successors)),
                    None if parent == self._virtual_root else parent,
                    tuple(sorted(self._dom_frontier.get(item.identifier, ()))),
                    True,
                    phi_values,
                    tuple(self._operation_results.get(item.identifier, ())),
                )
            )

        result = SSAFunction(
            address=self.function.address,
            blocks=tuple(frozen_blocks),
            values=tuple(self._values),
            phis=tuple(phis),
            operations=tuple(self._operation_list),
            entry_blocks=self._entry_blocks,
            live_in=tuple(
                (block, tuple(sorted(locations)))
                for block, locations in sorted(self._live_in.items())
                if block in self._active_blocks
            ),
            live_out=tuple(
                (block, tuple(sorted(locations)))
                for block, locations in sorted(self._live_out.items())
                if block in self._active_blocks
            ),
            diagnostics=tuple(self.diagnostics),
            complete=self.complete,
            return_sites=tuple(self._return_sites),
        )
        errors = self._validate(result)
        if errors:
            return replace(
                result,
                diagnostics=(*result.diagnostics, *errors),
                valid=False,
            )
        return replace(result, valid=True)

    def _validate(self, result: SSAFunction) -> tuple[str, ...]:
        errors: list[str] = []
        values = result.value_by_id
        blocks = result.block_by_id
        for identifier, value in enumerate(result.values):
            if identifier != value.identifier:
                errors.append(f"SSA value id/index mismatch at {identifier}")
            if value.source is not None and value.source not in values:
                errors.append(f"SSA value {identifier} references missing source")
            for _predecessor, incoming in value.incoming:
                if incoming not in values:
                    errors.append(f"SSA phi {identifier} references missing input")

        def dominates(definition: int, use_block: int) -> bool:
            seen: set[int] = set()
            cursor = use_block
            while cursor not in seen:
                if cursor == definition:
                    return True
                seen.add(cursor)
                parent = blocks[cursor].immediate_dominator
                if parent is None:
                    return False
                cursor = parent
            return False

        def check_use(value_id: int, block_id: int, sequence: int, where: str) -> None:
            definition = values.get(value_id)
            if definition is None:
                errors.append(f"{where} references missing SSA value {value_id}")
                return
            if definition.block_id is None:
                return
            if definition.block_id == block_id:
                if (
                    definition.kind != "phi"
                    and definition.operation_sequence is not None
                    and definition.operation_sequence >= sequence
                ):
                    errors.append(
                        f"SSA use before def for {definition.name} at {where}"
                    )
                return
            if not dominates(definition.block_id, block_id):
                errors.append(f"SSA def {definition.name} does not dominate {where}")

        for value in result.values:
            if value.kind == "call_stack_restore" and value.source is not None:
                check_use(
                    value.source,
                    value.block_id,
                    value.operation_sequence or 0,
                    f"call stack restore {value.instruction_address!r}",
                )

        for operation in result.operations:
            for output_id in (
                *operation.output_parts,
                *operation.call_effect_values,
                *(
                    (operation.output_value,)
                    if operation.output_value is not None
                    else ()
                ),
                *(
                    (operation.memory_output,)
                    if operation.memory_output is not None
                    else ()
                ),
            ):
                if output_id not in values:
                    errors.append(
                        f"operation {operation.index} references missing output {output_id}"
                    )
            for operand in operation.inputs:
                for value_id in operand.values:
                    check_use(
                        value_id,
                        operation.block_id,
                        operation.sequence,
                        f"{operation.instruction_address:#x}:{operation.sequence}",
                    )
                if operand.memory_version is not None:
                    check_use(
                        operand.memory_version,
                        operation.block_id,
                        operation.sequence,
                        f"memory operand {operation.instruction_address:#x}",
                    )
            for memory_value in operation.memory_inputs:
                check_use(
                    memory_value,
                    operation.block_id,
                    operation.sequence,
                    f"memory effect {operation.instruction_address:#x}",
                )
            if operation.memory_write_value is not None:
                for value_id in operation.memory_write_value.values:
                    if value_id == operation.output_value:
                        continue
                    check_use(
                        value_id,
                        operation.block_id,
                        operation.sequence,
                        f"store value {operation.instruction_address:#x}",
                    )

        for return_site in result.return_sites:
            if return_site.block_id not in blocks:
                errors.append(
                    f"return site at {return_site.instruction_address:#x} "
                    "references a missing block"
                )
                continue
            if not any(
                operation.block_id == return_site.block_id
                and operation.instruction_address == return_site.instruction_address
                and operation.sequence == return_site.operation_sequence
                and operation.opcode == "RETURN"
                for operation in result.operations
            ):
                errors.append(
                    f"return site at {return_site.instruction_address:#x} "
                    "has no matching RETURN operation"
                )
            for family, operand in return_site.register_values:
                for value_id in operand.values:
                    check_use(
                        value_id,
                        return_site.block_id,
                        return_site.operation_sequence,
                        f"return register {family} at "
                        f"{return_site.instruction_address:#x}",
                    )

        for phi in result.phis:
            block = blocks[phi.block_id]
            expected = set(block.predecessors)
            if phi.block_id in result.entry_blocks:
                expected.add(None)
            actual = {predecessor for predecessor, _value in phi.incoming}
            if expected != actual:
                errors.append(
                    f"phi {phi.output_value} has inputs {actual!r}, expected {expected!r}"
                )
            for predecessor, value_id in phi.incoming:
                if predecessor is None:
                    if values[value_id].block_id is not None:
                        errors.append(
                            f"entry phi {phi.output_value} has non-live-in input"
                        )
                    continue
                definition = values[value_id]
                if definition.block_id is not None and not dominates(
                    definition.block_id, predecessor
                ):
                    errors.append(
                        f"phi {phi.output_value} input {value_id} does not dominate edge"
                    )
        return tuple(errors)
