"""Experimental semantic pseudocode recovery for x86 and x86-64.

This module follows the broad stages visible in Ghidra's decompiler source:
instruction semantics are lifted into symbolic values, values are propagated
through the recovered control-flow graph, merge points receive phi values, and
the result is printed as C-like pseudocode.  It deliberately implements only a
small, explicit subset of x86 semantics; it is not a port of SLEIGH, p-code, or
the complete Ghidra decompiler.
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import ClassVar

try:
    from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64, Cs, CsError
    from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_OP_REG
except ImportError as exc:  # pragma: no cover - declared package dependency
    raise RuntimeError(
        "Falta Capstone. Instálalo con: python -m pip install capstone"
    ) from exc

from .abi import CallingConvention, calling_convention_for, default_calling_convention
from .model import (
    BasicBlock,
    Function,
    FunctionParameter,
    FunctionSignature,
    Instruction,
)


@dataclass(frozen=True, slots=True)
class _Expr:
    """Small immutable symbolic expression used as the decompiler's IR."""

    kind: str
    value: object = None
    args: tuple[_Expr, ...] = ()
    bits: int = 0


@dataclass(frozen=True, slots=True)
class _Flags:
    kind: str
    left: _Expr | None = None
    right: _Expr | None = None
    result: _Expr | None = None
    bits: int = 0
    carry: _Expr | None = None
    overflow: _Expr | None = None
    parity: _Expr | None = None


@dataclass
class _State:
    registers: dict[str, _Expr] = field(default_factory=dict)
    mmx_registers: dict[str, _Expr] = field(default_factory=dict)
    xmm_registers: dict[str, _Expr] = field(default_factory=dict)
    xmm_widths: dict[str, int] = field(default_factory=dict)
    stack: dict[tuple[int, int], _Expr] = field(default_factory=dict)
    x87_stack: list[_Expr] = field(default_factory=list)
    x87_flags: _Flags | None = None
    x87_status_flags: _Flags | None = None
    direction_flag: int | None = 0
    stack_delta: int | None = 0
    frame_delta: int | None = None
    flags: _Flags | None = None
    dirty_arguments: set[str] = field(default_factory=set)
    pending_pushes: list[tuple[int | None, _Expr]] = field(default_factory=list)

    def copy(self) -> _State:
        return _State(
            registers=dict(self.registers),
            mmx_registers=dict(self.mmx_registers),
            xmm_registers=dict(self.xmm_registers),
            xmm_widths=dict(self.xmm_widths),
            stack=dict(self.stack),
            x87_stack=list(self.x87_stack),
            x87_flags=self.x87_flags,
            x87_status_flags=self.x87_status_flags,
            direction_flag=self.direction_flag,
            stack_delta=self.stack_delta,
            frame_delta=self.frame_delta,
            flags=self.flags,
            dirty_arguments=set(self.dirty_arguments),
            pending_pushes=list(self.pending_pushes),
        )


@dataclass(frozen=True, slots=True)
class _Operand:
    kind: str
    size: int
    reg: str = ""
    imm: int = 0
    base: str = ""
    index: str = ""
    scale: int = 1
    displacement: int = 0
    segment: str = ""


@dataclass(frozen=True, slots=True)
class _Decoded:
    mnemonic: str
    operands: tuple[_Operand, ...]
    repeat: str = ""


class SemanticDecompiler:
    """Translate a recovered function into conservative C-like pseudocode.

    The first implementation handles common integer, data movement, stack,
    call, comparison, and control-flow instructions.  Register and stack values
    are propagated between basic blocks; values that disagree at a CFG join are
    represented with a ``phi(...)`` expression, following Ghidra's heritage /
    SSA merge model.  CFG transfers remain explicit gotos rather than claiming
    a complete structuring pass.
    """

    _ALIASES_64: ClassVar[dict[str, tuple[str, int, int]]] = {
        "rax": ("rax", 64, 0),
        "eax": ("rax", 32, 0),
        "ax": ("rax", 16, 0),
        "al": ("rax", 8, 0),
        "ah": ("rax", 8, 8),
        "rbx": ("rbx", 64, 0),
        "ebx": ("rbx", 32, 0),
        "bx": ("rbx", 16, 0),
        "bl": ("rbx", 8, 0),
        "bh": ("rbx", 8, 8),
        "rcx": ("rcx", 64, 0),
        "ecx": ("rcx", 32, 0),
        "cx": ("rcx", 16, 0),
        "cl": ("rcx", 8, 0),
        "ch": ("rcx", 8, 8),
        "rdx": ("rdx", 64, 0),
        "edx": ("rdx", 32, 0),
        "dx": ("rdx", 16, 0),
        "dl": ("rdx", 8, 0),
        "dh": ("rdx", 8, 8),
        "rsi": ("rsi", 64, 0),
        "esi": ("rsi", 32, 0),
        "si": ("rsi", 16, 0),
        "sil": ("rsi", 8, 0),
        "rdi": ("rdi", 64, 0),
        "edi": ("rdi", 32, 0),
        "di": ("rdi", 16, 0),
        "dil": ("rdi", 8, 0),
        "rbp": ("rbp", 64, 0),
        "ebp": ("rbp", 32, 0),
        "bp": ("rbp", 16, 0),
        "bpl": ("rbp", 8, 0),
        "rsp": ("rsp", 64, 0),
        "esp": ("rsp", 32, 0),
        "sp": ("rsp", 16, 0),
        "spl": ("rsp", 8, 0),
        "rip": ("rip", 64, 0),
        "eip": ("rip", 32, 0),
        "ip": ("rip", 16, 0),
    }
    _ALIASES_32: ClassVar[dict[str, tuple[str, int, int]]] = {
        "eax": ("eax", 32, 0),
        "ax": ("eax", 16, 0),
        "al": ("eax", 8, 0),
        "ah": ("eax", 8, 8),
        "ebx": ("ebx", 32, 0),
        "bx": ("ebx", 16, 0),
        "bl": ("ebx", 8, 0),
        "bh": ("ebx", 8, 8),
        "ecx": ("ecx", 32, 0),
        "cx": ("ecx", 16, 0),
        "cl": ("ecx", 8, 0),
        "ch": ("ecx", 8, 8),
        "edx": ("edx", 32, 0),
        "dx": ("edx", 16, 0),
        "dl": ("edx", 8, 0),
        "dh": ("edx", 8, 8),
        "esi": ("esi", 32, 0),
        "si": ("esi", 16, 0),
        "edi": ("edi", 32, 0),
        "di": ("edi", 16, 0),
        "ebp": ("ebp", 32, 0),
        "bp": ("ebp", 16, 0),
        "esp": ("esp", 32, 0),
        "sp": ("esp", 16, 0),
        "eip": ("eip", 32, 0),
        "ip": ("eip", 16, 0),
    }

    _CONDITION_ALIASES: ClassVar[dict[str, str]] = {
        "je": "eq",
        "jz": "eq",
        "jne": "ne",
        "jnz": "ne",
        "jb": "ult",
        "jnae": "ult",
        "jc": "ult",
        "jbe": "ule",
        "jna": "ule",
        "ja": "ugt",
        "jnbe": "ugt",
        "jae": "uge",
        "jnb": "uge",
        "jnc": "uge",
        "jl": "slt",
        "jnge": "slt",
        "jle": "sle",
        "jng": "sle",
        "jg": "sgt",
        "jnle": "sgt",
        "jge": "sge",
        "jnl": "sge",
        "jo": "of",
        "jno": "nof",
        "jp": "parity",
        "jpe": "parity",
        "jnp": "nparity",
        "jpo": "nparity",
    }

    _BINARY_OPERATORS: ClassVar[dict[str, str]] = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "imul": "*",
        "udiv": "/",
        "sdiv": "/",
        "urem": "%",
        "srem": "%",
        "and": "&",
        "or": "|",
        "xor": "^",
        "shl": "<<",
        "lshr": ">>",
        "ashr": ">>",
        "eq": "==",
        "ne": "!=",
        "ult": "<",
        "ule": "<=",
        "ugt": ">",
        "uge": ">=",
        "slt": "<",
        "sle": "<=",
        "sgt": ">",
        "sge": ">=",
        "land": "&&",
        "lor": "||",
    }

    def __init__(self, analyzer) -> None:
        self.analyzer = analyzer
        self.bits = analyzer.image.bits
        self.pointer_size = self.bits // 8
        mode = CS_MODE_32 if self.bits == 32 else CS_MODE_64
        self._md = Cs(CS_ARCH_X86, mode)
        self._md.detail = True
        self._decoded_cache: dict[int, _Decoded] = {}
        self._function: Function | None = None
        self._return_value_used_by_callers: bool | None = None
        self._saved_register_push_addresses: set[int] = set()
        self._blocks: list[BasicBlock] = []
        self._block_by_start: dict[int, BasicBlock] = {}
        self._predecessors: dict[int, list[int]] = defaultdict(list)
        self._out_states: dict[int, _State] = {}
        self._phi_inputs: dict[tuple[int, str], tuple[_Expr, ...]] = {}
        self._call_results: set[str] = set()
        self._saw_value_return = False
        self._return_type: str | None = None
        self._parameter_types: dict[str, str] = {}
        self._used_parameter_names: set[str] = set()
        self._calling_convention = default_calling_convention(analyzer)
        self._configure_calling_convention(self._calling_convention)
        self._string_by_address = {
            item.address: item.text for item in getattr(analyzer, "strings", ())
        }

    def decompile(self, function: Function) -> str:
        self._function = function
        self._calling_convention = calling_convention_for(self.analyzer, function)
        self._configure_calling_convention(self._calling_convention)
        self._return_value_used_by_callers = self._callers_use_return_value(function)
        self._parameter_types = {}
        self._used_parameter_names = set()
        if function.is_thunk:
            return self._decompile_thunk(function)

        self._blocks = sorted(function.blocks, key=lambda block: block.start)
        self._saved_register_push_addresses = self._find_prologue_saved_register_pushes(
            function
        )
        if not self._blocks and function.instructions:
            self._blocks = [
                BasicBlock(
                    function.instructions[0].address, tuple(function.instructions), ()
                )
            ]
        self._block_by_start = {block.start: block for block in self._blocks}
        self._predecessors = defaultdict(list)
        for block in self._blocks:
            for successor in block.successors:
                if successor in self._block_by_start:
                    self._predecessors[successor].append(block.start)
        for incoming in self._predecessors.values():
            incoming.sort()

        self._out_states = {}
        self._phi_inputs = {}
        self._analyze_dataflow()
        self._phi_inputs = {}
        input_states = {
            block.start: self._input_state_for(block.start, record_phis=True)
            for block in self._blocks
        }

        body: list[str] = []
        self._saw_value_return = False
        self._return_type = None
        self._call_results = set()
        for block in self._blocks:
            body.append(f"loc_{block.start:X}:")
            for (phi_block, location), values in sorted(self._phi_inputs.items()):
                if phi_block != block.start:
                    continue
                phi_name = self._phi_name(block.start, location)
                rendered_values = ", ".join(self._render(value) for value in values)
                body.append(f"  {phi_name} = phi({rendered_values});")

            input_state = input_states.get(block.start) or _State()
            output_state, statements = self._run_block(block, input_state, collect=True)
            body.extend(f"  {line}" for line in statements)
            body.extend(
                f"  {line}" for line in self._render_terminator(block, output_state)
            )

        body = self._prune_dead_phis(body)
        body = self._drop_unused_call_results(body)
        parameters = self._parameter_specs(body)
        params = (
            ", ".join(
                f"{parameter.type_name} {parameter.name}" for parameter in parameters
            )
            if parameters
            else "void"
        )
        return_type = (
            (self._return_type or f"uint{self.bits}_t")
            if self._saw_value_return
            else "void"
        )
        function.calling_convention = self._calling_convention.name
        function.signature = FunctionSignature(
            return_type, tuple(parameters), self._calling_convention.name
        )
        name = _c_identifier(function.name)
        convention_prefix = (
            f"{self._calling_convention.name} "
            if self._calling_convention.print_name
            else ""
        )
        lines = [
            "/* Experimental semantic pseudocode: symbolic register/stack propagation and CFG gotos.",
            " * Types, prototypes, and unsupported instruction effects remain conservative guesses;",
            " * this is not a byte-for-byte or feature-complete replacement for Ghidra. */",
            f"{return_type} {convention_prefix}{name}({params}) {{",
        ]
        lines.extend(
            f"  {declaration}" for declaration in self._local_declarations(body)
        )
        lines.extend(body)
        lines.append("}")
        return "\n".join(lines)

    def _decompile_thunk(self, function: Function) -> str:
        function.calling_convention = self._calling_convention.name
        function.signature = FunctionSignature(
            "void", (), self._calling_convention.name
        )
        name = _c_identifier(function.name)
        if function.thunk_target_name:
            target = _c_identifier(function.thunk_target_name)
        elif function.thunk_target is not None:
            target = f"loc_{function.thunk_target:X}"
        else:
            target = "indirect_target"
        lines = [
            "/* Tail thunk; the target prototype and forwarded arguments are unknown. */",
            f"void {name}(void) {{",
            f"  goto {target};",
            "}",
        ]
        return "\n".join(lines)

    def _configure_calling_convention(self, convention: CallingConvention) -> None:
        self._argument_registers = convention.argument_registers
        self._argument_index = {
            name: index for index, name in enumerate(self._argument_registers)
        }
        self._vector_argument_index = {
            name: index
            for index, name in enumerate(convention.vector_argument_registers)
        }
        self._tracked_argument_registers = set(self._argument_registers)
        self._tracked_argument_registers.update(convention.vector_argument_registers)
        if self.bits == 32:
            # Competing x86 models use ECX/EDX (MS fastcall/thiscall) or
            # EAX/EDX/ECX (GCC regparm); track writes for target-specific calls.
            self._tracked_argument_registers.update({"eax", "ecx", "edx"})
        else:
            self._tracked_argument_registers.update(
                {"rdi", "rsi", "rdx", "rcx", "r8", "r9"}
            )
            self._tracked_argument_registers.update(f"xmm{index}" for index in range(8))

    def _callers_use_return_value(self, function: Function) -> bool | None:
        return_family = self._calling_convention.integer_return_registers[0]
        return self._callers_use_register_before_write(function, return_family)

    def _callers_use_register_before_write(
        self, function: Function, return_family: str
    ) -> bool | None:
        call_sites = [
            xref
            for xref in self.analyzer.xrefs_to(function.address)
            if xref.kind == "call"
        ]
        if not call_sites:
            return None

        def register_families(register_ids: tuple[int, ...] | list[int]) -> set[str]:
            families: set[str] = set()
            for register_id in register_ids:
                name = self._reg_name(register_id)
                alias = self._register_alias(name)
                if alias is not None:
                    families.add(alias[0])
                elif re.fullmatch(r"xmm\d+", name):
                    families.add(name)
            return families

        resolved_sites = 0
        uncertain = False
        for xref in call_sites:
            try:
                caller = self.analyzer.function_at(xref.source)
            except ValueError:
                uncertain = True
                continue
            call_index = next(
                (
                    index
                    for index, instruction in enumerate(caller.instructions)
                    if instruction.address == xref.source
                ),
                None,
            )
            if call_index is None:
                uncertain = True
                continue
            resolved_sites += 1
            for candidate in caller.instructions[call_index + 1 : call_index + 17]:
                if candidate.is_return:
                    if not return_family.startswith("xmm"):
                        return True
                    uncertain = True
                    break
                if candidate.is_jump or candidate.is_trap:
                    uncertain = True
                    break
                try:
                    raw = next(
                        self._md.disasm(candidate.raw, candidate.address, count=1), None
                    )
                    if raw is None:
                        uncertain = True
                        break
                    reads, writes = raw.regs_access()
                except CsError:
                    uncertain = True
                    break
                read_families = register_families(reads)
                write_families = register_families(writes)
                if return_family in read_families:
                    return True
                if candidate.is_call or return_family in write_families:
                    break
            else:
                uncertain = True
        if uncertain or not resolved_sites:
            return None
        return False

    def _callers_use_register_pair_before_write(
        self, function: Function, low_register: str, high_register: str
    ) -> bool | None:
        call_sites = [
            xref
            for xref in self.analyzer.xrefs_to(function.address)
            if xref.kind == "call"
        ]
        if not call_sites:
            return None
        required = {low_register, high_register}
        unresolved = False
        for xref in call_sites:
            try:
                caller = self.analyzer.function_at(xref.source)
            except ValueError:
                unresolved = True
                continue
            call_index = next(
                (
                    index
                    for index, item in enumerate(caller.instructions)
                    if item.address == xref.source
                ),
                None,
            )
            if call_index is None:
                unresolved = True
                continue
            observed: set[str] = set()
            for candidate in caller.instructions[call_index + 1 : call_index + 17]:
                if candidate.is_return or candidate.is_jump or candidate.is_trap:
                    break
                try:
                    raw = next(
                        self._md.disasm(candidate.raw, candidate.address, count=1), None
                    )
                    if raw is None:
                        unresolved = True
                        break
                    reads, writes = raw.regs_access()
                except CsError:
                    unresolved = True
                    break

                def families(register_ids: tuple[int, ...] | list[int]) -> set[str]:
                    result: set[str] = set()
                    for register_id in register_ids:
                        alias = self._register_alias(self._reg_name(register_id))
                        if alias is not None:
                            result.add(alias[0])
                    return result

                read_families = families(reads)
                observed.update(read_families & required)
                if observed == required:
                    return True
                write_families = families(writes)
                if candidate.is_call or (write_families & (required - observed)):
                    break
        return None if unresolved else False

    def _decode(self, instruction: Instruction) -> _Decoded:
        cached = self._decoded_cache.get(instruction.address)
        if cached is not None:
            return cached
        try:
            raw = next(
                self._md.disasm(instruction.raw, instruction.address, count=1), None
            )
        except CsError:
            raw = None
        if raw is None:
            decoded = _Decoded(instruction.mnemonic.lower(), ())
        else:
            operands: list[_Operand] = []
            for operand in raw.operands:
                size = max(1, int(getattr(operand, "size", 0) or self.pointer_size))
                if operand.type == X86_OP_REG:
                    operands.append(
                        _Operand("reg", size, reg=self._reg_name(operand.reg))
                    )
                elif operand.type == X86_OP_IMM:
                    operands.append(_Operand("imm", size, imm=int(operand.imm)))
                elif operand.type == X86_OP_MEM:
                    memory = operand.mem
                    operands.append(
                        _Operand(
                            "mem",
                            size,
                            base=self._reg_name(memory.base),
                            index=self._reg_name(memory.index),
                            scale=int(memory.scale or 1),
                            displacement=int(memory.disp),
                            segment=self._reg_name(memory.segment),
                        )
                    )
                else:
                    operands.append(_Operand("other", size))
            mnemonic = str(raw.mnemonic).lower()
            repeat = ""
            repeat_match = re.match(r"^(rep(?:ne|nz|e|z)?)\s+(.+)$", mnemonic)
            if repeat_match:
                repeat, mnemonic = repeat_match.groups()
            decoded = _Decoded(mnemonic, tuple(operands), repeat)
        self._decoded_cache[instruction.address] = decoded
        return decoded

    def _reg_name(self, register_id: int) -> str:
        if not register_id:
            return ""
        try:
            return str(self._md.reg_name(register_id) or "").lower()
        except CsError:
            return ""

    def _register_alias(self, name: str) -> tuple[str, int, int] | None:
        name = name.lower()
        aliases = self._ALIASES_64 if self.bits == 64 else self._ALIASES_32
        alias = aliases.get(name)
        if alias is not None:
            return alias
        if self.bits == 64:
            match = re.fullmatch(r"r(8|9|1[0-5])(d|w|b)?", name)
            if match:
                family = f"r{match.group(1)}"
                width = {None: 64, "d": 32, "w": 16, "b": 8}[match.group(2)]
                return family, width, 0
        return None

    def _initial_register(self, family: str) -> _Expr:
        if family in self._argument_index:
            index = self._argument_index[family] + 1
            return _var(f"param_{index}", self.bits)
        return _var(f"in_{family}", self.bits)

    def _initial_xmm_register(self, register: str) -> _Expr:
        parameter_name = self._calling_convention.vector_parameter_name(register)
        if parameter_name is not None:
            return _var(parameter_name, 128)
        return _var(f"in_{register}", 128)

    def _note_vector_parameter_type(
        self, register: str, instruction: Instruction
    ) -> None:
        parameter_name = self._calling_convention.vector_parameter_name(register)
        if parameter_name is None:
            return
        mnemonic = instruction.mnemonic.lower()
        if "ss" in mnemonic:
            type_name = "float"
        elif "sd" in mnemonic:
            type_name = "double"
        elif "pd" in mnemonic:
            type_name = "__m128d"
        elif "ps" in mnemonic or mnemonic.endswith("aps"):
            type_name = "__m128"
        elif mnemonic in {"movdqa", "movdqu", "vmovdqa", "vmovdqu"}:
            type_name = "__m128i"
        else:
            type_name = "__m128"
        previous = self._parameter_types.get(parameter_name)
        if previous is None or previous.startswith("__m128"):
            self._parameter_types[parameter_name] = type_name

    def _initial_stack_value(self, offset: int, bits: int) -> _Expr:
        return _var(self._stack_name(offset), bits)

    def _stack_name(self, offset: int) -> str:
        if offset > 0 and offset % self.pointer_size == 0:
            convention = self._calling_convention
            if (
                convention.home_space_size
                and convention.home_space_start
                <= offset
                < convention.home_space_start + convention.home_space_size
            ):
                slot = (offset - convention.home_space_start) // self.pointer_size + 1
                return f"home_slot_{slot}"
            if offset >= convention.stack_argument_offset:
                slot = (offset - convention.stack_argument_offset) // self.pointer_size
                return f"param_{convention.parameter_register_count + slot + 1}"
        if offset < 0:
            return f"local_{abs(offset):X}"
        if offset == 0:
            return "stack_return_address"
        return f"stack_{offset:X}"

    def _stack_offset(self, operand: _Operand, state: _State) -> int | None:
        if operand.kind != "mem" or operand.index:
            return None
        if operand.base in {"rsp", "esp", "sp"}:
            if state.stack_delta is None:
                return None
            return state.stack_delta + operand.displacement
        if operand.base in {"rbp", "ebp", "bp"}:
            if state.frame_delta is None:
                return None
            return state.frame_delta + operand.displacement
        return None

    def _read_register(
        self, state: _State, name: str, instruction: Instruction
    ) -> _Expr:
        if name in {"rip", "eip", "ip"}:
            return _const(instruction.next_address, self.bits)
        alias = self._register_alias(name)
        if alias is None:
            return _var(f"in_{_c_identifier(name or 'reg')}", self.bits)
        family, width, offset = alias
        stack_family = "rsp" if self.bits == 64 else "esp"
        if family == stack_family and state.stack_delta is not None:
            value = _Expr("stack_pointer", state.stack_delta, bits=self.bits)
        else:
            value = state.registers.get(family, self._initial_register(family))
            if family in self._argument_index:
                state.dirty_arguments.add(family)
                self._used_parameter_names.add(
                    f"param_{self._argument_index[family] + 1}"
                )
        if offset:
            return _extract(value, offset, width)
        if width < value.bits:
            return _truncate(value, width)
        if width > value.bits:
            return _zero_extend(value, width)
        return value

    def _write_register(
        self,
        state: _State,
        name: str,
        value: _Expr,
        *,
        mark_argument: bool = True,
    ) -> str | None:
        alias = self._register_alias(name)
        if alias is None:
            return None
        family, width, offset = alias
        value = _resize(value, width)
        native_width = self.bits
        if offset == 0 and width == native_width:
            updated = value
        elif self.bits == 64 and offset == 0 and width == 32:
            # A write to a 32-bit x86-64 register zero-extends into its 64-bit parent.
            updated = _zero_extend(value, 64)
        else:
            old_value = state.registers.get(family, self._initial_register(family))
            updated = _insert_bits(old_value, value, offset, width, native_width)
        state.registers[family] = updated
        if mark_argument and family in self._tracked_argument_registers:
            state.dirty_arguments.add(family)
        return family

    def _read_operand(
        self, state: _State, operand: _Operand, instruction: Instruction
    ) -> _Expr:
        width = operand.size * 8
        if operand.kind == "reg":
            if re.fullmatch(r"mm[0-7]", operand.reg):
                value = state.mmx_registers.get(
                    operand.reg, _var(f"in_{operand.reg}", 64)
                )
                return _resize(value, width)
            if re.fullmatch(r"xmm\d+", operand.reg):
                self._note_vector_parameter_type(operand.reg, instruction)
                parameter_name = self._calling_convention.vector_parameter_name(
                    operand.reg
                )
                if parameter_name is not None:
                    self._used_parameter_names.add(parameter_name)
                    state.dirty_arguments.add(operand.reg)
                value = state.xmm_registers.get(
                    operand.reg, self._initial_xmm_register(operand.reg)
                )
                return _resize(value, width)
            return _resize(self._read_register(state, operand.reg, instruction), width)
        if operand.kind == "imm":
            return _const(operand.imm, width)
        if operand.kind == "mem":
            return self._read_memory(state, operand, instruction)
        return _var(f"unknown_operand_{instruction.address:X}", width)

    def _write_operand(
        self,
        state: _State,
        operand: _Operand,
        value: _Expr,
        instruction: Instruction,
        emitted: list[str],
        *,
        collect: bool,
    ) -> None:
        width = operand.size * 8
        if operand.kind == "reg" and re.fullmatch(r"mm[0-7]", operand.reg):
            state.mmx_registers[operand.reg] = _zero_extend(
                _resize(value, min(width, 64)), 64
            )
            return
        if operand.kind == "reg" and re.fullmatch(r"xmm\d+", operand.reg):
            state.xmm_registers[operand.reg] = _zero_extend(
                _resize(value, min(width, 128)), 128
            )
            mnemonic = instruction.mnemonic.lower()
            if "ss" in mnemonic:
                state.xmm_widths[operand.reg] = 32
            elif "sd" in mnemonic:
                state.xmm_widths[operand.reg] = 64
            elif mnemonic in {"movd", "vmovd"}:
                state.xmm_widths[operand.reg] = 32
            elif mnemonic in {"movq", "vmovq"}:
                state.xmm_widths[operand.reg] = 64
            else:
                state.xmm_widths[operand.reg] = 128
            if operand.reg in self._tracked_argument_registers:
                state.dirty_arguments.add(operand.reg)
            return
        value = _resize(value, width)
        if operand.kind == "reg":
            self._write_register(state, operand.reg, value)
            return
        if operand.kind != "mem":
            return
        offset = self._stack_offset(operand, state)
        if offset is not None:
            state.stack[(offset, width)] = value
            return
        if operand.segment in {"fs", "gs"}:
            # Segment-based TEB/SEH setup is not an outgoing call-argument list.
            state.pending_pushes.clear()
        if collect:
            address = self._memory_address(operand, state, instruction)
            emitted.append(
                f"*({_memory_type(width)} *)({self._render(address)}) = {self._render(value)};"
            )

    def _carry_flag(self, flags: _Flags | None) -> _Expr:
        if flags is None:
            return _var("flag_cf", 1)
        if flags.carry is not None:
            return flags.carry
        if flags.kind in {"logic", "test"}:
            return _const(0, 1)
        if flags.kind in {"cmp", "sub", "sbb"} and flags.left and flags.right:
            return _binary("ult", flags.left, flags.right, 1)
        return _var("flag_cf", 1)

    def _arithmetic_flags(
        self,
        operation: str,
        left: _Expr,
        right: _Expr,
        result: _Expr,
        bits: int,
        *,
        carry_in: _Expr | None = None,
        preserve_carry: _Expr | None = None,
    ) -> _Flags:
        left = _resize(left, bits)
        right = _resize(right, bits)
        result = _resize(result, bits)
        if operation in {"add", "adc"}:
            wide_bits = bits + 1
            wide_result = _binary(
                "add",
                _zero_extend(left, wide_bits),
                _zero_extend(right, wide_bits),
                wide_bits,
            )
            if carry_in is not None:
                wide_result = _binary(
                    "add", wide_result, _zero_extend(carry_in, wide_bits), wide_bits
                )
            carry = _extract(wide_result, bits, 1)
            same_sign = _unary(
                "boolnot",
                _binary(
                    "xor", _extract(left, bits - 1, 1), _extract(right, bits - 1, 1), 1
                ),
                1,
            )
            overflow = _binary(
                "and",
                same_sign,
                _binary(
                    "xor",
                    _extract(left, bits - 1, 1),
                    _extract(result, bits - 1, 1),
                    1,
                ),
                1,
            )
        else:
            wide_bits = bits + 1
            wide_left = _zero_extend(left, wide_bits)
            wide_right = _zero_extend(right, wide_bits)
            if carry_in is not None:
                wide_right = _binary(
                    "add", wide_right, _zero_extend(carry_in, wide_bits), wide_bits
                )
            carry = _binary("ult", wide_left, wide_right, 1)
            different_sign = _binary(
                "xor", _extract(left, bits - 1, 1), _extract(right, bits - 1, 1), 1
            )
            overflow = _binary(
                "and",
                different_sign,
                _binary(
                    "xor",
                    _extract(left, bits - 1, 1),
                    _extract(result, bits - 1, 1),
                    1,
                ),
                1,
            )
        if preserve_carry is not None:
            carry = preserve_carry
        return _Flags(
            operation,
            left,
            right,
            result,
            bits,
            carry,
            overflow,
            _parity(result),
        )

    def _logical_flags(
        self, kind: str, left: _Expr, right: _Expr, result: _Expr, bits: int
    ) -> _Flags:
        return _Flags(
            kind,
            left,
            right,
            result,
            bits,
            _const(0, 1),
            _const(0, 1),
            _parity(result),
        )

    def _x87_index(self, operand: _Operand) -> int | None:
        if operand.kind != "reg":
            return None
        match = re.fullmatch(r"st(?:\((\d+)\)|(\d+))?", operand.reg)
        if not match:
            return None
        return int(match.group(1) or match.group(2) or 0)

    def _x87_get(self, state: _State, index: int) -> _Expr:
        if 0 <= index < len(state.x87_stack):
            return state.x87_stack[index]
        return _var(f"unknown_x87_st{index}", 80)

    def _x87_set(self, state: _State, index: int, value: _Expr) -> None:
        if index < 0 or index >= 8:
            return
        while len(state.x87_stack) <= index:
            state.x87_stack.append(_var(f"unknown_x87_st{len(state.x87_stack)}", 80))
        state.x87_stack[index] = _resize(value, 80)

    def _x87_push(self, state: _State, value: _Expr) -> None:
        state.x87_stack.insert(0, _resize(value, 80))
        if len(state.x87_stack) > 8:
            state.x87_stack.pop()

    def _x87_pop(self, state: _State, count: int = 1) -> None:
        del state.x87_stack[: min(max(count, 0), len(state.x87_stack))]

    def _x87_read(
        self,
        state: _State,
        operand: _Operand,
        instruction: Instruction,
        *,
        integer: bool = False,
    ) -> _Expr:
        index = self._x87_index(operand)
        if index is not None:
            return self._x87_get(state, index)
        if operand.kind == "imm":
            return _fp_const(float(operand.imm))
        if operand.kind != "mem":
            return _var(f"unknown_x87_{instruction.address:X}", 80)
        width = operand.size * 8
        offset = self._stack_offset(operand, state)
        if offset is not None:
            stored = state.stack.get((offset, width))
            if stored is not None:
                if integer:
                    return _Expr("fp_from_int", args=(stored,), bits=80)
                if stored.kind in {
                    "fp_const",
                    "fp_load",
                    "fp_int_load",
                    "fp_op",
                    "fp_unary",
                    "fp_cast",
                    "fp_from_int",
                }:
                    return _resize(stored, 80)
        address = self._memory_address(operand, state, instruction)
        kind = "fp_int_load" if integer else "fp_load"
        return _Expr(kind, (width, integer), args=(address,), bits=80)

    def _x87_store(
        self,
        state: _State,
        operand: _Operand,
        value: _Expr,
        instruction: Instruction,
        emitted: list[str],
        *,
        collect: bool,
        integer: bool = False,
        truncate: bool = False,
    ) -> None:
        index = self._x87_index(operand)
        if index is not None:
            self._x87_set(state, index, value)
            return
        if operand.kind != "mem":
            return
        width = operand.size * 8
        if integer:
            stored = _Expr("fp_to_int", (width, truncate), args=(value,), bits=width)
        else:
            stored = _Expr("fp_cast", width, args=(value,), bits=width)
        offset = self._stack_offset(operand, state)
        if offset is not None:
            state.stack[(offset, width)] = stored
            return
        if collect:
            address = self._memory_address(operand, state, instruction)
            type_name = _signed_type(width) if integer else _float_type(width)
            emitted.append(
                f"/* {instruction.address:#x} */ *({type_name} *)({self._render(address)}) = {self._render(stored)};"
            )

    def _execute_x87(
        self,
        state: _State,
        instruction: Instruction,
        decoded: _Decoded,
        emitted: list[str],
        *,
        collect: bool,
    ) -> bool:
        mnemonic = decoded.mnemonic.lower()
        operands = decoded.operands

        def comment(text: str) -> None:
            if collect:
                emitted.append(f"/* {instruction.address:#x} */ /* {text} */")

        if mnemonic in {"wait", "fnop"}:
            return True
        if mnemonic in {"finit", "fninit", "emms"}:
            state.x87_stack.clear()
            if mnemonic == "emms":
                state.mmx_registers.clear()
            state.x87_flags = None
            state.x87_status_flags = None
            return True
        if mnemonic in {"fclex", "fnclex"}:
            state.x87_flags = None
            state.x87_status_flags = None
            return True
        if mnemonic in {
            "fld1",
            "fldz",
            "fldpi",
            "fldl2e",
            "fldl2t",
            "fldlg2",
            "fldln2",
        }:
            value = {
                "fld1": _fp_const(1.0),
                "fldz": _fp_const(0.0),
                "fldpi": _fp_const(math.pi),
                "fldl2e": _fp_const(math.log2(math.e)),
                "fldl2t": _fp_const(math.log2(10.0)),
                "fldlg2": _fp_const(math.log10(2.0)),
                "fldln2": _fp_const(math.log(2.0)),
            }[mnemonic]
            self._x87_push(state, value)
            return True
        if mnemonic == "fld" and operands:
            self._x87_push(state, self._x87_read(state, operands[0], instruction))
            return True
        if mnemonic in {"fst", "fstp"}:
            if operands:
                self._x87_store(
                    state,
                    operands[0],
                    self._x87_get(state, 0),
                    instruction,
                    emitted,
                    collect=collect,
                )
            if mnemonic == "fstp":
                self._x87_pop(state)
            return True
        if mnemonic in {"fist", "fistp", "fisttp"} and operands:
            self._x87_store(
                state,
                operands[0],
                self._x87_get(state, 0),
                instruction,
                emitted,
                collect=collect,
                integer=True,
                truncate=mnemonic == "fisttp",
            )
            if mnemonic in {"fistp", "fisttp"}:
                self._x87_pop(state)
            return True
        if mnemonic == "fild" and operands:
            self._x87_push(
                state, self._x87_read(state, operands[0], instruction, integer=True)
            )
            return True

        arithmetic_names = {
            "fadd",
            "faddp",
            "fiadd",
            "fsub",
            "fsubp",
            "fsubr",
            "fsubrp",
            "fisub",
            "fisubr",
            "fmul",
            "fmulp",
            "fimul",
            "fdiv",
            "fdivp",
            "fdivr",
            "fdivrp",
            "fidiv",
            "fidivr",
        }
        if mnemonic in arithmetic_names:
            pop = mnemonic.endswith("p")
            stem = mnemonic[:-1] if pop else mnemonic
            integer_source = stem.startswith("fi")
            operation_name = stem[2:] if integer_source else stem[1:]
            reverse = operation_name.endswith("r")
            operation_name = operation_name[:-1] if reverse else operation_name
            operation = {"add": "add", "sub": "sub", "mul": "mul", "div": "div"}.get(
                operation_name
            )
            if operation is None:
                return False
            if pop:
                destination = self._x87_index(operands[0]) if operands else 1
                destination = 1 if destination is None else destination
                left = self._x87_get(state, destination)
                right = self._x87_get(state, 0)
                if reverse:
                    left, right = right, left
                self._x87_set(state, destination, _fp_binary(operation, left, right))
                self._x87_pop(state)
            elif len(operands) >= 2 and all(
                item.kind == "reg" for item in operands[:2]
            ):
                destination = self._x87_index(operands[0]) or 0
                left = self._x87_get(state, destination)
                right = self._x87_get(state, self._x87_index(operands[1]) or 0)
                if reverse:
                    left, right = right, left
                self._x87_set(state, destination, _fp_binary(operation, left, right))
            elif operands:
                if operands[0].kind == "reg":
                    right = self._x87_get(state, self._x87_index(operands[0]) or 0)
                else:
                    right = self._x87_read(
                        state, operands[0], instruction, integer=integer_source
                    )
                left = self._x87_get(state, 0)
                if reverse:
                    left, right = right, left
                self._x87_set(state, 0, _fp_binary(operation, left, right))
            state.x87_flags = None
            return True

        if mnemonic in {"fcom", "fcomp", "fcompp", "fucom", "fucomp", "fucompp"}:
            left = self._x87_get(state, 0)
            right = (
                self._x87_read(state, operands[0], instruction)
                if operands
                else self._x87_get(state, 1)
            )
            state.x87_flags = _Flags("fp_compare", left, right, None, 80)
            if mnemonic.endswith("pp"):
                self._x87_pop(state, 2)
            elif mnemonic.endswith("p"):
                self._x87_pop(state)
            return True
        if mnemonic in {"fcomi", "fucomi", "fcomip", "fucomip"}:
            left = self._x87_get(state, 0)
            right = (
                self._x87_read(state, operands[-1], instruction)
                if operands
                else self._x87_get(state, 1)
            )
            state.flags = _Flags("fp_compare", left, right, None, 80)
            if mnemonic.endswith("p"):
                self._x87_pop(state)
            return True
        if mnemonic in {"fnstsw", "fstsw"}:
            state.x87_status_flags = state.x87_flags
            if operands:
                status = _var(f"x87_status_{instruction.address:X}", 16)
                self._write_operand(
                    state,
                    operands[0],
                    status,
                    instruction,
                    emitted,
                    collect=collect,
                )
            return True
        if mnemonic in {"fxch"}:
            index = self._x87_index(operands[0]) if operands else 1
            index = 1 if index is None else index
            left = self._x87_get(state, 0)
            right = self._x87_get(state, index)
            self._x87_set(state, 0, right)
            self._x87_set(state, index, left)
            return True
        if mnemonic in {"ftst"}:
            state.x87_flags = _Flags(
                "fp_compare", self._x87_get(state, 0), _fp_const(0.0), None, 80
            )
            return True
        if mnemonic in {
            "fabs",
            "fchs",
            "fsqrt",
            "fsin",
            "fcos",
            "fptan",
            "frndint",
            "f2xm1",
        }:
            value = self._x87_get(state, 0)
            operation = {
                "fabs": "abs",
                "fchs": "neg",
                "fsqrt": "sqrt",
                "fsin": "sin",
                "fcos": "cos",
                "fptan": "tan",
                "frndint": "round",
                "f2xm1": "exp2m1",
            }[mnemonic]
            result = _fp_unary(operation, value)
            self._x87_set(state, 0, result)
            if mnemonic == "fptan":
                self._x87_push(state, _fp_const(1.0))
            state.x87_flags = None
            return True
        if mnemonic in {"fyl2x", "fyl2xp1"}:
            x = self._x87_get(state, 0)
            y = self._x87_get(state, 1)
            if mnemonic == "fyl2xp1":
                x = _fp_binary("add", x, _fp_const(1.0))
            result = _Expr("fp_func", mnemonic, args=(y, x), bits=80)
            self._x87_set(state, 1, result)
            self._x87_pop(state)
            return True
        if mnemonic == "fpatan":
            result = _Expr(
                "fp_func",
                mnemonic,
                args=(self._x87_get(state, 1), self._x87_get(state, 0)),
                bits=80,
            )
            self._x87_set(state, 1, result)
            self._x87_pop(state)
            return True
        if mnemonic in {"fprem", "fprem1"}:
            self._x87_set(
                state,
                0,
                _Expr(
                    "fp_func",
                    mnemonic,
                    args=(self._x87_get(state, 0), self._x87_get(state, 1)),
                    bits=80,
                ),
            )
            return True
        if mnemonic == "fscale":
            self._x87_set(
                state,
                0,
                _Expr(
                    "fp_func",
                    mnemonic,
                    args=(self._x87_get(state, 0), self._x87_get(state, 1)),
                    bits=80,
                ),
            )
            return True
        if mnemonic in {"fincstp", "fdecstp"}:
            if state.x87_stack:
                amount = 1 if mnemonic == "fincstp" else -1
                state.x87_stack = state.x87_stack[amount:] + state.x87_stack[:amount]
            return True
        if mnemonic in {"fldcw", "fnstcw", "fstcw", "fldenv", "fnstenv", "fstenv"}:
            comment("x87 control/environment state is not fully modeled")
            return True
        if mnemonic in {"fnsave", "fsave", "frstor"}:
            comment("x87 save/restore image is not fully modeled")
            state.x87_stack = (
                []
                if mnemonic in {"fnsave", "fsave"}
                else [
                    _var(f"unknown_x87_st{index}_{instruction.address:X}", 80)
                    for index in range(8)
                ]
            )
            state.x87_flags = None
            state.x87_status_flags = None
            return True
        return False

    def _execute_mmx(
        self,
        state: _State,
        instruction: Instruction,
        decoded: _Decoded,
        emitted: list[str],
        *,
        collect: bool,
    ) -> bool:
        mnemonic = decoded.mnemonic
        operands = decoded.operands
        if any(
            operand.kind == "reg" and re.fullmatch(r"xmm\d+", operand.reg)
            for operand in operands
        ):
            return False

        if mnemonic == "cpuid":
            for register in ("eax", "ebx", "ecx", "edx"):
                self._write_register(
                    state,
                    register,
                    _var(f"cpuid_{instruction.address:X}_{register}", 32),
                    mark_argument=False,
                )
            if collect:
                emitted.append(
                    f"/* {instruction.address:#x} */ /* cpuid outputs are unknown */"
                )
            return True
        if mnemonic in {"rdtsc", "rdtscp"}:
            for register in ("eax", "edx"):
                self._write_register(
                    state,
                    register,
                    _var(f"{mnemonic}_{instruction.address:X}_{register}", 32),
                    mark_argument=False,
                )
            if mnemonic == "rdtscp":
                self._write_register(
                    state,
                    "ecx",
                    _var(f"rdtscp_{instruction.address:X}_ecx", 32),
                    mark_argument=False,
                )
            return True

        if mnemonic in {"movd", "movq"} and len(operands) >= 2:
            value = self._read_operand(state, operands[1], instruction)
            if mnemonic == "movd":
                if operands[0].kind == "reg" and re.fullmatch(
                    r"mm[0-7]", operands[0].reg
                ):
                    value = _zero_extend(value, 64)
                elif operands[1].kind == "reg" and re.fullmatch(
                    r"mm[0-7]", operands[1].reg
                ):
                    value = _truncate(value, 32)
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return True

        if mnemonic == "punpckldq" and len(operands) >= 2:
            left = self._read_operand(state, operands[0], instruction)
            right = self._read_operand(state, operands[1], instruction)
            result = _concat(_truncate(right, 32), _truncate(left, 32), 32)
            self._write_operand(
                state, operands[0], result, instruction, emitted, collect=collect
            )
            return True

        if mnemonic in {"pand", "pandn", "por", "pxor"} and len(operands) >= 2:
            left = self._read_operand(state, operands[0], instruction)
            right = self._read_operand(state, operands[1], instruction)
            operation = {"pand": "and", "pandn": "and", "por": "or", "pxor": "xor"}[
                mnemonic
            ]
            if mnemonic == "pandn":
                left = _unary("bitnot", left, 64)
                left, right = right, left
            result = _binary(operation, left, right, 64)
            self._write_operand(
                state, operands[0], result, instruction, emitted, collect=collect
            )
            return True

        lane_arithmetic = {
            "paddb": ("add", 8),
            "paddw": ("add", 16),
            "paddd": ("add", 32),
            "psubb": ("sub", 8),
            "psubw": ("sub", 16),
            "psubd": ("sub", 32),
            "pmullw": ("mul", 16),
        }
        if mnemonic in lane_arithmetic and len(operands) >= 2:
            operation, lane_width = lane_arithmetic[mnemonic]
            left = self._read_operand(state, operands[0], instruction)
            right = self._read_operand(state, operands[1], instruction)
            result = _mmx_lane_binary(operation, left, right, lane_width)
            self._write_operand(
                state, operands[0], result, instruction, emitted, collect=collect
            )
            return True

        lane_shifts = {
            "psllw": ("shl", 16),
            "pslld": ("shl", 32),
            "psllq": ("shl", 64),
            "psrlw": ("lshr", 16),
            "psrld": ("lshr", 32),
            "psrlq": ("lshr", 64),
            "psraw": ("ashr", 16),
            "psrad": ("ashr", 32),
        }
        if mnemonic in lane_shifts and len(operands) >= 2:
            operation, lane_width = lane_shifts[mnemonic]
            value = self._read_operand(state, operands[0], instruction)
            count = self._read_operand(state, operands[1], instruction)
            result = _mmx_lane_shift(operation, value, count, lane_width)
            self._write_operand(
                state, operands[0], result, instruction, emitted, collect=collect
            )
            return True
        return False

    def _execute_simd(
        self,
        state: _State,
        instruction: Instruction,
        decoded: _Decoded,
        emitted: list[str],
        *,
        collect: bool,
    ) -> bool:
        mnemonic = decoded.mnemonic
        operands = decoded.operands
        if not any(
            operand.kind == "reg" and re.fullmatch(r"xmm\d+", operand.reg)
            for operand in operands
        ):
            return False
        if any(operand.size * 8 > 128 for operand in operands):
            return False

        move_instructions = {
            "movaps",
            "movups",
            "movapd",
            "movupd",
            "movdqa",
            "movdqu",
            "vmovaps",
            "vmovups",
            "vmovapd",
            "vmovupd",
            "vmovdqa",
            "vmovdqu",
        }
        if mnemonic in move_instructions and len(operands) >= 2:
            source = self._read_operand(state, operands[1], instruction)
            self._write_operand(
                state, operands[0], source, instruction, emitted, collect=collect
            )
            return True

        if mnemonic == "cvtdq2pd" and len(operands) >= 2:
            source = self._read_operand(state, operands[-1], instruction)
            value = _simd_cvtdq2pd(source)
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return True

        int_to_float = {
            "cvtsi2ss": 32,
            "cvtsi2sd": 64,
            "vcvtsi2ss": 32,
            "vcvtsi2sd": 64,
        }
        if mnemonic in int_to_float and len(operands) >= 2:
            output_bits = int_to_float[mnemonic]
            source = self._read_operand(state, operands[-1], instruction)
            old_destination = self._read_operand(
                state, operands[1] if len(operands) >= 3 else operands[0], instruction
            )
            value = _simd_convert_int_to_fp(
                source, operands[-1].size * 8, output_bits, old_destination
            )
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return True

        float_conversions = {
            "cvtsd2ss": (64, 32),
            "cvtss2sd": (32, 64),
            "vcvtsd2ss": (64, 32),
            "vcvtss2sd": (32, 64),
        }
        if mnemonic in float_conversions and len(operands) >= 2:
            input_bits, output_bits = float_conversions[mnemonic]
            source = self._read_operand(state, operands[-1], instruction)
            old_destination = self._read_operand(
                state, operands[1] if len(operands) >= 3 else operands[0], instruction
            )
            value = _simd_convert_fp(source, input_bits, output_bits, old_destination)
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return True

        if mnemonic in {"minsd", "maxsd", "vminsd", "vmaxsd"} and len(operands) >= 2:
            operation = "min" if "min" in mnemonic else "max"
            if len(operands) >= 3:
                left_operand, right_operand = operands[-2], operands[-1]
                old_destination = self._read_operand(state, operands[1], instruction)
            else:
                left_operand, right_operand = operands[0], operands[1]
                old_destination = self._read_operand(state, operands[0], instruction)
            value = _simd_scalar_minmax(
                operation,
                self._read_operand(state, left_operand, instruction),
                self._read_operand(state, right_operand, instruction),
                old_destination,
            )
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return True

        scalar_float_ops = {
            "addss": ("add", 32),
            "subss": ("sub", 32),
            "mulss": ("mul", 32),
            "divss": ("div", 32),
            "addsd": ("add", 64),
            "subsd": ("sub", 64),
            "mulsd": ("mul", 64),
            "divsd": ("div", 64),
            "vaddss": ("add", 32),
            "vsubss": ("sub", 32),
            "vmulss": ("mul", 32),
            "vdivss": ("div", 32),
            "vaddsd": ("add", 64),
            "vsubsd": ("sub", 64),
            "vmulsd": ("mul", 64),
            "vdivsd": ("div", 64),
        }
        if mnemonic in scalar_float_ops and len(operands) >= 2:
            operation, scalar_bits = scalar_float_ops[mnemonic]
            if len(operands) >= 3:
                left_operand, right_operand = operands[-2], operands[-1]
                old_destination = self._read_operand(state, operands[1], instruction)
            else:
                left_operand, right_operand = operands[0], operands[1]
                old_destination = self._read_operand(state, operands[0], instruction)
            left = self._read_operand(state, left_operand, instruction)
            right = self._read_operand(state, right_operand, instruction)
            value = _simd_scalar_binary(
                operation, left, right, old_destination, scalar_bits
            )
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return True

        if (
            mnemonic in {"comiss", "ucomiss", "comisd", "ucomisd"}
            and len(operands) >= 2
        ):
            scalar_bits = 32 if mnemonic.endswith("ss") else 64
            left = _simd_float(
                self._read_operand(state, operands[0], instruction), scalar_bits
            )
            right = _simd_float(
                self._read_operand(state, operands[1], instruction), scalar_bits
            )
            state.flags = _Flags("fp_compare", left, right, None, 80)
            return True

        if mnemonic in {"cvttsd2si", "cvttss2si"} and len(operands) >= 2:
            input_bits = 32 if mnemonic == "cvttss2si" else 64
            source = self._read_operand(state, operands[-1], instruction)
            value = _simd_cvtt_to_int(
                source, operands[0].size * 8, input_bits=input_bits
            )
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return True

        if mnemonic in {"movd", "movq"} and len(operands) >= 2:
            source = self._read_operand(state, operands[1], instruction)
            if operands[0].kind == "reg" and re.fullmatch(r"xmm\d+", operands[0].reg):
                low_bits = 32 if mnemonic == "movd" else 64
                value = _zero_extend(_truncate(source, low_bits), 128)
            elif operands[1].kind == "reg" and re.fullmatch(r"xmm\d+", operands[1].reg):
                low_bits = 32 if mnemonic == "movd" else 64
                value = _truncate(source, low_bits)
            else:
                return False
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return True

        if mnemonic in {"movss", "movsd"} and len(operands) >= 2:
            scalar_bits = 32 if mnemonic == "movss" else 64
            source = self._read_operand(state, operands[1], instruction)
            if operands[0].kind == "reg" and re.fullmatch(r"xmm\d+", operands[0].reg):
                if operands[1].kind == "reg" and re.fullmatch(
                    r"xmm\d+", operands[1].reg
                ):
                    old = self._read_operand(state, operands[0], instruction)
                    value = _insert_bits(
                        old, _truncate(source, scalar_bits), 0, scalar_bits, 128
                    )
                else:
                    value = _zero_extend(_truncate(source, scalar_bits), 128)
                self._write_operand(
                    state, operands[0], value, instruction, emitted, collect=collect
                )
            elif operands[0].kind == "mem":
                self._write_operand(
                    state,
                    operands[0],
                    _truncate(source, scalar_bits),
                    instruction,
                    emitted,
                    collect=collect,
                )
            return True

        if (
            mnemonic
            in {
                "xorps",
                "xorpd",
                "andps",
                "andpd",
                "andnps",
                "andnpd",
                "orps",
                "orpd",
                "pxor",
                "pand",
                "pandn",
                "por",
            }
            and len(operands) >= 2
        ):
            if len(operands) >= 3:
                left_operand, right_operand = operands[-2], operands[-1]
            else:
                left_operand, right_operand = operands[0], operands[1]
            left = self._read_operand(state, left_operand, instruction)
            right = self._read_operand(state, right_operand, instruction)
            if mnemonic in {"andnps", "andnpd", "pandn"}:
                left = _unary("bitnot", left, 128)
                left, right = right, left
                operation = "and"
            else:
                operation = (
                    "xor"
                    if mnemonic in {"xorps", "xorpd", "pxor"}
                    else "and"
                    if mnemonic in {"andps", "andpd", "pand"}
                    else "or"
                )
            value = _binary(operation, left, right, 128)
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return True

        lane_operations = {
            "paddb": ("add", 8),
            "paddw": ("add", 16),
            "paddd": ("add", 32),
            "paddq": ("add", 64),
            "psubb": ("sub", 8),
            "psubw": ("sub", 16),
            "psubd": ("sub", 32),
            "psubq": ("sub", 64),
            "pmullw": ("mul", 16),
            "pcmpeqb": ("eq", 8),
            "pcmpeqw": ("eq", 16),
            "pcmpeqd": ("eq", 32),
            "pcmpeqq": ("eq", 64),
            "pminsb": ("smin", 8),
            "pminsw": ("smin", 16),
            "pminsd": ("smin", 32),
            "pmaxsb": ("smax", 8),
            "pmaxsw": ("smax", 16),
            "pmaxsd": ("smax", 32),
        }
        if mnemonic in lane_operations and len(operands) >= 2:
            operation, lane_width = lane_operations[mnemonic]
            if len(operands) >= 3:
                left_operand, right_operand = operands[-2], operands[-1]
            else:
                left_operand, right_operand = operands[0], operands[1]
            left = self._read_operand(state, left_operand, instruction)
            right = self._read_operand(state, right_operand, instruction)
            value = _simd_lane_binary(operation, left, right, lane_width)
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return True

        if mnemonic in {"pshufd", "vpshufd"} and len(operands) >= 3:
            source = self._read_operand(state, operands[-2], instruction)
            immediate = self._read_operand(state, operands[-1], instruction)
            value = _simd_shuffle_dwords(source, immediate)
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return True

        if mnemonic in {"psrldq", "pslldq"} and len(operands) >= 2:
            source = self._read_operand(state, operands[0], instruction)
            immediate = self._read_operand(state, operands[1], instruction)
            value = _simd_shift_bytes(source, immediate, left=mnemonic == "pslldq")
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return True
        return False

    def _execute_string(
        self,
        state: _State,
        instruction: Instruction,
        decoded: _Decoded,
        emitted: list[str],
        *,
        collect: bool,
    ) -> bool:
        mnemonic = decoded.mnemonic
        if any(
            operand.kind == "reg" and re.fullmatch(r"xmm\d+", operand.reg)
            for operand in decoded.operands
        ):
            return False
        sizes = {"b": 1, "w": 2, "d": 4, "q": 8}
        operation = next(
            (
                prefix
                for prefix in ("movs", "stos", "lods", "cmps", "scas")
                if mnemonic.startswith(prefix)
            ),
            None,
        )
        if operation is None:
            return False
        suffix = mnemonic[len(operation) :]
        if suffix not in sizes:
            return False
        repeated = bool(decoded.repeat)
        if not repeated and not any(
            operand.kind == "mem" for operand in decoded.operands
        ):
            return False

        size = sizes[suffix]
        width = size * 8
        type_name = _unsigned_type(width)
        source_register = "rsi" if self.bits == 64 else "esi"
        destination_register = "rdi" if self.bits == 64 else "edi"
        count_register = "rcx" if self.bits == 64 else "ecx"
        source = self._read_register(state, source_register, instruction)
        destination = self._read_register(state, destination_register, instruction)
        count = self._read_register(state, count_register, instruction)
        direction = state.direction_flag
        delta = (
            _binary("mul", count, _const(size, self.bits), self.bits)
            if repeated
            else _const(size, self.bits)
        )
        direction_sign = -1 if direction == 1 else 1

        def emit(text: str) -> None:
            if collect:
                emitted.append(f"/* {instruction.address:#x} */ {text}")

        def advance(register: str, pointer: _Expr, amount: _Expr) -> None:
            if direction is None:
                value = _var(
                    f"after_string_{instruction.address:X}_{register}", self.bits
                )
            elif direction_sign < 0:
                value = _binary("sub", pointer, amount, self.bits)
            else:
                value = _binary("add", pointer, amount, self.bits)
            self._write_register(state, register, value, mark_argument=False)

        if operation == "movs":
            if repeated and direction is not None:
                index = f"rep_i_{instruction.address:X}"
                sign = "-" if direction == 1 else ""
                emit(
                    f"for (uint32_t {index} = 0; {index} < {self._render(count)}; ++{index}) "
                    f"(({type_name} *)({self._render(destination)}))[{sign}{index}] = "
                    f"((const {type_name} *)({self._render(source)}))[{sign}{index}];"
                )
            elif repeated:
                emit("/* repeated string copy; direction flag is unknown */")
            else:
                loaded = _Expr("load", args=(source,), bits=width)
                emit(
                    f"*({type_name} *)({self._render(destination)}) = {self._render(loaded)};"
                )
            advance(source_register, source, delta)
            advance(destination_register, destination, delta)
            if repeated:
                self._write_register(
                    state, count_register, _const(0, self.bits), mark_argument=False
                )
            return True

        if operation == "stos":
            accumulator = {8: "al", 16: "ax", 32: "eax", 64: "rax"}[width]
            value = self._read_register(state, accumulator, instruction)
            if repeated and direction is None:
                emit("/* repeated stos; direction flag is unknown */")
            elif repeated:
                index = f"rep_i_{instruction.address:X}"
                sign = "-" if direction == 1 else ""
                emit(
                    f"for (uint32_t {index} = 0; {index} < {self._render(count)}; ++{index}) "
                    f"(({type_name} *)({self._render(destination)}))[{sign}{index}] = "
                    f"({_unsigned_type(width)})({self._render(value)});"
                )
            else:
                emit(
                    f"*({type_name} *)({self._render(destination)}) = "
                    f"({_unsigned_type(width)})({self._render(value)});"
                )
            advance(destination_register, destination, delta)
            if repeated:
                self._write_register(
                    state, count_register, _const(0, self.bits), mark_argument=False
                )
            return True

        if operation == "lods":
            accumulator = {8: "al", 16: "ax", 32: "eax", 64: "rax"}[width]
            if repeated:
                value = _var(f"rep_lods_result_{instruction.address:X}", width)
                self._write_register(state, accumulator, value, mark_argument=False)
                self._write_register(
                    state, count_register, _const(0, self.bits), mark_argument=False
                )
            else:
                value = _Expr("load", args=(source,), bits=width)
                self._write_register(state, accumulator, value, mark_argument=False)
            advance(source_register, source, delta)
            if repeated:
                emit("/* repeated lods result is represented symbolically */")
            return True

        if operation in {"cmps", "scas"}:
            if operation == "cmps":
                left = _Expr("load", args=(source,), bits=width)
                right = _Expr("load", args=(destination,), bits=width)
            else:
                accumulator = {8: "al", 16: "ax", 32: "eax", 64: "rax"}[width]
                left = self._read_register(state, accumulator, instruction)
                right = _Expr("load", args=(destination,), bits=width)
            if repeated:
                left = _var(f"rep_{operation}_left_{instruction.address:X}", width)
                right = _var(f"rep_{operation}_right_{instruction.address:X}", width)
                count_after = _var(
                    f"rep_{operation}_count_{instruction.address:X}", self.bits
                )
                self._write_register(
                    state, count_register, count_after, mark_argument=False
                )
                emit(
                    f"/* {decoded.repeat} {mnemonic}: final comparison and remaining count are symbolic */"
                )
            elif operation == "cmps":
                advance(source_register, source, delta)
                advance(destination_register, destination, delta)
            else:
                advance(destination_register, destination, delta)
            difference = _binary("sub", left, right, width)
            arithmetic_flags = self._arithmetic_flags(
                "sub", left, right, difference, width
            )
            state.flags = _Flags(
                "cmp",
                left,
                right,
                difference,
                width,
                arithmetic_flags.carry,
                arithmetic_flags.overflow,
                arithmetic_flags.parity,
            )
            if repeated and operation == "cmps":
                self._write_register(
                    state,
                    source_register,
                    _var(f"rep_cmps_src_{instruction.address:X}", self.bits),
                    mark_argument=False,
                )
                self._write_register(
                    state,
                    destination_register,
                    _var(f"rep_cmps_dst_{instruction.address:X}", self.bits),
                    mark_argument=False,
                )
            elif repeated and operation == "scas":
                self._write_register(
                    state,
                    destination_register,
                    _var(f"rep_scas_dst_{instruction.address:X}", self.bits),
                    mark_argument=False,
                )
            return True
        return False

    def _read_memory(
        self, state: _State, operand: _Operand, instruction: Instruction
    ) -> _Expr:
        width = operand.size * 8
        offset = self._stack_offset(operand, state)
        if offset is not None:
            exact = state.stack.get((offset, width))
            if exact is not None:
                return _resize(exact, width)
            for (stored_offset, stored_width), stored_value in state.stack.items():
                if stored_offset == offset and stored_width > width:
                    return _truncate(stored_value, width)
            return self._initial_stack_value(offset, width)
        return _Expr(
            "load",
            args=(self._memory_address(operand, state, instruction),),
            bits=width,
        )

    def _memory_address(
        self, operand: _Operand, state: _State, instruction: Instruction
    ) -> _Expr:
        offset = self._stack_offset(operand, state)
        if offset is not None:
            return _Expr(
                "address", value=("stack", self._stack_name(offset)), bits=self.bits
            )
        if operand.base == "rip":
            address = _Expr(
                "address",
                value=("absolute", instruction.next_address + operand.displacement),
                bits=self.bits,
            )
            if operand.segment:
                return _Expr(
                    "segment_address",
                    value=operand.segment,
                    args=(address,),
                    bits=self.bits,
                )
            return address
        if not operand.base and not operand.index:
            address = _Expr(
                "address",
                value=("absolute", operand.displacement & ((1 << self.bits) - 1)),
                bits=self.bits,
            )
            if operand.segment:
                return _Expr(
                    "segment_address",
                    value=operand.segment,
                    args=(address,),
                    bits=self.bits,
                )
            return address

        parts: list[_Expr] = []
        if operand.base:
            parts.append(self._read_register(state, operand.base, instruction))
        if operand.index:
            index = self._read_register(state, operand.index, instruction)
            if operand.scale != 1:
                index = _binary(
                    "mul", index, _const(operand.scale, self.bits), self.bits
                )
            parts.append(index)
        if operand.displacement:
            parts.append(_const(operand.displacement, self.bits))
        address = parts[0] if parts else _const(operand.displacement, self.bits)
        for part in parts[1:]:
            address = _binary("add", address, part, self.bits)
        if operand.segment:
            address = _Expr(
                "segment_address",
                value=operand.segment,
                args=(address,),
                bits=self.bits,
            )
        return address

    def _execute_instruction(
        self,
        state: _State,
        instruction: Instruction,
        *,
        collect: bool,
    ) -> list[str]:
        decoded = self._decode(instruction)
        mnemonic = decoded.mnemonic
        operands = decoded.operands
        emitted: list[str] = []

        def emit(text: str) -> None:
            if collect:
                emitted.append(f"/* {instruction.address:#x} */ {text}")

        if mnemonic in {"nop", "endbr32", "endbr64", "fnop"}:
            return emitted

        if instruction.is_return or mnemonic.startswith("ret"):
            return_family = self._calling_convention.integer_return_registers[0]
            initial_result = self._initial_register(return_family)
            result = state.registers.get(return_family, initial_result)
            call_result = (
                result.kind == "var"
                and re.fullmatch(r"call_result_[0-9A-F]+", str(result.value))
                and self._return_value_used_by_callers is False
            )
            xmm_result = None
            xmm_type = None
            if self.bits == 64:
                xmm_value = state.xmm_registers.get("xmm0")
                if (
                    xmm_value is not None
                    and xmm_value != self._initial_xmm_register("xmm0")
                    and not (
                        xmm_value.kind == "var"
                        and str(xmm_value.value).startswith("after_call_")
                    )
                    and self._callers_use_register_before_write(self._function, "xmm0")
                    is True
                ):
                    xmm_width = state.xmm_widths.get("xmm0", 128)
                    if xmm_width in {32, 64}:
                        xmm_result = _simd_float(
                            _extract(xmm_value, 0, xmm_width), xmm_width
                        )
                        xmm_type = "float" if xmm_width == 32 else "double"
                    else:
                        xmm_result = xmm_value
                        xmm_type = "__m128"
            wide_result = None
            wide_type = None
            if result != initial_result and not call_result:
                high_register = next(
                    (
                        register
                        for register in self._calling_convention.integer_return_registers[
                            1:
                        ]
                        if register in {"edx", "rdx"}
                    ),
                    None,
                )
                if high_register is not None:
                    initial_high = self._initial_register(high_register)
                    high_value = state.registers.get(high_register, initial_high)
                    if (
                        high_value != initial_high
                        and self._callers_use_register_pair_before_write(
                            self._function, return_family, high_register
                        )
                        is True
                    ):
                        low_bits = self.bits
                        wide_result = _concat(high_value, result, low_bits)
                        wide_type = "uint64_t" if self.bits == 32 else "__uint128_t"
            if xmm_result is not None:
                emit(f"return {self._render(xmm_result)};")
                if collect:
                    self._saw_value_return = True
                    self._return_type = xmm_type
            elif wide_result is not None:
                emit(f"return {self._render(wide_result)};")
                if collect:
                    self._saw_value_return = True
                    self._return_type = wide_type
            elif result != initial_result and not call_result:
                emit(f"return {self._render(result)};")
                if collect:
                    self._saw_value_return = True
                    self._return_type = f"uint{self.bits}_t"
            elif self.bits == 32 and state.x87_stack:
                emit(f"return {self._render(state.x87_stack[0])};")
                if collect:
                    self._saw_value_return = True
                    self._return_type = "long double"
            else:
                emit("return;")
            return emitted

        if instruction.is_trap or mnemonic in {"ud2", "int3", "hlt"}:
            emit("__builtin_trap();")
            return emitted

        if self._execute_x87(state, instruction, decoded, emitted, collect=collect):
            return emitted
        if self._execute_string(state, instruction, decoded, emitted, collect=collect):
            return emitted
        if self._execute_mmx(state, instruction, decoded, emitted, collect=collect):
            return emitted
        if self._execute_simd(state, instruction, decoded, emitted, collect=collect):
            return emitted

        if instruction.is_call:
            # call $+5 / call next is a common PC-acquisition idiom, not a real call.
            if instruction.branch_target == instruction.next_address:
                self._push(state, _const(instruction.next_address, self.bits))
                return emitted
            target_function = (
                self.analyzer.functions.get(instruction.branch_target)
                if instruction.branch_target is not None
                else None
            )
            call_convention = (
                calling_convention_for(self.analyzer, target_function)
                if target_function is not None
                else default_calling_convention(self.analyzer)
            )
            arguments = self._call_arguments(
                state, instruction, instruction.branch_target
            )
            target = self._call_target(instruction, operands, state)
            call = f"{target}({', '.join(self._render(arg) for arg in arguments)})"
            no_return = bool(self.analyzer._call_is_noreturn(instruction))
            if no_return:
                emit(f"{call}; /* no retorna */")
            else:
                result_name = f"call_result_{instruction.address:X}"
                emit(f"{result_name} = {call};")
                if collect:
                    self._call_results.add(result_name)
                result = _var(result_name, self.bits)
                return_family = call_convention.integer_return_registers[0]
                state.registers[return_family] = result
            caller_saved = call_convention.caller_saved_registers
            return_family = call_convention.integer_return_registers[0]
            for family in caller_saved:
                if family.startswith("xmm"):
                    state.xmm_registers[family] = _var(
                        f"after_call_{instruction.address:X}_{family}", 128
                    )
                    state.xmm_widths[family] = 128
                elif family != return_family:
                    state.registers[family] = _var(
                        f"after_call_{instruction.address:X}_{family}", self.bits
                    )
            if "ftol" in target.lower():
                self._x87_pop(state)
            else:
                state.x87_stack.clear()
                state.x87_flags = None
                state.x87_status_flags = None
            state.mmx_registers.clear()
            state.flags = None
            state.dirty_arguments.clear()
            purge_size = None
            if instruction.branch_target is not None:
                callee = self.analyzer.functions.get(instruction.branch_target)
                if callee is not None:
                    purge_size = callee.stack_purge_size
                    if (
                        purge_size is None
                        and callee.is_thunk
                        and callee.thunk_target is not None
                    ):
                        target_function = self.analyzer.functions.get(
                            callee.thunk_target
                        )
                        if target_function is not None:
                            purge_size = target_function.stack_purge_size
            if purge_size and state.stack_delta is not None:
                old_delta = state.stack_delta
                state.stack_delta += purge_size
                state.stack = {
                    (offset, width): value
                    for (offset, width), value in state.stack.items()
                    if not old_delta <= offset < state.stack_delta
                }
            state.pending_pushes.clear()
            return emitted

        if mnemonic in {"push", "pushf", "pushfd", "pushfq"}:
            value = (
                self._read_operand(state, operands[0], instruction)
                if operands
                else _var("flags", self.bits)
            )
            push_offset = (
                state.stack_delta - self.pointer_size
                if state.stack_delta is not None
                else None
            )
            self._push(state, value)
            is_saved_register = (
                instruction.address in self._saved_register_push_addresses
            )
            if not is_saved_register:
                state.pending_pushes.append((push_offset, value))
            return emitted

        if mnemonic in {"pop", "popf", "popfd", "popfq"}:
            pop_offset = state.stack_delta
            value = self._pop(state)
            for index in range(len(state.pending_pushes) - 1, -1, -1):
                pushed_offset, _pushed_value = state.pending_pushes[index]
                if pop_offset is None or pushed_offset == pop_offset:
                    state.pending_pushes.pop(index)
                    break
            if operands:
                self._write_operand(
                    state, operands[0], value, instruction, emitted, collect=collect
                )
            return emitted

        if mnemonic == "leave":
            if state.frame_delta is not None:
                state.stack_delta = state.frame_delta
            restored = self._pop(state)
            self._write_register(
                state,
                "rbp" if self.bits == 64 else "ebp",
                restored,
                mark_argument=False,
            )
            state.frame_delta = None
            return emitted

        if mnemonic in {"mov", "movabs"} and len(operands) >= 2:
            value = self._read_operand(state, operands[1], instruction)
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            self._update_frame_pointer(state, operands[0], operands[1])
            self._update_stack_pointer_move(state, operands[0], operands[1], value)
            return emitted

        if mnemonic in {"movzx", "movzb", "movzw"} and len(operands) >= 2:
            value = self._read_operand(state, operands[1], instruction)
            value = _zero_extend(value, operands[0].size * 8)
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return emitted

        if (
            mnemonic in {"movsx", "movsxd", "movsb", "movsw", "movsl"}
            and len(operands) >= 2
        ):
            value = self._read_operand(state, operands[1], instruction)
            value = _sign_extend(value, operands[0].size * 8)
            self._write_operand(
                state, operands[0], value, instruction, emitted, collect=collect
            )
            return emitted

        if mnemonic == "lea" and len(operands) >= 2 and operands[1].kind == "mem":
            address = self._memory_address(operands[1], state, instruction)
            self._write_operand(
                state, operands[0], address, instruction, emitted, collect=collect
            )
            destination = (
                self._register_alias(operands[0].reg)
                if operands[0].kind == "reg"
                else None
            )
            if destination and destination[0] == ("rsp" if self.bits == 64 else "esp"):
                state.stack_delta = self._stack_offset(operands[1], state)
            elif destination and destination[0] == (
                "rbp" if self.bits == 64 else "ebp"
            ):
                state.frame_delta = self._stack_offset(operands[1], state)
            return emitted

        if mnemonic in {"cdqe", "cltq", "cwde", "cbw"}:
            if mnemonic in {"cdqe", "cltq"}:
                value = _sign_extend(self._read_register(state, "eax", instruction), 64)
                self._write_register(state, "rax", value)
            elif mnemonic == "cwde":
                value = _sign_extend(self._read_register(state, "ax", instruction), 32)
                self._write_register(state, "eax", value)
            else:
                value = _sign_extend(self._read_register(state, "al", instruction), 16)
                self._write_register(state, "ax", value)
            return emitted

        if mnemonic in {"cwd", "cdq", "cqo"}:
            low_reg, high_reg, width = {
                "cwd": ("ax", "dx", 16),
                "cdq": ("eax", "edx", 32),
                "cqo": ("rax", "rdx", 64),
            }[mnemonic]
            low_value = self._read_register(state, low_reg, instruction)
            negative = _binary("slt", low_value, _const(0, width), 1)
            high_value = _select(
                negative, _const(_mask(width), width), _const(0, width), width
            )
            self._write_register(state, high_reg, high_value)
            return emitted

        if mnemonic in {"div", "idiv"} or (
            mnemonic in {"mul", "imul"} and len(operands) == 1
        ):
            if not operands:
                return emitted
            width = operands[0].size * 8
            if width <= 8:
                low_reg, high_reg = "al", "ah"
            elif width <= 16:
                low_reg, high_reg = "ax", "dx"
            elif width <= 32:
                low_reg, high_reg = "eax", "edx"
            else:
                low_reg, high_reg = "rax", "rdx"
            operand = self._read_operand(state, operands[0], instruction)
            if mnemonic in {"mul", "imul"}:
                accumulator = self._read_register(state, low_reg, instruction)
                if mnemonic == "imul":
                    lhs = _sign_extend(accumulator, width * 2)
                    rhs = _sign_extend(operand, width * 2)
                else:
                    lhs = _zero_extend(accumulator, width * 2)
                    rhs = _zero_extend(operand, width * 2)
                product = _binary("mul", lhs, rhs, width * 2)
                low = _truncate(product, width)
                high = _extract(product, width, width)
                self._write_register(state, low_reg, low)
                self._write_register(state, high_reg, high)
                if mnemonic == "imul":
                    expected_high = _select(
                        _extract(low, width - 1, 1),
                        _const(_mask(width), width),
                        _const(0, width),
                        width,
                    )
                    overflow = _binary("ne", high, expected_high, 1)
                else:
                    overflow = _binary("ne", high, _const(0, width), 1)
                state.flags = _Flags(
                    "mul", accumulator, operand, low, width, overflow, overflow
                )
            else:
                low = self._read_register(state, low_reg, instruction)
                high = self._read_register(state, high_reg, instruction)
                dividend = _concat(high, low, width)
                if mnemonic == "idiv":
                    divisor = _sign_extend(operand, width * 2)
                    operation = "sdiv"
                    remainder_operation = "srem"
                else:
                    divisor = _zero_extend(operand, width * 2)
                    operation = "udiv"
                    remainder_operation = "urem"
                quotient = _binary(operation, dividend, divisor, width)
                remainder = _binary(remainder_operation, dividend, divisor, width)
                self._write_register(state, low_reg, quotient)
                self._write_register(state, high_reg, remainder)
                state.flags = None
            return emitted

        if mnemonic == "cmp" and len(operands) >= 2:
            left = self._read_operand(state, operands[0], instruction)
            right = self._read_operand(state, operands[1], instruction)
            width = operands[0].size * 8
            result = _binary("sub", left, right, width)
            arithmetic_flags = self._arithmetic_flags("sub", left, right, result, width)
            state.flags = _Flags(
                "cmp",
                left,
                right,
                result,
                width,
                arithmetic_flags.carry,
                arithmetic_flags.overflow,
                arithmetic_flags.parity,
            )
            return emitted

        if mnemonic == "test" and len(operands) >= 2:
            left = self._read_operand(state, operands[0], instruction)
            right = self._read_operand(state, operands[1], instruction)
            width = operands[0].size * 8
            result = _binary("and", left, right, width)
            state.flags = self._logical_flags("test", left, right, result, width)
            return emitted

        if mnemonic in {"adc", "sbb"} and len(operands) >= 2:
            left = self._read_operand(state, operands[0], instruction)
            right = self._read_operand(state, operands[1], instruction)
            carry_in = self._carry_flag(state.flags)
            width = operands[0].size * 8
            operation = "add" if mnemonic == "adc" else "sub"
            partial = _binary(operation, left, right, width)
            result = _binary(operation, partial, carry_in, width)
            self._write_operand(
                state, operands[0], result, instruction, emitted, collect=collect
            )
            state.flags = self._arithmetic_flags(
                mnemonic, left, right, result, width, carry_in=carry_in
            )
            alias = (
                self._register_alias(operands[0].reg)
                if operands[0].kind == "reg"
                else None
            )
            stack_family = "rsp" if self.bits == 64 else "esp"
            if alias is not None and alias[0] == stack_family:
                if carry_in.kind == "const":
                    adjustment = _binary(operation, right, carry_in, width)
                    self._update_stack_pointer(
                        state, operands[0], operation, adjustment
                    )
                else:
                    state.stack_delta = None
                    state.pending_pushes.clear()
            return emitted

        if mnemonic in {"inc", "dec", "neg", "not"} and operands:
            old = self._read_operand(state, operands[0], instruction)
            width = operands[0].size * 8
            if mnemonic == "inc":
                one = _const(1, width)
                result = _binary("add", old, one, width)
                state.flags = self._arithmetic_flags(
                    "add",
                    old,
                    one,
                    result,
                    width,
                    preserve_carry=self._carry_flag(state.flags),
                )
            elif mnemonic == "dec":
                one = _const(1, width)
                result = _binary("sub", old, one, width)
                state.flags = self._arithmetic_flags(
                    "sub",
                    old,
                    one,
                    result,
                    width,
                    preserve_carry=self._carry_flag(state.flags),
                )
            elif mnemonic == "neg":
                result = _unary("neg", old, width)
                arithmetic_flags = self._arithmetic_flags(
                    "sub", _const(0, width), old, result, width
                )
                state.flags = _Flags(
                    "neg",
                    _const(0, width),
                    old,
                    result,
                    width,
                    _binary("ne", old, _const(0, width), 1),
                    _binary("eq", old, _const(1 << (width - 1), width), 1),
                    arithmetic_flags.parity,
                )
            else:
                result = _unary("bitnot", old, width)
            self._write_operand(
                state, operands[0], result, instruction, emitted, collect=collect
            )
            return emitted

        arithmetic = {
            "add": "add",
            "sub": "sub",
            "and": "and",
            "or": "or",
            "xor": "xor",
            "imul": "mul",
            "mul": "mul",
            "shl": "shl",
            "sal": "shl",
            "shr": "lshr",
            "sar": "ashr",
        }
        if mnemonic in arithmetic and len(operands) >= 2:
            width = operands[0].size * 8
            if mnemonic == "imul" and len(operands) >= 3:
                left = self._read_operand(state, operands[1], instruction)
                right = self._read_operand(state, operands[2], instruction)
            else:
                left = self._read_operand(state, operands[0], instruction)
                right = self._read_operand(state, operands[1], instruction)
            operation = arithmetic[mnemonic]
            result = _binary(operation, left, right, width)
            self._write_operand(
                state, operands[0], result, instruction, emitted, collect=collect
            )
            if operation in {"add", "sub"}:
                state.flags = self._arithmetic_flags(
                    operation, left, right, result, width
                )
            elif mnemonic == "imul":
                wide_left = _sign_extend(left, width * 2)
                wide_right = _sign_extend(right, width * 2)
                product = _binary("mul", wide_left, wide_right, width * 2)
                high = _extract(product, width, width)
                expected_high = _select(
                    _extract(result, width - 1, 1),
                    _const(_mask(width), width),
                    _const(0, width),
                    width,
                )
                overflow = _binary("ne", high, expected_high, 1)
                state.flags = _Flags(
                    "mul", left, right, result, width, overflow, overflow
                )
            else:
                state.flags = self._logical_flags("logic", left, right, result, width)
            self._update_stack_pointer(state, operands[0], operation, right)
            return emitted

        if mnemonic.startswith("cmov") and len(operands) >= 2:
            condition_name = "j" + mnemonic[4:]
            condition = self._condition_expression(
                condition_name, state.flags, state, instruction
            )
            old = self._read_operand(state, operands[0], instruction)
            value = self._read_operand(state, operands[1], instruction)
            self._write_operand(
                state,
                operands[0],
                _select(condition, value, old, operands[0].size * 8),
                instruction,
                emitted,
                collect=collect,
            )
            return emitted

        if mnemonic.startswith("set") and len(operands) == 1:
            condition = self._condition_expression(
                "j" + mnemonic[3:], state.flags, state, instruction
            )
            self._write_operand(
                state, operands[0], condition, instruction, emitted, collect=collect
            )
            return emitted

        if mnemonic in {
            "loop",
            "loope",
            "loopz",
            "loopne",
            "loopnz",
            "jrcxz",
            "jecxz",
            "jcxz",
        }:
            counter = "rcx" if self.bits == 64 else "ecx"
            if mnemonic.startswith("loop"):
                old = self._read_register(state, counter, instruction)
                new = _binary("sub", old, _const(1, self.bits), self.bits)
                self._write_register(state, counter, new)
            return emitted

        # Branches are emitted after the block has been evaluated, so the
        # condition can use the final flag expression from its predecessor ops.
        if instruction.is_jump:
            return emitted

        if mnemonic == "cld":
            state.direction_flag = 0
            return emitted
        if mnemonic == "std":
            state.direction_flag = 1
            return emitted
        if mnemonic == "lahf":
            self._write_register(
                state,
                "ah",
                _var(f"flags_ah_{instruction.address:X}", 8),
                mark_argument=False,
            )
            return emitted
        if mnemonic == "sahf":
            state.flags = state.x87_status_flags
            state.x87_status_flags = None
            return emitted

        # Explicitly invalidate an unsupported destination instead of silently
        # reusing a stale symbolic value.  The comment is a deliberate boundary
        # between known semantics and instructions awaiting a future lift.
        state.flags = None
        if operands:
            destination = operands[0]
            unknown = _var(
                f"unknown_{mnemonic}_{instruction.address:X}", destination.size * 8
            )
            if destination.kind == "reg":
                self._write_operand(
                    state, destination, unknown, instruction, emitted, collect=False
                )
            elif destination.kind == "mem":
                offset = self._stack_offset(destination, state)
                if offset is not None:
                    state.stack[(offset, destination.size * 8)] = unknown
        if collect:
            emit(
                f"/* unsupported x86 instruction: {mnemonic} {instruction.op_str}; effects are approximate */"
            )
        return emitted

    def _run_block(
        self, block: BasicBlock, input_state: _State, *, collect: bool
    ) -> tuple[_State, list[str]]:
        state = input_state.copy()
        statements: list[str] = []
        for instruction in block.instructions:
            statements.extend(
                self._execute_instruction(state, instruction, collect=collect)
            )
        return state, statements

    def _analyze_dataflow(self) -> None:
        if not self._blocks:
            return
        queued: set[int] = set()
        worklist: deque[int] = deque()

        def enqueue(start: int) -> None:
            if start in self._block_by_start and start not in queued:
                queued.add(start)
                worklist.append(start)

        entry = self._function.address if self._function else self._blocks[0].start
        enqueue(entry)
        for block in self._blocks:
            if not self._predecessors.get(block.start):
                enqueue(block.start)

        max_updates = max(256, len(self._blocks) * 128)
        updates = 0
        while worklist and updates < max_updates:
            start = worklist.popleft()
            queued.discard(start)
            block = self._block_by_start[start]
            input_state = self._input_state_for(start, record_phis=False)
            if input_state is None:
                continue
            output_state, _ = self._run_block(block, input_state, collect=False)
            previous = self._out_states.get(start)
            if previous == output_state:
                continue
            self._out_states[start] = output_state
            updates += 1
            for successor in block.successors:
                enqueue(successor)

        # Bodies can include a disconnected recovery fragment. Analyze it as a
        # fresh root rather than leaving it with no symbolic state.
        for block in self._blocks:
            if block.start in self._out_states:
                continue
            input_state = self._input_state_for(block.start, record_phis=False)
            if input_state is None:
                input_state = _State()
            output_state, _ = self._run_block(block, input_state, collect=False)
            self._out_states[block.start] = output_state

    def _input_state_for(self, start: int, *, record_phis: bool) -> _State | None:
        sources: list[_State] = []
        if self._function is not None and start == self._function.address:
            sources.append(_State())
        sources.extend(
            self._out_states[pred]
            for pred in self._predecessors.get(start, ())
            if pred in self._out_states
        )
        if not sources:
            if not self._predecessors.get(start):
                sources.append(_State())
            else:
                return None
        if len(sources) == 1:
            return sources[0].copy()
        return self._merge_states(start, sources, record_phis=record_phis)

    def _merge_states(
        self, block_start: int, sources: list[_State], *, record_phis: bool
    ) -> _State:
        merged = _State()
        register_names = set().union(*(source.registers.keys() for source in sources))
        for family in sorted(register_names):
            values = tuple(
                source.registers.get(family, self._initial_register(family))
                for source in sources
            )
            merged.registers[family] = self._merge_value(
                block_start, family, values, record_phis=record_phis
            )

        stack_keys = set().union(*(source.stack.keys() for source in sources))
        for offset, width in sorted(stack_keys):
            values = tuple(
                source.stack.get(
                    (offset, width), self._initial_stack_value(offset, width)
                )
                for source in sources
            )
            location = f"stack_{offset:+X}_{width}"
            merged.stack[(offset, width)] = self._merge_value(
                block_start, location, values, record_phis=record_phis
            )

        mmx_names = set().union(*(source.mmx_registers.keys() for source in sources))
        for name in sorted(mmx_names):
            values = tuple(
                source.mmx_registers.get(name, _var(f"in_{name}", 64))
                for source in sources
            )
            merged.mmx_registers[name] = self._merge_value(
                block_start, name, values, record_phis=record_phis
            )

        xmm_names = set().union(*(source.xmm_registers.keys() for source in sources))
        for name in sorted(xmm_names):
            values = tuple(
                source.xmm_registers.get(name, self._initial_xmm_register(name))
                for source in sources
            )
            merged.xmm_registers[name] = self._merge_value(
                block_start, name, values, record_phis=record_phis
            )
            widths = {source.xmm_widths.get(name, 128) for source in sources}
            merged.xmm_widths[name] = widths.pop() if len(widths) == 1 else 128

        x87_depths = {len(source.x87_stack) for source in sources}
        if len(x87_depths) == 1:
            for index in range(next(iter(x87_depths))):
                values = tuple(source.x87_stack[index] for source in sources)
                merged.x87_stack.append(
                    self._merge_value(
                        block_start,
                        f"x87_st{index}",
                        values,
                        record_phis=record_phis,
                    )
                )
        else:
            depth = max(x87_depths, default=0)
            merged.x87_stack = [
                _var(f"unknown_x87_st{index}_loc_{block_start:X}", 80)
                for index in range(depth)
            ]
        x87_flags = {source.x87_flags for source in sources}
        merged.x87_flags = x87_flags.pop() if len(x87_flags) == 1 else None
        x87_status_flags = {source.x87_status_flags for source in sources}
        merged.x87_status_flags = (
            x87_status_flags.pop() if len(x87_status_flags) == 1 else None
        )
        direction_flags = {source.direction_flag for source in sources}
        merged.direction_flag = (
            direction_flags.pop() if len(direction_flags) == 1 else None
        )

        stack_deltas = {source.stack_delta for source in sources}
        merged.stack_delta = stack_deltas.pop() if len(stack_deltas) == 1 else None
        frame_deltas = {source.frame_delta for source in sources}
        merged.frame_delta = frame_deltas.pop() if len(frame_deltas) == 1 else None
        flags = {source.flags for source in sources}
        merged.flags = flags.pop() if len(flags) == 1 else None
        merged.dirty_arguments = set().union(
            *(source.dirty_arguments for source in sources)
        )
        pending = [tuple(source.pending_pushes) for source in sources]
        if all(len(values) == len(pending[0]) for values in pending):
            merged_pending: list[tuple[int | None, _Expr]] = []
            for index in range(len(pending[0])):
                offsets = [values[index][0] for values in pending]
                if any(offset != offsets[0] for offset in offsets[1:]):
                    merged_pending = []
                    break
                values = tuple(source.pending_pushes[index][1] for source in sources)
                if all(value == values[0] for value in values[1:]):
                    value = values[0]
                else:
                    value = self._merge_value(
                        block_start,
                        f"pending_push_{index}",
                        values,
                        record_phis=record_phis,
                    )
                merged_pending.append((offsets[0], value))
            merged.pending_pushes = merged_pending
        else:
            merged.pending_pushes = []
        return merged

    def _merge_value(
        self,
        block_start: int,
        location: str,
        values: tuple[_Expr, ...],
        *,
        record_phis: bool,
    ) -> _Expr:
        if not values or all(value == values[0] for value in values[1:]):
            return values[0] if values else _var("undefined", self.bits)
        name = self._phi_name(block_start, location)
        if record_phis:
            self._phi_inputs[(block_start, location)] = values
        return _var(name, max((value.bits for value in values), default=self.bits))

    def _phi_name(self, block_start: int, location: str) -> str:
        safe = _c_identifier(location)
        return f"phi_{safe}_loc_{block_start:X}"

    def _render_terminator(self, block: BasicBlock, state: _State) -> list[str]:
        if not block.instructions:
            return []
        last = block.instructions[-1]
        lines: list[str] = []
        decoded = self._decode(last)
        if last.is_return or last.is_trap:
            return lines
        if last.is_call and self.analyzer._call_is_noreturn(last):
            return lines
        if last.is_conditional_jump:
            condition = self._condition_expression(
                decoded.mnemonic, state.flags, state, last
            )
            target = last.branch_target
            if target is None:
                lines.append(f"/* conditional indirect branch: {last.op_str} */")
            else:
                lines.append(
                    f"if ({self._render(condition)}) goto {self._branch_target(target)};"
                )
            fallthrough = last.next_address
            if fallthrough in self._block_by_start:
                lines.append(f"goto loc_{fallthrough:X};")
            return lines
        if last.is_jump:
            if last.computed_targets:
                targets = ", ".join(
                    self._branch_target(target) for target in last.computed_targets
                )
                indirect = (
                    self._read_operand(state, decoded.operands[0], last)
                    if decoded.operands
                    else _var("computed_target", self.bits)
                )
                lines.append(f"/* computed jump to one of: {targets} */")
                lines.append(f"goto *({self._render(indirect)});")
            elif last.branch_target is not None:
                target_function = self.analyzer.functions.get(last.branch_target)
                if (
                    target_function is not None
                    and target_function.address != self._function.address
                ):
                    target_name = _c_identifier(target_function.name)
                    if target_function.no_return:
                        lines.append(f"{target_name}(); /* tail jump; no retorna */")
                    else:
                        lines.append(
                            f"return {target_name}(/* argumentos reenviados, firma desconocida */);"
                        )
                        self._saw_value_return = True
                else:
                    lines.append(f"goto {self._branch_target(last.branch_target)};")
            else:
                indirect = (
                    self._read_operand(state, decoded.operands[0], last)
                    if decoded.operands
                    else _var("computed_target", self.bits)
                )
                lines.append(f"goto *({self._render(indirect)});")
            return lines

        successors = [
            target for target in block.successors if target in self._block_by_start
        ]
        if successors:
            lines.append(f"goto loc_{successors[0]:X};")
        elif last.next_address in self._block_by_start:
            lines.append(f"goto loc_{last.next_address:X};")
        elif not last.is_return:
            lines.append("/* recovered control flow ends here */")
        return lines

    def _branch_target(self, address: int) -> str:
        if address in self._block_by_start:
            return f"loc_{address:X}"
        name = self.analyzer.name_at(address)
        return _c_identifier(name) if name else f"loc_{address:X}"

    def _condition_expression(
        self,
        mnemonic: str,
        flags: _Flags | None,
        state: _State | None = None,
        instruction: Instruction | None = None,
    ) -> _Expr:
        mnemonic = mnemonic.lower()
        state = state or _State()
        instruction = instruction or Instruction(0, 0, b"", "", "")
        if mnemonic in {"jcxz", "jecxz", "jrcxz"}:
            counter = {
                "jcxz": "cx",
                "jecxz": "ecx",
                "jrcxz": "rcx" if self.bits == 64 else "ecx",
            }[mnemonic]
            value = self._read_register(state, counter, instruction)
            return _binary("eq", value, _const(0, self.bits), 1)
        if mnemonic.startswith("loop"):
            counter = "rcx" if self.bits == 64 else "ecx"
            count = self._read_register(state, counter, instruction)
            not_zero = _binary("ne", count, _const(0, self.bits), 1)
            if mnemonic in {"loope", "loopz"}:
                return _binary(
                    "land",
                    not_zero,
                    self._condition_expression("je", flags, state, instruction),
                    1,
                )
            if mnemonic in {"loopne", "loopnz"}:
                return _binary(
                    "land",
                    not_zero,
                    self._condition_expression("jne", flags, state, instruction),
                    1,
                )
            return not_zero
        if flags is None:
            return _var(f"condition_{mnemonic}", 1)

        if (
            flags.kind == "fp_compare"
            and flags.left is not None
            and flags.right is not None
        ):
            return _Expr(
                "fp_condition", mnemonic, args=(flags.left, flags.right), bits=1
            )

        condition = self._CONDITION_ALIASES.get(mnemonic)
        if (
            flags.kind == "cmp"
            and flags.left is not None
            and flags.right is not None
            and condition
            in {"eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"}
        ):
            return _binary(condition, flags.left, flags.right, 1)

        if flags.result is not None:
            zero = _binary("eq", flags.result, _const(0, flags.bits), 1)
            sign = _binary("slt", flags.result, _const(0, flags.bits), 1)
            carry = self._carry_flag(flags)
            overflow = flags.overflow or _var("flag_of", 1)
            parity = flags.parity or _var("flag_pf", 1)
            if mnemonic in {"je", "jz"}:
                return zero
            if mnemonic in {"jne", "jnz"}:
                return _unary("boolnot", zero, 1)
            if mnemonic in {"jb", "jnae", "jc"}:
                return carry
            if mnemonic in {"jae", "jnb", "jnc"}:
                return _unary("boolnot", carry, 1)
            if mnemonic in {"jbe", "jna"}:
                return _binary("lor", carry, zero, 1)
            if mnemonic in {"ja", "jnbe"}:
                return _binary(
                    "land", _unary("boolnot", carry, 1), _unary("boolnot", zero, 1), 1
                )
            signed_less = _binary("xor", sign, overflow, 1)
            if mnemonic in {"jl", "jnge"}:
                return signed_less
            if mnemonic in {"jge", "jnl"}:
                return _unary("boolnot", signed_less, 1)
            if mnemonic in {"jle", "jng"}:
                return _binary("lor", zero, signed_less, 1)
            if mnemonic in {"jg", "jnle"}:
                return _binary(
                    "land",
                    _unary("boolnot", zero, 1),
                    _unary("boolnot", signed_less, 1),
                    1,
                )
            if mnemonic == "js":
                return sign
            if mnemonic == "jns":
                return _unary("boolnot", sign, 1)
            if mnemonic == "jo":
                return overflow
            if mnemonic == "jno":
                return _unary("boolnot", overflow, 1)
            if mnemonic in {"jp", "jpe"}:
                return parity
            if mnemonic in {"jnp", "jpo"}:
                return _unary("boolnot", parity, 1)
        return _var(f"condition_{mnemonic}", 1)

    def _call_arguments(
        self,
        state: _State,
        instruction: Instruction,
        target_address: int | None,
    ) -> list[_Expr]:
        target_function = (
            self.analyzer.functions.get(target_address)
            if target_address is not None
            else None
        )
        if (
            target_function is not None
            and target_function.is_thunk
            and target_function.thunk_target is not None
        ):
            target_function = self.analyzer.functions.get(target_function.thunk_target)
        convention = (
            calling_convention_for(self.analyzer, target_function)
            if target_function is not None
            else default_calling_convention(self.analyzer)
        )
        args: list[_Expr] = []

        def is_available(register: str) -> bool:
            return register in state.dirty_arguments

        def read_argument(register: str) -> _Expr:
            if register.startswith("xmm"):
                self._note_vector_parameter_type(register, instruction)
                value = state.xmm_registers.get(
                    register, self._initial_xmm_register(register)
                )
                return _resize(value, 128)
            return self._read_register(state, register, instruction)

        def typed_argument(register: str, parameter_name: str) -> _Expr:
            value = read_argument(register)
            signature = target_function.signature if target_function else None
            parameter_type = (
                next(
                    (
                        parameter.type_name
                        for parameter in signature.parameters
                        if parameter.name == parameter_name
                    ),
                    None,
                )
                if signature
                else None
            )
            if register.startswith("xmm") and parameter_type in {"float", "double"}:
                width = 32 if parameter_type == "float" else 64
                return _simd_float(_extract(value, 0, width), width)
            return value

        if convention.register_argument_groups:
            active_groups = [
                index
                for index, group in enumerate(convention.register_argument_groups)
                if any(is_available(register) for register in group)
            ]
            for index in range(max(active_groups, default=-1) + 1):
                group = convention.register_argument_groups[index]
                selected = next(
                    (
                        register
                        for register in group
                        if register in state.dirty_arguments
                    ),
                    None,
                )
                if selected is None:
                    selected = next(
                        (
                            register
                            for register in group
                            if register in self._argument_index
                            or register in self._vector_argument_index
                        ),
                        group[0],
                    )
                if selected not in convention.hidden_argument_registers:
                    args.append(typed_argument(selected, f"param_{index + 1}"))
        else:
            active = [
                index
                for index, register in enumerate(convention.argument_registers)
                if is_available(register)
            ]
            if active:
                last_index = max(active)
                args.extend(
                    typed_argument(register, f"param_{index + 1}")
                    for index, register in enumerate(
                        convention.argument_registers[: last_index + 1]
                    )
                    if register not in convention.hidden_argument_registers
                )
            vector_active = [
                index
                for index, register in enumerate(convention.vector_argument_registers)
                if is_available(register)
            ]
            if vector_active:
                last_index = max(vector_active)
                args.extend(
                    typed_argument(
                        register,
                        convention.vector_parameter_name(register)
                        or f"float_param_{index + 1}",
                    )
                    for index, register in enumerate(
                        convention.vector_argument_registers[: last_index + 1]
                    )
                )

        stack_count = len(state.pending_pushes)
        if (
            self.bits == 32
            and target_function is not None
            and target_function.stack_purge_size
        ):
            stack_count = max(
                stack_count,
                target_function.stack_purge_size // self.pointer_size,
            )
        if stack_count == 0 and self.bits == 32:
            cleanup = self._callsite_stack_cleanup_bytes(instruction)
            if cleanup:
                stack_count = cleanup // self.pointer_size

        stack_values: list[_Expr] = []
        if state.stack_delta is not None:
            outgoing_offset = (
                state.stack_delta + convention.stack_argument_offset - self.pointer_size
            )
            if stack_count:
                for index in range(stack_count):
                    offset = outgoing_offset + index * self.pointer_size
                    value = state.stack.get((offset, self.bits))
                    if value is None:
                        if index < len(state.pending_pushes):
                            value = list(reversed(state.pending_pushes))[index][1]
                        else:
                            break
                    stack_values.append(value)
            elif self.bits == 64:
                # Stack-passed x64 arguments are written into the outgoing
                # argument area; Windows reserves 32 home bytes first.
                for index in range(16):
                    offset = outgoing_offset + index * self.pointer_size
                    value = state.stack.get((offset, self.bits))
                    if value is None:
                        break
                    stack_values.append(value)
        if not stack_values and state.pending_pushes:
            stack_values = [value for _offset, value in reversed(state.pending_pushes)]
        args.extend(stack_values)
        return args

    def _callsite_stack_cleanup_bytes(self, instruction: Instruction) -> int:
        if self._function is None:
            return 0
        call_index = next(
            (
                index
                for index, candidate in enumerate(self._function.instructions)
                if candidate.address == instruction.address
            ),
            None,
        )
        if call_index is None:
            return 0
        cleanup = 0
        for candidate in self._function.instructions[call_index + 1 : call_index + 5]:
            decoded = self._decode(candidate)
            operands = decoded.operands
            if (
                decoded.mnemonic == "add"
                and len(operands) >= 2
                and (
                    operands[0].kind == "reg"
                    and operands[0].reg == ("esp" if self.bits == 32 else "rsp")
                    and operands[1].kind == "imm"
                    and operands[1].imm > 0
                )
            ):
                return cleanup + operands[1].imm
            if decoded.mnemonic == "lea" and len(operands) >= 2:
                destination, source = operands[:2]
                if (
                    destination.kind == "reg"
                    and destination.reg == ("esp" if self.bits == 32 else "rsp")
                    and source.kind == "mem"
                    and source.base == destination.reg
                    and not source.index
                    and source.displacement > 0
                ):
                    return cleanup + source.displacement
            if decoded.mnemonic == "pop":
                cleanup += self.pointer_size
                continue
            if candidate.is_call or candidate.is_jump or candidate.is_return:
                break
        return cleanup

    def _call_target(
        self,
        instruction: Instruction,
        operands: tuple[_Operand, ...],
        state: _State,
    ) -> str:
        target = instruction.branch_target
        if target is not None:
            name = self.analyzer.name_at(target)
            if name:
                return _c_identifier(name)
            return f"loc_{target:X}"
        for reference in instruction.memory_references:
            import_name = getattr(self.analyzer.image, "import_slots", {}).get(
                reference
            )
            if not import_name:
                import_name = getattr(self.analyzer, "_import_name_by_address", {}).get(
                    reference
                )
            if not import_name:
                symbols = getattr(self.analyzer, "_imports_by_address", {}).get(
                    reference, ()
                )
                import_name = symbols[0].name if symbols else None
            if import_name:
                return _c_identifier(import_name)
        if operands:
            target_expr = self._read_operand(state, operands[0], instruction)
            return f"(*({self._render(target_expr)}))"
        return "indirect_call"

    def _is_saved_register(self, name: str) -> bool:
        alias = self._register_alias(name)
        if alias is None:
            return False
        return alias[0] in self._calling_convention.callee_saved_registers

    def _find_prologue_saved_register_pushes(self, function: Function) -> set[int]:
        restored = {
            decoded.operands[0].reg
            for candidate in function.instructions
            if (decoded := self._decode(candidate)).mnemonic == "pop"
            and decoded.operands
            and decoded.operands[0].kind == "reg"
        }
        saved_pushes: set[int] = set()
        for instruction in function.instructions:
            decoded = self._decode(instruction)
            if instruction.is_call or instruction.is_jump or instruction.is_return:
                break
            if (
                decoded.mnemonic == "push"
                and decoded.operands
                and decoded.operands[0].kind == "reg"
                and decoded.operands[0].reg in restored
                and self._is_saved_register(decoded.operands[0].reg)
            ):
                saved_pushes.add(instruction.address)
        return saved_pushes

    def _push(self, state: _State, value: _Expr) -> None:
        if state.stack_delta is not None:
            state.stack_delta -= self.pointer_size
            state.stack[(state.stack_delta, self.bits)] = _resize(value, self.bits)

    def _pop(self, state: _State) -> _Expr:
        offset = state.stack_delta
        if offset is None:
            result = _var("unknown_stack_value", self.bits)
        else:
            result = state.stack.get(
                (offset, self.bits), self._initial_stack_value(offset, self.bits)
            )
            state.stack_delta += self.pointer_size
        return result

    def _update_frame_pointer(
        self, state: _State, destination: _Operand, source: _Operand
    ) -> None:
        if destination.kind != "reg":
            return
        family_info = self._register_alias(destination.reg)
        if family_info is None or family_info[0] != (
            "rbp" if self.bits == 64 else "ebp"
        ):
            return
        if source.kind == "reg" and source.reg in {"rsp", "esp"}:
            state.frame_delta = state.stack_delta
        elif source.kind == "reg" and source.reg in {"rbp", "ebp"}:
            pass
        else:
            state.frame_delta = None

    def _update_stack_pointer_move(
        self, state: _State, destination: _Operand, source: _Operand, value: _Expr
    ) -> None:
        if destination.kind != "reg":
            return
        alias = self._register_alias(destination.reg)
        if alias is None or alias[0] != ("rsp" if self.bits == 64 else "esp"):
            return
        if source.kind == "reg" and source.reg in {"rbp", "ebp"}:
            state.stack_delta = state.frame_delta
        elif value.kind == "stack_pointer":
            state.stack_delta = int(value.value)
        else:
            state.stack_delta = None

    def _update_stack_pointer(
        self, state: _State, destination: _Operand, operation: str, right: _Expr
    ) -> None:
        if destination.kind != "reg":
            return
        alias = self._register_alias(destination.reg)
        if alias is None or alias[0] != ("rsp" if self.bits == 64 else "esp"):
            return
        if right.kind != "const" or state.stack_delta is None:
            state.stack_delta = None
            return
        value = int(right.value)
        if operation == "add":
            state.stack_delta += value
            state.pending_pushes.clear()
        elif operation == "sub":
            state.stack_delta -= value
            state.pending_pushes.clear()
        else:
            state.stack_delta = None
            state.pending_pushes.clear()

    def _render(self, expression: _Expr) -> str:
        kind = expression.kind
        if kind == "const":
            return _format_integer(int(expression.value))
        if kind == "var":
            return str(expression.value)
        if kind == "fp_const":
            value = _format_float(float(expression.value))
            return f"{value}L"
        if kind in {"fp_load", "fp_int_load"}:
            width, _is_integer = expression.value
            pointer = self._render(expression.args[0])
            type_name = (
                _signed_type(width) if kind == "fp_int_load" else _float_type(width)
            )
            loaded = f"*({type_name} *)({pointer})"
            if kind == "fp_int_load":
                return f"((long double)({loaded}))"
            return loaded
        if kind == "fp_cast":
            return f"(({_float_type(int(expression.value))})({self._render(expression.args[0])}))"
        if kind == "fp_from_int":
            return f"((long double)({self._render(expression.args[0])}))"
        if kind == "fp_to_int":
            width, truncate = expression.value
            conversion = "truncl" if truncate else "nearbyintl"
            return f"(({_signed_type(width)}){conversion}({self._render(expression.args[0])}))"
        if kind == "fp_unary":
            operand = self._render(expression.args[0])
            operation = str(expression.value)
            if operation == "neg":
                return f"(-({operand}))"
            functions = {
                "abs": "fabsl",
                "sqrt": "sqrtl",
                "sin": "sinl",
                "cos": "cosl",
                "tan": "tanl",
                "round": "nearbyintl",
            }
            if operation == "exp2m1":
                return f"(exp2l({operand}) - 1.0L)"
            return f"{functions.get(operation, operation)}({operand})"
        if kind == "fp_op":
            left, right = expression.args
            operator = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(
                str(expression.value), str(expression.value)
            )
            return f"({self._render(left)} {operator} {self._render(right)})"
        if kind == "fp_func":
            first, second = expression.args
            left, right = self._render(first), self._render(second)
            operation = str(expression.value)
            if operation in {"fyl2x", "fyl2xp1"}:
                return f"({left} * log2l({right}))"
            if operation == "fpatan":
                return f"atan2l({left}, {right})"
            if operation == "fprem":
                return f"fmodl({left}, {right})"
            if operation == "fprem1":
                return f"remainderl({left}, {right})"
            if operation == "fscale":
                return f"({left} * exp2l(truncl({right})))"
        if kind == "fp_condition":
            left, right = (self._render(arg) for arg in expression.args)
            unordered = f"(isnan({left}) || isnan({right}))"
            condition = str(expression.value)
            if condition in {"jp", "jpe"}:
                return unordered
            if condition in {"jnp", "jpo"}:
                return f"(!{unordered})"
            expressions = {
                "je": f"({unordered} || ({left} == {right}))",
                "jz": f"({unordered} || ({left} == {right}))",
                "jne": f"(!{unordered} && ({left} != {right}))",
                "jnz": f"(!{unordered} && ({left} != {right}))",
                "jb": f"({unordered} || ({left} < {right}))",
                "jc": f"({unordered} || ({left} < {right}))",
                "jnae": f"({unordered} || ({left} < {right}))",
                "jbe": f"({unordered} || ({left} <= {right}))",
                "jna": f"({unordered} || ({left} <= {right}))",
                "ja": f"(!{unordered} && ({left} > {right}))",
                "jnbe": f"(!{unordered} && ({left} > {right}))",
                "jae": f"(!{unordered} && ({left} >= {right}))",
                "jnb": f"(!{unordered} && ({left} >= {right}))",
                "jnc": f"(!{unordered} && ({left} >= {right}))",
            }
            return expressions.get(condition, f"condition_{condition}")
        if kind == "concat":
            high, low = expression.args
            lower_bits = int(expression.value)
            return (
                f"((({_unsigned_type(expression.bits)})({self._render(high)}) "
                f"<< {lower_bits}) | ({_unsigned_type(lower_bits)})("
                f"{self._render(low)}))"
            )
        if kind == "mmx_lanes":
            operation, lane_width = expression.value
            left, right = expression.args
            lane_mask = _mask(lane_width)
            operator = {"add": "+", "sub": "-", "mul": "*"}[operation]
            lanes = []
            for index in range(64 // lane_width):
                shift = index * lane_width
                left_lane = (
                    f"(({_unsigned_type(lane_width)})({self._render(left)} >> {shift}))"
                )
                right_lane = f"(({_unsigned_type(lane_width)})({self._render(right)} >> {shift}))"
                value = f"(({left_lane} {operator} {right_lane}) & {lane_mask:#x})"
                lanes.append(f"((uint64_t)({value}) << {shift})")
            return "(" + " | ".join(lanes) + ")"
        if kind == "mmx_shift":
            operation, lane_width = expression.value
            vector, count = expression.args
            value = self._render(vector)
            amount = self._render(count)
            if lane_width == 64:
                if operation == "shl":
                    return f"(({amount} >= 64) ? 0 : ((uint64_t)({value}) << {amount}))"
                if operation == "lshr":
                    return f"(({amount} >= 64) ? 0 : ((uint64_t)({value}) >> {amount}))"
                return f"(({amount} >= 64) ? ((int64_t)({value}) >> 63) : ((int64_t)({value}) >> {amount}))"
            lane_mask = _mask(lane_width)
            lanes = []
            for index in range(64 // lane_width):
                shift = index * lane_width
                lane = f"(({_unsigned_type(lane_width)})({value} >> {shift}))"
                if operation == "shl":
                    shifted = f"({lane} << {amount})"
                elif operation == "lshr":
                    shifted = f"({lane} >> {amount})"
                else:
                    signed_type = "int16_t" if lane_width == 16 else "int32_t"
                    shifted = f"((({signed_type}){lane}) >> {amount})"
                if operation == "ashr":
                    signed_type = "int16_t" if lane_width == 16 else "int32_t"
                    overflow_value = (
                        f"((({signed_type}){lane} < 0) ? {lane_mask:#x} : 0)"
                    )
                else:
                    overflow_value = "0"
                lanes.append(
                    f"((uint64_t)(({amount} >= {lane_width}) ? {overflow_value} : "
                    f"({shifted} & {lane_mask:#x})) << {shift})"
                )
            return "(" + " | ".join(lanes) + ")"
        if kind == "simd_float":
            width = int(expression.value)
            raw = self._render(expression.args[0])
            if width == 32:
                return f"((union {{ uint32_t i; float f; }}){{ .i = (uint32_t)({raw}) }}).f"
            return (
                f"((union {{ uint64_t i; double f; }}){{ .i = (uint64_t)({raw}) }}).f"
            )
        if kind == "simd_fp_binary":
            width = int(expression.value)
            result, old_destination = expression.args
            result_type = "float" if width == 32 else "double"
            integer_type = "uint32_t" if width == 32 else "uint64_t"
            shift = width
            raw_result = (
                f"((union {{ {result_type} f; {integer_type} i; }})"
                f"{{ .f = ({result_type})({self._render(result)}) }}).i"
            )
            old = self._render(old_destination)
            return (
                f"((((__uint128_t)({old}) >> {shift}) << {shift}) "
                f"| (__uint128_t)({raw_result}))"
            )
        if kind == "simd_convert_int_fp":
            input_bits, output_bits = expression.value
            source, old_destination = expression.args
            signed_type = _signed_type(input_bits)
            float_type = "float" if output_bits == 32 else "double"
            integer_type = "uint32_t" if output_bits == 32 else "uint64_t"
            integer = f"(({signed_type})({self._render(source)}))"
            packed = (
                f"((union {{ {float_type} f; {integer_type} i; }})"
                f"{{ .f = ({float_type})({integer}) }}).i"
            )
            old = self._render(old_destination)
            return (
                f"((((__uint128_t)({old}) >> {output_bits}) << {output_bits}) "
                f"| (__uint128_t)({packed}))"
            )
        if kind == "simd_convert_fp":
            _input_bits, output_bits = expression.value
            source, old_destination = expression.args
            float_type = "float" if output_bits == 32 else "double"
            integer_type = "uint32_t" if output_bits == 32 else "uint64_t"
            packed = (
                f"((union {{ {float_type} f; {integer_type} i; }})"
                f"{{ .f = ({float_type})({self._render(source)}) }}).i"
            )
            old = self._render(old_destination)
            return (
                f"((((__uint128_t)({old}) >> {output_bits}) << {output_bits}) "
                f"| (__uint128_t)({packed}))"
            )
        if kind == "simd_minmax":
            operation = str(expression.value)
            left, right, old_destination = expression.args
            lhs, rhs = self._render(left), self._render(right)
            comparator = "<" if operation == "min" else ">"
            float_result = (
                f"((isnan({lhs}) || isnan({rhs}) || !({lhs} {comparator} {rhs})) "
                f"? {rhs} : {lhs})"
            )
            raw_result = (
                "((union { double f; uint64_t i; })"
                f"{{ .f = (double)({float_result}) }}).i"
            )
            old = self._render(old_destination)
            return (
                f"((((__uint128_t)({old}) >> 64) << 64) | (__uint128_t)({raw_result}))"
            )
        if kind == "simd_cvtdq2pd":
            source = self._render(expression.args[0])
            lanes = []
            for index in range(2):
                shift = index * 32
                integer = f"((int32_t)(uint32_t)({source} >> {shift}))"
                raw = (
                    "((union { double f; uint64_t i; })"
                    f"{{ .f = (double)({integer}) }}).i"
                )
                lanes.append(f"((__uint128_t)({raw}) << {index * 64})")
            return "(" + " | ".join(lanes) + ")"
        if kind == "simd_cvtt_to_int":
            bits, _input_bits = expression.value
            value = self._render(expression.args[0])
            minimum = -(1 << (bits - 1))
            upper_bound = 1 << (bits - 1)
            invalid = 1 << (bits - 1)
            return (
                f"(({_unsigned_type(bits)})((!isfinite({value}) || "
                f"{value} < ({minimum}) || {value} >= ({upper_bound})) "
                f"? {invalid:#x} : ({_signed_type(bits)})trunc({value})))"
            )
        if kind == "simd_lanes":
            operation, lane_width = expression.value
            left, right = expression.args
            lane_mask = _mask(lane_width)
            lanes = []
            for index in range(expression.bits // lane_width):
                shift = index * lane_width
                lhs = (
                    f"(({_unsigned_type(lane_width)})({self._render(left)} >> {shift}))"
                )
                rhs = f"(({_unsigned_type(lane_width)})({self._render(right)} >> {shift}))"
                if operation == "eq":
                    lane = f"(({lhs} == {rhs}) ? {lane_mask:#x} : 0)"
                elif operation in {"smin", "smax"}:
                    signed_type = _signed_type(lane_width)
                    comparator = "<" if operation == "smin" else ">"
                    lane = (
                        f"((({signed_type}){lhs} {comparator} ({signed_type}){rhs}) "
                        f"? {lhs} : {rhs})"
                    )
                else:
                    operator = {"add": "+", "sub": "-", "mul": "*"}[operation]
                    lane = f"(({lhs} {operator} {rhs}) & {lane_mask:#x})"
                lanes.append(f"((__uint128_t)({lane}) << {shift})")
            return "(" + " | ".join(lanes) + ")"
        if kind == "simd_shuffle_dwords":
            vector = expression.args[0]
            shuffle = int(expression.value)
            lanes = []
            for index in range(4):
                source_index = (shuffle >> (index * 2)) & 3
                source_shift = source_index * 32
                result_shift = index * 32
                lane = f"((uint32_t)({self._render(vector)} >> {source_shift}))"
                lanes.append(f"((__uint128_t)({lane}) << {result_shift})")
            return "(" + " | ".join(lanes) + ")"
        if kind == "simd_shift_bytes":
            vector, amount = expression.args
            shift = f"({self._render(amount)} * 8)"
            value = self._render(vector)
            if expression.value == "left":
                shifted = f"((__uint128_t)({value}) << {shift})"
            else:
                shifted = f"((__uint128_t)({value}) >> {shift})"
            return f"(({self._render(amount)} >= 16) ? 0 : {shifted})"
        if kind == "parity":
            value = self._render(expression.args[0])
            return (
                f"((0x6996U >> (((uint8_t)({value}) ^ "
                f"((uint8_t)({value}) >> 4)) & 0xf)) & 1) == 0)"
            )
        if kind == "stack_pointer":
            delta = int(expression.value)
            base = "in_rsp" if self.bits == 64 else "in_esp"
            if delta == 0:
                return base
            operator = "+" if delta > 0 else "-"
            return f"({base} {operator} {abs(delta):#x})"
        if kind == "address":
            address_kind, value = expression.value
            if address_kind == "stack":
                return f"&{value}"
            address = int(value)
            symbol = self.analyzer.name_at(address)
            if symbol:
                return f"&{_c_identifier(symbol)}"
            text = self._string_by_address.get(address)
            if text is not None:
                return f"{json.dumps(text, ensure_ascii=True)}"
            return f"{address:#x}"
        if kind == "load":
            pointer = self._render(expression.args[0])
            return f"*({_memory_type(expression.bits)} *)({pointer})"
        if kind == "segment_address":
            return f"{expression.value}_base + ({self._render(expression.args[0])})"
        if kind == "zext":
            source_bits = expression.args[0].bits or expression.bits
            return f"(({_unsigned_type(expression.bits)})({_unsigned_type(source_bits)})({self._render(expression.args[0])}))"
        if kind == "sext":
            source_bits = expression.args[0].bits or expression.bits
            return f"(({_signed_type(expression.bits)})({_signed_type(source_bits)})({self._render(expression.args[0])}))"
        if kind == "trunc":
            return f"(({_unsigned_type(expression.bits)})({self._render(expression.args[0])}))"
        if kind == "extract":
            value, offset = expression.args
            if int(offset.value) == 0:
                return f"(({_unsigned_type(expression.bits)})({self._render(value)}))"
            mask = (1 << expression.bits) - 1
            return f"((({self._render(value)}) >> {self._render(offset)}) & {mask:#x})"
        if kind == "insert":
            old, new, offset = expression.args
            inserted = new.bits
            shift = int(offset.value)
            mask = ((1 << inserted) - 1) << shift
            return (
                f"((({self._render(old)}) & ~{mask:#x}) | "
                f"((({self._render(new)}) & {(1 << inserted) - 1:#x}) << {shift}))"
            )
        if kind == "unary":
            operator = {"neg": "-", "bitnot": "~", "boolnot": "!"}.get(
                str(expression.value), ""
            )
            return f"({operator}{self._render(expression.args[0])})"
        if kind == "select":
            condition, yes, no = expression.args
            return f"({self._render(condition)} ? {self._render(yes)} : {self._render(no)})"
        if kind == "op":
            op = str(expression.value)
            left, right = expression.args
            left_text = self._render(left)
            right_text = self._render(right)
            if op in {"ult", "ule", "ugt", "uge"}:
                operator = self._BINARY_OPERATORS[op]
                return (
                    f"(({_unsigned_type(max(left.bits, right.bits))})({left_text}) "
                    f"{operator} ({_unsigned_type(max(left.bits, right.bits))})({right_text}))"
                )
            if op in {"slt", "sle", "sgt", "sge"}:
                operator = self._BINARY_OPERATORS[op]
                return (
                    f"(({_signed_type(max(left.bits, right.bits))})({left_text}) "
                    f"{operator} ({_signed_type(max(left.bits, right.bits))})({right_text}))"
                )
            operator = self._BINARY_OPERATORS.get(op, op)
            return f"({left_text} {operator} {right_text})"
        return f"unknown_expr_{kind}"

    def _prune_dead_phis(self, body: list[str]) -> list[str]:
        phi_values = {
            self._phi_name(block_start, location): values
            for (block_start, location), values in self._phi_inputs.items()
        }
        if not phi_values:
            return body

        phi_names = set(phi_values)
        live: set[str] = set()
        for line in body:
            if any(
                re.match(rf"^\s*{re.escape(name)}\s*=\s*phi\(", line)
                for name in phi_names
            ):
                continue
            live.update(re.findall(r"\bphi_[A-Za-z0-9_]+\b", line))
        live.intersection_update(phi_names)

        pending = list(live)
        while pending:
            name = pending.pop()
            for value in phi_values.get(name, ()):
                for referenced in _expression_variables(value):
                    if referenced in phi_names and referenced not in live:
                        live.add(referenced)
                        pending.append(referenced)

        result = []
        for line in body:
            definition = next(
                (
                    name
                    for name in phi_names
                    if re.match(rf"^\s*{re.escape(name)}\s*=\s*phi\(", line)
                ),
                None,
            )
            if definition is None or definition in live:
                result.append(line)
        return result

    def _drop_unused_call_results(self, body: list[str]) -> list[str]:
        result = list(body)
        for name in self._call_results:
            occurrences = sum(
                len(re.findall(rf"\b{re.escape(name)}\b", line)) for line in result
            )
            if occurrences != 1:
                continue
            assignment = re.compile(rf"\b{re.escape(name)}\s*=\s*")
            for index, line in enumerate(result):
                if assignment.search(line):
                    result[index] = assignment.sub("", line, count=1)
                    break
        return result

    def _parameter_specs(self, body: list[str]) -> list[FunctionParameter]:
        used = {
            int(match.group(1))
            for line in body
            for match in re.finditer(r"\bparam_(\d+)\b", line)
        }
        used.update(
            int(match.group(1))
            for name in self._used_parameter_names
            if (match := re.fullmatch(r"param_(\d+)", name))
        )
        float_used = {
            int(match.group(1))
            for line in body
            for match in re.finditer(r"\bfloat_param_(\d+)\b", line)
        }
        float_used.update(
            int(match.group(1))
            for name in self._used_parameter_names
            if (match := re.fullmatch(r"float_param_(\d+)", name))
        )
        if not used and not float_used:
            return []
        parameters: list[FunctionParameter] = []
        if used:
            for index in range(1, max(used) + 1):
                name = f"param_{index}"
                type_name = self._parameter_types.get(name, _unsigned_type(self.bits))
                parameters.append(FunctionParameter(name, type_name))
        if float_used:
            for index in range(1, max(float_used) + 1):
                name = f"float_param_{index}"
                type_name = self._parameter_types.get(name, "double")
                parameters.append(FunctionParameter(name, type_name))
        return parameters

    def _local_declarations(self, body: list[str]) -> list[str]:
        names: set[str] = set()
        patterns = (
            r"\b(call_result_[0-9A-F]+)\b",
            r"\b(phi_[A-Za-z0-9_]+)\b",
            r"\b(local_[0-9A-F]+)\b",
            r"\b(stack_[0-9A-F]+)\b",
            r"\b(stack_arg_\d+)\b",
            r"\b(home_slot_\d+)\b",
            r"\b(stack_return_address)\b",
            r"\b(in_[A-Za-z0-9_]+)\b",
            r"\b(after_call_[0-9A-F]+_[A-Za-z0-9_]+)\b",
            r"\b(condition_[A-Za-z0-9_]+)\b",
            r"\b(unknown_[A-Za-z0-9_]+)\b",
            r"\b(flag_[A-Za-z0-9_]+)\b",
            r"\b(x87_status_[A-Za-z0-9_]+)\b",
            r"\b((?:cpuid|rdtsc|rdtscp)_[A-Za-z0-9_]+)\b",
            r"\b((?:rep_(?!i_)[A-Za-z0-9_]+|after_string_[A-F0-9]+_[A-Za-z0-9_]+))\b",
        )
        for line in body:
            for pattern in patterns:
                names.update(re.findall(pattern, line))
        names = {name for name in names if not re.fullmatch(r"param_\d+", name)}
        declarations = []
        for name in sorted(names):
            if name.startswith(("unknown_x87", "phi_x87")):
                type_name = "long double"
            elif name.startswith(("cpuid_", "rdtsc_", "rdtscp_")):
                type_name = "uint32_t"
            elif name.startswith(("in_mm", "phi_mm")):
                type_name = "uint64_t"
            elif name.startswith(("in_xmm", "phi_xmm")):
                type_name = "__uint128_t"
            elif name.startswith("x87_status_"):
                type_name = "uint16_t"
            elif name.startswith("flag_"):
                type_name = "uint8_t"
            else:
                type_name = _unsigned_type(self.bits)
            declarations.append(f"{type_name} {name};")
        return declarations

    def _call_is_noreturn(self, instruction: Instruction) -> bool:
        try:
            return bool(self.analyzer._call_is_noreturn(instruction))
        except (AttributeError, TypeError):
            return False


def _var(name: str, bits: int) -> _Expr:
    return _Expr("var", name, bits=bits)


def _expression_variables(expression: _Expr):
    if expression.kind == "var":
        yield str(expression.value)
    for argument in expression.args:
        yield from _expression_variables(argument)


def _const(value: int, bits: int) -> _Expr:
    return _Expr("const", int(value), bits=bits)


def _concat(high: _Expr, low: _Expr, low_bits: int) -> _Expr:
    total_bits = high.bits + low_bits
    if high.kind == "const" and low.kind == "const":
        value = ((int(high.value) & _mask(high.bits)) << low_bits) | (
            int(low.value) & _mask(low_bits)
        )
        return _const(value, total_bits)
    return _Expr("concat", low_bits, args=(high, low), bits=total_bits)


def _mmx_lane_binary(
    operation: str, left: _Expr, right: _Expr, lane_width: int
) -> _Expr:
    if left.kind == "const" and right.kind == "const":
        lane_mask = _mask(lane_width)
        result = 0
        for index in range(64 // lane_width):
            shift = index * lane_width
            lhs = (int(left.value) >> shift) & lane_mask
            rhs = (int(right.value) >> shift) & lane_mask
            lane = lhs + rhs if operation == "add" else lhs - rhs
            if operation == "mul":
                lane = lhs * rhs
            result |= (lane & lane_mask) << shift
        return _const(result, 64)
    return _Expr("mmx_lanes", (operation, lane_width), args=(left, right), bits=64)


def _mmx_lane_shift(
    operation: str, value: _Expr, count: _Expr, lane_width: int
) -> _Expr:
    if value.kind == "const" and count.kind == "const":
        lane_mask = _mask(lane_width)
        amount = int(count.value)
        result = 0
        for index in range(64 // lane_width):
            shift = index * lane_width
            lane = (int(value.value) >> shift) & lane_mask
            if amount >= lane_width:
                shifted = (
                    lane_mask
                    if operation == "ashr" and lane & (1 << (lane_width - 1))
                    else 0
                )
            elif operation == "shl":
                shifted = lane << amount
            elif operation == "lshr":
                shifted = lane >> amount
            else:
                shifted = _signed_value(lane, lane_width) >> amount
            result |= (shifted & lane_mask) << shift
        return _const(result, 64)
    return _Expr("mmx_shift", (operation, lane_width), args=(value, count), bits=64)


def _float_bits(value: float, bits: int) -> int:
    format_code = "<f" if bits == 32 else "<d"
    try:
        packed = struct.pack(format_code, value)
    except OverflowError:
        packed = struct.pack(format_code, math.copysign(math.inf, value))
    return int.from_bytes(packed, "little")


def _simd_float(value: _Expr, bits: int) -> _Expr:
    raw = _truncate(value, bits)
    if raw.kind == "const":
        format_code = "<f" if bits == 32 else "<d"
        packed = (int(raw.value) & _mask(bits)).to_bytes(bits // 8, "little")
        return _fp_const(struct.unpack(format_code, packed)[0])
    return _Expr("simd_float", bits, args=(raw,), bits=80)


def _simd_scalar_binary(
    operation: str, left: _Expr, right: _Expr, destination: _Expr, bits: int
) -> _Expr:
    result = _fp_binary(operation, _simd_float(left, bits), _simd_float(right, bits))
    if result.kind == "fp_const" and destination.kind == "const":
        raw_result = _float_bits(float(result.value), bits)
        upper_mask = _mask(128) ^ _mask(bits)
        return _const((int(destination.value) & upper_mask) | raw_result, 128)
    return _Expr("simd_fp_binary", bits, args=(result, destination), bits=128)


def _simd_convert_int_to_fp(
    value: _Expr, input_bits: int, output_bits: int, destination: _Expr
) -> _Expr:
    if value.kind == "const" and destination.kind == "const":
        integer = _signed_value(int(value.value), input_bits)
        raw_result = _float_bits(float(integer), output_bits)
        upper_mask = _mask(128) ^ _mask(output_bits)
        return _const((int(destination.value) & upper_mask) | raw_result, 128)
    return _Expr(
        "simd_convert_int_fp",
        (input_bits, output_bits),
        args=(value, destination),
        bits=128,
    )


def _simd_convert_fp(
    value: _Expr, input_bits: int, output_bits: int, destination: _Expr
) -> _Expr:
    floating = _simd_float(value, input_bits)
    if floating.kind == "fp_const" and destination.kind == "const":
        raw_result = _float_bits(float(floating.value), output_bits)
        upper_mask = _mask(128) ^ _mask(output_bits)
        return _const((int(destination.value) & upper_mask) | raw_result, 128)
    return _Expr(
        "simd_convert_fp",
        (input_bits, output_bits),
        args=(floating, destination),
        bits=128,
    )


def _simd_scalar_minmax(
    operation: str, left: _Expr, right: _Expr, destination: _Expr
) -> _Expr:
    lhs, rhs = _simd_float(left, 64), _simd_float(right, 64)
    if (
        lhs.kind == "fp_const"
        and rhs.kind == "fp_const"
        and destination.kind == "const"
    ):
        left_value, right_value = float(lhs.value), float(rhs.value)
        if (
            math.isnan(left_value)
            or math.isnan(right_value)
            or left_value == right_value
        ):
            result = right_value
        elif operation == "min":
            result = min(left_value, right_value)
        else:
            result = max(left_value, right_value)
        raw_result = _float_bits(result, 64)
        upper = int(destination.value) & (_mask(128) ^ _mask(64))
        return _const(upper | raw_result, 128)
    return _Expr("simd_minmax", operation, args=(lhs, rhs, destination), bits=128)


def _simd_cvtdq2pd(value: _Expr) -> _Expr:
    if value.kind == "const":
        result = 0
        for index in range(2):
            integer = _signed_value(int(value.value) >> (index * 32), 32)
            result |= _float_bits(float(integer), 64) << (index * 64)
        return _const(result, 128)
    return _Expr("simd_cvtdq2pd", args=(value,), bits=128)


def _simd_cvtt_to_int(value: _Expr, result_bits: int, *, input_bits: int = 64) -> _Expr:
    number = _simd_float(value, input_bits)
    if number.kind == "fp_const":
        floating = float(number.value)
        minimum = -(1 << (result_bits - 1))
        upper_bound = 1 << (result_bits - 1)
        if not math.isfinite(floating) or floating < minimum or floating >= upper_bound:
            return _const(1 << (result_bits - 1), result_bits)
        return _const(math.trunc(floating), result_bits)
    return _Expr(
        "simd_cvtt_to_int", (result_bits, input_bits), args=(number,), bits=result_bits
    )


def _simd_lane_binary(
    operation: str, left: _Expr, right: _Expr, lane_width: int
) -> _Expr:
    if left.kind == "const" and right.kind == "const":
        lane_mask = _mask(lane_width)
        result = 0
        for index in range(128 // lane_width):
            shift = index * lane_width
            lhs = (int(left.value) >> shift) & lane_mask
            rhs = (int(right.value) >> shift) & lane_mask
            if operation == "add":
                lane = lhs + rhs
            elif operation == "sub":
                lane = lhs - rhs
            elif operation == "mul":
                lane = lhs * rhs
            elif operation == "eq":
                lane = lane_mask if lhs == rhs else 0
            elif operation == "smin":
                lane = (
                    lhs
                    if _signed_value(lhs, lane_width) < _signed_value(rhs, lane_width)
                    else rhs
                )
            else:
                lane = (
                    lhs
                    if _signed_value(lhs, lane_width) > _signed_value(rhs, lane_width)
                    else rhs
                )
            result |= (lane & lane_mask) << shift
        return _const(result, 128)
    return _Expr("simd_lanes", (operation, lane_width), args=(left, right), bits=128)


def _simd_shuffle_dwords(value: _Expr, immediate: _Expr) -> _Expr:
    if value.kind == "const" and immediate.kind == "const":
        source = int(value.value)
        selector = int(immediate.value)
        result = 0
        for index in range(4):
            source_index = (selector >> (index * 2)) & 3
            lane = (source >> (source_index * 32)) & 0xFFFFFFFF
            result |= lane << (index * 32)
        return _const(result, 128)
    selector = int(immediate.value) if immediate.kind == "const" else 0
    return _Expr("simd_shuffle_dwords", selector, args=(value,), bits=128)


def _simd_shift_bytes(value: _Expr, amount: _Expr, *, left: bool) -> _Expr:
    direction = "left" if left else "right"
    if value.kind == "const" and amount.kind == "const":
        shift = int(amount.value) * 8
        if int(amount.value) >= 16:
            return _const(0, 128)
        if left:
            return _const(int(value.value) << shift, 128)
        return _const(int(value.value) >> shift, 128)
    return _Expr("simd_shift_bytes", direction, args=(value, amount), bits=128)


def _mask(bits: int) -> int:
    return (1 << bits) - 1 if bits > 0 else 0


def _parity(value: _Expr) -> _Expr:
    if value.kind == "const":
        return _const((int(value.value) & 0xFF).bit_count() % 2 == 0, 1)
    return _Expr("parity", args=(value,), bits=1)


def _signed_value(value: int, bits: int) -> int:
    if bits <= 0:
        return value
    value &= _mask(bits)
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _zero_extend(value: _Expr, bits: int) -> _Expr:
    if value.bits == bits:
        return value
    if value.kind == "const":
        return _const(int(value.value) & _mask(value.bits), bits)
    if value.kind == "zext" and value.bits == bits:
        return value
    return _Expr("zext", args=(value,), bits=bits)


def _sign_extend(value: _Expr, bits: int) -> _Expr:
    if value.bits == bits:
        return value
    if value.kind == "const":
        return _const(_signed_value(int(value.value), value.bits), bits)
    if value.kind == "sext" and value.bits == bits:
        return value
    return _Expr("sext", args=(value,), bits=bits)


def _truncate(value: _Expr, bits: int) -> _Expr:
    if value.bits == bits:
        return value
    if value.kind == "const":
        return _const(int(value.value) & _mask(bits), bits)
    if value.kind == "zext" and value.args[0].bits <= bits:
        return _resize(value.args[0], bits)
    if value.kind == "trunc" and value.bits == bits:
        return value
    return _Expr("trunc", args=(value,), bits=bits)


def _resize(value: _Expr, bits: int) -> _Expr:
    if bits == value.bits:
        return value
    if bits > value.bits:
        return _zero_extend(value, bits)
    return _truncate(value, bits)


def _extract(value: _Expr, offset: int, bits: int) -> _Expr:
    if offset == 0:
        return _truncate(value, bits)
    if value.kind == "const":
        return _const((int(value.value) >> offset) & _mask(bits), bits)
    return _Expr("extract", args=(value, _const(offset, 8)), bits=bits)


def _insert_bits(
    old: _Expr, new: _Expr, offset: int, width: int, total_bits: int
) -> _Expr:
    new = _resize(new, width)
    if offset == 0 and width == total_bits:
        return new
    if old.kind == "const" and new.kind == "const":
        mask = _mask(width) << offset
        value = (int(old.value) & ~mask) | ((int(new.value) & _mask(width)) << offset)
        return _const(value & _mask(total_bits), total_bits)
    return _Expr("insert", args=(old, new, _const(offset, 8)), bits=total_bits)


def _unary(op: str, value: _Expr, bits: int) -> _Expr:
    if value.kind == "const":
        raw = int(value.value) & _mask(bits)
        if op == "neg":
            return _const((-raw) & _mask(bits), bits)
        if op == "bitnot":
            return _const((~raw) & _mask(bits), bits)
        if op == "boolnot":
            return _const(0 if raw else 1, 1)
    return _Expr("unary", op, args=(value,), bits=bits)


def _binary(op: str, left: _Expr, right: _Expr, bits: int) -> _Expr:
    if op == "xor" and left == right:
        return _const(0, bits)
    if op == "sub" and left == right:
        return _const(0, bits)
    if op in {"and", "or"} and left == right:
        return left
    if op == "add" and right.kind == "const" and int(right.value) == 0:
        return _resize(left, bits)
    if op == "add" and left.kind == "const" and int(left.value) == 0:
        return _resize(right, bits)
    if op == "sub" and right.kind == "const" and int(right.value) == 0:
        return _resize(left, bits)
    if op == "mul" and (
        (left.kind == "const" and int(left.value) == 0)
        or (right.kind == "const" and int(right.value) == 0)
    ):
        return _const(0, bits)
    if op == "mul" and left.kind == "const" and int(left.value) == 1:
        return _resize(right, bits)
    if op == "mul" and right.kind == "const" and int(right.value) == 1:
        return _resize(left, bits)
    if op == "and" and right.kind == "const" and int(right.value) == _mask(bits):
        return _resize(left, bits)

    if left.kind == "const" and right.kind == "const":
        lhs = int(left.value)
        rhs = int(right.value)
        unsigned_lhs = lhs & _mask(max(left.bits, bits))
        unsigned_rhs = rhs & _mask(max(right.bits, bits))
        signed_lhs = _signed_value(lhs, max(left.bits, bits))
        signed_rhs = _signed_value(rhs, max(right.bits, bits))
        comparisons = {
            "eq": unsigned_lhs == unsigned_rhs,
            "ne": unsigned_lhs != unsigned_rhs,
            "ult": unsigned_lhs < unsigned_rhs,
            "ule": unsigned_lhs <= unsigned_rhs,
            "ugt": unsigned_lhs > unsigned_rhs,
            "uge": unsigned_lhs >= unsigned_rhs,
            "slt": signed_lhs < signed_rhs,
            "sle": signed_lhs <= signed_rhs,
            "sgt": signed_lhs > signed_rhs,
            "sge": signed_lhs >= signed_rhs,
            "land": bool(lhs) and bool(rhs),
            "lor": bool(lhs) or bool(rhs),
        }
        if op in comparisons:
            return _const(int(comparisons[op]), 1)
        try:
            signed_quotient = abs(signed_lhs) // abs(signed_rhs)
            if (signed_lhs < 0) != (signed_rhs < 0):
                signed_quotient = -signed_quotient
            shift_count_mask = 0x3F if bits >= 64 else 0x1F
            values = {
                "add": lhs + rhs,
                "sub": lhs - rhs,
                "mul": lhs * rhs,
                "udiv": unsigned_lhs // unsigned_rhs,
                "sdiv": signed_quotient,
                "urem": unsigned_lhs % unsigned_rhs,
                "srem": signed_lhs - signed_quotient * signed_rhs,
                "and": unsigned_lhs & unsigned_rhs,
                "or": unsigned_lhs | unsigned_rhs,
                "xor": unsigned_lhs ^ unsigned_rhs,
                "shl": unsigned_lhs << (unsigned_rhs & shift_count_mask),
                "lshr": unsigned_lhs >> (unsigned_rhs & shift_count_mask),
                "ashr": signed_lhs >> (unsigned_rhs & shift_count_mask),
            }
            if op in values:
                return _const(values[op] & _mask(bits), bits)
        except (ZeroDivisionError, ValueError, OverflowError):
            pass
    if op in {
        "eq",
        "ne",
        "ult",
        "ule",
        "ugt",
        "uge",
        "slt",
        "sle",
        "sgt",
        "sge",
        "land",
        "lor",
    }:
        result_bits = 1
    else:
        result_bits = bits
    return _Expr(
        "op",
        op,
        args=(
            _resize(left, max(left.bits, bits)),
            _resize(right, max(right.bits, bits)),
        ),
        bits=result_bits,
    )


def _select(condition: _Expr, yes: _Expr, no: _Expr, bits: int) -> _Expr:
    if condition.kind == "const":
        return _resize(yes if int(condition.value) else no, bits)
    if yes == no:
        return _resize(yes, bits)
    return _Expr(
        "select", args=(condition, _resize(yes, bits), _resize(no, bits)), bits=bits
    )


def _format_integer(value: int) -> str:
    if value < 0:
        return str(value)
    if value <= 9:
        return str(value)
    return f"{value:#x}"


def _fp_const(value: float) -> _Expr:
    return _Expr("fp_const", float(value), bits=80)


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "NAN"
    if math.isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    text = repr(value)
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text


def _fp_binary(operation: str, left: _Expr, right: _Expr) -> _Expr:
    if left.kind == "fp_const" and right.kind == "fp_const":
        lhs, rhs = float(left.value), float(right.value)
        try:
            values = {
                "add": lambda: lhs + rhs,
                "sub": lambda: lhs - rhs,
                "mul": lambda: lhs * rhs,
                "div": lambda: lhs / rhs,
            }
            if operation in values:
                return _fp_const(values[operation]())
        except (ArithmeticError, ValueError, OverflowError):
            pass
    return _Expr("fp_op", operation, args=(left, right), bits=80)


def _fp_unary(operation: str, value: _Expr) -> _Expr:
    if value.kind == "fp_const":
        number = float(value.value)
        try:
            values = {
                "abs": lambda: abs(number),
                "neg": lambda: -number,
                "sqrt": lambda: math.sqrt(number),
                "sin": lambda: math.sin(number),
                "cos": lambda: math.cos(number),
                "tan": lambda: math.tan(number),
                "round": lambda: float(round(number)),
                "exp2m1": lambda: math.exp2(number) - 1.0,
            }
            if operation in values:
                return _fp_const(values[operation]())
        except (ArithmeticError, ValueError, OverflowError):
            pass
    return _Expr("fp_unary", operation, args=(value,), bits=80)


def _float_type(bits: int) -> str:
    if bits <= 32:
        return "float"
    if bits <= 64:
        return "double"
    return "long double"


def _memory_type(bits: int) -> str:
    return _unsigned_type(bits)


def _unsigned_type(bits: int) -> str:
    if bits >= 128:
        return "__uint128_t"
    width = max(8, ((bits + 7) // 8) * 8)
    return f"uint{width}_t"


def _signed_type(bits: int) -> str:
    if bits >= 128:
        return "__int128"
    width = max(8, ((bits + 7) // 8) * 8)
    return f"int{width}_t"


def _c_identifier(value: str | None) -> str:
    if not value:
        return "unknown"
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not identifier or identifier[0].isdigit():
        identifier = "sym_" + identifier
    if identifier in {
        "void",
        "int",
        "char",
        "long",
        "short",
        "return",
        "if",
        "else",
        "while",
        "for",
        "switch",
    }:
        identifier = "func_" + identifier
    return identifier
