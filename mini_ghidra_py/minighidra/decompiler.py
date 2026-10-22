"""Experimental semantic pseudocode recovery for x86 and x86-64.

This module follows the broad stages visible in Ghidra's decompiler source:
instruction semantics are lifted into symbolic values, values are propagated
through the recovered control-flow graph, merge points receive phi values, and
the result is printed as C-like pseudocode. Dedicated lifters handle common
cases; a conservative interpreter also consumes supported per-instruction
SLEIGH p-code for instructions without a dedicated lift. This remains a
partial implementation, not the complete Ghidra decompiler.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import struct
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import ClassVar

try:
    from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64, Cs, CsError
    from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_OP_REG
except ImportError as exc:  # pragma: no cover - declared package dependency
    raise RuntimeError(
        "Falta Capstone. Instálalo con: python -m pip install capstone"
    ) from exc

from .abi import (
    CallingConvention,
    abi_argument_registers,
    calling_convention_for,
    callsite_register_writes,
    callsite_stack_writes,
    default_calling_convention,
)
from .model import (
    BasicBlock,
    Function,
    FunctionParameter,
    FunctionSignature,
    Instruction,
)
from .paramid import ParamActive, ParamIDAnalysis, ParamRank, PcodeLikeGraph
from .prototype import PrototypeResolver
from .sleigh import SleighInstruction, SleighTranslator, SleighVarnode
from .ssa import SSAFunction, SSAHeritage, SSALocation
from .typesys import StorageLocation, TypeDatabase, TypeKind


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
    zero: _Expr | None = None
    sign: _Expr | None = None


@dataclass
class _State:
    registers: dict[str, _Expr] = field(default_factory=dict)
    pcode_registers: dict[str, _Expr] = field(default_factory=dict)
    mmx_registers: dict[str, _Expr] = field(default_factory=dict)
    xmm_registers: dict[str, _Expr] = field(default_factory=dict)
    xmm_widths: dict[str, int] = field(default_factory=dict)
    stack: dict[tuple[int, int], _Expr] = field(default_factory=dict)
    memory: dict[tuple[int, int], _Expr] = field(default_factory=dict)
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
            pcode_registers=dict(self.pcode_registers),
            mmx_registers=dict(self.mmx_registers),
            xmm_registers=dict(self.xmm_registers),
            xmm_widths=dict(self.xmm_widths),
            stack=dict(self.stack),
            memory=dict(self.memory),
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


@dataclass
class _SleighExecutionPath:
    pc: int
    state: _State
    scratch: dict[tuple[str, int, int], _Expr]
    flag_updates: dict[str, _Expr]
    condition: _Expr
    steps: int = 0
    nonzero_sources: set[_Expr] = field(default_factory=set)
    zero_sources: set[_Expr] = field(default_factory=set)
    bit_facts: dict[tuple[_Expr, int], bool] = field(default_factory=dict)
    zero_high_from: dict[_Expr, int] = field(default_factory=dict)
    nonzero_high_from: dict[_Expr, int] = field(default_factory=dict)
    inconsistent: bool = False
    memory_stores: list[tuple[_Expr, _Expr, _Expr, int, str]] = field(
        default_factory=list
    )
    memory_epoch: int = 0


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
    _SLEIGH_OPCODES: ClassVar[frozenset[str]] = frozenset(
        {
            "IMARK",
            "BRANCH",
            "CBRANCH",
            "CALLOTHER",
            "COPY",
            "INT_ADD",
            "INT_SUB",
            "INT_MULT",
            "INT_DIV",
            "INT_SDIV",
            "INT_REM",
            "INT_SREM",
            "INT_AND",
            "INT_OR",
            "INT_XOR",
            "INT_LEFT",
            "INT_RIGHT",
            "INT_SRIGHT",
            "INT_EQUAL",
            "INT_NOTEQUAL",
            "INT_LESS",
            "INT_LESSEQUAL",
            "INT_SLESS",
            "INT_SLESSEQUAL",
            "INT_2COMP",
            "INT_NEGATE",
            "INT_ZEXT",
            "INT_SEXT",
            "INT_CARRY",
            "INT_SCARRY",
            "INT_SBORROW",
            "PIECE",
            "SUBPIECE",
            "BOOL_NEGATE",
            "BOOL_AND",
            "BOOL_OR",
            "BOOL_XOR",
            "POPCOUNT",
            "LZCOUNT",
            "FLOAT_EQUAL",
            "FLOAT_NOTEQUAL",
            "FLOAT_LESS",
            "FLOAT_LESSEQUAL",
            "FLOAT_NAN",
            "FLOAT_ADD",
            "FLOAT_DIV",
            "FLOAT_MULT",
            "FLOAT_SUB",
            "FLOAT_NEG",
            "FLOAT_ABS",
            "FLOAT_SQRT",
            "FLOAT_INT2FLOAT",
            "FLOAT_FLOAT2FLOAT",
            "FLOAT_TRUNC",
            "FLOAT_CEIL",
            "FLOAT_FLOOR",
            "FLOAT_ROUND",
            "LOAD",
            "STORE",
        }
    )
    _SLEIGH_ARITIES: ClassVar[dict[str, int]] = {
        "BRANCH": 1,
        "CBRANCH": 2,
        "COPY": 1,
        "INT_ADD": 2,
        "INT_SUB": 2,
        "INT_MULT": 2,
        "INT_DIV": 2,
        "INT_SDIV": 2,
        "INT_REM": 2,
        "INT_SREM": 2,
        "INT_AND": 2,
        "INT_OR": 2,
        "INT_XOR": 2,
        "INT_LEFT": 2,
        "INT_RIGHT": 2,
        "INT_SRIGHT": 2,
        "INT_EQUAL": 2,
        "INT_NOTEQUAL": 2,
        "INT_LESS": 2,
        "INT_LESSEQUAL": 2,
        "INT_SLESS": 2,
        "INT_SLESSEQUAL": 2,
        "INT_2COMP": 1,
        "INT_NEGATE": 1,
        "INT_ZEXT": 1,
        "INT_SEXT": 1,
        "INT_CARRY": 2,
        "INT_SCARRY": 2,
        "INT_SBORROW": 2,
        "PIECE": 2,
        "SUBPIECE": 2,
        "BOOL_NEGATE": 1,
        "BOOL_AND": 2,
        "BOOL_OR": 2,
        "BOOL_XOR": 2,
        "POPCOUNT": 1,
        "LZCOUNT": 1,
        "FLOAT_EQUAL": 2,
        "FLOAT_NOTEQUAL": 2,
        "FLOAT_LESS": 2,
        "FLOAT_LESSEQUAL": 2,
        "FLOAT_ADD": 2,
        "FLOAT_DIV": 2,
        "FLOAT_MULT": 2,
        "FLOAT_SUB": 2,
        "FLOAT_NAN": 1,
        "FLOAT_NEG": 1,
        "FLOAT_ABS": 1,
        "FLOAT_SQRT": 1,
        "FLOAT_INT2FLOAT": 1,
        "FLOAT_FLOAT2FLOAT": 1,
        "FLOAT_TRUNC": 1,
        "FLOAT_CEIL": 1,
        "FLOAT_FLOOR": 1,
        "FLOAT_ROUND": 1,
    }
    _PCODE_FLAG_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"cf", "zf", "sf", "of", "pf"}
    )
    _SLEIGH_PRESERVED_FLAG_MNEMONICS: ClassVar[frozenset[str]] = frozenset(
        {"rol", "ror", "rcl", "rcr"}
    )

    def __init__(self, analyzer) -> None:
        self.analyzer = analyzer
        self.bits = analyzer.image.bits
        self.pointer_size = self.bits // 8
        self._sleigh_translator = SleighTranslator(self.bits)
        self._sleigh_instructions: tuple[SleighInstruction, ...] = ()
        self._sleigh_instruction_by_address: dict[int, SleighInstruction] = {}
        self._ssa_function: SSAFunction | None = None
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
        self._parameter_evidence: dict[str, set[str]] = defaultdict(set)
        self._parameter_read_widths: dict[str, set[int]] = defaultdict(set)
        self._paramid_graph = PcodeLikeGraph()
        self._paramid_analysis: ParamIDAnalysis | None = None
        self._paramid_active = ParamActive()
        self._paramid_recording_enabled = False
        self._current_instruction: Instruction | None = None
        self._current_pcode_register_versions: dict[str, int] = {}
        self._current_pcode_read_nodes: list[int] = []
        self._declared_signature: FunctionSignature | None = None
        self._input_parameter_names_by_register: dict[str, str] = {}
        self._input_parameter_names_by_stack: dict[int, str] = {}
        self._type_database = getattr(analyzer, "type_database", None) or TypeDatabase(
            self.bits, getattr(analyzer.image, "format", "")
        )
        self._prototype_resolver = PrototypeResolver(self._type_database, self.bits)
        self._calling_convention = default_calling_convention(analyzer)
        self._configure_calling_convention(self._calling_convention)
        self._string_by_address = {
            item.address: item.text for item in getattr(analyzer, "strings", ())
        }

    @property
    def paramid_graph(self) -> PcodeLikeGraph:
        """Expose the reduced graph from the most recent decompilation."""
        return self._paramid_graph

    @property
    def paramid_ssa_graph(self) -> PcodeLikeGraph | None:
        """SSA p-code graph consumed by ParamID, if the last run built one."""
        if self._paramid_analysis is None or not self._paramid_analysis.uses_ssa_pcode:
            return None
        return self._paramid_analysis.graph

    @property
    def paramid_trials(self) -> ParamActive:
        """Expose active/selected/final-use trial state for inspection."""
        return self._paramid_active

    @property
    def sleigh_instructions(self) -> tuple[SleighInstruction, ...]:
        """Exact upstream SLEIGH translations from the latest decompilation."""
        return self._sleigh_instructions

    @property
    def ssa_function(self) -> SSAFunction | None:
        """The validated p-code SSA/heritage graph from the latest decompilation."""
        return self._ssa_function

    def decompile(self, function: Function) -> str:
        self._function = function
        self._calling_convention = calling_convention_for(self.analyzer, function)
        self._configure_calling_convention(self._calling_convention)
        self._return_value_used_by_callers = self._callers_use_return_value(function)
        self._parameter_types = {}
        self._used_parameter_names = set()
        self._parameter_evidence = defaultdict(set)
        self._parameter_read_widths = defaultdict(set)
        self._paramid_graph = PcodeLikeGraph()
        self._sleigh_instructions = self._sleigh_translator.translate_instructions(
            function.instructions
        )
        self._sleigh_instruction_by_address = {
            item.address: item for item in self._sleigh_instructions
        }
        self._ssa_function = SSAHeritage(
            function,
            self._sleigh_instructions,
            caller_saved_registers=self._calling_convention.caller_saved_registers,
            return_registers=(
                *self._calling_convention.integer_return_registers,
                *self._calling_convention.floating_return_registers,
            ),
            call_argument_registers=abi_argument_registers(
                self._calling_convention
            ),
            no_return_calls=(
                instruction.address
                for instruction in function.instructions
                if self.analyzer._call_is_noreturn(instruction)
                and instruction.branch_target != instruction.next_address
            ),
            registers=self._sleigh_translator.registers,
        ).build()
        self._paramid_graph.attach_sleigh_instructions(self._sleigh_instructions)
        self._paramid_analysis = None
        self._paramid_active = ParamActive()
        self._paramid_recording_enabled = False
        self._current_instruction = None
        self._current_pcode_register_versions = {}
        self._current_pcode_read_nodes = []
        self._declared_signature = None
        self._input_parameter_names_by_register = {}
        self._input_parameter_names_by_stack = {}
        if function.signature is not None and function.signature.locked:
            layout = self._prototype_resolver.resolve(
                function.signature, self._calling_convention
            )
            function.signature = layout.signature
            self._declared_signature = layout.signature
            self._bind_declared_parameters(layout.signature)
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
        self._paramid_recording_enabled = True
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

        self._paramid_recording_enabled = False
        body = self._prune_dead_phis(body)
        body = self._drop_unused_call_results(body)
        if self._declared_signature is not None:
            parameters = list(self._declared_signature.parameters)
            return_type = self._declared_signature.return_type
            signature = self._declared_signature
        else:
            parameters = self._parameter_specs(body)
            return_type = (
                (self._return_type or f"uint{self.bits}_t")
                if self._saw_value_return
                else "void"
            )
            signature = FunctionSignature(
                return_type, tuple(parameters), self._calling_convention.name
            )
        parameter_declarations = [
            _parameter_declaration(parameter) for parameter in parameters
        ]
        if signature.variadic:
            parameter_declarations.append("...")
        params = ", ".join(parameter_declarations) if parameter_declarations else "void"
        function.calling_convention = self._calling_convention.name
        resolved_signature = self._prototype_resolver.resolve(
            signature, self._calling_convention
        ).signature
        if self._declared_signature is not None:
            justproto_names = [
                parameter.name for parameter in resolved_signature.parameters
            ]
            if resolved_signature.hidden_return_location is not None:
                justproto_names.append("__hidden_return")
            self._run_paramid_analysis(
                justproto=True,
                input_names=justproto_names,
            )
        if self._paramid_analysis is not None:
            paramid_parameters = list(resolved_signature.parameters)
            if (
                self._declared_signature is not None
                and resolved_signature.hidden_return_location is not None
            ):
                hidden_location = resolved_signature.hidden_return_location
                paramid_parameters.append(
                    FunctionParameter(
                        "__hidden_return",
                        f"{resolved_signature.return_type} *",
                        locations=(hidden_location,),
                        implicit=True,
                    )
                )
            self._paramid_active = self._paramid_analysis.bind_parameters(
                paramid_parameters
            )
            self._paramid_active.finalize_selected_trials()
            resolved_signature = self._annotate_paramid_signature(
                resolved_signature
            )
        if self._declared_signature is None:
            resolved_signature = self._annotate_inferred_signature(
                function, resolved_signature
            )
        function.signature = resolved_signature
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
        signature = self._declared_signature
        if signature is None:
            signature = FunctionSignature("void", (), self._calling_convention.name)
            function.signature = signature
            parameters = "void"
            return_type = "void"
        else:
            parameters = ", ".join(
                _parameter_declaration(parameter) for parameter in signature.parameters
            )
            if signature.variadic:
                parameters = f"{parameters}, ..." if parameters else "..."
            parameters = parameters or "void"
            return_type = signature.return_type
        name = _c_identifier(function.name)
        if function.thunk_target_name:
            target = _c_identifier(function.thunk_target_name)
        elif function.thunk_target is not None:
            target = f"loc_{function.thunk_target:X}"
        else:
            target = "indirect_target"
        convention_prefix = (
            f"{self._calling_convention.name} "
            if self._calling_convention.print_name
            else ""
        )
        lines = [
            "/* Tail thunk; target forwarding remains represented as a goto. */",
            f"{return_type} {convention_prefix}{name}({parameters}) {{",
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

    def _bind_declared_parameters(self, signature: FunctionSignature) -> None:
        for parameter in signature.parameters:
            name = _c_identifier(parameter.name)
            self._parameter_types[name] = parameter.type_name
            for location in parameter.locations:
                if location.register:
                    self._input_parameter_names_by_register[location.register] = name
                if location.stack_offset is not None:
                    self._input_parameter_names_by_stack[location.stack_offset] = name
        hidden_location = signature.hidden_return_location
        if hidden_location is not None:
            if hidden_location.register:
                self._input_parameter_names_by_register[hidden_location.register] = (
                    "__hidden_return"
                )
            if hidden_location.stack_offset is not None:
                self._input_parameter_names_by_stack[hidden_location.stack_offset] = (
                    "__hidden_return"
                )

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

    @staticmethod
    def _sleigh_vector_register_slice(name: str) -> tuple[str, int, int] | None:
        match = re.fullmatch(r"xmm(\d+)_([bwdq])([a-p])", name.lower())
        if match is None:
            return None
        register_index = int(match.group(1))
        lane_kind = match.group(2)
        lane_index = ord(match.group(3)) - ord("a")
        lane_bits, lane_count = {
            "b": (8, 16),
            "w": (16, 8),
            "d": (32, 4),
            "q": (64, 2),
        }[lane_kind]
        if lane_index >= lane_count:
            return None
        return f"xmm{register_index}", lane_index * lane_bits, lane_bits

    def _record_parameter_sources(
        self,
        value: _Expr,
        evidence: str,
        width: int,
        *,
        instruction: Instruction | None = None,
        use_kind: str = "direct",
    ) -> set[str]:
        if self._declared_signature is not None:
            return set()
        sources = {
            name
            for name in _expression_variables(value)
            if re.fullmatch(r"(?:float_)?param_\d+", name)
        }
        if use_kind != "none":
            for name in sources:
                self._used_parameter_names.add(name)
                self._parameter_evidence[name].add(evidence)
                self._parameter_read_widths[name].add(width)
        if self._paramid_recording_enabled and use_kind != "none":
            self._paramid_graph.record_use(
                value,
                role=use_kind,
                source_names=sources,
                address=instruction.address if instruction is not None else None,
                size_bits=width,
                evidence=evidence,
                mnemonic=(
                    instruction.mnemonic.lower() if instruction is not None else None
                ),
            )
        return sources

    def _record_pcode_register_read(
        self,
        register: str,
        value: _Expr,
        width: int,
        instruction: Instruction,
    ) -> int | None:
        if not self._paramid_recording_enabled:
            return None
        node = self._paramid_graph.record_register_read(
            register,
            value,
            address=instruction.address,
            mnemonic=self._decode(instruction).mnemonic,
            size_bits=width,
            source_node=self._current_pcode_register_versions.get(register),
        )
        self._current_pcode_read_nodes.append(node)
        return node

    def _record_pcode_register_definition(
        self,
        register: str,
        value: _Expr,
        width: int,
        instruction: Instruction,
        *,
        input_nodes: tuple[int, ...] | None = None,
    ) -> None:
        if not self._paramid_recording_enabled:
            return
        output = self._paramid_graph.record_register_definition(
            register,
            value,
            address=instruction.address,
            mnemonic=self._decode(instruction).mnemonic,
            size_bits=width,
            input_nodes=(
                tuple(self._current_pcode_read_nodes)
                if input_nodes is None
                else input_nodes
            ),
        )
        self._current_pcode_register_versions[register] = output

    def _record_pcode_memory_load(
        self,
        value: _Expr,
        address_expression: _Expr,
        location: str,
        address_registers: tuple[str, ...],
        instruction: Instruction,
    ) -> None:
        if not self._paramid_recording_enabled:
            return
        output = self._paramid_graph.record_memory_load(
            value,
            address_expression,
            location=location,
            address=instruction.address,
            mnemonic=self._decode(instruction).mnemonic,
            size_bits=max(1, value.bits),
            address_registers=address_registers,
        )
        self._current_pcode_read_nodes.append(output)

    def _record_pcode_memory_store(
        self,
        value: _Expr,
        address_expression: _Expr,
        location: str,
        address_registers: tuple[str, ...],
        instruction: Instruction,
        width: int,
    ) -> None:
        if not self._paramid_recording_enabled:
            return
        self._paramid_graph.record_memory_store(
            value,
            address_expression,
            location=location,
            address=instruction.address,
            mnemonic=self._decode(instruction).mnemonic,
            size_bits=width,
            address_registers=address_registers,
        )

    def _memory_location(
        self, operand: _Operand, state: _State, instruction: Instruction
    ) -> str:
        offset = self._stack_offset(operand, state)
        if offset is not None:
            return f"stack[{offset:+#x}]"
        if operand.base == "rip":
            return f"absolute[{instruction.next_address + operand.displacement:#x}]"
        if not operand.base and not operand.index:
            absolute = operand.displacement & ((1 << self.bits) - 1)
            return f"absolute[{absolute:#x}]"
        parts = [operand.segment] if operand.segment else []
        if operand.base:
            parts.append(operand.base)
        if operand.index:
            parts.append(f"{operand.index}*{operand.scale}")
        if operand.displacement:
            parts.append(f"{operand.displacement:+#x}")
        return "memory[" + "+".join(parts or ["unknown"]) + "]"

    def _memory_address_registers(self, operand: _Operand) -> tuple[str, ...]:
        registers: list[str] = []
        for name in (operand.base, operand.index):
            alias = self._register_alias(name) if name else None
            if alias is not None and alias[0] not in registers:
                registers.append(alias[0])
        return tuple(registers)

    def _initial_register(self, family: str) -> _Expr:
        if self._declared_signature is not None:
            parameter_name = self._input_parameter_names_by_register.get(family)
            if parameter_name is not None:
                return _var(parameter_name, self.bits)
            return _var(f"in_{family}", self.bits)
        if family in self._argument_index:
            index = self._argument_index[family] + 1
            return _var(f"param_{index}", self.bits)
        return _var(f"in_{family}", self.bits)

    def _initial_xmm_register(self, register: str) -> _Expr:
        if self._declared_signature is not None:
            parameter_name = self._input_parameter_names_by_register.get(register)
            if parameter_name is not None:
                return _var(parameter_name, 128)
            return _var(f"in_{register}", 128)
        parameter_name = self._calling_convention.vector_parameter_name(register)
        if parameter_name is not None:
            return _var(parameter_name, 128)
        return _var(f"in_{register}", 128)

    def _note_vector_parameter_type(
        self,
        register: str,
        instruction: Instruction,
        parameter_name: str | None = None,
    ) -> None:
        if self._declared_signature is not None:
            return
        parameter_name = (
            parameter_name or self._calling_convention.vector_parameter_name(register)
        )
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
        self._parameter_evidence[parameter_name].add(f"vector-type-pattern:{mnemonic}")

    def _initial_stack_value(self, offset: int, bits: int) -> _Expr:
        if bits <= 0 or bits % 8:
            return _var(self._stack_name(offset), bits)
        slot_offset = (offset // self.pointer_size) * self.pointer_size
        slot_bit_offset = (offset - slot_offset) * 8
        if slot_bit_offset == 0 and bits <= self.bits:
            return _var(self._stack_name(offset), bits)
        if slot_bit_offset + bits <= self.bits:
            slot_value = _var(self._stack_name(slot_offset), self.bits)
            return _extract(slot_value, slot_bit_offset, bits)
        byte_values = [
            self._initial_stack_byte(byte_offset)
            for byte_offset in range(offset, offset + bits // 8)
        ]
        result = byte_values[0]
        for byte_value in byte_values[1:]:
            result = _concat(byte_value, result, result.bits)
        return _resize(result, bits)

    def _initial_stack_byte(self, offset: int) -> _Expr:
        slot_offset = (offset // self.pointer_size) * self.pointer_size
        bit_offset = (offset - slot_offset) * 8
        if bit_offset == 0:
            return _var(self._stack_name(slot_offset), 8)
        slot_value = _var(self._stack_name(slot_offset), self.bits)
        return _extract(slot_value, bit_offset, 8)

    @staticmethod
    def _stack_has_written(state: _State, offset: int, bits: int) -> bool:
        if bits <= 0:
            return False
        end = offset + (bits + 7) // 8
        return any(
            stored_offset < end and offset < stored_offset + (stored_bits + 7) // 8
            for stored_offset, stored_bits in state.stack
        )

    def _read_stack_value(self, state: _State, offset: int, bits: int) -> _Expr:
        if bits <= 0 or bits % 8:
            return self._initial_stack_value(offset, bits)
        exact = state.stack.get((offset, bits))
        if exact is not None:
            return _resize(exact, bits)
        end = offset + bits // 8
        for (stored_offset, stored_bits), stored_value in state.stack.items():
            stored_end = stored_offset + stored_bits // 8
            if stored_offset <= offset and stored_end >= end:
                return _extract(stored_value, (offset - stored_offset) * 8, bits)
        if not self._stack_has_written(state, offset, bits):
            return self._initial_stack_value(offset, bits)

        byte_values: list[_Expr] = []
        for byte_offset in range(offset, end):
            stored = next(
                (
                    (stored_offset, value)
                    for (stored_offset, stored_bits), value in state.stack.items()
                    if stored_offset <= byte_offset < stored_offset + stored_bits // 8
                ),
                None,
            )
            if stored is None:
                byte_values.append(self._initial_stack_byte(byte_offset))
            else:
                stored_offset, value = stored
                byte_values.append(
                    _extract(value, (byte_offset - stored_offset) * 8, 8)
                )

        result = byte_values[0]
        for byte_value in byte_values[1:]:
            result = _concat(byte_value, result, result.bits)
        return _resize(result, bits)

    def _write_stack_value(
        self, state: _State, offset: int, bits: int, value: _Expr
    ) -> None:
        value = _resize(value, bits)
        if bits <= 0 or bits % 8:
            state.stack[(offset, bits)] = value
            return
        end = offset + bits // 8
        remaining: dict[tuple[int, int], _Expr] = {}
        for (stored_offset, stored_bits), stored_value in state.stack.items():
            if stored_bits <= 0 or stored_bits % 8:
                remaining[(stored_offset, stored_bits)] = stored_value
                continue
            stored_end = stored_offset + stored_bits // 8
            if stored_end <= offset or stored_offset >= end:
                remaining[(stored_offset, stored_bits)] = stored_value
                continue
            if stored_offset < offset:
                left_bits = (offset - stored_offset) * 8
                remaining[(stored_offset, left_bits)] = _extract(
                    stored_value, 0, left_bits
                )
            if stored_end > end:
                right_offset = (end - stored_offset) * 8
                right_bits = (stored_end - end) * 8
                remaining[(end, right_bits)] = _extract(
                    stored_value, right_offset, right_bits
                )
        remaining[(offset, bits)] = value
        state.stack = remaining

    def _clear_stack_range(self, state: _State, start: int, end: int) -> None:
        if end <= start:
            return
        remaining: dict[tuple[int, int], _Expr] = {}
        for (stored_offset, stored_bits), stored_value in state.stack.items():
            if stored_bits <= 0 or stored_bits % 8:
                remaining[(stored_offset, stored_bits)] = stored_value
                continue
            stored_end = stored_offset + stored_bits // 8
            if stored_end <= start or stored_offset >= end:
                remaining[(stored_offset, stored_bits)] = stored_value
                continue
            if stored_offset < start:
                left_bits = (start - stored_offset) * 8
                remaining[(stored_offset, left_bits)] = _extract(
                    stored_value, 0, left_bits
                )
            if stored_end > end:
                right_offset = (end - stored_offset) * 8
                right_bits = (stored_end - end) * 8
                remaining[(end, right_bits)] = _extract(
                    stored_value, right_offset, right_bits
                )
        state.stack = remaining

    def _absolute_memory_offset(self, address: _Expr) -> int | None:
        if address.kind == "const":
            return int(address.value) & _mask(self.bits)
        if address.kind == "address":
            value = address.value
            if isinstance(value, tuple) and len(value) == 2 and value[0] == "absolute":
                offset = int(value[1])
                if 0 <= offset < 1 << self.bits:
                    return offset
            return None
        if (
            address.kind == "op"
            and address.value in {"add", "sub"}
            and len(address.args) == 2
        ):
            left, right = address.args
            if right.kind == "const":
                offset = self._absolute_memory_offset(left)
                if offset is None:
                    return None
                adjustment = int(right.value)
                if address.value == "sub":
                    adjustment = -adjustment
                return (offset + adjustment) & _mask(self.bits)
            if address.value == "add" and left.kind == "const":
                offset = self._absolute_memory_offset(right)
                if offset is not None:
                    return (offset + int(left.value)) & _mask(self.bits)
        return None

    @staticmethod
    def _memory_has_written(state: _State, offset: int, bits: int) -> bool:
        if bits <= 0:
            return False
        end = offset + (bits + 7) // 8
        return any(
            stored_offset < end and offset < stored_offset + (stored_bits + 7) // 8
            for stored_offset, stored_bits in state.memory
        )

    def _read_absolute_memory(self, state: _State, address: _Expr, bits: int) -> _Expr:
        offset = self._absolute_memory_offset(address)
        if (
            offset is None
            or bits <= 0
            or bits % 8
            or offset + bits // 8 > 1 << self.bits
        ):
            return _Expr("load", args=(address,), bits=bits)
        exact = state.memory.get((offset, bits))
        if exact is not None:
            return _resize(exact, bits)
        end = offset + bits // 8
        for (stored_offset, stored_bits), stored_value in state.memory.items():
            stored_end = stored_offset + stored_bits // 8
            if stored_offset <= offset and stored_end >= end:
                return _extract(stored_value, (offset - stored_offset) * 8, bits)
        if not self._memory_has_written(state, offset, bits):
            return _Expr("load", args=(address,), bits=bits)

        initial = _Expr("load", args=(address,), bits=bits)
        byte_values: list[_Expr] = []
        for byte_offset in range(offset, end):
            stored = next(
                (
                    (stored_offset, value)
                    for (stored_offset, stored_bits), value in state.memory.items()
                    if stored_offset <= byte_offset < stored_offset + stored_bits // 8
                ),
                None,
            )
            if stored is None:
                byte_values.append(_extract(initial, (byte_offset - offset) * 8, 8))
            else:
                stored_offset, value = stored
                byte_values.append(
                    _extract(value, (byte_offset - stored_offset) * 8, 8)
                )

        result = byte_values[0]
        for byte_value in byte_values[1:]:
            result = _concat(byte_value, result, result.bits)
        return _resize(result, bits)

    def _write_absolute_memory(
        self, state: _State, address: _Expr, bits: int, value: _Expr
    ) -> None:
        offset = self._absolute_memory_offset(address)
        if offset is None:
            state.memory.clear()
            state.stack.clear()
            return
        if bits <= 0 or bits % 8 or offset + bits // 8 > 1 << self.bits:
            state.memory.clear()
            return
        value = _resize(value, bits)
        end = offset + bits // 8
        remaining: dict[tuple[int, int], _Expr] = {}
        for (stored_offset, stored_bits), stored_value in state.memory.items():
            if stored_bits <= 0 or stored_bits % 8:
                remaining[(stored_offset, stored_bits)] = stored_value
                continue
            stored_end = stored_offset + stored_bits // 8
            if stored_end <= offset or stored_offset >= end:
                remaining[(stored_offset, stored_bits)] = stored_value
                continue
            if stored_offset < offset:
                left_bits = (offset - stored_offset) * 8
                remaining[(stored_offset, left_bits)] = _extract(
                    stored_value, 0, left_bits
                )
            if stored_end > end:
                right_offset = (end - stored_offset) * 8
                right_bits = (stored_end - end) * 8
                remaining[(end, right_bits)] = _extract(
                    stored_value, right_offset, right_bits
                )
        remaining[(offset, bits)] = value
        state.memory = remaining

    def _stack_name(self, offset: int) -> str:
        if self._declared_signature is not None:
            parameter_name = self._input_parameter_names_by_stack.get(offset)
            if parameter_name is not None:
                return parameter_name
            if offset > 0:
                return f"in_stack_{offset:X}"
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
        self,
        state: _State,
        name: str,
        instruction: Instruction,
        *,
        use_kind: str = "direct",
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
        self._record_pcode_register_read(family, value, width, instruction)
        self._record_parameter_sources(
            value,
            f"register-read:{family}",
            width,
            instruction=instruction,
            use_kind=use_kind,
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
            if self._current_instruction is not None:
                self._record_pcode_register_read(
                    family,
                    old_value,
                    native_width,
                    self._current_instruction,
                )
            updated = _insert_bits(old_value, value, offset, width, native_width)
        state.registers[family] = updated
        if self._current_instruction is not None:
            self._record_pcode_register_definition(
                family, updated, native_width, self._current_instruction
            )
        if mark_argument and family in self._tracked_argument_registers:
            state.dirty_arguments.add(family)
        return family

    def _read_xmm_register_slice(
        self,
        state: _State,
        register: str,
        offset: int,
        width: int,
        instruction: Instruction,
    ) -> _Expr:
        vector = state.xmm_registers.get(register, self._initial_xmm_register(register))
        value = _extract(vector, offset, width)
        sources = self._record_parameter_sources(
            value,
            f"vector-live-in-read:{register}",
            width,
            instruction=instruction,
        )
        self._record_pcode_register_read(register, value, width, instruction)
        expected_parameter = self._calling_convention.vector_parameter_name(register)
        if expected_parameter is not None:
            state.dirty_arguments.add(register)
            if expected_parameter in sources:
                self._note_vector_parameter_type(
                    register, instruction, expected_parameter
                )
        return value

    def _read_operand(
        self, state: _State, operand: _Operand, instruction: Instruction
    ) -> _Expr:
        width = operand.size * 8
        if operand.kind == "reg":
            if re.fullmatch(r"mm[0-7]", operand.reg):
                value = state.mmx_registers.get(
                    operand.reg, _var(f"in_{operand.reg}", 64)
                )
                self._record_pcode_register_read(operand.reg, value, width, instruction)
                self._record_parameter_sources(
                    value,
                    f"register-read:{operand.reg}",
                    width,
                    instruction=instruction,
                )
                return _resize(value, width)
            if re.fullmatch(r"xmm\d+", operand.reg):
                value = self._read_xmm_register_slice(
                    state, operand.reg, 0, min(width, 128), instruction
                )
                return _resize(value, width)
            return _resize(self._read_register(state, operand.reg, instruction), width)
        if operand.kind == "imm":
            value = _const(operand.imm, width)
            if self._paramid_recording_enabled:
                self._current_pcode_read_nodes.append(
                    self._paramid_graph.record_operand_read(
                        value,
                        address=instruction.address,
                        mnemonic=self._decode(instruction).mnemonic,
                        evidence="immediate-read",
                    )
                )
            return value
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
            updated = _zero_extend(_resize(value, min(width, 64)), 64)
            state.mmx_registers[operand.reg] = updated
            self._record_pcode_register_definition(
                operand.reg, updated, 64, instruction
            )
            return
        if operand.kind == "reg" and re.fullmatch(r"xmm\d+", operand.reg):
            updated = _zero_extend(_resize(value, min(width, 128)), 128)
            state.xmm_registers[operand.reg] = updated
            self._record_pcode_register_definition(
                operand.reg, updated, 128, instruction
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
            self._write_stack_value(state, offset, width, value)
            if self._paramid_recording_enabled:
                self._record_pcode_memory_store(
                    value,
                    self._memory_address(operand, state, instruction),
                    self._memory_location(operand, state, instruction),
                    self._memory_address_registers(operand),
                    instruction,
                    width,
                )
            return
        if operand.segment in {"fs", "gs"}:
            # Segment-based TEB/SEH setup is not an outgoing call-argument list.
            state.pending_pushes.clear()
        address = self._memory_address(operand, state, instruction)
        self._write_absolute_memory(state, address, width, value)
        if collect or self._paramid_recording_enabled:
            if self._paramid_recording_enabled:
                self._record_pcode_memory_store(
                    value,
                    address,
                    self._memory_location(operand, state, instruction),
                    self._memory_address_registers(operand),
                    instruction,
                    width,
                )
            if collect:
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
        value = (
            state.x87_stack[index]
            if 0 <= index < len(state.x87_stack)
            else _var(f"unknown_x87_st{index}", 80)
        )
        if self._current_instruction is not None:
            self._record_pcode_register_read(
                f"st{index}", value, 80, self._current_instruction
            )
        return value

    def _x87_set(self, state: _State, index: int, value: _Expr) -> None:
        if index < 0 or index >= 8:
            return
        while len(state.x87_stack) <= index:
            state.x87_stack.append(_var(f"unknown_x87_st{len(state.x87_stack)}", 80))
        updated = _resize(value, 80)
        state.x87_stack[index] = updated
        if self._current_instruction is not None:
            self._record_pcode_register_definition(
                f"st{index}", updated, 80, self._current_instruction
            )

    def _x87_push(self, state: _State, value: _Expr) -> None:
        state.x87_stack.insert(0, _resize(value, 80))
        if len(state.x87_stack) > 8:
            state.x87_stack.pop()
        if self._current_instruction is not None:
            for index, updated in enumerate(state.x87_stack):
                self._record_pcode_register_definition(
                    f"st{index}", updated, 80, self._current_instruction
                )

    def _x87_pop(self, state: _State, count: int = 1) -> None:
        del state.x87_stack[: min(max(count, 0), len(state.x87_stack))]
        if self._current_instruction is not None:
            for index in range(8):
                updated = (
                    state.x87_stack[index]
                    if index < len(state.x87_stack)
                    else _var(f"unknown_x87_st{index}", 80)
                )
                self._record_pcode_register_definition(
                    f"st{index}", updated, 80, self._current_instruction
                )

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
        if offset is not None and self._stack_has_written(state, offset, width):
            stored = self._read_stack_value(state, offset, width)
            if integer:
                value = _Expr("fp_from_int", args=(stored,), bits=80)
            elif stored.kind in {
                "fp_const",
                "fp_load",
                "fp_int_load",
                "fp_op",
                "fp_unary",
                "fp_cast",
                "fp_from_int",
            }:
                value = _resize(stored, 80)
            else:
                value = _var(
                    f"unknown_x87_stack_{abs(offset):X}_{instruction.address:X}", 80
                )
            self._record_pcode_memory_load(
                value,
                self._memory_address(operand, state, instruction),
                self._memory_location(operand, state, instruction),
                self._memory_address_registers(operand),
                instruction,
            )
            return value
        address = self._memory_address(operand, state, instruction)
        absolute_offset = self._absolute_memory_offset(address)
        if absolute_offset is not None and self._memory_has_written(
            state, absolute_offset, width
        ):
            stored = self._read_absolute_memory(state, address, width)
            if integer:
                value = _Expr("fp_from_int", args=(stored,), bits=80)
            elif stored.kind in {
                "fp_const",
                "fp_load",
                "fp_int_load",
                "fp_op",
                "fp_unary",
                "fp_cast",
                "fp_from_int",
            }:
                value = _resize(stored, 80)
            else:
                value = _var(
                    f"unknown_x87_memory_{absolute_offset:X}_{instruction.address:X}",
                    80,
                )
            self._record_pcode_memory_load(
                value,
                address,
                self._memory_location(operand, state, instruction),
                self._memory_address_registers(operand),
                instruction,
            )
            return value
        kind = "fp_int_load" if integer else "fp_load"
        value = _Expr(kind, (width, integer), args=(address,), bits=80)
        self._record_pcode_memory_load(
            value,
            address,
            self._memory_location(operand, state, instruction),
            self._memory_address_registers(operand),
            instruction,
        )
        return value

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
            self._write_stack_value(state, offset, width, stored)
            if self._paramid_recording_enabled:
                self._record_pcode_memory_store(
                    stored,
                    self._memory_address(operand, state, instruction),
                    self._memory_location(operand, state, instruction),
                    self._memory_address_registers(operand),
                    instruction,
                    width,
                )
            return
        address = self._memory_address(operand, state, instruction)
        self._write_absolute_memory(state, address, width, stored)
        if collect or self._paramid_recording_enabled:
            if self._paramid_recording_enabled:
                self._record_pcode_memory_store(
                    stored,
                    address,
                    self._memory_location(operand, state, instruction),
                    self._memory_address_registers(operand),
                    instruction,
                    width,
                )
            if collect:
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
            low_bits = 32 if mnemonic == "movd" else 64
            source_operand = operands[1]
            source_is_xmm = source_operand.kind == "reg" and re.fullmatch(
                r"xmm\d+", source_operand.reg
            )
            if source_is_xmm:
                source = self._read_xmm_register_slice(
                    state, source_operand.reg, 0, low_bits, instruction
                )
            else:
                source = self._read_operand(state, source_operand, instruction)
            if operands[0].kind == "reg" and re.fullmatch(r"xmm\d+", operands[0].reg):
                value = _zero_extend(_truncate(source, low_bits), 128)
            elif source_is_xmm:
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

        def read_memory(pointer: _Expr, register: str) -> _Expr:
            address = self._sleigh_address(pointer)
            stack_offset = self._sleigh_stack_offset(address)
            value = (
                self._read_stack_value(state, stack_offset, width)
                if stack_offset is not None
                else self._read_absolute_memory(state, address, width)
            )
            if self._paramid_recording_enabled:
                self._record_pcode_memory_load(
                    value,
                    address,
                    f"memory[{register}]",
                    (register,),
                    instruction,
                )
            return value

        def write_memory(value: _Expr, pointer: _Expr, register: str) -> None:
            if repeated and count.kind == "const" and int(count.value) == 0:
                return
            address = self._sleigh_address(pointer)
            if repeated and count.kind != "const":
                state.memory.clear()
                state.stack.clear()
            else:
                stack_offset = self._sleigh_stack_offset(address)
                if stack_offset is not None:
                    self._write_stack_value(state, stack_offset, width, value)
                else:
                    self._write_absolute_memory(state, address, width, value)
            self._record_pcode_memory_store(
                value,
                address,
                f"memory[{register}]",
                (register,),
                instruction,
                width,
            )

        if operation == "movs":
            loaded = read_memory(source, source_register)
            write_memory(loaded, destination, destination_register)
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
            write_memory(value, destination, destination_register)
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
                if self._paramid_recording_enabled:
                    self._record_pcode_memory_load(
                        value,
                        source,
                        f"memory[{source_register}]",
                        (source_register,),
                        instruction,
                    )
                self._write_register(state, accumulator, value, mark_argument=False)
                self._write_register(
                    state, count_register, _const(0, self.bits), mark_argument=False
                )
            else:
                value = read_memory(source, source_register)
                self._write_register(state, accumulator, value, mark_argument=False)
            advance(source_register, source, delta)
            if repeated:
                emit("/* repeated lods result is represented symbolically */")
            return True

        if operation in {"cmps", "scas"}:
            if operation == "cmps":
                left = read_memory(source, source_register)
                right = read_memory(destination, destination_register)
            else:
                accumulator = {8: "al", 16: "ax", 32: "eax", 64: "rax"}[width]
                left = self._read_register(state, accumulator, instruction)
                right = read_memory(destination, destination_register)
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
            value = self._read_stack_value(state, offset, width)
            self._record_parameter_sources(
                value,
                f"stack-live-in-read:{offset:+#x}",
                width,
                instruction=instruction,
            )
            if self._paramid_recording_enabled:
                self._record_pcode_memory_load(
                    value,
                    self._memory_address(operand, state, instruction),
                    self._memory_location(operand, state, instruction),
                    self._memory_address_registers(operand),
                    instruction,
                )
            return value
        address = self._memory_address(operand, state, instruction)
        value = self._read_absolute_memory(state, address, width)
        if self._paramid_recording_enabled:
            self._record_pcode_memory_load(
                value,
                address,
                self._memory_location(operand, state, instruction),
                self._memory_address_registers(operand),
                instruction,
            )
        return value

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
            parts.append(
                self._read_register(
                    state, operand.base, instruction, use_kind="indirect"
                )
            )
        if operand.index:
            index = self._read_register(
                state, operand.index, instruction, use_kind="indirect"
            )
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

    @staticmethod
    def _sleigh_lock_userop_name(operation) -> str | None:
        """Name the LOCK/UNLOCK userops in the packaged x86 SLEIGH specs.

        The local x86 and x86-64 languages assign these userops indices 0x11
        and 0x12. Other CALLOTHER operations are rejected unless a narrowly
        modeled local userop handler recognizes them.
        """
        if (
            operation.opcode != "CALLOTHER"
            or operation.output is not None
            or not operation.inputs
            or operation.inputs[0].space != "const"
        ):
            return None
        return {0x11: "LOCK", 0x12: "UNLOCK"}.get(operation.inputs[0].offset)

    @staticmethod
    def _sleigh_round_userop_name(operation) -> str | None:
        if (
            operation.opcode != "CALLOTHER"
            or not operation.inputs
            or operation.inputs[0].space != "const"
        ):
            return None
        return {
            0xE7: "roundps",
            0xE8: "roundss",
            0xE9: "roundpd",
            0xEA: "roundsd",
        }.get(operation.inputs[0].offset)

    @classmethod
    def _sleigh_lock_marker_positions(
        cls, operations: tuple
    ) -> tuple[int, int] | tuple[()] | None:
        """Validate the paired LOCK/UNLOCK delimiters in an atomic p-code op."""
        markers = [
            (index, name)
            for index, operation in enumerate(operations)
            if (name := cls._sleigh_lock_userop_name(operation)) is not None
        ]
        if not markers:
            return ()
        non_marker_ops = [
            index
            for index, operation in enumerate(operations)
            if operation.opcode != "IMARK"
        ]
        if (
            len(markers) != 2
            or [name for _index, name in markers] != ["LOCK", "UNLOCK"]
            or not non_marker_ops
            or markers[0][0] != non_marker_ops[0]
            or markers[1][0] != non_marker_ops[-1]
        ):
            return None
        return markers[0][0], markers[1][0]

    def _execute_sleigh_semantics(
        self,
        state: _State,
        instruction: Instruction,
        emitted: list[str],
        *,
        collect: bool,
    ) -> bool:
        """Lift a supported SLEIGH p-code sequence into symbolic state.

        The handwritten lifters above remain preferable for ABI-sensitive
        cases such as calls, returns, and x87. This interpreter handles generic
        integer p-code, ordinary memory operations, and bounded internal control
        flow for instructions that do not have a dedicated semantic lifter yet.
        """
        sleigh = self._sleigh_instruction_by_address.get(instruction.address)
        if sleigh is None or sleigh.error or not sleigh.ops:
            return False
        operations = sleigh.ops
        if any(
            operation.opcode not in self._SLEIGH_OPCODES for operation in operations
        ):
            return False
        for operation in operations:
            if operation.opcode == "CALLOTHER":
                if self._sleigh_lock_userop_name(operation) is not None:
                    continue
                round_name = self._sleigh_round_userop_name(operation)
                if (
                    round_name is None
                    or len(operation.inputs) != 4
                    or operation.output is None
                    or operation.output.space != "register"
                    or not re.fullmatch(r"xmm\d+", operation.output.display.lower())
                    or operation.output.size_bits != 128
                    or operation.inputs[3].space != "const"
                    or operation.inputs[3].offset & 0x4
                ):
                    return False
                continue
            expected_arity = self._SLEIGH_ARITIES.get(operation.opcode)
            if expected_arity is not None and len(operation.inputs) != expected_arity:
                return False
            if operation.opcode == "LOAD":
                if (
                    len(operation.inputs) != 2
                    or operation.inputs[0].space != "const"
                    or operation.output is None
                ):
                    return False
            elif operation.opcode == "STORE":
                if (
                    len(operation.inputs) != 3
                    or operation.inputs[0].space != "const"
                    or operation.output is not None
                ):
                    return False
            elif operation.opcode == "SUBPIECE":
                if len(operation.inputs) != 2 or operation.inputs[1].space != "const":
                    return False
            elif operation.opcode in {"BRANCH", "CBRANCH"}:
                target = operation.inputs[0]
                if (
                    operation.output is not None
                    or target.space not in {"const", "ram"}
                    or (
                        target.space == "ram"
                        and target.offset != instruction.next_address
                    )
                ):
                    return False
            elif (
                operation.opcode not in {"IMARK", "CALLOTHER"}
                and operation.output is None
            ):
                return False

        lock_markers = self._sleigh_lock_marker_positions(operations)
        if lock_markers is None:
            return False
        if any(operation.opcode in {"BRANCH", "CBRANCH"} for operation in operations):
            if lock_markers:
                lock_index, unlock_index = lock_markers
                for index, operation in enumerate(operations):
                    if operation.opcode not in {"BRANCH", "CBRANCH"}:
                        continue
                    target = index + _signed_value(
                        operation.inputs[0].offset,
                        operation.inputs[0].size_bits,
                    )
                    if not lock_index < target <= unlock_index:
                        return False
            return self._execute_branched_sleigh_semantics(
                state,
                instruction,
                operations,
                emitted,
                collect=collect,
                lock_markers=lock_markers,
            )

        scratch: dict[tuple[str, int, int], _Expr] = {}
        flag_updates: dict[str, _Expr] = {}
        previous_flags = state.flags
        for operation in operations:
            opcode = operation.opcode
            if opcode == "IMARK":
                continue
            if opcode == "CALLOTHER":
                marker = self._sleigh_lock_userop_name(operation)
                if marker is not None:
                    if collect:
                        emitted.append(f"/* {instruction.address:#x} */ {marker}();")
                    continue
                round_name = self._sleigh_round_userop_name(operation)
                assert round_name is not None and operation.output is not None
                values = self._read_sleigh_round_userop_values(
                    state, operation, scratch, instruction, round_name
                )
                value = self._evaluate_sleigh_round_userop(
                    round_name, values, operation.output.size_bits
                )
                if value is None:
                    return False
            elif opcode == "STORE":
                address = self._read_sleigh_varnode(
                    state, operation.inputs[1], scratch, instruction
                )
                value = self._read_sleigh_varnode(
                    state, operation.inputs[2], scratch, instruction
                )
                self._sleigh_store(
                    state,
                    address,
                    value,
                    operation.inputs[2].size_bits,
                    instruction,
                    emitted,
                    collect=collect,
                )
                continue
            elif opcode == "LOAD":
                address = self._read_sleigh_varnode(
                    state, operation.inputs[1], scratch, instruction
                )
                assert operation.output is not None
                value = self._sleigh_load(
                    state,
                    address,
                    operation.output.size_bits,
                    instruction,
                )
                stack_offset = self._sleigh_stack_offset(address)
                if collect and stack_offset is None:
                    size_bits = operation.output.size_bits
                    source = self._render(value)
                    name = (
                        f"pcode_load_{size_bits}_{instruction.address:X}_"
                        f"{operation.sequence}"
                    )
                    emitted.append(f"{name} = {source};")
                    value = _var(name, size_bits)
            else:
                values = tuple(
                    self._read_sleigh_varnode(state, varnode, scratch, instruction)
                    for varnode in operation.inputs
                )
                assert operation.output is not None
                value = self._evaluate_sleigh_operation(
                    opcode, values, operation.output.size_bits
                )
                if value is None:
                    return False
            output = operation.output
            assert output is not None
            value = _resize(value, output.size_bits)
            self._write_sleigh_varnode_scratch(scratch, output, value)
            if output.space == "register":
                register_name = output.display.lower()
                if register_name in self._PCODE_FLAG_NAMES:
                    flag_updates[register_name] = value
                    self._record_pcode_register_definition(
                        register_name, value, output.size_bits, instruction
                    )
                else:
                    self._write_sleigh_register(
                        state, register_name, value, output.size_bits, instruction
                    )
            elif output.space == "ram":
                self._sleigh_store(
                    state,
                    _const(output.offset, self.bits),
                    value,
                    output.size_bits,
                    instruction,
                    emitted,
                    collect=collect,
                )

        if flag_updates:
            state.flags = self._sleigh_flags_after_pcode(
                previous_flags, flag_updates, instruction
            )
        return True

    def _execute_branched_sleigh_semantics(
        self,
        state: _State,
        instruction: Instruction,
        operations: tuple,
        emitted: list[str],
        *,
        collect: bool,
        lock_markers: tuple[int, ...] = (),
    ) -> bool:
        """Interpret internal p-code branches with bounded symbolic path splitting.

        This covers control flow *inside one SLEIGH translation* (for example
        the bit-scan loops used by BSF/BSR). Machine-level branches remain the
        recovered CFG's responsibility. Memory effects are staged per path and
        committed to the output only after every path completes successfully.
        """
        for operation in operations:
            if operation.opcode == "IMARK":
                continue
            for index, varnode in enumerate(operation.inputs):
                if (
                    varnode.space == "ram"
                    and operation.opcode in {"BRANCH", "CBRANCH"}
                    and index == 0
                    and varnode.offset != instruction.next_address
                ):
                    return False
            if operation.output is not None and operation.output.space not in {
                "unique",
                "register",
                "ram",
            }:
                return False

        initial = state.copy()
        original_recording = self._paramid_recording_enabled
        used_before = set(self._used_parameter_names)
        evidence_before = {
            name: set(evidence) for name, evidence in self._parameter_evidence.items()
        }
        widths_before = {
            name: set(widths) for name, widths in self._parameter_read_widths.items()
        }
        self._paramid_recording_enabled = False
        worklist = [_SleighExecutionPath(0, initial.copy(), {}, {}, _const(1, 1))]
        completed: list[_SleighExecutionPath] = []
        memory_reads: dict[tuple[_Expr, _Expr], tuple[_Expr, _Expr, str]] = {}
        load_snapshots: list[tuple[_Expr, _Expr, _Expr, _Expr, int, str]] = []
        load_snapshot_indices: dict[tuple[_Expr, _Expr, int, int], int] = {}
        load_index = 0
        max_paths = max(128, self.bits + 16)
        max_steps = max(4096, len(operations) * (self.bits + 1) * 8)
        failed = False

        try:
            while worklist and not failed:
                path = worklist.pop()
                while path.pc < len(operations):
                    if path.steps >= max_steps:
                        failed = True
                        break
                    operation = operations[path.pc]
                    index = path.pc
                    path.pc += 1
                    path.steps += 1
                    opcode = operation.opcode
                    if opcode == "IMARK":
                        continue
                    if opcode == "CALLOTHER":
                        if self._sleigh_lock_userop_name(operation) is not None:
                            continue
                        round_name = self._sleigh_round_userop_name(operation)
                        if round_name is None or operation.output is None:
                            failed = True
                            break
                        values = self._read_sleigh_round_userop_values(
                            path.state,
                            operation,
                            path.scratch,
                            instruction,
                            round_name,
                        )
                        value = self._evaluate_sleigh_round_userop(
                            round_name, values, operation.output.size_bits
                        )
                        if value is None:
                            failed = True
                            break
                        output = operation.output
                        value = _resize(value, output.size_bits)
                        self._write_sleigh_varnode_scratch(path.scratch, output, value)
                        if output.space == "register":
                            self._write_sleigh_register(
                                path.state,
                                output.display.lower(),
                                value,
                                output.size_bits,
                                instruction,
                            )
                        continue
                    if opcode in {"BRANCH", "CBRANCH"}:
                        branch_target = operation.inputs[0]
                        if branch_target.space == "ram":
                            if branch_target.offset != instruction.next_address:
                                failed = True
                                break
                            target = len(operations)
                        else:
                            branch_offset = _signed_value(
                                branch_target.offset, branch_target.size_bits
                            )
                            target = index + branch_offset
                        if target < 0 or target > len(operations):
                            failed = True
                            break
                        if opcode == "BRANCH":
                            path.pc = target
                            continue

                        raw_condition = self._read_sleigh_varnode(
                            path.state,
                            operation.inputs[1],
                            path.scratch,
                            instruction,
                        )
                        raw_condition = _unwrap_sleigh_boolean(raw_condition)
                        condition = _binary(
                            "ne", raw_condition, _const(0, raw_condition.bits), 1
                        )
                        if condition.kind == "const":
                            if int(condition.value) != 0:
                                path.pc = target
                            continue

                        true_condition = _combine_sleigh_path_condition(
                            path.condition, condition
                        )
                        false_condition = _combine_sleigh_path_condition(
                            path.condition, _unary("boolnot", condition, 1)
                        )
                        true_path = _copy_sleigh_execution_path(
                            path, target, true_condition
                        )
                        false_path = _copy_sleigh_execution_path(
                            path, path.pc, false_condition
                        )
                        _add_sleigh_branch_fact(true_path, condition, True)
                        _add_sleigh_branch_fact(false_path, condition, False)
                        if _sleigh_path_constraints_consistent(false_path):
                            worklist.append(false_path)
                        if _sleigh_path_constraints_consistent(true_path):
                            worklist.append(true_path)
                        if len(worklist) + len(completed) > max_paths:
                            failed = True
                        break

                    if opcode == "STORE":
                        address = self._read_sleigh_varnode(
                            path.state,
                            operation.inputs[1],
                            path.scratch,
                            instruction,
                        )
                        stored_value = self._read_sleigh_varnode(
                            path.state,
                            operation.inputs[2],
                            path.scratch,
                            instruction,
                        )
                        size_bits = operation.inputs[2].size_bits
                        stack_offset = self._sleigh_stack_offset(address)
                        self._sleigh_store(
                            path.state,
                            address,
                            stored_value,
                            size_bits,
                            instruction,
                            [],
                            collect=False,
                        )
                        path.memory_epoch += 1
                        store_address = (
                            address
                            if stack_offset is not None
                            else self._sleigh_address(address)
                        )
                        location = (
                            f"stack[{stack_offset:+#x}]"
                            if stack_offset is not None
                            else f"memory[{self._render(store_address)}]"
                        )
                        path.memory_stores.append(
                            (
                                path.condition,
                                store_address,
                                _resize(stored_value, size_bits),
                                size_bits,
                                location,
                            )
                        )
                        continue

                    if opcode == "LOAD":
                        assert operation.output is not None
                        address = self._read_sleigh_varnode(
                            path.state,
                            operation.inputs[1],
                            path.scratch,
                            instruction,
                        )
                        value = self._sleigh_load(
                            path.state,
                            address,
                            operation.output.size_bits,
                            instruction,
                        )
                        stack_offset = self._sleigh_stack_offset(address)
                        load_address = (
                            address
                            if stack_offset is not None
                            else self._sleigh_address(address)
                        )
                        location = (
                            f"stack[{stack_offset:+#x}]"
                            if stack_offset is not None
                            else f"memory[{self._render(load_address)}]"
                        )
                        memory_reads[(load_address, value)] = (
                            load_address,
                            value,
                            location,
                        )
                        if collect and stack_offset is None:
                            snapshot_key = (
                                load_address,
                                value,
                                operation.output.size_bits,
                                path.memory_epoch,
                            )
                            snapshot_index = load_snapshot_indices.get(snapshot_key)
                            if snapshot_index is None:
                                name = (
                                    f"pcode_load_{operation.output.size_bits}_"
                                    f"{instruction.address:X}_{load_index}"
                                )
                                load_index += 1
                                temporary = _var(name, operation.output.size_bits)
                                load_snapshot_indices[snapshot_key] = len(
                                    load_snapshots
                                )
                                load_snapshots.append(
                                    (
                                        path.condition,
                                        load_address,
                                        value,
                                        temporary,
                                        operation.output.size_bits,
                                        location,
                                    )
                                )
                            else:
                                snapshot = load_snapshots[snapshot_index]
                                temporary = snapshot[3]
                                load_snapshots[snapshot_index] = (
                                    _combine_sleigh_path_condition_or(
                                        snapshot[0], path.condition
                                    ),
                                    *snapshot[1:],
                                )
                            value = temporary
                    else:
                        values_list = []
                        for varnode in operation.inputs:
                            operand_value = self._read_sleigh_varnode(
                                path.state, varnode, path.scratch, instruction
                            )
                            if varnode.space == "ram":
                                address = self._sleigh_address(
                                    _const(varnode.offset, self.bits)
                                )
                                location = f"memory[{self._render(address)}]"
                                memory_reads[(address, operand_value)] = (
                                    address,
                                    operand_value,
                                    location,
                                )
                                if collect:
                                    snapshot_key = (
                                        address,
                                        operand_value,
                                        varnode.size_bits,
                                        path.memory_epoch,
                                    )
                                    snapshot_index = load_snapshot_indices.get(
                                        snapshot_key
                                    )
                                    if snapshot_index is None:
                                        name = (
                                            f"pcode_load_{varnode.size_bits}_"
                                            f"{instruction.address:X}_{load_index}"
                                        )
                                        load_index += 1
                                        temporary = _var(name, varnode.size_bits)
                                        load_snapshot_indices[snapshot_key] = len(
                                            load_snapshots
                                        )
                                        load_snapshots.append(
                                            (
                                                path.condition,
                                                address,
                                                operand_value,
                                                temporary,
                                                varnode.size_bits,
                                                location,
                                            )
                                        )
                                    else:
                                        snapshot = load_snapshots[snapshot_index]
                                        temporary = snapshot[3]
                                        load_snapshots[snapshot_index] = (
                                            _combine_sleigh_path_condition_or(
                                                snapshot[0], path.condition
                                            ),
                                            *snapshot[1:],
                                        )
                                    operand_value = temporary
                            values_list.append(operand_value)
                        assert operation.output is not None
                        value = self._evaluate_sleigh_operation(
                            opcode, tuple(values_list), operation.output.size_bits
                        )
                    if value is None:
                        failed = True
                        break
                    output = operation.output
                    value = _resize(value, output.size_bits)
                    self._write_sleigh_varnode_scratch(path.scratch, output, value)
                    if output.space == "ram":
                        address = _const(output.offset, self.bits)
                        self._sleigh_store(
                            path.state,
                            address,
                            value,
                            output.size_bits,
                            instruction,
                            [],
                            collect=False,
                        )
                        path.memory_epoch += 1
                        stack_offset = self._sleigh_stack_offset(address)
                        store_address = (
                            address
                            if stack_offset is not None
                            else self._sleigh_address(address)
                        )
                        location = (
                            f"stack[{stack_offset:+#x}]"
                            if stack_offset is not None
                            else f"memory[{self._render(store_address)}]"
                        )
                        path.memory_stores.append(
                            (
                                path.condition,
                                store_address,
                                _resize(value, output.size_bits),
                                output.size_bits,
                                location,
                            )
                        )
                        continue
                    if output.space == "register":
                        register_name = output.display.lower()
                        if register_name in self._PCODE_FLAG_NAMES:
                            path.flag_updates[register_name] = value
                        else:
                            self._write_sleigh_register(
                                path.state,
                                register_name,
                                value,
                                output.size_bits,
                                instruction,
                            )
                if failed:
                    break
                if path.pc >= len(operations):
                    path.state.flags = self._sleigh_flags_after_pcode(
                        initial.flags, path.flag_updates, instruction
                    )
                    completed.append(path)
                    if len(completed) > max_paths:
                        failed = True

            if not completed:
                failed = True
            if failed:
                self._used_parameter_names = used_before
                self._parameter_evidence = defaultdict(set, evidence_before)
                self._parameter_read_widths = defaultdict(set, widths_before)
                return False
        finally:
            self._paramid_recording_enabled = original_recording

        memory_writes = {
            store: store for path in completed for store in path.memory_stores
        }
        merged = self._merge_sleigh_execution_paths(initial, completed, instruction)
        self._summarize_sleigh_bit_scan(merged, initial, instruction, load_snapshots)
        state.__dict__.update(merged.__dict__)
        if original_recording:
            self._record_branched_sleigh_register_effects(
                initial,
                state,
                operations,
                instruction,
                tuple(memory_reads.values()),
                tuple(memory_writes.values()),
            )
        if collect:
            if lock_markers:
                emitted.append(f"/* {instruction.address:#x} */ LOCK();")
            for (
                condition,
                _address,
                loaded_value,
                temporary,
                _size_bits,
                _location,
            ) in load_snapshots:
                if condition.kind == "const" and int(condition.value) == 0:
                    continue
                source = self._render(loaded_value)
                assignment = f"{temporary.value} = {source};"
                if condition.kind != "const":
                    assignment = f"if ({self._render(condition)}) {assignment}"
                emitted.append(assignment)
            for (
                condition,
                address,
                value,
                size_bits,
                location,
            ) in memory_writes.values():
                if location.startswith("stack[") or (
                    condition.kind == "const" and int(condition.value) == 0
                ):
                    continue
                store = (
                    f"*({_memory_type(size_bits)} *)({self._render(address)}) = "
                    f"{self._render(value)};"
                )
                if condition.kind != "const":
                    store = f"if ({self._render(condition)}) {store}"
                emitted.append(store)
            if lock_markers:
                emitted.append(f"/* {instruction.address:#x} */ UNLOCK();")
        return True

    def _merge_sleigh_execution_paths(
        self,
        initial: _State,
        paths: list[_SleighExecutionPath],
        instruction: Instruction,
    ) -> _State:
        merged = initial.copy()

        def merge_expression(get_value, fallback: _Expr) -> _Expr:
            result = fallback
            for path in reversed(paths):
                candidate = get_value(path.state)
                result = _select(
                    path.condition,
                    candidate,
                    result,
                    max(1, candidate.bits, result.bits),
                )
            return result

        def merge_mapping(name: str, fallback_for) -> None:
            initial_mapping = getattr(initial, name)
            merged_mapping = getattr(merged, name)
            keys = set(initial_mapping).union(
                *(getattr(path.state, name).keys() for path in paths)
            )
            for key in keys:
                fallback = initial_mapping.get(key)
                if fallback is None:
                    fallback = fallback_for(key, paths)
                base_value = getattr(paths[0].state, name).get(key, fallback)
                value = merge_expression(
                    lambda path_state, key=key, fallback=fallback: getattr(
                        path_state, name
                    ).get(key, fallback),
                    base_value,
                )
                if key in initial_mapping or value != fallback:
                    merged_mapping[key] = value
                else:
                    merged_mapping.pop(key, None)

        merge_mapping(
            "registers",
            lambda family, _paths: self._initial_register(family),
        )
        merge_mapping(
            "pcode_registers",
            lambda name, path_list: _var(
                f"in_{_c_identifier(name)}",
                next(
                    (
                        value.bits
                        for path in path_list
                        if (value := path.state.pcode_registers.get(name)) is not None
                    ),
                    self.bits,
                ),
            ),
        )
        merge_mapping(
            "mmx_registers",
            lambda name, _paths: _var(f"in_{name}", 64),
        )
        merge_mapping(
            "xmm_registers",
            lambda name, _paths: self._initial_xmm_register(name),
        )

        stack_boundaries = {
            boundary
            for candidate in (initial, *(path.state for path in paths))
            for offset, width in candidate.stack
            for boundary in (offset, offset + (width + 7) // 8)
        }
        ordered_boundaries = sorted(stack_boundaries)
        for start, end in itertools.pairwise(ordered_boundaries):
            if end <= start:
                continue
            width = (end - start) * 8
            fallback = self._read_stack_value(initial, start, width)
            if not any(
                self._stack_has_written(candidate, start, width)
                for candidate in (initial, *(path.state for path in paths))
            ):
                continue
            value = fallback
            for path in reversed(paths):
                candidate = self._read_stack_value(path.state, start, width)
                value = _select(
                    path.condition,
                    candidate,
                    value,
                    max(1, candidate.bits, value.bits),
                )
            self._write_stack_value(merged, start, width, value)

        memory_states = (initial, *(path.state for path in paths))
        memory_boundaries = {
            boundary
            for candidate in memory_states
            for offset, width in candidate.memory
            for boundary in (offset, offset + (width + 7) // 8)
        }
        for start, end in itertools.pairwise(sorted(memory_boundaries)):
            if end <= start:
                continue
            width = (end - start) * 8
            if not any(
                self._memory_has_written(candidate, start, width)
                for candidate in memory_states
            ):
                continue
            address = _Expr("address", value=("absolute", start), bits=self.bits)
            fallback = self._read_absolute_memory(initial, address, width)
            value = fallback
            for path in reversed(paths):
                candidate = self._read_absolute_memory(path.state, address, width)
                value = _select(
                    path.condition,
                    candidate,
                    value,
                    max(1, candidate.bits, value.bits),
                )
            self._write_absolute_memory(merged, address, width, value)

        xmm_names = set(initial.xmm_widths).union(
            *(path.state.xmm_widths.keys() for path in paths)
        )
        for name in xmm_names:
            widths = {path.state.xmm_widths.get(name, 128) for path in paths}
            merged.xmm_widths[name] = widths.pop() if len(widths) == 1 else 128

        flags = [path.state.flags for path in paths]
        if all(flag == flags[0] for flag in flags):
            merged.flags = flags[0]
        elif any(flag is not None for flag in flags):

            def merge_flag(name: str) -> _Expr:
                fallback = self._sleigh_flag_value(paths[0].state.flags, name)
                return merge_expression(
                    lambda path_state: self._sleigh_flag_value(path_state.flags, name),
                    fallback,
                )

            merged.flags = _Flags(
                "pcode",
                bits=self.bits,
                carry=merge_flag("cf"),
                zero=merge_flag("zf"),
                sign=merge_flag("sf"),
                overflow=merge_flag("of"),
                parity=merge_flag("pf"),
            )

        x87_depths = {len(path.state.x87_stack) for path in paths}
        if len(x87_depths) == 1:
            merged.x87_stack = [
                merge_expression(
                    lambda path_state, index=index: path_state.x87_stack[index],
                    paths[0].state.x87_stack[index],
                )
                for index in range(next(iter(x87_depths)))
            ]
        else:
            merged.x87_stack = [
                _var(f"unknown_x87_st{index}_loc_{instruction.address:X}", 80)
                for index in range(max(x87_depths, default=0))
            ]
        merged.x87_flags = (
            paths[0].state.x87_flags
            if all(path.state.x87_flags == paths[0].state.x87_flags for path in paths)
            else None
        )
        merged.x87_status_flags = (
            paths[0].state.x87_status_flags
            if all(
                path.state.x87_status_flags == paths[0].state.x87_status_flags
                for path in paths
            )
            else None
        )
        for attribute in ("direction_flag", "stack_delta", "frame_delta"):
            values = {getattr(path.state, attribute) for path in paths}
            setattr(merged, attribute, values.pop() if len(values) == 1 else None)
        pending_pushes = {tuple(path.state.pending_pushes) for path in paths}
        merged.pending_pushes = (
            list(pending_pushes.pop()) if len(pending_pushes) == 1 else []
        )
        merged.dirty_arguments = set().union(
            *(path.state.dirty_arguments for path in paths)
        )
        return merged

    def _summarize_sleigh_bit_scan(
        self,
        state: _State,
        initial: _State,
        instruction: Instruction,
        load_snapshots: list[tuple[_Expr, _Expr, _Expr, _Expr, int, str]],
    ) -> None:
        decoded = self._decode(instruction)
        if (
            decoded.mnemonic not in {"bsf", "bsr"}
            or len(decoded.operands) != 2
            or decoded.operands[0].kind != "reg"
            or decoded.operands[1].kind not in {"reg", "mem"}
        ):
            return
        original_recording = self._paramid_recording_enabled
        original_instruction = self._current_instruction
        self._paramid_recording_enabled = False
        self._current_instruction = None
        try:
            source_snapshot = next(
                (
                    snapshot[3]
                    for snapshot in load_snapshots
                    if decoded.operands[1].kind == "mem"
                    and snapshot[4] == decoded.operands[1].size * 8
                ),
                None,
            )
            source = source_snapshot or self._read_operand(
                initial.copy(), decoded.operands[1], instruction
            )
            if source.kind == "const":
                return
            result = _Expr(
                "pcode_builtin",
                decoded.mnemonic,
                args=(source,),
                bits=decoded.operands[0].size * 8,
            )
            self._write_sleigh_register(
                state,
                decoded.operands[0].reg,
                result,
                decoded.operands[0].size * 8,
                instruction,
            )
        finally:
            self._paramid_recording_enabled = original_recording
            self._current_instruction = original_instruction

    def _record_branched_sleigh_register_effects(
        self,
        initial: _State,
        final: _State,
        operations: tuple,
        instruction: Instruction,
        memory_reads: tuple[tuple[_Expr, _Expr, str], ...],
        memory_writes: tuple[tuple[_Expr, _Expr, _Expr, int, str], ...],
    ) -> None:
        written_registers: set[tuple[str, int, int]] = set()
        read_registers: dict[tuple[str, int, int], SleighVarnode] = {}
        for operation in operations:
            if operation.opcode == "IMARK":
                continue
            for varnode in operation.inputs:
                key = _sleigh_varnode_key(varnode)
                if varnode.space == "register" and key not in written_registers:
                    read_registers.setdefault(key, varnode)
            if operation.output is not None and operation.output.space == "register":
                written_registers.add(_sleigh_varnode_key(operation.output))
        self._current_pcode_read_nodes.clear()
        for address, value, location in memory_reads:
            self._record_pcode_memory_load(value, address, location, (), instruction)
        for varnode in read_registers.values():
            self._read_sleigh_varnode(initial, varnode, {}, instruction)
        inputs = tuple(self._current_pcode_read_nodes)
        for _, address, value, size_bits, location in memory_writes:
            self._record_pcode_memory_store(
                value, address, location, (), instruction, size_bits
            )

        for family in set(initial.registers).union(final.registers):
            old = initial.registers.get(family, self._initial_register(family))
            value = final.registers.get(family, self._initial_register(family))
            if value != old:
                self._record_pcode_register_definition(
                    family, value, value.bits, instruction, input_nodes=inputs
                )
        for name in set(initial.pcode_registers).union(final.pcode_registers):
            old = initial.pcode_registers.get(name)
            value = final.pcode_registers.get(name)
            if value is not None and value != old:
                self._record_pcode_register_definition(
                    name, value, value.bits, instruction, input_nodes=inputs
                )
        for name in set(initial.mmx_registers).union(final.mmx_registers):
            old = initial.mmx_registers.get(name, _var(f"in_{name}", 64))
            value = final.mmx_registers.get(name, _var(f"in_{name}", 64))
            if value != old:
                self._record_pcode_register_definition(
                    name, value, value.bits, instruction, input_nodes=inputs
                )
        for name in set(initial.xmm_registers).union(final.xmm_registers):
            old = initial.xmm_registers.get(name, self._initial_xmm_register(name))
            value = final.xmm_registers.get(name, self._initial_xmm_register(name))
            if value != old:
                self._record_pcode_register_definition(
                    name, value, value.bits, instruction, input_nodes=inputs
                )
        flag_names = {"cf", "zf", "sf", "of", "pf"}
        for name in flag_names:
            old = self._sleigh_flag_value(initial.flags, name)
            value = self._sleigh_flag_value(final.flags, name)
            if value != old:
                self._record_pcode_register_definition(
                    name, value, 1, instruction, input_nodes=inputs
                )
        self._current_pcode_read_nodes.clear()

    def _sleigh_flags_after_pcode(
        self,
        previous_flags: _Flags | None,
        flag_updates: dict[str, _Expr],
        instruction: Instruction,
    ) -> _Flags | None:
        if not flag_updates:
            return previous_flags
        preserve_unwritten = (
            self._decode(instruction).mnemonic in self._SLEIGH_PRESERVED_FLAG_MNEMONICS
        )

        def updated_flag(name: str) -> _Expr:
            value = flag_updates.get(name)
            if value is not None:
                return value
            if preserve_unwritten:
                return self._sleigh_flag_value(previous_flags, name)
            return _var(f"flag_{name}_{instruction.address:X}", 1)

        return _Flags(
            "pcode",
            bits=self.bits,
            carry=updated_flag("cf"),
            zero=updated_flag("zf"),
            sign=updated_flag("sf"),
            overflow=updated_flag("of"),
            parity=updated_flag("pf"),
        )

    def _read_sleigh_round_userop_values(
        self,
        state: _State,
        operation,
        scratch: dict[tuple[str, int, int], _Expr],
        instruction: Instruction,
        name: str,
    ) -> tuple[_Expr, ...]:
        values = []
        scalar = name.endswith(("ss", "sd"))
        for index, varnode in enumerate(operation.inputs[1:]):
            if index == 0 and not scalar:
                # Packed ROUND* replaces every lane; its destination is not read.
                values.append(_const(0, varnode.size_bits))
                continue
            # Scalar ROUND* preserves the high destination lanes.  Do not
            # count that implementation-only read as a live-in unless a later
            # instruction actually consumes those preserved lanes.
            value = self._read_sleigh_varnode(
                state,
                varnode,
                scratch,
                instruction,
                track_parameter_sources=index != 0,
            )
            values.append(value)
        return tuple(values)

    def _evaluate_sleigh_round_userop(
        self, name: str, values: tuple[_Expr, ...], output_bits: int
    ) -> _Expr | None:
        if len(values) != 3 or output_bits != 128 or values[2].kind != "const":
            return None
        destination, source, immediate = values
        control = int(immediate.value)
        if control & 0x4:
            return None
        mode = {0: "round", 1: "floor", 2: "ceil", 3: "trunc"}[control & 0x3]
        lane_bits = 32 if name.endswith(("ps", "ss")) else 64
        scalar = name.endswith(("ss", "sd"))
        if destination.bits < 128:
            return None
        result = _resize(destination, 128) if scalar else _const(0, 128)
        lane_count = 1 if scalar else 128 // lane_bits
        if source.bits < lane_count * lane_bits:
            return None
        for lane in range(lane_count):
            offset = lane * lane_bits
            lane_value = _extract(source, offset, lane_bits)
            floating = _simd_float(lane_value, lane_bits)
            rounded = _fp_unary(mode, floating)
            raw_result = _fp_to_bits(rounded, lane_bits)
            result = _insert_bits(result, raw_result, offset, lane_bits, 128)
        return result

    def _evaluate_sleigh_operation(
        self, opcode: str, values: tuple[_Expr, ...], output_bits: int
    ) -> _Expr | None:
        if opcode in {"COPY", "INT_ZEXT", "INT_SEXT"} and len(values) == 1:
            if opcode == "INT_SEXT":
                return _sign_extend(values[0], output_bits)
            if opcode == "INT_ZEXT":
                return _zero_extend(values[0], output_bits)
            return _resize(values[0], output_bits)
        if opcode in {"INT_2COMP", "INT_NEGATE", "BOOL_NEGATE"} and len(values) == 1:
            operator = {
                "INT_2COMP": "neg",
                "INT_NEGATE": "bitnot",
                "BOOL_NEGATE": "boolnot",
            }[opcode]
            return _unary(operator, values[0], output_bits)
        if opcode == "SUBPIECE" and len(values) == 2:
            return _extract(values[0], int(values[1].value) * 8, output_bits)
        if opcode == "PIECE" and len(values) == 2:
            return _concat(values[0], values[1], values[1].bits)
        if opcode in {"POPCOUNT", "LZCOUNT"} and len(values) == 1:
            source = values[0]
            if source.kind == "const":
                raw = int(source.value) & _mask(source.bits)
                result = (
                    raw.bit_count()
                    if opcode == "POPCOUNT"
                    else source.bits - raw.bit_length()
                )
                return _const(result, output_bits)
            return _Expr(
                "pcode_builtin",
                "popcount" if opcode == "POPCOUNT" else "lzcount",
                args=(source,),
                bits=output_bits,
            )
        if (
            opcode
            in {
                "FLOAT_EQUAL",
                "FLOAT_NOTEQUAL",
                "FLOAT_LESS",
                "FLOAT_LESSEQUAL",
            }
            and len(values) == 2
        ):
            if any(value.bits not in {32, 64} for value in values):
                return None
            comparison = {
                "FLOAT_EQUAL": "eq",
                "FLOAT_NOTEQUAL": "ne",
                "FLOAT_LESS": "lt",
                "FLOAT_LESSEQUAL": "le",
            }[opcode]
            left, right = (_simd_float(value, value.bits) for value in values)
            return _fp_compare(comparison, left, right)
        if opcode in {"FLOAT_ADD", "FLOAT_DIV", "FLOAT_MULT", "FLOAT_SUB"}:
            if (
                len(values) != 2
                or output_bits not in {32, 64}
                or any(value.bits not in {32, 64} for value in values)
            ):
                return None
            operation = {
                "FLOAT_ADD": "add",
                "FLOAT_DIV": "div",
                "FLOAT_MULT": "mul",
                "FLOAT_SUB": "sub",
            }[opcode]
            left, right = (_simd_float(value, value.bits) for value in values)
            return _fp_to_bits(_fp_binary(operation, left, right), output_bits)
        if opcode == "FLOAT_NAN" and len(values) == 1:
            if values[0].bits not in {32, 64}:
                return None
            return _fp_is_nan(_simd_float(values[0], values[0].bits))
        if opcode in {"FLOAT_NEG", "FLOAT_ABS", "FLOAT_SQRT"} and len(values) == 1:
            if values[0].bits not in {32, 64} or output_bits not in {32, 64}:
                return None
            operation = {
                "FLOAT_NEG": "neg",
                "FLOAT_ABS": "abs",
                "FLOAT_SQRT": "sqrt",
            }[opcode]
            source = _simd_float(values[0], values[0].bits)
            return _fp_to_bits(_fp_unary(operation, source), output_bits)
        if opcode == "FLOAT_INT2FLOAT" and len(values) == 1:
            if values[0].bits > 64 or output_bits not in {32, 64}:
                return None
            return _fp_to_bits(_fp_from_integer(values[0]), output_bits)
        if opcode == "FLOAT_FLOAT2FLOAT" and len(values) == 1:
            if values[0].bits not in {32, 64} or output_bits not in {32, 64}:
                return None
            source = _simd_float(values[0], values[0].bits)
            cast = _Expr("fp_cast", output_bits, args=(source,), bits=80)
            return _fp_to_bits(cast, output_bits)
        if opcode == "FLOAT_TRUNC" and len(values) == 1:
            if values[0].bits not in {32, 64} or output_bits not in {8, 16, 32, 64}:
                return None
            return _simd_cvtt_to_int(values[0], output_bits, input_bits=values[0].bits)
        if opcode in {"FLOAT_CEIL", "FLOAT_FLOOR", "FLOAT_ROUND"} and len(values) == 1:
            if values[0].bits not in {32, 64} or output_bits not in {32, 64}:
                return None
            operation = {
                "FLOAT_CEIL": "ceil",
                "FLOAT_FLOOR": "floor",
                "FLOAT_ROUND": "round",
            }[opcode]
            source = _simd_float(values[0], values[0].bits)
            return _fp_to_bits(_fp_unary(operation, source), output_bits)

        binary = {
            "INT_ADD": "add",
            "INT_SUB": "sub",
            "INT_MULT": "mul",
            "INT_DIV": "udiv",
            "INT_SDIV": "sdiv",
            "INT_REM": "urem",
            "INT_SREM": "srem",
            "INT_AND": "and",
            "INT_OR": "or",
            "INT_XOR": "xor",
            "INT_LEFT": "shl",
            "INT_RIGHT": "lshr",
            "INT_SRIGHT": "ashr",
            "INT_EQUAL": "eq",
            "INT_NOTEQUAL": "ne",
            "INT_LESS": "ult",
            "INT_LESSEQUAL": "ule",
            "INT_SLESS": "slt",
            "INT_SLESSEQUAL": "sle",
            "BOOL_AND": "land",
            "BOOL_OR": "lor",
            "BOOL_XOR": "xor",
        }
        if opcode in {"INT_LEFT", "INT_RIGHT", "INT_SRIGHT"} and len(values) == 2:
            source, amount = values
            if amount.kind == "const":
                shift = int(amount.value)
                if shift >= source.bits:
                    if opcode == "INT_SRIGHT":
                        sign = _binary("slt", source, _const(0, source.bits), 1)
                        return _unary("neg", _resize(sign, output_bits), output_bits)
                    return _const(0, output_bits)
                return _binary(binary[opcode], source, amount, output_bits)
            shifted = _binary(binary[opcode], source, amount, output_bits)
            in_range = _binary("ult", amount, _const(source.bits, amount.bits), 1)
            if opcode == "INT_SRIGHT":
                sign = _binary("slt", source, _const(0, source.bits), 1)
                too_large = _unary("neg", _resize(sign, output_bits), output_bits)
            else:
                too_large = _const(0, output_bits)
            return _select(in_range, shifted, too_large, output_bits)
        if opcode in binary and len(values) == 2:
            return _binary(binary[opcode], values[0], values[1], output_bits)
        if opcode in {"INT_CARRY", "INT_SCARRY", "INT_SBORROW"} and len(values) == 2:
            bits = max(values[0].bits, values[1].bits)
            kind = "sub" if opcode == "INT_SBORROW" else "add"
            result = _binary(kind, values[0], values[1], bits)
            if opcode == "INT_CARRY":
                wide_bits = bits + 1
                wide_result = _binary(
                    "add",
                    _zero_extend(values[0], wide_bits),
                    _zero_extend(values[1], wide_bits),
                    wide_bits,
                )
                return _extract(wide_result, bits, output_bits)
            flags = self._arithmetic_flags(kind, values[0], values[1], result, bits)
            return flags.overflow or _var("flag_of", output_bits)
        return None

    def _read_sleigh_unique(
        self,
        scratch: dict[tuple[str, int, int], _Expr],
        varnode: SleighVarnode,
        instruction: Instruction,
    ) -> _Expr:
        offset = varnode.offset
        size_bytes = varnode.size
        end = offset + size_bytes
        exact = scratch.get(_sleigh_varnode_key(varnode))
        if exact is not None:
            return _resize(exact, varnode.size_bits)
        for (space, stored_offset, stored_size), stored_value in scratch.items():
            if (
                space == "unique"
                and stored_offset <= offset
                and stored_offset + stored_size >= end
            ):
                return _extract(
                    stored_value,
                    (offset - stored_offset) * 8,
                    varnode.size_bits,
                )
        byte_values: list[_Expr] = []
        for byte_offset in range(offset, end):
            stored = next(
                (
                    (stored_offset, stored_size, value)
                    for (space, stored_offset, stored_size), value in scratch.items()
                    if space == "unique"
                    and stored_offset <= byte_offset < stored_offset + stored_size
                ),
                None,
            )
            if stored is None:
                byte_values.append(
                    _var(f"pcode_tmp_{instruction.address:X}_{byte_offset:X}", 8)
                )
            else:
                stored_offset, _stored_size, value = stored
                byte_values.append(
                    _extract(value, (byte_offset - stored_offset) * 8, 8)
                )
        value = byte_values[-1]
        for byte_value in reversed(byte_values[:-1]):
            value = _concat(value, byte_value, byte_value.bits)
        self._write_sleigh_varnode_scratch(scratch, varnode, value)
        return _resize(value, varnode.size_bits)

    @staticmethod
    def _write_sleigh_varnode_scratch(
        scratch: dict[tuple[str, int, int], _Expr],
        varnode: SleighVarnode,
        value: _Expr,
    ) -> None:
        key = _sleigh_varnode_key(varnode)
        value = _resize(value, varnode.size_bits)
        if varnode.space != "unique":
            scratch[key] = value
            return
        offset = varnode.offset
        end = offset + varnode.size
        remaining: dict[tuple[str, int, int], _Expr] = {}
        for (space, stored_offset, stored_size), stored_value in scratch.items():
            stored_end = stored_offset + stored_size
            if space != "unique" or stored_end <= offset or stored_offset >= end:
                remaining[(space, stored_offset, stored_size)] = stored_value
                continue
            if stored_offset < offset:
                left_size = offset - stored_offset
                remaining[(space, stored_offset, left_size)] = _extract(
                    stored_value, 0, left_size * 8
                )
            if stored_end > end:
                right_offset = end
                right_size = stored_end - end
                remaining[(space, right_offset, right_size)] = _extract(
                    stored_value,
                    (right_offset - stored_offset) * 8,
                    right_size * 8,
                )
        remaining[key] = value
        scratch.clear()
        scratch.update(remaining)

    def _read_sleigh_varnode(
        self,
        state: _State,
        varnode: SleighVarnode,
        scratch: dict[tuple[str, int, int], _Expr],
        instruction: Instruction,
        *,
        track_parameter_sources: bool = True,
    ) -> _Expr:
        size_bits = max(1, varnode.size_bits)
        if varnode.space == "const":
            return _const(varnode.offset, size_bits)
        key = _sleigh_varnode_key(varnode)
        if varnode.space in {"register", "ram"} and key in scratch:
            value = scratch[key]
            if varnode.space == "register":
                name = varnode.display.lower()
                alias = self._register_alias(name)
                vector_slice = self._sleigh_vector_register_slice(name)
                register = (
                    alias[0]
                    if alias is not None
                    else vector_slice[0]
                    if vector_slice is not None
                    else name
                )
                if track_parameter_sources:
                    self._record_pcode_register_read(
                        register, value, size_bits, instruction
                    )
            return _resize(value, size_bits)
        if varnode.space == "unique":
            return self._read_sleigh_unique(scratch, varnode, instruction)
        if varnode.space == "register":
            name = varnode.display.lower()
            if name in self._PCODE_FLAG_NAMES:
                return self._sleigh_flag_value(state.flags, name)
            if self._register_alias(name) is not None:
                return _resize(self._read_register(state, name, instruction), size_bits)
            vector_slice = self._sleigh_vector_register_slice(name)
            if vector_slice is not None:
                register, offset, slice_bits = vector_slice
                vector = state.xmm_registers.get(
                    register, self._initial_xmm_register(register)
                )
                value = _extract(vector, offset, min(slice_bits, size_bits))
                sources = self._record_parameter_sources(
                    value,
                    f"vector-live-in-read:{register}",
                    size_bits,
                    instruction=instruction,
                    use_kind="direct" if track_parameter_sources else "none",
                )
                if track_parameter_sources:
                    self._record_pcode_register_read(
                        register, value, size_bits, instruction
                    )
                expected = self._calling_convention.vector_parameter_name(register)
                if expected is not None and track_parameter_sources:
                    state.dirty_arguments.add(register)
                    if expected in sources:
                        self._note_vector_parameter_type(
                            register, instruction, expected
                        )
                return _resize(value, size_bits)
            if re.fullmatch(r"mm[0-7]", name):
                value = state.mmx_registers.get(name, _var(f"in_{name}", 64))
                self._record_pcode_register_read(name, value, size_bits, instruction)
                self._record_parameter_sources(
                    value,
                    f"register-read:{name}",
                    size_bits,
                    instruction=instruction,
                )
                return _resize(value, size_bits)
            if re.fullmatch(r"xmm\d+", name):
                value = state.xmm_registers.get(name, self._initial_xmm_register(name))
                sources = self._record_parameter_sources(
                    value,
                    f"vector-live-in-read:{name}",
                    size_bits,
                    instruction=instruction,
                    use_kind="direct" if track_parameter_sources else "none",
                )
                if track_parameter_sources:
                    self._record_pcode_register_read(
                        name, value, size_bits, instruction
                    )
                expected = self._calling_convention.vector_parameter_name(name)
                if (
                    track_parameter_sources
                    and expected is not None
                    and expected in sources
                ):
                    state.dirty_arguments.add(name)
                    self._note_vector_parameter_type(name, instruction, expected)
                return _resize(value, size_bits)
            if re.fullmatch(r"ymm\d+", name):
                value = state.pcode_registers.get(name)
                if value is None:
                    xmm_name = "xmm" + name[3:]
                    low = state.xmm_registers.get(
                        xmm_name, self._initial_xmm_register(xmm_name)
                    )
                    high_bits = max(1, size_bits - low.bits)
                    high = _var(f"in_{name}_hi", high_bits)
                    value = _concat(high, low, low.bits)
                self._record_pcode_register_read(name, value, size_bits, instruction)
                return _resize(value, size_bits)
            value = state.pcode_registers.get(
                name, _var(f"in_{_c_identifier(name)}", size_bits)
            )
            self._record_pcode_register_read(name, value, size_bits, instruction)
            return _resize(value, size_bits)
        if varnode.space == "ram":
            return self._sleigh_load(
                state, _const(varnode.offset, self.bits), size_bits, instruction
            )
        return _var(f"unknown_{varnode.space}_{varnode.offset:X}", size_bits)

    def _write_sleigh_register(
        self,
        state: _State,
        name: str,
        value: _Expr,
        size_bits: int,
        instruction: Instruction,
    ) -> None:
        alias = self._register_alias(name)
        if alias is not None:
            family = alias[0]
            old_stack_delta = state.stack_delta
            self._write_register(state, name, value)
            stack_family = "rsp" if self.bits == 64 else "esp"
            if family == stack_family:
                new_delta = self._sleigh_stack_offset(value)
                state.stack_delta = new_delta
                if new_delta is None or old_stack_delta != new_delta:
                    state.pending_pushes.clear()
            elif family == ("rbp" if self.bits == 64 else "ebp"):
                state.frame_delta = self._sleigh_stack_offset(value)
            return
        vector_slice = self._sleigh_vector_register_slice(name)
        if vector_slice is not None:
            register, offset, slice_bits = vector_slice
            old_value = state.xmm_registers.get(
                register, self._initial_xmm_register(register)
            )
            width = min(slice_bits, size_bits)
            updated = _insert_bits(old_value, value, offset, width, 128)
            state.xmm_registers[register] = updated
            state.xmm_widths[register] = max(
                state.xmm_widths.get(register, 0), offset + width
            )
            if register in self._tracked_argument_registers:
                state.dirty_arguments.add(register)
            self._record_pcode_register_definition(register, updated, 128, instruction)
            return
        if re.fullmatch(r"mm[0-7]", name):
            state.mmx_registers[name] = _resize(value, 64)
            self._record_pcode_register_definition(
                name, state.mmx_registers[name], 64, instruction
            )
            return
        if re.fullmatch(r"xmm\d+", name):
            old_value = state.xmm_registers.get(name, self._initial_xmm_register(name))
            if size_bits < 128:
                updated = _insert_bits(old_value, value, 0, size_bits, 128)
            else:
                updated = _resize(value, 128)
            state.xmm_registers[name] = updated
            state.xmm_widths[name] = min(size_bits, 128)
            if name in self._tracked_argument_registers:
                state.dirty_arguments.add(name)
            self._record_pcode_register_definition(name, updated, 128, instruction)
            return
        if re.fullmatch(r"ymm\d+", name) and size_bits >= 256:
            state.pcode_registers[name] = _resize(value, size_bits)
            xmm_name = "xmm" + name[3:]
            state.xmm_registers[xmm_name] = _extract(value, 0, 128)
            state.xmm_widths[xmm_name] = 128
            self._record_pcode_register_definition(
                name, state.pcode_registers[name], size_bits, instruction
            )
            return
        state.pcode_registers[name] = _resize(value, size_bits)
        self._record_pcode_register_definition(
            name, state.pcode_registers[name], size_bits, instruction
        )

    def _sleigh_stack_offset(self, expression: _Expr) -> int | None:
        if expression.kind == "stack_pointer":
            return int(expression.value)
        if expression.kind != "op" or len(expression.args) != 2:
            return None
        operation = str(expression.value)
        left, right = expression.args
        if operation == "add":
            if left.kind == "stack_pointer" and right.kind == "const":
                return int(left.value) + int(right.value)
            if right.kind == "stack_pointer" and left.kind == "const":
                return int(right.value) + int(left.value)
        if (
            operation == "sub"
            and left.kind == "stack_pointer"
            and right.kind == "const"
        ):
            return int(left.value) - int(right.value)
        return None

    def _sleigh_address(self, expression: _Expr) -> _Expr:
        if expression.kind == "const":
            return _Expr(
                "address",
                value=("absolute", int(expression.value) & _mask(self.bits)),
                bits=self.bits,
            )
        return expression

    def _sleigh_load(
        self,
        state: _State,
        address: _Expr,
        size_bits: int,
        instruction: Instruction,
    ) -> _Expr:
        offset = self._sleigh_stack_offset(address)
        if offset is not None:
            value = self._read_stack_value(state, offset, size_bits)
            location = f"stack[{offset:+#x}]"
            self._record_parameter_sources(
                value,
                f"stack-live-in-read:{offset:+#x}",
                size_bits,
                instruction=instruction,
                use_kind="direct",
            )
        else:
            address = self._sleigh_address(address)
            value = self._read_absolute_memory(state, address, size_bits)
            location = f"memory[{self._render(address)}]"
            self._record_parameter_sources(
                address,
                "sleigh-memory-address",
                self.bits,
                instruction=instruction,
                use_kind="indirect",
            )
        self._record_pcode_memory_load(value, address, location, (), instruction)
        return _resize(value, size_bits)

    def _sleigh_store(
        self,
        state: _State,
        address: _Expr,
        value: _Expr,
        size_bits: int,
        instruction: Instruction,
        emitted: list[str],
        *,
        collect: bool,
    ) -> None:
        value = _resize(value, size_bits)
        offset = self._sleigh_stack_offset(address)
        self._record_parameter_sources(
            value,
            "sleigh-memory-store-value",
            size_bits,
            instruction=instruction,
            use_kind="direct",
        )
        if offset is not None:
            self._write_stack_value(state, offset, size_bits, value)
            self._record_pcode_memory_store(
                value,
                address,
                f"stack[{offset:+#x}]",
                (),
                instruction,
                size_bits,
            )
            return
        address = self._sleigh_address(address)
        self._write_absolute_memory(state, address, size_bits, value)
        location = f"memory[{self._render(address)}]"
        self._record_pcode_memory_store(
            value, address, location, (), instruction, size_bits
        )
        if collect:
            emitted.append(
                f"*({_memory_type(size_bits)} *)({self._render(address)}) = "
                f"{self._render(value)};"
            )

    def _sleigh_flag_value(self, flags: _Flags | None, name: str) -> _Expr:
        name = name.lower()
        if flags is None:
            return _var(f"flag_{name}", 1)
        pcode_field = {
            "cf": flags.carry,
            "zf": flags.zero,
            "sf": flags.sign,
            "of": flags.overflow,
            "pf": flags.parity,
        }.get(name)
        if flags.kind == "pcode" and pcode_field is not None:
            return pcode_field
        if name == "cf":
            return self._carry_flag(flags)
        if name == "of":
            return flags.overflow or _var("flag_of", 1)
        if name == "pf":
            if flags.parity is not None:
                return flags.parity
            return (
                _parity(flags.result)
                if flags.result is not None
                else _var("flag_pf", 1)
            )
        if name == "zf":
            if (
                flags.kind == "cmp"
                and flags.left is not None
                and flags.right is not None
            ):
                return _binary("eq", flags.left, flags.right, 1)
            if flags.result is not None:
                return _binary("eq", flags.result, _const(0, flags.bits), 1)
        if name == "sf" and flags.result is not None:
            return _binary("slt", flags.result, _const(0, flags.bits), 1)
        return _var(f"flag_{name}", 1)

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
                self._record_pcode_register_read(
                    "xmm0", xmm_result, max(1, xmm_result.bits), instruction
                )
                self._record_parameter_sources(
                    xmm_result,
                    "returned-from-function",
                    xmm_result.bits,
                    instruction=instruction,
                    use_kind="return",
                )
                emit(f"return {self._render(xmm_result)};")
                if collect:
                    self._saw_value_return = True
                    self._return_type = xmm_type
            elif wide_result is not None:
                self._record_pcode_register_read(
                    return_family, result, max(1, result.bits), instruction
                )
                if high_register is not None:
                    self._record_pcode_register_read(
                        high_register, high_value, max(1, high_value.bits), instruction
                    )
                self._record_parameter_sources(
                    wide_result,
                    "returned-from-function",
                    wide_result.bits,
                    instruction=instruction,
                    use_kind="return",
                )
                emit(f"return {self._render(wide_result)};")
                if collect:
                    self._saw_value_return = True
                    self._return_type = wide_type
            elif result != initial_result and not call_result:
                self._record_pcode_register_read(
                    return_family, result, max(1, result.bits), instruction
                )
                self._record_parameter_sources(
                    result,
                    "returned-from-function",
                    result.bits,
                    instruction=instruction,
                    use_kind="return",
                )
                emit(f"return {self._render(result)};")
                if collect:
                    self._saw_value_return = True
                    self._return_type = f"uint{self.bits}_t"
            elif self.bits == 32 and state.x87_stack:
                self._record_pcode_register_read(
                    "st0", state.x87_stack[0], state.x87_stack[0].bits, instruction
                )
                self._record_parameter_sources(
                    state.x87_stack[0],
                    "returned-from-function",
                    state.x87_stack[0].bits,
                    instruction=instruction,
                    use_kind="return",
                )
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
            if any(_expression_contains_stack_pointer(arg) for arg in arguments) or any(
                _expression_contains_stack_pointer(value)
                for value in state.memory.values()
            ):
                state.stack.clear()
            target, target_value = self._call_target(instruction, operands, state)
            call = f"{target}({', '.join(self._render(arg) for arg in arguments)})"
            no_return = bool(self.analyzer._call_is_noreturn(instruction))
            result_name = f"call_result_{instruction.address:X}"
            result = None if no_return else _var(result_name, self.bits)
            if self._paramid_recording_enabled:
                self._paramid_graph.record_call(
                    arguments,
                    target=target_value,
                    result=result,
                    address=instruction.address,
                    mnemonic=mnemonic,
                    direct_target_address=instruction.branch_target,
                    call_target_name=target if target_value is None else None,
                    evidence=(
                        f"direct-target:{instruction.branch_target:#x}"
                        if instruction.branch_target is not None
                        else "indirect-or-import-target"
                    ),
                )
            self._current_pcode_read_nodes.clear()
            if no_return:
                emit(f"{call}; /* no retorna */")
            else:
                emit(f"{result_name} = {call};")
                if collect:
                    self._call_results.add(result_name)
                return_family = call_convention.integer_return_registers[0]
                self._write_register(state, return_family, result, mark_argument=False)
            caller_saved = call_convention.caller_saved_registers
            return_family = call_convention.integer_return_registers[0]
            for family in caller_saved:
                if family.startswith("xmm"):
                    clobbered = _var(
                        f"after_call_{instruction.address:X}_{family}", 128
                    )
                    state.xmm_registers[family] = clobbered
                    state.xmm_widths[family] = 128
                    self._record_pcode_register_definition(
                        family, clobbered, 128, instruction
                    )
                elif family != return_family:
                    self._write_register(
                        state,
                        family,
                        _var(f"after_call_{instruction.address:X}_{family}", self.bits),
                        mark_argument=False,
                    )
            if "ftol" in target.lower():
                self._x87_pop(state)
            else:
                state.x87_stack.clear()
                state.x87_flags = None
                state.x87_status_flags = None
            state.mmx_registers.clear()
            state.flags = None
            state.memory.clear()
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
                self._clear_stack_range(state, old_delta, state.stack_delta)
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

        # Use the local Ghidra SLEIGH semantics before conceding that an
        # instruction is unsupported by the higher-level lifters.
        if self._execute_sleigh_semantics(state, instruction, emitted, collect=collect):
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
                    self._write_stack_value(
                        state, offset, destination.size * 8, unknown
                    )
                else:
                    address = self._memory_address(destination, state, instruction)
                    self._write_absolute_memory(
                        state, address, destination.size * 8, unknown
                    )
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
        recording = collect and self._paramid_recording_enabled
        if recording:
            # Register versions are local to this straight-line block. Values
            # entering a block are represented by their symbolic expression or
            # its MULTIEQUAL node rather than an arbitrary predecessor version.
            self._current_pcode_register_versions = {}
        for instruction in block.instructions:
            if recording:
                mnemonic = self._decode(instruction).mnemonic
                self._current_instruction = instruction
                self._current_pcode_read_nodes = []
                self._paramid_graph.begin_instruction(instruction.address, mnemonic)
            else:
                self._current_instruction = None
            try:
                statements.extend(
                    self._execute_instruction(state, instruction, collect=collect)
                )
            finally:
                self._current_instruction = None
                self._current_pcode_read_nodes = []
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

        pcode_register_names = set().union(
            *(source.pcode_registers.keys() for source in sources)
        )
        for name in sorted(pcode_register_names):
            fallback_bits = next(
                (
                    value.bits
                    for source in sources
                    if (value := source.pcode_registers.get(name)) is not None
                ),
                self.bits,
            )
            values = tuple(
                source.pcode_registers.get(name, _var(f"in_{name}", fallback_bits))
                for source in sources
            )
            merged.pcode_registers[name] = self._merge_value(
                block_start, name, values, record_phis=record_phis
            )

        stack_boundaries = {
            boundary
            for source in sources
            for offset, width in source.stack
            for boundary in (offset, offset + (width + 7) // 8)
        }
        ordered_boundaries = sorted(stack_boundaries)
        for start, end in itertools.pairwise(ordered_boundaries):
            if end <= start:
                continue
            width = (end - start) * 8
            if not any(
                self._stack_has_written(source, start, width) for source in sources
            ):
                continue
            values = tuple(
                self._read_stack_value(source, start, width) for source in sources
            )
            location = f"stack_{start:+X}_{width}"
            value = self._merge_value(
                block_start, location, values, record_phis=record_phis
            )
            self._write_stack_value(merged, start, width, value)

        memory_boundaries = {
            boundary
            for source in sources
            for offset, width in source.memory
            for boundary in (offset, offset + (width + 7) // 8)
        }
        for start, end in itertools.pairwise(sorted(memory_boundaries)):
            if end <= start:
                continue
            width = (end - start) * 8
            if not any(
                self._memory_has_written(source, start, width) for source in sources
            ):
                continue
            address = _Expr("address", value=("absolute", start), bits=self.bits)
            values = tuple(
                self._read_absolute_memory(source, address, width) for source in sources
            )
            location = f"memory_{start:X}_{width}"
            merged.memory[(start, width)] = self._merge_value(
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
        size_bits = max((value.bits for value in values), default=self.bits)
        if record_phis:
            self._phi_inputs[(block_start, location)] = values
            if self._paramid_recording_enabled:
                self._paramid_graph.record_phi(
                    name, values, size_bits, address=block_start
                )
        return _var(name, size_bits)

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
        if flags.kind == "pcode":
            carry = self._sleigh_flag_value(flags, "cf")
            zero = self._sleigh_flag_value(flags, "zf")
            sign = self._sleigh_flag_value(flags, "sf")
            overflow = self._sleigh_flag_value(flags, "of")
            parity = self._sleigh_flag_value(flags, "pf")
            signed_less = _binary("xor", sign, overflow, 1)
            if condition == "eq":
                return zero
            if condition == "ne":
                return _unary("boolnot", zero, 1)
            if condition == "ult":
                return carry
            if condition == "ule":
                return _binary("lor", carry, zero, 1)
            if condition == "ugt":
                return _binary(
                    "land",
                    _unary("boolnot", carry, 1),
                    _unary("boolnot", zero, 1),
                    1,
                )
            if condition == "uge":
                return _unary("boolnot", carry, 1)
            if condition == "slt":
                return signed_less
            if condition == "sle":
                return _binary("lor", zero, signed_less, 1)
            if condition == "sgt":
                return _binary(
                    "land",
                    _unary("boolnot", zero, 1),
                    _unary("boolnot", signed_less, 1),
                    1,
                )
            if condition == "sge":
                return _unary("boolnot", signed_less, 1)
            if condition == "of":
                return overflow
            if condition == "nof":
                return _unary("boolnot", overflow, 1)
            if condition == "parity":
                return parity
            if condition == "nparity":
                return _unary("boolnot", parity, 1)
            if mnemonic == "js":
                return sign
            if mnemonic == "jns":
                return _unary("boolnot", sign, 1)

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

    def _record_call_argument_uses(
        self, arguments: list[_Expr], instruction: Instruction
    ) -> None:
        for index, argument in enumerate(arguments, start=1):
            self._record_parameter_sources(
                argument,
                f"forwarded-to-call-argument:{index}",
                max(1, argument.bits),
                instruction=instruction,
                use_kind="call",
            )

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
        if (
            target_function is not None
            and target_function.signature is not None
            and target_function.signature.locked
        ):
            return self._call_arguments_for_locked_signature(
                state, instruction, target_function, convention
            )
        args: list[_Expr] = []

        def is_available(register: str) -> bool:
            return register in state.dirty_arguments

        def read_argument(register: str) -> _Expr:
            if register.startswith("xmm"):
                value = state.xmm_registers.get(
                    register, self._initial_xmm_register(register)
                )
                sources = self._record_parameter_sources(
                    value,
                    f"vector-forwarded-to-call:{register}",
                    128,
                    instruction=instruction,
                    use_kind="none",
                )
                self._record_pcode_register_read(register, value, 128, instruction)
                expected_parameter = self._calling_convention.vector_parameter_name(
                    register
                )
                if expected_parameter in sources and not instruction.is_call:
                    self._note_vector_parameter_type(
                        register, instruction, expected_parameter
                    )
                elif expected_parameter in sources and expected_parameter is not None:
                    self._parameter_evidence[expected_parameter].add(
                        "forwarded-as-call-argument"
                    )
                return _resize(value, 128)
            return self._read_register(state, register, instruction, use_kind="none")

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
        stack_argument_offsets: list[int | None] = []
        if state.stack_delta is not None:
            outgoing_offset = (
                state.stack_delta + convention.stack_argument_offset - self.pointer_size
            )
            if stack_count:
                for index in range(stack_count):
                    offset = outgoing_offset + index * self.pointer_size
                    value = (
                        self._read_stack_value(state, offset, self.bits)
                        if self._stack_has_written(state, offset, self.bits)
                        else None
                    )
                    if value is None:
                        if index < len(state.pending_pushes):
                            value = list(reversed(state.pending_pushes))[index][1]
                        else:
                            break
                    stack_values.append(value)
                    stack_argument_offsets.append(offset)
            elif self.bits == 64:
                # Stack-passed x64 arguments are written into the outgoing
                # argument area; Windows reserves 32 home bytes first.
                for index in range(16):
                    offset = outgoing_offset + index * self.pointer_size
                    if not self._stack_has_written(state, offset, self.bits):
                        break
                    value = self._read_stack_value(state, offset, self.bits)
                    stack_values.append(value)
                    stack_argument_offsets.append(offset)
        if not stack_values and state.pending_pushes:
            pending = list(reversed(state.pending_pushes))
            stack_values = [value for _offset, value in pending]
            stack_argument_offsets = [offset for offset, _value in pending]
        if self._paramid_recording_enabled:
            stack_register = "rsp" if self.bits == 64 else "esp"
            for value, offset in zip(stack_values, stack_argument_offsets, strict=True):
                if offset is None:
                    continue
                address_expression = _Expr(
                    "address",
                    value=("stack", self._stack_name(offset)),
                    bits=self.bits,
                )
                self._record_pcode_memory_load(
                    value,
                    address_expression,
                    f"stack[{offset:+#x}]",
                    (stack_register,),
                    instruction,
                )
        args.extend(stack_values)
        self._record_call_argument_uses(args, instruction)
        return args

    def _call_arguments_for_locked_signature(
        self,
        state: _State,
        instruction: Instruction,
        target_function: Function,
        convention: CallingConvention,
    ) -> list[_Expr]:
        signature = target_function.signature
        if signature is None:
            return []
        args: list[_Expr] = []
        consumed_registers: set[str] = set()
        first_unused_stack_offset = convention.stack_argument_offset
        if signature.hidden_return_location is not None:
            hidden = signature.hidden_return_location
            if hidden.register:
                consumed_registers.add(hidden.register)
            if hidden.stack_offset is not None:
                first_unused_stack_offset = max(
                    first_unused_stack_offset,
                    hidden.stack_offset
                    + _align_abi_size(hidden.size, self.pointer_size),
                )

        for parameter_index, parameter in enumerate(signature.parameters):
            dtype = parameter.data_type or self._type_database.resolve(
                parameter.type_name
            )
            base = dtype.unwrapped()
            locations = parameter.locations
            pieces: list[_Expr] = []
            for location in locations:
                if location.register is not None:
                    consumed_registers.add(location.register)
                    if location.register.startswith("xmm"):
                        value = state.xmm_registers.get(
                            location.register,
                            self._initial_xmm_register(location.register),
                        )
                        self._record_pcode_register_read(
                            location.register, value, 128, instruction
                        )
                        shared_register_offsets = [
                            piece.value_offset
                            for piece in locations
                            if piece.register == location.register
                        ]
                        inner_offset = (
                            location.value_offset - min(shared_register_offsets)
                            if len(shared_register_offsets) > 1
                            else 0
                        )
                        width = min(location.size * 8, value.bits or 128)
                        piece = _extract(value, inner_offset * 8, width)
                    else:
                        value = self._read_register(
                            state,
                            location.register,
                            instruction,
                            use_kind="none",
                        )
                        piece = (
                            _truncate(value, location.size * 8)
                            if location.size * 8 < value.bits
                            else value
                        )
                else:
                    if location.stack_offset is None:
                        continue
                    first_unused_stack_offset = max(
                        first_unused_stack_offset,
                        location.stack_offset
                        + _align_abi_size(location.size, self.pointer_size),
                    )
                    if base.is_aggregate and not location.indirect:
                        piece = _Expr(
                            "address",
                            ("stack", f"stack_arg_{parameter_index + 1}"),
                            bits=self.bits,
                        )
                    else:
                        caller_offset = (
                            None
                            if state.stack_delta is None
                            else state.stack_delta
                            + location.stack_offset
                            - self.pointer_size
                        )
                        piece = self._call_stack_argument_value(
                            state,
                            caller_offset,
                            location,
                            convention,
                            parameter_index,
                        )
                        if caller_offset is not None:
                            address_expression = _Expr(
                                "address",
                                value=("stack", self._stack_name(caller_offset)),
                                bits=self.bits,
                            )
                            self._record_pcode_memory_load(
                                piece,
                                address_expression,
                                f"stack[{caller_offset:+#x}]",
                                ("rsp" if self.bits == 64 else "esp",),
                                instruction,
                            )
                pieces.append(piece)

            if not pieces:
                pieces.append(_var(f"stack_arg_{parameter_index + 1}", self.bits))
            if any(location.indirect for location in locations):
                args.append(
                    _Expr(
                        "typed_load",
                        dtype,
                        args=(pieces[0],),
                        bits=(dtype.size or self.pointer_size) * 8,
                    )
                )
            elif base.is_aggregate:
                if any(location.kind == "stack" for location in locations):
                    args.append(
                        _Expr(
                            "typed_load",
                            dtype,
                            args=(pieces[0],),
                            bits=(dtype.size or self.pointer_size) * 8,
                        )
                    )
                else:
                    args.append(
                        _Expr(
                            "aggregate_value",
                            (base, locations),
                            args=tuple(pieces),
                            bits=(base.size or 0) * 8,
                        )
                    )
            elif base.kind == TypeKind.FLOAT:
                value = pieces[0]
                width = (base.size or 8) * 8
                raw = _extract(value, 0, width) if value.bits > width else value
                args.append(_simd_float(raw, width))
            elif len(pieces) > 1:
                ordered = sorted(
                    zip(locations, pieces, strict=True),
                    key=lambda item: item[0].value_offset,
                )
                combined = ordered[0][1]
                for _location, piece in ordered[1:]:
                    combined = _concat(piece, combined, combined.bits)
                args.append(combined)
            else:
                value = pieces[0]
                if base.size is not None and base.size * 8 < value.bits:
                    value = _truncate(value, base.size * 8)
                args.append(value)

        if signature.variadic:
            args.extend(
                self._remaining_variadic_call_arguments(
                    state,
                    instruction,
                    convention,
                    consumed_registers,
                    first_unused_stack_offset,
                )
            )
        self._record_call_argument_uses(args, instruction)
        return args

    def _call_stack_argument_value(
        self,
        state: _State,
        caller_offset: int | None,
        location: StorageLocation,
        convention: CallingConvention,
        parameter_index: int,
    ) -> _Expr:
        if caller_offset is not None:
            width = location.size * 8
            if self._stack_has_written(state, caller_offset, width):
                return self._read_stack_value(state, caller_offset, width)
            if location.stack_offset is not None and state.pending_pushes:
                slot = max(
                    0,
                    (location.stack_offset - convention.stack_argument_offset)
                    // self.pointer_size,
                )
                pending = list(reversed(state.pending_pushes))
                if slot < len(pending):
                    return pending[slot][1]
        return _var(
            f"stack_arg_{parameter_index + 1}", max(self.bits, location.size * 8)
        )

    def _remaining_variadic_call_arguments(
        self,
        state: _State,
        instruction: Instruction,
        convention: CallingConvention,
        consumed_registers: set[str],
        first_unused_stack_offset: int,
    ) -> list[_Expr]:
        arguments: list[_Expr] = []

        def is_available(register: str) -> bool:
            return (
                register not in consumed_registers and register in state.dirty_arguments
            )

        def read(register: str) -> _Expr:
            if register.startswith("xmm"):
                value = state.xmm_registers.get(
                    register, self._initial_xmm_register(register)
                )
                self._record_pcode_register_read(register, value, 128, instruction)
                return _resize(value, 128)
            return self._read_register(state, register, instruction, use_kind="none")

        if convention.register_argument_groups:
            for group in convention.register_argument_groups:
                available = [register for register in group if is_available(register)]
                if not available:
                    continue
                # A duplicated Microsoft x64 variadic FP argument is represented
                # by its XMM value once, not again through the shadow GPR copy.
                selected = next(
                    (register for register in available if register.startswith("xmm")),
                    available[0],
                )
                arguments.append(read(selected))
        else:
            register_sets = (
                convention.argument_registers,
                convention.vector_argument_registers,
            )
            for registers in register_sets:
                active = [
                    index
                    for index, register in enumerate(registers)
                    if is_available(register)
                ]
                if active:
                    for register in registers[: max(active) + 1]:
                        if is_available(register):
                            arguments.append(read(register))

        if state.stack_delta is not None:
            outgoing_offset = (
                state.stack_delta + first_unused_stack_offset - self.pointer_size
            )
            pending = list(reversed(state.pending_pushes))
            pending_slot = max(
                0,
                (first_unused_stack_offset - convention.stack_argument_offset)
                // self.pointer_size,
            )
            for index in range(16):
                offset = outgoing_offset + index * self.pointer_size
                value = (
                    self._read_stack_value(state, offset, self.bits)
                    if self._stack_has_written(state, offset, self.bits)
                    else None
                )
                if value is None:
                    pending_index = pending_slot + index
                    if pending_index < len(pending):
                        value = pending[pending_index][1]
                    else:
                        break
                arguments.append(value)
        return arguments

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
    ) -> tuple[str, _Expr | None]:
        target = instruction.branch_target
        if target is not None:
            name = self.analyzer.name_at(target)
            if name:
                return _c_identifier(name), None
            return f"loc_{target:X}", None
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
                return _c_identifier(import_name), None
        if operands:
            target_expr = self._read_operand(state, operands[0], instruction)
            self._record_parameter_sources(
                target_expr,
                "indirect-call-target",
                max(1, target_expr.bits),
                instruction=instruction,
                use_kind="indirect",
            )
            return f"(*({self._render(target_expr)}))", target_expr
        return "indirect_call", None

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
            old_delta = state.stack_delta
            state.stack_delta -= self.pointer_size
            self._write_stack_value(state, state.stack_delta, self.bits, value)
            if (
                self._paramid_recording_enabled
                and self._current_instruction is not None
            ):
                instruction = self._current_instruction
                stack_register = "rsp" if self.bits == 64 else "esp"
                old_sp = _Expr("stack_pointer", old_delta, bits=self.bits)
                new_sp = _Expr("stack_pointer", state.stack_delta, bits=self.bits)
                sp_node = self._record_pcode_register_read(
                    stack_register, old_sp, self.bits, instruction
                )
                offset = state.stack_delta
                address = _Expr(
                    "address", value=("stack", self._stack_name(offset)), bits=self.bits
                )
                self._record_pcode_memory_store(
                    value,
                    address,
                    f"stack[{offset:+#x}]",
                    (stack_register,),
                    instruction,
                    self.bits,
                )
                self._record_pcode_register_definition(
                    stack_register,
                    new_sp,
                    self.bits,
                    instruction,
                    input_nodes=(sp_node,) if sp_node is not None else (),
                )

    def _pop(self, state: _State) -> _Expr:
        offset = state.stack_delta
        if offset is None:
            result = _var("unknown_stack_value", self.bits)
        else:
            result = self._read_stack_value(state, offset, self.bits)
            if (
                self._paramid_recording_enabled
                and self._current_instruction is not None
            ):
                instruction = self._current_instruction
                stack_register = "rsp" if self.bits == 64 else "esp"
                old_sp = _Expr("stack_pointer", offset, bits=self.bits)
                sp_node = self._record_pcode_register_read(
                    stack_register, old_sp, self.bits, instruction
                )
                address = _Expr(
                    "address", value=("stack", self._stack_name(offset)), bits=self.bits
                )
                self._record_pcode_memory_load(
                    result,
                    address,
                    f"stack[{offset:+#x}]",
                    (stack_register,),
                    instruction,
                )
            state.stack_delta += self.pointer_size
            if (
                self._paramid_recording_enabled
                and self._current_instruction is not None
            ):
                new_sp = _Expr("stack_pointer", state.stack_delta, bits=self.bits)
                self._record_pcode_register_definition(
                    "rsp" if self.bits == 64 else "esp",
                    new_sp,
                    self.bits,
                    self._current_instruction,
                    input_nodes=(sp_node,) if sp_node is not None else (),
                )
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
            source = expression.args[0]
            return (
                f"((long double)(({_signed_type(source.bits)})"
                f"({self._render(source)})))"
            )
        if kind == "fp_to_bits":
            width = int(expression.value)
            float_type = _float_type(width)
            integer_type = _unsigned_type(width)
            value = self._render(expression.args[0])
            return (
                f"((union {{ {integer_type} i; {float_type} f; }})"
                f"{{ .f = ({float_type})({value}) }}).i"
            )
        if kind == "fp_nan":
            return f"isnan({self._render(expression.args[0])})"
        if kind == "fp_compare":
            left, right = (self._render(arg) for arg in expression.args)
            operator = {"eq": "==", "ne": "!=", "lt": "<", "le": "<="}.get(
                str(expression.value), "=="
            )
            return f"({left} {operator} {right})"
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
                "ceil": "ceill",
                "floor": "floorl",
                "trunc": "truncl",
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
        if kind in {"typed_load", "aggregate_load"}:
            ctype = expression.value
            pointer = self._render(expression.args[0])
            return f"*(({ctype.name} *)({pointer}))"
        if kind == "aggregate_value":
            return self._render_aggregate_value(expression)
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
        if kind == "pcode_builtin":
            value = self._render(expression.args[0])
            if expression.value == "popcount":
                return f"__builtin_popcountll((unsigned long long)({value}))"
            if expression.value == "bsf":
                return (
                    f"(({value}) == 0 ? 0 : "
                    f"__builtin_ctzll((unsigned long long)({value})))"
                )
            if expression.value == "bsr":
                source_bits = expression.args[0].bits
                return (
                    f"(({value}) == 0 ? {source_bits - 1} : "
                    f"(63 - __builtin_clzll((unsigned long long)({value}))))"
                )
            if expression.value == "lzcount":
                source_bits = expression.args[0].bits
                adjustment = 64 - source_bits
                return (
                    f"(({value}) == 0 ? {source_bits} : "
                    f"(__builtin_clzll((unsigned long long)({value})) - {adjustment}))"
                )
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

    def _render_aggregate_value(self, expression: _Expr) -> str:
        ctype, locations = expression.value
        fields = ctype.fields[:1] if ctype.kind == TypeKind.UNION else ctype.fields
        initializers: list[str] = []
        for struct_field in fields:
            field_type = struct_field.type.unwrapped()
            field_size = field_type.size
            if field_size is None or field_size <= 0:
                continue
            pieces: list[tuple[int, _Expr]] = []
            field_start = struct_field.offset
            field_end = field_start + field_size
            for location, value in zip(locations, expression.args, strict=True):
                location_start = location.value_offset
                location_end = location_start + location.size
                overlap_start = max(field_start, location_start)
                overlap_end = min(field_end, location_end)
                if overlap_start >= overlap_end:
                    continue
                width = (overlap_end - overlap_start) * 8
                bit_offset = (overlap_start - location_start) * 8
                part = (
                    _extract(value, bit_offset, width)
                    if bit_offset or width < value.bits
                    else value
                )
                pieces.append((overlap_start, part))
            pieces.sort(key=lambda item: item[0])
            if not pieces or pieces[0][0] != field_start:
                continue
            raw = pieces[0][1]
            covered_until = field_start + raw.bits // 8
            complete = True
            for offset, piece in pieces[1:]:
                if offset != covered_until:
                    complete = False
                    break
                raw = _concat(piece, raw, raw.bits)
                covered_until += piece.bits // 8
            if not complete or covered_until < field_end:
                continue
            if field_type.kind == TypeKind.FLOAT and field_type.size in {4, 8}:
                rendered = self._render(_simd_float(raw, field_type.size * 8))
            elif field_type.is_aggregate:
                nested_pieces = [
                    (location, value)
                    for location, value in zip(locations, expression.args, strict=True)
                    if location.value_offset < field_end
                    and location.value_offset + location.size > field_start
                ]
                nested_locations = tuple(
                    replace(
                        location,
                        value_offset=location.value_offset - struct_field.offset,
                    )
                    for location, _value in nested_pieces
                )
                nested = _Expr(
                    "aggregate_value",
                    (field_type, nested_locations),
                    args=tuple(value for _location, value in nested_pieces),
                    bits=(field_type.size or 0) * 8,
                )
                rendered = self._render(nested)
            else:
                if field_size * 8 < raw.bits:
                    raw = _truncate(raw, field_size * 8)
                rendered = f"(({struct_field.type.name})({self._render(raw)}))"
            field_name = _c_identifier(
                struct_field.name or f"field_{len(initializers)}"
            )
            initializers.append(f".{field_name} = {rendered}")
        if not initializers:
            return f"(({ctype.name}){{ 0 }})"
        return f"(({ctype.name}){{ {', '.join(initializers)} }})"

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

    def _ssa_paramid_input_locations(
        self,
    ) -> tuple[dict[SSALocation, str], dict[str, int]]:
        """Name the machine-state bytes that hold incoming ABI parameters."""
        locations: dict[SSALocation, str] = {}
        widths: dict[str, int] = {}

        def bind(register: str, name: str) -> None:
            varnode = self._sleigh_translator.registers.get(register.lower())
            if varnode is None or not varnode.is_register:
                return
            widths[name] = max(widths.get(name, 0), varnode.size_bits)
            for byte_offset in range(varnode.size):
                locations.setdefault(
                    SSALocation("register", varnode.offset + byte_offset), name
                )

        if self._declared_signature is not None:
            for parameter in self._declared_signature.parameters:
                for location in parameter.locations:
                    if location.register is not None:
                        bind(location.register, parameter.name)
                data_type = parameter.data_type
                if data_type is not None:
                    widths[parameter.name] = max(
                        widths.get(parameter.name, 0), data_type.size * 8
                    )
            hidden_location = self._declared_signature.hidden_return_location
            if hidden_location is not None:
                if hidden_location.register is not None:
                    bind(hidden_location.register, "__hidden_return")
                widths["__hidden_return"] = self.bits
        else:
            for name, register in abi_argument_registers(self._calling_convention):
                bind(register, name)
        return locations, widths

    def _ssa_paramid_return_registers(self) -> tuple[str, ...]:
        """Limit return sinks to the class/width selected by the inferred C type."""
        if not self._saw_value_return:
            return ()
        type_name = (
            self._declared_signature.return_type
            if self._declared_signature is not None
            else self._return_type
            or ""
        ).strip()
        try:
            return_type = self._type_database.resolve(type_name)
        except KeyError:
            return_type = None
        if return_type is not None and return_type.kind in {
            TypeKind.FLOAT,
            TypeKind.VECTOR,
        }:
            registers = self._calling_convention.floating_return_registers[:1]
        else:
            registers = self._calling_convention.integer_return_registers
            if return_type is None or return_type.size <= self.pointer_size:
                registers = registers[:1]
            else:
                registers = registers[:2]
        families: list[str] = []
        for register in registers:
            alias = self._register_alias(register)
            name = alias[0] if alias is not None else register.lower()
            if name.startswith(("xmm", "ymm", "zmm")):
                match = re.match(r"(?:xmm|ymm|zmm)(\d+)", name)
                if match is not None:
                    name = f"xmm{match.group(1)}"
            families.append(name)
        return tuple(dict.fromkeys(families))

    def _run_paramid_analysis(
        self,
        *,
        justproto: bool = False,
        input_names: Iterable[str] | None = None,
    ) -> None:
        """Build the SSA-first ParamID view for inferred or declared formals."""
        paramid_graph = self._paramid_graph
        if self._ssa_function is not None and self._ssa_function.valid:
            parameter_locations, parameter_widths = (
                self._ssa_paramid_input_locations()
            )
            paramid_graph = PcodeLikeGraph.from_ssa(
                self._ssa_function,
                parameter_locations,
                parameter_widths=parameter_widths,
                return_registers=self._ssa_paramid_return_registers(),
                supplemental_graph=self._paramid_graph,
                stack_parameter_names=(
                    self._input_parameter_names_by_stack
                    if self._declared_signature is not None
                    else None
                ),
            )
        self._paramid_analysis = ParamIDAnalysis(paramid_graph)
        self._paramid_active = self._paramid_analysis.analyze(
            justproto=justproto,
            input_names=input_names,
        )

    def _parameter_specs(self, body: list[str]) -> list[FunctionParameter]:
        del body  # Candidate inputs come from the lifted value graph, not C text.
        self._run_paramid_analysis()
        active_names = {
            trial.name for trial in self._paramid_active.active_trials
        }
        for trial in self._paramid_active.active_trials:
            self._parameter_evidence[trial.name].update(trial.evidence)
        if self._paramid_analysis.uses_ssa_pcode:
            # Do not let the legacy C-like propagation resurrect an input that
            # the SSA ParamMeasure pass proved unreferenced.
            self._used_parameter_names = active_names
        else:
            self._used_parameter_names.update(active_names)

        used = {
            int(match.group(1))
            for name in self._used_parameter_names
            if (match := re.fullmatch(r"param_(\d+)", name))
        }
        float_used = {
            int(match.group(1))
            for name in self._used_parameter_names
            if (match := re.fullmatch(r"float_param_(\d+)", name))
        }
        parameters: list[FunctionParameter] = []
        for indices, prefix, default_type in (
            (used, "param", _unsigned_type(self.bits)),
            (float_used, "float_param", "double"),
        ):
            if not indices:
                continue
            for index in range(1, max(indices) + 1):
                name = f"{prefix}_{index}"
                evidence = set(self._parameter_evidence.get(name, ()))
                trial = self._paramid_active.get_trial(name)
                rank = trial.rank if trial is not None else None
                if evidence:
                    if rank is None or rank == ParamRank.DIRECT_READ:
                        evidence.add("live-in-value-reaches-an-instruction-read")
                        confidence = 0.76
                    elif rank == ParamRank.SUBFUNCTION_PARAMETER:
                        evidence.add("live-in-value-forwarded-to-subfunction")
                        confidence = 0.68
                    elif rank == ParamRank.FUNCTION_RETURN:
                        evidence.add("live-in-value-reaches-function-return")
                        confidence = 0.68
                    elif rank == ParamRank.INDIRECT:
                        evidence.add("live-in-value-used-indirectly")
                        confidence = 0.6
                    else:
                        evidence.add("live-in-value-use-not-ranked")
                        confidence = 0.55
                    widths = sorted(self._parameter_read_widths.get(name, ()))
                    if widths:
                        evidence.add(
                            "observed-read-widths-bits:"
                            + ",".join(str(width) for width in widths)
                        )
                else:
                    evidence.add("positional-gap-retained-for-ABI-order")
                    confidence = 0.2
                type_name = self._parameter_types.get(name)
                if type_name is None:
                    type_name = default_type
                    evidence.add(f"type-fallback:{default_type}")
                parameters.append(
                    FunctionParameter(
                        name,
                        type_name,
                        confidence=confidence,
                        evidence=tuple(sorted(evidence)),
                    )
                )
        return parameters

    def _annotate_paramid_signature(
        self, signature: FunctionSignature
    ) -> FunctionSignature:
        active = self._paramid_active
        parameters: list[FunctionParameter] = []
        for parameter in signature.parameters:
            trial = active.get_trial(parameter.name)
            evidence = set(parameter.evidence)
            if trial is not None:
                evidence.update(trial.evidence)
                evidence.add("paramid-active" if trial.active else "paramid-inactive")
                if trial.selected:
                    evidence.add("paramid-selected-candidate")
                if trial.used:
                    evidence.add("paramid-used-final")
            parameters.append(replace(parameter, evidence=tuple(sorted(evidence))))
        return replace(signature, parameters=tuple(parameters))

    def _annotate_inferred_signature(
        self, function: Function, signature: FunctionSignature
    ) -> FunctionSignature:
        call_sites = [
            xref
            for xref in self.analyzer.xrefs_to(function.address)
            if xref.kind == "call"
        ]
        candidate_registers = {
            location.register
            for parameter in signature.parameters
            for location in parameter.locations
            if location.register is not None
        }
        candidate_stack_offsets = {
            location.stack_offset - self.pointer_size
            for parameter in signature.parameters
            for location in parameter.locations
            if location.stack_offset is not None
        }
        writes_by_stack_offset = (
            callsite_stack_writes(self.analyzer, call_sites, candidate_stack_offsets)
            if call_sites and candidate_stack_offsets
            else {}
        )
        writes_by_register = (
            callsite_register_writes(self.analyzer, call_sites, candidate_registers)
            if call_sites and candidate_registers
            else {}
        )
        parameters: list[FunctionParameter] = []
        for parameter in signature.parameters:
            evidence = set(parameter.evidence)
            confidence = parameter.confidence or 0.2
            is_live_in = parameter.name in self._parameter_evidence
            registers = {
                location.register
                for location in parameter.locations
                if location.register is not None
            }
            stack_offsets = {
                location.stack_offset - self.pointer_size
                for location in parameter.locations
                if location.stack_offset is not None
            }
            register_sites = (
                set().union(
                    *(writes_by_register.get(register, set()) for register in registers)
                )
                if registers
                else set()
            )
            stack_sites = (
                set().union(
                    *(
                        writes_by_stack_offset.get(offset, set())
                        for offset in stack_offsets
                    )
                )
                if stack_offsets
                else set()
            )
            supplied_sites = register_sites | stack_sites
            if call_sites and supplied_sites:
                ratio = len(supplied_sites) / len(call_sites)
                if register_sites:
                    evidence.add(
                        f"callsite-register-write:{len(register_sites)}/{len(call_sites)}"
                    )
                if stack_sites:
                    evidence.add(
                        f"callsite-stack-write:{len(stack_sites)}/{len(call_sites)}"
                    )
                if is_live_in:
                    confidence = min(
                        0.97,
                        confidence + (0.15 if ratio >= 0.5 else 0.04),
                    )
                else:
                    evidence.add("callsite-supplies-unread-positional-slot")
                    confidence = max(
                        confidence,
                        0.45 if ratio >= 0.5 else 0.3,
                    )
            elif is_live_in and call_sites and registers:
                evidence.add("callsite-register-write-not-observed")
                confidence = max(0.55, confidence - 0.11)
            elif is_live_in and call_sites and stack_offsets:
                evidence.add("callsite-stack-write-not-observed")
            elif is_live_in and not call_sites:
                evidence.add("no-callsite-corroboration")
            parameters.append(
                replace(
                    parameter,
                    confidence=round(confidence, 2),
                    evidence=tuple(sorted(evidence)),
                )
            )

        inference_evidence = {"parameters-from-live-in-dataflow"}
        if self._paramid_analysis is not None:
            inference_evidence.add("pcode-like-paramid-analysis")
            if self._paramid_analysis.uses_ssa_pcode:
                inference_evidence.add("ssa-pcode-paramid-analysis")
                inference_evidence.add(
                    "ssa-pcode-paramid-operations:"
                    f"{self._paramid_analysis.graph.ssa_operation_count}"
                )
                if self._paramid_analysis.valid is False:
                    inference_evidence.add("ssa-pcode-paramid-invalid-ssa")
                if self._paramid_analysis.complete is False:
                    inference_evidence.add("ssa-pcode-paramid-incomplete-cfg")
            inference_evidence.add(
                "pcode-like-paramid-active-trials:"
                f"{len(self._paramid_active.active_trials)}"
            )
            inference_evidence.add(
                "pcode-like-paramid-selected-trials:"
                f"{self._paramid_active.get_num_selected()}"
            )
            inference_evidence.add(
                "pcode-like-paramid-used-trials:"
                f"{sum(trial.used for trial in self._paramid_active.trials)}"
            )
            inference_evidence.add(
                f"pcode-like-ir-ops:{self._paramid_graph.operation_count}"
            )
            if self._paramid_graph.depth_limited_uses:
                inference_evidence.add(
                    "pcode-like-expression-depth-limits:"
                    f"{self._paramid_graph.depth_limited_uses}"
                )
        if any(
            "positional-gap-retained-for-ABI-order" in parameter.evidence
            for parameter in parameters
        ):
            inference_evidence.add("unobserved-positional-gaps-filled-conservatively")
        if call_sites:
            inference_evidence.add(f"callsite-registers-checked:{len(call_sites)}")
            if candidate_stack_offsets:
                inference_evidence.add(
                    f"callsite-stack-slots-checked:{len(call_sites)}"
                )
        else:
            inference_evidence.add("no-resolved-callsite-corroboration")

        confidences = [parameter.confidence or 0.0 for parameter in parameters]
        inference_confidence = min(confidences) if confidences else 0.35
        if not parameters:
            inference_evidence.add(
                "no-live-in-parameter-read-observed; unused-formals-cannot-be-ruled-out"
            )

        has_sysv_gp = any(
            parameter.name.startswith("param_") for parameter in parameters
        )
        has_sysv_sse = any(
            parameter.name.startswith("float_param_") for parameter in parameters
        )
        if (
            self.bits == 64
            and self._calling_convention.name == "sysv_amd64"
            and has_sysv_gp
            and has_sysv_sse
        ):
            inference_evidence.add(
                "sysv-gp-sse-banks-do-not-reveal-source-parameter-order"
            )
            inference_confidence = min(inference_confidence, 0.45)
            parameters = [
                replace(
                    parameter,
                    confidence=min(parameter.confidence or 0.0, 0.55),
                    evidence=tuple(
                        sorted(
                            set(parameter.evidence) | {"sysv-gp-sse-order-ambiguous"}
                        )
                    ),
                )
                for parameter in parameters
            ]

        return replace(
            signature,
            parameters=tuple(parameters),
            inference_confidence=round(inference_confidence, 2),
            inference_evidence=tuple(sorted(inference_evidence)),
        )

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
            r"\b(__hidden_return)\b",
            r"\b(in_[A-Za-z0-9_]+)\b",
            r"\b(after_call_[0-9A-F]+_[A-Za-z0-9_]+)\b",
            r"\b(condition_[A-Za-z0-9_]+)\b",
            r"\b(unknown_[A-Za-z0-9_]+)\b",
            r"\b(flag_[A-Za-z0-9_]+)\b",
            r"\b(x87_status_[A-Za-z0-9_]+)\b",
            r"\b((?:cpuid|rdtsc|rdtscp)_[A-Za-z0-9_]+)\b",
            r"\b((?:rep_(?!i_)[A-Za-z0-9_]+|after_string_[A-F0-9]+_[A-Za-z0-9_]+))\b",
            r"\b(pcode_load_\d+_[A-F0-9]+_\d+)\b",
        )
        for line in body:
            for pattern in patterns:
                names.update(re.findall(pattern, line))
        names = {name for name in names if not re.fullmatch(r"param_\d+", name)}
        declarations = []
        for name in sorted(names):
            if name == "__hidden_return" and self._declared_signature is not None:
                type_name = f"{self._declared_signature.return_type} *"
            elif name.startswith(("unknown_x87", "phi_x87")):
                type_name = "long double"
            elif name.startswith(("cpuid_", "rdtsc_", "rdtscp_")):
                type_name = "uint32_t"
            elif name.startswith(("in_mm", "phi_mm")):
                type_name = "uint64_t"
            elif name.startswith(("in_xmm", "phi_xmm")):
                type_name = "__uint128_t"
            elif name.startswith("x87_status_"):
                type_name = "uint16_t"
            elif name.startswith("pcode_load_"):
                type_name = _unsigned_type(int(name.split("_")[2]))
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


def _align_abi_size(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) & -alignment


def _var(name: str, bits: int) -> _Expr:
    return _Expr("var", name, bits=bits)


def _expression_variables(expression: _Expr):
    if expression.kind == "var":
        yield str(expression.value)
    for argument in expression.args:
        yield from _expression_variables(argument)


def _expression_contains_stack_pointer(expression: _Expr) -> bool:
    if expression.kind == "stack_pointer":
        return True
    if (
        expression.kind == "address"
        and isinstance(expression.value, tuple)
        and expression.value[0] == "stack"
    ):
        return True
    return any(
        _expression_contains_stack_pointer(argument) for argument in expression.args
    )


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


def _fp_to_bits(value: _Expr, bits: int) -> _Expr:
    constant = value
    if value.kind == "fp_cast" and value.args:
        constant = value.args[0]
    if constant.kind == "fp_const":
        return _const(_float_bits(float(constant.value), bits), bits)
    return _Expr("fp_to_bits", bits, args=(value,), bits=bits)


def _fp_from_integer(value: _Expr) -> _Expr:
    if value.kind == "const":
        return _fp_const(float(_signed_value(int(value.value), value.bits)))
    return _Expr("fp_from_int", args=(value,), bits=80)


def _fp_is_nan(value: _Expr) -> _Expr:
    if value.kind == "fp_const":
        return _const(math.isnan(float(value.value)), 1)
    return _Expr("fp_nan", args=(value,), bits=1)


def _fp_compare(operation: str, left: _Expr, right: _Expr) -> _Expr:
    if left.kind == "fp_const" and right.kind == "fp_const":
        lhs, rhs = float(left.value), float(right.value)
        comparison = {
            "eq": lambda: lhs == rhs,
            "ne": lambda: lhs != rhs,
            "lt": lambda: lhs < rhs,
            "le": lambda: lhs <= rhs,
        }.get(operation)
        if comparison is not None:
            return _const(comparison(), 1)
    return _Expr("fp_compare", operation, args=(left, right), bits=1)


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


def _sleigh_varnode_key(varnode: SleighVarnode) -> tuple[str, int, int]:
    return varnode.space, varnode.offset, varnode.size


def _unwrap_sleigh_boolean(expression: _Expr) -> _Expr:
    while (
        expression.kind == "zext"
        and len(expression.args) == 1
        and expression.args[0].bits == 1
    ):
        expression = expression.args[0]
    return expression


def _combine_sleigh_path_condition(existing: _Expr, incoming: _Expr) -> _Expr:
    if existing.kind == "const":
        if int(existing.value) == 0:
            return _const(0, 1)
        return incoming
    if incoming.kind == "const":
        if int(incoming.value) == 0:
            return _const(0, 1)
        return existing
    return _binary("land", existing, incoming, 1)


def _combine_sleigh_path_condition_or(existing: _Expr, incoming: _Expr) -> _Expr:
    if existing.kind == "const":
        if int(existing.value) != 0:
            return _const(1, 1)
        return incoming
    if incoming.kind == "const":
        if int(incoming.value) != 0:
            return _const(1, 1)
        return existing
    return _binary("lor", existing, incoming, 1)


def _copy_sleigh_execution_path(
    path: _SleighExecutionPath, pc: int, condition: _Expr
) -> _SleighExecutionPath:
    return _SleighExecutionPath(
        pc,
        path.state.copy(),
        dict(path.scratch),
        dict(path.flag_updates),
        condition,
        path.steps,
        set(path.nonzero_sources),
        set(path.zero_sources),
        dict(path.bit_facts),
        dict(path.zero_high_from),
        dict(path.nonzero_high_from),
        path.inconsistent,
        list(path.memory_stores),
        path.memory_epoch,
    )


def _sleigh_shift_source(expression: _Expr) -> tuple[_Expr, int] | None:
    if (
        expression.kind == "op"
        and expression.value == "lshr"
        and len(expression.args) == 2
        and expression.args[1].kind == "const"
    ):
        return expression.args[0], int(expression.args[1].value)
    return None


def _sleigh_branch_fact(condition: _Expr):
    if condition.kind != "op" or condition.value not in {"eq", "ne"}:
        return None
    left, right = condition.args
    if right.kind != "const" or int(right.value) != 0:
        return None
    comparison = str(condition.value)
    if left.bits > 1:
        shift = _sleigh_shift_source(left)
        if shift is not None:
            source, amount = shift
            return "high", source, amount, comparison == "ne"
        if left.kind == "op" and left.value == "and" and len(left.args) == 2:
            shifted, mask = left.args
            if mask.kind == "const" and int(mask.value) == 1:
                shift = _sleigh_shift_source(shifted)
                if shift is not None:
                    source, amount = shift
                    return "bit", source, amount, comparison == "ne"
                return "bit", shifted, 0, comparison == "ne"
            if shifted.kind == "const" and int(shifted.value) == 1:
                shift = _sleigh_shift_source(mask)
                if shift is not None:
                    source, amount = shift
                    return "bit", source, amount, comparison == "ne"
                return "bit", mask, 0, comparison == "ne"
        return "nonzero", left, None, comparison == "ne"
    return None


def _add_sleigh_branch_fact(
    path: _SleighExecutionPath, condition: _Expr, truth: bool
) -> None:
    fact = _sleigh_branch_fact(condition)
    if fact is None:
        return
    kind, source, amount, condition_is_true = fact
    asserted = truth == condition_is_true
    if kind == "nonzero":
        (path.nonzero_sources if asserted else path.zero_sources).add(source)
    elif kind == "bit":
        key = (source, amount)
        previous = path.bit_facts.get(key)
        if previous is not None and previous != asserted:
            path.inconsistent = True
        path.bit_facts[key] = asserted
    elif kind == "high":
        target = path.nonzero_high_from if asserted else path.zero_high_from
        previous = target.get(source)
        if previous is None:
            target[source] = amount
        elif asserted:
            target[source] = max(previous, amount)
        else:
            target[source] = min(previous, amount)


def _sleigh_path_constraints_consistent(path: _SleighExecutionPath) -> bool:
    if path.inconsistent or path.nonzero_sources.intersection(path.zero_sources):
        return False
    for (source, bit), value in path.bit_facts.items():
        if (bit < 0 or bit >= source.bits) and value:
            return False
        if value and source in path.zero_sources:
            return False
        zero_from = path.zero_high_from.get(source)
        if value and zero_from is not None and bit >= zero_from:
            return False
    for source in path.zero_sources:
        if any(
            bit_source == source and value
            for (bit_source, _), value in path.bit_facts.items()
        ):
            return False
    for source in path.nonzero_sources:
        if all(
            path.bit_facts.get((source, bit)) is False for bit in range(source.bits)
        ):
            return False
        if path.zero_high_from.get(source, source.bits) <= 0:
            return False
    for source, amount in path.nonzero_high_from.items():
        if amount >= source.bits or source in path.zero_sources:
            return False
        zero_from = path.zero_high_from.get(source)
        if zero_from is not None and amount >= zero_from:
            return False
        if all(
            path.bit_facts.get((source, bit)) is False
            for bit in range(amount, source.bits)
        ):
            return False
    return True


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
    if value.kind == "insert":
        return _extract(value, 0, bits)
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
    if bits == value.bits and offset == 0:
        return value
    if value.kind == "select" and len(value.args) == 3:
        condition, yes, no = value.args
        return _select(
            condition,
            _extract(yes, offset, bits),
            _extract(no, offset, bits),
            bits,
        )
    if value.kind == "insert" and len(value.args) == 3:
        old, new, insertion_offset = value.args
        if insertion_offset.kind == "const":
            insertion_start = int(insertion_offset.value)
            insertion_end = insertion_start + new.bits
            requested_end = offset + bits
            if requested_end <= insertion_start or offset >= insertion_end:
                return _extract(old, offset, bits)
            if offset >= insertion_start and requested_end <= insertion_end:
                return _extract(new, offset - insertion_start, bits)
            boundaries = sorted(
                {
                    offset,
                    requested_end,
                    max(offset, insertion_start),
                    min(requested_end, insertion_end),
                }
            )
            pieces = []
            for start, end in itertools.pairwise(boundaries):
                if end <= start:
                    continue
                if start >= insertion_start and end <= insertion_end:
                    pieces.append(_extract(new, start - insertion_start, end - start))
                else:
                    pieces.append(_extract(old, start, end - start))
            result = pieces[-1]
            for piece in reversed(pieces[:-1]):
                result = _concat(result, piece, piece.bits)
            return result
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
    if (
        op == "boolnot"
        and value.kind == "op"
        and value.value in {"eq", "ne"}
        and len(value.args) == 2
        and value.args[1].kind == "const"
        and int(value.args[1].value) == 0
    ):
        inverse = "ne" if value.value == "eq" else "eq"
        return _binary(inverse, value.args[0], value.args[1], 1)
    return _Expr("unary", op, args=(value,), bits=bits)


def _binary(op: str, left: _Expr, right: _Expr, bits: int) -> _Expr:
    if op in {"eq", "ne"}:
        if right.kind == "const" and int(right.value) in {0, 1}:
            unwrapped = _unwrap_sleigh_boolean(left)
            if unwrapped.bits == 1 and unwrapped != left:
                return _binary(op, unwrapped, _const(int(right.value), 1), 1)
        if left.kind == "const" and int(left.value) in {0, 1}:
            unwrapped = _unwrap_sleigh_boolean(right)
            if unwrapped.bits == 1 and unwrapped != right:
                return _binary(op, _const(int(left.value), 1), unwrapped, 1)
        if left.bits == 1 and right.kind == "const" and int(right.value) in {0, 1}:
            if (op == "eq") == (int(right.value) == 1):
                return left
            return _unary("boolnot", left, 1)
        if right.bits == 1 and left.kind == "const" and int(left.value) in {0, 1}:
            if (op == "eq") == (int(left.value) == 1):
                return right
            return _unary("boolnot", right, 1)
    if op == "and" and (
        (left.kind == "const" and int(left.value) == 0)
        or (right.kind == "const" and int(right.value) == 0)
    ):
        return _const(0, bits)
    if op in {"eq", "ne"} and left == right:
        return _const(int(op == "eq"), 1)
    if op in {"add", "or", "xor"} and right.kind == "const" and int(right.value) == 0:
        return _resize(left, bits)
    if op in {"add", "or", "xor"} and left.kind == "const" and int(left.value) == 0:
        return _resize(right, bits)
    if (
        op in {"shl", "lshr", "ashr"}
        and right.kind == "const"
        and int(right.value) == 0
    ):
        return _resize(left, bits)
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
            shift_count_mask = 0x3F if bits >= 64 else 0x1F
            if op == "add":
                result = lhs + rhs
            elif op == "sub":
                result = lhs - rhs
            elif op == "mul":
                result = lhs * rhs
            elif op == "udiv" and unsigned_rhs != 0:
                result = unsigned_lhs // unsigned_rhs
            elif op == "urem" and unsigned_rhs != 0:
                result = unsigned_lhs % unsigned_rhs
            elif op in {"sdiv", "srem"} and signed_rhs != 0:
                signed_quotient = abs(signed_lhs) // abs(signed_rhs)
                if (signed_lhs < 0) != (signed_rhs < 0):
                    signed_quotient = -signed_quotient
                result = (
                    signed_quotient
                    if op == "sdiv"
                    else signed_lhs - signed_quotient * signed_rhs
                )
            elif op == "and":
                result = unsigned_lhs & unsigned_rhs
            elif op == "or":
                result = unsigned_lhs | unsigned_rhs
            elif op == "xor":
                result = unsigned_lhs ^ unsigned_rhs
            elif op == "shl":
                result = unsigned_lhs << (unsigned_rhs & shift_count_mask)
            elif op == "lshr":
                result = unsigned_lhs >> (unsigned_rhs & shift_count_mask)
            elif op == "ashr":
                result = signed_lhs >> (unsigned_rhs & shift_count_mask)
            else:
                result = None
            if result is not None:
                return _const(result & _mask(bits), bits)
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
                "ceil": lambda: float(math.ceil(number)),
                "floor": lambda: float(math.floor(number)),
                "trunc": lambda: float(math.trunc(number)),
                "exp2m1": lambda: math.exp2(number) - 1.0,
            }
            if operation in values:
                result = values[operation]()
                if (
                    operation in {"round", "ceil", "floor", "trunc"}
                    and result == 0.0
                    and math.copysign(1.0, number) < 0
                ):
                    result = -0.0
                return _fp_const(result)
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
    if bits > 64:
        return "__uint128_t"
    width = next(width for width in (8, 16, 32, 64) if bits <= width)
    return f"uint{width}_t"


def _signed_type(bits: int) -> str:
    if bits > 64:
        return "__int128"
    width = next(width for width in (8, 16, 32, 64) if bits <= width)
    return f"int{width}_t"


def _parameter_declaration(parameter: FunctionParameter) -> str:
    identifier = _c_identifier(parameter.name)
    if parameter.data_type is not None:
        return parameter.data_type.declaration(identifier)
    return f"{parameter.type_name} {identifier}"


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
