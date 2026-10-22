"""Bounded ParamID-style walks over SLEIGH SSA p-code and compatibility data.

The primary graph preserves each operation/value from the Python SSA builder;
a reduced symbolic graph remains supplemental for call-spec and stack evidence
that the current SSA model does not resolve. Neither layer implements Ghidra's
full heritage, type recovery, or prototype-recovery pipeline.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from .sleigh import SleighInstruction, SleighPcodeOp
from .ssa import SSAFunction, SSALocation
from .typesys import StorageLocation

_PARAM_NAME = re.compile(r"(?:float_)?param_\d+\Z")


class ParamRank(IntEnum):
    """Input-use ranking corresponding to the useful ParamMeasure ranks."""

    DIRECT_WRITE_WITHOUT_READ = 1
    DIRECT_READ = 2
    DIRECT_WRITE_UNKNOWN_READ = 3
    SUBFUNCTION_PARAMETER = 4
    FUNCTION_RETURN = 5
    INDIRECT = 6
    UNREFERENCED = 7

    @property
    def label(self) -> str:
        return {
            ParamRank.DIRECT_WRITE_WITHOUT_READ: "direct-write-without-read",
            ParamRank.DIRECT_READ: "direct-read",
            ParamRank.DIRECT_WRITE_UNKNOWN_READ: "direct-write-unknown-read",
            ParamRank.SUBFUNCTION_PARAMETER: "subfunction-parameter",
            ParamRank.FUNCTION_RETURN: "function-return",
            ParamRank.INDIRECT: "indirect",
            ParamRank.UNREFERENCED: "unreferenced",
        }[self]


@dataclass(frozen=True, slots=True)
class PcodeVarnode:
    """A named value from either the SSA graph or compatibility graph."""

    identifier: int
    name: str
    size_bits: int
    kind: str


@dataclass(frozen=True, slots=True)
class PcodeOp:
    """One SSA p-code op, compatibility definition, merge, or use sink."""

    opcode: str
    inputs: tuple[int, ...]
    output: int | None = None
    address: int | None = None
    terminal_rank: ParamRank | None = None
    evidence: str = ""
    mnemonic: str | None = None
    input_ranks: tuple[ParamRank | None, ...] = ()


@dataclass(frozen=True, slots=True)
class InstructionDataFlow:
    """Register/memory use-def summary for one decoded machine instruction.

    ``reads`` and ``writes`` are indices into the owning graph's varnode list;
    the named register/memory collections are stable display labels. A direct
    callee address, indirect target node, selected argument values, and call
    result are retained separately.
    """

    address: int
    mnemonic: str
    reads: tuple[int, ...] = ()
    writes: tuple[int, ...] = ()
    read_registers: tuple[str, ...] = ()
    written_registers: tuple[str, ...] = ()
    read_memory: tuple[str, ...] = ()
    written_memory: tuple[str, ...] = ()
    call_target: int | None = None
    call_target_name: str | None = None
    direct_target_address: int | None = None
    call_arguments: tuple[int, ...] = ()
    call_result: int | None = None
    operation_indices: tuple[int, ...] = ()
    sleigh_ops: tuple[SleighPcodeOp, ...] = ()
    sleigh_error: str | None = None


@dataclass(frozen=True, slots=True)
class ParamMeasurement:
    """Best input-use rank and bounded use counts for one entry value."""

    name: str
    rank: ParamRank
    direct_reads: int = 0
    call_uses: int = 0
    return_uses: int = 0
    indirect_uses: int = 0
    traversed_nodes: int = 0
    depth_limited: bool = False

    @property
    def active(self) -> bool:
        return self.rank != ParamRank.UNREFERENCED

    @property
    def evidence(self) -> tuple[str, ...]:
        result = [f"paramid-rank:{self.rank.label}"]
        if self.direct_reads:
            result.append(f"paramid-direct-reads:{self.direct_reads}")
        if self.call_uses:
            result.append(f"paramid-call-uses:{self.call_uses}")
        if self.return_uses:
            result.append(f"paramid-return-uses:{self.return_uses}")
        if self.indirect_uses:
            result.append(f"paramid-indirect-uses:{self.indirect_uses}")
        if self.depth_limited:
            result.append("paramid-walk-depth-limit-reached")
        return tuple(result)


@dataclass(slots=True)
class ParamTrial:
    """A tentative ABI input range measured against SSA and fallback evidence."""

    name: str
    slot: int
    size_bits: int
    locations: tuple[StorageLocation, ...] = ()
    rank: ParamRank = ParamRank.UNREFERENCED
    checked: bool = False
    active: bool = False
    selected: bool = False
    used: bool = False
    definitely_not_used: bool = False
    evidence: tuple[str, ...] = ()
    measurement: ParamMeasurement | None = None

    def mark_active(self) -> None:
        self.checked = True
        self.active = True
        self.definitely_not_used = False

    def mark_inactive(self) -> None:
        self.checked = True
        self.active = False

    def mark_selected(self) -> None:
        """Select this trial as a candidate without asserting it is active/used."""
        self.selected = True

    def mark_unselected(self) -> None:
        self.selected = False

    def mark_used(self) -> None:
        self.checked = True
        self.active = True
        self.selected = True
        self.used = True
        self.definitely_not_used = False

    def mark_no_use(self) -> None:
        self.checked = True
        self.active = False
        self.used = False
        self.definitely_not_used = True

    def is_checked(self) -> bool:
        return self.checked

    def is_active(self) -> bool:
        return self.active

    def is_selected(self) -> bool:
        return self.selected

    def is_used(self) -> bool:
        return self.used


class ParamActive:
    """Small candidate-trial container inspired by Ghidra's ParamActive."""

    def __init__(self, recover_subcalls: bool = False) -> None:
        self.recover_subcalls = recover_subcalls
        self._trials: list[ParamTrial] = []
        self._by_name: dict[str, ParamTrial] = {}

    @property
    def trials(self) -> tuple[ParamTrial, ...]:
        return tuple(self._trials)

    @property
    def active_trials(self) -> tuple[ParamTrial, ...]:
        return tuple(trial for trial in self._trials if trial.active)

    @property
    def inactive_trials(self) -> tuple[ParamTrial, ...]:
        return tuple(trial for trial in self._trials if not trial.active)

    @property
    def selected_trials(self) -> tuple[ParamTrial, ...]:
        return tuple(trial for trial in self._trials if trial.selected)

    def register_trial(
        self,
        name: str,
        size_bits: int,
        locations: Iterable[StorageLocation] = (),
    ) -> ParamTrial:
        """Register a candidate input, retaining its slot on duplicate reports."""
        found = self._by_name.get(name)
        supplied_locations = tuple(locations)
        if found is not None:
            found.size_bits = max(found.size_bits, size_bits)
            if supplied_locations:
                found.locations = supplied_locations
            return found
        trial = ParamTrial(
            name=name,
            slot=len(self._trials) + 1,
            size_bits=size_bits,
            locations=supplied_locations,
        )
        self._trials.append(trial)
        self._by_name[name] = trial
        return trial

    def get_trial(self, name: str) -> ParamTrial | None:
        return self._by_name.get(name)

    def get_num_trials(self) -> int:
        return len(self._trials)

    def get_num_used(self) -> int:
        """Return the leading used prefix, as the sorted Ghidra container does."""
        count = 0
        for trial in self._trials:
            if not trial.used:
                break
            count += 1
        return count

    def get_num_selected(self) -> int:
        return sum(trial.selected for trial in self._trials)

    def select_candidates(
        self,
        names: Iterable[str] | None = None,
        *,
        include_inactive: bool = False,
    ) -> tuple[ParamTrial, ...]:
        """Select candidate trials independently from activity and final use.

        With no explicit names, only active trials are selected. A caller that
        has independent positional evidence (for example, an ABI-resolved
        signature with a gap) may include inactive trials explicitly.
        """
        requested = (
            {trial.name for trial in self._trials if trial.active}
            if names is None
            else set(names)
        )
        selected: list[ParamTrial] = []
        for trial in self._trials:
            trial.mark_unselected()
            if trial.name in requested and (trial.active or include_inactive):
                trial.mark_selected()
                selected.append(trial)
        return tuple(selected)

    def sort_by_position(self) -> None:
        """Order trials by the current positional ABI names and renumber slots."""
        self._trials.sort(key=lambda trial: _parameter_order(trial.name))
        for slot, trial in enumerate(self._trials, start=1):
            trial.slot = slot
        self._by_name = {trial.name: trial for trial in self._trials}

    def delete_unused_trials(self) -> None:
        """Compact to trials that have reached a final used verdict."""
        self._trials = [trial for trial in self._trials if trial.used]
        self._by_name = {trial.name: trial for trial in self._trials}
        for slot, trial in enumerate(self._trials, start=1):
            trial.slot = slot


class PcodeLikeGraph:
    """SSA p-code adapter or legacy expression/use-site graph for one function."""

    MAX_EXPRESSION_DEPTH = 10

    def __init__(self) -> None:
        self.nodes: list[PcodeVarnode] = []
        self.ops: list[PcodeOp] = []
        self._variable_nodes: dict[tuple[str, int], int] = {}
        self._expression_nodes: dict[int, tuple[Any, int]] = {}
        self._definitions: dict[int, int] = {}
        self._source_names: dict[str, set[str]] = defaultdict(set)
        self._source_nodes: dict[str, set[int]] = defaultdict(set)
        self._use_keys: set[tuple[int, str, int | None, int, str]] = set()
        self._temporary_counter = 0
        self._register_versions: dict[str, int] = defaultdict(int)
        self._instruction_records: dict[int, dict[str, Any]] = {}
        self._source_widths: dict[str, int] = {}
        self.ssa_function: SSAFunction | None = None
        self.supplemental_graph: PcodeLikeGraph | None = None
        self.depth_limited_uses = 0

    @classmethod
    def from_ssa(
        cls,
        ssa_function: SSAFunction,
        parameter_locations: Mapping[SSALocation, str],
        *,
        parameter_widths: Mapping[str, int] | None = None,
        return_registers: Iterable[str] | None = None,
        supplemental_graph: PcodeLikeGraph | None = None,
    ) -> PcodeLikeGraph:
        """Adapt the complete reachable SLEIGH SSA graph for ParamID walks.

        Every p-code operation, phi input, output slice, and ABI return sink
        remains linked to the original SSA value ids. ``return_registers`` can
        restrict candidate sinks to families selected by the inferred return
        type. ``supplemental_graph`` is only for evidence not yet represented by
        SLEIGH SSA (notably outgoing stack arguments and call-spec locations).
        """
        graph = cls()
        graph.ssa_function = ssa_function
        graph.supplemental_graph = supplemental_graph
        graph._source_widths = {
            name: max(1, width) for name, width in (parameter_widths or {}).items()
        }
        for value in ssa_function.values:
            name = (
                parameter_locations.get(value.storage)
                if value.kind == "input" and value.storage is not None
                else None
            )
            size_bits = value.size_bits
            if name is not None:
                size_bits = max(size_bits, graph._source_widths.get(name, 1))
                graph._source_widths[name] = max(
                    graph._source_widths.get(name, 1), size_bits
                )
                graph._source_nodes[name].add(value.identifier)
            graph.nodes.append(
                PcodeVarnode(
                    value.identifier,
                    name or value.name,
                    size_bits,
                    "input" if name is not None else value.kind,
                )
            )

        for item in ssa_function.blocks:
            if item.reachable:
                graph.begin_instruction(item.instruction_address, item.instruction_mnemonic)

        values_by_id = ssa_function.value_by_id
        callother_passthrough_sources: dict[int, int] = {}
        round_width_by_mnemonic = {
            "roundss": 4,
            "roundsd": 8,
            "roundps": 16,
            "roundpd": 16,
        }
        for operation in ssa_function.operations:
            mnemonic = operation.instruction_mnemonic.lower()
            round_source_bytes = (
                round_width_by_mnemonic.get(mnemonic)
                if operation.opcode == "CALLOTHER"
                else None
            )
            is_round_userop = round_source_bytes is not None
            inputs: list[int] = []
            input_ranks: list[ParamRank | None] = []
            for input_index, operand in enumerate(operation.inputs):
                if operand.kind in {"constant", "address", "memory"}:
                    continue
                input_values = operand.values
                if operation.opcode == "SUBPIECE" and input_index == 0:
                    byte_offset = (
                        operation.inputs[1].constant
                        if len(operation.inputs) > 1
                        and operation.inputs[1].constant is not None
                        else 0
                    )
                    output_bits = (
                        values_by_id[operation.output_value].size_bits
                        if operation.output_value is not None
                        else operand.size_bits
                    )
                    output_bytes = max(1, (output_bits + 7) // 8)
                    input_values = input_values[
                        byte_offset : byte_offset + output_bytes
                    ]
                rank = (
                    None
                    if is_round_userop
                    else _ssa_input_rank(
                        operation.opcode,
                        input_index,
                        input_values,
                        operation.output_value is not None,
                    )
                )
                inputs.extend(input_values)
                input_ranks.extend(rank for _ in input_values)
            graph._set_definition(
                PcodeOp(
                    operation.opcode,
                    tuple(inputs),
                    operation.output_value,
                    address=operation.instruction_address,
                    evidence=f"ssa-pcode:{operation.opcode.lower()}",
                    mnemonic=operation.instruction_mnemonic,
                    input_ranks=(
                        tuple(input_ranks)
                        if is_round_userop or any(
                            rank is not None for rank in input_ranks
                        )
                        else ()
                    ),
                )
            )
            if is_round_userop and len(operation.inputs) > 2:
                destination_values = (
                    operation.inputs[1].values
                    if len(operation.inputs) > 1
                    else ()
                )
                source_values = operation.inputs[2].values[:round_source_bytes]
                for byte_index, value_id in enumerate(source_values):
                    if byte_index < len(operation.output_parts):
                        callother_passthrough_sources[
                            operation.output_parts[byte_index]
                        ] = value_id
                if mnemonic in {"roundss", "roundsd"}:
                    for byte_index in range(
                        round_source_bytes,
                        min(len(destination_values), len(operation.output_parts)),
                    ):
                        callother_passthrough_sources[
                            operation.output_parts[byte_index]
                        ] = destination_values[byte_index]
                if source_values:
                    graph._append_operation(
                        PcodeOp(
                            "CALLOTHER_INPUT",
                            tuple(source_values),
                            address=operation.instruction_address,
                            terminal_rank=ParamRank.DIRECT_READ,
                            evidence="ssa-round-source-read",
                            mnemonic=operation.instruction_mnemonic,
                        )
                    )
            for argument_name, value_ids in operation.call_argument_values:
                defined_values = tuple(
                    value_id
                    for value_id in value_ids
                    if values_by_id[value_id].kind not in {"input", "memory_input"}
                )
                if defined_values:
                    graph._append_operation(
                        PcodeOp(
                            "CALL",
                            tuple(dict.fromkeys(defined_values)),
                            address=operation.instruction_address,
                            terminal_rank=ParamRank.SUBFUNCTION_PARAMETER,
                            evidence=f"ssa-abi-call-argument:{argument_name}",
                            mnemonic=operation.instruction_mnemonic,
                        )
                    )

        blocks = ssa_function.block_by_id
        for value in ssa_function.values:
            address = value.instruction_address
            mnemonic = None
            if value.block_id is not None and value.block_id in blocks:
                block = blocks[value.block_id]
                address = address or block.instruction_address
                mnemonic = block.instruction_mnemonic
            if value.kind == "phi":
                graph._set_definition(
                    PcodeOp(
                        "MULTIEQUAL",
                        tuple(incoming for _predecessor, incoming in value.incoming),
                        value.identifier,
                        address=address,
                        evidence="ssa-phi",
                        mnemonic=mnemonic,
                    )
                )
            elif value.kind == "slice" and value.source is not None:
                pass_through = callother_passthrough_sources.get(value.identifier)
                if pass_through is None:
                    graph._set_definition(
                        PcodeOp(
                            "SUBPIECE",
                            (value.source,),
                            value.identifier,
                            address=address,
                            evidence="ssa-byte-slice",
                            mnemonic=mnemonic,
                            input_ranks=(None,),
                        )
                    )
                else:
                    graph._set_definition(
                        PcodeOp(
                            "COPY",
                            (pass_through,),
                            value.identifier,
                            address=address,
                            evidence="ssa-callother-passthrough",
                            mnemonic=mnemonic,
                            input_ranks=(None,),
                        )
                    )
            elif value.kind == "call_stack_restore" and value.source is not None:
                graph._set_definition(
                    PcodeOp(
                        "COPY",
                        (value.source,),
                        value.identifier,
                        address=address,
                        evidence="ssa-call-stack-restore",
                        mnemonic=mnemonic,
                        input_ranks=(None,),
                    )
                )
            elif value.kind in {"call_return", "call_clobber"}:
                graph._set_definition(
                    PcodeOp(
                        "CALL_RESULT",
                        (),
                        value.identifier,
                        address=address,
                        evidence=f"ssa-{value.kind}",
                        mnemonic=mnemonic,
                    )
                )

        allowed_return_registers = (
            None
            if return_registers is None
            else {register.lower() for register in return_registers}
        )
        for site in ssa_function.return_sites:
            return_values = tuple(
                value_id
                for register, operand in site.register_values
                if allowed_return_registers is None
                or register.lower() in allowed_return_registers
                for value_id in operand.values
                if values_by_id[value_id].kind not in {"input", "memory_input"}
            )
            if return_values:
                graph._append_operation(
                    PcodeOp(
                        "RETURN",
                        tuple(dict.fromkeys(return_values)),
                        address=site.instruction_address,
                        terminal_rank=ParamRank.FUNCTION_RETURN,
                        evidence="ssa-abi-return-registers",
                        mnemonic="ret",
                    )
                )
        return graph

    @property
    def is_ssa_backed(self) -> bool:
        return self.ssa_function is not None

    @property
    def ssa_operation_count(self) -> int:
        return len(self.ssa_function.operations) if self.ssa_function else 0

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._source_nodes, key=_parameter_order))

    @property
    def operation_count(self) -> int:
        return len(self.ops)

    @property
    def instruction_flows(self) -> tuple[InstructionDataFlow, ...]:
        """Return stable address-ordered instruction summaries."""
        result: list[InstructionDataFlow] = []
        for address, record in sorted(self._instruction_records.items()):
            result.append(
                InstructionDataFlow(
                    address=address,
                    mnemonic=record["mnemonic"],
                    reads=tuple(sorted(record["reads"])),
                    writes=tuple(sorted(record["writes"])),
                    read_registers=tuple(sorted(record["read_registers"])),
                    written_registers=tuple(sorted(record["written_registers"])),
                    read_memory=tuple(sorted(record["read_memory"])),
                    written_memory=tuple(sorted(record["written_memory"])),
                    call_target=record["call_target"],
                    call_target_name=record["call_target_name"],
                    direct_target_address=record["direct_target_address"],
                    call_arguments=tuple(record["call_arguments"]),
                    call_result=record["call_result"],
                    operation_indices=tuple(sorted(record["operations"])),
                    sleigh_ops=tuple(record["sleigh_ops"]),
                    sleigh_error=record["sleigh_error"],
                )
            )
        return tuple(result)

    @property
    def sleigh_operation_count(self) -> int:
        """Count exact SLEIGH ops, separately from the reduced graph's ops."""
        return sum(
            len(record["sleigh_ops"])
            for record in self._instruction_records.values()
        )

    def attach_sleigh_instructions(
        self, instructions: Iterable[SleighInstruction]
    ) -> None:
        """Attach upstream SLEIGH translations to their matching instructions."""
        for instruction in instructions:
            record = self._instruction_record(
                instruction.address, instruction.mnemonic
            )
            record["sleigh_ops"] = instruction.ops
            record["sleigh_error"] = instruction.error

    def begin_instruction(self, address: int, mnemonic: str) -> None:
        """Ensure even effect-free decoded instructions appear in the summary."""
        self._instruction_record(address, mnemonic)

    def record_register_read(
        self,
        register: str,
        expression: Any,
        *,
        address: int,
        mnemonic: str,
        size_bits: int,
        source_node: int | None = None,
    ) -> int:
        """Record a machine-register read and return its p-code-like value node."""
        node = source_node
        if node is None:
            node = self._materialize(
                expression, 0, self._expression_sources(expression)
            )
        operation = PcodeOp(
            f"READ_{_safe_opcode(register)}",
            (node,),
            address=address,
            evidence=f"register-read:{register}:{size_bits}",
            mnemonic=mnemonic,
        )
        self._append_operation(operation)
        self._annotate_instruction(
            address, mnemonic, read_registers=(register,)
        )
        return node

    def record_register_definition(
        self,
        register: str,
        expression: Any,
        *,
        address: int,
        mnemonic: str,
        size_bits: int,
        input_nodes: Iterable[int] = (),
    ) -> int:
        """Create a versioned register definition linked to instruction reads."""
        inputs = tuple(dict.fromkeys(input_nodes))
        if not inputs:
            inputs = (
                self._materialize(
                    expression, 0, self._expression_sources(expression)
                ),
            )
        version = self._register_versions[register]
        self._register_versions[register] += 1
        output = self._new_node(
            f"{register}@{address:X}.{version}", size_bits, "register"
        )
        operation = PcodeOp(
            f"X86_{_safe_opcode(mnemonic)}",
            inputs,
            output,
            address=address,
            evidence=f"register-definition:{register}",
            mnemonic=mnemonic,
        )
        self._set_definition(operation)
        self._annotate_instruction(
            address, mnemonic, written_registers=(register,)
        )
        return output

    def record_operand_read(
        self,
        expression: Any,
        *,
        address: int,
        mnemonic: str,
        evidence: str = "operand-read",
    ) -> int:
        """Record a non-register operand, such as an immediate constant."""
        node = self._materialize(
            expression, 0, self._expression_sources(expression)
        )
        self._append_operation(
            PcodeOp(
                "READ_OPERAND",
                (node,),
                address=address,
                evidence=evidence,
                mnemonic=mnemonic,
            )
        )
        return node

    def record_memory_load(
        self,
        expression: Any,
        address_expression: Any,
        *,
        location: str,
        address: int,
        mnemonic: str,
        size_bits: int,
        address_registers: Iterable[str] = (),
    ) -> int:
        """Record a load; stack loads retain the current symbolic contents."""
        address_node = self._materialize(
            address_expression, 0, self._expression_sources(address_expression)
        )
        value_node = self._materialize(
            expression, 0, self._expression_sources(expression)
        )
        if str(getattr(expression, "kind", "")).lower() in {
            "load",
            "fp_load",
            "fp_int_load",
            "typed_load",
        }:
            output = value_node
            inputs = (address_node,)
            opcode = "LOAD"
        else:
            output = self._new_temporary_node("load_stack", size_bits)
            inputs = (address_node, value_node)
            opcode = "LOAD_STACK"
        self._set_definition(
            PcodeOp(
                opcode,
                inputs,
                output,
                address=address,
                evidence=f"memory-read:{location}",
                mnemonic=mnemonic,
            )
        )
        self._annotate_instruction(
            address,
            mnemonic,
            read_registers=address_registers,
            read_memory=(location,),
        )
        return output

    def record_memory_store(
        self,
        value: Any,
        address_expression: Any,
        *,
        location: str,
        address: int,
        mnemonic: str,
        size_bits: int,
        address_registers: Iterable[str] = (),
    ) -> None:
        """Record a memory write without assuming alias-free memory SSA."""
        address_node = self._materialize(
            address_expression, 0, self._expression_sources(address_expression)
        )
        value_node = self._materialize(value, 0, self._expression_sources(value))
        self._append_operation(
            PcodeOp(
                "STORE_STACK" if location.startswith("stack[") else "STORE",
                (address_node, value_node),
                address=address,
                evidence=f"memory-write:{location}:{size_bits}",
                mnemonic=mnemonic,
            )
        )
        self._annotate_instruction(
            address,
            mnemonic,
            read_registers=address_registers,
            written_memory=(location,),
        )

    def record_call(
        self,
        arguments: Iterable[Any],
        *,
        target: Any | None = None,
        result: Any | None = None,
        address: int,
        mnemonic: str,
        evidence: str = "",
        direct_target_address: int | None = None,
        call_target_name: str | None = None,
    ) -> None:
        """Record a call operation; argument ranks are supplied by call sinks."""
        argument_nodes = tuple(
            self._materialize(argument, 0, self._expression_sources(argument))
            for argument in arguments
        )
        target_node = (
            self._materialize(target, 0, self._expression_sources(target))
            if target is not None
            else None
        )
        inputs = ((target_node,) if target_node is not None else ()) + argument_nodes
        output = (
            self._materialize(result, 0, self._expression_sources(result))
            if result is not None
            else None
        )
        self._set_definition(
            PcodeOp(
                "CALL",
                inputs,
                output,
                address=address,
                evidence=evidence,
                mnemonic=mnemonic,
            )
        )
        record = self._instruction_record(address, mnemonic)
        record["call_target"] = target_node
        record["call_target_name"] = call_target_name
        record["direct_target_address"] = direct_target_address
        record["call_arguments"] = argument_nodes
        record["call_result"] = output

    def record_phi(
        self,
        name: str,
        values: Iterable[Any],
        size_bits: int,
        address: int | None = None,
    ) -> None:
        """Add/update a MULTIEQUAL-like definition for a CFG merge value."""
        value_list = tuple(values)
        output = self._variable_node(name, size_bits, "merge")
        inputs = tuple(
            self._materialize(value, 0, self._expression_sources(value))
            for value in value_list
        )
        sources = set().union(
            *(self._expression_sources(value) for value in value_list)
        ) if value_list else set()
        self._source_names[name].update(sources)
        self._set_definition(
            PcodeOp("MULTIEQUAL", inputs, output, address=address)
        )

    def record_use(
        self,
        expression: Any,
        *,
        role: str,
        source_names: Iterable[str] = (),
        address: int | None = None,
        size_bits: int = 0,
        evidence: str = "",
        mnemonic: str | None = None,
    ) -> None:
        """Record an input read, call argument, return, or indirect use.

        Expression materialization is limited to ten levels. If a deeper
        expression still carries an input, it is represented by an opaque
        operation whose inputs are the known entry varnodes.
        """
        known_sources = set(source_names) | self._expression_sources(expression)
        if not known_sources and not self._contains_phi(expression):
            return
        input_node = self._materialize(expression, 0, known_sources)
        opcode, rank = _role_description(role)
        key = (input_node, role, address, size_bits, evidence)
        if key in self._use_keys:
            return
        self._use_keys.add(key)
        self._append_operation(
            PcodeOp(
                opcode,
                (input_node,),
                address=address,
                terminal_rank=rank,
                evidence=evidence,
                mnemonic=mnemonic,
            )
        )

    def measure(self, name: str, max_depth: int = 10) -> ParamMeasurement:
        """Walk forward from entry varnodes using bounded ParamMeasure ranks."""
        starts = self._source_nodes.get(name, set())
        if not starts:
            return ParamMeasurement(name, ParamRank.UNREFERENCED)

        users: dict[int, list[PcodeOp]] = defaultdict(list)
        for operation in self.ops:
            for input_node in operation.inputs:
                users[input_node].append(operation)

        queue = deque((node, 0) for node in starts)
        visited: set[int] = set()
        visited_operations: set[int] = set()
        best = ParamRank.UNREFERENCED
        direct_reads = call_uses = return_uses = indirect_uses = 0
        depth_limited = False

        def count_rank(rank: ParamRank) -> None:
            nonlocal best, direct_reads, call_uses, return_uses, indirect_uses
            best = min(best, rank)
            if rank == ParamRank.DIRECT_READ:
                direct_reads += 1
            elif rank == ParamRank.SUBFUNCTION_PARAMETER:
                call_uses += 1
            elif rank == ParamRank.FUNCTION_RETURN:
                return_uses += 1
            elif rank == ParamRank.INDIRECT:
                indirect_uses += 1

        while queue:
            node, depth = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if depth >= max_depth:
                if users.get(node):
                    depth_limited = True
                continue
            for operation in users.get(node, ()):
                if id(operation) in visited_operations:
                    continue
                visited_operations.add(id(operation))
                if operation.opcode == "MULTIEQUAL" and operation.output is not None:
                    if operation.output not in visited:
                        queue.append((operation.output, depth + 1))
                    continue
                if operation.evidence in {
                    "ssa-byte-slice",
                    "ssa-call-stack-restore",
                    "ssa-callother-passthrough",
                }:
                    if operation.output is not None:
                        queue.append((operation.output, depth + 1))
                    continue
                if operation.opcode == "CALL" and operation.terminal_rank is None:
                    # The structural call's result is not derived from its
                    # caller-side arguments. Ranked CALL sinks are distinct ops.
                    continue
                if operation.opcode == "LOAD":
                    # A load consumes its address indirectly; its result is not
                    # a copy of that address. Stack-content loads use LOAD_STACK.
                    count_rank(ParamRank.INDIRECT)
                    continue
                if operation.terminal_rank is not None:
                    count_rank(operation.terminal_rank)
                    continue
                if operation.input_ranks:
                    matched_ranks = tuple(
                        dict.fromkeys(
                            operation.input_ranks[index]
                            for index, input_node in enumerate(operation.inputs)
                            if input_node == node
                            and index < len(operation.input_ranks)
                        )
                    )
                    for rank in matched_ranks:
                        if rank is not None:
                            count_rank(rank)
                    continue
                if operation.output is not None:
                    # A normal p-code consumer reads its input. The SSA-backed
                    # walk also follows the result to recover later call/return
                    # sinks reached through COPY, arithmetic, and phi values.
                    count_rank(ParamRank.DIRECT_READ)
                    if self.is_ssa_backed and operation.output not in visited:
                        queue.append((operation.output, depth + 1))

        return ParamMeasurement(
            name=name,
            rank=best,
            direct_reads=direct_reads,
            call_uses=call_uses,
            return_uses=return_uses,
            indirect_uses=indirect_uses,
            traversed_nodes=len(visited),
            depth_limited=depth_limited,
        )

    def _materialize(
        self, expression: Any, depth: int, source_names: set[str]
    ) -> int:
        expression_id = id(expression)
        cached = self._expression_nodes.get(expression_id)
        if cached is not None and cached[0] is expression:
            return cached[1]

        kind = str(getattr(expression, "kind", "unknown")).lower()
        value = getattr(expression, "value", None)
        size_bits = max(1, int(getattr(expression, "bits", 0) or 0))
        if kind == "var":
            name = str(value)
            node = self._variable_node(name, size_bits)
            self._expression_nodes[expression_id] = (expression, node)
            return node
        if kind == "const":
            node = self._new_node(f"const_{value!s}", size_bits, "constant")
            self._expression_nodes[expression_id] = (expression, node)
            return node
        if depth >= self.MAX_EXPRESSION_DEPTH:
            self.depth_limited_uses += 1
            inputs = tuple(
                self._variable_node(name, size_bits)
                for name in sorted(source_names, key=_parameter_order)
            )
            output = self._new_temporary_node("opaque_expr", size_bits)
            self._set_definition(PcodeOp("OPAQUE_EXPR", inputs, output))
            self._expression_nodes[expression_id] = (expression, output)
            return output

        arguments = tuple(getattr(expression, "args", ()))
        inputs = tuple(
            self._materialize(argument, depth + 1, source_names)
            for argument in arguments
        )
        output = self._new_temporary_node(f"tmp_{kind}", size_bits)
        opcode = kind.upper().replace("-", "_")
        if value is not None and kind not in {"address", "load", "typed_load"}:
            opcode = f"{opcode}:{type(value).__name__}"
        self._set_definition(PcodeOp(opcode, inputs, output))
        self._expression_nodes[expression_id] = (expression, output)
        return output

    def _variable_node(self, name: str, size_bits: int, kind: str | None = None) -> int:
        width = max(1, size_bits)
        key = (name, width)
        existing = self._variable_nodes.get(key)
        if existing is not None:
            return existing
        node_kind = kind or ("input" if _PARAM_NAME.fullmatch(name) else "variable")
        identifier = self._new_node(name, width, node_kind)
        self._variable_nodes[key] = identifier
        if _PARAM_NAME.fullmatch(name):
            self._source_nodes[name].add(identifier)
        return identifier

    def _new_node(self, name: str, size_bits: int, kind: str) -> int:
        identifier = len(self.nodes)
        self.nodes.append(
            PcodeVarnode(identifier, name, max(1, size_bits), kind)
        )
        return identifier

    def _new_temporary_node(self, label: str, size_bits: int) -> int:
        name = f"{label}_{self._temporary_counter}"
        self._temporary_counter += 1
        return self._new_node(name, size_bits, "temporary")

    def _set_definition(self, operation: PcodeOp) -> int:
        if operation.output is None:
            return self._append_operation(operation)
        existing = self._definitions.get(operation.output)
        if existing is None:
            index = len(self.ops)
            self._definitions[operation.output] = index
            self.ops.append(operation)
        else:
            index = existing
            self.ops[index] = operation
        self._track_instruction_operation(index, operation)
        return index

    def _append_operation(self, operation: PcodeOp) -> int:
        index = len(self.ops)
        self.ops.append(operation)
        self._track_instruction_operation(index, operation)
        return index

    def _track_instruction_operation(
        self, index: int, operation: PcodeOp
    ) -> None:
        if operation.address is None or operation.mnemonic is None:
            return
        record = self._instruction_record(operation.address, operation.mnemonic)
        record["operations"].add(index)
        record["reads"].update(operation.inputs)
        if operation.output is not None:
            record["writes"].add(operation.output)

    def _instruction_record(self, address: int, mnemonic: str) -> dict[str, Any]:
        record = self._instruction_records.get(address)
        if record is None:
            record = {
                "mnemonic": mnemonic.lower(),
                "reads": set(),
                "writes": set(),
                "read_registers": set(),
                "written_registers": set(),
                "read_memory": set(),
                "written_memory": set(),
                "call_target": None,
                "call_target_name": None,
                "direct_target_address": None,
                "call_arguments": (),
                "call_result": None,
                "operations": set(),
                "sleigh_ops": (),
                "sleigh_error": None,
            }
            self._instruction_records[address] = record
        return record

    def _annotate_instruction(
        self,
        address: int,
        mnemonic: str,
        *,
        read_registers: Iterable[str] = (),
        written_registers: Iterable[str] = (),
        read_memory: Iterable[str] = (),
        written_memory: Iterable[str] = (),
    ) -> None:
        record = self._instruction_record(address, mnemonic)
        record["read_registers"].update(read_registers)
        record["written_registers"].update(written_registers)
        record["read_memory"].update(read_memory)
        record["written_memory"].update(written_memory)

    def _expression_sources(self, expression: Any) -> set[str]:
        result: set[str] = set()
        pending = [expression]
        visited: set[int] = set()
        while pending and len(visited) < 4096:
            current = pending.pop()
            current_id = id(current)
            if current_id in visited:
                continue
            visited.add(current_id)
            if str(getattr(current, "kind", "")).lower() == "var":
                name = str(getattr(current, "value", ""))
                if _PARAM_NAME.fullmatch(name):
                    result.add(name)
                result.update(self._source_names.get(name, ()))
            pending.extend(getattr(current, "args", ()))
        return result

    def _contains_phi(self, expression: Any) -> bool:
        pending = [expression]
        visited: set[int] = set()
        while pending and len(visited) < 4096:
            current = pending.pop()
            current_id = id(current)
            if current_id in visited:
                continue
            visited.add(current_id)
            if str(getattr(current, "kind", "")).lower() == "var" and str(
                getattr(current, "value", "")
            ).startswith("phi_"):
                return True
            pending.extend(getattr(current, "args", ()))
        return False


class ParamIDAnalysis:
    """Build trial rankings over exact SSA p-code, with a reduced-graph fallback."""

    def __init__(self, graph: PcodeLikeGraph, max_depth: int = 10) -> None:
        self.graph = graph
        self.supplemental_graph = graph.supplemental_graph
        self.max_depth = max(1, max_depth)
        self.input_active = ParamActive()
        self.measurements: dict[str, ParamMeasurement] = {}

    @property
    def uses_ssa_pcode(self) -> bool:
        return self.graph.is_ssa_backed

    @property
    def valid(self) -> bool | None:
        if self.graph.ssa_function is None:
            return None
        return self.graph.ssa_function.valid

    @property
    def complete(self) -> bool | None:
        if self.graph.ssa_function is None:
            return None
        return self.graph.ssa_function.complete

    def analyze(self) -> ParamActive:
        self.input_active = ParamActive()
        self.measurements = {}
        names = set(self.graph.parameter_names)
        if self.supplemental_graph is not None:
            names.update(self.supplemental_graph.parameter_names)
        for name in sorted(names, key=_parameter_order):
            measurement = self._measure(name)
            self.measurements[name] = measurement
            width = _source_width(self.graph, name)
            if self.supplemental_graph is not None:
                width = max(width, _source_width(self.supplemental_graph, name))
            trial = self.input_active.register_trial(name, width)
            self._apply_measurement(trial, measurement)
        return self.input_active

    def _measure(self, name: str) -> ParamMeasurement:
        primary = self.graph.measure(name, self.max_depth)
        fallback_graph = self.supplemental_graph
        if fallback_graph is None or name not in fallback_graph.parameter_names:
            return primary
        fallback = fallback_graph.measure(name, self.max_depth)
        if primary.rank == ParamRank.UNREFERENCED:
            rank = fallback.rank
        else:
            rank = primary.rank
            # Full SSA p-code is authoritative for ordinary reads and memory
            # indirection. The compatibility graph only contributes call/return
            # sinks that SSA's current CALL model does not encode as inputs.
            if fallback.call_uses:
                rank = min(rank, ParamRank.SUBFUNCTION_PARAMETER)
            if fallback.return_uses:
                rank = min(rank, ParamRank.FUNCTION_RETURN)
        primary_is_active = primary.rank != ParamRank.UNREFERENCED
        return ParamMeasurement(
            name,
            rank,
            direct_reads=(
                primary.direct_reads if primary_is_active else fallback.direct_reads
            ),
            call_uses=max(primary.call_uses, fallback.call_uses),
            return_uses=max(primary.return_uses, fallback.return_uses),
            indirect_uses=(
                primary.indirect_uses if primary_is_active else fallback.indirect_uses
            ),
            traversed_nodes=primary.traversed_nodes + fallback.traversed_nodes,
            depth_limited=primary.depth_limited or fallback.depth_limited,
        )

    def bind_parameters(self, parameters: Iterable[Any]) -> ParamActive:
        """Attach resolved ABI storage to active and positional-placeholder trials."""
        selected_names: list[str] = []
        for parameter in parameters:
            locations = tuple(getattr(parameter, "locations", ()))
            data_type = getattr(parameter, "data_type", None)
            size_bytes = getattr(data_type, "size", None)
            if size_bytes is None:
                size_bytes = max((location.size for location in locations), default=1)
            trial = self.input_active.register_trial(
                parameter.name, max(1, int(size_bytes) * 8), locations
            )
            measurement = self.measurements.get(parameter.name)
            if measurement is None:
                measurement = ParamMeasurement(parameter.name, ParamRank.UNREFERENCED)
                self.measurements[parameter.name] = measurement
            self._apply_measurement(trial, measurement)
            selected_names.append(parameter.name)
        self.input_active.select_candidates(selected_names, include_inactive=True)
        self.input_active.sort_by_position()
        return self.input_active

    def select_candidates(
        self,
        names: Iterable[str] | None = None,
        *,
        include_inactive: bool = False,
    ) -> tuple[ParamTrial, ...]:
        """Apply an explicit candidate-selection step without finalizing use."""
        return self.input_active.select_candidates(
            names, include_inactive=include_inactive
        )

    def _apply_measurement(
        self, trial: ParamTrial, measurement: ParamMeasurement
    ) -> None:
        trial.measurement = measurement
        trial.rank = measurement.rank
        trial.evidence = measurement.evidence
        if measurement.active:
            trial.mark_active()
        else:
            trial.mark_inactive()


def _ssa_input_rank(
    opcode: str,
    input_index: int,
    values: tuple[int, ...],
    has_output: bool,
) -> ParamRank | None:
    """Classify data uses whose p-code op has no ordinary output read."""
    if not values:
        return None
    if opcode == "STORE":
        if input_index == 1:
            return ParamRank.INDIRECT
        if input_index == 2:
            return ParamRank.DIRECT_READ
        return None
    if opcode == "LOAD":
        return ParamRank.INDIRECT if input_index == 1 else None
    if opcode == "CBRANCH":
        return ParamRank.DIRECT_READ if input_index == 1 else None
    if opcode in {"BRANCH", "BRANCHIND", "CALLIND"}:
        return ParamRank.INDIRECT if input_index == 0 else None
    if opcode in {"CALL", "RETURN", "IMARK"}:
        return None
    return ParamRank.DIRECT_READ if not has_output else None


def _role_description(role: str) -> tuple[str, ParamRank]:
    normalized = role.lower().replace("-", "_")
    if normalized in {"call", "call_argument", "subfunction_parameter"}:
        return "CALL", ParamRank.SUBFUNCTION_PARAMETER
    if normalized in {"return", "function_return"}:
        return "RETURN", ParamRank.FUNCTION_RETURN
    if normalized in {"indirect", "indirect_use"}:
        return "INDIRECT", ParamRank.INDIRECT
    return "READ", ParamRank.DIRECT_READ


def _source_width(graph: PcodeLikeGraph, name: str) -> int:
    widths = [
        node.size_bits
        for node in graph.nodes
        if node.name == name and node.kind == "input"
    ]
    return max(widths, default=1)


def _parameter_order(name: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"(float_)?param_(\d+)", name)
    if match is None:
        return (2, 0, name)
    return (1 if match.group(1) else 0, int(match.group(2)), name)


def _safe_opcode(value: str) -> str:
    """Make user/register names safe and deterministic in diagnostic opcodes."""
    return re.sub(r"[^A-Z0-9_]", "_", value.upper())
