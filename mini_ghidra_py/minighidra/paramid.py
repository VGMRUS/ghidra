"""Ghidra-compatible ParamID measurements over SLEIGH SSA p-code.

The rank walk ports ``ParamMeasure``'s directional input/output rules, bounded
phi traversal, and loop-edge suppression. The SSA graph is authoritative for
instruction uses; the reduced symbolic graph supplements stack-slot input
identity and call-spec evidence not represented by raw SLEIGH p-code. This
module does not port Ghidra's separate prototype-model resolution pipeline.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum, IntEnum, IntFlag
from functools import cmp_to_key
from typing import Any

from .sleigh import SleighInstruction, SleighPcodeOp
from .ssa import SSAFunction, SSALocation, _register_family
from .typesys import StorageLocation

_PARAM_NAME = re.compile(r"(?:float_)?param_\d+\Z")
PARAMID_MAX_DEPTH = 10


class ParamRank(IntEnum):
    """Numeric ranks shared by ParamMeasure's input and output directions."""

    DIRECT_WRITE_WITHOUT_READ = 1
    DIRECT_READ = 2
    DIRECT_WRITE_WITH_READ = 2
    DIRECT_WRITE_UNKNOWN_READ = 3
    SUBFUNCTION_PARAMETER = 4
    THIS_FUNCTION_PARAMETER = 4
    FUNCTION_RETURN = 5
    SUBFUNCTION_RETURN = 5
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


class ParamDirection(str, Enum):
    """Whether a ParamMeasure walks uses forward or definitions backward."""

    INPUT = "input"
    OUTPUT = "output"


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
    loop_inputs: tuple[bool, ...] = ()


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
    """Directional ParamID rank and bounded use counts for one storage."""

    name: str
    rank: ParamRank
    direct_reads: int = 0
    call_uses: int = 0
    return_uses: int = 0
    indirect_uses: int = 0
    traversed_nodes: int = 0
    depth_limited: bool = False
    direction: ParamDirection = ParamDirection.INPUT

    @property
    def active(self) -> bool:
        return self.rank != ParamRank.UNREFERENCED

    @property
    def numcalls(self) -> int:
        """Expose Ghidra ParamMeasure's ``numcalls`` under its source name."""
        return self.call_uses

    @property
    def rank_label(self) -> str:
        if self.direction == ParamDirection.INPUT:
            return self.rank.label
        return {
            ParamRank.DIRECT_WRITE_WITHOUT_READ: "direct-write-without-read",
            ParamRank.DIRECT_READ: "direct-write-with-read",
            ParamRank.DIRECT_WRITE_UNKNOWN_READ: "direct-write-unknown-read",
            ParamRank.SUBFUNCTION_PARAMETER: "this-function-parameter",
            ParamRank.FUNCTION_RETURN: "subfunction-return",
            ParamRank.INDIRECT: "indirect",
            ParamRank.UNREFERENCED: "unreferenced",
        }[self.rank]

    @property
    def evidence(self) -> tuple[str, ...]:
        result = [
            f"paramid-{self.direction.value}-rank:{self.rank_label}",
            f"paramid-rank:{self.rank_label}",
        ]
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


class ParamTrialFlag(IntFlag):
    """Bit flags used by Ghidra's ``ParamTrial`` storage-range model."""

    CHECKED = 0x001
    USED = 0x002
    DEFINITELY_NOT_USED = 0x004
    ACTIVE = 0x008
    UNREFERENCED = 0x010
    KILLED_BY_CALL = 0x020
    REM_FORMED = 0x040
    INDIRECT_CREATED = 0x080
    CONDITIONAL_EXECUTION = 0x100
    ANCESTOR_REALISTIC = 0x200
    ANCESTOR_SOLID = 0x400


@dataclass(frozen=True, slots=True)
class ParamTrialEntry:
    """The ordering properties a matched Ghidra ``ParamEntry`` supplies.

    The actual ParamList/ParamEntry mapping is outside this port. ``order`` is
    a caller-supplied stand-in for the source object's pointer ordering when
    entries share a group; object identity breaks unspecified ties. Optional
    alignment/base/slot metadata also models
    ``ParamEntry.getSlot`` for a contiguous storage range.
    """

    group: int
    order: int = 0
    exclusion: bool = False
    reverse_stack: bool = False
    alignment: int | None = None
    address_base: int | None = None
    num_slots: int = 1

    def __post_init__(self) -> None:
        if self.alignment is not None and self.alignment < 0:
            raise ValueError("La alineación de ParamTrialEntry no puede ser negativa")
        if self.num_slots < 1:
            raise ValueError("ParamTrialEntry necesita al menos un slot")

    @property
    def is_exclusion(self) -> bool:
        """Whether the model entry denotes a single-parameter exclusion."""
        return self.exclusion or self.alignment == 0

    def get_slot(self, address_offset: int, size_bytes: int) -> int:
        """Resolve the group slot of a byte range, mirroring ``getSlot``."""
        if size_bytes <= 0:
            raise ValueError("El rango de ParamTrialEntry debe tener tamaño positivo")
        if self.is_exclusion:
            return self.group + (self.num_slots - 1 if size_bytes > 1 else 0)
        if self.alignment is None:
            return self.group
        if self.address_base is None:
            raise ValueError("Falta address_base para calcular el slot alineado")
        base_slot = (address_offset + size_bytes - 1 - self.address_base) // self.alignment
        if self.reverse_stack:
            base_slot = self.num_slots - 1 - base_slot
        return self.group + base_slot


@dataclass(slots=True)
class ParamTrial:
    """A tentative storage range and its ParamID recovery flags."""

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
    unreferenced: bool = False
    killed_by_call: bool = False
    rem_formed: bool = False
    indirect_created: bool = False
    conditional_execution: bool = False
    ancestor_realistic: bool = False
    ancestor_solid: bool = False
    fixed_position: int | None = None
    entry: ParamTrialEntry | None = None
    offset: int = -1
    big_endian: bool = False

    @property
    def size_bytes(self) -> int:
        """Storage width in bytes (trials are byte-addressed in Ghidra)."""
        return (max(0, self.size_bits) + 7) // 8

    @property
    def flags(self) -> ParamTrialFlag:
        """Return the source-compatible flag mask; ``selected`` is Python-only."""
        result = ParamTrialFlag(0)
        for flag, field_name in _PARAM_TRIAL_FLAG_FIELDS:
            if getattr(self, field_name):
                result |= flag
        return result

    @flags.setter
    def flags(self, value: ParamTrialFlag | int) -> None:
        flags = ParamTrialFlag(value)
        for flag, field_name in _PARAM_TRIAL_FLAG_FIELDS:
            setattr(self, field_name, bool(flags & flag))

    def mark_active(self) -> None:
        self.checked = True
        self.active = True

    def mark_inactive(self) -> None:
        self.checked = True
        self.active = False

    def mark_selected(self) -> None:
        """Select a Python candidate without implying Ghidra's final verdict."""
        self.selected = True

    def mark_unselected(self) -> None:
        self.selected = False

    def mark_used(self) -> None:
        self.used = True

    def mark_no_use(self) -> None:
        self.checked = True
        self.active = False
        self.used = False
        self.definitely_not_used = True

    def mark_unreferenced(self) -> None:
        self.unreferenced = True
        self.checked = True
        self.slot = -1

    def mark_killed_by_call(self) -> None:
        self.killed_by_call = True

    def mark_rem_formed(self) -> None:
        self.rem_formed = True

    def set_rem_formed(self) -> None:
        self.mark_rem_formed()

    def mark_indirect_created(self) -> None:
        self.indirect_created = True

    def set_indirect_created(self) -> None:
        self.mark_indirect_created()

    def mark_conditional_execution(self) -> None:
        self.conditional_execution = True

    def set_conditional_execution(self) -> None:
        self.mark_conditional_execution()

    def mark_ancestor_realistic(self) -> None:
        self.ancestor_realistic = True

    def set_ancestor_realistic(self) -> None:
        self.mark_ancestor_realistic()

    def mark_ancestor_solid(self) -> None:
        self.ancestor_solid = True

    def set_ancestor_solid(self) -> None:
        self.mark_ancestor_solid()

    def get_size(self) -> int:
        """Return the storage width in bytes, like ``ParamTrial.getSize``."""
        return self.size_bytes

    def get_slot(self) -> int:
        return self.slot

    def set_slot(self, slot: int) -> None:
        self.slot = slot

    def get_entry(self) -> ParamTrialEntry | None:
        return self.entry

    def get_offset(self) -> int:
        return self.offset

    def get_address(self) -> tuple[StorageLocation, ...]:
        """Return the physical pieces representing this logical trial range."""
        return self.locations

    def set_entry(self, entry: ParamTrialEntry | None, offset: int = -1) -> None:
        self.entry = entry
        self.offset = offset if entry is not None else -1

    def slot_group(self) -> int:
        if self.entry is None:
            raise ValueError("El trial todavía no está asociado a un ParamEntry")
        if self.entry.alignment in {None, 0}:
            return self.entry.get_slot(0, self.size_bytes)
        if len(self.locations) != 1:
            raise ValueError(
                "ParamTrialEntry.getSlot requiere una pieza contigua de storage"
            )
        location = self.locations[0]
        if location.kind == "stack" and location.stack_offset is not None:
            address_offset = location.stack_offset
        elif location.kind == "register" and location.register is not None:
            address_offset = (
                location.register_offset + _register_alias_offset(location.register)
            )
        else:
            raise ValueError("El trial no tiene una dirección para resolver su slot")
        return self.entry.get_slot(address_offset, self.size_bytes)

    def set_fixed_position(self, position: int | None) -> None:
        self.fixed_position = position

    def is_checked(self) -> bool:
        return self.checked

    def is_active(self) -> bool:
        return self.active

    def is_selected(self) -> bool:
        return self.selected

    def is_used(self) -> bool:
        return self.used

    def is_definitely_not_used(self) -> bool:
        return self.definitely_not_used

    def is_unreferenced(self) -> bool:
        return self.unreferenced

    def is_killed_by_call(self) -> bool:
        return self.killed_by_call

    def is_rem_formed(self) -> bool:
        return self.rem_formed

    def is_indirect_created(self) -> bool:
        return self.indirect_created

    def has_conditional_execution(self) -> bool:
        return self.conditional_execution

    def has_ancestor_realistic(self) -> bool:
        return self.ancestor_realistic

    def has_ancestor_solid(self) -> bool:
        return self.ancestor_solid

    def set_address(
        self,
        locations: StorageLocation | Iterable[StorageLocation],
        size_bits: int,
    ) -> None:
        self.locations = _coerce_storage_locations(locations)
        _require_byte_size(size_bits)
        self.size_bits = size_bits

    def test_shrink(
        self,
        new_locations: StorageLocation | Iterable[StorageLocation],
        size_bits: int | None = None,
    ) -> bool:
        """Test Ghidra's endian-aware shrink rule for an unbound trial."""
        if self.entry is not None:
            return False
        locations = _coerce_storage_locations(new_locations)
        if not locations:
            return False
        if size_bits is None:
            size_bits = _storage_extent_bits(locations)
        try:
            _require_byte_size(size_bits)
        except ValueError:
            return False
        new_size = size_bits // 8
        old_size = self.size_bytes
        if new_size <= 0 or new_size > old_size:
            return False
        start = old_size - new_size if self.big_endian else 0
        try:
            expected = _slice_storage_locations(
                self.locations, start, start + new_size, old_size
            )
        except ValueError:
            return False
        return _physical_location_signature(expected) == _physical_location_signature(
            locations
        )

    def shrink(
        self,
        new_locations: StorageLocation | Iterable[StorageLocation],
        size_bits: int | None = None,
    ) -> None:
        if not self.test_shrink(new_locations, size_bits):
            raise ValueError("El nuevo rango no es un shrink válido de ParamTrial")
        locations = _coerce_storage_locations(new_locations)
        if size_bits is None:
            size_bits = _storage_extent_bits(locations)
        self.set_address(locations, size_bits)

    def split_hi(self, size_bits: int) -> ParamTrial:
        """Copy the low-address/first bytes into the original slot."""
        _require_split_size(size_bits, self.size_bits)
        if not self.locations:
            raise ValueError("No se puede dividir un trial sin storage asociado")
        size_bytes = size_bits // 8
        locations = _slice_storage_locations(
            self.locations, 0, size_bytes, self.size_bytes
        )
        return self._split_copy(locations, size_bits, self.slot)

    def split_lo(self, size_bits: int) -> ParamTrial:
        """Copy the high-address/last bytes into the following slot."""
        _require_split_size(size_bits, self.size_bits)
        if not self.locations:
            raise ValueError("No se puede dividir un trial sin storage asociado")
        size_bytes = size_bits // 8
        start = self.size_bytes - size_bytes
        locations = _slice_storage_locations(
            self.locations, start, self.size_bytes, self.size_bytes
        )
        return self._split_copy(locations, size_bits, self.slot + 1)

    def _split_copy(
        self,
        locations: tuple[StorageLocation, ...],
        size_bits: int,
        slot: int,
    ) -> ParamTrial:
        return replace(
            self,
            slot=slot,
            size_bits=size_bits,
            locations=locations,
            fixed_position=None,
            entry=None,
            offset=-1,
        )

    def fixed_position_before(self, other: ParamTrial) -> bool:
        return self.fixed_position_compare(self, other)

    @staticmethod
    def fixed_position_compare(left: ParamTrial, right: ParamTrial) -> bool:
        if left.fixed_position is None and right.fixed_position is None:
            return left < right
        if left.fixed_position is None:
            return False
        if right.fixed_position is None:
            return True
        return left.fixed_position < right.fixed_position

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ParamTrial):
            return NotImplemented
        if self.entry is None:
            return False
        if other.entry is None:
            return True
        return _param_trial_order_key(self) < _param_trial_order_key(other)


_PARAM_TRIAL_FLAG_FIELDS = (
    (ParamTrialFlag.CHECKED, "checked"),
    (ParamTrialFlag.USED, "used"),
    (ParamTrialFlag.DEFINITELY_NOT_USED, "definitely_not_used"),
    (ParamTrialFlag.ACTIVE, "active"),
    (ParamTrialFlag.UNREFERENCED, "unreferenced"),
    (ParamTrialFlag.KILLED_BY_CALL, "killed_by_call"),
    (ParamTrialFlag.REM_FORMED, "rem_formed"),
    (ParamTrialFlag.INDIRECT_CREATED, "indirect_created"),
    (ParamTrialFlag.CONDITIONAL_EXECUTION, "conditional_execution"),
    (ParamTrialFlag.ANCESTOR_REALISTIC, "ancestor_realistic"),
    (ParamTrialFlag.ANCESTOR_SOLID, "ancestor_solid"),
)


def _normalize_trial_size(size_bits: int) -> int:
    if size_bits <= 0:
        raise ValueError("El tamaño de un ParamTrial debe ser positivo")
    return ((size_bits + 7) // 8) * 8


def _require_byte_size(size_bits: int) -> None:
    if size_bits <= 0 or size_bits % 8:
        raise ValueError("El tamaño de un ParamTrial debe ser positivo y múltiplo de 8")


def _require_split_size(size_bits: int, total_bits: int) -> None:
    _require_byte_size(size_bits)
    if size_bits >= total_bits or size_bits % 8 or total_bits % 8:
        raise ValueError("El split debe dejar dos rangos de bytes no vacíos")


def _coerce_storage_locations(
    locations: StorageLocation | Iterable[StorageLocation],
) -> tuple[StorageLocation, ...]:
    if isinstance(locations, StorageLocation):
        return (locations,)
    return tuple(locations)


def _register_alias_offset(register: str | None) -> int:
    return 1 if (register or "").lower() in {"ah", "bh", "ch", "dh"} else 0


def _logical_location_offsets(
    locations: tuple[StorageLocation, ...],
) -> tuple[tuple[int, int], ...]:
    if not locations:
        return ()
    offsets = [location.value_offset for location in locations]
    if len(locations) > 1 and all(offset == 0 for offset in offsets):
        # Same-offset pieces are parallel representations (e.g. Windows x64
        # varargs copied into both XMM and GPR storage), not adjacent bytes.
        return tuple((0, location.size) for location in locations)
    origin = min(offsets)
    return tuple(
        (location.value_offset - origin, location.value_offset - origin + location.size)
        for location in locations
    )


def _storage_extent_bits(locations: tuple[StorageLocation, ...]) -> int:
    ranges = _logical_location_offsets(locations)
    if not ranges:
        return 0
    return (max(end for _start, end in ranges) - min(start for start, _end in ranges)) * 8


def _slice_storage_locations(
    locations: tuple[StorageLocation, ...],
    start: int,
    end: int,
    total_size: int,
) -> tuple[StorageLocation, ...]:
    if start < 0 or end <= start or end > total_size:
        raise ValueError("Rango ParamTrial fuera de límites")
    ranges = _logical_location_offsets(locations)
    sliced: list[StorageLocation] = []
    for location, (logical_start, logical_end) in zip(locations, ranges):
        piece_start = max(start, logical_start)
        piece_end = min(end, logical_end)
        if piece_start >= piece_end:
            continue
        delta = piece_start - logical_start
        size = piece_end - piece_start
        value_offset = piece_start - start
        if location.kind == "stack" and location.stack_offset is not None:
            sliced.append(
                replace(
                    location,
                    size=size,
                    stack_offset=location.stack_offset + delta,
                    value_offset=value_offset,
                )
            )
        elif location.kind == "register" and location.register is not None:
            sliced.append(
                replace(
                    location,
                    size=size,
                    register=_register_family(location.register.lower()),
                    register_offset=(
                        location.register_offset
                        + _register_alias_offset(location.register)
                        + delta
                    ),
                    value_offset=value_offset,
                )
            )
        elif delta == 0:
            sliced.append(replace(location, size=size, value_offset=value_offset))
        else:
            raise ValueError("No se puede calcular el subrango de esta ubicación")
    if not sliced:
        raise ValueError("El subrango no contiene storage")
    return tuple(
        sorted(
            sliced,
            key=lambda item: (
                item.value_offset,
                _storage_location_order_key(item),
            ),
        )
    )


def _physical_location_signature(
    locations: tuple[StorageLocation, ...],
) -> tuple[tuple[object, ...], ...]:
    result = []
    for location in locations:
        if location.kind == "stack" and location.stack_offset is not None:
            result.append(
                (
                    "stack",
                    location.stack_offset,
                    location.stack_offset + location.size,
                    location.indirect,
                    location.hidden,
                )
            )
        elif location.kind == "register" and location.register is not None:
            start = location.register_offset + _register_alias_offset(location.register)
            result.append(
                (
                    "register",
                    _register_family(location.register.lower()),
                    start,
                    start + location.size,
                    location.indirect,
                    location.hidden,
                )
            )
        else:
            result.append(
                (location.kind, location.register, location.stack_offset, location.size)
            )
    return tuple(sorted(result))


def _storage_location_order_key(
    location: StorageLocation, reverse_stack: bool = False
) -> tuple[object, ...]:
    if location.kind == "stack" and location.stack_offset is not None:
        offset = -location.stack_offset if reverse_stack else location.stack_offset
        return (1, "stack", offset)
    if location.kind == "register" and location.register is not None:
        return (
            0,
            _register_family(location.register.lower()),
            location.register_offset + _register_alias_offset(location.register),
        )
    return (
        2,
        location.kind,
        location.register or "",
        location.stack_offset if location.stack_offset is not None else 0,
    )


def _param_trial_order_key(trial: ParamTrial) -> tuple[object, ...]:
    entry = trial.entry
    if entry is None:
        return (1,)
    if entry.is_exclusion:
        address_key: tuple[object, ...] = (trial.offset,)
    else:
        address_key = tuple(
            _storage_location_order_key(location, entry.reverse_stack)
            for location in sorted(
                trial.locations,
                key=lambda item: (item.value_offset, _storage_location_order_key(item)),
            )
        ) or ((2, trial.name),)
    return (0, entry.group, entry.order, id(entry), address_key, trial.size_bits)


class ParamActive:
    """Trial-range container ported from Ghidra's ``ParamActive`` mechanics.

    The range/state operations mirror ``fspec.cc``. Prototype-model matching is
    represented by :class:`ParamTrialEntry`, not by the unported ParamList and
    ParamModel hierarchy. ``selected`` and name indexing remain Python-side
    conveniences for binding a recovered signature to trials.
    """

    def __init__(self, recover_subcalls: bool = False) -> None:
        self.recover_subcalls = recover_subcalls
        self._trials: list[ParamTrial] = []
        self._by_name: dict[str, ParamTrial] = {}
        self._slot_base = 1
        self._placeholder_slot = -1
        self._passes = 0
        self._max_pass = 0
        self._fully_checked = False
        self._needs_final_check = False
        self._join_most_sig_first = False

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

    @property
    def num_passes(self) -> int:
        return self._passes

    @property
    def max_pass(self) -> int:
        return self._max_pass

    @property
    def fully_checked(self) -> bool:
        return self._fully_checked

    @property
    def needs_final_check(self) -> bool:
        return self._needs_final_check

    @property
    def join_most_sig_first(self) -> bool:
        return self._join_most_sig_first

    @property
    def recover_subcall(self) -> bool:
        return self.recover_subcalls

    @property
    def placeholder_slot(self) -> int:
        return self._placeholder_slot

    @property
    def slot_base(self) -> int:
        return self._slot_base

    def register_trial(
        self,
        name: str,
        size_bits: int,
        locations: Iterable[StorageLocation] = (),
        *,
        merge_existing: bool = False,
    ) -> ParamTrial:
        """Register a storage range; use ``merge_existing`` for signature binding.

        Ghidra appends every range. ParamID's higher-level name binding often
        needs to enrich the already measured candidate instead, so that
        behavior is explicit and opt-in here.
        """
        size_bits = _normalize_trial_size(size_bits)
        supplied_locations = tuple(locations)
        found = self._by_name.get(name) if merge_existing else None
        if found is not None:
            if supplied_locations:
                found.size_bits = _storage_extent_bits(supplied_locations) or size_bits
                found.locations = supplied_locations
                if any(location.kind != "stack" for location in supplied_locations):
                    found.mark_killed_by_call()
            else:
                found.size_bits = max(found.size_bits, size_bits)
            return found
        trial = ParamTrial(
            name=name,
            slot=self._slot_base,
            size_bits=size_bits,
            locations=supplied_locations,
        )
        if any(location.kind != "stack" for location in supplied_locations):
            trial.mark_killed_by_call()
        self._slot_base += 1
        self._trials.append(trial)
        self._reindex_names()
        return trial

    def clear(self) -> None:
        """Clear trials while preserving max-pass and final-check controls."""
        self._trials.clear()
        self._by_name.clear()
        self._slot_base = 1
        self._placeholder_slot = -1
        self._passes = 0
        self._fully_checked = False
        self._join_most_sig_first = False

    def reregister_trial(self, old_trial: ParamTrial) -> ParamTrial:
        """Copy flags/range but clear model match and assign the next slot."""
        trial = replace(
            old_trial,
            slot=self._slot_base,
            entry=None,
            offset=-1,
            fixed_position=None,
        )
        self._slot_base += 1
        self._trials.append(trial)
        self._reindex_names()
        return trial

    def set_max_pass(self, value: int) -> None:
        self._max_pass = value

    def finish_pass(self) -> None:
        self._passes += 1

    def mark_fully_checked(self) -> None:
        self._fully_checked = True

    def mark_needs_final_check(self) -> None:
        self._needs_final_check = True

    def set_join_most_sig_first(self) -> None:
        self._join_most_sig_first = True

    def set_placeholder_slot(self) -> int:
        self._placeholder_slot = self._slot_base
        self._slot_base += 1
        return self._placeholder_slot

    def free_placeholder_slot(self) -> None:
        if self._placeholder_slot < 0:
            return
        for trial in self._trials:
            if trial.slot > self._placeholder_slot:
                trial.slot -= 1
        self._placeholder_slot = -2
        self._slot_base -= 1
        self._max_pass = 0

    def which_trial(
        self,
        locations: StorageLocation | Iterable[StorageLocation],
        size_bits: int | None = None,
    ) -> int:
        """Return the first trial whose storage overlaps the supplied range."""
        query = _coerce_storage_locations(locations)
        if size_bits is not None:
            _require_byte_size(size_bits)
            if len(query) == 1:
                query = (replace(query[0], size=size_bits // 8),)
            elif _storage_extent_bits(query) != size_bits:
                raise ValueError("El tamaño no coincide con las piezas de storage")
        for index, trial in enumerate(self._trials):
            if any(
                _storage_locations_overlap(candidate, existing)
                for candidate in query
                for existing in trial.locations
            ):
                return index
        return -1

    def get_trial_for_input_varnode(self, slot: int) -> ParamTrial:
        """Map a CALL input slot to its trial index, accounting for target/gap."""
        index = slot - (
            1 if self._placeholder_slot < 0 or slot < self._placeholder_slot else 2
        )
        if index < 0 or index >= len(self._trials):
            raise KeyError(f"No hay ParamTrial para el slot de entrada {slot}")
        return self._trials[index]

    def free_placeholder_slot_if_set(self) -> None:
        if self._placeholder_slot >= 0:
            self.free_placeholder_slot()

    def split_trial(self, index: int, size_bits: int) -> tuple[ParamTrial, ParamTrial]:
        if self._placeholder_slot >= 0:
            raise RuntimeError(
                "No se puede dividir un trial mientras no se resuelva el placeholder"
            )
        if index < 0 or index >= len(self._trials):
            raise IndexError("Índice de ParamTrial fuera de límites")
        original = self._trials[index]
        _require_split_size(size_bits, original.size_bits)
        slot = original.slot
        high_piece = original.split_hi(size_bits)
        low_piece = original.split_lo(original.size_bits - size_bits)
        updated: list[ParamTrial] = []
        for trial in self._trials[:index]:
            if trial.slot > slot:
                trial.slot += 1
            updated.append(trial)
        updated.extend((high_piece, low_piece))
        for trial in self._trials[index + 1 :]:
            if trial.slot > slot:
                trial.slot += 1
            updated.append(trial)
        self._trials = updated
        self._slot_base += 1
        self._reindex_names()
        return high_piece, low_piece

    def join_trial(
        self,
        slot: int,
        locations: StorageLocation | Iterable[StorageLocation] | None = None,
        size_bits: int | None = None,
        *,
        name: str | None = None,
    ) -> ParamTrial:
        """Join adjacent trial slots, requiring their exact combined size."""
        if self._placeholder_slot >= 0:
            raise RuntimeError(
                "No se pueden unir trials mientras no se resuelva el placeholder"
            )
        left = [trial for trial in self._trials if trial.slot == slot]
        right = [trial for trial in self._trials if trial.slot == slot + 1]
        if len(left) != 1 or len(right) != 1:
            raise ValueError("join_trial requiere un trial en cada slot adyacente")
        first, second = left[0], right[0]
        combined_size = first.size_bits + second.size_bits
        if size_bits is None:
            size_bits = combined_size
        _require_byte_size(size_bits)
        if combined_size != size_bits:
            raise ValueError("Size mismatch when joining parameters")
        if locations is None:
            first_locations = _rebase_location_offsets(first.locations, 0)
            second_locations = _rebase_location_offsets(
                second.locations, first.size_bytes
            )
            joined_locations = _merge_adjacent_locations(
                first_locations + second_locations
            )
        else:
            joined_locations = _coerce_storage_locations(locations)
        if joined_locations and _storage_extent_bits(joined_locations) != size_bits:
            raise ValueError("El storage unido no coincide con el tamaño solicitado")
        joined = ParamTrial(
            name=name or first.name,
            slot=slot,
            size_bits=size_bits,
            locations=joined_locations,
            rank=min(first.rank, second.rank),
            evidence=tuple(dict.fromkeys(first.evidence + second.evidence)),
        )
        joined.mark_used()
        joined.mark_active()
        updated: list[ParamTrial] = []
        emitted_join = False
        for trial in self._trials:
            if trial.slot < slot:
                updated.append(trial)
            elif trial.slot == slot:
                if not emitted_join:
                    updated.append(joined)
                    emitted_join = True
            elif trial.slot == slot + 1:
                continue
            else:
                trial.slot -= 1
                updated.append(trial)
        if not emitted_join:
            raise ValueError(f"No existe el slot {slot}")
        self._trials = updated
        self._slot_base -= 1
        self._reindex_names()
        return joined

    def test_shrink(
        self,
        index: int,
        locations: StorageLocation | Iterable[StorageLocation],
        size_bits: int | None = None,
    ) -> bool:
        return self._trials[index].test_shrink(locations, size_bits)

    def shrink(
        self,
        index: int,
        locations: StorageLocation | Iterable[StorageLocation],
        size_bits: int,
    ) -> None:
        """Apply a previously checked range shrink, like ``ParamActive.shrink``."""
        self._trials[index].set_address(locations, size_bits)

    def finalize_selected_trials(self) -> tuple[ParamTrial, ...]:
        """Set the final used bit on active selected candidates with storage."""
        finalized: list[ParamTrial] = []
        for trial in self._trials:
            if trial.selected and trial.active and trial.locations:
                trial.mark_used()
                finalized.append(trial)
        return tuple(finalized)

    def get_trial(self, name: str) -> ParamTrial | None:
        return self._by_name.get(name)

    def get_trials(self, name: str) -> tuple[ParamTrial, ...]:
        return tuple(trial for trial in self._trials if trial.name == name)

    def get_trial_at(self, index: int) -> ParamTrial:
        """Return the trial at a current list index, like ``getTrial``."""
        return self._trials[index]

    def get_num_trials(self) -> int:
        return len(self._trials)

    def get_num_passes(self) -> int:
        return self._passes

    def get_max_pass(self) -> int:
        return self._max_pass

    def is_fully_checked(self) -> bool:
        return self._fully_checked

    def is_recover_subcall(self) -> bool:
        return self.recover_subcalls

    def get_num_used(self) -> int:
        """Return the leading used prefix; callers must first sort trials."""
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

    def sort_trials(self) -> None:
        """Sort by model group/address without rewriting source trial slots."""
        self._trials.sort(key=cmp_to_key(_compare_param_trials))
        self._reindex_names()

    def sort_by_position(self) -> None:
        """Python signature convenience: order by ABI names and renumber slots."""
        self._trials.sort(key=lambda trial: _parameter_order(trial.name))
        for slot, trial in enumerate(self._trials, start=1):
            trial.slot = slot
        self._slot_base = len(self._trials) + 1
        self._reindex_names()

    def sort_fixed_position(self) -> None:
        """Sort by fixed position, then use ParamTrial's formal-order comparator."""
        self._trials.sort(key=cmp_to_key(_compare_fixed_position))
        self._reindex_names()

    def sort_by_fixed_position(self) -> None:
        """Python-friendly alias for :meth:`sort_fixed_position`."""
        self.sort_fixed_position()

    def delete_unused_trials(self) -> None:
        """Delete unused ranges and renumber the remaining input-varnode slots."""
        self._trials = [trial for trial in self._trials if trial.used]
        for slot, trial in enumerate(self._trials, start=1):
            trial.slot = slot
        self._reindex_names()

    def _reindex_names(self) -> None:
        self._by_name = {trial.name: trial for trial in self._trials}


def _compare_param_trials(left: ParamTrial, right: ParamTrial) -> int:
    if left < right:
        return -1
    if right < left:
        return 1
    return 0


def _compare_fixed_position(left: ParamTrial, right: ParamTrial) -> int:
    if ParamTrial.fixed_position_compare(left, right):
        return -1
    if ParamTrial.fixed_position_compare(right, left):
        return 1
    return 0


def _storage_locations_overlap(
    left: StorageLocation, right: StorageLocation
) -> bool:
    def interval(location: StorageLocation) -> tuple[tuple[object, ...], int, int]:
        if location.kind == "stack" and location.stack_offset is not None:
            start = location.stack_offset
            return ("stack",), start, start + location.size
        if location.kind == "register" and location.register is not None:
            start = (
                location.register_offset
                + _register_alias_offset(location.register)
            )
            return (
                ("register", _register_family(location.register.lower())),
                start,
                start + location.size,
            )
        return (
            (location.kind, location.register or ""),
            0,
            location.size,
        )

    left_space, left_start, left_end = interval(left)
    right_space, right_start, right_end = interval(right)
    return (
        left_space == right_space
        and max(left_start, right_start) < min(left_end, right_end)
    )


def _rebase_location_offsets(
    locations: tuple[StorageLocation, ...], base_offset: int
) -> tuple[StorageLocation, ...]:
    ranges = _logical_location_offsets(locations)
    return tuple(
        replace(location, value_offset=base_offset + start)
        for location, (start, _end) in zip(locations, ranges)
    )


def _merge_adjacent_locations(
    locations: tuple[StorageLocation, ...],
) -> tuple[StorageLocation, ...]:
    ordered = sorted(
        locations,
        key=lambda item: (item.value_offset, _storage_location_order_key(item)),
    )
    merged: list[StorageLocation] = []
    for location in ordered:
        if merged:
            previous = merged[-1]
            same_attributes = (
                previous.kind == location.kind
                and previous.indirect == location.indirect
                and previous.hidden == location.hidden
                and previous.value_offset + previous.size == location.value_offset
            )
            if same_attributes and previous.kind == "stack":
                adjacent = (
                    previous.stack_offset is not None
                    and location.stack_offset == previous.stack_offset + previous.size
                )
            elif same_attributes and previous.kind == "register":
                adjacent = (
                    previous.register is not None
                    and location.register is not None
                    and _register_family(previous.register.lower())
                    == _register_family(location.register.lower())
                    and location.register_offset + _register_alias_offset(location.register)
                    == previous.register_offset
                    + _register_alias_offset(previous.register)
                    + previous.size
                )
            else:
                adjacent = False
            if adjacent:
                merged[-1] = replace(previous, size=previous.size + location.size)
                continue
        merged.append(location)
    return tuple(merged)


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
        self._output_values: dict[str, set[int]] = defaultdict(set)
        self._output_sink_operations: dict[str, set[int]] = defaultdict(set)
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
        stack_parameter_names: Mapping[int, str] | None = None,
    ) -> PcodeLikeGraph:
        """Adapt the complete reachable SLEIGH SSA graph for ParamID walks.

        Every p-code operation, phi input, output slice, and ABI return sink
        remains linked to the original SSA value ids. ``return_registers`` can
        restrict candidate sinks to families selected by the inferred return
        type. ``stack_parameter_names`` binds declared stack formals to loads
        identified by the compatibility graph; the supplement also supplies
        outgoing stack-argument and call-spec evidence not represented by SSA.
        """
        graph = cls()
        graph.ssa_function = ssa_function
        graph.supplemental_graph = supplemental_graph
        graph._source_widths = {
            name: max(1, width) for name, width in (parameter_widths or {}).items()
        }
        values_by_id = ssa_function.value_by_id
        stack_input_by_value: dict[int, set[str]] = defaultdict(set)
        if supplemental_graph is not None:
            legacy_source_names = {
                node_id: name
                for name, node_ids in supplemental_graph._source_nodes.items()
                for node_id in node_ids
            }
            stack_sources_by_address: dict[int, set[str]] = defaultdict(set)
            for legacy_operation in supplemental_graph.ops:
                if (
                    legacy_operation.address is None
                    or legacy_operation.opcode not in {"LOAD_STACK", "LOAD"}
                    or not legacy_operation.evidence.startswith("memory-read:stack[")
                ):
                    continue
                source_names = {
                    legacy_source_names[node_id]
                    for node_id in legacy_operation.inputs
                    if node_id in legacy_source_names
                }
                offset_match = re.search(
                    r"memory-read:stack\[([+-])0x([0-9A-Fa-f]+)\]",
                    legacy_operation.evidence,
                )
                if offset_match is not None and stack_parameter_names is not None:
                    offset = int(offset_match.group(2), 16)
                    if offset_match.group(1) == "-":
                        offset = -offset
                    declared_name = stack_parameter_names.get(offset)
                    if declared_name is not None:
                        source_names.add(declared_name)
                stack_sources_by_address[legacy_operation.address].update(source_names)
            ssa_load_count_by_address: dict[int, int] = defaultdict(int)
            for operation in ssa_function.operations:
                if operation.opcode == "LOAD":
                    ssa_load_count_by_address[operation.instruction_address] += 1
            for operation in ssa_function.operations:
                if (
                    operation.opcode != "LOAD"
                    or operation.output_value is None
                    or ssa_load_count_by_address[operation.instruction_address] != 1
                ):
                    continue
                names = stack_sources_by_address.get(
                    operation.instruction_address, ()
                )
                stack_input_by_value[operation.output_value].update(names)
                for name in names:
                    graph._source_widths[name] = max(
                        graph._source_widths.get(name, 1),
                        _source_width(supplemental_graph, name),
                    )

        for value in ssa_function.values:
            register_name = (
                parameter_locations.get(value.storage)
                if value.kind == "input" and value.storage is not None
                else None
            )
            source_names = (
                (register_name,)
                if register_name is not None
                else tuple(
                    sorted(
                        stack_input_by_value.get(value.identifier, ()),
                        key=_parameter_order,
                    )
                )
            )
            size_bits = value.size_bits
            for name in source_names:
                size_bits = max(size_bits, graph._source_widths.get(name, 1))
                graph._source_widths[name] = max(
                    graph._source_widths.get(name, 1), size_bits
                )
                graph._source_nodes[name].add(value.identifier)
            graph.nodes.append(
                PcodeVarnode(
                    value.identifier,
                    source_names[0] if source_names else value.name,
                    size_bits,
                    "input" if source_names else value.kind,
                )
            )

        for item in ssa_function.blocks:
            if item.reachable:
                graph.begin_instruction(item.instruction_address, item.instruction_mnemonic)

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
                        if is_round_userop
                        or operation.opcode in {"RETURN", "IMARK"}
                        or any(rank is not None for rank in input_ranks)
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
                            input_ranks=tuple(
                                ParamRank.SUBFUNCTION_PARAMETER
                                for _value in defined_values
                            ),
                        )
                    )

        blocks = ssa_function.block_by_id
        immediate_dominators = {
            block.identifier: block.immediate_dominator
            for block in ssa_function.blocks
        }

        def is_loop_input(phi_block: int, predecessor: int | None) -> bool:
            current = predecessor
            visited_blocks: set[int] = set()
            while current is not None and current not in visited_blocks:
                if current == phi_block:
                    return True
                visited_blocks.add(current)
                current = immediate_dominators.get(current)
            return False

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
                        loop_inputs=tuple(
                            is_loop_input(value.block_id, predecessor)
                            for predecessor, _incoming in value.incoming
                        ),
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
            else {_register_family(register.lower()) for register in return_registers}
        )
        for site in ssa_function.return_sites:
            for register, operand in site.register_values:
                normalized_register = _register_family(register.lower())
                if (
                    allowed_return_registers is not None
                    and normalized_register not in allowed_return_registers
                ):
                    continue
                return_name = f"return_{normalized_register}"
                return_values = tuple(dict.fromkeys(operand.values))
                if not return_values:
                    continue
                graph._output_values[return_name].update(return_values)
                sink_index = graph._append_operation(
                    PcodeOp(
                        "RETURN",
                        return_values,
                        address=site.instruction_address,
                        terminal_rank=ParamRank.FUNCTION_RETURN,
                        evidence="ssa-abi-return-registers",
                        mnemonic="ret",
                        input_ranks=tuple(
                            ParamRank.FUNCTION_RETURN for _value in return_values
                        ),
                    )
                )
                graph._output_sink_operations[return_name].add(sink_index)
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
    def output_names(self) -> tuple[str, ...]:
        """Return ABI output storage names observed at reachable return sites."""
        return tuple(sorted(self._output_values))

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
                input_ranks=(
                    (ParamRank.DIRECT_READ,) if target_node is not None else ()
                )
                + tuple(
                    ParamRank.SUBFUNCTION_PARAMETER for _argument in argument_nodes
                ),
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
        *,
        loop_inputs: Iterable[bool] = (),
    ) -> None:
        """Add/update a MULTIEQUAL with optional loop-backedge annotations."""
        value_list = tuple(values)
        loop_flags = tuple(loop_inputs)
        if loop_flags and len(loop_flags) != len(value_list):
            raise ValueError("loop_inputs debe tener un flag por cada entrada phi")
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
            PcodeOp(
                "MULTIEQUAL",
                inputs,
                output,
                address=address,
                loop_inputs=loop_flags,
            )
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
                input_ranks=(rank,),
            )
        )

    def measure(self, name: str, max_depth: int = 10) -> ParamMeasurement:
        """Measure an input by Ghidra's forward ``walkforward`` rank rules."""
        starts = self._source_nodes.get(name, set())
        if not starts:
            return ParamMeasurement(name, ParamRank.UNREFERENCED)
        result = self._walk_forward(
            starts,
            max_depth=max_depth,
            best=True,
            ignore_operations=frozenset(),
            stop_rank=None,
        )
        return ParamMeasurement(
            name=name,
            rank=result[0],
            direct_reads=result[1],
            call_uses=result[2],
            return_uses=result[3],
            indirect_uses=result[4],
            traversed_nodes=result[5],
            depth_limited=result[6],
            direction=ParamDirection.INPUT,
        )

    def measure_output(
        self,
        name: str,
        values: Iterable[int] | None = None,
        max_depth: int = 10,
    ) -> ParamMeasurement:
        """Measure a return storage by Ghidra's backward ``walkbackward`` rules."""
        start_values = set(
            self._output_values.get(name, ()) if values is None else values
        )
        if not start_values:
            return ParamMeasurement(
                name, ParamRank.UNREFERENCED, direction=ParamDirection.OUTPUT
            )
        ignore_operations = frozenset(self._output_sink_operations.get(name, ()))
        visited: set[tuple[int, int | None]] = set()
        depth_limited = False

        def walk_back(value_id: int, ignore_operation: int | None) -> ParamRank:
            nonlocal depth_limited
            state = (value_id, ignore_operation)
            if state in visited:
                return ParamRank.UNREFERENCED
            visited.add(state)
            if value_id < 0 or value_id >= len(self.nodes):
                return ParamRank.UNREFERENCED
            value = self.nodes[value_id]
            if value.kind in {"input", "memory_input"}:
                return ParamRank.THIS_FUNCTION_PARAMETER
            definition_index = self._definitions.get(value_id)
            if definition_index is None:
                return ParamRank.THIS_FUNCTION_PARAMETER
            operation = self.ops[definition_index]
            if operation.opcode in {
                "BRANCH",
                "BRANCHIND",
                "CBRANCH",
                "CALL",
                "CALLIND",
                "CALL_RESULT",
            }:
                return ParamRank.UNREFERENCED
            if operation.opcode == "CALLOTHER":
                return ParamRank.DIRECT_READ
            if operation.opcode == "RETURN":
                return ParamRank.FUNCTION_RETURN
            if operation.opcode == "INDIRECT":
                return ParamRank.INDIRECT
            if operation.opcode == "MULTIEQUAL":
                ranks = [
                    walk_back(input_id, definition_index)
                    for input_index, input_id in enumerate(operation.inputs)
                    if input_index >= len(operation.loop_inputs)
                    or not operation.loop_inputs[input_index]
                ]
                return min(ranks, default=ParamRank.UNREFERENCED)

            ignored = {definition_index, *ignore_operations}
            if ignore_operation is not None:
                ignored.add(ignore_operation)
            forward = self._walk_forward(
                (value_id,),
                max_depth=max_depth,
                best=False,
                ignore_operations=frozenset(ignored),
                stop_rank=ParamRank.INDIRECT,
            )
            depth_limited = depth_limited or forward[6]
            return (
                ParamRank.DIRECT_READ
                if forward[0] == ParamRank.DIRECT_READ
                else ParamRank.DIRECT_WRITE_WITHOUT_READ
            )

        ranks = [walk_back(value_id, None) for value_id in sorted(start_values)]
        rank = min(ranks, default=ParamRank.UNREFERENCED)
        forward = self._walk_forward(
            start_values,
            max_depth=max_depth,
            best=False,
            ignore_operations=ignore_operations,
            stop_rank=ParamRank.INDIRECT,
        )
        depth_limited = depth_limited or forward[6]
        return ParamMeasurement(
            name=name,
            rank=rank,
            direct_reads=forward[1],
            call_uses=forward[2],
            return_uses=forward[3],
            indirect_uses=forward[4],
            traversed_nodes=len(visited) + forward[5],
            depth_limited=depth_limited,
            direction=ParamDirection.OUTPUT,
        )

    def _walk_forward(
        self,
        starts: Iterable[int],
        *,
        max_depth: int,
        best: bool,
        ignore_operations: frozenset[int],
        stop_rank: ParamRank | None,
    ) -> tuple[ParamRank, int, int, int, int, int, bool]:
        """Bounded forward walk, following only MULTIEQUAL and SSA aliases."""
        limit = min(PARAMID_MAX_DEPTH, max(1, max_depth))
        users: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for operation_index, operation in enumerate(self.ops):
            for input_index, input_node in enumerate(operation.inputs):
                users[input_node].append((operation_index, input_index))

        queue = deque((node, 0) for node in sorted(set(starts)))
        visited_nodes: set[int] = set()
        visited_edges: set[tuple[int, int, int]] = set()
        best_rank = ParamRank.UNREFERENCED if best else ParamRank.DIRECT_WRITE_WITHOUT_READ
        direct_reads = call_uses = return_uses = indirect_uses = 0
        counted_sites: set[tuple[int, ParamRank]] = set()
        depth_limited = False

        while queue:
            node, depth = queue.popleft()
            if node in visited_nodes:
                continue
            visited_nodes.add(node)
            if depth >= limit:
                if any(
                    operation_index not in ignore_operations
                    for operation_index, _slot in users.get(node, ())
                ):
                    depth_limited = True
                continue

            for operation_index, input_index in users.get(node, ()):
                if operation_index in ignore_operations:
                    continue
                edge = (node, operation_index, input_index)
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)
                operation = self.ops[operation_index]

                if operation.opcode == "MULTIEQUAL":
                    is_loop_input = (
                        input_index < len(operation.loop_inputs)
                        and operation.loop_inputs[input_index]
                    )
                    if (
                        not is_loop_input
                        and operation.output is not None
                        and operation.output not in visited_nodes
                    ):
                        queue.append((operation.output, depth + 1))
                    continue

                if operation.evidence in {
                    "ssa-byte-slice",
                    "ssa-call-stack-restore",
                    "ssa-callother-passthrough",
                }:
                    if operation.output is not None and operation.output not in visited_nodes:
                        queue.append((operation.output, depth + 1))
                    continue

                rank = _operation_input_rank(operation, input_index)
                if rank is None:
                    continue
                site = (operation_index, rank)
                if site not in counted_sites:
                    counted_sites.add(site)
                    if rank == ParamRank.DIRECT_READ:
                        direct_reads += 1
                    elif rank == ParamRank.SUBFUNCTION_PARAMETER:
                        call_uses += 1
                    elif rank == ParamRank.FUNCTION_RETURN:
                        return_uses += 1
                    elif rank == ParamRank.INDIRECT:
                        indirect_uses += 1
                best_rank = min(best_rank, rank) if best else max(best_rank, rank)
                if stop_rank is not None and rank == stop_rank:
                    queue.clear()
                    break

        return (
            best_rank,
            direct_reads,
            call_uses,
            return_uses,
            indirect_uses,
            len(visited_nodes),
            depth_limited,
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

    def __init__(self, graph: PcodeLikeGraph, max_depth: int = PARAMID_MAX_DEPTH) -> None:
        self.graph = graph
        self.supplemental_graph = graph.supplemental_graph
        self.max_depth = min(PARAMID_MAX_DEPTH, max(1, max_depth))
        self.input_active = ParamActive()
        self.measurements: dict[str, ParamMeasurement] = {}
        self.output_measurements: dict[str, ParamMeasurement] = {}
        self.justproto = False

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

    def analyze(
        self,
        *,
        justproto: bool = False,
        input_names: Iterable[str] | None = None,
    ) -> ParamActive:
        """Measure input trials and ABI outputs.

        ``justproto=True`` mirrors Ghidra's diagnostic mode for a recovered
        prototype: pass its formal storage names in ``input_names`` to avoid
        measuring unmodeled live-in locations. With ``justproto=False`` all
        discovered SSA and compatibility-graph inputs are measured.
        """
        if justproto and input_names is None:
            raise ValueError("justproto=True requiere los nombres de los formales")
        self.input_active = ParamActive()
        self.measurements = {}
        self.output_measurements = {}
        self.justproto = justproto
        if input_names is None:
            names = set(self.graph.parameter_names)
            if self.supplemental_graph is not None:
                names.update(self.supplemental_graph.parameter_names)
        else:
            names = set(input_names)
        for name in sorted(names, key=_parameter_order):
            measurement = self._measure(name)
            self.measurements[name] = measurement
            width = _source_width(self.graph, name)
            if self.supplemental_graph is not None:
                width = max(width, _source_width(self.supplemental_graph, name))
            trial = self.input_active.register_trial(name, width)
            self._apply_measurement(trial, measurement)

        for name in self.graph.output_names:
            self.output_measurements[name] = self.graph.measure_output(
                name, max_depth=self.max_depth
            )
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
            storage_bits = _storage_extent_bits(locations)
            data_type = getattr(parameter, "data_type", None)
            size_bytes = (
                storage_bits // 8
                if storage_bits
                else getattr(data_type, "size", None)
            )
            if size_bytes is None:
                size_bytes = 1
            trial = self.input_active.register_trial(
                parameter.name,
                max(1, int(size_bytes) * 8),
                locations,
                merge_existing=True,
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
    """Preserve Ghidra's input-slot exceptions in raw SLEIGH p-code."""
    del has_output  # Ordinary p-code inputs are direct reads with or without output.
    if not values:
        return None
    if opcode in {"BRANCH", "BRANCHIND"}:
        return ParamRank.DIRECT_READ if input_index == 0 else None
    if opcode == "CBRANCH":
        return ParamRank.DIRECT_READ if input_index < 2 else None
    if opcode in {"CALL", "CALLIND"}:
        if input_index == 0:
            return ParamRank.DIRECT_READ
        return ParamRank.SUBFUNCTION_PARAMETER
    if opcode == "CALLOTHER":
        return ParamRank.DIRECT_READ
    if opcode == "INDIRECT":
        return ParamRank.INDIRECT
    if opcode in {"RETURN", "IMARK"}:
        return None
    return None


def _operation_input_rank(
    operation: PcodeOp, input_index: int
) -> ParamRank | None:
    """Classify one ParamMeasure use, honoring explicit adapter annotations."""
    if input_index < len(operation.input_ranks):
        return operation.input_ranks[input_index]
    if operation.terminal_rank is not None:
        return operation.terminal_rank
    opcode = operation.opcode
    if opcode in {"BRANCH", "BRANCHIND"}:
        return ParamRank.DIRECT_READ if input_index == 0 else None
    if opcode == "CBRANCH":
        return ParamRank.DIRECT_READ if input_index < 2 else None
    if opcode in {"CALL", "CALLIND"}:
        return (
            ParamRank.DIRECT_READ
            if input_index == 0
            else ParamRank.SUBFUNCTION_PARAMETER
        )
    if opcode == "CALLOTHER":
        return ParamRank.DIRECT_READ
    if opcode == "RETURN":
        return ParamRank.FUNCTION_RETURN
    if opcode == "INDIRECT":
        return ParamRank.INDIRECT
    if opcode in {"MULTIEQUAL", "IMARK", "CALL_RESULT"}:
        return None
    return ParamRank.DIRECT_READ


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
