"""ParamEntry/ParamList resource models and prototype-resolution passes.

This module ports the storage-range and trial-selection algorithms in
Ghidra's ``fspec.hh/.cc``. It uses Python ``StorageLocation`` and a simplified
``JoinRecord`` instead of Ghidra address spaces, while preserving group, slot,
alignment, justification, exclusion, and reverse-stack semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from xml.etree import ElementTree

from .model import FunctionParameter, FunctionSignature
from .paramid import (
    ParamActive,
    ParamEntry,
    ParamTrial,
    ParamTypeClass,
    _slice_physical_location,
)
from .typesys import (
    CType,
    StorageLocation,
    TypeDatabase,
    TypeKind,
)


@dataclass(frozen=True, slots=True)
class Primitive:
    """One primitive C type and its byte offset in an aggregate."""

    data_type: CType
    offset: int

    @property
    def size(self) -> int:
        return self.data_type.unwrapped().size or 0

    @property
    def align_size(self) -> int:
        return self.data_type.unwrapped().align_size or 0


class PrimitiveExtractor:
    """Port of ``PrimitiveExtractor`` for supported Python ``CType`` layouts.

    It expands structures/arrays, computes the common refinement of union
    alternatives, and records unknown elements, misalignment, and unexplained
    holes. Callers decide which recorded conditions make a particular model
    rule ineligible.
    """

    Primitive = Primitive

    def __init__(
        self,
        data_type: CType,
        *,
        union_illegal: bool = False,
        array_primitive: bool = False,
        offset: int = 0,
        max_primitives: int = 1024,
    ) -> None:
        if offset < 0 or max_primitives < 0:
            raise ValueError("offset y max_primitives deben ser no negativos")
        self.primitives: list[Primitive] = []
        self.contains_unknown = False
        self.aligned = True
        self.holes = False
        self._union_illegal = union_illegal
        self._array_primitive = array_primitive
        self._max_primitives = max_primitives
        self.valid = self._extract(data_type, offset, 0)

    def size(self) -> int:
        return len(self.primitives)

    def get(self, index: int) -> Primitive:
        return self.primitives[index]

    def is_valid(self) -> bool:
        return self.valid

    def is_aligned(self) -> bool:
        return self.aligned

    def contains_holes(self) -> bool:
        return self.holes

    def _append(self, data_type: CType, offset: int) -> bool:
        if len(self.primitives) >= self._max_primitives:
            return False
        self.primitives.append(Primitive(data_type, offset))
        return True

    def _extract(self, data_type: CType, offset: int, depth: int) -> bool:
        base = data_type.unwrapped()
        if base.kind == TypeKind.UNKNOWN:
            self.contains_unknown = True
        if base.kind in {
            TypeKind.UNKNOWN,
            TypeKind.BOOL,
            TypeKind.INTEGER,
            TypeKind.FLOAT,
            TypeKind.POINTER,
            TypeKind.ENUM,
        }:
            if base.size is None or base.size <= 0:
                return False
            return self._append(data_type, offset)
        if base.kind == TypeKind.ARRAY:
            if self._array_primitive and depth != 0:
                return base.size is not None and self._append(data_type, offset)
            if base.target is None or base.count is None or base.target.size is None:
                return False
            element_size = base.target.unwrapped().align_size
            if element_size is None or element_size < 0:
                return False
            cursor = offset
            for _index in range(base.count):
                if not self._extract(base.target, cursor, depth + 1):
                    return False
                cursor += element_size
            return True
        if base.kind == TypeKind.UNION:
            return self._handle_union(data_type, offset)
        if base.kind != TypeKind.STRUCT:
            return False
        expected_offset = offset
        for field in base.fields:
            field_type = field.type.unwrapped()
            field_size = field_type.align_size
            alignment = field_type.alignment or 1
            current_offset = offset + field.offset
            if field_size is None or field_size < 0:
                return False
            if current_offset % alignment:
                self.aligned = False
            remainder = expected_offset % alignment
            if remainder:
                expected_offset += alignment - remainder
            if expected_offset != current_offset:
                self.holes = True
            if not self._extract(field.type, current_offset, depth + 1):
                return False
            expected_offset = current_offset + field_size
        return True

    def _handle_union(self, data_type: CType, offset: int) -> bool:
        base = data_type.unwrapped()
        if self._union_illegal or not base.fields:
            return False
        first = base.fields[0]
        common = PrimitiveExtractor(
            first.type,
            array_primitive=self._array_primitive,
            offset=offset + first.offset,
            max_primitives=self._max_primitives,
        )
        if not common.is_valid():
            return False
        for field in base.fields[1:]:
            alternate = PrimitiveExtractor(
                field.type,
                array_primitive=self._array_primitive,
                offset=offset + field.offset,
                max_primitives=self._max_primitives,
            )
            if not alternate.is_valid() or not self._common_refinement(
                common.primitives, alternate.primitives
            ):
                return False
        if len(self.primitives) + common.size() > self._max_primitives:
            return False
        self.primitives.extend(common.primitives)
        return True

    @classmethod
    def _check_overlap(
        cls,
        result: list[Primitive],
        smaller: list[Primitive],
        point: int,
        larger: Primitive,
    ) -> int:
        end_offset = larger.offset + larger.align_size
        prefer_smaller = larger.data_type.unwrapped().kind == TypeKind.FLOAT
        while point < len(smaller):
            current = smaller[point]
            if current.offset >= end_offset:
                break
            if current.offset + current.align_size > end_offset:
                return -1
            if prefer_smaller:
                result.append(current)
            point += 1
        if not prefer_smaller:
            result.append(larger)
        return point

    @classmethod
    def _common_refinement(
        cls, first: list[Primitive], second: list[Primitive]
    ) -> bool:
        first_point = 0
        second_point = 0
        common: list[Primitive] = []
        while first_point < len(first) and second_point < len(second):
            left = first[first_point]
            right = second[second_point]
            if (
                left.offset < right.offset
                and left.offset + left.align_size <= right.offset
            ):
                common.append(left)
                first_point += 1
                continue
            if (
                right.offset < left.offset
                and right.offset + right.align_size <= left.offset
            ):
                common.append(right)
                second_point += 1
                continue
            if left.align_size >= right.align_size:
                second_point = cls._check_overlap(
                    common, second, second_point, left
                )
                if second_point < 0:
                    return False
                first_point += 1
            else:
                first_point = cls._check_overlap(
                    common, first, first_point, right
                )
                if first_point < 0:
                    return False
                second_point += 1
        common.extend(first[first_point:])
        common.extend(second[second_point:])
        first[:] = common
        return True


class ParamListType(IntEnum):
    """List kinds corresponding to ``ParamList::p_*`` in ``fspec.hh``."""

    STANDARD = 0
    STANDARD_OUT = 1
    REGISTER = 2
    REGISTER_OUT = 3
    MERGED = 4


_HIDDEN_RETURN = object()


@dataclass(frozen=True, slots=True)
class TrialMapResult:
    """Summary of the mutations made by ``ParamListStandard.buildTrialMap``."""

    matched: tuple[str, ...]
    unmatched: tuple[str, ...]
    inserted_unreferenced: tuple[str, ...]


class ParamList:
    """Abstract input/output resource list used by prototype models."""

    list_type = ParamListType.STANDARD

    def allocate_status(self) -> list[int]:
        raise NotImplementedError

    def assign_map(
        self,
        parameters: Iterable[FunctionParameter | CType | str],
        *,
        types: TypeDatabase | None = None,
        status: list[int] | None = None,
    ) -> tuple[FunctionParameter, ...]:
        raise NotImplementedError

    def fillin_map(self, active: ParamActive) -> None:
        raise NotImplementedError

    def check_join(
        self,
        hi_location: StorageLocation,
        hi_size: int,
        lo_location: StorageLocation,
        lo_size: int,
    ) -> bool:
        raise NotImplementedError

    def check_split(
        self, location: StorageLocation, size: int, split_point: int
    ) -> bool:
        raise NotImplementedError

    def possible_param(self, location: StorageLocation, size: int) -> bool:
        raise NotImplementedError

    def possible_param_with_slot(
        self, location: StorageLocation, size: int
    ) -> tuple[int, int] | None:
        raise NotImplementedError


class ParamListStandard(ParamList):
    """Ordered resource groups with no-hole, exclusion, and stack-spill rules."""

    list_type = ParamListType.STANDARD

    def __init__(
        self,
        entries: Iterable[ParamEntry] = (),
        *,
        resource_start: Iterable[int] | None = None,
        this_before_return: bool = False,
        auto_killed_by_call: bool = False,
        split_float: bool = True,
        normal_stack: bool = True,
        model_rules: Iterable[object] = (),
    ) -> None:
        self.entries: list[ParamEntry] = []
        self.num_groups = 0
        self.max_delay = 0
        self.this_before_return = this_before_return
        self.auto_killed_by_call = auto_killed_by_call
        self.split_float = split_float
        self.normal_stack = normal_stack
        self.pointer_size: int | None = None
        self.resource_start: list[int] = []
        self._resource_start_supplied = resource_start is not None
        self._pending_resource_boundaries = list(resource_start or ())
        self.model_rules: list[object] = list(model_rules)
        self.killed_by_call_entries: list[ParamEntry] = []
        self._resolver: dict[tuple[str, ...], list[ParamEntry]] = {}
        self.unsupported_rules: tuple[CspecModelRule, ...] = ()
        for entry in entries:
            self.add_entry(entry)
        if self._resource_start_supplied:
            self.resource_start = list(self._pending_resource_boundaries)
        else:
            self._finish_resource_sections()
        self._rebuild_resolver()

    @classmethod
    def from_cspec(
        cls,
        cspec_path: str | Path,
        prototype_name: str,
        direction: str = "input",
        *,
        normal_stack: bool = True,
        list_type: ParamListType | None = None,
        big_endian: bool = False,
    ) -> ParamListStandard:
        """Decode a Ghidra Cspec ``<input>`` or ``<output>`` ParamList.

        This reader handles direct ``pentry`` and ``group`` resources, ranges,
        stack direction, overlap/join addresses, extension flags, and the rule
        actions currently supported by :class:`CspecModelRule`. Unsupported
        rules are retained in ``unsupported_rules`` rather than silently
        changing assignment behavior.
        """
        root = ElementTree.parse(cspec_path).getroot()
        prototype = next(
            (item for item in root.findall(".//prototype") if item.get("name") == prototype_name),
            None,
        )
        if prototype is None:
            raise ValueError(f"No existe el prototype {prototype_name!r} en {cspec_path}")
        direction = direction.lower()
        if direction not in {"input", "output"}:
            raise ValueError("direction debe ser 'input' o 'output'")
        resource_element = prototype.find(direction)
        if resource_element is None:
            raise ValueError(f"El prototype {prototype_name!r} no define {direction}")
        if list_type is None:
            list_type = (
                ParamListType.STANDARD_OUT
                if direction == "output"
                else ParamListType.STANDARD
            )
        if list_type == ParamListType.REGISTER:
            list_class: type[ParamListStandard] = ParamListRegister
        elif list_type == ParamListType.REGISTER_OUT:
            list_class = ParamListRegisterOut
        elif list_type == ParamListType.STANDARD_OUT:
            list_class = ParamListStandardOut
        else:
            list_class = cls
        auto_killed = resource_element.get("killedbycall", "false").lower() == "true"
        split_float = resource_element.get("separatefloat", "true").lower() != "false"
        list_model = list_class(
            this_before_return=resource_element.get("thisbeforeretpointer", "false").lower()
            == "true",
            auto_killed_by_call=auto_killed,
            split_float=split_float,
            normal_stack=normal_stack,
        )
        pointer_size_element = root.find("data_organization/pointer_size")
        if pointer_size_element is not None:
            try:
                list_model.pointer_size = int(pointer_size_element.get("value", ""))
            except ValueError:
                list_model.pointer_size = None
        group_index = 0
        for child in resource_element:
            if child.tag == "pentry":
                entry = _param_entry_from_cspec(child, group_index, big_endian)
                list_model.add_entry(entry)
                group_index = list_model.num_groups
            elif child.tag == "group":
                base_group = group_index
                group_entries: list[ParamEntry] = []
                for pentry in child.findall("pentry"):
                    entry = _param_entry_from_cspec(pentry, base_group, big_endian, grouped=True)
                    if group_entries:
                        ParamEntry.order_within_group(group_entries[-1], entry)
                        if len(group_entries) > 1:
                            ParamEntry.order_within_group(group_entries[-2], entry)
                    list_model.add_entry(entry)
                    group_entries.append(entry)
                group_index = list_model.num_groups
            elif child.tag == "rule":
                rule = CspecModelRule.from_element(child, direction)
                list_model.model_rules.append(rule)
            else:
                # Effects and notes are handled by ProtoModel, not ParamList.
                continue
        pointer_max = resource_element.get("pointermax")
        if pointer_max is not None:
            try:
                pointer_maximum = int(pointer_max, 0)
            except ValueError as exc:
                raise ValueError("pointermax debe ser un entero") from exc
            if pointer_maximum > 0:
                list_model.model_rules.append(
                    CspecModelRule(
                        "any",
                        "convert_to_ptr",
                        minimum_size=pointer_maximum + 1,
                    )
                )
        list_model._finish_resource_sections()
        list_model._rebuild_resolver()
        list_model.unsupported_rules = tuple(
            rule for rule in list_model.model_rules if not rule.is_supported
        )
        if isinstance(list_model, ParamListStandardOut):
            list_model.auto_killed_by_call = auto_killed
            list_model.use_fillin_fallback = None
            list_model.initialize_fillin_policy()
        return list_model

    def add_entry(self, entry: ParamEntry) -> None:
        """Append and resolve group/overlap metadata as Cspec decoding does."""
        previous = tuple(self.entries)
        if previous and entry.order == 0:
            entry.order = len(previous)
        entry.resolve_first(previous[-1] if previous else None)
        entry.resolve_join(previous)
        entry.resolve_overlap(previous)
        if not self.normal_stack and entry.alignment:
            if entry.max_size % entry.alignment:
                raise ValueError(
                    "Con reverse-stack, max_size debe ser múltiplo de alignment"
                )
            entry.reverse_stack = True
        self.entries.append(entry)
        self.num_groups = max(self.num_groups, max(entry.group_set) + 1)
        self.max_delay = max(self.max_delay, getattr(entry.location, "delay", 0))
        if not self._resource_start_supplied and self.split_float and len(self.entries) > 1:
            previous_entry = self.entries[-2]
            old_class = (
                ParamTypeClass.GENERAL
                if previous_entry.grouped
                else previous_entry.get_type()
            )
            new_class = ParamTypeClass.GENERAL if entry.grouped else entry.get_type()
            if old_class != new_class:
                if old_class < new_class:
                    raise ValueError(
                        "ParamEntry deben ordenarse por clase de storage"
                    )
                self._pending_resource_boundaries.append(entry.get_group())
        if self.auto_killed_by_call and not entry.is_stack_spill():
            self.killed_by_call_entries.append(entry)
        if self._resource_start_supplied:
            self.resource_start = list(self._pending_resource_boundaries)
        else:
            self._finish_resource_sections()
        self._rebuild_resolver()

    def _finish_resource_sections(self) -> None:
        starts = [0, *self._pending_resource_boundaries]
        if not starts or starts[-1] != self.num_groups:
            starts.append(self.num_groups)
        # Boundaries are group ids, and duplicate starts carry no information.
        self.resource_start = list(dict.fromkeys(starts))

    def _rebuild_resolver(self) -> None:
        resolver: dict[tuple[str, ...], list[ParamEntry]] = {}
        for entry in self.entries:
            for location in entry.locations:
                resolver.setdefault(_space_key(location), []).append(entry)
        self._resolver = resolver

    @property
    def stack_entry(self) -> ParamEntry | None:
        if self.entries and self.entries[-1].is_stack_spill():
            return self.entries[-1]
        return None

    @property
    def killed_by_call_ranges(self) -> tuple[ParamEntry, ...]:
        if self.auto_killed_by_call:
            return tuple(entry for entry in self.entries if not entry.is_stack_spill())
        return tuple(self.killed_by_call_entries)

    def get_stack_entry(self) -> ParamEntry | None:
        return self.stack_entry

    def get_num_groups(self) -> int:
        return self.num_groups

    def get_resource_start(self) -> tuple[int, ...]:
        return tuple(self.resource_start)

    def get_max_delay(self) -> int:
        return self.max_delay

    def get_range_list(self, kind: str) -> tuple[tuple[StorageLocation, int], ...]:
        """Return model ranges for a location kind (``register`` or ``stack``)."""
        ranges = []
        for entry in self.entries:
            for location in entry.locations:
                if location.kind == kind:
                    ranges.append((location, location.size))
        return tuple(ranges)

    def allocate_status(self) -> list[int]:
        return [0] * self.num_groups

    def find_entry(
        self, location: StorageLocation, size: int, *, justified: bool = True
    ) -> ParamEntry | None:
        for entry in self._resolver.get(_space_key(location), ()):
            if entry.get_min_size() > size:
                continue
            offset = entry.justified_contain(location, size)
            if offset == 0 or (not justified and offset >= 0):
                return entry
        return None

    def find_entry_for_locations(
        self, locations: Sequence[StorageLocation], size: int, *, justified: bool = True
    ) -> tuple[ParamEntry, int] | None:
        if len(locations) == 1:
            entry = self.find_entry(locations[0], size, justified=justified)
            if entry is not None:
                return entry, entry.justified_contain(locations[0], size)
            return None
        for entry in self.entries:
            if entry.get_min_size() > size:
                continue
            offset = _joined_containment_offset(entry, locations, size)
            if offset is not None and (offset == 0 or not justified):
                return entry, offset
        return None

    def select_unreferenced_entry(
        self, group: int, preferred_type: ParamTypeClass | str
    ) -> ParamEntry | None:
        preferred = _coerce_type_class(preferred_type)
        best: ParamEntry | None = None
        best_score = -1
        for entry in self.entries:
            if entry.get_group() != group:
                continue
            entry_type = entry.get_type()
            if entry_type == preferred:
                score = 2
            elif preferred == ParamTypeClass.GENERAL:
                score = 1
            else:
                score = 0
            if score > best_score:
                best, best_score = entry, score
        return best

    def _register_unreferenced(
        self, active: ParamActive, entry: ParamEntry, name: str
    ) -> ParamTrial:
        allocation_size = entry.get_size() if entry.is_exclusion else entry.get_align()
        if allocation_size <= 0:
            allocation_size = entry.get_size()
        location, _next_slot = entry.get_addr_by_slot(0, allocation_size, 1)
        if location is None:
            raise ValueError("No se pudo crear un trial ParamEntry no referenciado")
        base_name = name
        suffix = 1
        while any(trial.name == name for trial in active.trials):
            name = f"{base_name}_{suffix}"
            suffix += 1
        trial = active.register_trial(
            name,
            allocation_size * 8,
            (location,),
        )
        trial.mark_unreferenced()
        trial.set_entry(entry, 0)
        return trial

    def build_trial_map(self, active: ParamActive) -> TrialMapResult:
        """Port ``ParamListStandard::buildTrialMap`` and its inserted holes."""
        hitlist: list[ParamEntry | None] = []
        float_count = 0
        int_count = 0
        matched: list[str] = []
        unmatched: list[str] = []
        inserted: list[str] = []
        for index in range(active.get_num_trials()):
            trial = active.get_trial_at(index)
            trial.set_entry(None)
            match = self.find_entry_for_locations(
                trial.locations, trial.get_size(), justified=True
            )
            if match is None:
                trial.mark_no_use()
                unmatched.append(trial.name)
                continue
            entry, offset = match
            trial.set_entry(entry, offset)
            matched.append(trial.name)
            if trial.is_active():
                if entry.get_type() == ParamTypeClass.FLOAT:
                    float_count += 1
                else:
                    int_count += 1
            group = entry.get_group()
            while len(hitlist) <= group:
                hitlist.append(None)
            if hitlist[group] is None:
                hitlist[group] = entry

        for group, representative in enumerate(hitlist):
            if representative is None:
                representative = self.select_unreferenced_entry(
                    group,
                    ParamTypeClass.FLOAT
                    if float_count > int_count
                    else ParamTypeClass.GENERAL,
                )
                if representative is None:
                    continue
                name = f"__unreferenced_{group}"
                trial = self._register_unreferenced(active, representative, name)
                inserted.append(trial.name)
                continue
            if representative.is_exclusion:
                continue
            slot_occupancy: list[bool] = []
            for trial in active.trials:
                if trial.entry is not representative or not trial.locations:
                    continue
                location = trial.locations[0]
                first_slot = representative.get_slot(location, 0) - representative.get_group()
                last_slot = (
                    representative.get_slot(location, trial.get_size() - 1)
                    - representative.get_group()
                )
                if last_slot < first_slot:
                    first_slot, last_slot = last_slot, first_slot
                while len(slot_occupancy) <= last_slot:
                    slot_occupancy.append(False)
                for slot_index in range(max(0, first_slot), last_slot + 1):
                    slot_occupancy[slot_index] = True
            for slot_index, occupied in enumerate(slot_occupancy):
                if occupied:
                    continue
                location, _next_slot = representative.get_addr_by_slot(
                    slot_index, representative.get_align(), 1
                )
                if location is None:
                    continue
                name = f"__unreferenced_{group}_{slot_index}"
                base_name = name
                suffix = 1
                while any(trial.name == name for trial in active.trials):
                    name = f"{base_name}_{suffix}"
                    suffix += 1
                trial = active.register_trial(
                    name, representative.get_align() * 8, (location,)
                )
                trial.mark_unreferenced()
                trial.set_entry(representative, 0)
                inserted.append(trial.name)
        active.sort_trials()
        return TrialMapResult(tuple(matched), tuple(unmatched), tuple(inserted))

    def separate_sections(self, active: ParamActive) -> tuple[int, ...]:
        starts = self.resource_start or [0, self.num_groups]
        if len(starts) == 1:
            starts = [0, starts[0]]
        if starts[0] != 0:
            starts = [0, *starts]
        if starts[-1] != self.num_groups:
            starts = [*starts, self.num_groups]
        starts = list(dict.fromkeys(starts))
        boundaries = [0]
        next_section = 1
        next_group = starts[next_section] if next_section < len(starts) else self.num_groups
        for index, trial in enumerate(active.trials):
            if trial.entry is None:
                continue
            if trial.entry.get_group() >= next_group:
                boundaries.append(index)
                next_section += 1
                next_group = (
                    starts[next_section]
                    if next_section < len(starts)
                    else self.num_groups
                )
        boundaries.append(active.get_num_trials())
        return tuple(boundaries)

    @staticmethod
    def mark_group_no_use(active: ParamActive, active_trial: int, trial_start: int) -> None:
        number = active.get_num_trials()
        active_entry = active.get_trial_at(active_trial).entry
        if active_entry is None:
            return
        for index in range(trial_start, number):
            if index == active_trial:
                continue
            other = active.get_trial_at(index)
            if other.is_definitely_not_used():
                continue
            if other.entry is None or not other.entry.group_overlap(active_entry):
                break
            other.mark_no_use()

    @classmethod
    def mark_best_inactive(
        cls,
        active: ParamActive,
        group: int,
        group_start: int,
        preferred_type: ParamTypeClass | str,
    ) -> None:
        preferred = _coerce_type_class(preferred_type)
        best_index = -1
        best_score = -1
        for index in range(group_start, active.get_num_trials()):
            trial = active.get_trial_at(index)
            if trial.is_definitely_not_used():
                continue
            entry = trial.entry
            if entry is None or entry.get_group() != group:
                break
            if len(entry.get_all_groups()) > 1:
                continue
            score = 0
            if trial.has_ancestor_realistic():
                score += 5
                if trial.has_ancestor_solid():
                    score += 5
            if entry.get_type() == preferred:
                score += 1
            if score > best_score:
                best_index, best_score = index, score
        if best_index >= 0:
            cls.mark_group_no_use(active, best_index, group_start)

    @classmethod
    def force_exclusion_group(cls, active: ParamActive) -> None:
        current_group = -1
        group_start = -1
        inactive_count = 0
        for index in range(active.get_num_trials()):
            trial = active.get_trial_at(index)
            entry = trial.entry
            if trial.is_definitely_not_used() or entry is None or not entry.is_exclusion:
                continue
            group = entry.get_group()
            if group != current_group:
                if inactive_count > 1:
                    cls.mark_best_inactive(
                        active, current_group, group_start, ParamTypeClass.GENERAL
                    )
                current_group = group
                group_start = index
                inactive_count = 0
            if trial.is_active():
                cls.mark_group_no_use(active, index, group_start)
            else:
                inactive_count += 1
        if inactive_count > 1:
            cls.mark_best_inactive(
                active, current_group, group_start, ParamTypeClass.GENERAL
            )

    @staticmethod
    def force_no_use(active: ParamActive, start: int, stop: int) -> None:
        seen_definitely_not_used = False
        current_group = -1
        all_definitely_not_used = False
        for index in range(start, stop):
            trial = active.get_trial_at(index)
            entry = trial.entry
            if entry is None:
                continue
            group = entry.get_group()
            if group <= current_group and entry.is_exclusion:
                if not trial.is_definitely_not_used():
                    all_definitely_not_used = False
            else:
                if all_definitely_not_used:
                    seen_definitely_not_used = True
                all_definitely_not_used = trial.is_definitely_not_used()
                current_group = group
            if seen_definitely_not_used:
                trial.mark_inactive()

    @staticmethod
    def force_inactive_chain(
        active: ParamActive,
        max_chain: int,
        start: int,
        stop: int,
        group_start: int,
    ) -> None:
        seen_chain = False
        chain_length = 0
        last_active = -1
        for index in range(start, stop):
            trial = active.get_trial_at(index)
            if trial.is_definitely_not_used():
                continue
            if not trial.is_active():
                if (
                    trial.is_unreferenced()
                    and active.recover_subcall
                    and any(location.kind == "stack" for location in trial.locations)
                ):
                    seen_chain = True
                current_slot = trial.slot_group()
                if index == start:
                    chain_length += current_slot - group_start + 1
                else:
                    previous = active.get_trial_at(index - 1)
                    chain_length += current_slot - previous.slot_group()
                if chain_length > max_chain:
                    seen_chain = True
            else:
                chain_length = 0
                if not seen_chain:
                    last_active = index
            if seen_chain:
                trial.mark_inactive()
        for index in range(start, last_active + 1):
            trial = active.get_trial_at(index)
            if trial.is_definitely_not_used():
                continue
            if not trial.is_active():
                trial.mark_active()

    def fillin_map(self, active: ParamActive) -> TrialMapResult:
        """Resolve trial entries, resource holes, exclusions, and pass flags."""
        if active.get_num_trials() == 0:
            return TrialMapResult((), (), ())
        if not self.entries:
            raise ValueError("Cannot derive parameter storage without ParamEntry resources")
        result = self.build_trial_map(active)
        self.force_exclusion_group(active)
        boundaries = self.separate_sections(active)
        for section in range(len(boundaries) - 1):
            self.force_no_use(active, boundaries[section], boundaries[section + 1])
        starts = self.resource_start or [0, self.num_groups]
        if len(starts) == 1:
            starts = [0, starts[0]]
        if starts[0] != 0:
            starts = [0, *starts]
        for section in range(len(boundaries) - 1):
            section_start = starts[min(section, len(starts) - 1)]
            self.force_inactive_chain(
                active, 2, boundaries[section], boundaries[section + 1], section_start
            )
        for trial in active.trials:
            if trial.is_active():
                trial.mark_used()
        return result

    def assign_address_fallback(
        self,
        resource: ParamTypeClass | str,
        size: int,
        type_alignment: int,
        status: list[int],
        *,
        match_exact: bool = False,
    ) -> tuple[StorageLocation, ...] | None:
        requested = _coerce_type_class(resource)
        for entry in self.entries:
            group = entry.get_group()
            if group >= len(status) or status[group] < 0:
                continue
            entry_type = entry.get_type()
            if requested != entry_type and (
                match_exact or entry_type != ParamTypeClass.GENERAL
            ):
                continue
            locations, next_slot = entry.get_locations_by_slot(
                status[group], size, type_alignment
            )
            if locations is None:
                continue
            if entry.is_exclusion:
                for group_id in entry.get_all_groups():
                    if group_id < len(status):
                        status[group_id] = -1
            else:
                status[group] = next_slot
            return locations
        return None

    def assign_address(
        self,
        data_type: CType,
        status: list[int],
        *,
        prototype: FunctionSignature | None = None,
        position: int = -1,
        match_exact: bool = False,
        types: TypeDatabase | None = None,
    ) -> tuple[StorageLocation, ...] | None:
        for rule in self.model_rules:
            callback = getattr(rule, "assign_address", None)
            if callback is None:
                continue
            result = callback(data_type, prototype, position, status, self, types)
            if result is _HIDDEN_RETURN:
                return _HIDDEN_RETURN
            if result is not None:
                return tuple(result)
        return self.assign_address_fallback(
            _storage_class_for_type(data_type),
            data_type.size or 1,
            data_type.alignment or 1,
            status,
            match_exact=match_exact,
        )

    def assign_map(
        self,
        parameters: Iterable[FunctionParameter | CType | str],
        *,
        types: TypeDatabase | None = None,
        status: list[int] | None = None,
        match_exact: bool = False,
        prototype: FunctionSignature | None = None,
    ) -> tuple[FunctionParameter, ...]:
        resources = self.allocate_status() if status is None else status
        assignments: list[FunctionParameter] = []
        for index, parameter in enumerate(parameters):
            name, data_type, original = _coerce_parameter(parameter, index, types)
            locations = self.assign_address(
                data_type,
                resources,
                prototype=prototype,
                position=index,
                match_exact=match_exact,
                types=types,
            )
            if locations is _HIDDEN_RETURN:
                raise ValueError("ParamListStandard requiere un retorno oculto")
            if locations is None:
                raise ValueError(f"Cannot assign parameter storage for {name}")
            if original is None:
                assignments.append(
                    FunctionParameter(name, data_type.name, data_type, locations)
                )
            else:
                assignments.append(
                    replace(
                        original,
                        data_type=original.data_type or data_type,
                        locations=locations,
                    )
                )
        return tuple(assignments)

    def check_join(
        self,
        hi_location: StorageLocation,
        hi_size: int,
        lo_location: StorageLocation,
        lo_size: int,
    ) -> bool:
        high = self.find_entry(hi_location, hi_size, justified=True)
        low = self.find_entry(lo_location, lo_size, justified=True)
        if high is None or low is None:
            return False
        if high.get_group() == low.get_group():
            if high.is_exclusion or low.is_exclusion:
                return False
            if not _contiguous(high_location=hi_location, high_size=hi_size,
                               low_location=lo_location, low_size=lo_size):
                return False
            high_base = high.address_base
            low_base = low.address_base
            high_offset = _offset(hi_location)
            low_offset = _offset(lo_location)
            if (
                high.alignment
                and high_base is not None
                and high_offset is not None
                and (high_offset - high_base) % high.alignment
            ):
                return False
            return not (
                low.alignment
                and low_base is not None
                and low_offset is not None
                and (low_offset - low_base) % low.alignment
            )
        total_size = hi_size + lo_size
        for entry in self.entries:
            if entry.get_size() < total_size:
                continue
            if entry.justified_contain(lo_location, lo_size) != 0:
                continue
            if entry.justified_contain(hi_location, hi_size) != lo_size:
                continue
            return True
        return False

    def check_split(
        self, location: StorageLocation, size: int, split_point: int
    ) -> bool:
        if split_point <= 0 or split_point >= size:
            return False
        first = _slice_physical_location(location, 0, split_point)
        second = _slice_physical_location(location, split_point, size - split_point)
        return (
            self.find_entry(first, split_point, justified=True) is not None
            and self.find_entry(second, size - split_point, justified=True) is not None
        )

    def characterize_as_param(self, location: StorageLocation, size: int) -> int:
        contains = False
        contained_by = False
        for entry in self._resolver.get(_space_key(location), ()):
            offset = entry.justified_contain(location, size)
            if offset == 0:
                return 2
            if offset > 0:
                contains = True
            if entry.is_exclusion and entry.contained_by(location, size):
                contained_by = True
        if contains:
            return 1
        if contained_by:
            return 3
        for entry in self.entries:
            if entry.is_exclusion and entry.contained_by(location, size):
                return 3
        return 0

    def possible_param(self, location: StorageLocation, size: int) -> bool:
        return self.find_entry(location, size, justified=True) is not None

    def possible_param_with_slot(
        self, location: StorageLocation, size: int
    ) -> tuple[int, int] | None:
        entry = self.find_entry(location, size, justified=True)
        if entry is None:
            return None
        slot = entry.get_slot(location, 0)
        if entry.is_exclusion:
            slot_size = len(entry.get_all_groups())
        else:
            slot_size = (size + entry.get_align() - 1) // entry.get_align()
        return slot, slot_size

    def get_biggest_contained_param(
        self, location: StorageLocation, size: int
    ) -> StorageLocation | None:
        candidates = [
            entry for entry in self.entries if entry.contained_by(location, size)
        ]
        if not candidates:
            return None
        largest = max(candidates, key=lambda entry: entry.get_size())
        return largest.location if largest.is_exclusion else None

    def unjustified_container(
        self, location: StorageLocation, size: int
    ) -> StorageLocation | None:
        for entry in self.entries:
            if entry.get_min_size() > size:
                continue
            offset = entry.justified_contain(location, size)
            if offset < 0:
                continue
            if offset == 0:
                return None
            return entry.get_container(location, size)
        return None

    def assumed_extension(
        self, location: StorageLocation, size: int
    ) -> tuple[str, StorageLocation | None]:
        for entry in self.entries:
            if entry.get_min_size() > size:
                continue
            opcode, container = entry.assumed_extension(location, size)
            if opcode != "COPY":
                return opcode, container
        return "COPY", None

    def get_spacebase(self) -> str | None:
        stack_entry = self.stack_entry
        if stack_entry is None or stack_entry.location is None:
            return None
        return stack_entry.location.kind

    def fillin_map_fallback(self, active: ParamActive, first_only: bool) -> None:
        best_entry: ParamEntry | None = None
        best_cover = 0
        best_class = ParamTypeClass.POINTER
        for entry in self.entries:
            if (
                first_only
                and not entry.is_first_in_class()
                and entry.is_exclusion
                and len(entry.get_all_groups()) == 1
            ):
                continue
            putative = False
            for trial in active.trials:
                if trial.is_active():
                    offset = (
                        entry.justified_contain(
                            trial.locations[0], trial.get_size()
                        )
                        if trial.locations
                        else -1
                    )
                    if offset >= 0:
                        trial.set_entry(entry, offset)
                        putative = True
                    else:
                        trial.set_entry(None)
                else:
                    trial.set_entry(None)
            if not putative:
                continue
            active.sort_trials()
            offset_match = 0
            matched_count = 0
            for trial in active.trials:
                if trial.entry is None:
                    continue
                if offset_match != trial.get_offset():
                    break
                if (
                    (offset_match == 0 and entry.is_param_check_low())
                    or (offset_match != 0 and entry.is_param_check_high())
                ) and (trial.is_rem_formed() or trial.is_indirect_created()):
                    break
                offset_match += trial.get_size()
                matched_count += 1
            if offset_match < entry.get_min_size():
                matched_count = 0
            if (
                matched_count == active.get_num_trials()
                and (
                    entry.get_type() < best_class
                    or offset_match > best_cover
                )
            ):
                best_entry = entry
                best_cover = offset_match
                best_class = entry.get_type()
        if best_entry is None:
            for trial in active.trials:
                trial.mark_no_use()
            return
        for trial in active.trials:
            if not trial.is_active():
                trial.mark_no_use()
                trial.set_entry(None)
                continue
            offset = (
                best_entry.justified_contain(
                    trial.locations[0], trial.get_size()
                )
                if trial.locations
                else -1
            )
            if offset >= 0:
                trial.mark_used()
                trial.set_entry(best_entry, offset)
            else:
                trial.mark_no_use()
                trial.set_entry(None)
        active.sort_trials()


class ParamListStandardOut(ParamListStandard):
    """Standard return resource list with single-return fill-in semantics."""

    list_type = ParamListType.STANDARD_OUT

    def __init__(self, *args, use_fillin_fallback: bool | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_fillin_fallback = use_fillin_fallback
        self.initialize_fillin_policy()

    def initialize_fillin_policy(self) -> None:
        if self.use_fillin_fallback is None:
            affects_output = False
            for rule in self.model_rules:
                predicate = getattr(rule, "can_affect_fillin_output", False)
                if callable(predicate):
                    predicate = predicate()
                if predicate:
                    affects_output = True
                    break
            self.use_fillin_fallback = not affects_output
        if self.use_fillin_fallback:
            self.auto_killed_by_call = True

    def assign_map(
        self,
        parameters: Iterable[FunctionParameter | CType | str],
        *,
        types: TypeDatabase | None = None,
        status: list[int] | None = None,
        prototype: FunctionSignature | None = None,
        match_exact: bool = False,
    ) -> tuple[FunctionParameter, ...]:
        requested = tuple(parameters)
        if len(requested) > 1:
            raise ValueError("ParamListStandardOut solo asigna un resultado")
        if not requested:
            return ()
        resources = self.allocate_status() if status is None else status
        name, data_type, original = _coerce_parameter(requested[0], -1, types)
        if data_type.unwrapped().kind == TypeKind.VOID:
            return (original or FunctionParameter(name, data_type.name, data_type),)
        locations = self.assign_address(
            data_type,
            resources,
            prototype=prototype,
            position=-1,
            match_exact=match_exact,
            types=types,
        )
        if locations is _HIDDEN_RETURN or locations is None:
            if locations is _HIDDEN_RETURN or any(
                getattr(rule, "requires_hidden_return", lambda *_args: False)(
                    data_type, prototype, -1
                )
                for rule in self.model_rules
            ):
                raise ValueError(
                    "El modelo requiere un retorno oculto; resuélvelo con PrototypeModel"
                )
            raise ValueError(f"Cannot assign return storage for {data_type.name}")
        if original is None:
            return (FunctionParameter(name, data_type.name, data_type, locations),)
        return (
            replace(
                original,
                data_type=original.data_type or data_type,
                locations=locations,
            ),
        )

    def fillin_map(self, active: ParamActive) -> None:
        if active.get_num_trials() == 0:
            return
        if self.use_fillin_fallback:
            self.fillin_map_fallback(active, first_only=False)
            return
        for trial in active.trials:
            trial.set_entry(None)
            if not trial.is_active():
                continue
            match = self.find_entry_for_locations(
                trial.locations, trial.get_size(), justified=False
            )
            if match is None:
                trial.mark_no_use()
                continue
            if (
                (trial.is_rem_formed() or trial.is_indirect_created())
                and not match[0].is_first_in_class()
            ):
                trial.mark_no_use()
                continue
            trial.set_entry(match[0], match[1])
        active.sort_trials()
        for rule in self.model_rules:
            callback = getattr(rule, "fillin_output_map", None)
            if callback is not None and callback(active, self):
                for trial in active.trials:
                    if trial.is_active():
                        trial.mark_used()
                    else:
                        trial.mark_no_use()
                        trial.set_entry(None)
                return
        self.fillin_map_fallback(active, first_only=True)

    def possible_param(self, location: StorageLocation, size: int) -> bool:
        return any(entry.justified_contain(location, size) >= 0 for entry in self.entries)


@dataclass(frozen=True, slots=True)
class CspecModelRule:
    """A decoded ``<rule>`` filter/action pair from an x86 Cspec.

    The x86 rules used by this port are decoded and applied to resource lists.
    Dual-class joining expands aggregate primitives, refines union layouts,
    classifies each resource tile, and commits register/stack consumption only
    after the full assignment succeeds. Qualifier filters and less common
    architecture-specific actions are retained and reported as unsupported.
    """

    datatype_name: str
    action: str
    minimum_size: int = 0
    maximum_size: int = 0
    sizes: tuple[int, ...] = ()
    attributes: tuple[tuple[str, str], ...] = ()
    qualifiers: tuple[str, ...] = ()
    base_storage: ParamTypeClass = ParamTypeClass.GENERAL
    alternate_storage: ParamTypeClass = ParamTypeClass.FLOAT
    consume_from_stack: bool = False
    consume_most_significant: bool = False
    justify_right: bool = False
    fill_alternate: bool = False
    hidden_strategy: str = ""
    hidden_voidlock: bool = False

    @classmethod
    def from_element(cls, element: ElementTree.Element, direction: str) -> CspecModelRule:
        datatype = element.find("datatype")
        if datatype is None:
            return cls("any", "unsupported", qualifiers=("missing-datatype",))
        actions = [
            child
            for child in element
            if child.tag not in {"datatype", "position", "datatype_at", "varargs"}
        ]
        action_element = actions[0] if actions else None
        action = action_element.tag if action_element is not None else "unsupported"
        attributes = tuple(sorted(action_element.attrib.items())) if action_element is not None else ()
        qualifiers = tuple(
            child.tag
            for child in element
            if child.tag in {"position", "datatype_at", "varargs"}
        )
        if action == "hidden_return" and direction != "output":
            qualifiers += ("hidden_return_not_output",)
        action_attributes = dict(attributes)
        sizes_text = datatype.get("sizes", "")
        try:
            sizes = tuple(sorted({int(item.strip()) for item in sizes_text.split(",") if item.strip()}))
            minimum = int(datatype.get("minsize", "0"))
            maximum = int(datatype.get("maxsize", "0"))
        except ValueError as exc:
            raise ValueError("Atributos de tamaño inválidos en regla Cspec") from exc
        if sizes and (minimum or maximum):
            raise ValueError("Una regla Cspec no puede combinar sizes con min/max")
        return cls(
            datatype_name=datatype.get("name", "any"),
            action=action,
            minimum_size=minimum,
            maximum_size=maximum,
            sizes=sizes,
            attributes=attributes,
            qualifiers=qualifiers,
            base_storage=_coerce_type_class(
                action_attributes.get("storage", action_attributes.get("a", "general"))
            ),
            alternate_storage=_coerce_type_class(action_attributes.get("b", "float")),
            consume_from_stack=action_attributes.get("stackspill", "false").lower() == "true",
            consume_most_significant=action_attributes.get("reversesignif", "false").lower() == "true",
            justify_right=action_attributes.get("reversejustify", "false").lower() == "true",
            fill_alternate=action_attributes.get("fillalternate", "false").lower() == "true",
            hidden_strategy=(
                action_attributes.get("strategy", "special")
                if action == "hidden_return"
                else ""
            ),
            hidden_voidlock=action_attributes.get("voidlock", "false").lower() == "true",
        )

    @property
    def is_supported(self) -> bool:
        allowed_attributes = {
            "goto_stack": set(),
            "convert_to_ptr": set(),
            "hidden_return": {"strategy", "voidlock"},
            "join_dual_class": {
                "storage",
                "a",
                "b",
                "stackspill",
                "reversesignif",
                "reversejustify",
                "fillalternate",
            },
        }
        if self.action not in allowed_attributes or self.qualifiers:
            return False
        if any(key not in allowed_attributes[self.action] for key, _value in self.attributes):
            return False
        return not (
            self.action == "hidden_return"
            and self.hidden_strategy not in {"normalparam", "special"}
        )

    def can_affect_fillin_output(self) -> bool:
        return self.action in {"goto_stack", "join", "join_dual_class", "consume"}

    def matches_type(self, data_type: CType) -> bool:
        base = data_type.unwrapped()
        type_matches = {
            "any": True,
            "struct": base.kind == TypeKind.STRUCT,
            "union": base.kind == TypeKind.UNION,
            "array": base.kind == TypeKind.ARRAY,
            "float": base.kind == TypeKind.FLOAT,
            "ptr": base.kind == TypeKind.POINTER,
            "pointer": base.kind == TypeKind.POINTER,
            "int": base.kind in {TypeKind.INTEGER, TypeKind.BOOL, TypeKind.ENUM},
            "uint": base.kind in {TypeKind.INTEGER, TypeKind.BOOL, TypeKind.ENUM},
            "vector": base.kind == TypeKind.VECTOR,
        }.get(self.datatype_name)
        if not type_matches:
            return False
        size = data_type.size or 0
        if self.sizes and size not in self.sizes:
            return False
        return not (
            (self.minimum_size and size < self.minimum_size)
            or (self.maximum_size and size > self.maximum_size)
        )

    def requires_hidden_return(
        self,
        data_type: CType,
        prototype: FunctionSignature | None = None,
        position: int = -1,
    ) -> bool:
        return (
            self.action == "hidden_return"
            and self.matches_type(data_type)
            and position < 0
        )

    def assign_address(
        self,
        data_type: CType,
        prototype: FunctionSignature | None,
        position: int,
        status: list[int],
        resource: ParamListStandard,
        types: TypeDatabase | None,
    ) -> tuple[StorageLocation, ...] | None:
        if not self.is_supported or not self.matches_type(data_type):
            return None
        if self.action == "hidden_return" and position < 0:
            return _HIDDEN_RETURN
        if self.action == "join_dual_class":
            return self._assign_dual_class(data_type, status, resource)
        if self.action == "goto_stack":
            stack_entry = resource.stack_entry
            if stack_entry is None:
                return None
            group = stack_entry.get_group()
            location, _next_slot = stack_entry.get_addr_by_slot(
                status[group], data_type.size or 0, data_type.alignment or 1
            )
            if location is None:
                return None
            return (location,)
        if self.action == "convert_to_ptr":
            pointer_size = resource.pointer_size or _infer_pointer_size(resource)
            assigned = resource.assign_address_fallback(
                ParamTypeClass.POINTER,
                pointer_size,
                pointer_size,
                status,
            )
            if assigned is None:
                return None
            return tuple(replace(location, indirect=True) for location in assigned)
        return None

    def _assign_dual_class(
        self,
        data_type: CType,
        status: list[int],
        resource: ParamListStandard,
    ) -> tuple[StorageLocation, ...] | None:
        base_tiles = [
            entry
            for entry in resource.entries
            if entry.is_exclusion
            and len(entry.get_all_groups()) == 1
            and entry.get_type() == self.base_storage
        ]
        alternate_tiles = [
            entry
            for entry in resource.entries
            if entry.is_exclusion
            and len(entry.get_all_groups()) == 1
            and entry.get_type() == self.alternate_storage
        ]
        if not base_tiles or not alternate_tiles:
            return None
        tile_size = base_tiles[0].get_size()
        if tile_size <= 0 or alternate_tiles[0].get_size() != tile_size:
            return None

        primitives = PrimitiveExtractor(data_type, max_primitives=1024)
        size = data_type.unwrapped().size
        if (
            not primitives.is_valid()
            or primitives.size() == 0
            or primitives.contains_holes()
            or size is None
            or size <= 0
        ):
            return None

        trial_status = list(status)
        locations: list[StorageLocation] = []
        primitive_index = 0
        base_index = 0
        alternate_index = 0
        size_left = size
        offset = 0

        while size_left > 0:
            tile_class, primitive_index = self._get_tile_class(
                primitives, offset, primitive_index, tile_size
            )
            if tile_class is None:
                return None
            tiles = base_tiles if tile_class == 0 else alternate_tiles
            tile_index = base_index if tile_class == 0 else alternate_index
            while tile_index < len(tiles):
                group = tiles[tile_index].get_group()
                if group < len(trial_status) and trial_status[group] == 0:
                    break
                tile_index += 1
            if tile_class == 0:
                base_index = tile_index
            else:
                alternate_index = tile_index

            if tile_index == len(tiles):
                if not self.consume_from_stack:
                    return None
                break

            entry = tiles[tile_index]
            group = entry.get_group()
            location, _next_slot = entry.get_addr_by_slot(
                trial_status[group], entry.get_size(), 1
            )
            if location is None:
                return None
            trial_status[group] = -1
            locations.append(
                replace(location, size=entry.get_size(), value_offset=offset)
            )
            size_left -= entry.get_size()
            offset = size - size_left

        if size_left > 0:
            if not self.consume_from_stack:
                return None
            stack_entry = resource.stack_entry
            if stack_entry is None:
                return None
            group = stack_entry.get_group()
            if group >= len(trial_status):
                return None
            stack_location, next_slot = stack_entry.get_addr_by_slot(
                trial_status[group],
                size_left,
                data_type.alignment or 1,
                justify_right=self.justify_right,
            )
            if stack_location is None:
                return None
            trial_status[group] = next_slot
            locations.append(
                replace(stack_location, value_offset=offset)
            )
        elif size_left < 0:
            excess = -size_left
            piece_index = 0 if self.justify_right else len(locations) - 1
            if piece_index < 0:
                return None
            piece = locations[piece_index]
            if excess >= piece.size:
                return None
            add_offset = (
                _resource_is_big_endian(resource)
                ^ self.consume_most_significant
                ^ self.justify_right
            )
            updates: dict[str, int] = {"size": piece.size - excess}
            if add_offset:
                if piece.kind == "register":
                    updates["register_offset"] = piece.register_offset + excess
                elif piece.kind == "stack" and piece.stack_offset is not None:
                    updates["stack_offset"] = piece.stack_offset + excess
                else:
                    return None
            locations[piece_index] = replace(piece, **updates)

        status[:] = trial_status
        return tuple(locations)

    def _get_tile_class(
        self,
        primitives: PrimitiveExtractor,
        offset: int,
        index: int,
        tile_size: int,
    ) -> tuple[int | None, int]:
        if index >= primitives.size():
            return None, index
        first = primitives.get(index)
        boundary = offset + tile_size
        count = 0
        tile_class = 1
        while index < primitives.size():
            primitive = primitives.get(index)
            if primitive.offset < offset:
                return None, index
            if primitive.offset >= boundary:
                break
            if primitive.offset + primitive.size > boundary:
                return None, index
            count += 1
            index += 1
            primitive_class = (
                ParamTypeClass.FLOAT
                if primitive.data_type.unwrapped().kind == TypeKind.FLOAT
                else ParamTypeClass.GENERAL
            )
            if primitive_class != self.alternate_storage:
                tile_class = 0
        if count == 0:
            return None, index
        if self.fill_alternate and (count > 1 or first.size != tile_size):
            tile_class = 0
        return tile_class, index

    def fillin_output_map(
        self, active: ParamActive, resource: ParamListStandard
    ) -> bool:
        if not self.is_supported:
            return False
        if self.action == "goto_stack":
            stack_entry = resource.stack_entry
            if stack_entry is None:
                return False
            count = 0
            for trial in active.trials:
                if trial.entry is None:
                    break
                if trial.entry is not stack_entry:
                    return False
                count += 1
                if count > 1:
                    return False
            return count == 1
        if self.action != "join_dual_class":
            return False
        count = 0
        current_group = -1
        resource_type: ParamTypeClass | None = None
        partial_index = -1
        for index, trial in enumerate(active.trials):
            entry = trial.entry
            if entry is None:
                break
            if count == 0:
                resource_type = entry.get_type()
                if resource_type not in {self.base_storage, self.alternate_storage}:
                    return False
            elif entry.get_type() != resource_type:
                return False
            if count == 0:
                if not entry.is_first_in_class():
                    return False
            elif entry.get_group() != current_group + 1:
                return False
            current_group = entry.get_group()
            if trial.get_size() != entry.get_size():
                if partial_index != -1:
                    return False
                partial_index = index
            count += 1
        if partial_index != -1:
            if self.justify_right:
                if partial_index != 0:
                    return False
            elif partial_index != count - 1:
                return False
            trial = active.get_trial_at(partial_index)
            if self.justify_right == self.consume_most_significant:
                if trial.get_offset() != 0:
                    return False
            elif trial.get_offset() + trial.get_size() != trial.entry.get_size():
                return False
        if count and self.consume_most_significant:
            active.set_join_most_sig_first()
        return count > 0


class ParamListRegister(ParamListStandard):
    """Unordered input resources where holes are permitted."""

    list_type = ParamListType.REGISTER

    def fillin_map(self, active: ParamActive) -> None:
        for trial in active.trials:
            match = self.find_entry_for_locations(
                trial.locations, trial.get_size(), justified=True
            )
            if match is None:
                trial.mark_no_use()
                trial.set_entry(None)
            else:
                trial.set_entry(*match)
                if trial.is_active():
                    trial.mark_used()
        active.sort_trials()


class ParamListRegisterOut(ParamListStandardOut):
    """Output register set using standard single-result fill-in selection."""

    list_type = ParamListType.REGISTER_OUT


class ParamListMerged(ParamListStandard):
    """Union of resource lists used while a calling convention is unresolved."""

    list_type = ParamListType.MERGED

    def __init__(self) -> None:
        super().__init__(resource_start=(0,))

    def fold_in(self, other: ParamListStandard) -> None:
        for incoming in other.entries:
            duplicate = False
            replacement_index: int | None = None
            for index, current in enumerate(self.entries):
                if current.subsumes_definition(incoming):
                    if current.get_min_size() == incoming.get_min_size():
                        duplicate = True
                        break
                elif (
                    incoming.subsumes_definition(current)
                    and current.get_min_size() == incoming.get_min_size()
                ):
                    replacement_index = index
                    break
            if duplicate:
                continue
            cloned = replace(
                incoming,
                group_set=tuple(incoming.group_set),
                join_pieces=tuple(incoming.join_pieces),
            )
            if replacement_index is not None:
                self.entries[replacement_index] = cloned
            else:
                self.entries.append(cloned)
        self.num_groups = max(
            (max(entry.group_set) + 1 for entry in self.entries), default=0
        )
        self._finish_resource_sections()
        self._rebuild_resolver()

    def finalize(self) -> None:
        self._rebuild_resolver()

    def assign_map(self, *args, **kwargs):
        raise RuntimeError("Resolve ProtoModelMerged before assigning parameter storage")

    def fillin_map(self, active: ParamActive) -> None:
        raise RuntimeError("Resolve ProtoModelMerged before deriving a parameter map")

    def clone(self) -> ParamListMerged:
        result = ParamListMerged()
        result.entries = [
            replace(entry, group_set=tuple(entry.group_set), join_pieces=tuple(entry.join_pieces))
            for entry in self.entries
        ]
        result.num_groups = self.num_groups
        result.resource_start = list(self.resource_start)
        result._rebuild_resolver()
        return result


@dataclass(slots=True)
class PrototypeModel:
    """Input/output ParamLists and metadata for one concrete prototype model."""

    name: str
    input_list: ParamListStandard
    output_list: ParamListStandardOut | ParamListStandard
    this_before_return_pointer: bool = False
    stack_alignment: int = 1
    extra_pop: int | None = None
    stack_shift: int = 0

    @classmethod
    def from_cspec(
        cls,
        cspec_path: str | Path,
        model_name: str,
        *,
        normal_stack: bool = True,
    ) -> PrototypeModel:
        root = ElementTree.parse(cspec_path).getroot()
        element = next(
            (item for item in root.findall(".//prototype") if item.get("name") == model_name),
            None,
        )
        if element is None:
            raise ValueError(f"No existe el prototype {model_name!r} en {cspec_path}")
        input_list = ParamListStandard.from_cspec(
            cspec_path, model_name, "input", normal_stack=normal_stack
        )
        output_list = ParamListStandard.from_cspec(
            cspec_path, model_name, "output", normal_stack=normal_stack
        )
        try:
            extra_pop = int(element.get("extrapop", ""), 0)
        except ValueError:
            extra_pop = None
        try:
            stack_shift = int(element.get("stackshift", "0"), 0)
        except ValueError:
            stack_shift = 0
        return cls(
            model_name,
            input_list,
            output_list,
            this_before_return_pointer=input_list.this_before_return,
            stack_alignment=(
                input_list.stack_entry.alignment
                if input_list.stack_entry is not None
                and input_list.stack_entry.alignment
                else 1
            ),
            extra_pop=extra_pop,
            stack_shift=stack_shift,
        )

    def possible_input_param_with_slot(
        self, location: StorageLocation, size: int
    ) -> tuple[int, int] | None:
        return self.input_list.possible_param_with_slot(location, size)

    def possible_output_param_with_slot(
        self, location: StorageLocation, size: int
    ) -> tuple[int, int] | None:
        return self.output_list.possible_param_with_slot(location, size)

    def fillin_map(self, active: ParamActive, *, output: bool = False) -> TrialMapResult | None:
        resource = self.output_list if output else self.input_list
        return resource.fillin_map(active)

    def assign_parameters(
        self,
        parameters: Iterable[FunctionParameter | CType | str],
        *,
        types: TypeDatabase | None = None,
        prototype: FunctionSignature | None = None,
    ) -> tuple[FunctionParameter, ...]:
        return self.input_list.assign_map(
            parameters, types=types, prototype=prototype
        )

    def assign_signature(
        self,
        signature: FunctionSignature,
        *,
        types: TypeDatabase,
    ) -> FunctionSignature:
        """Assign all formal storage, including a hidden structure-return pointer."""
        return_type = signature.return_data_type or types.resolve(signature.return_type)
        hidden_rule = next(
            (
                rule
                for rule in self.output_list.model_rules
                if getattr(rule, "requires_hidden_return", lambda *_args: False)(
                    return_type, signature, -1
                )
            ),
            None,
        )
        if hidden_rule is not None and not getattr(hidden_rule, "is_supported", False):
            raise NotImplementedError("La regla Cspec de retorno oculto no está soportada")
        hidden_strategy = getattr(hidden_rule, "hidden_strategy", "normalparam")
        hidden_voidlock = bool(getattr(hidden_rule, "hidden_voidlock", False))
        output_status = self.output_list.allocate_status()
        output_parameter = FunctionParameter("", return_type.name, return_type)
        hidden_return = False
        try:
            output_assignment = self.output_list.assign_map(
                (output_parameter,),
                types=types,
                status=output_status,
                prototype=signature,
            )[0]
            return_locations = output_assignment.locations
        except ValueError:
            if return_type.unwrapped().kind == TypeKind.VOID:
                return_locations = ()
            else:
                hidden_return = True
                if hidden_voidlock:
                    return_locations = ()
                else:
                    pointer_type = types.pointer(return_type)
                    pointer_locations = self.output_list.assign_address(
                        pointer_type,
                        output_status,
                        prototype=signature,
                        position=-1,
                        types=types,
                    )
                    if pointer_locations is None or pointer_locations is _HIDDEN_RETURN:
                        raise ValueError(
                            f"No se pudo asignar el puntero de retorno de {return_type.name}"
                        )
                    return_locations = tuple(
                        replace(location, hidden=True)
                        for location in pointer_locations
                    )
        input_parameters = list(signature.parameters)
        hidden_location: StorageLocation | None = None
        if hidden_return:
            hidden_name = "__ghidra_hidden_return__"
            while any(parameter.name == hidden_name for parameter in input_parameters):
                hidden_name += "_"
            hidden_pointer = types.pointer(return_type)
            hidden_parameter = FunctionParameter(
                hidden_name,
                hidden_pointer.name,
                hidden_pointer,
                implicit=True,
            )
            insertion = (
                1
                if self.this_before_return_pointer
                and input_parameters
                and input_parameters[0].implicit
                else 0
            )
            input_parameters.insert(insertion, hidden_parameter)
        if hidden_return and hidden_strategy == "special":
            status = self.input_list.allocate_status()
            before = self.input_list.assign_map(
                input_parameters[:insertion],
                types=types,
                status=status,
                prototype=signature,
            )
            hidden_pointer = input_parameters[insertion]
            hidden_locations = self.input_list.assign_address_fallback(
                ParamTypeClass.HIDDEN_RETURN,
                hidden_pointer.data_type.size if hidden_pointer.data_type else types.pointer_size,
                types.pointer_size,
                status,
            )
            if hidden_locations is None:
                raise ValueError("No se pudo asignar el parámetro sret especial")
            hidden_assigned = replace(hidden_pointer, locations=hidden_locations)
            after = self.input_list.assign_map(
                input_parameters[insertion + 1 :],
                types=types,
                status=status,
                prototype=signature,
            )
            located = (*before, hidden_assigned, *after)
        else:
            located = self.input_list.assign_map(
                input_parameters,
                types=types,
                prototype=signature,
            )
        if hidden_return:
            hidden_assigned = next(
                parameter for parameter in located if parameter.name == hidden_name
            )
            hidden_location = (
                replace(hidden_assigned.locations[0], hidden=True, indirect=True)
                if hidden_assigned.locations
                else None
            )
            located = tuple(
                parameter for parameter in located if parameter.name != hidden_name
            )
        return replace(
            signature,
            return_type=return_type.name,
            return_data_type=return_type,
            parameters=tuple(located),
            return_locations=return_locations,
            hidden_return=hidden_return,
            hidden_return_location=hidden_location,
            calling_convention=self.name,
        )


@dataclass(frozen=True, slots=True)
class _ScoredParameter:
    original_index: int
    slot: int
    size: int


class ScoreProtoModel:
    """Ghidra ``ScoreProtoModel`` slot-hole/overlap scoring algorithm."""

    def __init__(
        self, is_input: bool, model: PrototypeModel, num_parameters: int = 0
    ) -> None:
        self.is_input = is_input
        self.model = model
        self.entries: list[_ScoredParameter] = []
        self.final_score = -1
        self.mismatch = 0
        self.expected_parameters = num_parameters

    def add_parameter(
        self,
        location: StorageLocation | Sequence[StorageLocation],
        size: int,
    ) -> None:
        original_index = len(self.entries)
        locations = (location,) if isinstance(location, StorageLocation) else tuple(location)
        resource = self.model.input_list if self.is_input else self.model.output_list
        match: tuple[int, int] | None = None
        for piece in locations:
            match = resource.possible_param_with_slot(piece, size)
            if match is not None:
                break
        if match is None and locations:
            joined = resource.find_entry_for_locations(locations, size, justified=True)
            if joined is not None:
                entry, _offset = joined
                slot = entry.get_slot(locations[0], 0)
                slot_size = (
                    len(entry.get_all_groups())
                    if entry.is_exclusion
                    else (size + entry.get_align() - 1) // entry.get_align()
                )
                match = slot, slot_size
        if match is None:
            self.mismatch += 1
            return
        self.entries.append(_ScoredParameter(original_index, match[0], match[1]))

    def do_score(self) -> int:
        self.entries.sort(key=lambda item: item.slot)
        next_free = 0
        base_score = 0
        penalties = (16, 10, 7, 5)
        final_penalty = 3
        mismatch_penalty = 20
        for entry in self.entries:
            if entry.slot > next_free:
                while next_free < entry.slot:
                    base_score += (
                        penalties[next_free]
                        if next_free < len(penalties)
                        else final_penalty
                    )
                    next_free += 1
                next_free += entry.size
            elif next_free > entry.slot:
                base_score += mismatch_penalty
                next_free = max(next_free, entry.slot + entry.size)
            else:
                next_free = entry.slot + entry.size
        self.final_score = base_score + mismatch_penalty * self.mismatch
        return self.final_score

    def get_score(self) -> int:
        return self.final_score

    def get_num_mismatch(self) -> int:
        return self.mismatch


class ProtoModelMerged:
    """Merged resource model whose concrete member is chosen by slot scoring."""

    @classmethod
    def from_cspec(
        cls,
        cspec_path: str | Path,
        resolve_name: str,
        *,
        normal_stack: bool = True,
    ) -> ProtoModelMerged:
        root = ElementTree.parse(cspec_path).getroot()
        resolve = next(
            (
                element
                for element in root.findall(".//resolveprototype")
                if element.get("name") == resolve_name
            ),
            None,
        )
        if resolve is None:
            raise ValueError(f"No existe resolveprototype {resolve_name!r} en {cspec_path}")
        models = [
            PrototypeModel.from_cspec(cspec_path, child.get("name", ""), normal_stack=normal_stack)
            for child in resolve.findall("model")
        ]
        if not models:
            raise ValueError(f"resolveprototype {resolve_name!r} no contiene modelos")
        return cls(models, name=resolve_name)

    def __init__(
        self,
        models: Iterable[PrototypeModel] = (),
        *,
        name: str = "merged",
    ) -> None:
        self.name = name
        self.models: list[PrototypeModel] = []
        self.input_list = ParamListMerged()
        self.output_list: ParamListStandardOut | ParamListStandard | None = None
        for model in models:
            self.fold_in(model)

    def fold_in(self, model: PrototypeModel) -> None:
        if not isinstance(model.input_list, ParamListStandard):
            raise TypeError("ProtoModelMerged requiere ParamListStandard de entrada")
        if self.output_list is None:
            self.output_list = model.output_list
        elif _param_list_definition(self.output_list) != _param_list_definition(
            model.output_list
        ):
            raise ValueError("Los modelos fusionados deben compartir el ParamList de salida")
        self.input_list.fold_in(model.input_list)
        self.models.append(model)

    def finalize(self) -> None:
        self.input_list.finalize()

    def select_model(self, active: ParamActive) -> PrototypeModel:
        best_model: PrototypeModel | None = None
        best_score = 500
        for model in self.models:
            score = ScoreProtoModel(True, model, active.get_num_trials())
            for trial in active.trials:
                if trial.is_active():
                    score.add_parameter(trial.locations, trial.get_size())
            model_score = score.do_score()
            if best_score is None or model_score < best_score:
                best_model = model
                best_score = model_score
                if model_score == 0:
                    break
        if best_model is None:
            raise ValueError("No hay PrototypeModel concreto en el modelo fusionado")
        return best_model

    def resolve_model(self, active: ParamActive) -> PrototypeModel:
        """Equivalent to ``FuncProto::resolveModel`` for model selection."""
        return self.select_model(active)

    def fillin_map(
        self, active: ParamActive, *, output: bool = False
    ) -> TrialMapResult | None:
        model = self.select_model(active)
        return model.fillin_map(active, output=output)


def _param_list_definition(resource: ParamListStandard) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            entry.get_group(),
            entry.get_all_groups(),
            entry.locations,
            entry.min_size,
            entry.max_size,
            entry.alignment,
            entry.get_type(),
            entry.extension,
        )
        for entry in resource.entries
    )


def _param_entry_from_cspec(
    element: ElementTree.Element,
    group: int,
    big_endian: bool,
    *,
    grouped: bool = False,
) -> ParamEntry:
    try:
        minimum_size = int(element.get("minsize", "0"), 0)
        maximum_size = int(element.get("maxsize", "0"), 0)
        alignment_text = element.get("align", element.get("size", "0"))
        alignment = int(alignment_text, 0)
    except ValueError as exc:
        raise ValueError("Atributos de tamaño inválidos en <pentry>") from exc
    if minimum_size <= 0 or maximum_size <= 0:
        raise ValueError("<pentry> requiere minsize y maxsize positivos")
    if alignment == maximum_size:
        alignment = 0
    type_class = element.get("metatype", element.get("storage", "general"))
    extension = element.get("extension", "none")
    location_element = element.find("register")
    if location_element is not None:
        location = _cspec_register_location(
            location_element.get("name", ""), maximum_size
        )
        join_pieces: tuple[StorageLocation, ...] = ()
    else:
        location_element = element.find("addr")
        if location_element is None:
            raise ValueError("<pentry> debe especificar <register> o <addr>")
        space = location_element.get("space", "")
        if space == "join":
            piece_names = [
                value
                for _key, value in sorted(
                    location_element.attrib.items(),
                    key=lambda pair: int(pair[0][5:]) if pair[0].startswith("piece") else 0,
                )
                if _key.startswith("piece")
            ]
            if not piece_names:
                raise ValueError("Un <addr space='join'> debe tener piezas")
            sizes = [_cspec_register_size(name, maximum_size) for name in piece_names]
            cursor = 0
            value_offsets = [0] * len(piece_names)
            for index in range(len(piece_names) - 1, -1, -1):
                value_offsets[index] = cursor
                cursor += sizes[index]
            join_pieces = tuple(
                _cspec_register_location(name, piece_size, value_offset=value_offset)
                for name, piece_size, value_offset in zip(
                    piece_names, sizes, value_offsets, strict=True
                )
            )
            location = join_pieces[0]
        else:
            try:
                offset = int(location_element.get("offset", "0"), 0)
            except ValueError as exc:
                raise ValueError("Offset inválido en <addr>") from exc
            kind = "stack" if space == "stack" else space or "memory"
            if kind == "stack":
                location = StorageLocation("stack", maximum_size, stack_offset=offset)
            else:
                location = StorageLocation(kind, maximum_size, stack_offset=offset)
            join_pieces = ()
    return ParamEntry(
        group=group,
        alignment=alignment,
        location=location,
        min_size=minimum_size,
        max_size=maximum_size,
        type_class=type_class,
        join_pieces=join_pieces,
        big_endian=big_endian,
        extension=extension,
        grouped=grouped,
    )


def _cspec_register_location(
    name: str, size: int, *, value_offset: int = 0
) -> StorageLocation:
    register = name.lower()
    register_offset = 0
    if register.endswith("_qa"):
        register = register[:-3]
    elif register.endswith("_qb"):
        register = register[:-3]
        register_offset = 8
    elif register.endswith("_da"):
        register = register[:-3]
    elif register.endswith("_db"):
        register = register[:-3]
        register_offset = 4
    if register in {"ah", "bh", "ch", "dh"}:
        register_offset += 1
    return StorageLocation(
        "register",
        size,
        register=register,
        value_offset=value_offset,
        register_offset=register_offset,
    )


def _cspec_register_size(name: str, fallback: int) -> int:
    register = name.lower()
    if register.endswith(("_qa", "_qb")):
        return 8
    if register.endswith(("_da", "_db")):
        return 4
    if register.startswith(("zmm",)):
        return 64
    if register.startswith(("ymm",)):
        return 32
    if register.startswith(("xmm",)):
        return 16
    if register.startswith("st"):
        return 10
    if register in {"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp"}:
        return 8
    if register.startswith("r") and register[1:2].isdigit():
        return 8
    if register in {"eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp"}:
        return 4
    if register in {"ax", "bx", "cx", "dx", "si", "di", "bp", "sp"}:
        return 2
    if register in {"al", "bl", "cl", "dl", "ah", "bh", "ch", "dh"}:
        return 1
    return fallback


def _infer_pointer_size(resource: ParamListStandard) -> int:
    if resource.pointer_size:
        return resource.pointer_size
    stack_entry = resource.stack_entry
    if stack_entry is not None and stack_entry.alignment:
        return stack_entry.alignment
    for entry in resource.entries:
        for location in entry.locations:
            if location.kind == "register" and location.size in {4, 8}:
                return location.size
    return 4


def _resource_is_big_endian(resource: ParamListStandard) -> bool:
    return bool(resource.entries and resource.entries[0].big_endian)


def _coerce_type_class(value: ParamTypeClass | str) -> ParamTypeClass:
    if isinstance(value, ParamTypeClass):
        return value
    aliases = {
        "general": ParamTypeClass.GENERAL,
        "unknown": ParamTypeClass.GENERAL,
        "float": ParamTypeClass.FLOAT,
        "pointer": ParamTypeClass.POINTER,
        "ptr": ParamTypeClass.POINTER,
        "hiddenret": ParamTypeClass.HIDDEN_RETURN,
        "hidden_return": ParamTypeClass.HIDDEN_RETURN,
        "vector": ParamTypeClass.VECTOR,
        "class1": ParamTypeClass.CLASS1,
        "class2": ParamTypeClass.CLASS2,
        "class3": ParamTypeClass.CLASS3,
        "class4": ParamTypeClass.CLASS4,
    }
    normalized = str(value).lower().replace("-", "_")
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Clase de storage desconocida: {value}") from exc


def _storage_class_for_type(data_type: CType) -> ParamTypeClass:
    base = data_type.unwrapped()
    if base.kind == TypeKind.FLOAT:
        return ParamTypeClass.FLOAT
    if base.kind == TypeKind.POINTER:
        return ParamTypeClass.POINTER
    if base.kind == TypeKind.VECTOR:
        return ParamTypeClass.VECTOR
    return ParamTypeClass.GENERAL


def _coerce_parameter(
    parameter: FunctionParameter | CType | str,
    index: int,
    types: TypeDatabase | None,
) -> tuple[str, CType, FunctionParameter | None]:
    if isinstance(parameter, FunctionParameter):
        data_type = parameter.data_type
        if data_type is None:
            if types is None:
                raise ValueError(f"Falta TypeDatabase para {parameter.type_name}")
            data_type = types.resolve(parameter.type_name)
        return parameter.name, data_type, parameter
    if isinstance(parameter, CType):
        return f"param_{index + 1}", parameter, None
    if isinstance(parameter, str):
        if types is None:
            raise ValueError(f"Falta TypeDatabase para {parameter}")
        data_type = types.resolve(parameter)
        return f"param_{index + 1}", data_type, None
    raise TypeError(f"Tipo de parámetro no admitido: {type(parameter).__name__}")


def _space_key(location: StorageLocation) -> tuple[str, ...]:
    from .paramid import _location_space_key

    return _location_space_key(location)


def _offset(location: StorageLocation) -> int | None:
    from .paramid import _location_start_offset

    return _location_start_offset(location)


def _joined_containment_offset(
    entry: ParamEntry, locations: Sequence[StorageLocation], size: int
) -> int | None:
    offsets: list[int] = []
    for location in locations:
        offset = entry.justified_contain(location, location.size)
        if offset < 0:
            return None
        offsets.append(offset)
    order = sorted(
        zip(locations, offsets), key=lambda item: item[0].value_offset
    )
    cursor = 0
    first_offset = order[0][1]
    for location, offset in order:
        if offset != first_offset + cursor:
            return None
        cursor += location.size
    if cursor < size:
        return None
    return first_offset


def _contiguous(
    *,
    high_location: StorageLocation,
    high_size: int,
    low_location: StorageLocation,
    low_size: int,
) -> bool:
    if _space_key(high_location) != _space_key(low_location):
        return False
    high_offset = _offset(high_location)
    low_offset = _offset(low_location)
    return (
        high_offset is not None
        and low_offset == high_offset + high_size
        and low_size > 0
    )


__all__ = [
    "CspecModelRule",
    "ParamList",
    "ParamListMerged",
    "ParamListRegister",
    "ParamListRegisterOut",
    "ParamListStandard",
    "ParamListStandardOut",
    "ParamListType",
    "ProtoModelMerged",
    "PrototypeModel",
    "ScoreProtoModel",
    "TrialMapResult",
]
