from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path
from xml.etree import ElementTree as ET

from minighidra.model import FunctionParameter, FunctionSignature
from minighidra.paramid import (
    ParamActive,
    ParamEntry,
    ParamEntryFlag,
    ParamIDAnalysis,
    ParamTypeClass,
    PcodeLikeGraph,
)
from minighidra.paramlist import (
    CspecModelRule,
    ParamListRegister,
    ParamListStandard,
    ParamListStandardOut,
    PrimitiveExtractor,
    PrototypeModel,
    ScoreProtoModel,
)
from minighidra.prototype import PrototypeResolver
from minighidra.ssa import SSAFunction, SSAOperand, SSAReturnSite, SSAValue
from minighidra.typesys import CType, StorageLocation, TypeDatabase, TypeKind

ROOT = Path(__file__).resolve().parents[2]
CSPECS = ROOT / "Ghidra" / "Processors" / "x86" / "data" / "languages"


class ParamEntryTests(unittest.TestCase):
    def test_flags_storage_class_and_range_containment(self) -> None:
        entry = ParamEntry(
            group=3,
            location=StorageLocation("stack", 16, stack_offset=8),
            alignment=8,
            min_size=1,
            max_size=16,
            type_class="float",
        )
        entry.flags = (
            ParamEntryFlag.REVERSE_STACK
            | ParamEntryFlag.SMALL_SIZE_ZERO_EXTEND
            | ParamEntryFlag.FIRST_STORAGE
        )
        self.assertEqual(entry.flags & ParamEntryFlag.REVERSE_STACK, ParamEntryFlag.REVERSE_STACK)
        self.assertEqual(entry.get_type(), ParamTypeClass.FLOAT)
        self.assertTrue(entry.is_first_in_class())
        self.assertEqual(
            entry.justified_contain(StorageLocation("stack", 4, stack_offset=8), 4),
            0,
        )
        self.assertEqual(
            entry.assumed_extension(
                StorageLocation("stack", 4, stack_offset=8), 4
            )[0],
            "INT_ZEXT",
        )
        self.assertTrue(
            entry.contains(
                ParamEntry(
                    group=8,
                    location=StorageLocation("stack", 4, stack_offset=8),
                    alignment=0,
                    min_size=1,
                    max_size=4,
                )
            )
        )

    def test_slot_allocation_alignment_endian_and_reverse_stack(self) -> None:
        entry = ParamEntry(
            group=4,
            location=StorageLocation("stack", 12, stack_offset=0),
            alignment=4,
            min_size=1,
            max_size=12,
        )
        first, next_slot = entry.get_addr_by_slot(0, 3, 1)
        second, next_slot = entry.get_addr_by_slot(next_slot, 4, 4)
        self.assertEqual(first.stack_offset, 0)
        self.assertEqual(second.stack_offset, 4)
        self.assertEqual(next_slot, 2)

        reverse = ParamEntry(
            group=4,
            location=StorageLocation("stack", 16, stack_offset=0),
            alignment=8,
            min_size=1,
            max_size=16,
            reverse_stack=True,
        )
        high, next_slot = reverse.get_addr_by_slot(0, 8)
        low, _ = reverse.get_addr_by_slot(next_slot, 8)
        self.assertEqual(high.stack_offset, 8)
        self.assertEqual(low.stack_offset, 0)
        self.assertEqual(reverse.get_slot(high, 7), 4)
        self.assertEqual(reverse.get_slot(low, 7), 5)

        big_endian = ParamEntry(
            group=1,
            location=StorageLocation("stack", 16, stack_offset=0),
            alignment=8,
            min_size=1,
            max_size=16,
            big_endian=True,
        )
        right_justified, _ = big_endian.get_addr_by_slot(0, 4)
        self.assertEqual(right_justified.stack_offset, 4)
        big_endian_exclusion = ParamEntry(
            group=1,
            location=StorageLocation("stack", 8, stack_offset=16),
            alignment=0,
            min_size=1,
            max_size=8,
            big_endian=True,
        )
        narrow_right_justified, _ = big_endian_exclusion.get_addr_by_slot(0, 4)
        self.assertEqual(narrow_right_justified.stack_offset, 20)
        self.assertEqual(
            big_endian.justified_contain(
                StorageLocation("stack", 4, stack_offset=4), 4
            ),
            0,
        )

    def test_float_extension_returns_logical_join_with_full_physical_piece(self) -> None:
        entry = ParamEntry(
            group=0,
            location=StorageLocation("register", 10, register="st0"),
            alignment=0,
            min_size=4,
            max_size=10,
            type_class=ParamTypeClass.FLOAT,
            big_endian=True,
            extension="float",
        )

        float_location, next_slot = entry.get_addr_by_slot(
            0, 4, justify_right=True
        )

        self.assertEqual(next_slot, 1)
        self.assertIsNotNone(float_location)
        self.assertEqual((float_location.register, float_location.size), ("st0", 4))
        record = float_location.join_record
        self.assertIsNotNone(record)
        self.assertTrue(record.float_extension)
        self.assertEqual(record.logical_size, 4)
        self.assertEqual(
            [(piece.register, piece.size, piece.register_offset) for piece in record.pieces],
            [("st0", 10, 0)],
        )
        self.assertEqual(
            record.get_equivalent_piece(3),
            (record.pieces[0], 3),
        )
        self.assertIsNone(record.get_equivalent_piece(4))
        self.assertEqual(float_location.physical_pieces, record.pieces)

        full_precision, _ = entry.get_addr_by_slot(0, 10)
        self.assertIsNone(full_precision.join_record)
        self.assertEqual(full_precision.size, 10)

    def test_cspec_float_extension_marks_output_join_record(self) -> None:
        types = TypeDatabase(32, "ELF")
        model = PrototypeModel.from_cspec(CSPECS / "x86gcc.cspec", "__cdecl")
        self.assertEqual(model.output_list.entries[0].extension, "float")

        for type_name, logical_size in (("float", 4), ("double", 8)):
            with self.subTest(type_name=type_name):
                assigned = model.output_list.assign_map(
                    (types.get(type_name),), types=types
                )[0]
                location = assigned.locations[0]
                self.assertEqual(location.size, logical_size)
                self.assertEqual(location.register, "st0")
                self.assertIsNotNone(location.join_record)
                self.assertEqual(location.join_record.logical_size, logical_size)
                self.assertEqual(location.join_record.pieces[0].size, 10)

        signature = model.assign_signature(
            FunctionSignature(
                "float", (), "__cdecl", return_data_type=types.get("float")
            ),
            types=types,
        )
        self.assertTrue(signature.return_locations[0].join_record.float_extension)
        serialized = asdict(signature)
        serialized_record = serialized["return_locations"][0]["join_record"]
        self.assertEqual(serialized_record["logical_size"], 4)
        self.assertEqual(serialized_record["pieces"][0]["size"], 10)

    def test_overlap_groups_and_storage_container_classification(self) -> None:
        narrow = ParamEntry(
            0,
            location=StorageLocation("register", 4, register="edi"),
            alignment=0,
            max_size=4,
        )
        wide = ParamEntry(
            1,
            location=StorageLocation("register", 8, register="rdi"),
            alignment=0,
            max_size=8,
        )
        wide.resolve_overlap((narrow,))
        self.assertEqual(wide.get_all_groups(), (0,))
        self.assertTrue(wide.is_overlap())
        self.assertTrue(wide.is_param_check_high())

        aligned = ParamEntry(
            0,
            location=StorageLocation("stack", 16, stack_offset=0),
            alignment=8,
            max_size=16,
        )
        resources = ParamListStandard((aligned,), split_float=False)
        contained = StorageLocation("stack", 4, stack_offset=4)
        self.assertEqual(resources.characterize_as_param(contained, 4), 1)
        container = resources.unjustified_container(contained, 4)
        self.assertEqual((container.stack_offset, container.size), (0, 8))

        exclusions = ParamListStandard(
            (
                ParamEntry(
                    0,
                    location=StorageLocation("stack", 4, stack_offset=0),
                    alignment=0,
                    max_size=4,
                ),
                ParamEntry(
                    1,
                    location=StorageLocation("stack", 8, stack_offset=4),
                    alignment=0,
                    max_size=8,
                ),
            ),
            split_float=False,
        )
        largest = exclusions.get_biggest_contained_param(
            StorageLocation("stack", 12, stack_offset=0), 12
        )
        self.assertEqual((largest.stack_offset, largest.size), (4, 8))

        grouped = ParamEntry(
            2,
            group_set=(2, 4, 6),
            location=StorageLocation("register", 16, register="xmm0"),
            alignment=0,
            max_size=16,
        )
        self.assertEqual(grouped.get_slot(grouped.location, 0), 2)
        self.assertEqual(grouped.get_slot(grouped.location, 1), 6)

    def test_join_entry_resolves_groups_and_expands_logical_value(self) -> None:
        low = ParamEntry(
            group=1,
            location=StorageLocation("register", 4, register="eax"),
            alignment=0,
            min_size=1,
            max_size=4,
        )
        high = ParamEntry(
            group=2,
            location=StorageLocation("register", 4, register="edx"),
            alignment=0,
            min_size=1,
            max_size=4,
        )
        joined = ParamEntry(
            group=3,
            location=StorageLocation("register", 4, register="edx"),
            join_pieces=(
                StorageLocation("register", 4, register="edx", value_offset=4),
                StorageLocation("register", 4, register="eax", value_offset=0),
            ),
            alignment=0,
            min_size=5,
            max_size=8,
        )
        # The join resolves against prior physical resources in source order.
        resources = ParamListStandard((low, high, joined), split_float=False)
        self.assertEqual(joined.get_all_groups(), (1, 2))
        pieces, next_slot = joined.get_locations_by_slot(0, 8)
        self.assertEqual(next_slot, 1)
        self.assertEqual(
            [(piece.register, piece.size, piece.value_offset) for piece in pieces],
            [("eax", 4, 0), ("edx", 4, 4)],
        )
        self.assertTrue(resources.possible_param(pieces[0], 4))


class ParamListTests(unittest.TestCase):
    def test_primitive_extractor_expands_and_refines_aggregate_layouts(self) -> None:
        types = TypeDatabase(64, "ELF")
        float_or_integer = types.union(
            "FloatOrInteger", [("fraction", "float"), ("bits", "int")]
        )
        refined_union = PrimitiveExtractor(float_or_integer)
        self.assertTrue(refined_union.is_valid())
        self.assertEqual(
            [
                (
                    primitive.data_type.unwrapped().kind,
                    primitive.offset,
                    primitive.size,
                )
                for primitive in refined_union.primitives
            ],
            [(types.get("int").kind, 0, 4)],
        )

        pair = types.structure(
            "IntegerPair", [("low", "int"), ("high", "int")]
        )
        integer_or_pair = types.union(
            "IntegerOrPair", [("wide", "long long"), ("pair", pair)]
        )
        wide_refinement = PrimitiveExtractor(integer_or_pair)
        self.assertTrue(wide_refinement.is_valid())
        self.assertEqual(
            [
                (primitive.data_type.name, primitive.offset)
                for primitive in wide_refinement.primitives
            ],
            [("long long", 0)],
        )

        nested = types.structure(
            "NestedUnion", [("common", float_or_integer), ("value", "double", 8)]
        )
        nested_extraction = PrimitiveExtractor(nested)
        self.assertTrue(nested_extraction.is_valid())
        self.assertFalse(nested_extraction.contains_holes())
        self.assertEqual(
            [(primitive.offset, primitive.size) for primitive in nested_extraction.primitives],
            [(0, 4), (8, 8)],
        )

        array_extraction = PrimitiveExtractor(types.array("int", 3))
        self.assertTrue(array_extraction.is_valid())
        self.assertEqual(
            [primitive.offset for primitive in array_extraction.primitives],
            [0, 4, 8],
        )
        odd_element = types.structure(
            "OddArrayElement",
            [("word", "int"), ("tail", "char")],
            size=5,
            alignment=4,
        )
        self.assertEqual(odd_element.align_size, 8)
        odd_array = types.array(odd_element, 2)
        self.assertEqual(odd_array.size, 16)
        odd_array_extraction = PrimitiveExtractor(odd_array)
        self.assertTrue(odd_array_extraction.is_valid())
        self.assertEqual(
            [primitive.offset for primitive in odd_array_extraction.primitives],
            [0, 4, 8, 12],
        )

        hole = types.structure(
            "PrimitiveHole", [("first", "int"), ("later", "int", 8)], size=16
        )
        hole_extraction = PrimitiveExtractor(hole)
        self.assertTrue(hole_extraction.is_valid())
        self.assertTrue(hole_extraction.contains_holes())

        packed = types.structure(
            "MisalignedPrimitive",
            [("prefix", "char"), ("word", "int", 1)],
            packed=True,
            size=5,
        )
        packed_extraction = PrimitiveExtractor(packed)
        self.assertTrue(packed_extraction.is_valid())
        self.assertFalse(packed_extraction.is_aligned())
        self.assertTrue(packed_extraction.contains_holes())

        unknown_extraction = PrimitiveExtractor(
            CType("unknown4", TypeKind.UNKNOWN, 4, 4)
        )
        self.assertTrue(unknown_extraction.is_valid())
        self.assertTrue(unknown_extraction.contains_unknown)

        partial_overlap = types.structure(
            "PartialOverlap", [("first", "int"), ("second", "int")]
        )
        invalid_union = types.union(
            "InvalidRefinement",
            [
                ("normal", partial_overlap),
                ("misaligned", types.structure("Misaligned", [("value", "int", 2)], size=8)),
            ],
        )
        invalid_extraction = PrimitiveExtractor(invalid_union)
        self.assertFalse(invalid_extraction.is_valid())

    def test_dual_class_tiles_commit_or_roll_back_transactionally(self) -> None:
        types = TypeDatabase(64, "ELF")
        mixed = types.structure(
            "DualClassMixed", [("fraction", "double"), ("whole", "int", 8)]
        )
        entries = (
            ParamEntry(
                0,
                location=StorageLocation("register", 8, register="rdi"),
                alignment=0,
                max_size=8,
                type_class=ParamTypeClass.GENERAL,
            ),
            ParamEntry(
                1,
                location=StorageLocation("register", 8, register="xmm0"),
                alignment=0,
                max_size=8,
                type_class=ParamTypeClass.FLOAT,
            ),
            ParamEntry(
                2,
                location=StorageLocation("stack", 32, stack_offset=8),
                alignment=8,
                max_size=32,
            ),
        )
        spilling = ParamListStandard(
            entries,
            model_rules=(
                CspecModelRule("any", "join_dual_class", consume_from_stack=True),
            ),
            split_float=False,
        )
        status = [-1, 0, 0]

        locations = spilling.assign_address(mixed, status, types=types)

        self.assertEqual(
            [(location.kind, location.register, location.stack_offset, location.value_offset)
             for location in locations],
            [("register", "xmm0", None, 0), ("stack", None, 8, 8)],
        )
        self.assertEqual(status, [-1, -1, 1])

        no_spill = ParamListStandard(
            entries[:2],
            model_rules=(
                CspecModelRule("any", "join_dual_class", consume_from_stack=False),
            ),
            split_float=False,
        )
        exhausted_status = [-1, 0]
        self.assertIsNone(
            no_spill.assign_address(mixed, exhausted_status, types=types)
        )
        self.assertEqual(exhausted_status, [-1, 0])

        float_pair = types.structure(
            "DualFloatPair", [("first", "float"), ("second", "float")]
        )
        fill_alternate = ParamListStandard(
            entries[:2],
            model_rules=(
                CspecModelRule("any", "join_dual_class", fill_alternate=True),
            ),
            split_float=False,
        )
        filled = fill_alternate.assign_address(
            float_pair, fill_alternate.allocate_status(), types=types
        )
        self.assertEqual(
            [(location.register, location.size, location.value_offset) for location in filled],
            [("rdi", 8, 0)],
        )

    def test_paramid_analysis_fills_in_selected_paramlist(self) -> None:
        resources = ParamListStandard(
            (
                ParamEntry(
                    0,
                    location=StorageLocation("register", 8, register="rdi"),
                    alignment=0,
                    max_size=8,
                ),
            ),
            resource_start=(0, 1),
            split_float=False,
        )
        analysis = ParamIDAnalysis(PcodeLikeGraph())
        trial = analysis.input_active.register_trial(
            "arg", 32, (StorageLocation("register", 4, register="rdi"),)
        )
        trial.mark_active()

        result = analysis.fillin_param_list(resources)

        self.assertEqual(result.matched, ("arg",))
        self.assertTrue(trial.used)
        empty_output = analysis.fillin_param_list(resources, output=True)
        self.assertEqual(empty_output.matched, ())
        self.assertEqual(analysis.output_active.get_num_trials(), 0)

    def test_paramid_output_trials_fill_in_selected_output_paramlist(self) -> None:
        ssa = SSAFunction(
            address=0x1000,
            blocks=(),
            values=(SSAValue(0, "input", "rax", 64),),
            phis=(),
            operations=(),
            entry_blocks=(),
            live_in=(),
            live_out=(),
            return_sites=(
                SSAReturnSite(
                    0,
                    0x1001,
                    0,
                    (("rax", SSAOperand("register", 64, values=(0,))),),
                ),
            ),
        )
        analysis = ParamIDAnalysis(PcodeLikeGraph.from_ssa(ssa, {}))
        analysis.analyze()
        output_list = ParamListStandardOut(
            (
                ParamEntry(
                    0,
                    location=StorageLocation("register", 8, register="rax"),
                    alignment=0,
                    max_size=8,
                ),
            ),
            split_float=False,
        )

        analysis.fillin_param_list(output_list, output=True)

        trial = analysis.output_active.get_trial("return_rax")
        self.assertEqual(
            analysis.output_measurements["return_rax"].rank_label,
            "this-function-parameter",
        )
        self.assertEqual(trial.locations[0].register, "rax")
        self.assertEqual(trial.locations[0].size, 8)
        self.assertTrue(trial.used)
        self.assertIs(trial.entry, output_list.entries[0])

    def test_build_trial_map_inserts_missing_exclusion_group(self) -> None:
        resources = ParamListStandard(
            (
                ParamEntry(
                    0,
                    location=StorageLocation("register", 8, register="rdi"),
                    alignment=0,
                    max_size=8,
                ),
                ParamEntry(
                    1,
                    location=StorageLocation("register", 8, register="rsi"),
                    alignment=0,
                    max_size=8,
                ),
            ),
            resource_start=(0, 2),
            split_float=False,
        )
        active = ParamActive()
        trial = active.register_trial(
            "param_2", 64, (StorageLocation("register", 8, register="rsi"),)
        )
        trial.mark_active()

        result = resources.fillin_map(active)

        self.assertEqual(result.matched, ("param_2",))
        self.assertEqual(result.unmatched, ())
        self.assertEqual(result.inserted_unreferenced, ("__unreferenced_0",))
        self.assertEqual(
            [(item.name, item.entry.get_group()) for item in active.trials],
            [("__unreferenced_0", 0), ("param_2", 1)],
        )
        self.assertTrue(active.get_trial("param_2").used)
        self.assertTrue(active.get_trial("__unreferenced_0").unreferenced)

    def test_build_trial_map_inserts_hole_inside_stack_range(self) -> None:
        entry = ParamEntry(
            0,
            location=StorageLocation("stack", 24, stack_offset=0),
            alignment=8,
            min_size=1,
            max_size=24,
        )
        resources = ParamListStandard((entry,), split_float=False)
        active = ParamActive()
        trial = active.register_trial(
            "stack_arg_2", 64, (StorageLocation("stack", 8, stack_offset=8),)
        )
        trial.mark_active()

        result = resources.fillin_map(active)

        self.assertEqual(result.inserted_unreferenced, ("__unreferenced_0_0",))
        self.assertEqual(
            [(item.name, item.locations[0].stack_offset) for item in active.trials],
            [("__unreferenced_0_0", 0), ("stack_arg_2", 8)],
        )
        self.assertTrue(active.get_trial("stack_arg_2").used)

    def test_exclusion_pass_keeps_only_one_active_alternative(self) -> None:
        resources = ParamListStandard(
            (
                ParamEntry(
                    0,
                    location=StorageLocation("register", 8, register="rdi"),
                    alignment=0,
                    max_size=8,
                ),
                ParamEntry(
                    0,
                    location=StorageLocation("register", 8, register="xmm0"),
                    alignment=0,
                    max_size=8,
                    type_class="float",
                ),
            ),
            resource_start=(0, 1),
            split_float=False,
        )
        active = ParamActive()
        integer = active.register_trial(
            "int_trial", 64, (StorageLocation("register", 8, register="rdi"),)
        )
        floating = active.register_trial(
            "float_trial", 64, (StorageLocation("register", 8, register="xmm0"),)
        )
        integer.mark_active()
        floating.mark_active()

        resources.fillin_map(active)

        self.assertTrue(integer.used)
        self.assertTrue(integer.active)
        self.assertTrue(floating.definitely_not_used)
        self.assertFalse(floating.active)

    def test_force_no_use_makes_later_groups_inactive(self) -> None:
        active = ParamActive()
        first = active.register_trial("first", 32)
        second = active.register_trial("second", 32)
        first.set_entry(ParamEntry(0, alignment=0, max_size=4))
        second.set_entry(ParamEntry(1, alignment=0, max_size=4))
        first.mark_no_use()
        second.mark_active()

        ParamListStandard.force_no_use(active, 0, 2)

        self.assertFalse(second.active)
        self.assertFalse(second.definitely_not_used)
        self.assertTrue(second.checked)

    def test_check_join_and_split_use_model_ranges(self) -> None:
        resources = ParamListStandard(
            (
                ParamEntry(
                    0,
                    location=StorageLocation("stack", 16, stack_offset=0),
                    alignment=4,
                    min_size=1,
                    max_size=16,
                ),
            ),
            split_float=False,
        )
        self.assertTrue(
            resources.check_join(
                StorageLocation("stack", 4, stack_offset=0),
                4,
                StorageLocation("stack", 4, stack_offset=4),
                4,
            )
        )
        self.assertTrue(
            resources.check_split(
                StorageLocation("stack", 8, stack_offset=0), 8, 4
            )
        )
        self.assertEqual(
            resources.possible_param_with_slot(
                StorageLocation("stack", 8, stack_offset=4), 8
            ),
            (1, 2),
        )

    def test_register_paramlist_marks_unmatched_trials_without_hole_fill(self) -> None:
        resources = ParamListRegister(
            (
                ParamEntry(
                    0,
                    location=StorageLocation("register", 8, register="rdi"),
                    alignment=0,
                    max_size=8,
                ),
            ),
            split_float=False,
        )
        active = ParamActive()
        trial = active.register_trial(
            "not_a_parameter", 64, (StorageLocation("register", 8, register="rsi"),)
        )
        trial.mark_active()

        resources.fillin_map(active)

        self.assertTrue(trial.definitely_not_used)
        self.assertIsNone(trial.entry)
        self.assertEqual(active.get_num_trials(), 1)

    def test_output_fillin_partial_trials_and_remainder_filter(self) -> None:
        partial_resources = ParamListStandardOut(
            (
                ParamEntry(
                    0,
                    location=StorageLocation("stack", 4, stack_offset=0),
                    alignment=0,
                    min_size=4,
                    max_size=4,
                ),
            ),
            split_float=False,
            use_fillin_fallback=True,
        )
        partial_active = ParamActive()
        low = partial_active.register_trial(
            "low", 16, (StorageLocation("stack", 2, stack_offset=0),)
        )
        high = partial_active.register_trial(
            "high", 16, (StorageLocation("stack", 2, stack_offset=2),)
        )
        low.mark_active()
        high.mark_active()

        partial_resources.fillin_map(partial_active)

        self.assertTrue(low.used)
        self.assertTrue(high.used)
        self.assertIs(low.entry, partial_resources.entries[0])
        self.assertIs(high.entry, partial_resources.entries[0])

        regular_resources = ParamListStandardOut(
            (
                ParamEntry(
                    0,
                    location=StorageLocation("register", 8, register="rax"),
                    alignment=0,
                    max_size=8,
                ),
                ParamEntry(
                    1,
                    location=StorageLocation("register", 8, register="rdx"),
                    alignment=0,
                    max_size=8,
                ),
            ),
            resource_start=(0, 2),
            split_float=False,
            use_fillin_fallback=False,
        )
        active = ParamActive()
        remainder = active.register_trial(
            "remainder", 64, (StorageLocation("register", 8, register="rdx"),)
        )
        remainder.rem_formed = True
        remainder.mark_active()

        regular_resources.fillin_map(active)

        self.assertTrue(remainder.definitely_not_used)
        self.assertIsNone(remainder.entry)

    def test_standard_x86_cspec_assigns_joined_return_and_hidden_sret(self) -> None:
        types = TypeDatabase(32, "ELF")
        model = PrototypeModel.from_cspec(CSPECS / "x86gcc.cspec", "__cdecl")
        joined = model.output_list.assign_map((types.get("long long"),), types=types)[0]
        self.assertEqual(
            [(piece.register, piece.size, piece.value_offset) for piece in joined.locations],
            [("eax", 4, 0), ("edx", 4, 4)],
        )

        result_type = types.structure(
            "LargeResult", [("a", "long long"), ("b", "long long")]
        )
        signature = FunctionSignature(
            result_type.name,
            (FunctionParameter("arg", "int", types.get("int")),),
            "__cdecl",
            return_data_type=result_type,
        )
        assigned = model.assign_signature(signature, types=types)
        self.assertTrue(assigned.hidden_return)
        self.assertEqual(assigned.hidden_return_location.stack_offset, 4)
        self.assertEqual(assigned.parameters[0].locations[0].stack_offset, 8)
        self.assertEqual(assigned.return_locations[0].register, "eax")

    def test_hidden_return_special_and_normal_parameter_strategies(self) -> None:
        types = TypeDatabase(64, "ELF")
        result_type = types.structure(
            "HiddenResult", [("a", "long long"), ("b", "long long"), ("c", "int")]
        )
        signature = FunctionSignature(
            result_type.name,
            (FunctionParameter("arg", "int", types.get("int")),),
            "custom",
            return_data_type=result_type,
        )
        output = ParamListStandardOut(
            (
                ParamEntry(
                    0,
                    location=StorageLocation("register", 8, register="rax"),
                    alignment=0,
                    max_size=8,
                ),
            ),
            model_rules=(
                CspecModelRule(
                    "struct", "hidden_return", hidden_strategy="special"
                ),
            ),
            split_float=False,
            use_fillin_fallback=False,
        )
        special_input = ParamListStandard(
            (
                ParamEntry(
                    0,
                    location=StorageLocation("register", 8, register="rsi"),
                    alignment=0,
                    max_size=8,
                    type_class=ParamTypeClass.HIDDEN_RETURN,
                ),
                ParamEntry(
                    1,
                    location=StorageLocation("register", 8, register="rdi"),
                    alignment=0,
                    max_size=8,
                ),
                ParamEntry(
                    2,
                    location=StorageLocation("register", 8, register="rdx"),
                    alignment=0,
                    max_size=8,
                ),
            ),
            split_float=False,
        )
        special = PrototypeModel("special", special_input, output)
        assigned_special = special.assign_signature(signature, types=types)
        self.assertEqual(assigned_special.hidden_return_location.register, "rsi")
        self.assertEqual(assigned_special.parameters[0].locations[0].register, "rdi")

        voidlocked_output = ParamListStandardOut(
            output.entries,
            model_rules=(
                CspecModelRule(
                    "struct",
                    "hidden_return",
                    hidden_strategy="special",
                    hidden_voidlock=True,
                ),
            ),
            split_float=False,
            use_fillin_fallback=False,
        )
        voidlocked = PrototypeModel("voidlocked", special_input, voidlocked_output)
        assigned_voidlocked = voidlocked.assign_signature(signature, types=types)
        self.assertTrue(assigned_voidlocked.hidden_return)
        self.assertEqual(assigned_voidlocked.return_locations, ())

        normal_input = ParamListStandard(
            (
                ParamEntry(
                    0,
                    location=StorageLocation("register", 8, register="rdi"),
                    alignment=0,
                    max_size=8,
                ),
                ParamEntry(
                    1,
                    location=StorageLocation("register", 8, register="rsi"),
                    alignment=0,
                    max_size=8,
                ),
            ),
            split_float=False,
        )
        normal_output = ParamListStandardOut(
            output.entries,
            model_rules=(
                CspecModelRule(
                    "struct", "hidden_return", hidden_strategy="normalparam"
                ),
            ),
            split_float=False,
            use_fillin_fallback=False,
        )
        normal = PrototypeModel("normal", normal_input, normal_output)
        assigned_normal = normal.assign_signature(signature, types=types)
        self.assertEqual(assigned_normal.hidden_return_location.register, "rdi")
        self.assertEqual(assigned_normal.parameters[0].locations[0].register, "rsi")

    def test_x86_thiscall_places_this_before_hidden_return_pointer(self) -> None:
        types = TypeDatabase(32, "PE")
        model = PrototypeModel.from_cspec(CSPECS / "x86win.cspec", "__thiscall")
        result_type = types.structure(
            "ThiscallResult", [("first", "long long"), ("second", "long long")]
        )
        signature = FunctionSignature(
            result_type.name,
            (
                FunctionParameter(
                    "this", "void *", types.pointer("void"), implicit=True
                ),
                FunctionParameter("value", "int", types.get("int")),
            ),
            "__thiscall",
            return_data_type=result_type,
        )

        assigned = model.assign_signature(signature, types=types)

        self.assertTrue(assigned.hidden_return)
        self.assertEqual(assigned.parameters[0].locations[0].register, "ecx")
        self.assertEqual(assigned.hidden_return_location.stack_offset, 4)
        self.assertEqual(assigned.parameters[1].locations[0].stack_offset, 8)
        self.assertEqual(assigned.parameters[1].locations[0].value_offset, 0)

    def test_sysv_cspec_dual_class_rules_assign_registers_or_stack(self) -> None:
        types = TypeDatabase(64, "ELF")
        model = PrototypeModel.from_cspec(
            CSPECS / "x86-64-gcc.cspec", "__stdcall"
        )
        self.assertIn(
            "join_dual_class", [rule.action for rule in model.input_list.model_rules]
        )
        self.assertEqual(model.input_list.unsupported_rules, ())
        parameters = model.input_list.assign_map(
            (
                types.get("int"),
                types.get("double"),
                types.get("long long"),
            ),
            types=types,
        )
        self.assertEqual(parameters[0].locations[0].register, "rdi")
        self.assertEqual(parameters[1].locations[0].register, "xmm0")
        self.assertEqual(parameters[2].locations[0].register, "rsi")

        mixed = types.structure(
            "Mixed", [("fraction", "double"), ("whole", "int", 8)], size=16
        )
        mixed_parameter = model.input_list.assign_map((mixed,), types=types)[0]
        self.assertEqual(
            [(piece.register, piece.value_offset) for piece in mixed_parameter.locations],
            [("xmm0", 0), ("rdi", 8)],
        )
        padded_hole = types.structure(
            "Hole", [("first", "int"), ("second", "int", 8)], size=16
        )
        spilled = model.input_list.assign_map((padded_hole,), types=types)[0]
        self.assertEqual(spilled.locations[0].kind, "stack")

        return_location = model.output_list.assign_map(
            (types.get("long long"),), types=types
        )[0]
        self.assertEqual(return_location.locations[0].register, "rax")
        mixed_signature = FunctionSignature(
            mixed.name, (), "__stdcall", return_data_type=mixed
        )
        mixed_return = model.assign_signature(mixed_signature, types=types)
        self.assertEqual(
            [(piece.register, piece.value_offset) for piece in mixed_return.return_locations],
            [("xmm0", 0), ("rax", 8)],
        )

    def test_windows_x64_group_consumption_and_hidden_return_order(self) -> None:
        types = TypeDatabase(64, "PE")
        model = PrototypeModel.from_cspec(
            CSPECS / "x86-64-win.cspec", "__fastcall"
        )
        parameters = model.input_list.assign_map(
            (types.get("double"), types.get("int")), types=types
        )
        self.assertEqual(parameters[0].locations[0].register, "xmm0")
        self.assertEqual(parameters[1].locations[0].register, "rdx")
        large_argument = types.structure(
            "WinArgument", [("first", "long long"), ("second", "long long")]
        )
        indirect_argument = model.input_list.assign_map(
            (large_argument,), types=types
        )[0]
        self.assertEqual(indirect_argument.locations[0].register, "rcx")
        self.assertTrue(indirect_argument.locations[0].indirect)

        large_result = types.structure(
            "WinResult", [("a", "long long"), ("b", "int")]
        )
        signature = FunctionSignature(
            large_result.name,
            (FunctionParameter("this", "void *", types.pointer("void"), implicit=True),),
            "__fastcall",
            return_data_type=large_result,
        )
        assigned = model.assign_signature(signature, types=types)
        self.assertTrue(assigned.hidden_return)
        self.assertEqual(assigned.hidden_return_location.register, "rcx")
        self.assertEqual(assigned.parameters[0].locations[0].register, "rdx")

    def test_merged_model_scoring_selects_matching_cspec(self) -> None:
        resolver = PrototypeResolver(TypeDatabase(32, "PE"), 32)
        merged = resolver.load_merged_prototype_model(
            str(CSPECS / "x86win.cspec"), "__fastcall/__thiscall/__stdcall"
        )
        active = ParamActive()
        trial = active.register_trial(
            "param_1", 32, (StorageLocation("register", 4, register="ecx"),)
        )
        trial.mark_active()

        selected, fillin = resolver.resolve_param_trials(active, merged)

        self.assertEqual(selected.name, "__fastcall")
        self.assertIsNotNone(fillin)
        self.assertEqual(fillin.matched, ("param_1",))
        score = ScoreProtoModel(True, selected, 1)
        score.add_parameter(trial.locations, trial.get_size())
        self.assertEqual(score.do_score(), 0)
        self.assertEqual(score.get_num_mismatch(), 0)

    def test_score_proto_model_penalizes_holes_overlaps_and_mismatches(self) -> None:
        resources = ParamListStandard(
            (
                ParamEntry(
                    0,
                    location=StorageLocation("register", 8, register="rdi"),
                    alignment=0,
                    max_size=8,
                ),
                ParamEntry(
                    1,
                    location=StorageLocation("register", 8, register="rsi"),
                    alignment=0,
                    max_size=8,
                ),
            ),
            split_float=False,
        )
        model = PrototypeModel("test", resources, ParamListStandardOut())

        hole = ScoreProtoModel(True, model)
        hole.add_parameter(StorageLocation("register", 8, register="rsi"), 8)
        self.assertEqual(hole.do_score(), 16)

        overlap = ScoreProtoModel(True, model)
        overlap.add_parameter(StorageLocation("register", 8, register="rdi"), 8)
        overlap.add_parameter(StorageLocation("register", 8, register="rdi"), 8)
        self.assertEqual(overlap.do_score(), 20)

        mismatch = ScoreProtoModel(True, model)
        mismatch.add_parameter(StorageLocation("register", 8, register="rax"), 8)
        self.assertEqual(mismatch.do_score(), 20)
        self.assertEqual(mismatch.get_num_mismatch(), 1)

    def test_all_local_x86_cspec_prototype_resources_load(self) -> None:
        for filename in (
            "x86gcc.cspec",
            "x86win.cspec",
            "x86-64-gcc.cspec",
            "x86-64-win.cspec",
        ):
            path = CSPECS / filename
            root = ET.parse(path).getroot()
            for prototype in root.findall(".//prototype"):
                with self.subTest(cspec=filename, model=prototype.get("name")):
                    model = PrototypeModel.from_cspec(path, prototype.get("name", ""))
                    self.assertTrue(model.input_list.entries)
                    self.assertTrue(model.output_list.entries)

    def test_rule_filter_tracks_local_cspec_model_limitations(self) -> None:
        element = ET.Element("rule")
        ET.SubElement(element, "datatype", {"name": "struct", "sizes": "3,5,6,7"})
        ET.SubElement(element, "convert_to_ptr")
        rule = CspecModelRule.from_element(element, "input")
        types = TypeDatabase(64, "PE")
        odd = types.structure("Odd", [("a", "int"), ("b", "char")], size=7)
        regular = types.structure("Regular", [("a", "long long")])
        self.assertTrue(rule.matches_type(odd))
        self.assertFalse(rule.matches_type(regular))
        self.assertTrue(rule.is_supported)

        hidden_element = ET.Element("rule")
        ET.SubElement(hidden_element, "datatype", {"name": "struct"})
        ET.SubElement(
            hidden_element,
            "hidden_return",
            {"strategy": "normalparam", "voidlock": "true"},
        )
        hidden_rule = CspecModelRule.from_element(hidden_element, "output")
        self.assertTrue(hidden_rule.is_supported)
        self.assertEqual(hidden_rule.hidden_strategy, "normalparam")
        self.assertTrue(hidden_rule.hidden_voidlock)

        unsupported_element = ET.Element("rule")
        ET.SubElement(unsupported_element, "datatype", {"name": "struct"})
        ET.SubElement(unsupported_element, "hidden_return", {"strategy": "magic"})
        self.assertFalse(
            CspecModelRule.from_element(unsupported_element, "output").is_supported
        )


if __name__ == "__main__":
    unittest.main()
