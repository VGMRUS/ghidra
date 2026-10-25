from __future__ import annotations

import unittest

from minighidra.action_controlflow import ActionCopyMarker, ActionShadowVar
from minighidra.action_name import ActionNameVars
from minighidra.datatype_split import SplitDatatype
from minighidra.merge import Merge, VariablePiece
from minighidra.paramid import PcodeLikeGraph, PcodeOp
from minighidra.rule import RuleSplitCopy, RuleSplitLoad, RuleSplitStore
from minighidra.typesys import TypeDatabase, TypeKind


class TypeDatabaseSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.types = TypeDatabase(64)
        self.left = self.types.structure(
            "LeftPair",
            [("first", "uint32_t"), ("second", "uint32_t")],
            packed=True,
        )
        self.right = self.types.structure(
            "RightPair",
            [("first", "uint32_t"), ("second", "uint32_t")],
            packed=True,
        )

    def test_partial_view_components_are_relative_to_view(self) -> None:
        partial = self.types.partial(self.left, 4, offset=4)
        components = self.types.components(partial)
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].offset, 0)
        self.assertEqual(components[0].size, 4)
        self.assertEqual(components[0].type.name, "uint32_t")

    def test_split_copy_rewrites_aggregate_to_components(self) -> None:
        graph = PcodeLikeGraph()
        source = graph.new_unique(64, "source")
        output = graph.new_unique(64, "output")
        in_type = self.types.partial(self.left, 8)
        out_type = self.types.partial(self.right, 8)
        graph.set_node_type(source, in_type)
        graph.set_node_type(output, out_type)
        copy = PcodeOp("COPY", (source,), output, address=0x1000)
        graph.append_operation(copy)

        self.assertTrue(SplitDatatype(graph, self.types).split_copy(copy))
        self.assertEqual(graph.get_def(output).opcode, "PIECE")
        self.assertEqual(
            sum(op.evidence == "splitdatatype-copy-piece" for op in graph.ops), 2
        )
        self.assertEqual(
            sum(op.evidence == "splitdatatype-input-piece" for op in graph.ops), 2
        )

    def test_split_constant_copy_extracts_little_endian_fields(self) -> None:
        graph = PcodeLikeGraph()
        graph.type_database = self.types
        source = graph.new_constant(64, 0x1122334455667788)
        output = graph.new_unique(64, "output")
        graph.set_node_type(output, self.types.partial(self.right, 8))
        copy = PcodeOp("COPY", (source,), output)
        graph.append_operation(copy)

        self.assertTrue(SplitDatatype(graph, self.types).split_copy(copy))
        values = [
            graph.get_constant_value(op.inputs[0])
            for op in graph.ops
            if op.evidence == "splitdatatype-copy-piece"
        ]
        self.assertEqual(values, [0x55667788, 0x11223344])

    def test_split_copy_metadata_drives_internal_and_padding_markers(self) -> None:
        padded = self.types.structure(
            "PaddedPair",
            [("first", "uint32_t", 0), ("second", "uint32_t", 8)],
            packed=True,
            size=12,
        )
        graph = PcodeLikeGraph()
        graph.type_database = self.types
        zero = graph.new_constant(96, 0)
        output = graph.new_unique(96, "padded_output")
        graph.set_node_type(output, padded)
        copy = PcodeOp("COPY", (zero,), output)
        graph.append_operation(copy)

        self.assertTrue(SplitDatatype(graph, self.types).split_copy(copy))
        piece_copies = [
            operation
            for operation in graph.ops
            if operation.evidence == "splitdatatype-copy-piece"
        ]
        self.assertEqual(len(piece_copies), 3)
        self.assertEqual(len(graph.padding_nodes), 2)

        ActionCopyMarker("merge").apply(graph)
        marked_piece_copies = [
            operation
            for operation in graph.ops
            if operation.evidence == "splitdatatype-copy-piece"
        ]
        self.assertTrue(all(operation.non_printing for operation in marked_piece_copies))
        padding_piece = next(
            operation
            for operation in marked_piece_copies
            if operation.output in graph.padding_nodes
        )
        self.assertEqual(padding_piece.non_printing_reason, "pure-join-padding")
        reassembly = [
            operation
            for operation in graph.ops
            if operation.evidence == "splitdatatype-reassemble"
        ]
        self.assertTrue(reassembly)
        self.assertTrue(all(operation.non_printing for operation in reassembly))

    def test_split_copy_metadata_supports_a_shared_aggregate_source(self) -> None:
        graph = PcodeLikeGraph()
        source = graph.new_unique(64, "source")
        first_output = graph.new_unique(64, "first_output")
        second_output = graph.new_unique(64, "second_output")
        graph.set_node_type(source, self.types.partial(self.left, 8))
        graph.set_node_type(first_output, self.types.partial(self.right, 8))
        graph.set_node_type(second_output, self.types.partial(self.right, 8))
        first_copy = PcodeOp("COPY", (source,), first_output)
        second_copy = PcodeOp("COPY", (source,), second_output)
        graph.append_operation(first_copy)
        graph.append_operation(second_copy)

        splitter = SplitDatatype(graph, self.types)
        self.assertTrue(splitter.split_copy(first_copy))
        self.assertTrue(splitter.split_copy(second_copy))
        ActionCopyMarker("merge").apply(graph)

        input_pieces = [
            operation
            for operation in graph.ops
            if operation.evidence == "splitdatatype-input-piece"
        ]
        self.assertEqual(len(input_pieces), 4)
        self.assertTrue(all(operation.non_printing for operation in input_pieces))

    def test_split_constant_copy_extracts_big_endian_fields(self) -> None:
        graph = PcodeLikeGraph()
        graph.type_database = self.types
        graph.big_endian = True
        source = graph.new_constant(64, 0x1122334455667788)
        output = graph.new_unique(64, "output")
        graph.set_node_type(output, self.types.partial(self.right, 8))
        copy = PcodeOp("COPY", (source,), output)
        graph.append_operation(copy)

        self.assertTrue(SplitDatatype(graph, self.types).split_copy(copy))
        values = [
            graph.get_constant_value(op.inputs[0])
            for op in graph.ops
            if op.evidence == "splitdatatype-copy-piece"
        ]
        self.assertEqual(values, [0x11223344, 0x55667788])
        self.assertEqual(graph.get_def(output).opcode, "PIECE")

    def test_split_load_creates_typed_field_accesses(self) -> None:
        graph = PcodeLikeGraph()
        pointer = graph.new_unique(64, "pointer")
        output = graph.new_unique(64, "loaded")
        graph.set_node_type(pointer, self.types.pointer(self.right))
        graph.set_node_type(output, self.right)
        load = PcodeOp("LOAD", (pointer,), output, address=0x2000, input_slots=(1,))
        graph.append_operation(load)

        self.assertTrue(SplitDatatype(graph, self.types).split_load(load))
        loads = [op for op in graph.ops if op.evidence == "splitdatatype-load-piece"]
        self.assertEqual(len(loads), 2)
        self.assertEqual(graph.get_def(output).opcode, "PIECE")
        for operation in loads:
            self.assertEqual(
                graph.get_node_type(operation.output).kind,
                TypeKind.INTEGER,
            )

    def test_split_store_creates_field_writes(self) -> None:
        graph = PcodeLikeGraph()
        pointer = graph.new_unique(64, "pointer")
        value = graph.new_unique(64, "value")
        graph.set_node_type(pointer, self.types.pointer(self.right))
        graph.set_node_type(value, self.types.partial(self.left, 8))
        store = PcodeOp(
            "STORE",
            (pointer, value),
            address=0x3000,
            input_slots=(1, 2),
        )
        graph.append_operation(store)

        self.assertTrue(SplitDatatype(graph, self.types).split_store(store))
        stores = [op for op in graph.ops if op.opcode == "STORE"]
        self.assertEqual(len(stores), 2)
        self.assertTrue(
            all(op.evidence == "splitdatatype-store-piece" for op in stores)
        )

    def test_split_rules_use_compatible_base_constructor(self) -> None:
        rules = (
            (RuleSplitCopy("splitcopy"), "splitcopy", "splitcopy"),
            (RuleSplitLoad("splitpointer"), "splitpointer", "splitload"),
            (RuleSplitStore("splitpointer"), "splitpointer", "splitstore"),
        )
        for rule, expected_group, expected_name in rules:
            self.assertEqual(rule.group, expected_group)
            self.assertEqual(rule.name, expected_name)

    def test_split_rules_dispatch_to_typed_rewrites(self) -> None:
        copy_graph = PcodeLikeGraph()
        copy_graph.type_database = self.types
        copy_source = copy_graph.new_unique(64, "copy_source")
        copy_output = copy_graph.new_unique(64, "copy_output")
        copy_graph.set_node_type(copy_source, self.types.partial(self.left, 8))
        copy_graph.set_node_type(copy_output, self.types.partial(self.right, 8))
        copy = PcodeOp("COPY", (copy_source,), copy_output)
        copy_graph.append_operation(copy)

        load_graph = PcodeLikeGraph()
        load_graph.type_database = self.types
        pointer = load_graph.new_unique(64, "pointer")
        load_output = load_graph.new_unique(64, "loaded")
        load_graph.set_node_type(pointer, self.types.pointer(self.right))
        load_graph.set_node_type(load_output, self.types.partial(self.right, 8))
        load = PcodeOp("LOAD", (pointer,), load_output, input_slots=(1,))
        load_graph.append_operation(load)

        store_graph = PcodeLikeGraph()
        store_graph.type_database = self.types
        store_pointer = store_graph.new_unique(64, "store_pointer")
        store_value = store_graph.new_unique(64, "store_value")
        store_graph.set_node_type(store_pointer, self.types.pointer(self.right))
        store_graph.set_node_type(store_value, self.types.partial(self.left, 8))
        store = PcodeOp("STORE", (store_pointer, store_value), input_slots=(1, 2))
        store_graph.append_operation(store)

        self.assertEqual(RuleSplitCopy("splitcopy").apply_op(copy, copy_graph), 1)
        self.assertEqual(RuleSplitLoad("splitpointer").apply_op(load, load_graph), 1)
        self.assertEqual(RuleSplitStore("splitpointer").apply_op(store, store_graph), 1)


class AliasAndMergeTests(unittest.TestCase):
    def test_stale_pcode_reference_tracks_immutable_operation_rewrites(self) -> None:
        graph = PcodeLikeGraph()
        first = graph.new_unique(32, "first")
        second = graph.new_unique(32, "second")
        replacement = graph.new_unique(32, "replacement")
        output = graph.new_unique(32, "output")
        original = PcodeOp("INT_ADD", (first, second), output)
        graph.append_operation(original)

        graph.op_remove_input(original, 1)
        graph.op_set_opcode(original, "COPY")
        graph.op_set_input(original, replacement, 0)
        current = graph.get_def(output)
        self.assertEqual(current.opcode, "COPY")
        self.assertEqual(current.inputs, (replacement,))

    def test_shadow_var_rewrites_duplicate_phi_to_copy(self) -> None:
        graph = PcodeLikeGraph()
        first, second, output_a, output_b = [
            graph.new_unique(32, "node") for _ in range(4)
        ]
        phi_a = PcodeOp(
            "MULTIEQUAL", (first, second), output_a, address=0x4000, block_id=7
        )
        phi_b = PcodeOp(
            "MULTIEQUAL", (first, second), output_b, address=0x4000, block_id=7
        )
        graph.append_operation(phi_a)
        graph.append_operation(phi_b)

        self.assertEqual(ActionShadowVar("analysis").apply(graph), 1)
        self.assertEqual(graph.get_def(output_b).opcode, "COPY")
        self.assertEqual(graph.shadow_aliases[output_b], output_a)

    def test_copy_marker_marks_nonprinting_copy_after_explicit_merge(self) -> None:
        graph = PcodeLikeGraph()
        source = graph.new_unique(32, "source")
        output = graph.new_unique(32, "copy")
        graph.set_node_type(source, self._int_type())
        graph.set_node_type(output, self._int_type())
        copy = PcodeOp("COPY", (source,), output)
        graph.append_operation(copy)
        merge = Merge(graph)
        merge.assign_highs()
        merge.merge_op("COPY")
        graph.merge = merge

        self.assertGreaterEqual(ActionCopyMarker("merge").apply(graph), 1)
        rewritten = graph.get_def(output)
        self.assertTrue(rewritten.non_printing)
        self.assertEqual(rewritten.non_printing_reason, "internal-copy")
        self.assertIs(
            graph.merge.node_to_high[source], graph.merge.node_to_high[output]
        )

    def test_copy_marker_does_not_merge_high_variables(self) -> None:
        graph = PcodeLikeGraph()
        source = graph.new_unique(32, "source")
        output = graph.new_unique(32, "copy")
        graph.append_operation(PcodeOp("COPY", (source,), output))

        ActionCopyMarker("merge").apply(graph)
        self.assertIsNot(
            graph.merge.node_to_high[source], graph.merge.node_to_high[output]
        )

    def test_shadow_alias_name_propagates_without_highvar_merge(self) -> None:
        graph = PcodeLikeGraph()
        first = graph.new_unique(32, "first")
        second = graph.new_unique(32, "second")
        other = graph.new_unique(32, "other")
        phi_output = graph._new_node("canonical_phi", 32, "phi")
        duplicate_output = graph._new_node("duplicate_phi", 32, "phi")
        use_output = graph.new_unique(32, "use")
        graph.node_names[phi_output] = "canonical_phi"
        graph.append_operation(
            PcodeOp("MULTIEQUAL", (first, second), phi_output, block_id=11)
        )
        graph.append_operation(
            PcodeOp("MULTIEQUAL", (first, second), duplicate_output, block_id=11)
        )
        graph.append_operation(PcodeOp("INT_ADD", (phi_output, other), use_output))

        ActionShadowVar("analysis").apply(graph)
        merge = Merge(graph)
        merge.assign_highs()
        merge.merge_op("MULTIEQUAL")
        merge.merge_op("COPY")
        graph.merge = merge
        self.assertIsNot(
            merge.node_to_high[phi_output], merge.node_to_high[duplicate_output]
        )

        ActionNameVars("merge").apply(graph)
        self.assertEqual(graph.get_node_name(duplicate_output), "canonical_phi")

    def test_variable_names_follow_copy_and_phi_aliases(self) -> None:
        graph = PcodeLikeGraph()
        parameter = graph._new_node("param_0", 32, "input")
        copied = graph.new_unique(32, "copy")
        graph.append_operation(PcodeOp("COPY", (parameter,), copied))
        merge = Merge(graph)
        merge.assign_highs()
        merge.merge_op("COPY")
        graph.merge = merge
        ActionCopyMarker("merge").apply(graph)
        ActionNameVars("merge").apply(graph)
        self.assertEqual(graph.get_node_name(copied), "param_0")

        second = graph.new_unique(32, "second")
        phi_output = graph.new_unique(32, "phi")
        duplicate_output = graph.new_unique(32, "phi")
        graph.append_operation(
            PcodeOp("MULTIEQUAL", (parameter, second), phi_output, block_id=9)
        )
        graph.append_operation(
            PcodeOp("MULTIEQUAL", (parameter, second), duplicate_output, block_id=9)
        )
        ActionShadowVar("analysis").apply(graph)
        merge.assign_highs()
        merge.merge_op("MULTIEQUAL")
        merge.merge_op("COPY")
        ActionCopyMarker("merge").apply(graph)
        ActionNameVars("merge").apply(graph)
        self.assertEqual(
            graph.get_node_name(duplicate_output), graph.get_node_name(phi_output)
        )

    def _int_type(self):
        return TypeDatabase(64).get("int")

    def test_piece_layout_marks_padding_initialization(self) -> None:
        graph = PcodeLikeGraph()
        zero = graph.new_constant(32, 0)
        padding = graph.new_unique(32, "padding")
        copy = PcodeOp("COPY", (zero,), padding)
        graph.append_operation(copy)
        graph.padding_nodes.add(padding)
        merge = Merge(graph)
        merge.assign_highs()
        self.assertEqual(merge.mark_pure_padding_copies(), 1)
        self.assertEqual(graph.get_def(padding).non_printing_reason, "pure-join-padding")

    def test_variable_piece_rejects_invalid_ranges(self) -> None:
        with self.assertRaises(ValueError):
            VariablePiece("aggregate", -1, 4)
        with self.assertRaises(ValueError):
            VariablePiece("aggregate", 0, 0)


if __name__ == "__main__":
    unittest.main()
