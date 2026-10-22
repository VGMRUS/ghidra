from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from minighidra import Analyzer, BinaryImage
from minighidra.abi import X86_FASTCALL
from minighidra.decompiler import SemanticDecompiler
from minighidra.model import FunctionParameter, FunctionSignature
from minighidra.paramid import (
    ParamActive,
    ParamDirection,
    ParamIDAnalysis,
    ParamRank,
    PcodeLikeGraph,
    PcodeOp,
)
from minighidra.typesys import StorageLocation


@dataclass(frozen=True)
class Expr:
    kind: str
    value: object = None
    args: tuple[Expr, ...] = ()
    bits: int = 64


class ParamIDTests(unittest.TestCase):
    def test_expression_use_becomes_direct_read_trial(self) -> None:
        graph = PcodeLikeGraph()
        source = Expr("var", "param_1")
        constant = Expr("const", 5)
        addition = Expr("add", args=(source, constant))
        graph.record_use(
            addition,
            role="direct",
            source_names={"param_1"},
            address=0x401000,
            size_bits=64,
            evidence="test-arithmetic-read",
        )

        analysis = ParamIDAnalysis(graph)
        active = analysis.analyze()
        trial = active.get_trial("param_1")

        self.assertIsNotNone(trial)
        assert trial is not None
        self.assertEqual(trial.rank, ParamRank.DIRECT_READ)
        self.assertTrue(trial.checked)
        self.assertTrue(trial.active)
        self.assertEqual(analysis.measurements["param_1"].direct_reads, 1)
        self.assertTrue(any(operation.opcode == "ADD" for operation in graph.ops))

    def test_multiequal_forwards_a_trial_to_a_call_use(self) -> None:
        graph = PcodeLikeGraph()
        source = Expr("var", "param_2")
        other_path = Expr("const", 0)
        graph.record_phi(
            "phi_rdi_loc_401020",
            (source, other_path),
            64,
            address=0x401020,
        )
        graph.record_use(
            Expr("var", "phi_rdi_loc_401020"),
            role="call",
            address=0x401030,
            size_bits=64,
            evidence="forwarded-to-callee",
        )

        analysis = ParamIDAnalysis(graph)
        active = analysis.analyze()
        trial = active.get_trial("param_2")

        self.assertIsNotNone(trial)
        assert trial is not None
        self.assertEqual(trial.rank, ParamRank.SUBFUNCTION_PARAMETER)
        self.assertTrue(trial.active)
        self.assertEqual(analysis.measurements["param_2"].call_uses, 1)
        self.assertTrue(any(operation.opcode == "MULTIEQUAL" for operation in graph.ops))

    def test_return_and_indirect_uses_keep_distinct_ranks(self) -> None:
        graph = PcodeLikeGraph()
        for name, role in (("param_3", "return"), ("param_4", "indirect")):
            graph.record_use(
                Expr("var", name),
                role=role,
                address=0x402000,
                size_bits=64,
                evidence=f"{role}-sink",
            )

        analysis = ParamIDAnalysis(graph)
        active = analysis.analyze()

        self.assertEqual(
            active.get_trial("param_3").rank, ParamRank.FUNCTION_RETURN
        )
        self.assertEqual(active.get_trial("param_4").rank, ParamRank.INDIRECT)
        self.assertIn("paramid-return-uses:1", active.get_trial("param_3").evidence)
        self.assertIn("paramid-indirect-uses:1", active.get_trial("param_4").evidence)

    def test_input_ranks_match_ghidra_call_and_memory_operand_roles(self) -> None:
        graph = PcodeLikeGraph()
        pointer = Expr("var", "param_1")
        callee = Expr("var", "param_2")
        graph.record_memory_load(
            Expr("load", args=(pointer,)),
            pointer,
            location="memory[param_1]",
            address=0x402080,
            mnemonic="mov",
            size_bits=64,
        )
        graph.record_call(
            (pointer,),
            target=callee,
            address=0x402090,
            mnemonic="call",
            evidence="test-call",
        )

        analysis = ParamIDAnalysis(graph)
        analysis.analyze()

        self.assertEqual(
            analysis.measurements["param_1"].rank, ParamRank.DIRECT_READ
        )
        self.assertEqual(
            analysis.measurements["param_2"].rank, ParamRank.DIRECT_READ
        )
        self.assertEqual(analysis.measurements["param_1"].call_uses, 1)
        self.assertEqual(analysis.measurements["param_1"].numcalls, 1)

        indirect = PcodeLikeGraph()
        source = indirect._variable_node("param_3", 64)
        output = indirect._new_temporary_node("indirect", 64)
        indirect._set_definition(PcodeOp("INDIRECT", (source,), output))
        self.assertEqual(
            indirect.measure("param_3").rank, ParamRank.INDIRECT
        )

    def test_all_forward_opcode_roles_are_ranked_by_paramid(self) -> None:
        cases = (
            ("BRANCH", ParamRank.DIRECT_READ),
            ("BRANCHIND", ParamRank.DIRECT_READ),
            ("CBRANCH", ParamRank.DIRECT_READ),
            ("CALL", ParamRank.DIRECT_READ),
            ("CALLIND", ParamRank.DIRECT_READ),
            ("CALLOTHER", ParamRank.DIRECT_READ),
            ("RETURN", ParamRank.FUNCTION_RETURN),
            ("INDIRECT", ParamRank.INDIRECT),
        )
        for index, (opcode, expected) in enumerate(cases, start=1):
            with self.subTest(opcode=opcode):
                graph = PcodeLikeGraph()
                name = f"param_{index}"
                source = graph._variable_node(name, 64)
                graph._append_operation(PcodeOp(opcode, (source,)))
                self.assertEqual(graph.measure(name).rank, expected)

        call_graph = PcodeLikeGraph()
        argument = Expr("var", "param_9")
        call_graph.record_call(
            (argument,),
            address=0x4020C0,
            mnemonic="call",
            evidence="argument-only-call-input",
        )
        self.assertEqual(
            call_graph.measure("param_9").rank,
            ParamRank.SUBFUNCTION_PARAMETER,
        )

    def test_backward_output_ranks_cover_special_ops_and_later_reads(self) -> None:
        cases = (
            ("CALLOTHER", ParamRank.DIRECT_READ, "direct-write-with-read"),
            ("RETURN", ParamRank.FUNCTION_RETURN, "subfunction-return"),
            ("INDIRECT", ParamRank.INDIRECT, "indirect"),
        )
        for index, (opcode, expected, label) in enumerate(cases):
            with self.subTest(opcode=opcode):
                graph = PcodeLikeGraph()
                value = graph._new_temporary_node(f"return_{index}", 64)
                graph._output_values["return_test"].add(value)
                graph._set_definition(PcodeOp(opcode, (), value))
                measurement = graph.measure_output("return_test")
                self.assertEqual(measurement.rank, expected)
                self.assertEqual(measurement.rank_label, label)

        graph = PcodeLikeGraph()
        returned = graph._new_temporary_node("return_value", 64)
        later_use = graph._new_temporary_node("later_use", 64)
        graph._output_values["return_test"].add(returned)
        graph._set_definition(PcodeOp("COPY", (), returned))
        graph._set_definition(PcodeOp("INT_ADD", (returned,), later_use))
        measurement = graph.measure_output("return_test")
        self.assertEqual(measurement.rank, ParamRank.DIRECT_READ)
        self.assertEqual(measurement.rank_label, "direct-write-with-read")

    def test_loop_phi_backedge_does_not_propagate_paramid(self) -> None:
        graph = PcodeLikeGraph()
        graph.record_phi(
            "phi_loop",
            (Expr("const", 0), Expr("var", "param_1")),
            64,
            address=0x4020A0,
            loop_inputs=(False, True),
        )
        graph.record_use(
            Expr("var", "phi_loop"),
            role="direct",
            address=0x4020B0,
            evidence="phi-read",
        )

        measurement = ParamIDAnalysis(graph).analyze().get_trial("param_1")

        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertEqual(measurement.rank, ParamRank.UNREFERENCED)

    def test_ssa_paramid_excludes_backedge_inputs_from_phi_walks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paramid-loop-phi.bin"
            path.write_bytes(
                bytes.fromhex("48 89 f8 48 83 c0 01 48 85 c0 75 f7 c3")
            )
            base = 0x4020D0
            image = BinaryImage.load(path, raw_arch="x86_64", base=base)
            analyzer = Analyzer(image)
            function = analyzer.functions[base]
            decompiler = SemanticDecompiler(analyzer)

            decompiler.decompile(function)

            graph = decompiler.paramid_ssa_graph
            self.assertIsNotNone(graph)
            assert graph is not None
            phis = [op for op in graph.ops if op.opcode == "MULTIEQUAL"]
            self.assertTrue(phis)
            self.assertTrue(any(any(op.loop_inputs) for op in phis))
            self.assertTrue(all(len(op.inputs) == len(op.loop_inputs) for op in phis))
            self.assertEqual(
                decompiler._paramid_analysis.measurements["param_1"].rank,
                ParamRank.DIRECT_READ,
            )

    def test_param_active_binds_storage_and_compacts_only_final_used_trials(self) -> None:
        active = ParamActive()
        first = active.register_trial(
            "param_1",
            32,
            (StorageLocation("register", 4, register="rdi"),),
        )
        second = active.register_trial("param_2", 32)
        first.mark_active()
        second.mark_no_use()
        self.assertFalse(second.is_active())
        self.assertEqual(active.get_num_trials(), 2)
        self.assertEqual(active.get_num_used(), 0)

        selected = active.select_candidates(
            ("param_2",), include_inactive=True
        )
        self.assertEqual([trial.name for trial in selected], ["param_2"])
        self.assertTrue(second.is_selected())
        self.assertFalse(second.is_active())
        self.assertFalse(second.is_used())
        self.assertFalse(first.is_selected())

        first.mark_used()
        self.assertTrue(first.is_active())
        self.assertTrue(first.is_selected())
        self.assertTrue(first.is_used())
        active.delete_unused_trials()
        self.assertEqual([trial.name for trial in active.trials], ["param_1"])
        self.assertEqual(active.trials[0].locations[0].register, "rdi")
        self.assertEqual(active.get_num_used(), 1)

    def test_param_active_reregisters_trial_state_with_a_fresh_slot(self) -> None:
        active = ParamActive()
        original = active.register_trial(
            "param_1",
            64,
            (StorageLocation("register", 8, register="rdi"),),
        )
        original.mark_active()

        copied = active.reregister_trial(original)

        self.assertEqual(active.get_num_trials(), 2)
        self.assertEqual(original.slot, 1)
        self.assertEqual(copied.slot, 2)
        self.assertTrue(copied.active)
        self.assertTrue(copied.killed_by_call)
        self.assertIs(active.get_trial("param_1"), copied)

    def test_justproto_requires_explicit_formal_names(self) -> None:
        analysis = ParamIDAnalysis(PcodeLikeGraph())

        with self.assertRaisesRegex(ValueError, "nombres de los formales"):
            analysis.analyze(justproto=True)

    def test_binding_selects_signature_candidates_without_marking_them_used(self) -> None:
        graph = PcodeLikeGraph()
        graph.record_use(
            Expr("var", "param_1"), role="direct", address=0x402100
        )
        analysis = ParamIDAnalysis(graph)
        analysis.analyze()
        active = analysis.bind_parameters(
            (
                FunctionParameter("param_1", "uint64_t"),
                FunctionParameter("param_2", "uint64_t"),
            )
        )

        first = active.get_trial("param_1")
        second = active.get_trial("param_2")
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertTrue(first.is_active())
        self.assertTrue(first.is_selected())
        self.assertFalse(first.is_used())
        self.assertFalse(second.is_active())
        self.assertTrue(second.is_selected())
        self.assertFalse(second.is_used())
        self.assertEqual(active.get_num_selected(), 2)
        self.assertEqual(active.get_num_used(), 0)

    def test_expression_depth_is_bounded_and_preserves_input_measurement(self) -> None:
        graph = PcodeLikeGraph()
        expression = Expr("var", "param_5")
        for _ in range(PcodeLikeGraph.MAX_EXPRESSION_DEPTH + 3):
            expression = Expr("add", args=(expression, Expr("const", 1)))
        graph.record_use(
            expression,
            role="direct",
            source_names={"param_5"},
            address=0x403000,
            size_bits=64,
            evidence="deep-expression",
        )

        measurement = graph.measure("param_5")

        self.assertEqual(measurement.rank, ParamRank.DIRECT_READ)
        self.assertGreater(graph.depth_limited_uses, 0)
        self.assertTrue(any(operation.opcode == "OPAQUE_EXPR" for operation in graph.ops))

    def test_semantic_instruction_flow_tracks_register_and_memory_effects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paramid-memory.bin"
            path.write_bytes(bytes.fromhex("48 8b 07 48 89 37 c3"))
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x405000)
            analyzer = Analyzer(image)
            function = analyzer.functions[0x405000]
            decompiler = SemanticDecompiler(analyzer)

            decompiler.decompile(function)

            flows = {
                flow.address: flow
                for flow in decompiler._paramid_graph.instruction_flows
            }
            load = flows[0x405000]
            store = flows[0x405003]
            self.assertEqual(load.mnemonic, "mov")
            self.assertIn("rdi", load.read_registers)
            self.assertIn("rax", load.written_registers)
            self.assertEqual(load.read_memory, ("memory[rdi]",))
            self.assertTrue(load.reads)
            self.assertTrue(load.writes)
            self.assertTrue(
                any(
                    decompiler._paramid_graph.ops[index].opcode == "LOAD"
                    for index in load.operation_indices
                )
            )

            self.assertIn("rdi", store.read_registers)
            self.assertIn("rsi", store.read_registers)
            self.assertEqual(store.written_memory, ("memory[rdi]",))
            self.assertTrue(
                any(
                    decompiler._paramid_graph.ops[index].opcode == "STORE"
                    for index in store.operation_indices
                )
            )
            self.assertEqual(
                decompiler._paramid_analysis.measurements["param_1"].rank,
                ParamRank.DIRECT_READ,
            )
            self.assertGreaterEqual(
                decompiler._paramid_analysis.measurements["param_1"].direct_reads,
                1,
            )
            ssa_graph = decompiler.paramid_ssa_graph
            self.assertIsNotNone(ssa_graph)
            assert ssa_graph is not None
            self.assertEqual(ssa_graph.measure("param_1").rank, ParamRank.DIRECT_READ)

    def test_semantic_paramid_consumes_complete_ssa_pcode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paramid-ssa.bin"
            path.write_bytes(bytes.fromhex("48 89 f8 c3"))  # mov rax, rdi; ret
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x405800)
            analyzer = Analyzer(image)
            function = analyzer.functions[0x405800]
            decompiler = SemanticDecompiler(analyzer)

            decompiler.decompile(function)

            analysis = decompiler._paramid_analysis
            graph = decompiler.paramid_ssa_graph
            self.assertIsNotNone(graph)
            assert graph is not None
            self.assertTrue(analysis.uses_ssa_pcode)
            self.assertIs(analysis.graph, graph)
            self.assertIs(graph.ssa_function, decompiler.ssa_function)
            self.assertEqual(
                graph.ssa_operation_count, len(decompiler.ssa_function.operations)
            )
            self.assertEqual(
                sum(operation.evidence.startswith("ssa-pcode:") for operation in graph.ops),
                graph.ssa_operation_count,
            )
            self.assertTrue(
                any(
                    operation.evidence == "ssa-abi-return-registers"
                    for operation in graph.ops
                )
            )
            measurement = analysis.measurements["param_1"]
            self.assertEqual(measurement.rank, ParamRank.DIRECT_READ)
            self.assertGreaterEqual(measurement.return_uses, 1)

    def test_output_measurement_walks_back_from_the_inferred_return_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paramid-output.bin"
            path.write_bytes(bytes.fromhex("48 89 f8 c3"))  # mov rax, rdi; ret
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x406800)
            analyzer = Analyzer(image)
            function = analyzer.functions[0x406800]
            decompiler = SemanticDecompiler(analyzer)

            decompiler.decompile(function)

            outputs = decompiler._paramid_analysis.output_measurements
            self.assertIn("return_rax", outputs)
            output = outputs["return_rax"]
            self.assertEqual(output.direction, ParamDirection.OUTPUT)
            self.assertEqual(output.rank, ParamRank.DIRECT_WRITE_WITHOUT_READ)
            self.assertEqual(output.rank_label, "direct-write-without-read")

    def test_output_backward_phi_ignores_parent_phi_in_forward_probe(self) -> None:
        graph = PcodeLikeGraph()
        source = graph._new_temporary_node("source", 64)
        incoming = graph._new_temporary_node("incoming", 64)
        graph._set_definition(PcodeOp("COPY", (source,), incoming))
        other_incoming = graph._new_temporary_node("other_incoming", 64)
        output = graph._new_temporary_node("return_phi", 64)
        graph._set_definition(
            PcodeOp(
                "MULTIEQUAL",
                (incoming, other_incoming),
                output,
                loop_inputs=(False, False),
            )
        )
        phi_index = graph._definitions[output]
        graph._output_values["return_test"].add(output)
        sink_index = graph._append_operation(PcodeOp("RETURN", (output,)))
        graph._output_sink_operations["return_test"].add(sink_index)
        direct_read = graph._new_temporary_node("direct_read", 64)
        graph._set_definition(PcodeOp("INT_ADD", (output,), direct_read))

        measurement = graph.measure_output("return_test")

        self.assertEqual(measurement.rank, ParamRank.DIRECT_WRITE_WITHOUT_READ)
        self.assertNotEqual(phi_index, sink_index)

    def test_paramid_reads_incoming_stack_trials_in_x86_and_x64_ssa(self) -> None:
        cases = (
            ("x86", "8b 44 24 04 c3", "", 0x510000, "param_1"),
            ("x86_64", "48 8b 44 24 08 c3", "", 0x511000, "param_7"),
            ("x86_64", "48 8b 44 24 28 c3", "PE", 0x512000, "param_5"),
        )
        for architecture, code, format_name, base, name in cases:
            with self.subTest(architecture=architecture, format=format_name):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "paramid-stack-input.bin"
                    path.write_bytes(bytes.fromhex(code))
                    image = BinaryImage.load(
                        path, raw_arch=architecture, base=base
                    )
                    if format_name:
                        image.format = format_name
                    analyzer = Analyzer(image)
                    function = analyzer.functions[base]
                    decompiler = SemanticDecompiler(analyzer)

                    decompiler.decompile(function)

                    graph = decompiler.paramid_ssa_graph
                    self.assertIsNotNone(graph)
                    assert graph is not None
                    self.assertIn(name, graph.parameter_names)
                    self.assertEqual(
                        decompiler._paramid_analysis.measurements[name].rank,
                        ParamRank.DIRECT_READ,
                    )
                    self.assertTrue(
                        any(
                            value.name == name and value.kind == "input"
                            for value in graph.nodes
                        )
                    )
                    if architecture == "x86":
                        self.assertIn(
                            "return_rax",
                            decompiler._paramid_analysis.output_measurements,
                        )

    def test_declared_sret_pointer_is_measured_as_hidden_formal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paramid-declared-sret.bin"
            path.write_bytes(bytes.fromhex("48 89 07 c3"))
            base = 0x512200
            image = BinaryImage.load(path, raw_arch="x86_64", base=base)
            analyzer = Analyzer(image)
            function = analyzer.functions[base]
            decompiler = SemanticDecompiler(analyzer)
            result_type = decompiler._type_database.structure(
                "ParamIDResult",
                (
                    ("first", "uint64_t"),
                    ("second", "uint64_t"),
                    ("third", "uint64_t"),
                ),
            )
            function.signature = FunctionSignature(
                "struct ParamIDResult",
                (),
                "sysv_amd64",
                return_data_type=result_type,
                source="declared",
                locked=True,
            )

            decompiler.decompile(function)

            analysis = decompiler._paramid_analysis
            self.assertTrue(analysis.justproto)
            self.assertEqual(tuple(analysis.measurements), ("__hidden_return",))
            self.assertEqual(
                analysis.measurements["__hidden_return"].rank,
                ParamRank.DIRECT_READ,
            )
            trial = decompiler._paramid_active.get_trial("__hidden_return")
            self.assertIsNotNone(trial)
            assert trial is not None
            self.assertTrue(trial.used)
            self.assertTrue(trial.locations[0].hidden)

    def test_declared_stack_formal_is_measured_in_justproto_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paramid-declared-stack.bin"
            path.write_bytes(bytes.fromhex("8b 44 24 04 c3"))
            base = 0x512400
            image = BinaryImage.load(path, raw_arch="x86", base=base)
            analyzer = Analyzer(image)
            function = analyzer.functions[base]
            function.signature = FunctionSignature(
                "uint32_t",
                (FunctionParameter("argument", "uint32_t"),),
                "__cdecl",
                source="declared",
                locked=True,
            )
            decompiler = SemanticDecompiler(analyzer)

            decompiler.decompile(function)

            analysis = decompiler._paramid_analysis
            self.assertTrue(analysis.justproto)
            self.assertEqual(tuple(analysis.measurements), ("argument",))
            self.assertEqual(
                analysis.measurements["argument"].rank,
                ParamRank.DIRECT_READ,
            )
            trial = decompiler._paramid_active.get_trial("argument")
            self.assertIsNotNone(trial)
            assert trial is not None
            self.assertTrue(trial.used)

    def test_paramid_tracks_x86_fastcall_register_argument(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paramid-x86-fastcall.bin"
            path.write_bytes(bytes.fromhex("8b c1 c3"))  # mov eax, ecx; ret
            base = 0x512800
            image = BinaryImage.load(path, raw_arch="x86", base=base)
            analyzer = Analyzer(image)
            analyzer._calling_convention_overrides = {base: X86_FASTCALL}
            function = analyzer.functions[base]
            decompiler = SemanticDecompiler(analyzer)

            decompiler.decompile(function)

            graph = decompiler.paramid_ssa_graph
            self.assertIsNotNone(graph)
            assert graph is not None
            self.assertEqual(
                decompiler._paramid_analysis.measurements["param_1"].rank,
                ParamRank.DIRECT_READ,
            )
            parameter = next(
                item
                for item in function.signature.parameters
                if item.name == "param_1"
            )
            self.assertEqual(parameter.locations[0].register, "ecx")
            self.assertTrue(
                decompiler._paramid_active.get_trial("param_1").used
            )

    def test_declared_prototype_uses_justproto_paramid_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paramid-justproto.bin"
            path.write_bytes(bytes.fromhex("48 89 f8 c3"))
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x513000)
            analyzer = Analyzer(image)
            function = analyzer.functions[0x513000]
            function.signature = FunctionSignature(
                "uint64_t",
                (FunctionParameter("value", "uint64_t"),),
                "sysv_amd64",
                source="declared",
                locked=True,
            )
            decompiler = SemanticDecompiler(analyzer)

            decompiler.decompile(function)

            analysis = decompiler._paramid_analysis
            self.assertTrue(analysis.justproto)
            self.assertEqual(tuple(analysis.measurements), ("value",))
            self.assertEqual(
                analysis.measurements["value"].rank, ParamRank.DIRECT_READ
            )
            self.assertIn("return_rax", analysis.output_measurements)
            parameter = function.signature.parameters[0]
            self.assertIn("paramid-used-final", parameter.evidence)

    def test_call_flow_keeps_arguments_and_return_definition_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paramid-call.bin"
            path.write_bytes(bytes.fromhex("48 89 f7 e8 01 00 00 00 c3 c3"))
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x407000)
            analyzer = Analyzer(image)
            function = analyzer.functions[0x407000]
            decompiler = SemanticDecompiler(analyzer)

            decompiler.decompile(function)

            call_flow = next(
                flow
                for flow in decompiler._paramid_graph.instruction_flows
                if flow.mnemonic == "call"
            )
            self.assertEqual(call_flow.direct_target_address, 0x407009)
            self.assertIsNotNone(call_flow.call_target_name)
            self.assertIsNone(call_flow.call_target)
            self.assertTrue(call_flow.call_arguments)
            self.assertIsNotNone(call_flow.call_result)
            self.assertTrue(
                any(
                    decompiler._paramid_graph.ops[index].opcode == "CALL"
                    and decompiler._paramid_graph.ops[index].output
                    == call_flow.call_result
                    for index in call_flow.operation_indices
                )
            )
            ssa_graph = decompiler.paramid_ssa_graph
            self.assertIsNotNone(ssa_graph)
            assert ssa_graph is not None
            self.assertTrue(
                any(
                    operation.evidence.startswith("ssa-abi-call-argument:")
                    for operation in ssa_graph.ops
                )
            )
            self.assertGreaterEqual(
                decompiler._paramid_analysis.measurements["param_2"].call_uses,
                1,
            )
            self.assertTrue(
                any(
                    trial.name == "param_2" and trial.used
                    for trial in decompiler._paramid_active.trials
                )
            )
            self.assertTrue(
                all(not trial.used for trial in decompiler._paramid_active.inactive_trials)
            )
            self.assertTrue(
                any(trial.selected for trial in decompiler._paramid_active.trials)
            )

    def test_indirect_call_records_target_value_without_fabricating_address(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paramid-call-indirect.bin"
            path.write_bytes(bytes.fromhex("ff d0 c3"))
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x408800)
            analyzer = Analyzer(image)
            function = analyzer.functions[0x408800]
            decompiler = SemanticDecompiler(analyzer)

            decompiler.decompile(function)

            call = next(
                flow
                for flow in decompiler._paramid_graph.instruction_flows
                if flow.mnemonic == "call"
            )
            self.assertIsNone(call.direct_target_address)
            self.assertIsNotNone(call.call_target)
            self.assertIsNotNone(call.call_result)
            self.assertIn("rax", call.read_registers)

    def test_call_flow_tracks_stack_argument_store_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paramid-call-stack.bin"
            path.write_bytes(bytes.fromhex("56 e8 01 00 00 00 c3 c3"))
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x408000)
            analyzer = Analyzer(image)
            function = analyzer.functions[0x408000]
            decompiler = SemanticDecompiler(analyzer)

            decompiler.decompile(function)

            flows = {
                flow.address: flow
                for flow in decompiler._paramid_graph.instruction_flows
            }
            push = flows[0x408000]
            call = flows[0x408001]
            self.assertIn("rsp", push.read_registers)
            self.assertIn("rsp", push.written_registers)
            self.assertEqual(push.written_memory, ("stack[-0x8]",))
            self.assertEqual(call.read_memory, ("stack[-0x8]",))
            self.assertTrue(call.call_arguments)

    def test_semantic_decompiler_links_phi_input_to_function_return(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paramid-phi.bin"
            path.write_bytes(
                bytes.fromhex("85 ff 74 05 48 89 f0 eb 05 b8 01 00 00 00 c3")
            )
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x404000)
            analyzer = Analyzer(image)
            function = analyzer.functions[0x404000]
            decompiler = SemanticDecompiler(analyzer)

            decompiler.decompile(function)

            self.assertTrue(
                any(operation.opcode == "MULTIEQUAL" for operation in decompiler._paramid_graph.ops)
            )
            measurement = decompiler._paramid_analysis.measurements["param_2"]
            self.assertGreaterEqual(measurement.return_uses, 1)
            parameter = next(
                parameter
                for parameter in function.signature.parameters
                if parameter.name == "param_2"
            )
            self.assertIn("paramid-active", parameter.evidence)
            self.assertIn("paramid-selected-candidate", parameter.evidence)
            self.assertIn("paramid-return-uses:1", parameter.evidence)


if __name__ == "__main__":
    unittest.main()
