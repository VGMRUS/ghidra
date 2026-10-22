from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from minighidra import Analyzer, BinaryImage
from minighidra.decompiler import SemanticDecompiler
from minighidra.model import FunctionParameter
from minighidra.paramid import (
    ParamActive,
    ParamIDAnalysis,
    ParamRank,
    PcodeLikeGraph,
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
            self.assertGreaterEqual(
                decompiler._paramid_analysis.measurements["param_1"].indirect_uses,
                1,
            )

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
            self.assertTrue(
                all(not trial.used for trial in decompiler._paramid_active.trials)
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
