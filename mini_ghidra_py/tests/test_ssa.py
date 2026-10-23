from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from minighidra import (
    Analyzer,
    BinaryImage,
    SleighTranslator,
    SSAHeritage,
    SSAReturnSite,
)
from minighidra.decompiler import SemanticDecompiler


class SSAHeritageTests(unittest.TestCase):
    def _analyze(self, raw: bytes, arch: str, base: int = 0x410000):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "ssa-regression.bin"
        path.write_bytes(raw)
        analyzer = Analyzer(BinaryImage.load(path, raw_arch=arch, base=base))
        return analyzer, analyzer.functions[base]

    def test_instruction_cfg_diamond_places_live_register_phis_x86_and_x64(
        self,
    ) -> None:
        raw = bytes.fromhex(
            "b8 01 00 00 00 85 c9 74 07 b8 02 00 00 00 eb 05 b8 03 00 00 00 83 c0 01 c3"
        )
        for arch in ("x86", "x86_64"):
            with self.subTest(arch=arch):
                analyzer, function = self._analyze(raw, arch)
                result = analyzer.ssa(function)

                self.assertTrue(result.valid, result.diagnostics)
                self.assertTrue(result.complete, result.diagnostics)
                register_phis = [
                    phi for phi in result.phis if phi.location.space == "register"
                ]
                self.assertGreaterEqual(len(register_phis), 4)
                self.assertTrue(all(len(phi.incoming) == 2 for phi in register_phis))
                self.assertTrue(
                    any(
                        len(block.predecessors) == 2
                        and block.phi_outputs
                        and block.reachable
                        for block in result.blocks
                    )
                )

    def test_return_sites_capture_abi_register_values_without_calls(self) -> None:
        raw = bytes.fromhex("b8 2a 00 00 00 c3")  # mov eax, 42; ret
        analyzer, function = self._analyze(raw, "x86_64")
        result = analyzer.ssa(function)

        self.assertTrue(result.valid, result.diagnostics)
        self.assertTrue(result.complete, result.diagnostics)
        self.assertEqual(len(result.return_sites), 1)
        self.assertIsInstance(result.return_sites[0], SSAReturnSite)
        return_values = dict(result.return_sites[0].register_values)
        self.assertIn("rax", return_values)
        self.assertEqual(return_values["rax"].size_bits, 64)
        self.assertTrue(return_values["rax"].values)

    def test_call_return_register_reaches_function_return_site(self) -> None:
        raw = bytes.fromhex("e8 01 00 00 00 c3 90 c3")
        analyzer, function = self._analyze(raw, "x86_64")
        result = analyzer.ssa(function)

        self.assertTrue(result.valid, result.diagnostics)
        self.assertTrue(result.complete, result.diagnostics)
        self.assertEqual(len(result.return_sites), 1)
        return_values = dict(result.return_sites[0].register_values)
        self.assertTrue(return_values["rax"].values)
        self.assertTrue(
            all(
                result.value(value_id).kind == "call_return"
                for value_id in return_values["rax"].values
            )
        )

    def test_dead_register_merges_are_pruned_by_liveness(self) -> None:
        raw = bytes.fromhex("85 c9 74 07 b8 01 00 00 00 eb 05 b8 02 00 00 00 c3")
        analyzer, function = self._analyze(raw, "x86_64")
        result = analyzer.ssa(function)

        self.assertTrue(result.valid, result.diagnostics)
        self.assertFalse(any(phi.location.space == "register" for phi in result.phis))

    def test_internal_sleigh_branches_form_micro_cfg_and_phis(self) -> None:
        raw = bytes.fromhex("0f bc c3 c3")  # bsf eax, ebx; ret
        _analyzer, function = self._analyze(raw, "x86_64")
        translations = SleighTranslator(64).translate_instructions(
            function.instructions
        )
        self.assertTrue(any("CBRANCH" in item.operation_names for item in translations))

        result = SSAHeritage(
            function,
            translations,
            caller_saved_registers=("rax", "rcx", "rdx", "xmm0"),
            return_registers=("rax", "xmm0"),
        ).build()

        self.assertTrue(result.valid, result.diagnostics)
        self.assertTrue(result.complete, result.diagnostics)
        first_instruction_blocks = [
            block
            for block in result.blocks
            if block.instruction_address == function.address and block.reachable
        ]
        self.assertGreater(len(first_instruction_blocks), 1)
        self.assertTrue(
            any(
                len(block.predecessors) > 1 and block.phi_outputs
                for block in first_instruction_blocks
            )
        )

    def test_unresolved_indirect_branch_reports_incomplete_cfg(self) -> None:
        raw = bytes.fromhex("ff e0 c3")  # jmp eax; ret (target not recovered)
        analyzer, function = self._analyze(raw, "x86")
        result = analyzer.ssa(function)

        self.assertTrue(result.valid, result.diagnostics)
        self.assertFalse(result.complete)
        self.assertTrue(
            any(
                "unresolved indirect p-code branch" in item
                for item in result.diagnostics
            )
        )

    def test_loop_header_gets_phi_for_carried_register_x86_and_x64(self) -> None:
        cases = (
            (
                "x86",
                bytes.fromhex("b8 00 00 00 00 40 83 f8 03 7c fa c3"),
            ),
            (
                "x86_64",
                bytes.fromhex("48 89 f8 48 83 c0 01 48 85 c0 75 f7 c3"),
            ),
        )
        for arch, raw in cases:
            with self.subTest(arch=arch):
                analyzer, function = self._analyze(raw, arch)
                result = analyzer.ssa(function)

                self.assertTrue(result.valid, result.diagnostics)
                loop_phis = [
                    phi
                    for phi in result.phis
                    if phi.location.space == "register" and len(phi.incoming) >= 2
                ]
                self.assertTrue(loop_phis)
                self.assertTrue(
                    any(
                        predecessor > phi.block_id
                        for phi in loop_phis
                        for predecessor, _value in phi.incoming
                        if predecessor is not None
                    )
                )

    def test_partial_register_aliasing_uses_byte_cell_phi(self) -> None:
        raw = bytes.fromhex("31 c0 85 ff 74 02 b0 01 04 01 c3")
        analyzer, function = self._analyze(raw, "x86_64")
        result = analyzer.ssa(function)
        al = next(
            varnode
            for translation in SleighTranslator(64).translate_instructions(
                function.instructions
            )
            for operation in translation.ops
            for varnode in (
                *operation.inputs,
                *((operation.output,) if operation.output else ()),
            )
            if varnode.space == "register" and varnode.display.upper() == "AL"
        )

        self.assertTrue(result.valid, result.diagnostics)
        self.assertTrue(
            any(
                phi.location.space == "register" and phi.location.offset == al.offset
                for phi in result.phis
            )
        )

    def test_memory_ssa_phi_merges_branching_store_effects(self) -> None:
        raw = bytes.fromhex(
            "85 ff 74 0c "
            "c7 05 f2 2f 00 00 01 00 00 00 eb 0a "
            "c7 05 e6 2f 00 00 02 00 00 00 "
            "8b 05 e0 2f 00 00 c3"
        )
        analyzer, function = self._analyze(raw, "x86_64")
        result = analyzer.ssa(function)

        memory_phis = [phi for phi in result.phis if phi.location.space == "__memory__"]
        self.assertTrue(result.valid, result.diagnostics)
        self.assertTrue(result.complete, result.diagnostics)
        self.assertEqual(len(memory_phis), 1)
        self.assertEqual(len(memory_phis[0].incoming), 2)
        self.assertTrue(
            all(
                result.value(value).kind in {"memory_write", "memory_input"}
                for _predecessor, value in memory_phis[0].incoming
            )
        )
        self.assertTrue(
            any(operation.opcode == "LOAD" for operation in result.operations)
        )
        self.assertTrue(
            any(
                operation.opcode in {"COPY", "STORE"}
                and operation.memory_output is not None
                for operation in result.operations
            )
        )

    def test_call_next_pc_thunk_preserves_the_pushed_return_address(self) -> None:
        raw = bytes.fromhex("e8 00 00 00 00 58 c3")  # call $+5; pop rax; ret
        analyzer, function = self._analyze(raw, "x86_64")
        result = analyzer.ssa(function)
        call_branch = next(
            operation
            for operation in result.operations
            if operation.instruction_address == function.address
            and operation.opcode == "BRANCH"
        )

        self.assertTrue(result.valid, result.diagnostics)
        self.assertFalse(call_branch.memory_clobber)
        self.assertFalse(call_branch.call_effect_values)
        self.assertTrue(
            any(
                operation.instruction_address == function.address + 5
                and operation.opcode == "LOAD"
                for operation in result.operations
            )
        )

    def test_call_edges_return_to_caller_and_apply_abi_effects(self) -> None:
        raw = bytes.fromhex(
            "b8 2a 00 00 00 e8 05 00 00 00 83 c0 02 c3 90 90 90 90 90 c3"
        )
        analyzer, function = self._analyze(raw, "x86_64")
        result = analyzer.ssa(function)
        call = next(
            operation
            for operation in result.operations
            if operation.instruction_address == function.address + 5
            and operation.opcode == "CALL"
        )

        self.assertTrue(result.valid, result.diagnostics)
        self.assertTrue(call.memory_clobber)
        self.assertTrue(call.call_effect_values)
        self.assertIn("param_1", dict(call.call_argument_values))
        self.assertTrue(dict(call.call_argument_values)["param_1"])
        effects = [result.value(value) for value in call.call_effect_values]
        self.assertTrue(any(value.kind == "call_return" for value in effects))
        self.assertTrue(any(value.kind == "call_stack_restore" for value in effects))
        self.assertTrue(
            any(
                operation.instruction_address == function.address + 10
                and operation.opcode == "INT_ADD"
                for operation in result.operations
            )
        )

    def test_known_noreturn_call_terminates_the_pcode_cfg(self) -> None:
        raw = bytes.fromhex(
            "b8 2a 00 00 00 e8 05 00 00 00 83 c0 02 c3 90 90 90 90 90 c3"
        )
        analyzer, function = self._analyze(raw, "x86_64")
        analyzer.no_return_addresses.add(function.address + 0xF)
        result = analyzer.ssa(function)

        self.assertTrue(result.valid, result.diagnostics)
        self.assertFalse(
            any(
                operation.instruction_address >= function.address + 10
                for operation in result.operations
            )
        )

    def test_semantic_decompiler_exposes_the_same_validated_ssa_result(self) -> None:
        raw = bytes.fromhex("b8 2a 00 00 00 c3")
        analyzer, function = self._analyze(raw, "x86_64")
        decompiler = SemanticDecompiler(analyzer)

        pseudocode = decompiler.decompile(function)

        self.assertIn("return 0x2a;", pseudocode)
        self.assertIsNotNone(decompiler.ssa_function)
        self.assertTrue(decompiler.ssa_function.valid)
        self.assertTrue(decompiler.ssa_function.complete)


if __name__ == "__main__":
    unittest.main()
