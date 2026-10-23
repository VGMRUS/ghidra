from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from minighidra import Analyzer, BinaryImage, SleighTranslator
from minighidra.decompiler import SemanticDecompiler


class SleighTranslationTests(unittest.TestCase):
    def test_sleigh_translator_exposes_register_alias_metadata(self) -> None:
        for bits, register, size in (
            (32, "eax", 4),
            (64, "rax", 8),
            (64, "xmm0", 16),
        ):
            with self.subTest(bits=bits, register=register):
                varnode = SleighTranslator(bits).registers[register]
                self.assertEqual(varnode.space, "register")
                self.assertEqual(varnode.size, size)
                self.assertTrue(varnode.display)

    def test_real_sleigh_pcode_for_x86_32(self) -> None:
        instruction = SleighTranslator(32).translate_instruction(
            bytes.fromhex("83 c0 01"),
            0x401000,
            mnemonic="add",
            op_str="eax, 1",
            expected_length=3,
        )

        self.assertTrue(instruction.translated)
        self.assertEqual(instruction.length, 3)
        self.assertIn("INT_ADD", instruction.operation_names)
        add = next(op for op in instruction.ops if op.opcode == "INT_ADD")
        self.assertIsNotNone(add.output)
        assert add.output is not None
        self.assertTrue(add.output.is_register)
        self.assertEqual(add.output.display.upper(), "EAX")
        self.assertEqual(add.output.size_bits, 32)
        self.assertTrue(any(node.is_constant for node in add.inputs))
        self.assertEqual(instruction.ops[0].opcode, "IMARK")

    def test_real_sleigh_pcode_for_x86_64(self) -> None:
        instruction = SleighTranslator(64).translate_instruction(
            bytes.fromhex("48 83 c0 01"),
            0x402000,
            mnemonic="add",
            op_str="rax, 1",
            expected_length=4,
        )

        self.assertTrue(instruction.translated)
        self.assertEqual(instruction.length, 4)
        self.assertIn("INT_ADD", instruction.operation_names)
        add = next(op for op in instruction.ops if op.opcode == "INT_ADD")
        self.assertIsNotNone(add.output)
        assert add.output is not None
        self.assertTrue(add.output.is_register)
        self.assertEqual(add.output.display.upper(), "RAX")
        self.assertEqual(add.output.size_bits, 64)
        self.assertTrue(any(node.is_constant for node in add.inputs))

    def test_memory_instruction_exposes_real_load_and_varnodes(self) -> None:
        instruction = SleighTranslator(64).translate_instruction(
            bytes.fromhex("48 8b 43 08"), 0x403000, mnemonic="mov"
        )

        load = next(op for op in instruction.ops if op.opcode == "LOAD")
        self.assertIsNotNone(load.output)
        assert load.output is not None
        self.assertTrue(load.output.is_unique)
        self.assertTrue(
            any(
                node.display.upper() == "RBX"
                for op in instruction.ops
                for node in op.inputs
            )
        )
        self.assertTrue(any(node.is_unique for node in load.inputs))

    def test_sleigh_pcode_fallback_lifts_register_integer_operations(self) -> None:
        cases = (
            (
                "bswap32",
                "x86",
                "b8 12 34 56 78 0f c8 c3",
                "return 0x12345678;",
            ),
            (
                "bswap64",
                "x86_64",
                "48 b8 01 02 03 04 05 06 07 08 48 0f c8 c3",
                "return 0x102030405060708;",
            ),
            (
                "xadd",
                "x86",
                "b8 01 00 00 00 bb 02 00 00 00 0f c1 d8 89 d8 c3",
                "return 1;",
            ),
            (
                "shld",
                "x86",
                "b8 01 00 00 00 bb 02 00 00 00 0f a4 d8 04 c3",
                "return 0x10;",
            ),
        )
        for index, (name, arch, raw, expected) in enumerate(cases):
            with (
                self.subTest(instruction=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                base = 0x406000 + index * 0x100
                path = Path(directory) / f"{name}.bin"
                path.write_bytes(bytes.fromhex(raw))
                analyzer = Analyzer(BinaryImage.load(path, raw_arch=arch, base=base))
                decompiler = SemanticDecompiler(analyzer)

                pseudocode = decompiler.decompile(analyzer.functions[base])

                self.assertIn(expected, pseudocode)
                self.assertNotIn("unsupported x86 instruction", pseudocode)

    def test_locked_memory_updates_preserve_sleigh_atomic_markers(self) -> None:
        translations = (
            (32, bytes.fromhex("f0 01 08"), "lock add"),
            (64, bytes.fromhex("f0 48 01 08"), "lock add"),
            (64, bytes.fromhex("f0 48 09 08"), "lock or"),
            (64, bytes.fromhex("f0 48 0f b1 0f"), "lock cmpxchg"),
            (32, bytes.fromhex("87 08"), "xchg"),
            (64, bytes.fromhex("48 87 08"), "xchg"),
        )
        for bits, raw, mnemonic in translations:
            with self.subTest(bits=bits, mnemonic=mnemonic):
                translated = SleighTranslator(bits).translate_instruction(
                    raw, 0x407400, mnemonic=mnemonic
                )
                markers = [
                    operation
                    for operation in translated.ops
                    if operation.opcode == "CALLOTHER"
                ]

                self.assertEqual(
                    [operation.inputs[0].offset for operation in markers],
                    [0x11, 0x12],
                )

        rip_cmpxchg = SleighTranslator(64).translate_instruction(
            bytes.fromhex("f0 48 0f b1 0d 00 20 00 00"),
            0x407400,
            mnemonic="lock cmpxchg",
        )
        self.assertTrue(
            any(
                operation.inputs[0].space == "ram" and operation.opcode == "COPY"
                for operation in rip_cmpxchg.ops
                if operation.inputs
            )
        )
        self.assertTrue(
            any(
                operation.output is not None and operation.output.space == "ram"
                for operation in rip_cmpxchg.ops
            )
        )

        cases = (
            (
                "x86",
                "8b 44 24 04 8b 4c 24 08 f0 01 08 c3",
                " + param_2)",
            ),
            (
                "x86_64",
                "48 89 f8 48 89 f1 f0 48 01 08 c3",
                " + param_2)",
            ),
            (
                "x86_64",
                "48 89 f8 48 89 f1 f0 48 09 08 c3",
                " | param_2)",
            ),
            (
                "x86_64",
                "48 89 f0 48 89 d1 f0 48 0f b1 0f c3",
                " = param_3;",
            ),
            (
                "x86_64",
                "f0 48 0f b1 0d 00 20 00 00 c3",
                " = param_4;",
            ),
            (
                "x86_64",
                "f0 48 0f b1 0d 00 20 00 00 48 8b 15 f9 1f 00 00 48 89 d0 c3",
                " ? param_4 : *(uint64_t *)",
            ),
            (
                "x86",
                "8b 44 24 04 8b 4c 24 08 87 08 c3",
                " = param_2;",
            ),
            (
                "x86_64",
                "48 89 f8 48 89 f1 48 87 08 c3",
                " = param_2;",
            ),
        )
        for index, (arch, raw, expected) in enumerate(cases):
            with (
                self.subTest(arch=arch, expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                base = 0x407500 + index * 0x100
                path = Path(directory) / "locked-update.bin"
                path.write_bytes(bytes.fromhex(raw))
                analyzer = Analyzer(BinaryImage.load(path, raw_arch=arch, base=base))
                decompiler = SemanticDecompiler(analyzer)

                pseudocode = decompiler.decompile(analyzer.functions[base])

                self.assertIn("LOCK();", pseudocode)
                self.assertIn("UNLOCK();", pseudocode)
                self.assertIn(expected, pseudocode)
                self.assertIn("*(uint", pseudocode)
                self.assertNotIn("unsupported x86 instruction", pseudocode)
                if "f0 48 0f b1 0d" in raw:
                    self.assertIn("pcode_load_64_", pseudocode)
                    self.assertLess(
                        pseudocode.index(" = *(uint64_t *)"),
                        pseudocode.index("if ("),
                    )
                locked_instruction = next(
                    item
                    for item in decompiler.sleigh_instructions
                    if item.mnemonic in {"lock add", "lock or", "lock cmpxchg", "xchg"}
                )
                self.assertIn("CALLOTHER", locked_instruction.operation_names)

    def test_sleigh_internal_branches_and_bit_scan_semantics(self) -> None:
        instructions = (
            (32, bytes.fromhex("0f bc c3"), "bsf"),
            (32, bytes.fromhex("0f bd c3"), "bsr"),
            (64, bytes.fromhex("48 0f bc c3"), "bsf"),
            (64, bytes.fromhex("48 0f bd c3"), "bsr"),
        )
        for bits, raw, mnemonic in instructions:
            with self.subTest(bits=bits, mnemonic=mnemonic):
                instruction = SleighTranslator(bits).translate_instruction(
                    raw, 0x407800, mnemonic=mnemonic
                )

                self.assertTrue(instruction.translated)
                self.assertIn("CBRANCH", instruction.operation_names)
                self.assertIn("BRANCH", instruction.operation_names)
                for operation in instruction.ops:
                    if operation.opcode in {"BRANCH", "CBRANCH"}:
                        self.assertIsNone(operation.output)
                        self.assertEqual(operation.inputs[0].space, "const")

        cases = (
            (
                "bsf32-constant",
                "x86",
                "bb 00 01 00 00 0f bc c3 c3",
                "return 8;",
                None,
            ),
            (
                "bsr32-constant",
                "x86",
                "bb 00 01 00 00 0f bd c3 c3",
                "return 8;",
                None,
            ),
            (
                "bsf32-zero",
                "x86",
                "31 db 0f bc c3 c3",
                "return 0;",
                None,
            ),
            (
                "bsr32-zero",
                "x86",
                "31 db 0f bd c3 c3",
                "return 0x1f;",
                None,
            ),
            (
                "bsf64-constant",
                "x86_64",
                "48 bb 00 01 00 00 00 00 00 00 48 0f bc c3 c3",
                "return 8;",
                None,
            ),
            (
                "bsr64-constant",
                "x86_64",
                "48 bb 00 01 00 00 00 00 00 00 48 0f bd c3 c3",
                "return 8;",
                None,
            ),
            (
                "bsf64-zero",
                "x86_64",
                "31 db 48 0f bc c3 c3",
                "return 0;",
                None,
            ),
            (
                "bsr64-zero",
                "x86_64",
                "31 db 48 0f bd c3 c3",
                "return 0x3f;",
                None,
            ),
            (
                "bsf64-parameter",
                "x86_64",
                "48 0f bc c7 c3",
                "__builtin_ctzll",
                "uint64_t param_1",
            ),
            (
                "bsr64-parameter",
                "x86_64",
                "48 0f bd c7 c3",
                "__builtin_clzll",
                "uint64_t param_1",
            ),
            (
                "bsf64-memory",
                "x86_64",
                "48 0f bc 07 c3",
                "__builtin_ctzll",
                None,
            ),
            (
                "bsf64-zero-flag",
                "x86_64",
                "48 0f bc c7 0f 94 c3 0f b6 db 89 d8 c3",
                "(param_1 == 0)",
                "uint64_t param_1",
            ),
            (
                "cmpxchg32-memory",
                "x86",
                "0f b1 0b c3",
                "pcode_load_32_",
                None,
            ),
            (
                "cmpxchg64-memory",
                "x86_64",
                "48 0f b1 0b c3",
                "pcode_load_64_",
                None,
            ),
        )
        for index, (name, arch, raw, expected, signature) in enumerate(cases):
            with (
                self.subTest(instruction=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                base = 0x408000 + index * 0x100
                path = Path(directory) / f"{name}.bin"
                path.write_bytes(bytes.fromhex(raw))
                analyzer = Analyzer(BinaryImage.load(path, raw_arch=arch, base=base))
                decompiler = SemanticDecompiler(analyzer)

                pseudocode = decompiler.decompile(analyzer.functions[base])

                self.assertIn(expected, pseudocode)
                self.assertNotIn("unsupported x86 instruction", pseudocode)
                if signature is not None:
                    self.assertIn(signature, pseudocode)
                    self.assertIn("param_1", decompiler._used_parameter_names)
                if name == "bsf64-memory":
                    self.assertIn("pcode_load_64_", pseudocode)
                    flow = next(
                        item
                        for item in decompiler.paramid_graph.instruction_flows
                        if item.address == base
                    )
                    self.assertTrue(flow.read_memory)
                    self.assertTrue(
                        any(operation.opcode == "LOAD" for operation in flow.sleigh_ops)
                    )
                if name.startswith("cmpxchg"):
                    self.assertIn("if (", pseudocode)
                    flow = next(
                        item
                        for item in decompiler.paramid_graph.instruction_flows
                        if item.address == base
                    )
                    self.assertTrue(flow.read_memory)
                    self.assertTrue(flow.written_memory)
                    self.assertTrue(
                        any(
                            operation.opcode == "STORE" for operation in flow.sleigh_ops
                        )
                    )

    def test_sleigh_pcode_fallback_tracks_flags_and_stack_memory(self) -> None:
        cases = (
            (
                "rol-carry",
                "b8 01 00 00 80 c1 c0 01 0f 92 c0 0f b6 c0 c3",
                "return 1;",
            ),
            (
                "rol-overflow",
                "b8 01 00 00 80 c1 c0 01 0f 90 c0 0f b6 c0 c3",
                "return 1;",
            ),
            (
                "rol-preserved-zf",
                "31 c0 b8 01 00 00 80 c1 c0 01 0f 94 c0 0f b6 c0 c3",
                "return 1;",
            ),
            (
                "bts-stack",
                "6a 00 0f ba 2c 24 01 58 c3",
                "return 2;",
            ),
            (
                "bts-absolute",
                "0f ba 2d 00 20 00 00 01 c3",
                "*(uint32_t *)(0x2000) =",
            ),
        )
        for index, (name, raw, expected) in enumerate(cases):
            with (
                self.subTest(instruction=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                base = 0x407000 + index * 0x100
                path = Path(directory) / f"{name}.bin"
                path.write_bytes(bytes.fromhex(raw))
                analyzer = Analyzer(BinaryImage.load(path, raw_arch="x86", base=base))
                decompiler = SemanticDecompiler(analyzer)

                pseudocode = decompiler.decompile(analyzer.functions[base])

                self.assertIn(expected, pseudocode)
                self.assertNotIn("unsupported x86 instruction", pseudocode)
                if name == "bts-stack":
                    bts = next(
                        item
                        for item in decompiler.sleigh_instructions
                        if item.mnemonic == "bts"
                    )
                    self.assertIn("LOAD", bts.operation_names)
                    self.assertIn("STORE", bts.operation_names)
                    reduced_ops = [
                        operation
                        for operation in decompiler.paramid_graph.ops
                        if operation.address == bts.address
                    ]
                    self.assertTrue(
                        any(
                            operation.opcode == "LOAD_STACK"
                            for operation in reduced_ops
                        )
                    )
                    self.assertTrue(
                        any(
                            operation.opcode == "STORE_STACK"
                            for operation in reduced_ops
                        )
                    )
                elif name == "bts-absolute":
                    bts = next(
                        item
                        for item in decompiler.sleigh_instructions
                        if item.mnemonic == "bts"
                    )
                    self.assertTrue(
                        any(
                            operation.output is not None
                            and operation.output.space == "ram"
                            for operation in bts.ops
                        )
                    )

    def test_overlapping_stack_stores_and_loads_merge_by_byte(self) -> None:
        cases = (
            (
                "partial-store-32",
                "x86",
                "6a 00 c6 44 24 01 7f 8b 04 24 c3",
                ("return 0x7f00;",),
            ),
            (
                "partial-store-64",
                "x86_64",
                (
                    "48 b8 88 77 66 55 44 33 22 11 48 83 ec 08 "
                    "48 89 04 24 c6 44 24 01 7f 48 8b 04 24 "
                    "48 83 c4 08 c3"
                ),
                ("return 0x1122334455667f88;",),
            ),
            (
                "branch-partial-stores",
                "x86",
                "6a 00 85 c0 74 07 c6 44 24 01 11 eb 05 c6 44 24 02 22 58 c3",
                (
                    "phi_stack__2_8_loc_",
                    "phi_stack__3_8_loc_",
                    "= 0x11;",
                    "= 0x22;",
                ),
            ),
            (
                "unaligned-x86-parameter-byte",
                "x86",
                "0f b6 44 24 05 c3",
                ("param_1", ">> 8"),
            ),
            (
                "unaligned-x64-stack-parameter-byte",
                "x86_64",
                "0f b6 44 24 09 c3",
                ("param_7", ">> 8"),
            ),
        )
        for index, (name, arch, raw, expected) in enumerate(cases):
            with (
                self.subTest(case=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                base = 0x409000 + index * 0x100
                path = Path(directory) / f"{name}.bin"
                path.write_bytes(bytes.fromhex(raw))
                analyzer = Analyzer(BinaryImage.load(path, raw_arch=arch, base=base))
                decompiler = SemanticDecompiler(analyzer)

                pseudocode = decompiler.decompile(analyzer.functions[base])

                for token in expected:
                    self.assertIn(token, pseudocode)
                self.assertNotIn("unsupported x86 instruction", pseudocode)
                self.assertNotIn("uint24_t", pseudocode)
                self.assertNotIn("uint40_t", pseudocode)

    def test_absolute_memory_forwarding_merges_and_invalidates(self) -> None:
        cases = (
            (
                "mov-store-then-load",
                "x86",
                "c7 05 00 20 00 00 78 56 34 12 a1 00 20 00 00 c3",
                ("return 0x12345678;",),
                ("unsupported x86 instruction",),
            ),
            (
                "sleigh-store-then-load",
                "x86",
                "0f ba 2d 00 20 00 00 01 a1 00 20 00 00 c3",
                ("| 2", "0x2000"),
                ("unsupported x86 instruction",),
            ),
            (
                "cfg-memory-merge",
                "x86",
                (
                    "85 c0 74 0c c7 05 00 20 00 00 01 00 00 00 eb 0a "
                    "c7 05 00 20 00 00 02 00 00 00 a1 00 20 00 00 c3"
                ),
                ("phi_memory_2000_32_loc_", "= 1;", "= 2;"),
                ("unsupported x86 instruction",),
            ),
            (
                "dynamic-store-kills-known-global",
                "x86",
                "c7 05 00 20 00 00 78 56 34 12 c7 00 02 00 00 00 a1 00 20 00 00 c3",
                ("return *(uint32_t *)(0x2000);",),
                ("return 0x12345678;",),
            ),
            (
                "call-kills-known-global",
                "x86",
                "c7 05 00 20 00 00 78 56 34 12 ff d0 a1 00 20 00 00 c3",
                ("return *(uint32_t *)(0x2000);",),
                ("return 0x12345678;",),
            ),
            (
                "dynamic-store-kills-aliased-stack-local",
                "x86",
                (
                    "83 ec 04 c7 04 24 78 56 34 12 c7 00 02 00 00 00 "
                    "8b 04 24 83 c4 04 c3"
                ),
                ("return local_4;",),
                ("return 0x12345678;",),
            ),
            (
                "call-kills-escaped-stack-local",
                "x86_64",
                (
                    "48 83 ec 08 48 c7 04 24 78 56 34 12 48 8d 3c 24 "
                    "ff d0 48 8b 04 24 48 83 c4 08 c3"
                ),
                ("return local_8;", "(&local_8"),
                ("return 0x12345678;",),
            ),
            (
                "rep-zero-count-preserves-memory",
                "x86",
                (
                    "c7 05 04 20 00 00 78 56 34 12 be 00 20 00 00 "
                    "bf 04 20 00 00 31 c9 f3 a5 a1 04 20 00 00 c3"
                ),
                ("return 0x12345678;",),
                (),
            ),
            (
                "rep-unknown-count-invalidates-memory",
                "x86",
                (
                    "c7 05 04 20 00 00 78 56 34 12 be 00 20 00 00 "
                    "bf 04 20 00 00 f3 a5 a1 04 20 00 00 c3"
                ),
                ("return *(uint32_t *)(0x2004);",),
                ("return 0x12345678;",),
            ),
        )
        for index, (name, raw_arch, raw, expected, absent) in enumerate(cases):
            with (
                self.subTest(case=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                base = 0x40A000 + index * 0x100
                path = Path(directory) / f"{name}.bin"
                path.write_bytes(bytes.fromhex(raw))
                analyzer = Analyzer(
                    BinaryImage.load(path, raw_arch=raw_arch, base=base)
                )
                pseudocode = SemanticDecompiler(analyzer).decompile(
                    analyzer.functions[base]
                )

                for token in expected:
                    self.assertIn(token, pseudocode)
                for token in absent:
                    self.assertNotIn(token, pseudocode)

    def test_sleigh_float_fallback_and_rounding_userops(self) -> None:
        cases = (
            (
                "sqrtss-x86",
                "x86",
                "b8 00 00 80 40 66 0f 6e c8 f3 0f 51 c1 66 0f 7e c0 c3",
                "return 0x40000000;",
            ),
            (
                "sqrtss-x86_64",
                "x86_64",
                "b8 00 00 80 40 66 0f 6e c8 f3 0f 51 c1 66 0f 7e c0 c3",
                "return 0x40000000;",
            ),
            (
                "sqrtsd-x86_64",
                "x86_64",
                (
                    "48 b8 00 00 00 00 00 00 10 40 66 48 0f 6e c8 "
                    "f2 0f 51 c1 66 48 0f 7e c0 c3"
                ),
                "return 0x4000000000000000;",
            ),
            (
                "addps-lane-a",
                "x86_64",
                (
                    "b8 00 00 80 3f 66 0f 6e c0 b8 00 00 00 40 "
                    "66 0f 6e c8 0f 58 c1 66 0f 7e c0 c3"
                ),
                "return 0x40400000;",
            ),
            (
                "mulps-lane-a",
                "x86_64",
                (
                    "b8 00 00 00 40 66 0f 6e c0 b8 00 00 40 40 "
                    "66 0f 6e c8 0f 59 c1 66 0f 7e c0 c3"
                ),
                "return 0x40c00000;",
            ),
            (
                "cvtps2pd-unique-subpiece",
                "x86_64",
                ("b8 00 00 c0 3f 66 0f 6e c8 0f 5a c1 66 48 0f 7e c0 c3"),
                "return 0x3ff8000000000000;",
            ),
            (
                "cvttps2dq-lane-a",
                "x86_64",
                ("b8 33 33 f3 3f 66 0f 6e c8 f3 0f 5b c1 66 0f 7e c0 c3"),
                "return 1;",
            ),
            (
                "roundss-nearest",
                "x86_64",
                (
                    "b8 00 00 c0 3f 66 0f 6e c8 31 c0 66 0f 6e c0 "
                    "66 0f 3a 0a c1 00 66 0f 7e c0 c3"
                ),
                "return 0x40000000;",
            ),
            (
                "roundss-toward-zero",
                "x86_64",
                (
                    "b8 00 00 e0 3f 66 0f 6e c8 31 c0 66 0f 6e c0 "
                    "66 0f 3a 0a c1 03 66 0f 7e c0 c3"
                ),
                "return 0x3f800000;",
            ),
            (
                "roundss-negative-nearest-zero-sign",
                "x86_64",
                (
                    "b8 00 00 00 bf 66 0f 6e c8 31 c0 66 0f 6e c0 "
                    "66 0f 3a 0a c1 00 66 0f 7e c0 c3"
                ),
                "return 0x80000000;",
            ),
            (
                "roundss-negative-floor",
                "x86_64",
                (
                    "b8 00 00 00 bf 66 0f 6e c8 31 c0 66 0f 6e c0 "
                    "66 0f 3a 0a c1 01 66 0f 7e c0 c3"
                ),
                "return 0xbf800000;",
            ),
            (
                "roundss-negative-ceil-zero-sign",
                "x86_64",
                (
                    "b8 00 00 00 bf 66 0f 6e c8 31 c0 66 0f 6e c0 "
                    "66 0f 3a 0a c1 02 66 0f 7e c0 c3"
                ),
                "return 0x80000000;",
            ),
            (
                "roundss-negative-trunc-zero-sign",
                "x86_64",
                (
                    "b8 00 00 00 bf 66 0f 6e c8 31 c0 66 0f 6e c0 "
                    "66 0f 3a 0a c1 03 66 0f 7e c0 c3"
                ),
                "return 0x80000000;",
            ),
            (
                "roundps-lane-a",
                "x86_64",
                (
                    "b8 00 00 c0 3f 66 0f 6e c8 31 c0 66 0f 6e c0 "
                    "66 0f 3a 08 c1 00 66 0f 7e c0 c3"
                ),
                "return 0x40000000;",
            ),
            (
                "roundpd-lane-a",
                "x86_64",
                (
                    "48 b8 00 00 00 00 00 00 f8 3f 66 48 0f 6e c8 "
                    "31 c0 66 0f 6e c0 66 0f 3a 09 c1 00 "
                    "66 48 0f 7e c0 c3"
                ),
                "return 0x4000000000000000;",
            ),
            (
                "roundss-memory-source",
                "x86_64",
                "66 0f 3a 0a 05 00 20 00 00 00 66 0f 7e c0 c3",
                "nearbyintl(",
            ),
            (
                "roundss-preserve-upper-lanes",
                "x86_64",
                ("b8 00 00 c0 3f 66 0f 6e c8 66 0f 3a 0a c1 00 66 48 0f 7e c0 c3"),
                "float_param_1",
            ),
            (
                "roundss-memory-forward",
                "x86_64",
                (
                    "c7 05 00 20 00 00 00 00 c0 3f "
                    "66 0f 3a 0a 05 f6 1f 00 00 00 "
                    "66 0f 7e c0 c3"
                ),
                "return 0x40000000;",
            ),
            (
                "roundps-memory-source",
                "x86_64",
                "66 0f 3a 08 05 00 20 00 00 00 66 0f 7e c0 c3",
                "nearbyintl(",
            ),
            (
                "roundpd-memory-source",
                "x86_64",
                "66 0f 3a 09 05 00 20 00 00 00 66 48 0f 7e c0 c3",
                "nearbyintl(",
            ),
            (
                "roundsd-memory-source",
                "x86_64",
                "66 0f 3a 0b 05 00 20 00 00 00 66 48 0f 7e c0 c3",
                "nearbyintl(",
            ),
            (
                "roundps-two-lanes",
                "x86_64",
                (
                    "b8 00 00 c0 3f 66 0f 6e c8 b9 00 00 60 40 "
                    "66 0f 3a 22 c9 01 66 0f 3a 08 c1 00 "
                    "66 48 0f 7e c0 c3"
                ),
                "return 0x4080000040000000;",
            ),
            (
                "roundpd-two-lanes-high",
                "x86_64",
                (
                    "48 b8 00 00 00 00 00 00 f8 3f 66 48 0f 6e c8 "
                    "48 b9 00 00 00 00 00 00 0c 40 "
                    "66 48 0f 3a 22 c9 01 66 0f 3a 09 c1 00 "
                    "66 0f 73 d8 08 66 48 0f 7e c0 c3"
                ),
                "return 0x4010000000000000;",
            ),
            (
                "roundps-branch-path",
                "x86_64",
                (
                    "85 ff 74 1c b8 00 00 c0 3f 66 0f 6e c8 "
                    "b9 00 00 60 40 66 0f 3a 22 c9 01 "
                    "66 0f 3a 08 c1 00 eb 09 b8 00 00 40 3f "
                    "66 0f 6e c0 66 48 0f 7e c0 c3"
                ),
                "phi_xmm0_loc_",
            ),
            (
                "minss-taken-branch",
                "x86_64",
                (
                    "b8 00 00 80 3f 66 0f 6e c0 b8 00 00 80 40 "
                    "66 0f 6e c8 f3 0f 5d c1 66 0f 7e c0 c3"
                ),
                "return 0x3f800000;",
            ),
            (
                "minss-fallthrough-branch",
                "x86_64",
                (
                    "b8 00 00 80 40 66 0f 6e c0 b8 00 00 80 3f "
                    "66 0f 6e c8 f3 0f 5d c1 66 0f 7e c0 c3"
                ),
                "return 0x3f800000;",
            ),
        )
        for index, (name, arch, raw, expected) in enumerate(cases):
            with (
                self.subTest(instruction=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                base = 0x40B000 + index * 0x100
                path = Path(directory) / f"{name}.bin"
                path.write_bytes(bytes.fromhex(raw))
                analyzer = Analyzer(BinaryImage.load(path, raw_arch=arch, base=base))
                decompiler = SemanticDecompiler(analyzer)
                pseudocode = decompiler.decompile(analyzer.functions[base])

                self.assertIn(expected, pseudocode)
                if name == "roundps-branch-path":
                    self.assertIn("= 0x4080000040000000;", pseudocode)
                    self.assertIn("= 0x3f400000;", pseudocode)
                self.assertNotIn("unsupported x86 instruction", pseudocode)
                if name == "roundss-preserve-upper-lanes":
                    self.assertIn("float_param_1", pseudocode)
                else:
                    self.assertNotIn("float_param_", pseudocode)
                if name in {"roundps-branch-path", "roundss-preserve-upper-lanes"}:
                    self.assertIn("param_1", pseudocode)
                else:
                    self.assertNotIn("param_1", pseudocode)
                lifted = next(
                    item
                    for item in decompiler.sleigh_instructions
                    if item.mnemonic.startswith("sqrt")
                    or item.mnemonic.startswith("round")
                    or item.mnemonic
                    in {"minss", "addps", "mulps", "cvtps2pd", "cvttps2dq"}
                )
                if name.startswith("sqrt"):
                    self.assertIn("FLOAT_SQRT", lifted.operation_names)
                elif name.startswith("round"):
                    self.assertIn("CALLOTHER", lifted.operation_names)
                    if name.endswith(("memory-source", "memory-forward")):
                        userop = next(
                            operation
                            for operation in lifted.ops
                            if operation.opcode == "CALLOTHER"
                        )
                        self.assertEqual(userop.inputs[2].space, "ram")
                elif name.startswith("minss"):
                    self.assertIn("FLOAT_LESS", lifted.operation_names)
                    self.assertIn("CBRANCH", lifted.operation_names)
                elif name.startswith("mulps"):
                    self.assertIn("FLOAT_MULT", lifted.operation_names)
                elif name.startswith("cvtps"):
                    self.assertIn("FLOAT_FLOAT2FLOAT", lifted.operation_names)
                elif name.startswith("cvtt"):
                    self.assertIn("FLOAT_TRUNC", lifted.operation_names)
                else:
                    self.assertIn("FLOAT_ADD", lifted.operation_names)

    def test_translation_errors_are_reported_without_raising(self) -> None:
        instruction = SleighTranslator(64).translate_instruction(
            bytes.fromhex("ff ff"), 0x403000, expected_length=2
        )

        self.assertFalse(instruction.translated)
        self.assertEqual(instruction.ops, ())
        self.assertIn("BadDataError", instruction.error or "")

    def test_empty_instruction_bytes_are_reported_without_raising(self) -> None:
        instruction = SleighTranslator(32).translate_instruction(b"", 0x403000)

        self.assertFalse(instruction.translated)
        self.assertEqual(instruction.ops, ())
        self.assertIn("empty instruction", instruction.error or "")

    def test_decompiler_attaches_sleigh_ops_per_instruction_for_both_modes(
        self,
    ) -> None:
        samples = (
            ("x86", 32, bytes.fromhex("83 c0 01 c3"), 0x404000),
            ("x86_64", 64, bytes.fromhex("48 83 c0 01 c3"), 0x405000),
        )
        for raw_arch, bits, code, base in samples:
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"sleigh-{bits}.bin"
                path.write_bytes(code)
                image = BinaryImage.load(path, raw_arch=raw_arch, base=base)
                analyzer = Analyzer(image)
                function = analyzer.functions[base]
                decompiler = SemanticDecompiler(analyzer)

                decompiler.decompile(function)

                self.assertEqual(len(decompiler.sleigh_instructions), 2)
                self.assertEqual(
                    decompiler.sleigh_instructions[0].operation_names[0], "IMARK"
                )
                self.assertIn(
                    "INT_ADD", decompiler.sleigh_instructions[0].operation_names
                )
                flows = {
                    flow.address: flow
                    for flow in decompiler.paramid_graph.instruction_flows
                }
                self.assertEqual(
                    flows[base].sleigh_ops,
                    decompiler.sleigh_instructions[0].ops,
                )
                self.assertIsNone(flows[base].sleigh_error)
                self.assertGreater(decompiler.paramid_graph.sleigh_operation_count, 0)


if __name__ == "__main__":
    unittest.main()
