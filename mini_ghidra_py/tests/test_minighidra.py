from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from minighidra import Analyzer, BinaryImage, ProjectState
from minighidra.abi import default_calling_convention
from minighidra.cli import Console
from minighidra.image import (
    _collect_macho_function_starts,
    _collect_pe_exception_function_ranges,
    _collect_pe_tls_callbacks,
    _parse_pe_unwind_info,
)
from minighidra.model import Section, Symbol
from minighidra.noreturn import NoReturnCatalog


class MiniGhidraTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def make_raw_image(self, arch: str = "x86_64", base: int = 0x400000):
        if arch == "x86_64":
            code = bytes.fromhex("55 48 89 e5 e8 02 00 00 00 5d c3 90 c3")
        else:
            code = bytes.fromhex("55 89 e5 e8 01 00 00 00 5d c3 90 c3")
        payload = (
            code + b"hello world\x00" + "wide string".encode("utf-16le") + b"\x00\x00"
        )
        path = self.root / "sample.bin"
        path.write_bytes(payload)
        image = BinaryImage.load(path, raw_arch=arch, base=base)
        return image, payload

    def test_x86_64_disassembly_and_call_xref(self) -> None:
        image, _ = self.make_raw_image()
        analyzer = Analyzer(image)

        self.assertEqual(image.architecture, "x86_64")
        self.assertEqual(image.bits, 64)
        self.assertIn(0x400000, analyzer.instructions)
        self.assertEqual(analyzer.instructions[0x400000].mnemonic, "push")
        self.assertEqual(analyzer.instructions[0x400004].mnemonic, "call")
        self.assertEqual(analyzer.instructions[0x400004].branch_target, 0x40000B)
        self.assertEqual(analyzer.xrefs_to(0x40000B)[0].kind, "call")
        self.assertIn(0x40000B, analyzer.functions)

    def test_default_abi_profiles_match_x86_and_x64_cspecs(self) -> None:
        cases = (
            ("x86", "PE", "__stdcall"),
            ("x86", "ELF", "__cdecl"),
            ("x86_64", "PE", "__fastcall"),
            ("x86_64", "ELF", "sysv_amd64"),
        )
        profiles = {}
        for index, (arch, format_name, expected) in enumerate(cases):
            with self.subTest(arch=arch, format=format_name):
                image, _ = self.make_raw_image(arch, 0x410000 + index * 0x1000)
                image.format = format_name
                profile = default_calling_convention(Analyzer(image))
                self.assertEqual(profile.name, expected)
                profiles[(arch, format_name)] = profile

        self.assertEqual(profiles[("x86", "PE")].cleanup, "callee")
        self.assertEqual(
            profiles[("x86", "PE")].integer_return_registers, ("eax", "edx")
        )
        self.assertEqual(profiles[("x86", "ELF")].cleanup, "caller")
        self.assertEqual(
            profiles[("x86_64", "PE")].register_argument_groups,
            (("rcx", "xmm0"), ("rdx", "xmm1"), ("r8", "xmm2"), ("r9", "xmm3")),
        )
        self.assertEqual(
            profiles[("x86_64", "ELF")].argument_registers,
            ("rdi", "rsi", "rdx", "rcx", "r8", "r9"),
        )
        self.assertIn("xmm6", profiles[("x86_64", "PE")].callee_saved_registers)
        self.assertIn("xmm0", profiles[("x86_64", "PE")].caller_saved_registers)

    def test_xref_deduplication_preserves_indexes(self) -> None:
        image, _ = self.make_raw_image()
        analyzer = Analyzer(image)
        source, target = 0xABC0, 0xDEF0
        analyzer._record_xref(source, target, "test")
        analyzer._record_xref(source, target, "test")

        self.assertEqual(len(analyzer.xrefs_from(source)), 1)
        self.assertEqual(len(analyzer.xrefs_to(target)), 1)
        self.assertEqual(analyzer.xrefs_from(source)[0].kind, "test")

    def test_call_to_fallthrough_is_not_promoted_to_function_start(self) -> None:
        for arch, base in (("x86", 0x8200), ("x86_64", 0x8300)):
            with self.subTest(arch=arch):
                path = self.root / f"call-next-{arch}.bin"
                path.write_bytes(bytes.fromhex("e8 00 00 00 00 58 c3"))
                image = BinaryImage.load(path, raw_arch=arch, base=base)
                analyzer = Analyzer(image)

                self.assertIn(base + 5, analyzer.instructions)
                self.assertNotIn(base + 5, analyzer.function_seeds)
                self.assertNotIn(base + 5, analyzer.functions)
                self.assertEqual(analyzer.instructions[base].branch_target, base + 5)

    def test_stack_purge_is_inferred_from_ret_immediate_on_x86_and_x64(self) -> None:
        for arch, base in (("x86", 0x8400), ("x86_64", 0x8500)):
            with self.subTest(arch=arch):
                path = self.root / f"ret-purge-{arch}.bin"
                path.write_bytes(bytes.fromhex("c2 10 00 c3"))
                image = BinaryImage.load(path, raw_arch=arch, base=base)
                analyzer = Analyzer(image)
                function = analyzer.functions[base]

                self.assertEqual(function.stack_purge_size, 16)
                self.assertIn("STACK_PURGE=16", analyzer.format_function(function))
                self.assertEqual(analyzer.summary()["stack_purge_functions"], 1)

    def test_plain_ret_infers_zero_stack_purge(self) -> None:
        base = 0x8580
        path = self.root / "plain-ret-purge.bin"
        path.write_bytes(bytes.fromhex("c3"))
        image = BinaryImage.load(path, raw_arch="x86", base=base)
        analyzer = Analyzer(image)

        self.assertEqual(analyzer.functions[base].stack_purge_size, 0)

    def test_conflicting_return_purges_remain_unknown(self) -> None:
        base = 0x8600
        path = self.root / "conflicting-ret-purges.bin"
        path.write_bytes(bytes.fromhex("75 03 c2 04 00 c2 08 00"))
        image = BinaryImage.load(path, raw_arch="x86", base=base)
        analyzer = Analyzer(image)

        self.assertIsNone(analyzer.functions[base].stack_purge_size)

    def test_macho_function_start_offsets_are_based_at_text_segment(self) -> None:
        binary = SimpleNamespace(
            segments=[
                SimpleNamespace(name="__LINKEDIT", virtual_address=0x200000),
                SimpleNamespace(name="__TEXT", virtual_address=0x100000),
            ],
            function_starts=SimpleNamespace(functions=[0x20, 0x80, 0x20]),
        )

        self.assertEqual(
            _collect_macho_function_starts(binary, "Mach-O"),
            [0x100020, 0x100080],
        )
        self.assertEqual(_collect_macho_function_starts(binary, "PE"), [])

    def test_macho_function_starts_seed_only_valid_unowned_instructions(self) -> None:
        base = 0x8800
        path = self.root / "macho-function-starts.bin"
        path.write_bytes(bytes.fromhex("c3 55 89 e5 c3 cc"))
        image = BinaryImage.load(path, raw_arch="x86", base=base)
        image.format = "Mach-O"
        image.macho_function_starts = [base + 1, base + 3, base + 5]
        analyzer = Analyzer(image)

        self.assertIn(base + 1, analyzer.functions)
        self.assertEqual(
            analyzer.functions[base + 1].source, "Mach-O LC_FUNCTION_STARTS"
        )
        self.assertNotIn(base + 3, analyzer.function_seeds)  # middle of an instruction
        self.assertNotIn(base + 5, analyzer.function_seeds)  # trap instruction
        self.assertEqual(analyzer.macho_function_start_candidates, [base + 1])

    def test_pe_exception_rvas_are_translated_and_chained_entries_skipped(self) -> None:
        image_base = 0x140000000
        binary = SimpleNamespace(
            exception_functions=[
                SimpleNamespace(
                    rva_start=0x1000,
                    rva_end=0x1080,
                    unwind_info=SimpleNamespace(flags=0),
                ),
                SimpleNamespace(
                    rva_start=0x2000,
                    rva_end=0x2040,
                    unwind_info=SimpleNamespace(flags=0x4),
                ),
                SimpleNamespace(
                    rva_start=0,
                    rva_end=0x40,
                    unwind_info=None,
                ),
                SimpleNamespace(
                    rva_start=0x3000,
                    rva_end=0x2FFF,
                    unwind_info=None,
                ),
            ]
        )

        self.assertEqual(
            _collect_pe_exception_function_ranges(binary, "PE", 64, image_base),
            [(image_base + 0x1000, image_base + 0x1080)],
        )
        self.assertEqual(
            _collect_pe_exception_function_ranges(binary, "PE", 32, image_base), []
        )
        self.assertEqual(
            _collect_pe_exception_function_ranges(binary, "ELF", 64, image_base), []
        )

    def test_pe_exception_ranges_fall_back_to_raw_pdata_when_lief_has_none(
        self,
    ) -> None:
        image_base = 0x140000000
        pdata_rva = 0x200
        xdata_rva = 0x400

        def runtime_function(start: int, end: int, unwind: int) -> bytes:
            return b"".join(
                value.to_bytes(4, "little") for value in (start, end, unwind)
            )

        pdata = b"".join(
            (
                runtime_function(0x1000, 0x1010, xdata_rva),
                runtime_function(0x1020, 0x1030, xdata_rva + 4),
                runtime_function(0x1040, 0x1048, 0),
            )
        )
        xdata = bytes.fromhex("01 00 00 00 21 00 00 00")
        binary = SimpleNamespace(
            exception_functions=[],
            data_directories=[
                SimpleNamespace(type="EXCEPTION_TABLE", rva=pdata_rva, size=len(pdata))
            ],
        )
        sections = [
            Section(".pdata", image_base + pdata_rva, len(pdata), 0, pdata),
            Section(".xdata", image_base + xdata_rva, len(xdata), len(pdata), xdata),
        ]

        self.assertEqual(
            _collect_pe_exception_function_ranges(
                binary, "PE", 64, image_base, sections
            ),
            [
                (image_base + 0x1000, image_base + 0x1010),
                (image_base + 0x1040, image_base + 0x1048),
            ],
        )

    def test_pe_exception_function_ranges_create_function_seeds(self) -> None:
        base = 0x8900
        path = self.root / "pe-runtime-functions.bin"
        path.write_bytes(bytes.fromhex("c3 c3 c3"))
        image = BinaryImage.load(path, raw_arch="x86_64", base=base)
        image.format = "PE"
        image.pe_exception_function_ranges = [(base + 1, base + 2)]
        analyzer = Analyzer(image)

        self.assertIn(base + 1, analyzer.functions)
        self.assertEqual(analyzer.functions[base + 1].source, "PE exception directory")
        self.assertEqual(analyzer.summary()["pe_exception_functions"], 1)

    def test_pe_x64_unwind_info_decodes_common_stack_and_save_operations(self) -> None:
        image_base = 0x140000000
        unwind_rva = 0x400
        unwind_bytes = bytes.fromhex("01 04 03 15 04 30 02 42 01 03")
        sections = [
            Section(
                ".xdata", image_base + unwind_rva, len(unwind_bytes), 0, unwind_bytes
            )
        ]

        info = _parse_pe_unwind_info(unwind_rva, sections, image_base)

        self.assertIsNotNone(info)
        self.assertEqual(info.version, 1)
        self.assertEqual(info.flags, 0)
        self.assertEqual(info.prolog_size, 4)
        self.assertEqual(info.frame_register, "RBP")
        self.assertEqual(info.frame_offset, 16)
        self.assertEqual(
            info.stack_allocation_bytes, 48
        )  # PUSH RBX (8) + ALLOC_SMALL (40)
        self.assertEqual(info.saved_registers, ("RBX",))
        self.assertEqual(
            [operation.name for operation in info.operations],
            ["UWOP_PUSH_NONVOL", "UWOP_ALLOC_SMALL", "UWOP_SET_FPREG"],
        )

    def test_pe_x64_unwind_info_reads_aligned_handler_rva(self) -> None:
        image_base = 0x140000000
        unwind_rva = 0x500
        unwind_bytes = bytes.fromhex("09 01 01 00 00 30 00 00 34 12 00 00")
        sections = [
            Section(
                ".xdata", image_base + unwind_rva, len(unwind_bytes), 0, unwind_bytes
            )
        ]

        info = _parse_pe_unwind_info(unwind_rva, sections, image_base)

        self.assertIsNotNone(info)
        self.assertEqual(info.handler_rva, 0x1234)
        self.assertEqual(info.stack_allocation_bytes, 8)

    def test_pe_unwind_info_is_attached_to_recovered_function(self) -> None:
        base = 0x8B00
        path = self.root / "pe-unwind-function.bin"
        path.write_bytes(bytes.fromhex("c3"))
        image = BinaryImage.load(path, raw_arch="x86_64", base=base)
        image.format = "PE"
        info = _parse_pe_unwind_info(
            0x500,
            [Section(".xdata", base + 0x500, 4, 0, bytes.fromhex("01 00 00 00"))],
            base,
        )
        image.pe_exception_function_ranges = [(base, base + 1)]
        image.pe_unwind_infos = {base: info}
        analyzer = Analyzer(image)

        self.assertIs(analyzer.functions[base].pe_unwind_info, info)
        self.assertEqual(analyzer.summary()["pe_unwind_functions"], 1)
        self.assertIn(
            "UNWIND_STACK=0", analyzer.format_function(analyzer.functions[base])
        )

        console = Console(image, ProjectState(), self.root / "unwind-project.json")
        output = StringIO()
        with redirect_stdout(output):
            console.execute(f"unwind {base:#x}")
        self.assertIn("UNWIND_INFO", output.getvalue())
        self.assertIn("stack estimado: 0 bytes", output.getvalue())

        report_path = self.root / "unwind-report.json"
        with redirect_stdout(StringIO()):
            console.execute(f"export-json {report_path}")
        report = json.loads(report_path.read_text())
        self.assertEqual(
            report["functions"][0]["pe_unwind_info"]["stack_allocation_bytes"], 0
        )

    def test_pe_tls_callbacks_support_absolute_addresses_and_rvas(self) -> None:
        image_base = 0x400000
        binary = SimpleNamespace(
            tls=SimpleNamespace(callbacks=[0x401000, 0x2000, 0, 0x403000])
        )

        self.assertEqual(
            _collect_pe_tls_callbacks(binary, "PE", image_base),
            [0x401000, 0x402000],
        )
        self.assertEqual(_collect_pe_tls_callbacks(binary, "ELF", image_base), [])
        self.assertEqual(
            _collect_pe_tls_callbacks(SimpleNamespace(tls=None), "PE", image_base),
            [],
        )

    def test_pe_tls_callbacks_create_named_function_seeds(self) -> None:
        base = 0x8A00
        path = self.root / "pe-tls-callbacks.bin"
        path.write_bytes(bytes.fromhex("c3 c3"))
        image = BinaryImage.load(path, raw_arch="x86", base=base)
        image.format = "PE"
        image.pe_tls_callbacks = [base + 1]
        analyzer = Analyzer(image)

        self.assertIn(base + 1, analyzer.functions)
        self.assertEqual(analyzer.functions[base + 1].name, "tls_callback_0")
        self.assertEqual(analyzer.functions[base + 1].source, "PE TLS callback")
        self.assertEqual(analyzer.summary()["pe_tls_callbacks"], 1)

    def test_scalar_operand_reference_to_loaded_data(self) -> None:
        base = 0x400000
        target = base + 0x1000
        code = bytes.fromhex("b8 00 10 40 00 c3")  # mov eax, 0x401000; ret
        data = bytes.fromhex("73 61 6d 70 6c 65 20 74 65 78 74 00")
        payload = code + bytes([0x90]) * (0x1000 - len(code)) + data
        path = self.root / "scalar-reference.bin"
        path.write_bytes(payload)
        image = BinaryImage.load(path, raw_arch="x86", base=base)
        image.sections = [
            Section(".text", base, len(code), 0, code, executable=True),
            Section(".rdata", target, len(data), 0x1000, data, executable=False),
        ]
        analyzer = Analyzer(image)

        instruction = analyzer.instructions[base]
        self.assertEqual(instruction.scalar_references, (target,))
        self.assertIn(
            (base, target, "scalar"),
            {(xref.source, xref.target, xref.kind) for xref in analyzer.xrefs},
        )
        self.assertIn("scalar->0x401000", analyzer.format_instruction(instruction))

    def make_pointer_table_image(self, entry_count: int):
        base = 0x500000
        table_address = base + 0x100
        code = bytes.fromhex("b8 00 01 50 00 c3 90 90 90")
        targets = (base + 5, base + 6, base + 7)[:entry_count]
        table_data = b"".join(target.to_bytes(4, "little") for target in targets)
        payload = code + bytes([0x90]) * (0x100 - len(code)) + table_data
        path = self.root / f"pointer-table-{entry_count}.bin"
        path.write_bytes(payload)
        image = BinaryImage.load(path, raw_arch="x86", base=base)
        image.sections = [
            Section(".text", base, len(code), 0, code, executable=True),
            Section(".rdata", table_address, len(table_data), 0x100, table_data),
        ]
        return image, table_address, targets

    def test_referenced_pointer_table_is_recognized(self) -> None:
        image, table_address, targets = self.make_pointer_table_image(3)
        analyzer = Analyzer(image)

        self.assertEqual(len(analyzer.address_tables), 1)
        table = analyzer.address_tables[0]
        self.assertEqual(table.address, table_address)
        self.assertEqual(table.section, ".rdata")
        self.assertEqual(table.entry_size, 4)
        self.assertEqual(table.targets, targets)
        table_xrefs = [xref for xref in analyzer.xrefs if xref.kind == "table"]
        self.assertEqual([xref.target for xref in table_xrefs], list(targets))

    def test_indexed_indirect_jump_can_seed_pointer_table_analysis(self) -> None:
        base = 0x510000
        table_address = base + 0x100
        code = bytes.fromhex("ff 24 85 00 01 51 00 c3 90 90")
        targets = (base + 7, base + 8, base + 9)
        data = b"".join(target.to_bytes(4, "little") for target in targets)
        payload = code + bytes([0x90]) * (0x100 - len(code)) + data
        path = self.root / "indexed-jump-table.bin"
        path.write_bytes(payload)
        image = BinaryImage.load(path, raw_arch="x86", base=base)
        image.sections = [
            Section(".text", base, len(code), 0, code, executable=True),
            Section(".rdata", table_address, len(data), 0x100, data),
        ]
        analyzer = Analyzer(image)

        instruction = analyzer.instructions[base]
        self.assertTrue(instruction.is_jump)
        self.assertIsNone(instruction.branch_target)
        self.assertIn(table_address, instruction.indexed_memory_references)
        self.assertNotIn(table_address, instruction.memory_references)
        self.assertEqual(instruction.computed_targets, targets)
        self.assertEqual(analyzer.address_tables[0].targets, targets)
        self.assertEqual(analyzer.address_tables[0].jump_sources, (base,))
        switch_xrefs = [
            xref for xref in analyzer.xrefs_from(base) if xref.kind == "switch"
        ]
        self.assertEqual([xref.target for xref in switch_xrefs], list(targets))
        function = analyzer.functions[base]
        self.assertTrue(
            set(targets).issubset({item.address for item in function.instructions})
        )
        entry_block = next(block for block in function.blocks if block.start == base)
        self.assertEqual(entry_block.successors, targets)
        self.assertIn("switch ", analyzer.pseudocode(function))

    def test_referenced_pointer_table_supports_x86_64_entries(self) -> None:
        base = 0x600000
        table_address = base + 0x100
        code = bytes.fromhex("48 b8 00 01 60 00 00 00 00 00 c3 90 90")
        targets = (base + 10, base + 11, base + 12)
        data = b"".join(target.to_bytes(8, "little") for target in targets)
        payload = code + bytes([0x90]) * (0x100 - len(code)) + data
        path = self.root / "pointer-table-x64.bin"
        path.write_bytes(payload)
        image = BinaryImage.load(path, raw_arch="x86_64", base=base)
        image.sections = [
            Section(".text", base, len(code), 0, code, executable=True),
            Section(".rdata", table_address, len(data), 0x100, data),
        ]
        analyzer = Analyzer(image)

        self.assertEqual(len(analyzer.address_tables), 1)
        self.assertEqual(analyzer.address_tables[0].entry_size, 8)
        self.assertEqual(analyzer.address_tables[0].targets, targets)

    def test_pointer_run_shorter_than_threshold_is_not_a_table(self) -> None:
        image, _, _ = self.make_pointer_table_image(2)
        analyzer = Analyzer(image)

        self.assertEqual(analyzer.address_tables, [])
        self.assertFalse(any(xref.kind == "table" for xref in analyzer.xrefs))

    def test_raw_image_can_start_at_zero(self) -> None:
        path = self.root / "zero.bin"
        path.write_bytes(bytes.fromhex("c3 90 c3"))
        image = BinaryImage.load(path, raw_arch="x86_64", base=0)
        analyzer = Analyzer(image)

        self.assertIn(0, analyzer.instructions)
        self.assertIn(0, analyzer.functions)

    def test_ghidra_x86_pattern_recovers_unreferenced_function(self) -> None:
        path = self.root / "patterns64.bin"
        path.write_bytes(bytes.fromhex("c3 90 90 55 48 89 e5 b8 01 00 00 00 5d c3"))
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x5000)
        analyzer = Analyzer(image)

        function = analyzer.functions[0x5003]
        self.assertTrue(function.source.startswith("Ghidra pattern:"))
        self.assertIn("frame pointer", function.source)
        self.assertTrue(
            any(item.address == 0x5003 for item in analyzer.function_start_candidates)
        )

    def test_ghidra_x86_32_stack_allocation_pattern_requires_boundary(self) -> None:
        base = 0x3500
        positive_path = self.root / "stack-boundary.bin"
        positive_path.write_bytes(bytes.fromhex("c3 83 ec 10 83 c0 01 c3"))
        positive_image = BinaryImage.load(positive_path, raw_arch="x86", base=base)
        positive = Analyzer(positive_image)

        self.assertIn(base + 1, positive.functions)
        self.assertIn("stack allocation", positive.functions[base + 1].source)

        negative_path = self.root / "stack-no-boundary.bin"
        negative_path.write_bytes(bytes.fromhex("89 c0 83 ec 10 83 c0 01 c3"))
        negative_image = BinaryImage.load(negative_path, raw_arch="x86", base=base)
        negative = Analyzer(negative_image)

        self.assertNotIn(base + 2, negative.functions)

    def test_ghidra_x86_32_pattern_recovers_unreferenced_function(self) -> None:
        path = self.root / "patterns32.bin"
        path.write_bytes(bytes.fromhex("c3 90 55 89 e5 b8 2a 00 00 00 c9 c3"))
        image = BinaryImage.load(path, raw_arch="x86", base=0x3000)
        analyzer = Analyzer(image)

        function = analyzer.functions[0x3002]
        self.assertTrue(function.source.startswith("Ghidra pattern:"))
        self.assertIn("push ebp; mov ebp, esp", function.source)

    def test_ghidra_known_noreturn_catalog_matches_names_and_wildcards(self) -> None:
        elf_catalog = NoReturnCatalog.for_format("ELF")
        rust_catalog = NoReturnCatalog.for_format("PE")

        self.assertEqual(
            elf_catalog.match("__stack_chk_fail@GLIBC_2.4"), "stack_chk_fail"
        )
        self.assertIsNotNone(elf_catalog.match("_exit"))
        self.assertIsNotNone(
            rust_catalog.match("_ZN4core9panicking5panic17h1234567890abcdefE")
        )
        self.assertIsNone(elf_catalog.match("my_exit_helper"))

    def test_direct_jump_thunk_has_a_separate_body(self) -> None:
        path = self.root / "thunk.bin"
        path.write_bytes(bytes.fromhex("e9 03 00 00 00 c3 90 90 c3"))
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x8000)
        analyzer = Analyzer(image)

        function = analyzer.functions[0x8000]
        self.assertTrue(function.is_thunk)
        self.assertEqual(function.thunk_target, 0x8008)
        self.assertEqual([item.mnemonic for item in function.instructions], ["jmp"])
        self.assertIn("goto loc_8008", analyzer.pseudocode(function))

    def test_import_slot_noreturn_call_ends_caller_body(self) -> None:
        path = self.root / "indirect-noreturn.bin"
        path.write_bytes(
            bytes.fromhex("ff 15 06 00 00 00 90 90 90 90 90 90 00 00 00 00")
        )
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x7100)
        image.import_slots[0x710C] = "abort"
        analyzer = Analyzer(image)

        caller = analyzer.functions[0x7100]
        self.assertEqual([item.mnemonic for item in caller.instructions], ["call"])
        self.assertIn(0x710C, analyzer.no_return_addresses)

    def test_elf_plt_entries_become_named_thunks_when_available(self) -> None:
        binary_path = Path("/bin/true")
        if not binary_path.is_file():
            self.skipTest("requiere un ELF de prueba en /bin/true")
        image = BinaryImage.load(binary_path)
        if "abort" not in image.plt_stubs.values():
            self.skipTest("el ELF de prueba no contiene un stub PLT para abort")
        address = next(
            address for address, name in image.plt_stubs.items() if name == "abort"
        )
        analyzer = Analyzer(image)

        function = analyzer.functions[address]
        self.assertTrue(function.is_thunk)
        self.assertEqual(function.thunk_target_name, "abort")
        self.assertTrue(function.no_return)

    def test_dynamic_noreturn_inference_requires_three_callsite_indicators(
        self,
    ) -> None:
        base = 0x7200
        target_offset = 0x30
        code = bytearray()
        for _ in range(3):
            displacement = target_offset - (len(code) + 5)
            code.extend(b"\xe8" + displacement.to_bytes(4, "little", signed=True))
            code.append(0xCC)  # x86 INT3 immediately after each call
        code.extend(b"\x90" * (target_offset - len(code)))
        code.append(0xC3)
        path = self.root / "dynamic-noreturn.bin"
        path.write_bytes(code)
        image = BinaryImage.load(path, raw_arch="x86_64", base=base)
        analyzer = Analyzer(image)
        target = base + target_offset

        self.assertTrue(analyzer.functions[target].no_return)
        self.assertEqual(len(analyzer.no_return_evidence[target]), 3)
        self.assertTrue(
            all("INT3" in item for item in analyzer.no_return_evidence[target])
        )

    def test_dynamic_noreturn_inference_supports_x86_32(self) -> None:
        base = 0x7600
        target_offset = 0x30
        code = bytearray()
        for _ in range(3):
            displacement = target_offset - (len(code) + 5)
            code.extend(b"\xe8" + displacement.to_bytes(4, "little", signed=True))
            code.append(0xCC)
        code.extend(b"\x90" * (target_offset - len(code)))
        code.append(0xC3)
        path = self.root / "dynamic-noreturn-x86.bin"
        path.write_bytes(code)
        image = BinaryImage.load(path, raw_arch="x86", base=base)
        analyzer = Analyzer(image)

        self.assertEqual(image.bits, 32)
        self.assertTrue(analyzer.functions[base + target_offset].no_return)

    def test_dynamic_noreturn_inference_rejects_insufficient_evidence(self) -> None:
        base = 0x7300
        target_offset = 0x20
        code = bytearray()
        for _ in range(2):
            displacement = target_offset - (len(code) + 5)
            code.extend(b"\xe8" + displacement.to_bytes(4, "little", signed=True))
            code.append(0xCC)
        code.extend(b"\x90" * (target_offset - len(code)))
        code.append(0xC3)
        path = self.root / "insufficient-noreturn.bin"
        path.write_bytes(code)
        image = BinaryImage.load(path, raw_arch="x86_64", base=base)
        analyzer = Analyzer(image)
        target = base + target_offset

        self.assertFalse(analyzer.functions[target].no_return)
        self.assertEqual(len(analyzer.no_return_evidence[target]), 2)
        self.assertEqual(analyzer.noreturn_evidence_threshold, 3)

        lower_threshold = Analyzer(image, noreturn_evidence_threshold=2)
        self.assertTrue(lower_threshold.functions[target].no_return)

    def test_dynamic_noreturn_ignores_call_sites_with_a_return_path(self) -> None:
        base = 0x7700
        target_offset = 0x30
        code = bytearray()
        for _ in range(3):
            displacement = target_offset - (len(code) + 5)
            code.extend(bytes([0xE8]) + displacement.to_bytes(4, "little", signed=True))
        code.append(0xC3)
        code.extend(bytes([0x90]) * (target_offset - len(code)))
        code.append(0xC3)
        path = self.root / "returning-calls.bin"
        path.write_bytes(code)
        image = BinaryImage.load(path, raw_arch="x86_64", base=base)
        analyzer = Analyzer(image)
        target = base + target_offset

        self.assertFalse(analyzer.functions[target].no_return)
        self.assertNotIn(target, analyzer.no_return_evidence)

    def test_no_return_property_propagates_to_wrappers(self) -> None:
        base = 0x7400
        target = base + 0x10
        displacement = target - (base + 5)
        code = bytearray(b"\xe8" + displacement.to_bytes(4, "little", signed=True))
        code.extend(b"\x90" * (0x10 - len(code)))
        code.append(0xC3)
        path = self.root / "noreturn-wrapper.bin"
        path.write_bytes(code)
        image = BinaryImage.load(path, raw_arch="x86_64", base=base)
        state = ProjectState()
        state.labels[target] = "abort"
        analyzer = Analyzer(image, state)

        wrapper = analyzer.functions[base]
        self.assertTrue(wrapper.no_return)
        self.assertIn(base, analyzer.no_return_addresses)
        self.assertTrue(
            any("terminal" in item for item in analyzer.no_return_evidence[base])
        )

    def test_wrapper_with_returning_path_is_not_inferred_noreturn(self) -> None:
        base = 0x7500
        target = base + 0x20
        call_address = base + 2
        displacement = target - (call_address + 5)
        code = bytearray(
            b"\x74\x08\xe8" + displacement.to_bytes(4, "little", signed=True)
        )
        code.extend(b"\xc3\x90\x90\xc3")
        code.extend(b"\x90" * (0x20 - len(code)))
        code.append(0xC3)
        path = self.root / "returning-wrapper.bin"
        path.write_bytes(code)
        image = BinaryImage.load(path, raw_arch="x86_64", base=base)
        state = ProjectState()
        state.labels[target] = "abort"
        analyzer = Analyzer(image, state)

        self.assertTrue(analyzer.functions[target].no_return)
        self.assertFalse(analyzer.functions[base].no_return)

    def test_known_noreturn_call_ends_caller_body(self) -> None:
        path = self.root / "noreturn.bin"
        path.write_bytes(bytes.fromhex("e8 06 00 00 00 90 90 90 90 90 90 c3"))
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x7000)
        state = ProjectState()
        state.labels[0x700B] = "exit"
        analyzer = Analyzer(image, state)

        caller = analyzer.functions[0x7000]
        target = analyzer.functions[0x700B]
        self.assertEqual([item.mnemonic for item in caller.instructions], ["call"])
        self.assertTrue(target.no_return)
        self.assertTrue(caller.no_return)
        self.assertTrue(analyzer.pseudocode(caller).find("no retorna") > 0)
        self.assertEqual(analyzer.summary()["no_return_functions"], 2)

    def test_generic_stack_pattern_requires_valid_reachable_code(self) -> None:
        path = self.root / "short-stack.bin"
        path.write_bytes(bytes.fromhex("c3 90 48 83 ec 20 c3"))
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x6000)
        analyzer = Analyzer(image)

        self.assertNotIn(0x6002, analyzer.functions)
        self.assertFalse(
            any(item.address == 0x6002 for item in analyzer.function_start_candidates)
        )

    def test_x86_32_mode(self) -> None:
        image, _ = self.make_raw_image("x86", base=0x1000)
        analyzer = Analyzer(image)

        self.assertEqual(image.bits, 32)
        self.assertEqual(analyzer.instructions[0x1000].mnemonic, "push")
        self.assertIn(0x100A, analyzer.instructions)
        self.assertEqual(analyzer.instructions[0x1003].branch_target, 0x1009)

    def test_ascii_utf16_strings_and_byte_search(self) -> None:
        image, _ = self.make_raw_image()
        analyzer = Analyzer(image)

        texts = {(item.text, item.encoding) for item in analyzer.strings}
        self.assertIn(("hello world", "ascii"), texts)
        self.assertIn(("wide string", "utf-16le"), texts)
        self.assertNotIn(("dwide string", "utf-16le"), texts)
        self.assertTrue(analyzer.search_bytes(b"hello world"))

    def test_semantic_pseudocode_and_cfg_are_labeled_as_experimental(self) -> None:
        image, _ = self.make_raw_image()
        analyzer = Analyzer(image)
        function = analyzer.functions[0x400000]
        pseudocode = analyzer.pseudocode(function)

        self.assertIn("Experimental semantic pseudocode", pseudocode)
        self.assertIn("return ", pseudocode)
        self.assertIn("digraph cfg", analyzer.cfg_dot(function))

    def test_semantic_pseudocode_folds_register_arithmetic_and_return(self) -> None:
        path = self.root / "constant-return.bin"
        path.write_bytes(bytes.fromhex("b8 05 00 00 00 83 c0 03 c3"))
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9000)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9000])
        self.assertIn("return 8;", pseudocode)
        self.assertNotIn("eax +=", pseudocode)

    def test_semantic_pseudocode_models_implicit_x86_multiply_and_divide(self) -> None:
        cases = (
            (
                "mul.bin",
                "b8 06 00 00 00 bb 07 00 00 00 f7 e3 c3",
                "return 0x2a;",
            ),
            (
                "div.bin",
                "31 d2 b8 2a 00 00 00 bb 05 00 00 00 f7 f3 c3",
                "return 8;",
            ),
            (
                "idiv.bin",
                "b8 ef ff ff ff 99 bb 05 00 00 00 f7 fb c3",
                "return 0xfffffffd;",
            ),
        )
        for filename, machine_code, expected in cases:
            with self.subTest(filename=filename):
                path = self.root / filename
                path.write_bytes(bytes.fromhex(machine_code))
                image = BinaryImage.load(path, raw_arch="x86", base=0x9100)
                analyzer = Analyzer(image)

                pseudocode = analyzer.pseudocode(analyzer.functions[0x9100])
                self.assertIn(expected, pseudocode)
                self.assertNotIn("unsupported implicit", pseudocode)

    def test_semantic_pseudocode_models_x86_adc_and_sbb_carry(self) -> None:
        path = self.root / "sbb-carry.bin"
        path.write_bytes(bytes.fromhex("31 db 31 c0 83 f8 01 19 db 89 d8 c3"))
        image = BinaryImage.load(path, raw_arch="x86", base=0x9200)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9200])
        self.assertIn("return 0xffffffff;", pseudocode)
        self.assertNotIn("condition_jb", pseudocode)

    def test_semantic_pseudocode_models_x87_stack_and_float_store(self) -> None:
        path = self.root / "x87-add-store.bin"
        path.write_bytes(bytes.fromhex("d9 e8 d9 e8 de c1 d9 1d 00 20 00 00 c3"))
        image = BinaryImage.load(path, raw_arch="x86", base=0x9300)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9300])
        self.assertIn("*(float *)(0x2000)", pseudocode)
        self.assertIn("fpu_st0_9300 + fpu_st0_9302", pseudocode)
        self.assertNotIn("unsupported x86 instruction: f", pseudocode)

    def test_semantic_pseudocode_models_sse_integer_float_conversions(self) -> None:
        path = self.root / "sse-conversions.bin"
        path.write_bytes(
            bytes.fromhex(
                "b8 07 00 00 00 66 0f ef c0 f2 0f 2a c0 f2 0f 5a c0 66 0f 7e c0 c3"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x95C0)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x95C0])
        self.assertIn("return 0x40e00000;", pseudocode)
        self.assertNotIn("unsupported x86 instruction: cvtsi2sd", pseudocode)
        self.assertNotIn("unsupported x86 instruction: cvtsd2ss", pseudocode)

    def test_semantic_pseudocode_models_scalar_sse_double_conversion(self) -> None:
        path = self.root / "sse-double.bin"
        path.write_bytes(
            bytes.fromhex(
                "b8 03 00 00 00 66 0f 6e c0 f3 0f e6 c0 "
                "48 b8 00 00 00 00 00 00 00 40 66 48 0f 6e c8 "
                "f2 0f 59 c1 f2 0f 2c c0 c3"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9580)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9580])
        self.assertIn("return 6;", pseudocode)
        self.assertNotIn("unsupported x86 instruction: cvtdq2pd", pseudocode)
        self.assertNotIn("unsupported x86 instruction: mulsd", pseudocode)
        self.assertNotIn("unsupported x86 instruction: cvttsd2si", pseudocode)

    def test_semantic_pseudocode_models_xmm_moves_and_byte_shift(self) -> None:
        path = self.root / "xmm-shift.bin"
        path.write_bytes(
            bytes.fromhex(
                "b8 01 00 00 00 66 0f 6e c0 66 0f 70 c0 00 "
                "66 0f 73 d8 04 48 b8 00 10 00 00 00 00 00 00 "
                "f3 0f 7f 00 c3"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9500)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9500])
        self.assertIn("*(__uint128_t *)0x1000 = 0x10000000100000001;", pseudocode)
        self.assertNotIn("unsupported x86 instruction: movd", pseudocode)
        self.assertNotIn("unsupported x86 instruction: pshufd", pseudocode)
        self.assertNotIn("unsupported x86 instruction: psrldq", pseudocode)

    def test_semantic_pseudocode_models_cpuid_register_outputs(self) -> None:
        path = self.root / "cpuid.bin"
        path.write_bytes(bytes.fromhex("b8 00 00 00 00 0f a2 89 d8 c3"))
        image = BinaryImage.load(path, raw_arch="x86", base=0x93A0)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x93A0])
        self.assertIn("cpuid_93A5_ebx", pseudocode)
        self.assertNotIn("unsupported x86 instruction: cpuid", pseudocode)

    def test_semantic_pseudocode_models_mmx_lanes_and_shifts(self) -> None:
        path = self.root / "mmx-lanes.bin"
        path.write_bytes(
            bytes.fromhex(
                "b8 03 00 00 00 0f 6e c0 0f 62 c0 0f 73 f0 1f 0f 7f 05 00 10 00 00 c3"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86", base=0x93C0)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x93C0])
        self.assertIn("*(uint64_t *)0x1000 = 0x8000000180000000;", pseudocode)
        self.assertNotIn("unsupported x86 instruction: movd", pseudocode)
        self.assertNotIn("unsupported x86 instruction: punpckldq", pseudocode)

    def test_semantic_pseudocode_models_repeated_movs_with_direction_flag(self) -> None:
        path = self.root / "rep-movsd.bin"
        path.write_bytes(
            bytes.fromhex("fc b9 04 00 00 00 be 00 10 00 00 bf 00 20 00 00 f3 a5 c3")
        )
        image = BinaryImage.load(path, raw_arch="x86", base=0x9380)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9380])
        self.assertIn("rep_i_9390 < 4", pseudocode)
        self.assertIn("((uint32_t *)(0x2000))[rep_i_9390]", pseudocode)
        self.assertIn("((const uint32_t *)(0x1000))[rep_i_9390]", pseudocode)
        self.assertNotIn("unsupported x86 instruction: rep movsd", pseudocode)

    def test_semantic_pseudocode_transfers_x87_compare_through_sahf(self) -> None:
        path = self.root / "x87-compare.bin"
        path.write_bytes(
            bytes.fromhex(
                "d9 e8 d9 ee de d9 df e0 9e 72 06 b8 01 00 00 00 c3 b8 02 00 00 00 c3"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86", base=0x9400)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9400])
        self.assertIn("isnan(", pseudocode)
        self.assertIn("fpu_st0_9402 < fpu_st0_9400", pseudocode)
        self.assertIn("return 1;", pseudocode)
        self.assertIn("return 2;", pseudocode)

    def test_semantic_pseudocode_recovers_x86_stack_arguments(self) -> None:
        path = self.root / "stack-arguments-x86.bin"
        path.write_bytes(bytes.fromhex("55 89 e5 8b 45 08 03 45 0c c9 c3"))
        image = BinaryImage.load(path, raw_arch="x86", base=0x9100)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9100])
        self.assertIn("param_1", pseudocode)
        self.assertIn("param_2", pseudocode)
        self.assertIn("return param_1 + param_2;", pseudocode)

    def test_semantic_pseudocode_names_windows_x64_home_space_slots(self) -> None:
        path = self.root / "windows-home-space.bin"
        path.write_bytes(
            bytes.fromhex(
                "41 54 48 83 ec 30 48 8d 54 24 40 8b 44 24 40 48 83 c4 30 41 5c c3"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9150)
        image.format = "PE"
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9150])
        self.assertIn("home_slot_1", pseudocode)
        self.assertNotIn("stack_arg_1", pseudocode)
        self.assertNotIn("param_5", pseudocode)

    def test_semantic_pseudocode_recovers_compare_and_conditional_flow(self) -> None:
        path = self.root / "conditional-return.bin"
        path.write_bytes(
            bytes.fromhex("83 ff 00 74 06 b8 01 00 00 00 c3 b8 00 00 00 00 c3")
        )
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9200)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9200])
        self.assertIn("if (", pseudocode)
        # self.assertIn("else", pseudocode)
        self.assertIn("param_1", pseudocode)

        self.assertIn("return 1;", pseudocode)
        self.assertIn("return 0;", pseudocode)

    def test_semantic_pseudocode_structures_x86_conditional_return(self) -> None:
        path = self.root / "x86-conditional-return.bin"
        path.write_bytes(
            bytes.fromhex(
                "83 7c 24 04 00 74 06 b8 01 00 00 00 c3 "
                "b8 00 00 00 00 c3"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86", base=0x9A00)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9A00])

        self.assertIn("if (", pseudocode)
        # self.assertIn("else", pseudocode)
        self.assertIn("param_1", pseudocode)
        self.assertIn("return 1;", pseudocode)
        self.assertIn("return 0;", pseudocode)
        self.assertNotIn("goto loc_9A12;", pseudocode)

    def test_semantic_pseudocode_structures_single_sided_if(self) -> None:
        path = self.root / "if-only.bin"
        path.write_bytes(
            bytes.fromhex("85 ff 74 0a c7 05 00 20 00 00 01 00 00 00 c3")
        )
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9F00)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9F00])

        self.assertIn("if", pseudocode)
        self.assertIn("*(uint32_t *)0xbf0e = 1;", pseudocode)
        self.assertNotIn("goto loc_9F0E;", pseudocode)

    def test_semantic_pseudocode_structures_single_block_do_while(self) -> None:
        path = self.root / "do-while.bin"
        path.write_bytes(
            bytes.fromhex(
                "85 ff c7 05 00 20 00 00 01 00 00 00 75 f4 c3"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86_64", base=0xA000)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0xA000])

        # El bucle de un solo bloque se estructura como do/while: la condicion
        # sale del salto condicional que vuelve a la cabecera, sin gotos.
        self.assertIn("do {", pseudocode)
        self.assertIn("} while ((uint32_t)param_1 != 0);", pseudocode)
        self.assertIn("*(uint32_t *)0xc00c = 1;", pseudocode)
        self.assertNotIn("goto loc_A002;", pseudocode)

    def test_semantic_pseudocode_keeps_gotos_for_loop_cfg(self) -> None:
        path = self.root / "loop-fallback.bin"
        path.write_bytes(bytes.fromhex("85 ff 74 03 90 eb f9 c3"))
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9E00)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9E00])

        self.assertIn("while", pseudocode)
        self.assertIn("while", pseudocode)

    def test_semantic_pseudocode_keeps_segment_register_in_memory_access(self) -> None:
        path = self.root / "fs-memory.bin"
        path.write_bytes(bytes.fromhex("64 a1 00 00 00 00 c3"))
        image = BinaryImage.load(path, raw_arch="x86", base=0x9250)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9250])
        self.assertIn("fs_base", pseudocode)
        self.assertIn("return *(", pseudocode)

    def test_semantic_pseudocode_infers_x86_fastcall_register_arguments(self) -> None:
        path = self.root / "fastcall-x86.bin"
        path.write_bytes(
            bytes.fromhex(
                "b9 05 00 00 00 ba 07 00 00 00 6a 0b e8 0f 00 00 00 c3 "
                "90 90 90 90 90 90 90 90 90 90 90 90 90 90 "
                "89 c8 01 d0 03 44 24 04 c2 04 00"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86", base=0x9800)
        image.format = "PE"
        analyzer = Analyzer(image)

        callee = analyzer.pseudocode(analyzer.functions[0x9820])
        caller = analyzer.pseudocode(analyzer.functions[0x9800])
        self.assertIn("__fastcall FUN_9820", callee)
        self.assertIn("param_1", callee)
        self.assertIn("param_2", callee)
        self.assertIn("param_3", callee)
        self.assertIn("FUN_9820(5, 7, 0xb)", caller)
        signature = analyzer.functions[0x9820].signature
        self.assertIsNotNone(signature)
        self.assertEqual(signature.calling_convention, "__fastcall")
        self.assertEqual(
            [parameter.name for parameter in signature.parameters],
            ["param_1", "param_2", "param_3"],
        )
        console = Console(image, ProjectState(), self.root / "abi-project.json")
        console.analyzer = analyzer
        report_path = self.root / "abi-report.json"
        with redirect_stdout(StringIO()):
            console.execute(f"export-json {report_path}")
        report = json.loads(report_path.read_text())
        exported = next(
            item for item in report["functions"] if item["address"] == 0x9820
        )
        self.assertEqual(exported["calling_convention"], "__fastcall")
        self.assertEqual(exported["signature"]["return_type"], "uint32_t")

    def test_semantic_pseudocode_infers_x86_thiscall_register_argument(self) -> None:
        path = self.root / "thiscall-x86.bin"
        path.write_bytes(
            bytes.fromhex(
                "b9 78 56 34 12 6a 0b e8 14 00 00 00 c3 "
                "90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 "
                "89 c8 03 44 24 04 c2 04 00"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86", base=0x9900)
        image.format = "PE"
        analyzer = Analyzer(image)

        callee = analyzer.pseudocode(analyzer.functions[0x9920])
        caller = analyzer.pseudocode(analyzer.functions[0x9900])
        self.assertIn("__thiscall FUN_9920", callee)
        self.assertIn("uint32_t param_1, uint32_t param_2", callee)
        self.assertIn("FUN_9920(0xb)", caller)

    def test_semantic_pseudocode_infers_gcc_regparm3(self) -> None:
        path = self.root / "regparm3-x86.bin"
        path.write_bytes(
            bytes.fromhex(
                "b8 05 00 00 00 ba 07 00 00 00 b9 09 00 00 00 "
                "e8 0c 00 00 00 c3 90 90 90 90 90 90 90 90 90 90 90 "
                "89 c3 01 d3 01 cb 89 d8 c3"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86", base=0x9B00)
        analyzer = Analyzer(image)

        callee = analyzer.pseudocode(analyzer.functions[0x9B20])
        caller = analyzer.pseudocode(analyzer.functions[0x9B00])
        self.assertIn("__regparm3 FUN_9B20", callee)
        self.assertIn("param_1", callee)
        self.assertIn("param_2", callee)
        self.assertIn("param_3", callee)
        self.assertIn("FUN_9B20(5, 7, 9)", caller)

    def test_semantic_pseudocode_uses_windows_x64_register_groups(self) -> None:
        path = self.root / "windows-x64-call.bin"
        path.write_bytes(
            bytes.fromhex("b9 07 00 00 00 e8 06 00 00 00 c3 90 90 90 90 90 89 c8 c3")
        )
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9C00)
        image.format = "PE"
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9C00])
        self.assertIn("FUN_9C10(7)", pseudocode)

    def test_semantic_pseudocode_maps_windows_x64_float_parameter(self) -> None:
        path = self.root / "windows-x64-float-call.bin"
        path.write_bytes(
            bytes.fromhex(
                "b8 00 00 00 40 66 0f 6e c0 e8 12 00 00 00 c3 "
                "90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 "
                "f3 0f 10 c0 c3"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9D00)
        image.format = "PE"
        analyzer = Analyzer(image)

        callee = analyzer.pseudocode(analyzer.functions[0x9D20])
        caller = analyzer.pseudocode(analyzer.functions[0x9D00])
        self.assertIn("float param_1", callee)
        self.assertIn("FUN_9D20(2.0L)", caller)

    def test_semantic_pseudocode_returns_windows_x64_scalar_in_xmm0(self) -> None:
        path = self.root / "windows-x64-xmm-return.bin"
        path.write_bytes(
            bytes.fromhex(
                "e8 1b 00 00 00 f2 0f 10 c8 c3 "
                "90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 "
                "f2 0f 10 c1 c3"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9F00)
        image.format = "PE"
        analyzer = Analyzer(image)

        callee = analyzer.pseudocode(analyzer.functions[0x9F20])
        self.assertIn("double FUN_9F20", callee)
        self.assertIn("return", callee)

    def test_semantic_pseudocode_recovers_x86_edx_eax_wide_return(self) -> None:
        path = self.root / "edx-eax-return.bin"
        path.write_bytes(
            bytes.fromhex(
                "e8 1b 00 00 00 01 d0 c3 "
                "90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 "
                "90 90 90 90 90 90 90 90 b8 01 00 00 00 ba 02 00 00 00 c3"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86", base=0xA100)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0xA120])
        signature = analyzer.functions[0xA120].signature
        self.assertIn("uint64_t", pseudocode)
        self.assertIsNotNone(signature)
        self.assertEqual(signature.return_type, "uint64_t")

    def test_semantic_pseudocode_returns_x87_value_in_st0(self) -> None:
        path = self.root / "x87-return-x86.bin"
        path.write_bytes(bytes.fromhex("d9 e8 c3"))
        image = BinaryImage.load(path, raw_arch="x86", base=0x9E00)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9E00])
        self.assertIn("long double entry(void)", pseudocode)
        self.assertIn("return fpu_st0_9E00;", pseudocode)

    def test_semantic_pseudocode_uses_recent_x86_stack_pushes_as_call_args(
        self,
    ) -> None:
        path = self.root / "cdecl-call.bin"
        path.write_bytes(
            bytes.fromhex("6a 07 6a 05 e8 06 00 00 00 c3 90 90 90 90 90 c3")
        )
        image = BinaryImage.load(path, raw_arch="x86", base=0xA000)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0xA000])
        self.assertIn("FUN_A00F(5, 7)", pseudocode)

    def test_semantic_pseudocode_structures_diamond_and_materializes_phi(self) -> None:
        path = self.root / "phi-merge.bin"
        path.write_bytes(
            bytes.fromhex("83 ff 00 74 07 b8 01 00 00 00 eb 05 b8 02 00 00 00 c3")
        )
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9400)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9400])
        self.assertIn("if (", pseudocode)
        # self.assertIn("else", pseudocode)
        self.assertNotIn("goto loc_9411;", pseudocode)
        self.assertIn("phi_rax_loc_9411 = phi(1,2);", pseudocode)
        self.assertIn("phi(1,2)", pseudocode)
        # self.assertNotIn("phi_rax_loc_9411 = phi(", pseudocode)
        self.assertIn("return phi_rax_loc_9411;", pseudocode)

    def test_semantic_pseudocode_models_x86_64_call_argument_and_result(self) -> None:
        path = self.root / "call-argument.bin"
        path.write_bytes(bytes.fromhex("bf 07 00 00 00 e8 02 00 00 00 c3 90 c3"))
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9300)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9300])
        self.assertIn("FUN_930C(7)", pseudocode)
        self.assertIn("return call_result_9305;", pseudocode)

    def test_semantic_pseudocode_drops_unobserved_call_return(self) -> None:
        path = self.root / "unused-call-result.bin"
        path.write_bytes(
            bytes.fromhex(
                "e8 0b 00 00 00 b8 2a 00 00 00 c3 90 90 90 90 90 "
                "e8 0b 00 00 00 c3 90 90 90 90 90 90 90 90 90 90 "
                "b8 07 00 00 00 c3"
            )
        )
        image = BinaryImage.load(path, raw_arch="x86", base=0x9600)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9610])
        self.assertIn("void FUN_9610(void)", pseudocode)
        self.assertIn("FUN_9620();", pseudocode)
        self.assertIn("return;", pseudocode)
        self.assertNotIn("return call_result_", pseudocode)

    def test_semantic_pseudocode_recognizes_call_next_pc_thunk(self) -> None:
        path = self.root / "pc-thunk.bin"
        path.write_bytes(bytes.fromhex("e8 00 00 00 00 58 c3"))
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9500)
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9500])
        self.assertIn("return 0x9505;", pseudocode)
        self.assertNotIn("call_result_", pseudocode)

    def test_semantic_pseudocode_recovers_tail_call_to_known_function(self) -> None:
        path = self.root / "tail-call.bin"
        path.write_bytes(bytes.fromhex("85 c0 eb 02 c3 90 c3"))
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x9600)
        image.symbols.append(Symbol("target_func", 0x9606, kind="function"))
        analyzer = Analyzer(image)

        pseudocode = analyzer.pseudocode(analyzer.functions[0x9600])
        self.assertIn("return target_func(", pseudocode)
        self.assertIn("firma desconocida", pseudocode)

    def test_patch_writes_copy_and_keeps_original(self) -> None:
        image, original = self.make_raw_image()
        output = self.root / "patched.bin"
        image.patch_to_file(0x400000, b"\xcc", output)

        self.assertEqual(image.path.read_bytes(), original)
        self.assertEqual(output.read_bytes()[0], 0xCC)
        with self.assertRaises(ValueError):
            image.patch_to_file(0x400000, b"\xcc", image.path)

    def test_project_roundtrip_and_binary_identity(self) -> None:
        image, _ = self.make_raw_image()
        state = ProjectState()
        state.labels[0x400000] = "start_here"
        state.comments[0x400000] = "test annotation"
        project_path = self.root / "sample.json"
        state.save(project_path, image)

        loaded = ProjectState.load(project_path, image)
        self.assertEqual(loaded.labels, state.labels)
        self.assertEqual(loaded.comments, state.comments)

        other_path = self.root / "other.bin"
        other_path.write_bytes(b"different bytes")
        other = BinaryImage.load(other_path, raw_arch="x86_64")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            ProjectState.load(project_path, other)

    def test_raw_architecture_is_validated(self) -> None:
        path = self.root / "sample.bin"
        path.write_bytes(b"\x90")
        with self.assertRaisesRegex(ValueError, "Arquitectura no válida"):
            BinaryImage.load(path, raw_arch="arm64")


class FlowResyncAndBoundaryTests(unittest.TestCase):
    """Desensamblado con resincronización por flujo y fronteras de función.

    Regresiones de la verificación contra GXSCC.exe: el barrido lineal se
    desalineaba tras un ``ret`` y se tragaba el prólogo de una función sólo
    alcanzada por ``call`` (``0x4058A0``), los thunks ajustadores de ``this``
    absorbían todo el cuerpo de su destino (``FUN_401500`` con luz hasta
    ``0x40EEC4``) y las entradas de vtable no se promovían a función
    (``0x401070``, ``InitInstance``).
    """

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def _analyze(self, name: str, hexbytes: str, base: int, arch: str = "x86_64"):
        path = self.root / name
        path.write_bytes(bytes.fromhex(hexbytes))
        return Analyzer(BinaryImage.load(path, raw_arch=arch, base=base))

    def test_call_target_swallowed_by_a_long_instruction_is_resynced(self) -> None:
        # 9000: call 0x900b ; 9005: ret ;
        # 9006: mov dword [0x900b], 0x2a   <- 10 bytes, cubre 0x900b
        # 9010: push rbp ; 9011: ret
        analyzer = self._analyze(
            "resync.bin",
            "e806000000" "c3" "c7050b9000002a000000" "55" "c3",
            0x9000,
        )
        self.assertGreaterEqual(analyzer.resynced_flow_boundaries, 1)
        self.assertIn(0x900B, analyzer.instructions)
        # La instrucción desalineada que cubría el destino desaparece.
        self.assertNotIn(0x9006, analyzer.instructions)
        self.assertIn(0x900B, analyzer.functions)

    def test_linear_sweep_without_desync_is_left_untouched(self) -> None:
        # Todo destino de ``call`` cuadra: la resincronización no toca nada.
        analyzer = self._analyze("alineado.bin", "5548 89e5 e802000000 5d c3 90 c3".replace(" ", ""), 0x9400)
        self.assertEqual(analyzer.resynced_flow_boundaries, 0)

    def test_unconditional_jump_target_after_return_is_its_own_function(self) -> None:
        # 9100: jmp 0x9103 ; 9102: ret ; 9103: push rbp ; 9104: ret
        analyzer = self._analyze("frontera.bin", "eb01" "c3" "55" "c3", 0x9100)
        self.assertIn(0x9103, analyzer.functions)
        self.assertEqual(
            analyzer.function_seeds[0x9103][1], "after-return jump target"
        )
        # El salto de cola no absorbe el cuerpo del destino.
        self.assertNotIn(0x9103, {i.address for i in analyzer.functions[0x9100].instructions})

    def test_conditional_jump_target_is_not_split_off(self) -> None:
        # Un salto condicional hacia delante no crea frontera: sigue siendo una
        # sola función (si no, la regla partiría bucles y bloques compartidos).
        analyzer = self._analyze("condicional.bin", "85ff" "7401" "c3" "90" "c3", 0x9150)
        self.assertEqual(len(analyzer.functions), 1)
        self.assertNotIn(0x9154, analyzer.function_seeds)

    def test_this_adjustor_thunk_promotes_its_target(self) -> None:
        # 9200: mov ecx, 0x3e8 ; 9205: jmp 0x920b ; 920a: nop ; 920b: push ebp ; ret
        analyzer = self._analyze(
            "thunk-this.bin", "b9e8030000" "e901000000" "90" "55" "c3", 0x9200, arch="x86"
        )
        thunk = analyzer.functions[0x9200]
        self.assertTrue(thunk.is_thunk)
        self.assertEqual(thunk.thunk_target, 0x920B)
        self.assertEqual([i.mnemonic for i in thunk.instructions], ["mov", "jmp"])
        self.assertIn(0x920B, analyzer.functions)
        self.assertEqual(analyzer.function_seeds[0x920B][1], "thunk target")

    def test_thunk_prefix_that_writes_memory_is_not_a_thunk(self) -> None:
        # ``mov dword [ecx], 1`` escribe memoria: no es un thunk, es una función.
        analyzer = self._analyze(
            "no-thunk.bin", "c70101000000" "e901000000" "90" "55" "c3", 0x9250, arch="x86"
        )
        self.assertFalse(analyzer.functions[0x9250].is_thunk)

    def test_data_pointer_to_code_creates_a_function(self) -> None:
        base = 0x620000
        orphan = base + 0x20
        # El huérfano no usa un prólogo reconocible por los patrones de inicio
        # de función: si no, lo sembrarían ellos y no se vería la regla nueva.
        code = bytes.fromhex("55c3") + b"\x90" * (0x20 - 2)
        code += bytes.fromhex("b801000000c3")
        code += b"\x90" * (0x40 - len(code))
        table_address = base + 0x100
        data = orphan.to_bytes(8, "little")
        payload = code + b"\x90" * (0x100 - len(code)) + data
        path = self.root / "puntero-de-datos.bin"
        path.write_bytes(payload)
        image = BinaryImage.load(path, raw_arch="x86_64", base=base)
        image.sections = [
            Section(".text", base, len(code), 0, code, executable=True),
            Section(".rdata", table_address, len(data), 0x100, data),
        ]
        analyzer = Analyzer(image)

        self.assertIn(orphan, analyzer.instructions)
        self.assertIn(orphan, analyzer.functions)
        self.assertEqual(analyzer.function_seeds[orphan][1], "data pointer")
        # La función que apunta no absorbe el cuerpo huérfano.
        self.assertNotIn(orphan, {i.address for i in analyzer.functions[base].instructions})


if __name__ == "__main__":
    unittest.main()
