from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from minighidra import Analyzer, BinaryImage, ProjectState
from minighidra.cli import Console
from minighidra.image import (
    _collect_macho_function_starts,
    _collect_pe_exception_function_ranges,
    _collect_pe_tls_callbacks,
    _parse_pe_unwind_info,
)
from minighidra.noreturn import NoReturnCatalog
from minighidra.model import Section


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
        payload = code + b"hello world\x00" + "wide string".encode("utf-16le") + b"\x00\x00"
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
        self.assertEqual(analyzer.functions[base + 1].source, "Mach-O LC_FUNCTION_STARTS")
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
        self.assertEqual(_collect_pe_exception_function_ranges(binary, "PE", 32, image_base), [])
        self.assertEqual(_collect_pe_exception_function_ranges(binary, "ELF", 64, image_base), [])

    def test_pe_exception_ranges_fall_back_to_raw_pdata_when_lief_has_none(self) -> None:
        image_base = 0x140000000
        pdata_rva = 0x200
        xdata_rva = 0x400

        def runtime_function(start: int, end: int, unwind: int) -> bytes:
            return b"".join(value.to_bytes(4, "little") for value in (start, end, unwind))

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
            _collect_pe_exception_function_ranges(binary, "PE", 64, image_base, sections),
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
            Section(".xdata", image_base + unwind_rva, len(unwind_bytes), 0, unwind_bytes)
        ]

        info = _parse_pe_unwind_info(unwind_rva, sections, image_base)

        self.assertIsNotNone(info)
        self.assertEqual(info.version, 1)
        self.assertEqual(info.flags, 0)
        self.assertEqual(info.prolog_size, 4)
        self.assertEqual(info.frame_register, "RBP")
        self.assertEqual(info.frame_offset, 16)
        self.assertEqual(info.stack_allocation_bytes, 48)  # PUSH RBX (8) + ALLOC_SMALL (40)
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
            Section(".xdata", image_base + unwind_rva, len(unwind_bytes), 0, unwind_bytes)
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
        self.assertIn("UNWIND_STACK=0", analyzer.format_function(analyzer.functions[base]))

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
        self.assertEqual(report["functions"][0]["pe_unwind_info"]["stack_allocation_bytes"], 0)

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
        switch_xrefs = [xref for xref in analyzer.xrefs_from(base) if xref.kind == "switch"]
        self.assertEqual([xref.target for xref in switch_xrefs], list(targets))
        function = analyzer.functions[base]
        self.assertTrue(set(targets).issubset({item.address for item in function.instructions}))
        entry_block = next(block for block in function.blocks if block.start == base)
        self.assertEqual(entry_block.successors, targets)
        self.assertIn("computed jump to one of", analyzer.pseudocode(function))

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
        self.assertTrue(any(item.address == 0x5003 for item in analyzer.function_start_candidates))

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

        self.assertEqual(elf_catalog.match("__stack_chk_fail@GLIBC_2.4"), "stack_chk_fail")
        self.assertIsNotNone(elf_catalog.match("_exit"))
        self.assertIsNotNone(rust_catalog.match("_ZN4core9panicking5panic17h1234567890abcdefE"))
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
        path.write_bytes(bytes.fromhex("ff 15 06 00 00 00 90 90 90 90 90 90 00 00 00 00"))
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
        address = next(address for address, name in image.plt_stubs.items() if name == "abort")
        analyzer = Analyzer(image)

        function = analyzer.functions[address]
        self.assertTrue(function.is_thunk)
        self.assertEqual(function.thunk_target_name, "abort")
        self.assertTrue(function.no_return)

    def test_dynamic_noreturn_inference_requires_three_callsite_indicators(self) -> None:
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
        self.assertTrue(all("INT3" in item for item in analyzer.no_return_evidence[target]))

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
        self.assertTrue(any("terminal" in item for item in analyzer.no_return_evidence[base]))

    def test_wrapper_with_returning_path_is_not_inferred_noreturn(self) -> None:
        base = 0x7500
        target = base + 0x20
        call_address = base + 2
        displacement = target - (call_address + 5)
        code = bytearray(b"\x74\x08\xe8" + displacement.to_bytes(4, "little", signed=True))
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
        self.assertFalse(any(item.address == 0x6002 for item in analyzer.function_start_candidates))

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

    def test_pseudocode_and_cfg_are_labeled_as_heuristic(self) -> None:
        image, _ = self.make_raw_image()
        analyzer = Analyzer(image)
        function = analyzer.functions[0x400000]

        self.assertIn("no es una decompilación semántica", analyzer.pseudocode(function))
        self.assertIn("return;", analyzer.pseudocode(function))
        self.assertIn("digraph cfg", analyzer.cfg_dot(function))

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


if __name__ == "__main__":
    unittest.main()
