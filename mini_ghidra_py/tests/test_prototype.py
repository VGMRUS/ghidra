from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from minighidra import Analyzer, BinaryImage, FunctionParameter, FunctionSignature
from minighidra.abi import (
    SYSV_AMD64,
    WINDOWS_X64,
    WINDOWS_X64_THISCALL,
    X86_CDECL,
    X86_FASTCALL,
    X86_REGPARM3,
    X86_THISCALL,
)
from minighidra.prototype import PrototypeResolver
from minighidra.typesys import TypeDatabase, TypeKind


class CTypeDatabaseTests(unittest.TestCase):
    def test_builtin_pointer_typedef_array_and_aggregate_layout(self) -> None:
        types = TypeDatabase(64, "ELF")
        pointer = types.resolve("const int * *")
        self.assertEqual(pointer.kind, TypeKind.POINTER)
        self.assertEqual(pointer.target.kind, TypeKind.POINTER)
        self.assertEqual(pointer.target.target.name, "int")

        alias = types.typedef("Count", "unsigned long")
        self.assertEqual(types.resolve("Count"), alias)
        array = types.array("int", 3)
        self.assertEqual((array.size, array.alignment), (12, 4))

        structure = types.structure("Record", [("tag", "char"), ("value", "double")])
        self.assertEqual(structure.size, 16)
        self.assertEqual(
            [(field.name, field.offset) for field in structure.fields],
            [("tag", 0), ("value", 8)],
        )
        union = types.union("Scalar", [("integer", "int"), ("floating", "double")])
        self.assertEqual(union.size, 8)
        self.assertEqual(types.get("long double").size, 16)
        self.assertEqual(TypeDatabase(32, "ELF").get("long double").size, 12)
        self.assertEqual(TypeDatabase(64, "PE").get("long double").size, 8)

        forward_node = types.forward_structure("Node")
        with self.assertRaisesRegex(ValueError, "incompleto"):
            types.structure("InvalidNode", [("child", forward_node)])
        node_pointer = types.pointer(forward_node)
        node = types.structure("Node", [("value", "int"), ("next", node_pointer)])
        self.assertEqual(node.size, 16)
        self.assertEqual(types.resolve("struct Node").size, 16)
        self.assertEqual(node.fields[1].type.target.name, "struct Node")

    def test_pointer_spacing_and_c_default_variadic_promotions(self) -> None:
        types = TypeDatabase(32, "PE")
        self.assertEqual(types.resolve("int * *").size, 4)
        self.assertEqual(types.resolve("unsigned char").signed, False)
        self.assertEqual(types.promote_vararg("float").name, "double")
        self.assertEqual(types.promote_vararg("unsigned char").name, "int")

        function_type = types.function(
            "int", ("int", types.array("char", 4)), variadic=True
        )
        self.assertEqual(
            function_type.declaration("handler"), "int handler(int, char *, ...)"
        )
        function_pointer = types.pointer(function_type)
        self.assertEqual(
            function_pointer.declaration("handler"),
            "int (*handler)(int, char *, ...)",
        )
        self.assertEqual(types.resolve(function_pointer.name), function_pointer)
        self.assertEqual(types.promote_vararg(function_type), function_pointer)
        self.assertEqual(types.promote_vararg("short").name, "int")
        self.assertEqual(
            types.promote_vararg(types.array("char", 8)).kind, TypeKind.POINTER
        )
        small_enum = types.enum("SmallEnum", "unsigned char")
        self.assertEqual(types.promote_vararg(small_enum).name, "int")

        windows_types = TypeDatabase(32, "PE")
        self.assertEqual(windows_types.get("wchar_t").size, 2)
        self.assertEqual(windows_types.get("long double").size, 8)
        self.assertEqual(windows_types.get("long").size, 4)


class PrototypeResolverTests(unittest.TestCase):
    def test_sysv_amd64_mixed_aggregate_argument_and_split_return(self) -> None:
        types = TypeDatabase(64, "ELF")
        mixed = types.structure("Mixed", [("value", "double"), ("tag", "int")])
        pair = types.structure("Pair", [("left", "double"), ("right", "double")])
        resolver = PrototypeResolver(types, 64)

        argument_layout = resolver.resolve(
            FunctionSignature(
                "void",
                (FunctionParameter("item", mixed),),
                SYSV_AMD64.name,
            ),
            SYSV_AMD64,
        ).signature.parameters[0]
        self.assertEqual(
            [
                (location.register, location.value_offset)
                for location in argument_layout.locations
            ],
            [("xmm0", 0), ("rdi", 8)],
        )

        returned = resolver.resolve(
            FunctionSignature("Pair", (), SYSV_AMD64.name, return_data_type=pair),
            SYSV_AMD64,
        ).signature
        self.assertFalse(returned.hidden_return)
        self.assertEqual(
            [
                (location.register, location.value_offset)
                for location in returned.return_locations
            ],
            [("xmm0", 0), ("xmm1", 8)],
        )

    def test_sysv_unaligned_and_large_aggregates_use_memory(self) -> None:
        types = TypeDatabase(64, "ELF")
        packed = types.structure(
            "Packed", [("tag", "char"), ("value", "double")], packed=True, size=9
        )
        large = types.structure(
            "Large", [("a", "uint64_t"), ("b", "uint64_t"), ("c", "uint64_t")]
        )
        resolver = PrototypeResolver(types, 64)
        signature = FunctionSignature(
            "void",
            (FunctionParameter("packed", packed), FunctionParameter("large", large)),
            SYSV_AMD64.name,
        )
        resolved = resolver.resolve(signature, SYSV_AMD64).signature
        self.assertEqual(resolved.parameters[0].locations[0].kind, "stack")
        self.assertEqual(resolved.parameters[1].locations[0].kind, "stack")

    def test_sysv_hidden_return_shifts_integer_arguments(self) -> None:
        types = TypeDatabase(64, "ELF")
        large = types.structure(
            "LargeResult", [("a", "uint64_t"), ("b", "uint64_t"), ("c", "uint64_t")]
        )
        resolver = PrototypeResolver(types, 64)
        resolved = resolver.resolve(
            FunctionSignature(
                "LargeResult",
                (FunctionParameter("value", "int"),),
                SYSV_AMD64.name,
                return_data_type=large,
            ),
            SYSV_AMD64,
        ).signature
        self.assertTrue(resolved.hidden_return)
        self.assertEqual(resolved.hidden_return_location.register, "rdi")
        self.assertEqual(resolved.parameters[0].locations[0].register, "rsi")
        self.assertEqual(resolved.return_locations[0].register, "rax")

    def test_windows_x64_aggregate_pointers_hidden_return_and_this_order(self) -> None:
        types = TypeDatabase(64, "PE")
        three_bytes = types.structure(
            "ThreeBytes", [("a", "char"), ("b", "char"), ("c", "char")], size=3
        )
        sixteen_bytes = types.structure(
            "SixteenBytes", [("a", "uint64_t"), ("b", "uint64_t")]
        )
        resolver = PrototypeResolver(types, 64)

        passed = resolver.resolve(
            FunctionSignature(
                "void",
                (
                    FunctionParameter("item", three_bytes),
                    FunctionParameter("tail", "int"),
                ),
                WINDOWS_X64.name,
            ),
            WINDOWS_X64,
        ).signature
        self.assertTrue(passed.parameters[0].locations[0].indirect)
        self.assertEqual(passed.parameters[0].locations[0].register, "rcx")
        self.assertEqual(passed.parameters[1].locations[0].register, "rdx")

        returned = resolver.resolve(
            FunctionSignature(
                "SixteenBytes", (), WINDOWS_X64.name, return_data_type=sixteen_bytes
            ),
            WINDOWS_X64,
        ).signature
        self.assertTrue(returned.hidden_return)
        self.assertEqual(returned.hidden_return_location.register, "rcx")

        method = resolver.resolve(
            FunctionSignature(
                "SixteenBytes",
                (
                    FunctionParameter("this", "void *", implicit=True),
                    FunctionParameter("value", "int"),
                ),
                WINDOWS_X64_THISCALL.name,
                return_data_type=sixteen_bytes,
            ),
            WINDOWS_X64_THISCALL,
        ).signature
        self.assertEqual(method.parameters[0].locations[0].register, "rcx")
        self.assertEqual(method.hidden_return_location.register, "rdx")
        self.assertEqual(method.parameters[1].locations[0].register, "r8")

    def test_x86_gcc_and_microsoft_hidden_return_locations(self) -> None:
        gcc_types = TypeDatabase(32, "ELF")
        small_struct = gcc_types.structure("Small", [("value", "int")])
        gcc_resolver = PrototypeResolver(gcc_types, 32)
        gcc_result = gcc_resolver.resolve(
            FunctionSignature(
                "Small",
                (FunctionParameter("value", "int"),),
                X86_CDECL.name,
                return_data_type=small_struct,
            ),
            X86_CDECL,
        ).signature
        self.assertTrue(gcc_result.hidden_return)
        self.assertEqual(gcc_result.hidden_return_location.stack_offset, 4)
        self.assertEqual(gcc_result.parameters[0].locations[0].stack_offset, 8)

        regparm = (
            PrototypeResolver(gcc_types, 32)
            .resolve(
                FunctionSignature(
                    "Small",
                    (FunctionParameter("value", "int"),),
                    X86_REGPARM3.name,
                    return_data_type=small_struct,
                ),
                X86_REGPARM3,
            )
            .signature
        )
        self.assertTrue(regparm.hidden_return)
        self.assertEqual(regparm.hidden_return_location.register, "eax")
        self.assertEqual(regparm.parameters[0].locations[0].register, "edx")

        windows_types = TypeDatabase(32, "PE")
        small_struct = windows_types.structure("Small", [("value", "uint32_t")])
        large_struct = windows_types.structure(
            "Large", [("a", "uint64_t"), ("b", "uint64_t")]
        )
        win_resolver = PrototypeResolver(windows_types, 32)
        small_argument = win_resolver.resolve(
            FunctionSignature(
                "void",
                (
                    FunctionParameter("item", small_struct),
                    FunctionParameter("tail", "int"),
                ),
                X86_FASTCALL.name,
            ),
            X86_FASTCALL,
        ).signature
        self.assertEqual(small_argument.parameters[0].locations[0].register, "ecx")
        self.assertEqual(small_argument.parameters[1].locations[0].register, "edx")

        fastcall = win_resolver.resolve(
            FunctionSignature(
                "Large",
                (FunctionParameter("value", "int"),),
                X86_FASTCALL.name,
                return_data_type=large_struct,
            ),
            X86_FASTCALL,
        ).signature
        self.assertEqual(fastcall.hidden_return_location.register, "ecx")
        self.assertEqual(fastcall.parameters[0].locations[0].register, "edx")

        method = win_resolver.resolve(
            FunctionSignature(
                "Large",
                (
                    FunctionParameter("this", "void *", implicit=True),
                    FunctionParameter("value", "int"),
                ),
                X86_THISCALL.name,
                return_data_type=large_struct,
            ),
            X86_THISCALL,
        ).signature
        self.assertEqual(method.parameters[0].locations[0].register, "ecx")
        self.assertEqual(method.hidden_return_location.stack_offset, 4)
        self.assertEqual(method.parameters[1].locations[0].stack_offset, 8)

    def test_variadic_promotions_and_sysv_sse_count(self) -> None:
        types = TypeDatabase(64, "ELF")
        resolver = PrototypeResolver(types, 64)
        layout = resolver.resolve(
            FunctionSignature(
                "int",
                (FunctionParameter("format", "double"),),
                SYSV_AMD64.name,
                variadic=True,
            ),
            SYSV_AMD64,
            variadic_types=("float", "unsigned char"),
        )
        self.assertEqual(layout.signature.varargs_sse_count_register, "al")
        self.assertEqual(layout.varargs_sse_count, 2)
        self.assertEqual(
            [parameter.type_name for parameter in layout.variadic_parameters],
            ["double", "int"],
        )
        self.assertEqual(layout.variadic_parameters[0].locations[0].register, "xmm1")
        self.assertEqual(layout.variadic_parameters[1].locations[0].register, "rdi")

    def test_windows_x64_variadic_float_is_duplicated_to_integer_register(self) -> None:
        types = TypeDatabase(64, "PE")
        resolver = PrototypeResolver(types, 64)
        layout = resolver.resolve(
            FunctionSignature(
                "int",
                (FunctionParameter("count", "int"),),
                WINDOWS_X64.name,
                variadic=True,
            ),
            WINDOWS_X64,
            variadic_types=("float",),
        )
        locations = layout.variadic_parameters[0].locations
        self.assertEqual(
            {(location.kind, location.register) for location in locations},
            {("register", "xmm1"), ("register", "rdx")},
        )

    def test_x86_variadic_arguments_are_promoted_and_stack_aligned(self) -> None:
        types = TypeDatabase(32, "ELF")
        resolver = PrototypeResolver(types, 32)
        layout = resolver.resolve(
            FunctionSignature(
                "int",
                (FunctionParameter("count", "int"),),
                X86_CDECL.name,
                variadic=True,
            ),
            X86_CDECL,
            variadic_types=("float", "unsigned char"),
        )
        self.assertEqual(layout.variadic_parameters[0].type_name, "double")
        self.assertEqual(layout.variadic_parameters[0].locations[0].stack_offset, 8)
        self.assertEqual(layout.variadic_parameters[0].locations[0].size, 8)
        self.assertEqual(layout.variadic_parameters[1].type_name, "int")
        self.assertEqual(layout.variadic_parameters[1].locations[0].stack_offset, 16)

    def test_variadic_layout_rejects_nonvariadic_prototypes(self) -> None:
        resolver = PrototypeResolver(TypeDatabase(64, "ELF"), 64)
        with self.assertRaisesRegex(ValueError, "no está marcado"):
            resolver.layout_variadic_arguments(
                FunctionSignature("void", (), SYSV_AMD64.name), SYSV_AMD64, ("int",)
            )


class IntegratedPrototypeTests(unittest.TestCase):
    def test_inferred_prototype_uses_live_in_reads_and_callsite_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.bin"
            path.write_bytes(bytes.fromhex("bf 07 00 00 00 e8 01 00 00 00 c3 89 f8 c3"))
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x130000)
            image.format = "ELF"
            analyzer = Analyzer(image)

            candidate = analyzer.infer_prototype(0x13000B)
            self.assertEqual(candidate.source, "inferred")
            self.assertFalse(candidate.locked)
            self.assertEqual(len(candidate.parameters), 1)
            parameter = candidate.parameters[0]
            self.assertEqual(parameter.name, "param_1")
            self.assertEqual(parameter.locations[0].register, "rdi")
            self.assertEqual(parameter.confidence, 0.91)
            self.assertIn("callsite-register-write:1/1", parameter.evidence)
            self.assertIn("paramid-rank:direct-read", parameter.evidence)
            self.assertIn("paramid-active", parameter.evidence)
            self.assertIn("pcode-like-paramid-analysis", candidate.inference_evidence)
            self.assertEqual(candidate.inference_confidence, 0.91)

            overwritten_path = Path(directory) / "overwritten.bin"
            overwritten_path.write_bytes(bytes.fromhex("bf 01 00 00 00 89 f8 c3"))
            overwritten_image = BinaryImage.load(
                overwritten_path, raw_arch="x86_64", base=0x131000
            )
            overwritten_image.format = "ELF"
            overwritten = Analyzer(overwritten_image)
            overwritten_candidate = overwritten.infer_prototype(0x131000)
            self.assertEqual(overwritten_candidate.parameters, ())
            self.assertEqual(overwritten_candidate.inference_confidence, 0.35)
            self.assertIn(
                "no-live-in-parameter-read-observed; unused-formals-cannot-be-ruled-out",
                overwritten_candidate.inference_evidence,
            )

    def test_stack_parameter_candidate_has_storage_and_read_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stack-candidate.bin"
            path.write_bytes(bytes.fromhex("8b 44 24 04 c3"))
            image = BinaryImage.load(path, raw_arch="x86", base=0x132000)
            analyzer = Analyzer(image)

            candidate = analyzer.infer_prototype(0x132000)
            self.assertEqual(len(candidate.parameters), 1)
            parameter = candidate.parameters[0]
            self.assertEqual(parameter.locations[0].stack_offset, 4)
            self.assertEqual(parameter.confidence, 0.76)
            self.assertTrue(
                any(
                    evidence.startswith("stack-live-in-read:")
                    for evidence in parameter.evidence
                )
            )

            call_path = Path(directory) / "stack-callsite.bin"
            call_path.write_bytes(
                bytes.fromhex(
                    "6a 07 e8 04 00 00 00 83 c4 04 c3 8b 44 24 04 c3"
                )
            )
            call_image = BinaryImage.load(call_path, raw_arch="x86", base=0x133000)
            call_analyzer = Analyzer(call_image)
            call_candidate = call_analyzer.infer_prototype(0x13300B)
            call_parameter = call_candidate.parameters[0]
            self.assertEqual(call_parameter.confidence, 0.91)
            self.assertIn("callsite-stack-write:1/1", call_parameter.evidence)

    def test_x64_callsite_stack_stores_support_unread_positional_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x64-stack-callsite.bin"
            path.write_bytes(
                bytes.fromhex(
                    "48 83 ec 38 "
                    "48 c7 44 24 20 07 00 00 00 "
                    "48 c7 44 24 28 08 00 00 00 "
                    "e8 01 00 00 00 c3 "
                    "8b 44 24 30 c3"
                )
            )
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x134000)
            image.format = "PE"
            analyzer = Analyzer(image)

            candidate = analyzer.infer_prototype(0x13401C)
            self.assertEqual(len(candidate.parameters), 6)
            unread_gap = candidate.parameters[4]
            self.assertEqual(unread_gap.name, "param_5")
            self.assertEqual(unread_gap.locations[0].stack_offset, 40)
            self.assertEqual(unread_gap.confidence, 0.45)
            self.assertIn("callsite-stack-write:1/1", unread_gap.evidence)
            self.assertIn("callsite-supplies-unread-positional-slot", unread_gap.evidence)

            observed = candidate.parameters[5]
            self.assertEqual(observed.name, "param_6")
            self.assertEqual(observed.locations[0].stack_offset, 48)
            self.assertEqual(observed.confidence, 0.91)
            self.assertIn("callsite-stack-write:1/1", observed.evidence)

    def test_x64_stack_evidence_stops_at_unmodeled_stack_pointer_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x64-unknown-stack-adjustment.bin"
            path.write_bytes(
                bytes.fromhex(
                    "48 c7 44 24 20 07 00 00 00 "
                    "48 83 e4 f0 "
                    "e8 01 00 00 00 c3 "
                    "8b 44 24 28 c3"
                )
            )
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x135000)
            image.format = "PE"
            analyzer = Analyzer(image)

            candidate = analyzer.infer_prototype(0x135013)
            parameter = candidate.parameters[4]
            self.assertEqual(parameter.name, "param_5")
            self.assertEqual(parameter.confidence, 0.76)
            self.assertIn("callsite-stack-write-not-observed", parameter.evidence)
            self.assertNotIn("callsite-stack-write:1/1", parameter.evidence)

    def test_locked_prototype_controls_pseudocode_and_json_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proto.bin"
            path.write_bytes(bytes.fromhex("89 f8 c3"))  # mov eax, edi; ret
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x100000)
            image.format = "ELF"
            analyzer = Analyzer(image)
            function = analyzer.functions[0x100000]
            signature = FunctionSignature(
                "int",
                (FunctionParameter("value", "int"),),
                SYSV_AMD64.name,
            )

            resolved = analyzer.set_prototype(function, signature)
            self.assertTrue(resolved.locked)
            self.assertEqual(resolved.source, "user")
            self.assertEqual(resolved.parameters[0].locations[0].register, "rdi")

            pseudocode = analyzer.pseudocode(function)
            self.assertIn("int entry(int value)", pseudocode)
            self.assertRegex(pseudocode, r"return .*value")
            self.assertEqual(function.signature.parameters[0].name, "value")
            self.assertEqual(function.calling_convention, "sysv_amd64")

            exported = json.loads(json.dumps({"signature": asdict(function.signature)}))
            self.assertEqual(
                exported["signature"]["parameters"][0]["locations"][0]["register"],
                "rdi",
            )
            self.assertEqual(exported["signature"]["return_data_type"]["name"], "int")

            callback = analyzer.type_database.function("int", ("double",))
            analyzer.set_prototype(
                function,
                FunctionSignature(
                    "void",
                    (FunctionParameter("callback", callback),),
                    SYSV_AMD64.name,
                ),
            )
            self.assertIn(
                "void entry(int (*callback)(double))", analyzer.pseudocode(function)
            )

    def test_locked_prototype_is_preserved_for_thunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thunk.bin"
            path.write_bytes(bytes.fromhex("c3"))
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x110000)
            image.format = "ELF"
            analyzer = Analyzer(image)
            function = analyzer.functions[0x110000]
            function.is_thunk = True
            analyzer.set_prototype(
                function,
                FunctionSignature(
                    "int",
                    (FunctionParameter("value", "int"),),
                    SYSV_AMD64.name,
                ),
            )

            pseudocode = analyzer.pseudocode(function)
            self.assertIn("int entry(int value)", pseudocode)
            self.assertNotIn("void entry(void)", pseudocode)
            self.assertTrue(function.signature.locked)

    def test_decompiler_preserves_variadic_marker_and_hidden_sret_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prototypes.bin"
            path.write_bytes(bytes.fromhex("c3 c3"))
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x120000)
            image.format = "ELF"
            analyzer = Analyzer(image)
            function = analyzer.functions[0x120000]

            variadic = analyzer.set_prototype(
                function,
                FunctionSignature(
                    "int",
                    (FunctionParameter("count", "int"),),
                    SYSV_AMD64.name,
                    variadic=True,
                ),
            )
            pseudocode = analyzer.pseudocode(function)
            self.assertIn("int entry(int count, ...)", pseudocode)
            self.assertEqual(variadic.varargs_sse_count_register, "al")

            aggregate = analyzer.type_database.structure(
                "LargeResult",
                [("first", "uint64_t"), ("second", "uint64_t"), ("third", "uint64_t")],
            )
            hidden = analyzer.set_prototype(
                function,
                FunctionSignature(
                    aggregate.name,
                    (FunctionParameter("value", "int"),),
                    SYSV_AMD64.name,
                    return_data_type=aggregate,
                ),
            )
            self.assertTrue(hidden.hidden_return)
            self.assertEqual(hidden.hidden_return_location.register, "rdi")
            self.assertEqual(hidden.parameters[0].locations[0].register, "rsi")
            self.assertIn(
                "struct LargeResult entry(int value)", analyzer.pseudocode(function)
            )

    def test_locked_prototype_formats_aggregate_calls_and_hides_sret_argument(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sysv_path = Path(directory) / "aggregate-call.bin"
            sysv_path.write_bytes(
                bytes.fromhex("bf 01 00 00 00 be 02 00 00 00 e8 01 00 00 00 c3 c3")
            )
            sysv_image = BinaryImage.load(sysv_path, raw_arch="x86_64", base=0x150000)
            sysv_image.format = "ELF"
            sysv = Analyzer(sysv_image)
            pair = sysv.type_database.structure(
                "Pair", [("first", "uint64_t"), ("second", "uint64_t")]
            )
            sysv.set_prototype(
                sysv.functions[0x150010],
                FunctionSignature(
                    "void",
                    (FunctionParameter("item", pair),),
                    SYSV_AMD64.name,
                ),
            )
            sysv_pseudocode = sysv.pseudocode(sysv.functions[0x150000])
            self.assertIn(
                "((struct Pair){ .first = ((uint64_t)(1)), .second = ((uint64_t)(2)) })",
                sysv_pseudocode,
            )

            win_path = Path(directory) / "hidden-return-call.bin"
            win_path.write_bytes(
                bytes.fromhex("48 8d 4c 24 20 ba 07 00 00 00 e8 01 00 00 00 c3 c3")
            )
            win_image = BinaryImage.load(win_path, raw_arch="x86_64", base=0x160000)
            win_image.format = "PE"
            windows = Analyzer(win_image)
            result_type = windows.type_database.structure(
                "LargeResult", [("first", "uint64_t"), ("second", "uint64_t")]
            )
            windows.set_prototype(
                windows.functions[0x160010],
                FunctionSignature(
                    result_type.name,
                    (FunctionParameter("value", "int"),),
                    WINDOWS_X64.name,
                    return_data_type=result_type,
                ),
            )
            win_pseudocode = windows.pseudocode(windows.functions[0x160000])
            self.assertIn("FUN_160010(7)", win_pseudocode)
            self.assertNotIn("FUN_160010(in_rcx, 7)", win_pseudocode)

            indirect_path = Path(directory) / "indirect-aggregate-call.bin"
            indirect_path.write_bytes(
                bytes.fromhex("48 8d 4c 24 20 e8 01 00 00 00 c3 c3")
            )
            indirect_image = BinaryImage.load(
                indirect_path, raw_arch="x86_64", base=0x170000
            )
            indirect_image.format = "PE"
            indirect = Analyzer(indirect_image)
            small = indirect.type_database.structure(
                "ThreeBytes",
                [("first", "char"), ("second", "char"), ("third", "char")],
                size=3,
            )
            indirect.set_prototype(
                indirect.functions[0x17000B],
                FunctionSignature(
                    "void",
                    (FunctionParameter("item", small),),
                    WINDOWS_X64.name,
                ),
            )
            indirect_pseudocode = indirect.pseudocode(indirect.functions[0x170000])
            self.assertIn(
                "FUN_17000B(*((struct ThreeBytes *)(&home_slot_4)))",
                indirect_pseudocode,
            )

    def test_explicit_unknown_types_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proto.bin"
            path.write_bytes(bytes.fromhex("c3"))
            image = BinaryImage.load(path, raw_arch="x86_64", base=0x110000)
            analyzer = Analyzer(image)
            with self.assertRaisesRegex(KeyError, "Tipo no definido"):
                analyzer.set_prototype(
                    analyzer.functions[0x110000],
                    FunctionSignature(
                        "MissingType",
                        (FunctionParameter("argument", "int"),),
                        SYSV_AMD64.name,
                    ),
                )
            incomplete = analyzer.type_database.forward_structure("Opaque")
            with self.assertRaisesRegex(ValueError, "agregado incompleto"):
                analyzer.set_prototype(
                    analyzer.functions[0x110000],
                    FunctionSignature(
                        incomplete.name,
                        (FunctionParameter("argument", "int"),),
                        SYSV_AMD64.name,
                        return_data_type=incomplete,
                    ),
                )
            with self.assertRaisesRegex(ValueError, "por valor"):
                analyzer.set_prototype(
                    analyzer.functions[0x110000],
                    FunctionSignature(
                        "void",
                        (
                            FunctionParameter(
                                "opaque", incomplete.name, data_type=incomplete
                            ),
                        ),
                        SYSV_AMD64.name,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
