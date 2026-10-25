from __future__ import annotations

import io
import re
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from types import SimpleNamespace

from minighidra import (
    Analyzer,
    BinaryImage,
    FunctionParameter,
    FunctionSignature,
    PrintC,
    PrintLanguage,
    TypeResolver,
    build_printable_function,
)
from minighidra.decompiler import SemanticDecompiler, _Expr
from minighidra.emit import BraceStyle, EmitMarkup, SyntaxHighlight
from minighidra.marshal import ELEM_CLANG_DOCUMENT, PackedDecode
from minighidra.model import Function
from minighidra.printc import (
    _expr_mentions,
    _for_loop_form,
    _opaque_segments,
    _split_opaque,
    _split_phi_inputs,
)
from minighidra.printdoc import (
    ExpressionConverter,
    PrintExpr,
    PrintOp,
    PrintStatement,
    PrintSymbol,
    SymbolTable,
)
from minighidra.printlanguage import Modifier, OpTokenType, TagType
from minighidra.typesys import TypeDatabase


# A call, a conditional branch and a phi merge, plus a small callee.
CALL_SAMPLE = (
    "55" "4889e5" "8b4510" "83c00a" "89c7" "e80f000000" "85c0" "7405"
    "b82a000000" "5d" "c3" "90909090" "55" "4889e5" "89f8" "0fafc0" "5d" "c3"
)
# cmp/jge/loop-back: a while loop with a back edge.
LOOP_SAMPLE = (
    "55" "4889e5" "b800000000" "83f80a" "7d09" "8b4d10" "01c8" "ffc0" "ebf2"
    "5d" "c3"
)


def _printer(markup: bool = False, width: int = 100):
    printer = PrintC(None)
    stream = io.StringIO()
    printer.set_output_stream(stream)
    printer.set_markup(markup)
    printer.set_max_line_size(width)
    printer.resolver = TypeResolver(TypeDatabase(64, "ELF"), 64)
    printer.initialize_from_architecture()
    return printer, stream


def _emit_expression(printer: PrintC, node: PrintExpr) -> None:
    document = printer.emit.begin_document()
    printer.emit.tag_line()
    printer.push_node(node, None, printer.mods)
    printer.recurse()
    printer.emit.print(";")
    printer.emit.tag_line()
    printer.emit.end_document(document)
    printer.emit.flush()


def _symbols(printer: PrintC) -> SymbolTable:
    function = Function(address=0x401000, name="test")
    return SymbolTable(None, printer.resolver, 64, function)


def _var(symbols: SymbolTable, name: str, bits: int = 32) -> PrintExpr:
    symbol = symbols.anonymous(name, bits)
    return PrintExpr(
        kind="var", symbol=symbol, ctype=symbol.ctype, name=name, bits=bits
    )


def _const(printer: PrintC, value: int, bits: int = 32) -> PrintExpr:
    return PrintExpr(
        kind="const",
        int_value=value,
        bits=bits,
        ctype=printer.resolver.integer(bits, False),
    )


def _op(name: str, *args: PrintExpr, bits: int = 32) -> PrintExpr:
    return PrintExpr(kind="op", name=name, args=args, bits=bits)


class PrintLanguageHelperTests(unittest.TestCase):
    def test_most_natural_base_matches_ghidra(self) -> None:
        cases = {
            0: 10,
            1: 16,
            10: 10,
            11: 16,
            100: 10,
            1000: 10,
            0x10: 16,
            0xFF: 16,
            0x100: 16,
            9: 16,
            99: 10,
            999: 10,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(PrintLanguage.most_natural_base(value), expected)

    def test_format_binary_pads_to_byte_boundaries(self) -> None:
        for value, expected in ((0, "0"), (1, "00000001"), (0x1FF, "0000000111111111")):
            with self.subTest(value=value):
                stream = io.StringIO()
                PrintLanguage.format_binary(stream, value)
                self.assertEqual(stream.getvalue(), expected)

    def test_unicode_needs_escape(self) -> None:
        self.assertTrue(PrintLanguage.unicode_needs_escape(0x00))
        self.assertTrue(PrintLanguage.unicode_needs_escape(ord("\\")))
        self.assertTrue(PrintLanguage.unicode_needs_escape(ord('"')))
        self.assertFalse(PrintLanguage.unicode_needs_escape(ord("a")))
        self.assertTrue(PrintLanguage.unicode_needs_escape(0x7F))
        self.assertFalse(PrintLanguage.unicode_needs_escape(0xE9))
        self.assertTrue(PrintLanguage.unicode_needs_escape(0x2028))
        self.assertTrue(PrintLanguage.unicode_needs_escape(0xD800))
        self.assertFalse(PrintLanguage.unicode_needs_escape(0x4E00))

    def test_integer_format_option_sets_modifiers(self) -> None:
        printer, _ = _printer()
        printer.set_integer_format("hex")
        self.assertTrue(printer.is_set(Modifier.force_hex.value))
        printer.set_integer_format("dec")
        self.assertFalse(printer.is_set(Modifier.force_hex.value))
        self.assertTrue(printer.is_set(Modifier.force_dec.value))
        printer.set_integer_format("best")
        self.assertFalse(printer.is_set(Modifier.force_dec.value))
        with self.assertRaises(Exception):
            printer.set_integer_format("octal")


class OpTokenTableTests(unittest.TestCase):
    def test_token_table_matches_printc_cc(self) -> None:
        # token: (print1, print2, stage, precedence, associative, type, spacing, bump)
        expected = {
            "hidden": ("", "", 1, 70, False, OpTokenType.hiddenfunction, 0, 0),
            "scope": ("::", "", 2, 70, True, OpTokenType.binary, 0, 0),
            "object_member": (".", "", 2, 66, True, OpTokenType.binary, 0, 0),
            "pointer_member": ("->", "", 2, 66, True, OpTokenType.binary, 0, 0),
            "subscript": ("[", "]", 2, 66, False, OpTokenType.postsurround, 0, 0),
            "function_call": ("(", ")", 2, 66, False, OpTokenType.postsurround, 0, 10),
            "bitwise_not": ("~", "", 1, 62, False, OpTokenType.unary_prefix, 0, 0),
            "boolean_not": ("!", "", 1, 62, False, OpTokenType.unary_prefix, 0, 0),
            "unary_minus": ("-", "", 1, 62, False, OpTokenType.unary_prefix, 0, 0),
            "addressof": ("&", "", 1, 62, False, OpTokenType.unary_prefix, 0, 0),
            "dereference": ("*", "", 1, 62, False, OpTokenType.unary_prefix, 0, 0),
            "typecast": ("(", ")", 2, 62, False, OpTokenType.presurround, 0, 0),
            "multiply": ("*", "", 2, 54, True, OpTokenType.binary, 1, 0),
            "divide": ("/", "", 2, 54, False, OpTokenType.binary, 1, 0),
            "modulo": ("%", "", 2, 54, False, OpTokenType.binary, 1, 0),
            "binary_plus": ("+", "", 2, 50, True, OpTokenType.binary, 1, 0),
            "binary_minus": ("-", "", 2, 50, False, OpTokenType.binary, 1, 0),
            "shift_left": ("<<", "", 2, 46, False, OpTokenType.binary, 1, 0),
            "shift_right": (">>", "", 2, 46, False, OpTokenType.binary, 1, 0),
            "shift_sright": (">>", "", 2, 46, False, OpTokenType.binary, 1, 0),
            "less_than": ("<", "", 2, 42, False, OpTokenType.binary, 1, 0),
            "less_equal": ("<=", "", 2, 42, False, OpTokenType.binary, 1, 0),
            "greater_than": (">", "", 2, 42, False, OpTokenType.binary, 1, 0),
            "greater_equal": (">=", "", 2, 42, False, OpTokenType.binary, 1, 0),
            "equal": ("==", "", 2, 38, False, OpTokenType.binary, 1, 0),
            "not_equal": ("!=", "", 2, 38, False, OpTokenType.binary, 1, 0),
            "bitwise_and": ("&", "", 2, 34, True, OpTokenType.binary, 1, 0),
            "bitwise_xor": ("^", "", 2, 30, True, OpTokenType.binary, 1, 0),
            "bitwise_or": ("|", "", 2, 26, True, OpTokenType.binary, 1, 0),
            "boolean_and": ("&&", "", 2, 22, False, OpTokenType.binary, 1, 0),
            "boolean_xor": ("^^", "", 2, 20, False, OpTokenType.binary, 1, 0),
            "boolean_or": ("||", "", 2, 18, False, OpTokenType.binary, 1, 0),
            "assignment": ("=", "", 2, 14, False, OpTokenType.binary, 1, 5),
            "comma": (",", "", 2, 2, True, OpTokenType.binary, 0, 0),
            "multequal": ("*=", "", 2, 14, False, OpTokenType.binary, 1, 5),
            "plusequal": ("+=", "", 2, 14, False, OpTokenType.binary, 1, 5),
            "type_expr_space": ("", "", 2, 10, False, OpTokenType.space, 1, 0),
            "type_expr_nospace": ("", "", 2, 10, False, OpTokenType.space, 0, 0),
            "ptr_expr": ("*", "", 1, 62, False, OpTokenType.unary_prefix, 0, 0),
            "array_expr": ("[", "]", 2, 66, False, OpTokenType.postsurround, 1, 0),
            "enum_cat": ("|", "", 2, 26, True, OpTokenType.binary, 0, 0),
        }
        for name, values in expected.items():
            with self.subTest(token=name):
                token = getattr(PrintC, name)
                self.assertEqual(
                    (
                        token.print1,
                        token.print2,
                        token.stage,
                        token.precedence,
                        token.associative,
                        token.type,
                        token.spacing,
                        token.bump,
                    ),
                    values,
                )

    def test_negation_pairs(self) -> None:
        self.assertIs(PrintC.less_than.negate, PrintC.greater_equal)
        self.assertIs(PrintC.less_equal.negate, PrintC.greater_than)
        self.assertIs(PrintC.greater_than.negate, PrintC.less_equal)
        self.assertIs(PrintC.greater_equal.negate, PrintC.less_than)
        self.assertIs(PrintC.equal.negate, PrintC.not_equal)
        self.assertIs(PrintC.not_equal.negate, PrintC.equal)

    def test_default_printc_options(self) -> None:
        printer, _ = _printer()
        self.assertTrue(printer.option_convention)
        self.assertTrue(printer.option_hide_exts)
        self.assertFalse(printer.option_inplace_ops)
        self.assertFalse(printer.option_nocasts)
        self.assertFalse(printer.option_NULL)
        self.assertEqual(printer.option_brace_func, BraceStyle.skip_line)
        self.assertEqual(printer.option_brace_ifelse, BraceStyle.same_line)
        self.assertEqual(printer.null_token, "NULL")
        self.assertEqual(printer.get_scope_delimiter(), "::")


class ExpressionPrintingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.printer, self.stream = _printer()
        self.symbols = _symbols(self.printer)

    def _print(self, node: PrintExpr) -> str:
        self.stream.truncate(0)
        self.stream.seek(0)
        _emit_expression(self.printer, node)
        return self.stream.getvalue().strip()

    def test_precedence_and_associativity(self) -> None:
        a = _var(self.symbols, "a")
        b = _var(self.symbols, "b")
        c = _var(self.symbols, "c")
        self.assertEqual(self._print(_op("add", a, _op("mul", b, c))), "a + b * c;")
        self.assertEqual(self._print(_op("mul", _op("add", a, b), c)), "(a + b) * c;")
        self.assertEqual(
            self._print(_op("add", _op("add", a, b), c)), "a + b + c;"
        )
        self.assertEqual(
            self._print(_op("sub", _op("sub", a, b), c)), "(a - b) - c;"
        )

    def test_signed_and_unsigned_comparisons_keep_explicit_casts(self) -> None:
        a = _var(self.symbols, "a")
        b = _var(self.symbols, "b")
        self.assertEqual(
            self._print(_op("slt", a, b)), "(int32_t)a < (int32_t)b;"
        )
        self.assertEqual(
            self._print(_op("uge", a, b)), "(uint32_t)a >= (uint32_t)b;"
        )
        self.assertEqual(self._print(_op("eq", a, b)), "a == b;")

    def test_negate_token_flips_the_operator(self) -> None:
        # opBoolNegate flips the next comparison token instead of printing '!'.
        a = _var(self.symbols, "a")
        b = _var(self.symbols, "b")
        negated = PrintExpr(
            kind="unary", name="boolnot", args=(_op("slt", a, b),), bits=1
        )
        self.assertEqual(self._print(negated), "(int32_t)a >= (int32_t)b;")
        self.assertFalse(self.printer.is_set(Modifier.negatetoken.value))

        # A non-flippable operand keeps the explicit '!'.
        plain = PrintExpr(
            kind="unary", name="boolnot", args=(_var(self.symbols, "flag"),), bits=1
        )
        self.assertEqual(self._print(plain), "!flag;")

    def test_unary_operators(self) -> None:
        a = _var(self.symbols, "a")
        self.assertEqual(
            self._print(PrintExpr(kind="unary", name="neg", args=(a,), bits=32)), "-a;"
        )
        self.assertEqual(
            self._print(PrintExpr(kind="unary", name="bitnot", args=(a,), bits=32)),
            "~a;",
        )
        self.assertEqual(
            self._print(PrintExpr(kind="unary", name="boolnot", args=(a,), bits=32)),
            "!a;",
        )

    def test_extension_and_truncation_casts(self) -> None:
        a = _var(self.symbols, "a", 32)
        zext = PrintExpr(
            kind="zext",
            args=(a,),
            bits=64,
            ctype=self.printer.resolver.integer(64, False),
        )
        self.assertEqual(self._print(zext), "(uint64_t)(uint32_t)a;")
        sext = PrintExpr(
            kind="sext",
            args=(a,),
            bits=64,
            ctype=self.printer.resolver.integer(64, True),
        )
        self.assertEqual(self._print(sext), "(int64_t)(int32_t)a;")
        trunc = PrintExpr(
            kind="trunc",
            args=(a,),
            bits=16,
            ctype=self.printer.resolver.integer(16, False),
        )
        self.assertEqual(self._print(trunc), "(uint16_t)a;")

    def test_load_uses_a_typed_pointer_dereference(self) -> None:
        pointer = _var(self.symbols, "ptr", 64)
        load = PrintExpr(
            kind="load",
            args=(pointer,),
            bits=32,
            ctype=self.printer.resolver.integer(32, False),
        )
        self.assertEqual(self._print(load), "*(uint32_t *)ptr;")
        nested = PrintExpr(
            kind="load",
            args=(_op("add", pointer, _const(self.printer, 8, 64)),),
            bits=32,
            ctype=self.printer.resolver.integer(32, False),
        )
        self.assertEqual(self._print(nested), "*(uint32_t *)(ptr + 8);")

    def test_address_of_and_select(self) -> None:
        local = _var(self.symbols, "local_10", 64)
        address_of = PrintExpr(
            kind="address_of",
            args=(local,),
            bits=64,
            ctype=self.printer.resolver.pointer(local.ctype),
        )
        self.assertEqual(self._print(address_of), "&local_10;")
        select = PrintExpr(
            kind="select",
            args=(
                _op("eq", _var(self.symbols, "flag"), _const(self.printer, 0)),
                _var(self.symbols, "yes"),
                _var(self.symbols, "no"),
            ),
            bits=32,
        )
        self.assertEqual(self._print(select), "flag == 0 ? yes : no;")

        # A ternary inside an addition needs parentheses (looser precedence).
        nested = _op("add", _var(self.symbols, "total"), select)
        self.assertEqual(
            self._print(nested), "total + (flag == 0 ? yes : no);"
        )

        # Nesting in the false branch is right associative: no parentheses.
        chained = PrintExpr(
            kind="select",
            args=(_var(self.symbols, "a"), _var(self.symbols, "b"), select),
            bits=32,
        )
        self.assertEqual(self._print(chained), "a ? b : flag == 0 ? yes : no;")

        # Nesting in the true branch must be parenthesised.
        inner = PrintExpr(
            kind="select",
            args=(_var(self.symbols, "a"), select, _var(self.symbols, "c")),
            bits=32,
        )
        self.assertEqual(
            self._print(inner), "a ? (flag == 0 ? yes : no) : c;"
        )

    def test_phi_and_opaque_nodes(self) -> None:
        phi = PrintExpr(
            kind="phi",
            args=(_var(self.symbols, "a"), _var(self.symbols, "b")),
            bits=32,
        )
        self.assertEqual(self._print(phi), "phi(a,b);")
        opaque = PrintExpr(kind="opaque", text="__builtin_popcountll(x)", bits=64)
        self.assertEqual(self._print(opaque), "__builtin_popcountll(x);")


class ConstantAndTypePrintingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.printer, self.stream = _printer()

    def _constant(self, node: PrintExpr) -> str:
        self.stream.truncate(0)
        self.stream.seek(0)
        _emit_expression(self.printer, node)
        return self.stream.getvalue().strip().rstrip(";")

    def test_best_fit_integer_formatting(self) -> None:
        cases = {
            0: "0",
            10: "10",
            11: "0xb",
            100: "100",
            0x1000: "0x1000",
            0xFFFFFFFF: "0xffffffff",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                printer, stream = _printer()
                self.printer = printer
                self.stream = stream
                self.assertEqual(
                    self._constant(_const(printer, value, 32)), expected
                )

    def test_forced_formats(self) -> None:
        printer, stream = _printer()
        printer.set_integer_format("hex")
        self.printer, self.stream = printer, stream
        self.assertEqual(self._constant(_const(printer, 10, 32)), "0xa")
        printer, stream = _printer()
        printer.set_integer_format("dec")
        self.printer, self.stream = printer, stream
        self.assertEqual(self._constant(_const(printer, 0x1000, 32)), "4096")

    def test_signed_constants_print_negative_values(self) -> None:
        ctype = self.printer.resolver.integer(32, True)
        node = PrintExpr(
            kind="const", int_value=0xFFFFFFFF, bits=32, ctype=ctype, signed=True
        )
        document = self.printer.emit.begin_document()
        self.printer.emit.tag_line()
        # pushConstant receives the raw value; the signed type drives the sign.
        self.printer.push_constant(0xFFFFFFFF, ctype, TagType.vartoken, node, None, 0)
        self.printer.recurse()
        self.printer.emit.tag_line()
        self.printer.emit.end_document(document)
        self.printer.emit.flush()
        self.assertEqual(self.stream.getvalue().strip(), "-1")

    def test_float_constants(self) -> None:
        ctype = self.printer.resolver.floating(64)
        node = PrintExpr(kind="fp_const", float_value=1.5, bits=64, ctype=ctype)
        self.assertEqual(self._constant(node), "1.5")
        whole = PrintExpr(kind="fp_const", float_value=2.0, bits=64, ctype=ctype)
        self.assertEqual(self._constant(whole), "2.0")

    def test_pointer_array_and_function_declarators(self) -> None:
        resolver = self.printer.resolver
        base = resolver.from_spec("uint32_t")
        pointer = resolver.pointer(base)
        array = resolver.from_spec("uint32_t[10]")
        symbols = _symbols(self.printer)

        def declaration(ctype, name: str) -> str:
            printer, stream = _printer()
            printer.resolver = resolver
            symbol = PrintSymbol(name=name, ctype=ctype, scope=symbols.local_scope)
            document = printer.emit.begin_document()
            printer.emit.tag_line()
            printer.emit_var_decl(symbol)
            printer.emit.print(";")
            printer.emit.tag_line()
            printer.emit.end_document(document)
            printer.emit.flush()
            return stream.getvalue().strip()

        self.assertEqual(declaration(base, "count"), "uint32_t count;")
        self.assertEqual(declaration(pointer, "ptr"), "uint32_t *ptr;")
        # array_expr keeps Ghidra's spacing=1 in front of the subscript.
        self.assertEqual(declaration(array, "table"), "uint32_t table [10];")
        self.assertEqual(
            declaration(resolver.pointer(array), "rows"), "uint32_t (*rows) [10];"
        )

    def test_struct_definition(self) -> None:
        database = TypeDatabase(64, "ELF")
        record = database.structure(
            "Record", [("tag", "uint32_t"), ("value", "double")]
        )
        printer, stream = _printer()
        printer.resolver = TypeResolver(database, 64)
        document = printer.emit.begin_document()
        printer.emit_type_definition(printer.resolver.from_spec(record))
        printer.emit.tag_line()
        printer.emit.end_document(document)
        printer.emit.flush()
        text = stream.getvalue()
        self.assertIn("typedef struct", text)
        self.assertIn("uint32_t tag,", text)
        self.assertIn("double value", text)
        self.assertIn("} Record;", text)
        self.assertNotIn("} struct Record;", text)


class FunctionPrintingTests(unittest.TestCase):
    def _analyze(self, hexcode: str, base: int = 0x401000):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "sample.bin"
        path.write_bytes(bytes.fromhex(hexcode))
        image = BinaryImage.load(path, raw_arch="x86_64", base=base)
        analyzer = Analyzer(image)
        return analyzer, analyzer.functions[base]

    def test_complete_function_is_emitted(self) -> None:
        analyzer, function = self._analyze(CALL_SAMPLE)
        text = analyzer.print_function(function)
        self.assertIn("uint64_t entry(", text)
        self.assertIn("uint64_t param_1,", text)
        self.assertIn("{", text)
        self.assertIn("uint64_t call_result_40100C;", text)
        self.assertIn("call_result_40100C = FUN_401020(", text)
        self.assertIn("if ((uint32_t)call_result_40100C != 0) {", text)
        self.assertIn("phi_rax_loc_40101A = phi(call_result_40100C,0x2a);", text)
        self.assertIn("return phi_rax_loc_40101A;", text)
        self.assertTrue(text.rstrip().endswith("}"))

    def test_markup_uses_the_ghidra_element_vocabulary(self) -> None:
        analyzer, function = self._analyze(CALL_SAMPLE)
        markup = analyzer.print_function(function, markup=True)
        root = ElementTree.fromstring(markup)
        self.assertEqual(root.tag, "clang_document")
        tags = {element.tag for element in root.iter()}
        for expected in (
            "function",
            "funcproto",
            "return_type",
            "type",
            "funcname",
            "vardecl",
            "block",
            "statement",
            "op",
            "variable",
            "break",
            "syntax",
        ):
            self.assertIn(expected, tags)
        funcname = root.find(".//funcname")
        self.assertEqual(funcname.attrib.get("color"), "0x3")
        variables = {
            element.text: element.attrib.get("color")
            for element in root.iter("variable")
        }
        self.assertEqual(variables.get("param_1"), "0x6")  # param_color
        self.assertEqual(variables.get("call_result_40100C"), "0x4")  # var_color
        statement = root.find(".//statement")
        self.assertIsNotNone(statement.attrib.get("opref", None) or None)

    def test_flat_mode_emits_labels_and_gotos(self) -> None:
        analyzer, function = self._analyze(LOOP_SAMPLE)
        flat = analyzer.print_function(function, flat=True, max_line_size=200)
        self.assertIn("code_r0x00401000:", flat)
        self.assertIn("if (", flat)
        self.assertIn("goto code_r0x00401017;", flat)
        self.assertNotIn("while (", flat)

    def test_structured_loop_negates_the_exit_condition(self) -> None:
        analyzer, function = self._analyze(LOOP_SAMPLE)
        text = analyzer.print_function(function, max_line_size=400)
        # jge exits the loop, so the printed condition must be the negation.
        self.assertIn("< (int32_t)10", text)
        self.assertNotIn(">= (int32_t)10", text)

    def test_loop_variable_is_recovered_as_a_for_loop(self) -> None:
        analyzer, function = self._analyze(LOOP_SAMPLE)
        text = analyzer.print_function(function, max_line_size=400)
        # emitForLoop: the merge at the loop head becomes the three ;-separated
        # parts of a `for` header instead of a comma separated while condition.
        self.assertIn("for (phi_rax_loc_401009 = 0;", text)
        self.assertIn("; phi_rax_loc_401009 =", text)
        self.assertNotIn("phi(", text)
        self.assertNotIn("while (", text)

    def test_for_loops_can_be_disabled(self) -> None:
        analyzer, function = self._analyze(LOOP_SAMPLE)
        text = analyzer.print_function(
            function, max_line_size=400, for_loops=False
        )
        # Architecture::analyze_for_loops off: plain while syntax, and the merge
        # statements move into the body instead of the condition.
        self.assertIn("while ((int32_t)", text)
        self.assertIn("phi_rax_loc_401009 = phi(0,", text)
        self.assertNotIn("for (", text)

    def test_merge_statements_move_into_the_loop_body(self) -> None:
        analyzer, function = self._analyze(LOOP_SAMPLE)
        text = analyzer.print_function(function, max_line_size=400, for_loops=False)
        body = text.split("while (", 1)[1]
        condition, rest = body.split(") {", 1)
        self.assertNotIn("phi(", condition)
        self.assertIn("phi(", rest)

    def test_options_reach_the_emitter(self) -> None:
        analyzer, function = self._analyze(CALL_SAMPLE)
        no_casts = analyzer.print_function(function, casts=False, max_line_size=200)
        self.assertIn("DISPLAY WARNING: Type casts are NOT being printed", no_casts)
        self.assertNotIn("(uint32_t)", no_casts)

        forced_hex = analyzer.print_function(
            function, integer_format="hex", max_line_size=200
        )
        self.assertIn("+ 0xa", forced_hex)

        same_line = analyzer.print_function(
            function, brace="same_line", max_line_size=200
        )
        self.assertIn(") {", same_line)

        cplusplus = analyzer.print_function(
            function, comment_style="cplusplus", max_line_size=200
        )
        self.assertIn("// Experimental semantic pseudocode", cplusplus)

        no_labels = analyzer.print_function(function, labels="none", flat=True)
        # Label definitions disappear; goto targets still name their block.
        self.assertNotIn("\ncode_r0x", no_labels)
        self.assertIn("goto code_r0x", no_labels)

        narrow = analyzer.print_function(function, max_line_size=40)
        self.assertTrue(all(len(line) <= 40 for line in narrow.splitlines()), narrow)

    def test_declared_prototypes_drive_symbol_and_type_resolution(self) -> None:
        analyzer, function = self._analyze(CALL_SAMPLE)
        analyzer.set_prototype(
            function,
            FunctionSignature(
                return_type="int32_t",
                parameters=(
                    FunctionParameter("count", "int32_t"),
                    FunctionParameter("name", "char *"),
                ),
                calling_convention="sysv_amd64",
            ),
        )
        text = analyzer.print_function(function, max_line_size=200)
        self.assertIn("int32_t entry(int32_t count,char *name)", text)

        markup = analyzer.print_function(function, markup=True)
        root = ElementTree.fromstring(markup)
        parameters = {
            element.text: element.attrib.get("color")
            for element in root.iter("variable")
        }
        self.assertEqual(parameters.get("count"), "0x6")
        types = [element.text for element in root.iter("type")]
        self.assertIn("int32_t", types)
        self.assertIn("char", types)

    def test_artifacts_expose_statements_terminators_and_calls(self) -> None:
        analyzer, function = self._analyze(CALL_SAMPLE)
        artifacts = SemanticDecompiler(analyzer).decompile_artifacts(function)
        self.assertEqual(artifacts.is_thunk, False)
        self.assertIn(0x401000, artifacts.statements_by_block)
        kinds = {start: item.kind for start, item in artifacts.terminators.items()}
        self.assertEqual(kinds[0x401000], "cbranch")
        self.assertIn("return", kinds.values())
        cbranch = artifacts.terminators[0x401000]
        self.assertEqual(cbranch.true_target, 0x40101A)
        self.assertEqual(cbranch.fallthrough, 0x401015)
        self.assertIsNotNone(cbranch.condition)
        calls = [record for records in artifacts.calls.values() for record in records]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].target_address, 0x401020)
        self.assertEqual(len(calls[0].arguments), 2)
        self.assertTrue(artifacts.render_catalog)

        document = build_printable_function(artifacts)
        self.assertEqual(document.name, "entry")
        self.assertEqual(len(document.prototype.parameters), 7)
        self.assertEqual(document.prototype.output_type.name, "uint64_t")
        identifiers = [symbol.symbol_id for symbol in document.symbols.values()]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_call_arguments_survive_the_print(self) -> None:
        analyzer, function = self._analyze(CALL_SAMPLE)
        text = analyzer.print_function(function, max_line_size=400)
        backend = analyzer.pseudocode(function)
        self.assertIn("FUN_401020(", text)
        self.assertIn("FUN_401020(", backend)
        # The recovered argument list is the same in both renderings.
        self.assertIn("in_rbp", text)
        self.assertIn("param_7 + 10", text)


class ConsoleEmitTests(unittest.TestCase):
    def test_emit_command_prints_and_can_write_a_file(self) -> None:
        from contextlib import redirect_stdout

        from minighidra.cli import Console
        from minighidra.project import ProjectState

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        path = root / "sample.bin"
        path.write_bytes(bytes.fromhex(LOOP_SAMPLE))
        image = BinaryImage.load(path, raw_arch="x86_64", base=0x401000)
        console = Console(image, ProjectState(), root / "project.json")

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertTrue(console.execute("emit 0x401000 --width 400"))
        self.assertIn("for (phi_rax_loc_401009 = 0;", buffer.getvalue())

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            console.execute("emit 0x401000 --width 400 --no-for-loops")
        self.assertIn("while (", buffer.getvalue())

        buffer = io.StringIO()
        output = root / "out.xml"
        with redirect_stdout(buffer):
            console.execute(f"emit entry --markup --output {output}")
        self.assertIn("Emisión guardada en", buffer.getvalue())
        self.assertEqual(ElementTree.fromstring(output.read_text()).tag, "clang_document")

        # --packed writes Ghidra's binary marshaling format.
        buffer = io.StringIO()
        packed_path = root / "out.packed"
        with redirect_stdout(buffer):
            console.execute(f"emit entry --packed --output {packed_path}")
        data = packed_path.read_bytes()
        self.assertIn(f"({len(data)} bytes)", buffer.getvalue())
        self.assertTrue(data)
        self.assertNotIn(0, data)  # every byte of the format is non-zero
        decoder = PackedDecode()
        decoder.ingest_stream(data)
        self.assertEqual(decoder.open_element(), ELEM_CLANG_DOCUMENT.id)

        # Without --output the bytes are dumped instead of written raw.
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            console.execute("emit entry --packed")
        shown = buffer.getvalue()
        self.assertIn("PackedEncode:", shown)
        rows = re.findall(r"^[0-9a-f]{8}  (.+?)  \|", shown, re.MULTILINE)
        self.assertTrue(rows)
        dumped = bytes.fromhex("".join(rows).replace(" ", ""))
        # Paren ids come from a process wide counter (TokenSplit::countbase), so
        # two separate emissions differ in those bytes; everything else matches.
        self.assertEqual(len(dumped), len(data))
        decoder = PackedDecode()
        decoder.ingest_stream(dumped)
        self.assertEqual(decoder.open_element(), ELEM_CLANG_DOCUMENT.id)

        with self.assertRaises(ValueError):
            console.execute("emit")
        with self.assertRaises(ValueError):
            console.execute("emit 0x401000 --width")
        with self.assertRaises(ValueError):
            console.execute("emit 0x401000 --nope")


class MarkupEmitterTests(unittest.TestCase):
    def test_markup_emitter_is_available_standalone(self) -> None:
        stream = io.StringIO()
        emit = EmitMarkup()
        emit.set_output_stream(stream)
        emit.begin_document()
        emit.tag_line()
        emit.print("x")
        emit.end_document(0)
        self.assertIn("<clang_document>", stream.getvalue())
        self.assertIn("<syntax>x</syntax>", stream.getvalue())
        self.assertTrue(emit.emits_markup())


class FloatingPointNodeTests(unittest.TestCase):
    """The x87/scalar SSE forms of the backend become real expression nodes."""

    def render(
        self, build, width: int = 200, markup: bool = False, catalog_text="(rendered)"
    ):
        printer, stream = _printer(markup=markup, width=width)
        symbols = _symbols(printer)
        expression = build(symbols)
        catalog = {} if catalog_text is None else {catalog_text: expression}
        converter = ExpressionConverter(printer.resolver, symbols, 64, catalog)
        node = converter.convert(expression)
        _emit_expression(printer, node)
        return node, stream.getvalue().strip()

    @staticmethod
    def _var(name: str, bits: int = 64):
        return _Expr("var", name, (), bits)

    def test_float_arithmetic_uses_the_c_operator_tokens(self) -> None:
        for operation, expected in (
            ("add", "+"),
            ("sub", "-"),
            ("mul", "*"),
            ("div", "/"),
        ):
            with self.subTest(operation=operation):
                node, text = self.render(
                    lambda _s, op=operation: _Expr(
                        "fp_op", op, (self._var("a"), self._var("b")), 64
                    )
                )
                self.assertEqual(node.kind, "op")
                self.assertEqual(node.name, f"f{operation[:3]}")
                self.assertEqual(node.ctype.name, "double")
                self.assertIn(f"a {expected} b", text)

    def test_float_comparisons_do_not_get_integer_casts(self) -> None:
        for operation, expected in (
            ("eq", "=="),
            ("ne", "!="),
            ("lt", "<"),
            ("le", "<="),
        ):
            with self.subTest(operation=operation):
                node, text = self.render(
                    lambda _s, op=operation: _Expr(
                        "fp_compare", op, (self._var("a"), self._var("b")), 1
                    )
                )
                self.assertEqual(node.kind, "op")
                self.assertIn(f"a {expected} b", text)
                self.assertNotIn("(int", text)
                self.assertNotIn("(uint", text)

    def test_float_unary_forms(self) -> None:
        node, text = self.render(
            lambda _s: _Expr("fp_unary", "neg", (self._var("a"),), 64)
        )
        self.assertEqual(node.kind, "unary")
        self.assertEqual(text, "-a;")

        node, text = self.render(
            lambda _s: _Expr("fp_unary", "sqrt", (self._var("a"),), 80)
        )
        self.assertEqual(node.kind, "func")
        self.assertEqual(node.name, "sqrtl")
        self.assertEqual(text, "sqrtl(a);")

        node, text = self.render(
            lambda _s: _Expr("fp_unary", "exp2m1", (self._var("a"),), 80)
        )
        self.assertEqual(node.kind, "op")
        self.assertIn("exp2l(a)", text)
        self.assertIn("1.0L", text)

    def test_float_conversions(self) -> None:
        node, text = self.render(
            lambda _s: _Expr("fp_cast", 64, (self._var("a"),), 64)
        )
        self.assertEqual(node.kind, "cast")
        self.assertEqual(text, "(double)a;")

        node, text = self.render(
            lambda _s: _Expr("fp_from_int", None, (self._var("a", 32),), 80)
        )
        self.assertEqual(node.kind, "cast")
        self.assertEqual(node.ctype.name, "long double")
        self.assertIn("(long double)", text)

        node, text = self.render(
            lambda _s: _Expr("fp_to_int", (32, False), (self._var("a"),), 32)
        )
        self.assertEqual(node.kind, "cast")
        self.assertEqual(text, "(int32_t)nearbyintl(a);")

        node, text = self.render(
            lambda _s: _Expr("fp_to_int", (32, True), (self._var("a"),), 32)
        )
        self.assertEqual(text, "(int32_t)truncl(a);")

    def test_float_loads_and_the_nan_test(self) -> None:
        node, text = self.render(
            lambda _s: _Expr("fp_load", (64, False), (self._var("p"),), 64)
        )
        self.assertEqual(node.kind, "load_typed")
        self.assertEqual(node.ctype.name, "double")
        self.assertIn("*(double *)", text)

        node, text = self.render(
            lambda _s: _Expr("fp_int_load", (32, True), (self._var("p"),), 80)
        )
        self.assertEqual(node.kind, "cast")
        self.assertIn("(long double)", text)
        self.assertIn("*(int32_t *)", text)

        node, text = self.render(
            lambda _s: _Expr("fp_nan", None, (self._var("a"),), 1)
        )
        self.assertEqual(node.kind, "func")
        self.assertEqual(text, "isnan(a);")

    def test_x87_two_operand_instructions(self) -> None:
        cases = {
            "fpatan": "atan2l(a,b);",
            "fprem": "fmodl(a,b);",
            "fprem1": "remainderl(a,b);",
            "fyl2x": "a * log2l(b);",
        }
        for operation, expected in cases.items():
            with self.subTest(operation=operation):
                _node, text = self.render(
                    lambda _s, op=operation: _Expr(
                        "fp_func", op, (self._var("a"), self._var("b")), 80
                    )
                )
                self.assertEqual(text, expected)

        _node, text = self.render(
            lambda _s: _Expr(
                "fp_func", "fscale", (self._var("a"), self._var("b")), 80
            )
        )
        self.assertEqual(text, "a * exp2l(truncl(b));")

    def test_float_jump_conditions_rebuild_the_parity_test(self) -> None:
        node, text = self.render(
            lambda _s: _Expr(
                "fp_condition", "jae", (self._var("a"), self._var("b")), 1
            )
        )
        self.assertEqual(node.kind, "op")
        self.assertEqual(node.name, "land")
        self.assertIn("isnan(a)", text)
        self.assertIn("isnan(b)", text)
        self.assertIn("||", text)
        self.assertIn("&&", text)
        self.assertIn("a >= b", text)
        self.assertTrue(text.startswith("!"))

        _node, text = self.render(
            lambda _s: _Expr("fp_condition", "jp", (self._var("a"), self._var("b")), 1)
        )
        self.assertEqual(text, "isnan(a) || isnan(b);")

        _node, text = self.render(
            lambda _s: _Expr(
                "fp_condition", "jnp", (self._var("a"), self._var("b")), 1
            )
        )
        self.assertEqual(text, "!(isnan(a) || isnan(b));")

        _node, text = self.render(
            lambda _s: _Expr("fp_condition", "jbe", (self._var("a"), self._var("b")), 1)
        )
        self.assertIn("a <= b", text)

    def test_concat_becomes_a_shift_and_or_tree(self) -> None:
        node, text = self.render(
            lambda _s: _Expr(
                "concat", 32, (self._var("high", 32), self._var("low", 32)), 64
            )
        )
        self.assertEqual(node.kind, "op")
        self.assertEqual(node.name, "or")
        self.assertIn("<<", text)
        self.assertIn("|", text)
        self.assertIn("(uint64_t)", text)
        self.assertIn("(uint32_t)", text)

    def test_wide_simd_forms_stay_opaque_but_keep_their_text(self) -> None:
        # Without a catalog entry the placeholder name is kept.
        node, text = self.render(
            lambda _s: _Expr("simd_float", 64, (self._var("x"),), 64),
            width=200,
            catalog_text=None,
        )
        self.assertEqual(node.kind, "opaque")
        self.assertIn("unknown_expr_simd_float", text)

        # With one, the exact rendering of the backend is emitted verbatim.
        rendered = "((union { uint64_t i; double f; }){ .i = (uint64_t)(x) }).f"
        printer, stream = _printer(width=400)
        symbols = _symbols(printer)
        expression = _Expr("simd_float", 64, (_Expr("var", "x", (), 64),), 64)
        converter = ExpressionConverter(
            printer.resolver, symbols, 64, {rendered: expression}
        )
        node = converter.convert(expression)
        self.assertEqual(node.kind, "opaque")
        self.assertEqual(node.text, rendered)
        _emit_expression(printer, node)
        self.assertIn(rendered, stream.getvalue())

    def test_modelled_forms_are_marked_up_token_by_token(self) -> None:
        _node, markup = self.render(
            lambda _s: _Expr(
                "fp_condition", "jb", (self._var("a"), self._var("b")), 1
            ),
            markup=True,
        )
        root = ElementTree.fromstring(markup)
        operators = {element.text for element in root.iter("op")}
        self.assertTrue({"||", "&&", "<"} & operators)
        # PrintC::opFunc pushes the name as an operator token (optoken), not as
        # a funcname, exactly like the C++ code.
        self.assertIn("isnan", operators)
        self.assertNotIn("<", {element.text for element in root.iter("funcname")})


class OpaqueTokenTests(unittest.TestCase):
    """Opaque text is split into C tokens instead of one unbreakable blob."""

    SAMPLES = (
        "((union { uint64_t i; double f; }){ .i = (uint64_t)(x) }).f",
        "(((uint64_t)(a) << 32) | (uint32_t)(b))",
        'call("text, with spaces", 0x1f, -3)',
        "isnan(a) || isnan(b)",
        "a->b.c[3] = *p++;",
        "x",
        "",
    )

    def test_splitting_never_loses_a_character(self) -> None:
        for text in self.SAMPLES:
            with self.subTest(text=text):
                pieces = _split_opaque(text)
                self.assertEqual("".join(piece for _kind, piece in pieces), text)

    def test_segments_keep_the_text_and_merge_whitespace_left(self) -> None:
        for text in self.SAMPLES:
            with self.subTest(text=text):
                segments = _opaque_segments(text)
                self.assertEqual("".join(piece for piece, _hl in segments), text)
                # Only a leading run can start with whitespace.
                for piece, _highlight in segments[1:]:
                    self.assertFalse(piece.startswith((" ", "\t", "\n")))

    def test_a_break_is_never_placed_after_a_dot(self) -> None:
        segments = _opaque_segments("{ .i = x }.field->other")
        self.assertIn(".i", [piece.strip() for piece, _hl in segments])
        self.assertIn(".field", [piece.strip() for piece, _hl in segments])
        self.assertIn("->other", [piece.strip() for piece, _hl in segments])

    def test_identifiers_numbers_and_types_get_their_own_colour(self) -> None:
        printer, _stream = _printer()
        symbols = _symbols(printer)
        inner = _Expr("var", "local_10", (), 64)
        symbol = symbols.anonymous("local_10", 64)
        expression = _Expr("simd_float", 64, (inner,), 64)
        text = "((union { uint64_t i; double f; }){ .i = (uint64_t)(local_10) }).f"
        converter = ExpressionConverter(
            printer.resolver, symbols, 64, {text: expression}
        )
        node = converter.convert(expression)
        self.assertEqual(node.kind, "opaque")
        self.assertEqual(node.text, text)

        document = SimpleNamespace(symbols={"local_10": symbol})
        highlights = {
            piece.strip(): highlight
            for piece, highlight in _opaque_segments(node.text, document)
        }
        self.assertEqual(highlights["uint64_t"], SyntaxHighlight.type_color)
        self.assertEqual(highlights["double"], SyntaxHighlight.type_color)
        self.assertEqual(highlights["union"], SyntaxHighlight.keyword_color)
        self.assertEqual(highlights["local_10"], SyntaxHighlight.var_color)
        self.assertEqual(highlights[".i"], SyntaxHighlight.no_color)

    def test_opaque_markup_emits_one_element_per_token(self) -> None:
        printer, stream = _printer(markup=True, width=400)
        text = "((union { uint64_t i; double f; }){ .i = (uint64_t)(x) }).f"
        node = PrintExpr(kind="opaque", text=text)
        _emit_expression(printer, node)
        root = ElementTree.fromstring(stream.getvalue())
        syntax = [element.text or "" for element in root.iter("syntax")]
        self.assertEqual("".join(syntax), text + ";")
        self.assertGreater(len(syntax), 10)
        colours = {
            (element.text or ""): element.attrib.get("color")
            for element in root.iter("syntax")
        }
        self.assertEqual(colours["uint64_t"], "0x2")  # type_color

    def test_a_long_opaque_expression_wraps_instead_of_overflowing(self) -> None:
        printer, stream = _printer(width=60)
        text = (
            "((((__uint128_t)(old_destination) >> 64) << 64) | "
            "(__uint128_t)(((union { double f; uint64_t i; })"
            "{ .f = (double)(raw_result) }).i)) + "
            "((((__uint128_t)(other_destination) >> 64) << 64) | "
            "(__uint128_t)(((union { double f; uint64_t i; })"
            "{ .f = (double)(other_result) }).i))"
        )
        _emit_expression(printer, PrintExpr(kind="opaque", text=text))
        rendered = stream.getvalue()
        lines = [line for line in rendered.splitlines() if line.strip()]
        self.assertGreater(len(lines), 2)
        for line in lines:
            self.assertLessEqual(len(line.rstrip()), 80)
        # Removing the inserted breaks gives the original text back.
        joined = "".join(line.strip() for line in lines)
        self.assertEqual(
            "".join(joined.split()), "".join((text + ";").split())
        )


class ForLoopHelperTests(unittest.TestCase):
    def test_expr_mentions_walks_casts_and_opaque_text(self) -> None:
        variable = PrintExpr(kind="var", name="loop_i")
        self.assertTrue(_expr_mentions(variable, "loop_i"))
        self.assertFalse(_expr_mentions(variable, "loop_j"))
        cast = PrintExpr(kind="cast", args=(variable,))
        self.assertTrue(_expr_mentions(cast, "loop_i"))
        opaque = PrintExpr(kind="opaque", text="(loop_i + 1) * 2")
        self.assertTrue(_expr_mentions(opaque, "loop_i"))
        self.assertFalse(_expr_mentions(opaque, "loop_i2"))
        self.assertFalse(_expr_mentions(None, "loop_i"))

    def test_split_phi_inputs_finds_the_iterator(self) -> None:
        loop = PrintExpr(kind="var", name="i")
        statement = PrintStatement(
            kind="phi",
            op=PrintOp(op_time=1),
            out_name="i",
            values=(
                PrintExpr(kind="const", int_value=0),
                PrintExpr(kind="op", name="add", args=(loop, PrintExpr(kind="const", int_value=1))),
            ),
        )
        init, iterate = _split_phi_inputs(statement, "i")
        self.assertEqual(init.kind, "const")
        self.assertEqual(iterate.kind, "op")

        # Neither input reads the variable: no iterator statement can be formed.
        plain = PrintStatement(
            kind="phi",
            op=PrintOp(op_time=2),
            out_name="i",
            values=(
                PrintExpr(kind="const", int_value=0),
                PrintExpr(kind="const", int_value=1),
            ),
        )
        self.assertEqual(_split_phi_inputs(plain, "i"), (None, None))

    def test_for_loops_can_be_switched_off(self) -> None:
        printer, _stream = _printer()
        self.assertTrue(printer.analyze_for_loops)
        printer.set_analyze_for_loops(False)
        self.assertFalse(printer.analyze_for_loops)
        self.assertIsNone(_for_loop_form(printer, None))
        printer.reset_defaults()
        self.assertTrue(printer.analyze_for_loops)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
