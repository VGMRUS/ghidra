from __future__ import annotations

import io
import unittest

from minighidra.emit import (
    ATTRIB_COLOR,
    ATTRIB_CONTENT,
    ATTRIB_INDENT,
    ATTRIB_OFF,
    ATTRIB_OPEN,
    ATTRIB_SPACE,
    ELEM_BREAK,
    ELEM_CLANG_DOCUMENT,
    BraceStyle,
    CircularQueue,
    Emit,
    EmitError,
    EmitMarkup,
    EmitNoMarkup,
    EmitPrettyPrint,
    PendPrint,
    PrintClass,
    SyntaxHighlight,
    TagType,
    TokenSplit,
    XmlEncoder,
    xml_escape,
)


class _Ref:
    """Stand-in for the objects Ghidra references from markup attributes."""

    def __init__(self, **attributes) -> None:
        for name, value in attributes.items():
            setattr(self, name, value)


class XmlEncoderTests(unittest.TestCase):
    def test_xml_escape_replaces_special_characters(self) -> None:
        self.assertEqual(
            xml_escape('a < b & c > d "e" f\'g'),
            "a &lt; b &amp; c &gt; d &quot;e&quot; f&apos;g",
        )
        self.assertEqual(xml_escape("plain 0x10"), "plain 0x10")

    def test_element_with_content_and_empty_element(self) -> None:
        stream = io.StringIO()
        encoder = XmlEncoder(stream)
        encoder.open_element(ELEM_CLANG_DOCUMENT)
        encoder.write_string(ATTRIB_CONTENT, "<x>")
        encoder.close_element(ELEM_CLANG_DOCUMENT)
        encoder.open_element(ELEM_BREAK)
        encoder.write_signed_integer(ATTRIB_INDENT, 4)
        encoder.close_element(ELEM_BREAK)
        self.assertEqual(
            stream.getvalue(),
            '\n<clang_document>&lt;x&gt;</clang_document>\n<break indent="4"/>',
        )

    def test_unsigned_attributes_are_hexadecimal(self) -> None:
        stream = io.StringIO()
        encoder = XmlEncoder(stream)
        encoder.open_element(ELEM_CLANG_DOCUMENT)
        encoder.write_unsigned_integer(ATTRIB_COLOR, 10)
        encoder.write_string(ATTRIB_CONTENT, "tok")
        encoder.close_element(ELEM_CLANG_DOCUMENT)
        self.assertIn('color="0xa"', stream.getvalue())
        self.assertIn(">tok</clang_document>", stream.getvalue())

    def test_formatting_can_be_disabled(self) -> None:
        stream = io.StringIO()
        encoder = XmlEncoder(stream, do_format=False)
        encoder.open_element(ELEM_BREAK)
        encoder.close_element(ELEM_BREAK)
        self.assertEqual(stream.getvalue(), "<break/>")


class EmitNoMarkupTests(unittest.TestCase):
    def _emitter(self):
        stream = io.StringIO()
        emit = EmitNoMarkup()
        emit.set_output_stream(stream)
        return emit, stream

    def test_tokens_are_dumped_straight_to_the_stream(self) -> None:
        emit, stream = self._emitter()
        emit.begin_document()
        emit.tag_type("uint32_t", SyntaxHighlight.type_color, None)
        emit.spaces(1)
        emit.tag_func_name("main", SyntaxHighlight.funcname_color, None, None)
        emit.open_paren("(")
        emit.tag_variable("argc", SyntaxHighlight.param_color, None, None)
        emit.close_paren(")", 0)
        emit.tag_line()
        emit.print("{")
        emit.end_document(0)
        self.assertEqual(stream.getvalue(), "uint32_t main(argc)\n{")
        self.assertEqual(emit.get_paren_level(), 0)
        self.assertFalse(emit.emits_markup())

    def test_tag_line_honours_explicit_indent(self) -> None:
        emit, stream = self._emitter()
        emit.start_indent()
        emit.tag_line()
        emit.tag_line(6)
        emit.stop_indent(0)
        self.assertEqual(stream.getvalue(), "\n  \n      ")


class EmitMarkupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = io.StringIO()
        self.emit = EmitMarkup()
        self.emit.set_output_stream(self.stream)

    def test_document_and_function_markup(self) -> None:
        function = _Ref()
        block = _Ref(index=3)
        self.emit.begin_document()
        self.emit.begin_function(function)
        self.emit.begin_block(block)
        self.emit.end_block(0)
        self.emit.end_function(0)
        self.emit.end_document(0)
        text = self.stream.getvalue()
        self.assertIn("<clang_document>", text)
        self.assertIn("<function>", text)
        self.assertIn('<block blockref="3"/>', text)
        self.assertTrue(self.emit.emits_markup())

    def test_token_markup_matches_ghidra_elements(self) -> None:
        varnode = _Ref(create_index=7)
        op = _Ref(op_time=11)
        ctype = _Ref(unsized_id=2, name="uint32_t", ctype=None)
        self.emit.tag_variable("local_10", SyntaxHighlight.var_color, varnode, op)
        self.emit.tag_op("=", SyntaxHighlight.no_color, op)
        self.emit.tag_func_name("FUN_1", SyntaxHighlight.funcname_color, None, op)
        self.emit.tag_type("uint32_t", SyntaxHighlight.type_color, ctype)
        self.emit.tag_field("field", SyntaxHighlight.no_color, ctype, 4, op)
        self.emit.tag_bit_field("bits", SyntaxHighlight.no_color, ctype, 3, op)
        self.emit.tag_comment("// hi", SyntaxHighlight.comment_color, _Ref(name="ram"), 0x20)
        self.emit.tag_label("code_r0x20", SyntaxHighlight.no_color, _Ref(name="ram"), 0x20)
        self.emit.tag_case_label("case", SyntaxHighlight.keyword_color, op, 9)
        self.emit.print(";", SyntaxHighlight.no_color)
        text = self.stream.getvalue()
        self.assertIn(
            '<variable color="0x4" varref="0x7" opref="0xb">local_10</variable>', text
        )
        self.assertIn('<op opref="0xb">=</op>', text)
        self.assertIn('<funcname color="0x3" opref="0xb">FUN_1</funcname>', text)
        self.assertIn('<type color="0x2" id="0x2">uint32_t</type>', text)
        self.assertIn('<field name="uint32_t" id="0x2" off="4"', text)
        self.assertIn("<bitfield ", text)
        self.assertIn('<comment color="0x1" space="ram" off="0x20"', text)
        self.assertIn('<label space="ram" off="0x20">code_r0x20</label>', text)
        self.assertIn('<value color="0x0" off="0x9" opref="0xb">case</value>', text)
        self.assertIn("<syntax>;</syntax>", text)

    def test_structure_markup_and_parentheses(self) -> None:
        self.emit.begin_func_proto()
        self.emit.begin_return_type(_Ref(create_index=1))
        self.emit.end_return_type(0)
        self.emit.begin_var_decl(_Ref(symbol_id=2))
        self.emit.end_var_decl(0)
        self.emit.end_func_proto(0)
        identifier = self.emit.open_paren("(", 4)
        self.emit.close_paren(")", identifier)
        self.emit.begin_statement(_Ref(op_time=6))
        self.emit.end_statement(0)
        text = self.stream.getvalue()
        self.assertIn("<funcproto>", text)
        self.assertIn('<return_type varref="0x1"/>', text)
        self.assertIn('<vardecl symref="0x2"/>', text)
        self.assertIn('<syntax open="4">(</syntax>', text)
        self.assertIn('<syntax close="0">)</syntax>', text)
        self.assertIn('<statement opref="0x6"/>', text)
        self.assertEqual(self.emit.get_paren_level(), 0)

    def test_break_records_the_current_indent(self) -> None:
        self.emit.start_indent()
        self.emit.tag_line()
        self.emit.tag_line(0)
        self.assertIn('<break indent="2"/>', self.stream.getvalue())
        self.assertIn('<break indent="0"/>', self.stream.getvalue())

    def test_packed_output_switches_the_codec(self) -> None:
        from minighidra.marshal import (
            ATTRIB_CONTENT,
            ELEM_CLANG_DOCUMENT,
            ELEM_SYNTAX,
            PackedDecode,
        )

        # The text stream stays untouched: the packed bytes live in the encoder.
        self.emit.set_packed_output(True)
        self.emit.begin_document()
        self.emit.print("uint32_t", SyntaxHighlight.type_color)
        self.emit.end_document(0)
        self.assertEqual(self.stream.getvalue(), "")
        data = self.emit.get_packed_bytes()
        self.assertTrue(data)
        self.assertTrue(all(byte != 0 for byte in data))

        decoder = PackedDecode()
        decoder.ingest_stream(data)
        self.assertEqual(decoder.open_element(), ELEM_CLANG_DOCUMENT.id)
        self.assertEqual(decoder.open_element(), ELEM_SYNTAX.id)
        self.assertEqual(decoder.read_unsigned_integer(), SyntaxHighlight.type_color)
        self.assertEqual(decoder.read_string(ATTRIB_CONTENT), "uint32_t")
        decoder.close_element(ELEM_SYNTAX.id)
        decoder.close_element(ELEM_CLANG_DOCUMENT.id)

        # Switching back restores the XML codec on the same stream.
        self.emit.set_packed_output(False)
        self.emit.begin_document()
        self.emit.end_document(0)
        self.assertIn("<clang_document/>", self.stream.getvalue())
        self.assertEqual(self.emit.get_packed_bytes(), b"")

    def test_packed_output_accepts_a_binary_stream(self) -> None:
        stream = io.BytesIO()
        self.emit.set_output_stream(stream)
        self.emit.set_packed_output(True)
        self.emit.begin_document()
        self.emit.print("x")
        self.emit.end_document(0)
        self.assertEqual(stream.getvalue(), self.emit.get_packed_bytes())


class CircularQueueTests(unittest.TestCase):
    def test_stack_and_queue_behaviour(self) -> None:
        queue = CircularQueue(4, int)
        self.assertTrue(queue.empty())
        queue.push(1)
        queue.push(2)
        self.assertEqual(queue.top(), 2)
        self.assertEqual(queue.bottom(), 1)
        self.assertEqual(queue.pop(), 2)
        self.assertEqual(queue.popbottom(), 1)
        self.assertTrue(queue.empty())

    def test_expand_preserves_contents(self) -> None:
        queue = CircularQueue(4, int)
        for value in (5, 6, 7):
            queue.push(value)
        queue.expand(4)
        self.assertEqual(queue.get_max(), 8)
        self.assertEqual(
            [queue.popbottom() for _ in range(3)],
            [5, 6, 7],
        )


class TokenSplitTests(unittest.TestCase):
    def test_begin_end_pairs_get_unique_counts(self) -> None:
        first = TokenSplit()
        second = TokenSplit()
        identifier = first.begin_document()
        self.assertEqual(second.begin_function(None), identifier + 1)
        self.assertEqual(first.get_class(), PrintClass.begin)
        self.assertEqual(first.get_tag(), TagType.docu_b)

    def test_print_dispatches_to_the_matching_emit_call(self) -> None:
        calls: list[str] = []

        class Recorder(EmitNoMarkup):
            def __init__(self) -> None:
                super().__init__()
                self.set_output_stream(io.StringIO())

            def tag_variable(self, name, hl, vn, op) -> None:
                calls.append(f"variable:{name}:{int(hl)}")

            def tag_op(self, name, hl, op) -> None:
                calls.append(f"op:{name}")

            def spaces(self, num, bump=0) -> None:
                calls.append(f"spaces:{num}")

        token = TokenSplit()
        token.tag_variable("x", SyntaxHighlight.var_color, None, None)
        token.do_print(Recorder())
        token2 = TokenSplit()
        token2.tag_op("+", SyntaxHighlight.no_color, None)
        token2.do_print(Recorder())
        token3 = TokenSplit()
        token3.spaces(3, 0)
        token3.do_print(Recorder())
        self.assertEqual(calls, ["variable:x:4", "op:+", "spaces:3"])

    def test_line_tokens_are_never_printed_directly(self) -> None:
        token = TokenSplit()
        token.tag_line()
        with self.assertRaises(EmitError):
            token.do_print(EmitNoMarkup())


class _PendingOpenBrace(PendPrint):
    def __init__(self) -> None:
        self.issued = -1

    def get_indent_id(self) -> int:
        return self.issued

    def callback(self, emit: Emit) -> None:
        self.issued = emit.open_brace_indent("{", BraceStyle.same_line)


class EmitPrettyPrintTests(unittest.TestCase):
    def _printer(self, markup: bool = False, width: int = 100):
        stream = io.StringIO()
        emit = EmitPrettyPrint()
        emit.set_output_stream(stream)
        emit.set_markup(markup)
        emit.set_max_line_size(width)
        return emit, stream

    def _document(self, emit, body) -> None:
        document = emit.begin_document()
        function = emit.begin_function(None)
        body(emit)
        emit.end_function(function)
        emit.end_document(document)
        emit.flush()

    def test_default_line_size_and_indent_increment(self) -> None:
        emit, _ = self._printer()
        self.assertEqual(emit.get_max_line_size(), 100)
        self.assertEqual(emit.get_indent_increment(), 2)
        with self.assertRaises(EmitError):
            emit.set_max_line_size(10)

    def test_indentation_follows_nesting_levels(self) -> None:
        emit, stream = self._printer()

        def body(emitter):
            indent = emitter.open_brace_indent("{", BraceStyle.next_line)
            emitter.tag_line()
            emitter.print("a;")
            inner = emitter.start_indent()
            emitter.tag_line()
            emitter.print("b;")
            emitter.stop_indent(inner)
            emitter.close_brace_indent("}", indent)

        self._document(emit, body)
        self.assertEqual(stream.getvalue(), "\n{\n  a;\n    b;\n}")

    def test_long_lines_break_at_group_boundaries(self) -> None:
        emit, stream = self._printer(width=30)

        def body(emitter):
            emitter.tag_line()
            emitter.tag_variable("result", SyntaxHighlight.var_color, None, None)
            emitter.spaces(1)
            emitter.tag_op("=", SyntaxHighlight.no_color, None)
            emitter.spaces(1)
            emitter.open_group()
            emitter.tag_func_name(
                "some_long_name", SyntaxHighlight.funcname_color, None, None
            )
            identifier = emitter.open_paren("(")
            emitter.tag_variable("first_argument", SyntaxHighlight.var_color, None, None)
            emitter.print(",")
            emitter.spaces(1)
            emitter.tag_variable("second_argument", SyntaxHighlight.var_color, None, None)
            emitter.close_paren(")", identifier)
            emitter.close_group(0)
            emitter.print(";")

        self._document(emit, body)
        text = stream.getvalue()
        self.assertIn("result = some_long_name", text)
        self.assertIn("\n         (first_argument,", text)
        self.assertTrue(all(len(line) <= 30 for line in text.splitlines()), text)

    def test_comment_fill_is_repeated_after_forced_breaks(self) -> None:
        emit, stream = self._printer(width=24)
        emit.set_comment_fill(" * ")

        def body(emitter):
            identifier = emitter.start_comment()
            for word in ("one", "two", "three", "four", "five", "six", "seven"):
                emitter.tag_comment(word, SyntaxHighlight.comment_color, None, 0)
                emitter.spaces(1)
            emitter.stop_comment(identifier)

        self._document(emit, body)
        lines = [line for line in stream.getvalue().splitlines() if line.strip()]
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(line.startswith(" * ") for line in lines[1:]), lines)

    def test_markup_mode_switches_the_low_level_emitter(self) -> None:
        emit, stream = self._printer(markup=True, width=60)
        self.assertTrue(emit.emits_markup())

        def body(emitter):
            emitter.tag_line()
            identifier = emitter.begin_statement(None)
            emitter.tag_variable("x", SyntaxHighlight.var_color, None, None)
            emitter.end_statement(identifier)

        self._document(emit, body)
        text = stream.getvalue()
        self.assertIn("<clang_document>", text)
        self.assertIn("<statement>", text)
        self.assertIn('<variable color="0x4">x</variable>', text)
        emit.set_markup(False)
        self.assertFalse(emit.emits_markup())

    def test_flush_requires_balanced_groups(self) -> None:
        emit, _ = self._printer()
        emit.begin_document()
        emit.print("dangling")
        with self.assertRaises(EmitError):
            emit.flush()

    def test_clear_resets_state(self) -> None:
        emit, stream = self._printer()
        document = emit.begin_document()
        emit.open_paren("(")
        emit.close_paren(")", 0)
        emit.end_document(document)
        emit.flush()
        emit.clear()
        self.assertEqual(emit.get_paren_level(), 0)
        self.assertEqual(emit.indentlevel, 0)
        # A fresh document can still be emitted after clear().
        document = emit.begin_document()
        emit.print("x")
        emit.end_document(document)
        emit.flush()
        self.assertIn("x", stream.getvalue())

    def test_pending_print_is_cancelable(self) -> None:
        emit, stream = self._printer(width=60)
        document = emit.begin_document()
        function = emit.begin_function(None)
        pending = _PendingOpenBrace()
        emit.set_pending_print(pending)
        self.assertTrue(emit.has_pending_print(pending))
        emit.cancel_pending_print()
        self.assertFalse(emit.has_pending_print(pending))
        emit.tag_line()
        self.assertEqual(pending.issued, -1)

        emit.set_pending_print(pending)
        emit.tag_line()  # the callback runs before the line break
        self.assertGreaterEqual(pending.issued, 0)
        emit.tag_line()
        emit.print("b;")
        emit.close_brace_indent("}", pending.get_indent_id())
        emit.end_function(function)
        emit.end_document(document)
        emit.flush()
        self.assertIn("\n {", stream.getvalue())

    def test_brace_styles(self) -> None:
        for style, expected in (
            (BraceStyle.same_line, "\n\na; {\n  b;\n}"),
            (BraceStyle.next_line, "\n\na;\n{\n  b;\n}"),
            (BraceStyle.skip_line, "\n\na;\n\n{\n  b;\n}"),
        ):
            with self.subTest(style=style.name):
                emit, stream = self._printer(width=60)
                document = emit.begin_document()
                emit.tag_line()
                emit.tag_line()
                emit.print("a;")
                indent = emit.open_brace_indent("{", style)
                emit.tag_line()
                emit.print("b;")
                emit.close_brace_indent("}", indent)
                emit.end_document(document)
                emit.flush()
                self.assertEqual(stream.getvalue(), expected)


class EmitInterfaceTests(unittest.TestCase):
    def test_spaces_uses_the_precomputed_table(self) -> None:
        stream = io.StringIO()
        emit = EmitNoMarkup()
        emit.set_output_stream(stream)
        emit.spaces(3)
        emit.spaces(12)
        self.assertEqual(stream.getvalue(), "   " + " " * 12)

    def test_markup_ids_match_ghidra(self) -> None:
        self.assertEqual((ELEM_CLANG_DOCUMENT.name, ELEM_CLANG_DOCUMENT.id), ("clang_document", 18))
        self.assertEqual((ELEM_BREAK.name, ELEM_BREAK.id), ("break", 17))
        self.assertEqual((ATTRIB_CONTENT.name, ATTRIB_CONTENT.id), ("XMLcontent", 1))
        self.assertEqual((ATTRIB_OFF.name, ATTRIB_OFF.id), ("off", 39))
        self.assertEqual((ATTRIB_OPEN.name, ATTRIB_OPEN.id), ("open", 40))
        self.assertEqual((ATTRIB_SPACE.name, ATTRIB_SPACE.id), ("space", 20))
        self.assertEqual(SyntaxHighlight.special_color.value, 10)
        self.assertEqual(BraceStyle.skip_line.value, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
