"""Tests for the headless helpers of the Tkinter front-end.

``launch_gui`` itself needs a display, but everything it uses to turn the
printer markup into coloured text lives in :mod:`minighidra.gui` as plain
functions, so the printing tab can be checked without Tk.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from minighidra import Analyzer, BinaryImage
from minighidra.clangdoc import ClangDocument, ClangToken
from minighidra.emit import BraceStyle, SyntaxHighlight
from minighidra.gui import (
    BRACE_STYLES,
    COMMENT_STYLES,
    HIGHLIGHT_COLORS,
    INTEGER_FORMATS,
    LABEL_MODES,
    PRINT_MODES,
    document_report,
    hex_dump,
    highlight_tag,
    packed_report,
)

# push/pop rbp, a couple of moves and a return: enough for a small document.
TINY_SAMPLE = "55" "4889e5" "8b4510" "83c001" "894514" "5d" "c3"


def _document():
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "sample.bin"
    path.write_bytes(bytes.fromhex(TINY_SAMPLE))
    image = BinaryImage.load(path, raw_arch="x86_64", base=0x401000)
    analyzer = Analyzer(image)
    function = analyzer.functions[0x401000]
    return analyzer.clang_document(function), analyzer, function, directory


class HighlightPaletteTests(unittest.TestCase):
    def test_every_syntax_highlight_has_a_colour(self) -> None:
        for highlight in SyntaxHighlight:
            self.assertIn(highlight.value, HIGHLIGHT_COLORS, highlight.name)
            colour, bold = HIGHLIGHT_COLORS[highlight.value]
            self.assertTrue(colour.startswith("#"))
            self.assertEqual(len(colour), 7)
            self.assertIsInstance(bold, bool)

    def test_tags_are_unique_per_colour_code(self) -> None:
        tags = {highlight_tag(code) for code in HIGHLIGHT_COLORS}
        self.assertEqual(len(tags), len(HIGHLIGHT_COLORS))
        self.assertEqual(highlight_tag(ClangToken.KEYWORD_COLOR), "hl0")
        self.assertEqual(highlight_tag(ClangToken.DEFAULT_COLOR), "hl8")

    def test_the_clang_colour_codes_match_the_emitter(self) -> None:
        self.assertEqual(ClangToken.KEYWORD_COLOR, SyntaxHighlight.keyword_color)
        self.assertEqual(ClangToken.COMMENT_COLOR, SyntaxHighlight.comment_color)
        self.assertEqual(ClangToken.TYPE_COLOR, SyntaxHighlight.type_color)
        self.assertEqual(ClangToken.FUNCTION_COLOR, SyntaxHighlight.funcname_color)
        self.assertEqual(ClangToken.VARIABLE_COLOR, SyntaxHighlight.var_color)
        self.assertEqual(ClangToken.CONST_COLOR, SyntaxHighlight.const_color)
        self.assertEqual(ClangToken.PARAMETER_COLOR, SyntaxHighlight.param_color)
        self.assertEqual(ClangToken.GLOBAL_COLOR, SyntaxHighlight.global_color)
        self.assertEqual(ClangToken.DEFAULT_COLOR, SyntaxHighlight.no_color)
        self.assertEqual(ClangToken.ERROR_COLOR, SyntaxHighlight.error_color)
        self.assertEqual(ClangToken.SPECIAL_COLOR, SyntaxHighlight.special_color)


class OptionValueTests(unittest.TestCase):
    def test_the_view_options_are_the_ones_the_printer_accepts(self) -> None:
        document, analyzer, function, directory = _document()
        self.addCleanup(directory.cleanup)
        self.assertEqual(len(PRINT_MODES), 3)
        for brace in BRACE_STYLES:
            BraceStyle[brace]  # raises KeyError for an unknown style
            analyzer.print_function(function, brace=brace, max_line_size=200)
        for integer_format in INTEGER_FORMATS:
            analyzer.print_function(
                function, integer_format=integer_format, max_line_size=200
            )
        for labels in LABEL_MODES:
            analyzer.print_function(function, labels=labels, max_line_size=200)
        for comment_style in COMMENT_STYLES:
            analyzer.print_function(
                function, comment_style=comment_style, max_line_size=200
            )


class HexDumpTests(unittest.TestCase):
    def test_rows_carry_offset_hex_and_ascii(self) -> None:
        data = bytes(range(0x20))
        dump = hex_dump(data)
        rows = dump.splitlines()
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0].startswith("00000000  "))
        self.assertTrue(rows[1].startswith("00000010  "))
        self.assertIn("00 01 02 03", rows[0])
        # Non printable bytes become dots.
        self.assertIn("|" + "." * 16 + "|", rows[0])

    def test_printable_bytes_show_up_in_the_ascii_column(self) -> None:
        dump = hex_dump(b"AB\x00\xff")
        self.assertIn("|AB..", dump)
        self.assertIn("41 42 00 ff", dump)

    def test_a_partial_last_row_is_padded(self) -> None:
        rows = hex_dump(bytes(range(18))).splitlines()
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), len(rows[1]))

    def test_the_empty_stream_dumps_nothing(self) -> None:
        self.assertEqual(hex_dump(b""), "")

    def test_every_byte_is_listed_exactly_once(self) -> None:
        data = bytes(range(256)) * 2
        rows = hex_dump(data).splitlines()
        self.assertEqual(len(rows), len(data) // 16)
        recovered = bytearray()
        for row in rows:
            _offset, rest = row.split("  ", 1)
            recovered += bytes.fromhex(rest.split("|")[0].replace(" ", ""))
        self.assertEqual(bytes(recovered), data)


class ReportTests(unittest.TestCase):
    def test_packed_report_compares_both_formats(self) -> None:
        report = packed_report(b"\x52\x60", "<clang_document></clang_document>")
        self.assertIn("PackedEncode: 2 bytes", report)
        self.assertIn("XmlEncode: 33 bytes", report)
        self.assertIn("6%", report)
        self.assertTrue(report.endswith("\n\n"))

    def test_packed_report_survives_an_empty_xml_document(self) -> None:
        self.assertIn("0%", packed_report(b"", ""))

    def test_document_report_counts_lines_tokens_and_signature(self) -> None:
        document, _analyzer, _function, directory = _document()
        self.addCleanup(directory.cleanup)
        report = document_report(document)
        self.assertTrue(report.startswith("/*"))
        self.assertTrue(report.rstrip().endswith("*/"))
        self.assertIn(f"{len(document.lines)} líneas", report)
        self.assertIn("tokens marcados", report)
        self.assertIn("firma: uint64_t entry(", report)

    def test_the_spans_of_the_document_are_the_text_of_the_tab(self) -> None:
        document, analyzer, function, directory = _document()
        self.addCleanup(directory.cleanup)
        spans = document.spans()
        self.assertEqual(
            "".join(text for text, _colour in spans).rstrip("\n"),
            analyzer.print_function(function).rstrip("\n"),
        )
        for _text, colour in spans:
            self.assertIn(colour, HIGHLIGHT_COLORS)

    def test_a_document_without_a_prototype_has_no_signature(self) -> None:
        document = ClangDocument.parse(
            "<clang_document><syntax>x</syntax></clang_document>"
        )
        self.assertIsNone(document.signature())
        self.assertNotIn("firma", document_report(document))
        self.assertEqual(document.text().strip(), "x")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
