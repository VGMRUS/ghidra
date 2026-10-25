"""Tests for the Clang document model (``ghidra.app.decompiler.Clang*``)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from minighidra import Analyzer, BinaryImage
from minighidra.clangdoc import (
    INDENT_STRING,
    ClangBlock,
    ClangBreak,
    ClangCommentToken,
    ClangDocument,
    ClangFuncNameToken,
    ClangFuncProto,
    ClangFunction,
    ClangLabelToken,
    ClangLine,
    ClangOpToken,
    ClangPrettyPrinter,
    ClangReferenceResolver,
    ClangReturnType,
    ClangStatement,
    ClangSyntaxToken,
    ClangToken,
    ClangTokenGroup,
    ClangTypeToken,
    ClangVariableDecl,
    ClangVariableToken,
    build_clang_tree,
    to_lines,
)
from minighidra.marshal import DecoderError

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
# An unconditional loop plus a goto: exercises labels and comments.
GOTO_SAMPLE = (
    "55" "4889e5" "eb04" "90" "9090" "b801000000" "ebf6" "5d" "c3"
)


class _Recording(ClangReferenceResolver):
    """A resolver that records every reference id it is asked about."""

    def __init__(self) -> None:
        self.refs: list[int] = []
        self.op_refs: list[int] = []
        self.symbols: list[int] = []
        self.blocks: list[int] = []

    def get_ref(self, refid: int):
        self.refs.append(refid)
        return f"varnode#{refid}"

    def get_op_ref(self, refid: int):
        self.op_refs.append(refid)
        return f"op#{refid}"

    def get_symbol(self, symref: int):
        self.symbols.append(symref)
        return f"symbol#{symref}"

    def get_block(self, blockref: int):
        self.blocks.append(blockref)
        return f"block#{blockref}"


class ClangDocumentTestBase(unittest.TestCase):
    def _analyze(self, hexcode: str, base: int = 0x401000):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "sample.bin"
        path.write_bytes(bytes.fromhex(hexcode))
        image = BinaryImage.load(path, raw_arch="x86_64", base=base)
        analyzer = Analyzer(image)
        return analyzer, analyzer.functions[base]

    def _markup(self, hexcode: str = CALL_SAMPLE, **options) -> str:
        analyzer, function = self._analyze(hexcode)
        return analyzer.print_function(function, markup=True, **options)

    def _document(self, hexcode: str = CALL_SAMPLE, **options):
        analyzer, function = self._analyze(hexcode)
        return analyzer.clang_document(function, **options), analyzer, function


class BuildClangTreeTests(ClangDocumentTestBase):
    def test_the_tree_follows_the_markup_hierarchy(self) -> None:
        root = build_clang_tree(self._markup())
        self.assertIsInstance(root, ClangTokenGroup)
        function = root.child(0)
        self.assertIsInstance(function, ClangFunction)
        self.assertIs(function.get_clang_function(), function)
        kinds = {type(node) for node in _walk(root)}
        for expected in (
            ClangFunction,
            ClangFuncProto,
            ClangReturnType,
            ClangVariableDecl,
            ClangBlock,
            ClangStatement,
            ClangSyntaxToken,
            ClangVariableToken,
            ClangTypeToken,
            ClangFuncNameToken,
            ClangBreak,
        ):
            self.assertIn(expected, kinds)
        # <return_type> belongs to the prototype, not to the function group.
        proto = next(
            node for node in _walk(root) if isinstance(node, ClangFuncProto)
        )
        self.assertTrue(
            any(isinstance(node, ClangReturnType) for node in _walk(proto))
        )

    def test_packed_markup_builds_the_same_tree(self) -> None:
        analyzer, function = self._analyze(CALL_SAMPLE)
        from_xml = build_clang_tree(
            analyzer.print_function(function, markup=True)
        )
        from_packed = build_clang_tree(
            analyzer.print_function(function, markup=True, packed=True)
        )
        self.assertEqual(
            [(type(node).__name__, str(node)) for node in _flatten(from_xml)],
            [(type(node).__name__, str(node)) for node in _flatten(from_packed)],
        )

    def test_token_attributes_survive_the_round_trip(self) -> None:
        root = build_clang_tree(self._markup())
        tokens = [node for node in _flatten(root) if isinstance(node, ClangToken)]
        self.assertTrue(tokens)

        parens = [
            token
            for token in tokens
            if isinstance(token, ClangSyntaxToken) and token.get_open() >= 0
        ]
        self.assertTrue(parens)
        self.assertEqual(parens[0].get_text(), "(")

        variables = [token for token in tokens if isinstance(token, ClangVariableToken)]
        self.assertTrue(variables)
        self.assertEqual(
            variables[0].get_syntax_type(), ClangToken.PARAMETER_COLOR
        )

        funcnames = [token for token in tokens if isinstance(token, ClangFuncNameToken)]
        self.assertTrue(funcnames)
        self.assertEqual(funcnames[0].get_syntax_type(), ClangToken.FUNCTION_COLOR)

        types = [token for token in tokens if isinstance(token, ClangTypeToken)]
        self.assertTrue(types)
        self.assertEqual(types[0].get_syntax_type(), ClangToken.TYPE_COLOR)

        breaks = [token for token in tokens if isinstance(token, ClangBreak)]
        self.assertTrue(breaks)
        self.assertEqual(breaks[0].get_text(), "")
        self.assertGreaterEqual(breaks[-1].get_indent(), 0)

    def test_comments_keep_their_address(self) -> None:
        root = build_clang_tree(self._markup())
        comments = [
            node for node in _flatten(root) if isinstance(node, ClangCommentToken)
        ]
        self.assertTrue(comments)
        self.assertEqual(comments[0].get_syntax_type(), ClangToken.COMMENT_COLOR)
        self.assertIsNotNone(comments[0].srcaddr)
        self.assertEqual(comments[0].srcaddr.space, "ram")
        self.assertEqual(
            root.min_address().offset, min(c.srcaddr.offset for c in comments)
        )

    def test_labels_keep_their_block_address(self) -> None:
        root = build_clang_tree(self._markup(GOTO_SAMPLE, labels="join"))
        labels = [node for node in _flatten(root) if isinstance(node, ClangLabelToken)]
        self.assertTrue(labels)
        self.assertIsNotNone(labels[0].blockaddr)
        self.assertIn("code_r", labels[0].get_text())

    def test_statements_blocks_and_declarations_carry_their_references(self) -> None:
        resolver = _Recording()
        root = build_clang_tree(self._markup(), resolver=resolver)
        statements = [
            node for node in _walk(root) if isinstance(node, ClangStatement)
        ]
        self.assertTrue(statements)
        self.assertTrue(any(node.opref is not None for node in statements))
        self.assertTrue(any(node.op is not None for node in statements))
        self.assertTrue(resolver.op_refs)
        self.assertTrue(resolver.refs)
        declarations = [
            node for node in _walk(root) if isinstance(node, ClangVariableDecl)
        ]
        self.assertTrue(declarations)
        self.assertTrue(resolver.symbols)
        blocks = [node for node in _walk(root) if isinstance(node, ClangBlock)]
        self.assertTrue(blocks)

    def test_operators_become_op_tokens(self) -> None:
        root = build_clang_tree(self._markup())
        operators = [node for node in _flatten(root) if isinstance(node, ClangOpToken)]
        texts = {token.get_text() for token in operators}
        self.assertIn("=", texts)
        self.assertTrue(texts & {"+", "!=", "return"})

    def test_an_unknown_element_is_a_decode_error(self) -> None:
        with self.assertRaises(DecoderError):
            build_clang_tree("<clang_document><mystery/></clang_document>")

    def test_a_function_root_is_accepted_directly(self) -> None:
        root = build_clang_tree("<function><syntax>x</syntax></function>")
        self.assertIsInstance(root, ClangFunction)
        self.assertEqual(str(root), "x")


class LineLayoutTests(ClangDocumentTestBase):
    def test_lines_split_at_breaks_and_keep_the_indent(self) -> None:
        root = build_clang_tree(self._markup())
        lines = to_lines(root)
        self.assertGreater(len(lines), 3)
        self.assertEqual(lines[0].get_line_number(), 1)
        self.assertEqual(
            [line.get_line_number() for line in lines],
            list(range(1, len(lines) + 1)),
        )
        indents = {line.get_indent() for line in lines}
        self.assertIn(0, indents)
        self.assertIn(2, indents)
        self.assertEqual(lines[0].get_indent_string(), "")
        body = next(line for line in lines if line.get_indent() == 2)
        self.assertEqual(body.get_indent_string(), INDENT_STRING * 2)

    def test_every_token_knows_its_line(self) -> None:
        root = build_clang_tree(self._markup())
        lines = to_lines(root)
        for line in lines:
            for token in line.get_all_tokens():
                self.assertIs(token.get_line_parent(), line)
                self.assertIs(line.get_token(line.index_of_token(token)), token)

    def test_a_comment_is_glued_back_into_one_token(self) -> None:
        root = build_clang_tree(self._markup())
        comments = [
            token
            for line in to_lines(root)
            for token in line.get_all_tokens()
            if isinstance(token, ClangCommentToken)
        ]
        self.assertTrue(comments)
        # The emitter splits comments word by word; toLines collapses them.
        self.assertIn("Experimental semantic pseudocode", comments[0].get_text())
        for token in comments:
            self.assertNotIn("\n", token.get_text())

    def test_empty_lines_get_a_spacer_token(self) -> None:
        document, _analyzer, _function = self._document()
        padded = [line for line in document.lines if line.get_all_tokens()]
        self.assertEqual(len(padded), len(document.lines))
        spacer = ClangToken.build_spacer(None, 3)
        self.assertIsInstance(spacer, ClangSyntaxToken)
        self.assertEqual(spacer.get_text(), INDENT_STRING * 3)


class PrettyPrinterTests(ClangDocumentTestBase):
    def test_the_rendered_text_matches_the_plain_emission(self) -> None:
        analyzer, function = self._analyze(CALL_SAMPLE)
        plain = analyzer.print_function(function, max_line_size=100)
        markup = analyzer.print_function(function, markup=True, max_line_size=100)
        printer = ClangPrettyPrinter(build_clang_tree(markup))
        self.assertEqual(printer.print().code.rstrip("\n"), plain.rstrip("\n"))

    def test_the_rendered_text_matches_for_every_width(self) -> None:
        analyzer, function = self._analyze(LOOP_SAMPLE)
        for width in (20, 60, 100, 200):
            plain = analyzer.print_function(function, max_line_size=width)
            markup = analyzer.print_function(
                function, markup=True, max_line_size=width
            )
            document = ClangDocument.parse(markup)
            self.assertEqual(
                document.text().rstrip("\n"),
                plain.rstrip("\n"),
                f"width {width}",
            )

    def test_the_signature_is_recovered_from_the_prototype(self) -> None:
        document, _analyzer, _function = self._document()
        signature = document.signature()
        self.assertIsNotNone(signature)
        self.assertTrue(signature.endswith(";"))
        self.assertIn("entry(", signature)
        self.assertIn("uint64_t", signature)

    def test_spans_reproduce_the_text_with_colours(self) -> None:
        document, _analyzer, _function = self._document()
        spans = document.spans()
        self.assertEqual(
            "".join(text for text, _colour in spans).rstrip("\n"),
            document.text().rstrip("\n"),
        )
        colours = {colour for _text, colour in spans}
        self.assertIn(ClangToken.KEYWORD_COLOR, colours)
        self.assertIn(ClangToken.DEFAULT_COLOR, colours)
        # The keyword colour only ever comes from a real token.
        keywords = [
            text
            for text, colour in spans
            if colour == ClangToken.KEYWORD_COLOR
        ]
        self.assertTrue(keywords)
        self.assertTrue(all(text.strip() for text in keywords))

    def test_line_text_ignores_the_indent_of_the_previous_line(self) -> None:
        document, _analyzer, _function = self._document()
        line = next(line for line in document.lines if line.get_indent() == 2)
        text = ClangPrettyPrinter.get_text(line)
        self.assertTrue(text.startswith(INDENT_STRING * 2))
        self.assertEqual(text, line.get_indent_string() + "".join(
            token.get_text() or "" for token in line.get_all_tokens()
        ))

    def test_a_transformer_is_applied_to_symbol_tokens_only(self) -> None:
        document, _analyzer, _function = self._document()
        printer = ClangPrettyPrinter(
            document.root, transformer=lambda text: text.upper()
        )
        rendered = printer.print().code
        # Names of functions, variables and types are cleaned ...
        self.assertIn("ENTRY(", rendered)
        self.assertIn("UINT64_T", rendered)
        # ... punctuation and constants are not.
        self.assertIn("(", rendered)
        self.assertIn("0x2a", rendered)
        self.assertNotIn("0X2A", rendered)

    def test_token_iterator_runs_in_both_directions(self) -> None:
        document, _analyzer, _function = self._document()
        forward = list(document.root.token_iterator(True))
        backward = list(document.root.token_iterator(False))
        self.assertEqual(forward, list(reversed(backward)))
        self.assertTrue(forward)


class ClangDocumentApiTests(ClangDocumentTestBase):
    def test_the_analyzer_returns_a_document_for_both_formats(self) -> None:
        analyzer, function = self._analyze(CALL_SAMPLE)
        unpacked = analyzer.clang_document(function)
        packed = analyzer.clang_document(function, packed=True)
        self.assertEqual(unpacked.text(), packed.text())
        self.assertIs(unpacked.function, function)
        self.assertIsInstance(unpacked.function_group, ClangFunction)
        self.assertEqual(unpacked.signature(), packed.signature())

    def test_document_options_reach_the_printer(self) -> None:
        analyzer, function = self._analyze(LOOP_SAMPLE)
        loops = analyzer.clang_document(function)
        self.assertIn("for (", loops.text())
        whiles = analyzer.clang_document(function, for_loops=False)
        self.assertIn("while (", whiles.text())
        self.assertNotIn("for (", whiles.text())
        wide = analyzer.clang_document(function, max_line_size=400)
        self.assertLess(len(wide.lines), len(loops.lines) + 3)

    def test_a_resolver_is_handed_every_reference(self) -> None:
        analyzer, function = self._analyze(CALL_SAMPLE)
        resolver = _Recording()
        document = analyzer.clang_document(function, resolver=resolver)
        self.assertTrue(resolver.refs)
        self.assertTrue(resolver.op_refs)
        variables = [
            token
            for token in document.root.token_iterator(True)
            if isinstance(token, ClangVariableToken)
        ]
        self.assertTrue(variables)
        resolved = [token for token in variables if token.varnode is not None]
        self.assertTrue(resolved)
        self.assertTrue(resolved[0].varnode.startswith("varnode#"))
        self.assertTrue(
            all(token.varnode is None for token in variables if token.varref is None)
        )

    def test_str_of_a_group_joins_names_with_a_space(self) -> None:
        group = ClangTokenGroup(None)
        group.add_token_group(ClangSyntaxToken(group, "uint64_t"))
        group.add_token_group(ClangSyntaxToken(group, "entry"))
        self.assertEqual(str(group), "uint64_t entry")
        group.add_token_group(ClangSyntaxToken(group, "("))
        group.add_token_group(ClangSyntaxToken(group, ")"))
        self.assertEqual(str(group), "uint64_t entry()")

    def test_an_empty_line_renders_as_its_indent(self) -> None:
        line = ClangLine(1, 4)
        self.assertEqual(line.get_all_tokens(), [])
        self.assertEqual(ClangPrettyPrinter.get_text(line), INDENT_STRING * 4)
        self.assertEqual(line.get_num_tokens(), 0)


def _flatten(node):
    out = []
    node.flatten(out)
    return out


def _walk(node):
    """Every node of the tree, groups included (depth first)."""
    yield node
    for index in range(node.num_children()):
        child = node.child(index)
        if child is not None:
            yield from _walk(child)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
