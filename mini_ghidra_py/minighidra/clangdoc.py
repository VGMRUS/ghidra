"""Port of Ghidra's *Clang* document model (``ghidra.app.decompiler.Clang*``).

The printing engine does not write plain text when markup is on: it writes the
``<clang_document>`` vocabulary of :mod:`minighidra.emit` through one of the
codecs in :mod:`minighidra.marshal`.  Ghidra's GUI consumes that stream by
rebuilding a tree of ``ClangNode``/``ClangToken`` objects (``ClangMarkup
.buildClangTree``), splitting it into ``ClangLine`` objects at every
``ClangBreak`` (``DecompilerUtils.toLines``) and rendering each token with the
syntax-highlight colour carried by the markup (``PrettyPrinter``).

This module ports that consumer side so a front-end can show *exactly* what the
decompiler emitted, with the original colours, instead of re-deriving
highlighting from the text:

* :func:`build_clang_tree` decodes XML **or** packed markup into the tree.
* :func:`to_lines` flattens the tree and splits it into lines at the breaks.
* :class:`ClangPrettyPrinter` renders lines back to text and recovers the
  function signature from the ``<funcproto>`` group.

Deviations from the Java code
----------------------------
* There is no ``PcodeFactory``: markup references (``varref``/``opref``/
  ``symref``/``blockref``) are resolved through an optional
  :class:`ClangReferenceResolver` and otherwise kept as plain integers on the
  token, so nothing is lost when no resolver is supplied.
* Data types are not looked up in a ``DataTypeManager``; ``<type>``/``<field>``
  tokens keep the marshaled type name and id.
* ``ClangVariableDecl`` tolerates a missing ``symref`` (MiniGhidraPy only writes
  the attribute when the symbol has an id) where the Java code would fail.
* ``NameTransformer`` is reduced to a plain callable applied to symbol-like
  tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

from . import marshal as M
from .marshal import DecoderError
from .printdoc import PrintAddress

#: ``PrettyPrinter.INDENT_STRING``: one break indent unit is one space.
INDENT_STRING = " "

__all__ = [
    "ClangBitFieldToken",
    "ClangBlock",
    "ClangBreak",
    "ClangCaseToken",
    "ClangCommentToken",
    "ClangDocument",
    "ClangFieldToken",
    "ClangFuncNameToken",
    "ClangFuncProto",
    "ClangFunction",
    "ClangLabelToken",
    "ClangLine",
    "ClangNode",
    "ClangOpToken",
    "ClangPrettyPrinter",
    "ClangReferenceResolver",
    "ClangReturnType",
    "ClangStatement",
    "ClangSyntaxToken",
    "ClangToken",
    "ClangTokenGroup",
    "ClangTypeToken",
    "ClangVariableDecl",
    "ClangVariableToken",
    "DecompiledFunction",
    "INDENT_STRING",
    "build_clang_tree",
    "iter_tokens",
    "to_lines",
]


class ClangReferenceResolver:
    """Stand-in for Ghidra's ``PcodeFactory`` when decoding markup references."""

    def get_ref(self, refid: int) -> Any:
        """``PcodeFactory.getRef``: the varnode a ``varref`` points at."""
        return None

    def get_op_ref(self, refid: int) -> Any:
        """``PcodeFactory.getOpRef``: the p-code op an ``opref`` points at."""
        return None

    def get_symbol(self, symref: int) -> Any:
        """``PcodeFactory.getSymbol``: the symbol a ``symref`` points at."""
        return None

    def get_block(self, blockref: int) -> Any:
        """The structured block a ``blockref`` points at."""
        return None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class ClangNode:
    """``ClangNode``: anything that can appear in the token tree."""

    def __init__(self, parent: Optional["ClangNode"] = None) -> None:
        self.parent = parent

    def parent_node(self) -> Optional["ClangNode"]:
        return self.parent

    def min_address(self) -> Optional[PrintAddress]:
        return None

    def max_address(self) -> Optional[PrintAddress]:
        return None

    def num_children(self) -> int:
        return 0

    def child(self, index: int) -> Optional["ClangNode"]:
        return None

    def get_clang_function(self) -> Optional["ClangFunction"]:
        if self.parent is not None:
            return self.parent.get_clang_function()
        return None

    def flatten(self, nodes: list["ClangNode"]) -> None:
        nodes.append(self)

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        """Read this node from the current position of ``decoder``."""

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        return ""


def _is_letter_digit_or_underscore(char: str) -> bool:
    return char.isalnum() or char == "_"


class ClangTokenGroup(ClangNode):
    """``ClangTokenGroup``: a sequence of tokens forming a source construct."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.tokgroup: list[ClangNode] = []
        self.minaddress: Optional[PrintAddress] = None
        self.maxaddress: Optional[PrintAddress] = None

    def add_token_group(self, obj: ClangNode) -> None:
        minaddr = obj.min_address()
        maxaddr = obj.max_address()
        if minaddr is not None:
            if self.minaddress is None or minaddr.offset < self.minaddress.offset:
                self.minaddress = minaddr
        if maxaddr is not None:
            if self.maxaddress is None or maxaddr.offset > self.maxaddress.offset:
                self.maxaddress = maxaddr
        self.tokgroup.append(obj)

    def min_address(self) -> Optional[PrintAddress]:
        return self.minaddress

    def max_address(self) -> Optional[PrintAddress]:
        return self.maxaddress

    def num_children(self) -> int:
        return len(self.tokgroup)

    def child(self, index: int) -> Optional[ClangNode]:
        if 0 <= index < len(self.tokgroup):
            return self.tokgroup[index]
        return None

    def flatten(self, nodes: list[ClangNode]) -> None:
        for element in self.tokgroup:
            element.flatten(nodes)

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        """``ClangTokenGroup.decode``: dispatch on every child element."""
        while True:
            elem = decoder.open_element()
            if elem == 0:
                break
            if elem == M.ELEM_RETURN_TYPE.id:
                child: ClangNode = ClangReturnType(self)
            elif elem == M.ELEM_VARDECL.id:
                child = ClangVariableDecl(self)
            elif elem == M.ELEM_STATEMENT.id:
                child = ClangStatement(self)
            elif elem == M.ELEM_FUNCPROTO.id:
                child = ClangFuncProto(self)
            elif elem == M.ELEM_BLOCK.id:
                # The Java code keeps a plain ClangTokenGroup here; this port
                # also records the blockref so a front-end can link the group
                # back to the structured block it came from.
                child = ClangBlock(self)
            elif elem == M.ELEM_FUNCTION.id:
                # buildClangTree expects <function> to be the root element; the
                # printer wraps it in <clang_document>, so a function group can
                # also show up as a child.
                child = ClangFunction(self)
            else:
                token = ClangToken.build_token(elem, self, decoder, resolver)
                self.add_token_group(token)
                decoder.close_element(elem)
                continue
            child.decode(decoder, resolver)
            self.add_token_group(child)
            decoder.close_element(elem)

    def token_iterator(self, forward: bool = True) -> Iterator["ClangToken"]:
        """``ClangTokenGroup.tokenIterator``: tokens in (or against) display order."""
        nodes: list[ClangNode] = []
        self.flatten(nodes)
        tokens = [node for node in nodes if isinstance(node, ClangToken)]
        if not forward:
            tokens.reverse()
        return iter(tokens)

    def __iter__(self) -> Iterator[ClangNode]:
        return iter(self.tokgroup)

    def __str__(self) -> str:
        last: Optional[str] = None
        out: list[str] = []
        for node in self.tokgroup:
            text = str(node)
            if not text:
                continue
            if last is not None and _is_letter_digit_or_underscore(
                text[0]
            ) and _is_letter_digit_or_underscore(last[-1]):
                # Avoid concatenating two names together.
                out.append(" ")
            last = text
            out.append(text)
        return "".join(out)


class ClangBlock(ClangTokenGroup):
    """A ``<block>`` group; ``blockref`` is the structured block index."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.blockref: Optional[int] = None
        self.block: Any = None

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        while True:
            attrib = decoder.get_next_attribute_id()
            if attrib == 0:
                break
            if attrib == M.ATTRIB_BLOCKREF.id:
                self.blockref = decoder.read_signed_integer()
                if resolver is not None:
                    self.block = resolver.get_block(self.blockref)
        decoder.rewind_attributes()
        super().decode(decoder, resolver)


class ClangFunction(ClangTokenGroup):
    """``ClangFunction``: a grouping of tokens representing a whole function."""

    def get_clang_function(self) -> "ClangFunction":
        return self


class ClangStatement(ClangTokenGroup):
    """``ClangStatement``: tokens of a single statement."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.opref: Optional[int] = None
        self.op: Any = None

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        while True:
            attrib = decoder.get_next_attribute_id()
            if attrib == 0:
                break
            if attrib == M.ATTRIB_OPREF.id:
                self.opref = decoder.read_unsigned_integer()
                if resolver is not None:
                    self.op = resolver.get_op_ref(self.opref)
                break
        super().decode(decoder, resolver)

    def min_address(self) -> Optional[PrintAddress]:
        return None


class ClangReturnType(ClangTokenGroup):
    """``ClangReturnType``: the declared return type of the function."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.varref: Optional[int] = None
        self.varnode: Any = None

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        while True:
            attrib = decoder.get_next_attribute_id()
            if attrib == 0:
                break
            if attrib == M.ATTRIB_VARREF.id:
                self.varref = decoder.read_unsigned_integer()
                if resolver is not None:
                    self.varnode = resolver.get_ref(self.varref)
                break
        super().decode(decoder, resolver)


class ClangVariableDecl(ClangTokenGroup):
    """``ClangVariableDecl``: one local variable declaration."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.symref: Optional[int] = None
        self.symbol: Any = None

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        try:
            self.symref = decoder.read_unsigned_integer(M.ATTRIB_SYMREF)
        except DecoderError:
            # MiniGhidraPy omits symref for symbols without an id.
            self.symref = None
        if self.symref is not None and resolver is not None:
            self.symbol = resolver.get_symbol(self.symref)
        super().decode(decoder, resolver)


class ClangFuncProto(ClangTokenGroup):
    """``ClangFuncProto``: the function prototype tokens."""


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


class ClangToken(ClangNode):
    """``ClangToken``: one display token with a syntax-highlight colour."""

    # Constants must match Emit::syntax_highlight (ClangToken in the Java code).
    KEYWORD_COLOR = 0
    COMMENT_COLOR = 1
    TYPE_COLOR = 2
    FUNCTION_COLOR = 3
    VARIABLE_COLOR = 4
    CONST_COLOR = 5
    PARAMETER_COLOR = 6
    GLOBAL_COLOR = 7
    DEFAULT_COLOR = 8
    ERROR_COLOR = 9
    SPECIAL_COLOR = 10
    MAX_COLOR = 11

    def __init__(
        self,
        parent: Optional[ClangNode] = None,
        text: Optional[str] = None,
        color: int = DEFAULT_COLOR,
    ) -> None:
        super().__init__(parent)
        self.text = text
        self.syntax_type = color
        self.lineparent: Optional["ClangLine"] = None
        self.matching_token = False

    def get_line_parent(self) -> Optional["ClangLine"]:
        return self.lineparent

    def set_line_parent(self, line: "ClangLine") -> None:
        self.lineparent = line

    def num_children(self) -> int:
        return 0

    def child(self, index: int) -> Optional[ClangNode]:
        return None

    def set_matching_token(self, value: bool) -> None:
        self.matching_token = bool(value)

    def is_matching_token(self) -> bool:
        return self.matching_token

    def is_variable_ref(self) -> bool:
        return False

    def get_syntax_type(self) -> int:
        return self.syntax_type

    def set_syntax_type(self, value: int) -> None:
        self.syntax_type = value

    def get_text(self) -> Optional[str]:
        return self.text

    def set_text(self, text: Optional[str]) -> None:
        self.text = text

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        """``ClangToken.decode``: read the colour attribute and the content."""
        self.syntax_type = self.DEFAULT_COLOR
        while True:
            attrib = decoder.get_next_attribute_id()
            if attrib == 0:
                break
            if attrib == M.ATTRIB_COLOR.id:
                self.syntax_type = decoder.read_unsigned_integer()
                break
        try:
            self.text = decoder.read_string(M.ATTRIB_CONTENT)
        except DecoderError:
            self.text = None
        if self.syntax_type < 0 or self.syntax_type >= self.MAX_COLOR:
            self.syntax_type = self.DEFAULT_COLOR

    def flatten(self, nodes: list[ClangNode]) -> None:
        nodes.append(self)

    @staticmethod
    def build_token(
        elem: int,
        parent: ClangNode,
        decoder: Any,
        resolver: Optional[ClangReferenceResolver] = None,
    ) -> "ClangToken":
        """``ClangToken.buildToken``: allocate the right token for an element."""
        token: ClangToken
        if elem == M.ELEM_VARIABLE.id:
            token = ClangVariableToken(parent)
        elif elem == M.ELEM_OP.id:
            token = ClangOpToken(parent)
        elif elem == M.ELEM_SYNTAX.id:
            token = ClangSyntaxToken(parent)
        elif elem == M.ELEM_BREAK.id:
            token = ClangBreak(parent)
        elif elem == M.ELEM_FUNCNAME.id:
            token = ClangFuncNameToken(parent)
        elif elem == M.ELEM_TYPE.id:
            token = ClangTypeToken(parent)
        elif elem == M.ELEM_COMMENT.id:
            token = ClangCommentToken(parent)
        elif elem == M.ELEM_LABEL.id:
            token = ClangLabelToken(parent)
        elif elem == M.ELEM_FIELD.id:
            token = ClangFieldToken(parent)
        elif elem == M.ELEM_BITFIELD.id:
            token = ClangBitFieldToken(parent)
        elif elem == M.ELEM_VALUE.id:
            token = ClangCaseToken(parent)
        else:
            raise DecoderError("Expecting token element")
        token.decode(decoder, resolver)
        return token

    @staticmethod
    def build_spacer(
        parent: Optional[ClangNode], indent: int, indent_str: str = INDENT_STRING
    ) -> "ClangToken":
        """``ClangToken.buildSpacer``: a syntax token holding a line's indent."""
        return ClangSyntaxToken(parent, indent_str * indent)

    def __str__(self) -> str:
        return self.text or ""


class ClangSyntaxToken(ClangToken):
    """``ClangSyntaxToken``: plain syntax, optionally a paren open/close."""

    def __init__(
        self,
        parent: Optional[ClangNode] = None,
        text: Optional[str] = None,
        color: int = ClangToken.DEFAULT_COLOR,
    ) -> None:
        super().__init__(parent, text, color)
        self.open = -1
        self.close = -1

    def is_variable_ref(self) -> bool:
        return isinstance(self.parent, ClangVariableDecl)

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        while True:
            attrib = decoder.get_next_attribute_id()
            if attrib == 0:
                break
            if attrib == M.ATTRIB_OPEN.id:
                self.open = decoder.read_signed_integer()
            elif attrib == M.ATTRIB_CLOSE.id:
                self.close = decoder.read_signed_integer()
        decoder.rewind_attributes()
        super().decode(decoder, resolver)

    def get_open(self) -> int:
        return self.open

    def get_close(self) -> int:
        return self.close


class ClangVariableToken(ClangToken):
    """``ClangVariableToken``: a variable/constant in the data-flow."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.varref: Optional[int] = None
        self.opref: Optional[int] = None
        self.varnode: Any = None
        self.op: Any = None

    def is_variable_ref(self) -> bool:
        return True

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        while True:
            attrib = decoder.get_next_attribute_id()
            if attrib == 0:
                break
            if attrib == M.ATTRIB_VARREF.id:
                self.varref = decoder.read_unsigned_integer()
                if resolver is not None:
                    self.varnode = resolver.get_ref(self.varref)
            elif attrib == M.ATTRIB_OPREF.id:
                self.opref = decoder.read_unsigned_integer()
                if resolver is not None:
                    self.op = resolver.get_op_ref(self.opref)
        decoder.rewind_attributes()
        super().decode(decoder, resolver)


class ClangOpToken(ClangToken):
    """``ClangOpToken``: an operator token tied to a p-code op."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.opref: Optional[int] = None
        self.op: Any = None

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        while True:
            attrib = decoder.get_next_attribute_id()
            if attrib == 0:
                break
            if attrib == M.ATTRIB_OPREF.id:
                self.opref = decoder.read_unsigned_integer()
                if resolver is not None:
                    self.op = resolver.get_op_ref(self.opref)
        decoder.rewind_attributes()
        super().decode(decoder, resolver)


class ClangTypeToken(ClangToken):
    """``ClangTypeToken``: a data-type name."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.type_id: Optional[int] = None

    def is_variable_ref(self) -> bool:
        return isinstance(self.parent, ClangVariableDecl)

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        while True:
            attrib = decoder.get_next_attribute_id()
            if attrib == 0:
                break
            if attrib == M.ATTRIB_ID.id:
                self.type_id = decoder.read_unsigned_integer()
        decoder.rewind_attributes()
        super().decode(decoder, resolver)


class ClangFuncNameToken(ClangToken):
    """``ClangFuncNameToken``: the name of a function being called/declared."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.opref: Optional[int] = None
        self.op: Any = None

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        while True:
            attrib = decoder.get_next_attribute_id()
            if attrib == 0:
                break
            if attrib == M.ATTRIB_OPREF.id:
                self.opref = decoder.read_unsigned_integer()
                if resolver is not None:
                    self.op = resolver.get_op_ref(self.opref)
                break
        decoder.rewind_attributes()
        super().decode(decoder, resolver)


class ClangCommentToken(ClangToken):
    """``ClangCommentToken``: a piece of a comment attached to an address."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.srcaddr: Optional[PrintAddress] = None

    def min_address(self) -> Optional[PrintAddress]:
        return self.srcaddr

    def max_address(self) -> Optional[PrintAddress]:
        return self.srcaddr

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        space = ""
        offset = 0
        try:
            space = decoder.read_space(M.ATTRIB_SPACE)
            offset = decoder.read_unsigned_integer(M.ATTRIB_OFF)
            self.srcaddr = PrintAddress(space, offset)
        except DecoderError:
            self.srcaddr = None
        super().decode(decoder, resolver)

    @staticmethod
    def derive(source: "ClangCommentToken", text: str) -> "ClangCommentToken":
        """``ClangCommentToken.derive``: clone a token with new text."""
        token = ClangCommentToken(source.parent)
        token.set_text(text)
        token.set_line_parent(source.get_line_parent())  # type: ignore[arg-type]
        token.set_syntax_type(source.get_syntax_type())
        token.srcaddr = source.srcaddr
        return token


class ClangLabelToken(ClangToken):
    """``ClangLabelToken``: a goto label tied to a block address."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.blockaddr: Optional[PrintAddress] = None

    def min_address(self) -> Optional[PrintAddress]:
        return self.blockaddr

    def max_address(self) -> Optional[PrintAddress]:
        return self.blockaddr

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        try:
            space = decoder.read_space(M.ATTRIB_SPACE)
            offset = decoder.read_unsigned_integer(M.ATTRIB_OFF)
            self.blockaddr = PrintAddress(space, offset)
        except DecoderError:
            self.blockaddr = None
        super().decode(decoder, resolver)


class ClangFieldToken(ClangToken):
    """``ClangFieldToken``: a structure/union field name."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.datatype_name: Optional[str] = None
        self.type_id: int = 0
        self.offset: int = 0
        self.opref: Optional[int] = None

    def is_variable_ref(self) -> bool:
        return True

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        while True:
            attrib = decoder.get_next_attribute_id()
            if attrib == 0:
                break
            if attrib == M.ATTRIB_NAME.id:
                self.datatype_name = decoder.read_string()
            elif attrib == M.ATTRIB_ID.id:
                self.type_id = decoder.read_unsigned_integer()
            elif attrib == M.ATTRIB_OFF.id:
                self.offset = decoder.read_signed_integer()
            elif attrib == M.ATTRIB_OPREF.id:
                self.opref = decoder.read_unsigned_integer()
        decoder.rewind_attributes()
        super().decode(decoder, resolver)


class ClangBitFieldToken(ClangFieldToken):
    """``ClangBitFieldToken``: a bit-field within a structure."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.ident = -1

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        self.ident = -1
        while True:
            attrib = decoder.get_next_attribute_id()
            if attrib == 0:
                break
            if attrib == M.ATTRIB_NAME.id:
                self.datatype_name = decoder.read_string()
            elif attrib == M.ATTRIB_ID.id:
                self.type_id = decoder.read_unsigned_integer()
            elif attrib == M.ATTRIB_OPREF.id:
                self.opref = decoder.read_unsigned_integer()
            elif attrib == M.ATTRIB_OFF.id:
                self.ident = decoder.read_signed_integer()
        decoder.rewind_attributes()
        ClangToken.decode(self, decoder, resolver)


class ClangCaseToken(ClangToken):
    """``ClangCaseToken``: a ``case`` value of a switch body."""

    def __init__(self, parent: Optional[ClangNode] = None) -> None:
        super().__init__(parent)
        self.value: int = 0
        self.opref: Optional[int] = None
        self.op: Any = None

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        while True:
            attrib = decoder.get_next_attribute_id()
            if attrib == 0:
                break
            if attrib == M.ATTRIB_OPREF.id:
                self.opref = decoder.read_unsigned_integer()
                if resolver is not None:
                    self.op = resolver.get_op_ref(self.opref)
            elif attrib == M.ATTRIB_OFF.id:
                self.value = decoder.read_unsigned_integer()
        decoder.rewind_attributes()
        super().decode(decoder, resolver)


class ClangBreak(ClangToken):
    """``ClangBreak``: a line break plus the indent of the next line."""

    def __init__(self, parent: Optional[ClangNode] = None, indent: int = 0) -> None:
        super().__init__(parent)
        self.indent = indent

    def get_indent(self) -> int:
        return self.indent

    def decode(self, decoder: Any, resolver: Optional[ClangReferenceResolver]) -> None:
        self.indent = decoder.read_signed_integer(M.ATTRIB_INDENT)
        self.set_text("")


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------


class ClangLine:
    """``ClangLine``: one line of C code and the tokens it owns."""

    def __init__(self, line_number: int, indent: int) -> None:
        self.indent_level = indent
        self.tokens: list[ClangToken] = []
        self.line_number = line_number

    def get_indent_string(self) -> str:
        return INDENT_STRING * self.indent_level

    def get_indent(self) -> int:
        return self.indent_level

    def add_token(self, token: ClangToken) -> None:
        self.tokens.append(token)
        token.set_line_parent(self)

    def get_all_tokens(self) -> list[ClangToken]:
        return self.tokens

    def get_num_tokens(self) -> int:
        return len(self.tokens)

    def get_line_number(self) -> int:
        return self.line_number

    def get_token(self, index: int) -> ClangToken:
        return self.tokens[index]

    def index_of_token(self, token: ClangToken) -> int:
        return self.tokens.index(token)

    def __str__(self) -> str:
        return f"{self.line_number}: " + "".join(token.get_text() or "" for token in self.tokens)


def _consume_comment_tokens(
    alltoks: list[ClangNode],
    index: int,
    first: ClangCommentToken,
    current: ClangLine,
) -> int:
    """``DecompilerUtils.consumeCommentTokens``: glue a comment back together.

    The emitter splits comments into word tokens with blank syntax tokens in
    between (that is how ``Emit::setCommentFill`` works); a front-end wants the
    whole comment as one token again.
    """
    parts = [first.get_text() or ""]
    index += 1
    while index < len(alltoks):
        token = alltoks[index]
        if isinstance(token, ClangCommentToken):
            if first.get_syntax_type() != token.get_syntax_type():
                break
            parts.append(token.get_text() or "")
        elif isinstance(token, ClangSyntaxToken):
            # Comments can have blank space tokens embedded in them.
            value = token.get_text() or ""
            if not value.strip():
                parts.append(value)
            else:
                break
        else:
            break
        index += 1
    comment = ClangCommentToken.derive(first, "".join(parts))
    comment.set_line_parent(current)
    current.add_token(comment)
    return index


def to_lines(group: ClangTokenGroup) -> list[ClangLine]:
    """``DecompilerUtils.toLines``: split a token tree into lines at the breaks."""
    alltoks: list[ClangNode] = []
    group.flatten(alltoks)
    if not alltoks:
        return []

    index = 0
    line_number = 1
    if isinstance(alltoks[0], ClangBreak):
        current = ClangLine(line_number, alltoks[0].get_indent())
        line_number += 1
        index = 1
    else:
        current = ClangLine(line_number, 0)
        line_number += 1

    lines: list[ClangLine] = []
    while index < len(alltoks):
        node = alltoks[index]
        if isinstance(node, ClangBreak):
            lines.append(current)
            current = ClangLine(line_number, node.get_indent())
            line_number += 1
        elif isinstance(node, ClangCommentToken):
            index = _consume_comment_tokens(alltoks, index, node, current)
            index -= 1
        elif isinstance(node, ClangToken):
            node.set_line_parent(current)
            current.add_token(node)
        index += 1
    lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------


def build_clang_tree(
    markup: Any,
    packed: Optional[bool] = None,
    resolver: Optional[ClangReferenceResolver] = None,
) -> ClangTokenGroup:
    """``ClangMarkup.buildClangTree``: decode markup into the token tree.

    ``markup`` may be the XML text, the packed bytes, or an already built
    :class:`~minighidra.marshal.Decoder`.  ``packed`` is inferred from the type
    when it is not given.
    """
    if isinstance(markup, M.Decoder):
        decoder = markup
    else:
        if packed is None:
            packed = isinstance(markup, (bytes, bytearray))
        decoder = M.PackedDecode() if packed else M.XmlDecode()
        decoder.ingest_stream(markup)
    elem = decoder.open_element()
    if elem == M.ELEM_FUNCTION.id:
        docroot: ClangTokenGroup = ClangFunction(None)
    else:
        docroot = ClangTokenGroup(None)
    docroot.decode(decoder, resolver)
    decoder.close_element(elem)
    return docroot


@dataclass(frozen=True)
class DecompiledFunction:
    """``DecompiledFunction``: the printed C code plus its signature."""

    signature: Optional[str]
    code: str


class ClangPrettyPrinter:
    """``PrettyPrinter``: turn a token tree back into readable C code."""

    INDENT_STRING = INDENT_STRING

    def __init__(
        self,
        tokgroup: ClangTokenGroup,
        function: Any = None,
        transformer: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.function = function
        self.tokgroup = tokgroup
        self.transformer = transformer
        self.lines: list[ClangLine] = []
        self._flatten_lines()
        self._pad_empty_lines()

    def get_function(self) -> Any:
        return self.function

    def get_lines(self) -> list[ClangLine]:
        return self.lines

    def _flatten_lines(self) -> None:
        self.lines = to_lines(self.tokgroup)

    def _pad_empty_lines(self) -> None:
        for line in self.lines:
            tokens = line.get_all_tokens()
            if not tokens:
                spacer = ClangToken.build_spacer(None, line.get_indent())
                spacer.set_line_parent(line)
                tokens.insert(0, spacer)

    def print(self) -> DecompiledFunction:
        """``PrettyPrinter.print``: the whole document as text."""
        out = []
        for line in self.lines:
            out.append(self.get_text(line, self.transformer))
        return DecompiledFunction(self.find_signature(), "\n".join(out) + "\n")

    @staticmethod
    def get_text(
        line: ClangLine, transformer: Optional[Callable[[str], str]] = None
    ) -> str:
        """``PrettyPrinter.getText``: one line, with the indent and the tokens."""
        out = [line.get_indent_string()]
        for token in line.get_all_tokens():
            clean = isinstance(
                token,
                (
                    ClangFuncNameToken,
                    ClangVariableToken,
                    ClangTypeToken,
                    ClangFieldToken,
                    ClangLabelToken,
                ),
            )
            # Do not clean constant tokens.
            if clean and token.get_syntax_type() == ClangToken.CONST_COLOR:
                clean = False
            text = token.get_text() or ""
            if clean and transformer is not None:
                text = transformer(text)
            out.append(text)
        return "".join(out)

    def find_signature(self) -> Optional[str]:
        """``PrettyPrinter.findSignature``: text of the ``<funcproto>`` group."""
        group = self.tokgroup
        if not isinstance(group, ClangFunction):
            # The printer wraps the function in <clang_document>; Ghidra hands
            # the <function> group to the pretty printer directly.
            for index in range(group.num_children()):
                node = group.child(index)
                if isinstance(node, ClangFunction):
                    group = node
                    break
        for index in range(group.num_children()):
            node = group.child(index)
            if isinstance(node, ClangFuncProto):
                return str(node) + ";"
        return None


def iter_tokens(group: ClangTokenGroup) -> Iterable[ClangToken]:
    """Every token of a document, in display order."""
    return group.token_iterator(True)


@dataclass
class ClangDocument:
    """A decoded document plus the line layout a front-end renders.

    Bundles :func:`build_clang_tree` and :class:`ClangPrettyPrinter`, which is
    what ``DecompileResults`` does for Ghidra's GUI.
    """

    root: ClangTokenGroup
    printer: ClangPrettyPrinter = field(init=False)
    function: Any = None
    resolver: Optional[ClangReferenceResolver] = None

    def __post_init__(self) -> None:
        self.printer = ClangPrettyPrinter(self.root, self.function)

    @classmethod
    def parse(
        cls,
        markup: Any,
        packed: Optional[bool] = None,
        function: Any = None,
        resolver: Optional[ClangReferenceResolver] = None,
    ) -> "ClangDocument":
        root = build_clang_tree(markup, packed=packed, resolver=resolver)
        return cls(root=root, function=function, resolver=resolver)

    @property
    def lines(self) -> list[ClangLine]:
        return self.printer.get_lines()

    def spans(self) -> list[tuple[str, int]]:
        """ ``(text, syntax colour)`` pairs covering the whole document.

        The indent of every line is emitted as its own pair so a front-end can
        insert the document in one pass, tagging each piece with the colour the
        printer chose (``Emit::syntax_highlight``).  Joining the texts reproduces
        :meth:`text` exactly.
        """
        out: list[tuple[str, int]] = []
        for line in self.lines:
            out.append((line.get_indent_string(), ClangToken.DEFAULT_COLOR))
            for token in line.get_all_tokens():
                out.append((token.get_text() or "", token.get_syntax_type()))
            out.append(("\n", ClangToken.DEFAULT_COLOR))
        return out

    @property
    def function_group(self) -> Optional[ClangFunction]:
        """The ``<function>`` group (``DecompileResults.getCCodeMarkup``)."""
        root = self.root
        if isinstance(root, ClangFunction):
            return root
        for index in range(root.num_children()):
            node = root.child(index)
            if isinstance(node, ClangFunction):
                return node
        return None

    def text(self) -> str:
        """The document rendered back to plain C text."""
        return self.printer.print().code

    def signature(self) -> Optional[str]:
        return self.printer.find_signature()
