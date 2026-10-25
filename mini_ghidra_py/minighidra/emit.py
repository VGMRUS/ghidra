"""Port of Ghidra's high-level emitter stack (``prettyprint.hh``/``prettyprint.cc``).

Ghidra separates *what* is printed from *how* it is printed.  ``PrintC`` (and any
other ``PrintLanguage``) only produces a stream of tagged tokens; this module
reproduces the receiving end of that stream:

* :class:`Emit` -- the abstract token interface (``beginDocument``/``tagOp``/...).
* :class:`EmitMarkup` -- the markup emitter.  Every token is wrapped in the same
  XML elements and attributes the C++ engine writes through ``XmlEncode``
  (``<clang_document>``, ``<function>``, ``<statement>``, ``<syntax>``, ...).
* :class:`EmitNoMarkup` -- the trivial emitter that dumps tokens straight out.
* :class:`EmitPrettyPrint` -- Derek C. Oppen's pretty printing algorithm, with
  the token queue (:class:`TokenSplit`), the circular buffers
  (:class:`CircularQueue`), ``scan``/``advanceleft``/``overflow`` and the
  ``checkstart``/``checkstring``/``checkend``/``checkbreak`` whitespace rules,
  wrapping one of the low-level emitters above.

The C++ classes are parameterised on ``Funcdata``/``FlowBlock``/``Varnode``/
``PcodeOp``/``Symbol``/``Datatype``/``AddrSpace`` pointers.  MiniGhidraPy's
decompiler backend still runs on its own expression AST, so those pointers are
replaced by the equivalent Python objects described in ``printdoc.py``.  Markup
references are read through the attribute names used by the C++ accessors
(``create_index`` for ``Varnode::getCreateIndex``, ``op_time`` for
``PcodeOp::getTime``, ``symbol_id`` for ``Symbol::getId``, ``index`` for
``FlowBlock::getIndex``, ``unsized_id`` for ``Datatype::getUnsizedId``); when an
object does not carry the attribute the reference is simply omitted, exactly as
the C++ code omits it for a null pointer.

Only the unpacked XML encoder is ported.  ``PackedEncode`` (Ghidra's binary
marshaling format) is not reproduced; :meth:`Emit.set_packed_output` rejects it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional, TextIO


class EmitError(RuntimeError):
    """Raised for emitter misuse, mirroring C++ ``LowlevelError``."""


@dataclass(frozen=True, slots=True)
class ElementId:
    """Marshaling element identifier (``ElementId`` in ``marshal.hh``)."""

    name: str
    id: int


@dataclass(frozen=True, slots=True)
class AttributeId:
    """Marshaling attribute identifier (``AttributeId`` in ``marshal.hh``)."""

    name: str
    id: int


# Attribute ids, matching the numeric values assigned in marshal.cc/prettyprint.cc.
ATTRIB_CONTENT = AttributeId("XMLcontent", 1)
ATTRIB_ID = AttributeId("id", 9)
ATTRIB_NAME = AttributeId("name", 14)
ATTRIB_SPACE = AttributeId("space", 20)
ATTRIB_BLOCKREF = AttributeId("blockref", 35)
ATTRIB_CLOSE = AttributeId("close", 36)
ATTRIB_COLOR = AttributeId("color", 37)
ATTRIB_INDENT = AttributeId("indent", 38)
ATTRIB_OFF = AttributeId("off", 39)
ATTRIB_OPEN = AttributeId("open", 40)
ATTRIB_OPREF = AttributeId("opref", 41)
ATTRIB_VARREF = AttributeId("varref", 42)
ATTRIB_SYMREF = AttributeId("symref", 68)

# Element ids, matching the numeric values assigned across the C++ sources.
ELEM_VALUE = ElementId("value", 9)
ELEM_BREAK = ElementId("break", 17)
ELEM_CLANG_DOCUMENT = ElementId("clang_document", 18)
ELEM_FUNCNAME = ElementId("funcname", 19)
ELEM_FUNCPROTO = ElementId("funcproto", 20)
ELEM_LABEL = ElementId("label", 21)
ELEM_RETURN_TYPE = ElementId("return_type", 22)
ELEM_STATEMENT = ElementId("statement", 23)
ELEM_SYNTAX = ElementId("syntax", 24)
ELEM_VARDECL = ElementId("vardecl", 25)
ELEM_VARIABLE = ElementId("variable", 26)
ELEM_OP = ElementId("op", 27)
ELEM_FIELD = ElementId("field", 49)
ELEM_TYPE = ElementId("type", 60)
ELEM_COMMENT = ElementId("comment", 86)
ELEM_BLOCK = ElementId("block", 103)
ELEM_FUNCTION = ElementId("function", 116)
ELEM_BITFIELD = ElementId("bitfield", 289)


def xml_escape(text: str) -> str:
    """Escape the characters with special XML meaning (``xml_escape`` in xml.cc)."""
    out = []
    for char in text:
        if char < "?":
            if char == "<":
                out.append("&lt;")
                continue
            if char == ">":
                out.append("&gt;")
                continue
            if char == "&":
                out.append("&amp;")
                continue
            if char == '"':
                out.append("&quot;")
                continue
            if char == "'":
                out.append("&apos;")
                continue
        out.append(char)
    return "".join(out)


class XmlEncoder:
    """Unpacked XML encoder, a port of ``XmlEncode`` in ``marshal.cc``."""

    SPACES = "\n                        "
    MAX_SPACES = 24 + 1

    TAG_START = 0
    TAG_CONTENT = 1
    TAG_STOP = 2

    def __init__(self, stream: TextIO, do_format: bool = True) -> None:
        self.stream = stream
        self.depth = 0
        self.tag_status = self.TAG_STOP
        self.do_formatting = do_format

    def _write(self, text: str) -> None:
        self.stream.write(text)

    def _new_line(self) -> None:
        if not self.do_formatting:
            return
        num_spaces = self.depth * 2 + 1
        if num_spaces > self.MAX_SPACES:
            num_spaces = self.MAX_SPACES
        self._write(self.SPACES[:num_spaces])

    def open_element(self, elem_id: ElementId) -> None:
        if self.tag_status == self.TAG_START:
            self._write(">")
        else:
            self.tag_status = self.TAG_START
        self._new_line()
        self._write("<" + elem_id.name)
        self.depth += 1

    def close_element(self, elem_id: ElementId) -> None:
        self.depth -= 1
        if self.tag_status == self.TAG_START:
            self._write("/>")
            self.tag_status = self.TAG_STOP
            return
        if self.tag_status != self.TAG_CONTENT:
            self._new_line()
        else:
            self.tag_status = self.TAG_STOP
        self._write("</" + elem_id.name + ">")

    def _a_v(self, attrib_id: AttributeId, value: str) -> None:
        self._write(" " + attrib_id.name + '="' + xml_escape(value) + '"')

    def _a_v_i(self, attrib_id: AttributeId, value: int) -> None:
        self._write(f' {attrib_id.name}="{value:d}"')

    def _a_v_u(self, attrib_id: AttributeId, value: int) -> None:
        self._write(f' {attrib_id.name}="0x{value:x}"')

    def _a_v_b(self, attrib_id: AttributeId, value: bool) -> None:
        self._write(f' {attrib_id.name}="{"true" if value else "false"}"')

    def _content_prefix(self) -> None:
        if self.tag_status == self.TAG_START:
            self._write(">")

    def write_bool(self, attrib_id: AttributeId, value: bool) -> None:
        if attrib_id == ATTRIB_CONTENT:
            self._content_prefix()
            self._write("true" if value else "false")
            self.tag_status = self.TAG_CONTENT
            return
        self._a_v_b(attrib_id, value)

    def write_signed_integer(self, attrib_id: AttributeId, value: int) -> None:
        if attrib_id == ATTRIB_CONTENT:
            self._content_prefix()
            self._write(f"{value:d}")
            self.tag_status = self.TAG_CONTENT
            return
        self._a_v_i(attrib_id, value)

    def write_unsigned_integer(self, attrib_id: AttributeId, value: int) -> None:
        if attrib_id == ATTRIB_CONTENT:
            self._content_prefix()
            self._write(f"0x{value:x}")
            self.tag_status = self.TAG_CONTENT
            return
        self._a_v_u(attrib_id, value)

    def write_string(self, attrib_id: AttributeId, value: str) -> None:
        if attrib_id == ATTRIB_CONTENT:
            self._content_prefix()
            self._write(xml_escape(value))
            self.tag_status = self.TAG_CONTENT
            return
        self._a_v(attrib_id, value)

    def write_space(self, attrib_id: AttributeId, space: Any) -> None:
        name = getattr(space, "name", space)
        if attrib_id == ATTRIB_CONTENT:
            self._content_prefix()
            self._write(xml_escape(str(name)))
            self.tag_status = self.TAG_CONTENT
            return
        self._a_v(attrib_id, str(name))


class SyntaxHighlight(IntEnum):
    """``Emit::syntax_highlight``; values must match ``ClangToken``."""

    keyword_color = 0
    comment_color = 1
    type_color = 2
    funcname_color = 3
    var_color = 4
    const_color = 5
    param_color = 6
    global_color = 7
    no_color = 8
    error_color = 9
    special_color = 10


class BraceStyle(IntEnum):
    """``Emit::brace_style``."""

    same_line = 0
    next_line = 1
    skip_line = 2


def _reference(obj: Any, attribute: str) -> Optional[int]:
    """Read a marshaling reference id off an AST/document object, if present."""
    if obj is None:
        return None
    value = getattr(obj, attribute, None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


class PendPrint:
    """Base class for cancelable pending print commands (``PendPrint``)."""

    def callback(self, emit: "Emit") -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class Emit(ABC):
    """Interface for emitting the Decompiler's formal output: source code."""

    EMPTY_STRING = ""

    def __init__(self) -> None:
        self.indentlevel = 0
        self.parenlevel = 0
        self.pend_print: Optional[PendPrint] = None
        self.reset_defaults_internal()

    # -- options ---------------------------------------------------------
    def reset_defaults_internal(self) -> None:
        self.indentincrement = 2

    def emit_pending(self) -> None:
        if self.pend_print is not None:
            pending = self.pend_print
            self.pend_print = None
            pending.callback(self)

    # -- document structure ---------------------------------------------
    @abstractmethod
    def begin_document(self) -> int:
        ...

    @abstractmethod
    def end_document(self, id: int) -> None:
        ...

    @abstractmethod
    def begin_function(self, fd: Any) -> int:
        ...

    @abstractmethod
    def end_function(self, id: int) -> None:
        ...

    @abstractmethod
    def begin_block(self, bl: Any) -> int:
        ...

    @abstractmethod
    def end_block(self, id: int) -> None:
        ...

    @abstractmethod
    def tag_line(self, indent: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def begin_return_type(self, vn: Any) -> int:
        ...

    @abstractmethod
    def end_return_type(self, id: int) -> None:
        ...

    @abstractmethod
    def begin_var_decl(self, symbol: Any) -> int:
        ...

    @abstractmethod
    def end_var_decl(self, id: int) -> None:
        ...

    @abstractmethod
    def begin_statement(self, op: Any) -> int:
        ...

    @abstractmethod
    def end_statement(self, id: int) -> None:
        ...

    @abstractmethod
    def begin_func_proto(self) -> int:
        ...

    @abstractmethod
    def end_func_proto(self, id: int) -> None:
        ...

    # -- tokens ----------------------------------------------------------
    @abstractmethod
    def tag_variable(self, name: str, hl: SyntaxHighlight, vn: Any, op: Any) -> None:
        ...

    @abstractmethod
    def tag_op(self, name: str, hl: SyntaxHighlight, op: Any) -> None:
        ...

    @abstractmethod
    def tag_func_name(self, name: str, hl: SyntaxHighlight, fd: Any, op: Any) -> None:
        ...

    @abstractmethod
    def tag_type(self, name: str, hl: SyntaxHighlight, ct: Any) -> None:
        ...

    @abstractmethod
    def tag_field(
        self, name: str, hl: SyntaxHighlight, ct: Any, off: int, op: Any
    ) -> None:
        ...

    @abstractmethod
    def tag_bit_field(
        self, name: str, hl: SyntaxHighlight, ct: Any, id: int, op: Any
    ) -> None:
        ...

    @abstractmethod
    def tag_comment(self, name: str, hl: SyntaxHighlight, spc: Any, off: int) -> None:
        ...

    @abstractmethod
    def tag_label(self, name: str, hl: SyntaxHighlight, spc: Any, off: int) -> None:
        ...

    @abstractmethod
    def tag_case_label(
        self, name: str, hl: SyntaxHighlight, op: Any, value: int
    ) -> None:
        ...

    @abstractmethod
    def print(self, data: str, hl: SyntaxHighlight = SyntaxHighlight.no_color) -> None:
        ...

    @abstractmethod
    def open_paren(self, paren: str, id: int = 0) -> int:
        ...

    @abstractmethod
    def close_paren(self, paren: str, id: int) -> None:
        ...

    def open_group(self) -> int:
        return 0

    def close_group(self, id: int) -> None:
        return None

    def clear(self) -> None:
        self.parenlevel = 0
        self.indentlevel = 0
        self.pend_print = None

    @abstractmethod
    def set_output_stream(self, stream: Optional[TextIO]) -> None:
        ...

    @abstractmethod
    def get_output_stream(self) -> Optional[TextIO]:
        ...

    def set_markup(self, val: bool) -> None:
        return None

    def set_packed_output(self, val: bool) -> None:
        return None

    def spaces(self, num: int, bump: int = 0) -> None:
        space_array = [
            "",
            " ",
            "  ",
            "   ",
            "    ",
            "     ",
            "      ",
            "       ",
            "        ",
            "         ",
            "          ",
        ]
        if num <= 10:
            self.print(space_array[num])
        else:
            self.print(" " * num)

    def start_indent(self) -> int:
        self.indentlevel += self.indentincrement
        return 0

    def stop_indent(self, id: int) -> None:
        self.indentlevel -= self.indentincrement

    def start_comment(self) -> int:
        return 0

    def stop_comment(self, id: int) -> None:
        return None

    def flush(self) -> None:
        return None

    def set_max_line_size(self, mls: int) -> None:
        return None

    def get_max_line_size(self) -> int:
        return -1

    def set_comment_fill(self, fill: str) -> None:
        return None

    @abstractmethod
    def emits_markup(self) -> bool:
        ...

    def reset_defaults(self) -> None:
        self.reset_defaults_internal()

    def get_paren_level(self) -> int:
        return self.parenlevel

    def get_indent_increment(self) -> int:
        return self.indentincrement

    def set_indent_increment(self, val: int) -> None:
        self.indentincrement = val

    def set_pending_print(self, pend: PendPrint) -> None:
        self.pend_print = pend

    def cancel_pending_print(self) -> None:
        self.pend_print = None

    def has_pending_print(self, pend: PendPrint) -> bool:
        return self.pend_print is pend

    def open_brace_indent(self, brace: str, style: BraceStyle) -> int:
        if style == BraceStyle.same_line:
            self.spaces(1)
        elif style == BraceStyle.skip_line:
            self.tag_line()
            self.tag_line()
        else:
            self.tag_line()
        id = self.start_indent()
        self.print(brace)
        return id

    def open_brace(self, brace: str, style: BraceStyle) -> None:
        if style == BraceStyle.same_line:
            self.spaces(1)
        elif style == BraceStyle.skip_line:
            self.tag_line()
            self.tag_line()
        else:
            self.tag_line()
        self.print(brace)

    def close_brace_indent(self, brace: str, id: int) -> None:
        self.stop_indent(id)
        self.tag_line()
        self.print(brace)


class EmitMarkup(Emit):
    """Emitter that associates markup with individual tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.s: Optional[TextIO] = None
        self.encoder: Optional[XmlEncoder] = None

    def _encoder(self) -> XmlEncoder:
        if self.encoder is None:
            raise EmitError("EmitMarkup sin stream de salida")
        return self.encoder

    def begin_document(self) -> int:
        self._encoder().open_element(ELEM_CLANG_DOCUMENT)
        return 0

    def end_document(self, id: int) -> None:
        self._encoder().close_element(ELEM_CLANG_DOCUMENT)

    def begin_function(self, fd: Any) -> int:
        self._encoder().open_element(ELEM_FUNCTION)
        return 0

    def end_function(self, id: int) -> None:
        self._encoder().close_element(ELEM_FUNCTION)

    def begin_block(self, bl: Any) -> int:
        encoder = self._encoder()
        encoder.open_element(ELEM_BLOCK)
        index = _reference(bl, "index")
        if index is not None:
            encoder.write_signed_integer(ATTRIB_BLOCKREF, index)
        return 0

    def end_block(self, id: int) -> None:
        self._encoder().close_element(ELEM_BLOCK)

    def tag_line(self, indent: Optional[int] = None) -> None:
        self.emit_pending()
        encoder = self._encoder()
        encoder.open_element(ELEM_BREAK)
        encoder.write_signed_integer(
            ATTRIB_INDENT, self.indentlevel if indent is None else indent
        )
        encoder.close_element(ELEM_BREAK)

    def begin_return_type(self, vn: Any) -> int:
        encoder = self._encoder()
        encoder.open_element(ELEM_RETURN_TYPE)
        varref = _reference(vn, "create_index")
        if varref is not None:
            encoder.write_unsigned_integer(ATTRIB_VARREF, varref)
        return 0

    def end_return_type(self, id: int) -> None:
        self._encoder().close_element(ELEM_RETURN_TYPE)

    def begin_var_decl(self, symbol: Any) -> int:
        encoder = self._encoder()
        encoder.open_element(ELEM_VARDECL)
        symref = _reference(symbol, "symbol_id")
        if symref is not None:
            encoder.write_unsigned_integer(ATTRIB_SYMREF, symref)
        return 0

    def end_var_decl(self, id: int) -> None:
        self._encoder().close_element(ELEM_VARDECL)

    def begin_statement(self, op: Any) -> int:
        encoder = self._encoder()
        encoder.open_element(ELEM_STATEMENT)
        opref = _reference(op, "op_time")
        if opref is not None:
            encoder.write_unsigned_integer(ATTRIB_OPREF, opref)
        return 0

    def end_statement(self, id: int) -> None:
        self._encoder().close_element(ELEM_STATEMENT)

    def begin_func_proto(self) -> int:
        self._encoder().open_element(ELEM_FUNCPROTO)
        return 0

    def end_func_proto(self, id: int) -> None:
        self._encoder().close_element(ELEM_FUNCPROTO)

    def tag_variable(self, name: str, hl: SyntaxHighlight, vn: Any, op: Any) -> None:
        encoder = self._encoder()
        encoder.open_element(ELEM_VARIABLE)
        if hl != SyntaxHighlight.no_color:
            encoder.write_unsigned_integer(ATTRIB_COLOR, int(hl))
        varref = _reference(vn, "create_index")
        if varref is not None:
            encoder.write_unsigned_integer(ATTRIB_VARREF, varref)
        opref = _reference(op, "op_time")
        if opref is not None:
            encoder.write_unsigned_integer(ATTRIB_OPREF, opref)
        encoder.write_string(ATTRIB_CONTENT, name)
        encoder.close_element(ELEM_VARIABLE)

    def tag_op(self, name: str, hl: SyntaxHighlight, op: Any) -> None:
        encoder = self._encoder()
        encoder.open_element(ELEM_OP)
        if hl != SyntaxHighlight.no_color:
            encoder.write_unsigned_integer(ATTRIB_COLOR, int(hl))
        opref = _reference(op, "op_time")
        if opref is not None:
            encoder.write_unsigned_integer(ATTRIB_OPREF, opref)
        encoder.write_string(ATTRIB_CONTENT, name)
        encoder.close_element(ELEM_OP)

    def tag_func_name(self, name: str, hl: SyntaxHighlight, fd: Any, op: Any) -> None:
        encoder = self._encoder()
        encoder.open_element(ELEM_FUNCNAME)
        if hl != SyntaxHighlight.no_color:
            encoder.write_unsigned_integer(ATTRIB_COLOR, int(hl))
        opref = _reference(op, "op_time")
        if opref is not None:
            encoder.write_unsigned_integer(ATTRIB_OPREF, opref)
        encoder.write_string(ATTRIB_CONTENT, name)
        encoder.close_element(ELEM_FUNCNAME)

    def tag_type(self, name: str, hl: SyntaxHighlight, ct: Any) -> None:
        encoder = self._encoder()
        encoder.open_element(ELEM_TYPE)
        if hl != SyntaxHighlight.no_color:
            encoder.write_unsigned_integer(ATTRIB_COLOR, int(hl))
        type_id = _reference(ct, "unsized_id")
        if type_id not in (None, 0):
            encoder.write_unsigned_integer(ATTRIB_ID, type_id)
        encoder.write_string(ATTRIB_CONTENT, name)
        encoder.close_element(ELEM_TYPE)

    def tag_field(
        self, name: str, hl: SyntaxHighlight, ct: Any, off: int, op: Any
    ) -> None:
        encoder = self._encoder()
        encoder.open_element(ELEM_FIELD)
        if hl != SyntaxHighlight.no_color:
            encoder.write_unsigned_integer(ATTRIB_COLOR, int(hl))
        if ct is not None:
            encoder.write_string(ATTRIB_NAME, _type_name(ct))
            type_id = _reference(ct, "unsized_id")
            if type_id not in (None, 0):
                encoder.write_unsigned_integer(ATTRIB_ID, type_id)
            encoder.write_signed_integer(ATTRIB_OFF, off)
            opref = _reference(op, "op_time")
            if opref is not None:
                encoder.write_unsigned_integer(ATTRIB_OPREF, opref)
        encoder.write_string(ATTRIB_CONTENT, name)
        encoder.close_element(ELEM_FIELD)

    def tag_bit_field(
        self, name: str, hl: SyntaxHighlight, ct: Any, id: int, op: Any
    ) -> None:
        encoder = self._encoder()
        encoder.open_element(ELEM_BITFIELD)
        if hl != SyntaxHighlight.no_color:
            encoder.write_unsigned_integer(ATTRIB_COLOR, int(hl))
        encoder.write_string(ATTRIB_NAME, _type_name(ct))
        type_id = _reference(ct, "unsized_id")
        if type_id not in (None, 0):
            encoder.write_unsigned_integer(ATTRIB_ID, type_id)
        encoder.write_signed_integer(ATTRIB_OFF, id)
        opref = _reference(op, "op_time")
        if opref is not None:
            encoder.write_unsigned_integer(ATTRIB_OPREF, opref)
        encoder.write_string(ATTRIB_CONTENT, name)
        encoder.close_element(ELEM_BITFIELD)

    def tag_comment(self, name: str, hl: SyntaxHighlight, spc: Any, off: int) -> None:
        encoder = self._encoder()
        encoder.open_element(ELEM_COMMENT)
        if hl != SyntaxHighlight.no_color:
            encoder.write_unsigned_integer(ATTRIB_COLOR, int(hl))
        encoder.write_space(ATTRIB_SPACE, spc)
        encoder.write_unsigned_integer(ATTRIB_OFF, off)
        encoder.write_string(ATTRIB_CONTENT, name)
        encoder.close_element(ELEM_COMMENT)

    def tag_label(self, name: str, hl: SyntaxHighlight, spc: Any, off: int) -> None:
        encoder = self._encoder()
        encoder.open_element(ELEM_LABEL)
        if hl != SyntaxHighlight.no_color:
            encoder.write_unsigned_integer(ATTRIB_COLOR, int(hl))
        encoder.write_space(ATTRIB_SPACE, spc)
        encoder.write_unsigned_integer(ATTRIB_OFF, off)
        encoder.write_string(ATTRIB_CONTENT, name)
        encoder.close_element(ELEM_LABEL)

    def tag_case_label(
        self, name: str, hl: SyntaxHighlight, op: Any, value: int
    ) -> None:
        encoder = self._encoder()
        encoder.open_element(ELEM_VALUE)
        if hl != SyntaxHighlight.no_color:
            encoder.write_unsigned_integer(ATTRIB_COLOR, int(hl))
        encoder.write_unsigned_integer(ATTRIB_OFF, value)
        opref = _reference(op, "op_time")
        if opref is not None:
            encoder.write_unsigned_integer(ATTRIB_OPREF, opref)
        encoder.write_string(ATTRIB_CONTENT, name)
        encoder.close_element(ELEM_VALUE)

    def print(self, data: str, hl: SyntaxHighlight = SyntaxHighlight.no_color) -> None:
        encoder = self._encoder()
        encoder.open_element(ELEM_SYNTAX)
        if hl != SyntaxHighlight.no_color:
            encoder.write_unsigned_integer(ATTRIB_COLOR, int(hl))
        encoder.write_string(ATTRIB_CONTENT, data)
        encoder.close_element(ELEM_SYNTAX)

    def open_paren(self, paren: str, id: int = 0) -> int:
        encoder = self._encoder()
        encoder.open_element(ELEM_SYNTAX)
        encoder.write_signed_integer(ATTRIB_OPEN, id)
        encoder.write_string(ATTRIB_CONTENT, paren)
        encoder.close_element(ELEM_SYNTAX)
        self.parenlevel += 1
        return 0

    def close_paren(self, paren: str, id: int) -> None:
        encoder = self._encoder()
        encoder.open_element(ELEM_SYNTAX)
        encoder.write_signed_integer(ATTRIB_CLOSE, id)
        encoder.write_string(ATTRIB_CONTENT, paren)
        encoder.close_element(ELEM_SYNTAX)
        self.parenlevel -= 1

    def set_output_stream(self, stream: Optional[TextIO]) -> None:
        self.s = stream
        self.encoder = XmlEncoder(stream) if stream is not None else None

    def get_output_stream(self) -> Optional[TextIO]:
        return self.s

    def set_packed_output(self, val: bool) -> None:
        if self.encoder is None:
            return
        if val:
            raise EmitError(
                "PackedEncode (marshaling binario de Ghidra) no está portado; "
                "usa set_packed_output(False) para XML"
            )
        self.encoder = XmlEncoder(self.s) if self.s is not None else None

    def emits_markup(self) -> bool:
        return True


def _type_name(ct: Any) -> str:
    """Best-effort data-type name for markup attributes."""
    if ct is None:
        return ""
    name = getattr(ct, "name", None)
    if name:
        return str(name)
    ctype = getattr(ct, "ctype", None)
    if ctype is not None:
        name = getattr(ctype, "name", None)
        if name:
            return str(name)
    return ""


class EmitNoMarkup(Emit):
    """A trivial emitter that outputs syntax straight to the stream."""

    def __init__(self) -> None:
        super().__init__()
        self.s: Optional[TextIO] = None

    def _write(self, text: str) -> None:
        if self.s is None:
            raise EmitError("EmitNoMarkup sin stream de salida")
        self.s.write(text)

    def begin_document(self) -> int:
        return 0

    def end_document(self, id: int) -> None:
        return None

    def begin_function(self, fd: Any) -> int:
        return 0

    def end_function(self, id: int) -> None:
        return None

    def begin_block(self, bl: Any) -> int:
        return 0

    def end_block(self, id: int) -> None:
        return None

    def _line(self, indent: int) -> None:
        self._write("\n")
        for _ in range(indent, 0, -1):
            self._write(" ")

    def tag_line(self, indent: Optional[int] = None) -> None:
        self._line(self.indentlevel if indent is None else indent)

    def begin_return_type(self, vn: Any) -> int:
        return 0

    def end_return_type(self, id: int) -> None:
        return None

    def begin_var_decl(self, symbol: Any) -> int:
        return 0

    def end_var_decl(self, id: int) -> None:
        return None

    def begin_statement(self, op: Any) -> int:
        return 0

    def end_statement(self, id: int) -> None:
        return None

    def begin_func_proto(self) -> int:
        return 0

    def end_func_proto(self, id: int) -> None:
        return None

    def tag_variable(self, name: str, hl: SyntaxHighlight, vn: Any, op: Any) -> None:
        self._write(name)

    def tag_op(self, name: str, hl: SyntaxHighlight, op: Any) -> None:
        self._write(name)

    def tag_func_name(self, name: str, hl: SyntaxHighlight, fd: Any, op: Any) -> None:
        self._write(name)

    def tag_type(self, name: str, hl: SyntaxHighlight, ct: Any) -> None:
        self._write(name)

    def tag_field(
        self, name: str, hl: SyntaxHighlight, ct: Any, off: int, op: Any
    ) -> None:
        self._write(name)

    def tag_bit_field(
        self, name: str, hl: SyntaxHighlight, ct: Any, id: int, op: Any
    ) -> None:
        self._write(name)

    def tag_comment(self, name: str, hl: SyntaxHighlight, spc: Any, off: int) -> None:
        self._write(name)

    def tag_label(self, name: str, hl: SyntaxHighlight, spc: Any, off: int) -> None:
        self._write(name)

    def tag_case_label(
        self, name: str, hl: SyntaxHighlight, op: Any, value: int
    ) -> None:
        self._write(name)

    def print(self, data: str, hl: SyntaxHighlight = SyntaxHighlight.no_color) -> None:
        self._write(data)

    def open_paren(self, paren: str, id: int = 0) -> int:
        self._write(paren)
        self.parenlevel += 1
        return id

    def close_paren(self, paren: str, id: int) -> None:
        self._write(paren)
        self.parenlevel -= 1

    def set_output_stream(self, stream: Optional[TextIO]) -> None:
        self.s = stream

    def get_output_stream(self) -> Optional[TextIO]:
        return self.s

    def emits_markup(self) -> bool:
        return False


class PrintClass(IntEnum):
    """``TokenSplit::printclass``."""

    begin = 0
    end = 1
    tokenstring = 2
    tokenbreak = 3
    begin_indent = 4
    end_indent = 5
    begin_comment = 6
    end_comment = 7
    ignore = 8


class TagType(IntEnum):
    """``TokenSplit::tag_type``."""

    docu_b = 0
    docu_e = 1
    func_b = 2
    func_e = 3
    bloc_b = 4
    bloc_e = 5
    rtyp_b = 6
    rtyp_e = 7
    vard_b = 8
    vard_e = 9
    stat_b = 10
    stat_e = 11
    prot_b = 12
    prot_e = 13
    vari_t = 14
    op_t = 15
    fnam_t = 16
    type_t = 17
    field_t = 18
    bitfield_t = 19
    comm_t = 20
    label_t = 21
    case_t = 22
    synt_t = 23
    opar_t = 24
    cpar_t = 25
    oinv_t = 26
    cinv_t = 27
    spac_t = 28
    bump_t = 29
    line_t = 30


class TokenSplit:
    """A token/command object in the pretty printing stream."""

    _countbase = 0

    def __init__(self) -> None:
        self.tagtype = TagType.synt_t
        self.delimtype = PrintClass.ignore
        self.tok = ""
        self.hl = SyntaxHighlight.no_color
        self.op: Any = None
        self.ptr_second: Any = None
        self.off = 0
        self.indentbump = 0
        self.numspaces = 0
        self.size = 0
        self.count = 0

    @classmethod
    def _next_count(cls) -> int:
        count = cls._countbase
        cls._countbase += 1
        return count

    # -- begin/end commands ---------------------------------------------
    def begin_document(self) -> int:
        self.tagtype = TagType.docu_b
        self.delimtype = PrintClass.begin
        self.size = 0
        self.count = self._next_count()
        return self.count

    def end_document(self, id: int) -> None:
        self.tagtype = TagType.docu_e
        self.delimtype = PrintClass.end
        self.size = 0
        self.count = id

    def begin_function(self, f: Any) -> int:
        self.tagtype = TagType.func_b
        self.delimtype = PrintClass.begin
        self.size = 0
        self.ptr_second = f
        self.count = self._next_count()
        return self.count

    def end_function(self, id: int) -> None:
        self.tagtype = TagType.func_e
        self.delimtype = PrintClass.end
        self.size = 0
        self.count = id

    def begin_block(self, b: Any) -> int:
        self.tagtype = TagType.bloc_b
        self.delimtype = PrintClass.ignore
        self.ptr_second = b
        self.count = self._next_count()
        return self.count

    def end_block(self, id: int) -> None:
        self.tagtype = TagType.bloc_e
        self.delimtype = PrintClass.ignore
        self.count = id

    def begin_return_type(self, v: Any) -> int:
        self.tagtype = TagType.rtyp_b
        self.delimtype = PrintClass.begin
        self.ptr_second = v
        self.count = self._next_count()
        return self.count

    def end_return_type(self, id: int) -> None:
        self.tagtype = TagType.rtyp_e
        self.delimtype = PrintClass.end
        self.count = id

    def begin_var_decl(self, symbol: Any) -> int:
        self.tagtype = TagType.vard_b
        self.delimtype = PrintClass.begin
        self.ptr_second = symbol
        self.count = self._next_count()
        return self.count

    def end_var_decl(self, id: int) -> None:
        self.tagtype = TagType.vard_e
        self.delimtype = PrintClass.end
        self.count = id

    def begin_statement(self, o: Any) -> int:
        self.tagtype = TagType.stat_b
        self.delimtype = PrintClass.begin
        self.op = o
        self.count = self._next_count()
        return self.count

    def end_statement(self, id: int) -> None:
        self.tagtype = TagType.stat_e
        self.delimtype = PrintClass.end
        self.count = id

    def begin_func_proto(self) -> int:
        self.tagtype = TagType.prot_b
        self.delimtype = PrintClass.begin
        self.count = self._next_count()
        return self.count

    def end_func_proto(self, id: int) -> None:
        self.tagtype = TagType.prot_e
        self.delimtype = PrintClass.end
        self.count = id

    # -- content tokens --------------------------------------------------
    def tag_variable(
        self, name: str, h: SyntaxHighlight, v: Any, o: Any
    ) -> None:
        self.tok = name
        self.size = len(self.tok)
        self.tagtype = TagType.vari_t
        self.delimtype = PrintClass.tokenstring
        self.hl = h
        self.ptr_second = v
        self.op = o

    def tag_op(self, name: str, h: SyntaxHighlight, o: Any) -> None:
        self.tok = name
        self.size = len(self.tok)
        self.tagtype = TagType.op_t
        self.delimtype = PrintClass.tokenstring
        self.hl = h
        self.op = o

    def tag_func_name(self, name: str, h: SyntaxHighlight, f: Any, o: Any) -> None:
        self.tok = name
        self.size = len(self.tok)
        self.tagtype = TagType.fnam_t
        self.delimtype = PrintClass.tokenstring
        self.hl = h
        self.ptr_second = f
        self.op = o

    def tag_type(self, name: str, h: SyntaxHighlight, ct: Any) -> None:
        self.tok = name
        self.size = len(self.tok)
        self.tagtype = TagType.type_t
        self.delimtype = PrintClass.tokenstring
        self.hl = h
        self.ptr_second = ct

    def tag_field(
        self, name: str, h: SyntaxHighlight, ct: Any, o: int, in_op: Any
    ) -> None:
        self.tok = name
        self.size = len(self.tok)
        self.tagtype = TagType.field_t
        self.delimtype = PrintClass.tokenstring
        self.hl = h
        self.ptr_second = ct
        self.off = o
        self.op = in_op

    def tag_bit_field(
        self, name: str, h: SyntaxHighlight, ct: Any, id: int, in_op: Any
    ) -> None:
        self.tok = name
        self.size = len(self.tok)
        self.tagtype = TagType.bitfield_t
        self.delimtype = PrintClass.tokenstring
        self.hl = h
        self.ptr_second = ct
        self.off = id
        self.op = in_op

    def tag_comment(self, name: str, h: SyntaxHighlight, s: Any, o: int) -> None:
        self.tok = name
        self.size = len(self.tok)
        self.ptr_second = s
        self.off = o
        self.tagtype = TagType.comm_t
        self.delimtype = PrintClass.tokenstring
        self.hl = h

    def tag_label(self, name: str, h: SyntaxHighlight, s: Any, o: int) -> None:
        self.tok = name
        self.size = len(self.tok)
        self.ptr_second = s
        self.off = o
        self.tagtype = TagType.label_t
        self.delimtype = PrintClass.tokenstring
        self.hl = h

    def tag_case_label(
        self, name: str, h: SyntaxHighlight, in_op: Any, int_value: int
    ) -> None:
        self.tok = name
        self.size = len(self.tok)
        self.op = in_op
        self.off = int_value
        self.tagtype = TagType.case_t
        self.delimtype = PrintClass.tokenstring
        self.hl = h

    def print(self, data: str, h: SyntaxHighlight) -> None:
        self.tok = data
        self.size = len(self.tok)
        self.tagtype = TagType.synt_t
        self.delimtype = PrintClass.tokenstring
        self.hl = h

    def open_paren(self, paren: str, id: int) -> None:
        self.tok = paren
        self.size = 1
        self.tagtype = TagType.opar_t
        self.delimtype = PrintClass.tokenstring
        self.count = id

    def close_paren(self, paren: str, id: int) -> None:
        self.tok = paren
        self.size = 1
        self.tagtype = TagType.cpar_t
        self.delimtype = PrintClass.tokenstring
        self.count = id

    def open_group(self) -> int:
        self.tagtype = TagType.oinv_t
        self.delimtype = PrintClass.begin
        self.count = self._next_count()
        return self.count

    def close_group(self, id: int) -> None:
        self.tagtype = TagType.cinv_t
        self.delimtype = PrintClass.end
        self.count = id

    def start_indent(self, bump: int) -> int:
        self.tagtype = TagType.bump_t
        self.delimtype = PrintClass.begin_indent
        self.indentbump = bump
        self.size = 0
        self.count = self._next_count()
        return self.count

    def stop_indent(self, id: int) -> None:
        self.tagtype = TagType.bump_t
        self.delimtype = PrintClass.end_indent
        self.size = 0
        self.count = id

    def start_comment(self) -> int:
        self.tagtype = TagType.oinv_t
        self.delimtype = PrintClass.begin_comment
        self.count = self._next_count()
        return self.count

    def stop_comment(self, id: int) -> None:
        self.tagtype = TagType.cinv_t
        self.delimtype = PrintClass.end_comment
        self.count = id

    def spaces(self, num: int, bump: int) -> None:
        self.tagtype = TagType.spac_t
        self.delimtype = PrintClass.tokenbreak
        self.numspaces = num
        self.indentbump = bump

    def tag_line(self, indent: Optional[int] = None) -> None:
        if indent is None:
            self.tagtype = TagType.bump_t
            self.delimtype = PrintClass.tokenbreak
            self.numspaces = 999999
            self.indentbump = 0
        else:
            self.tagtype = TagType.line_t
            self.delimtype = PrintClass.tokenbreak
            self.numspaces = 999999
            self.indentbump = indent

    # -- accessors -------------------------------------------------------
    def do_print(self, emit: Emit) -> None:
        """Send ``this`` token to the given low-level emitter."""
        tag = self.tagtype
        if tag == TagType.docu_b:
            emit.begin_document()
        elif tag == TagType.docu_e:
            emit.end_document(self.count)
        elif tag == TagType.func_b:
            emit.begin_function(self.ptr_second)
        elif tag == TagType.func_e:
            emit.end_function(self.count)
        elif tag == TagType.bloc_b:
            emit.begin_block(self.ptr_second)
        elif tag == TagType.bloc_e:
            emit.end_block(self.count)
        elif tag == TagType.rtyp_b:
            emit.begin_return_type(self.ptr_second)
        elif tag == TagType.rtyp_e:
            emit.end_return_type(self.count)
        elif tag == TagType.vard_b:
            emit.begin_var_decl(self.ptr_second)
        elif tag == TagType.vard_e:
            emit.end_var_decl(self.count)
        elif tag == TagType.stat_b:
            emit.begin_statement(self.op)
        elif tag == TagType.stat_e:
            emit.end_statement(self.count)
        elif tag == TagType.prot_b:
            emit.begin_func_proto()
        elif tag == TagType.prot_e:
            emit.end_func_proto(self.count)
        elif tag == TagType.vari_t:
            emit.tag_variable(self.tok, self.hl, self.ptr_second, self.op)
        elif tag == TagType.op_t:
            emit.tag_op(self.tok, self.hl, self.op)
        elif tag == TagType.fnam_t:
            emit.tag_func_name(self.tok, self.hl, self.ptr_second, self.op)
        elif tag == TagType.type_t:
            emit.tag_type(self.tok, self.hl, self.ptr_second)
        elif tag == TagType.field_t:
            emit.tag_field(self.tok, self.hl, self.ptr_second, self.off, self.op)
        elif tag == TagType.bitfield_t:
            emit.tag_bit_field(self.tok, self.hl, self.ptr_second, self.off, self.op)
        elif tag == TagType.comm_t:
            emit.tag_comment(self.tok, self.hl, self.ptr_second, self.off)
        elif tag == TagType.label_t:
            emit.tag_label(self.tok, self.hl, self.ptr_second, self.off)
        elif tag == TagType.case_t:
            emit.tag_case_label(self.tok, self.hl, self.op, self.off)
        elif tag == TagType.synt_t:
            emit.print(self.tok, self.hl)
        elif tag == TagType.opar_t:
            emit.open_paren(self.tok, self.count)
        elif tag == TagType.cpar_t:
            emit.close_paren(self.tok, self.count)
        elif tag in (TagType.oinv_t, TagType.cinv_t):
            return
        elif tag == TagType.spac_t:
            emit.spaces(self.numspaces)
        else:  # line_t / bump_t
            raise EmitError("Should never get called")

    def get_indent_bump(self) -> int:
        return self.indentbump

    def get_num_spaces(self) -> int:
        return self.numspaces

    def get_size(self) -> int:
        return self.size

    def set_size(self, sz: int) -> None:
        self.size = sz

    def get_class(self) -> PrintClass:
        return self.delimtype

    def get_tag(self) -> TagType:
        return self.tagtype

    def get_count(self) -> int:
        return self.count

    def print_debug(self) -> str:
        return self.tagtype.name


class CircularQueue:
    """Circular buffer usable as a stack or a queue (``circularqueue<_type>``)."""

    def __init__(self, sz: int, factory: Any = int) -> None:
        self.max = sz
        self.left = 1
        self.right = 0
        self.factory = factory
        self.cache = [factory() for _ in range(sz)]

    def set_max(self, sz: int) -> None:
        if self.max != sz:
            self.max = sz
            self.cache = [self.factory() for _ in range(sz)]
        self.left = 1
        self.right = 0

    def get_max(self) -> int:
        return self.max

    def expand(self, amount: int) -> None:
        newcache = [self.factory() for _ in range(self.max + amount)]
        i = self.left
        j = 0
        while i != self.right:
            newcache[j] = self.cache[i]
            j += 1
            i = (i + 1) % self.max
        newcache[j] = self.cache[i]
        self.left = 0
        self.right = j
        self.cache = newcache
        self.max += amount

    def clear(self) -> None:
        self.left = 1
        self.right = 0

    def empty(self) -> bool:
        return self.left == (self.right + 1) % self.max

    def topref(self) -> int:
        return self.right

    def bottomref(self) -> int:
        return self.left

    def ref(self, r: int) -> Any:
        return self.cache[r]

    def set_ref(self, r: int, value: Any) -> None:
        self.cache[r] = value

    def top(self) -> Any:
        return self.cache[self.right]

    def bottom(self) -> Any:
        return self.cache[self.left]

    def push(self, value: Any = None) -> Any:
        self.right = (self.right + 1) % self.max
        if value is not None:
            self.cache[self.right] = value
        elif isinstance(self.cache[self.right], TokenSplit):
            # Reuse the slot but start from a clean token, as the C++ template
            # hands out a freshly assignable object.
            self.cache[self.right] = TokenSplit()
        return self.cache[self.right]

    def pop(self) -> Any:
        tmp = self.right
        self.right = (self.right + self.max - 1) % self.max
        return self.cache[tmp]

    def popbottom(self) -> Any:
        tmp = self.left
        self.left = (self.left + 1) % self.max
        return self.cache[tmp]


class EmitPrettyPrint(Emit):
    """Generic source code pretty printer (Derek C. Oppen's algorithm)."""

    def __init__(self) -> None:
        super().__init__()
        self.lowlevel: Emit = EmitNoMarkup()
        self.indentstack: list[int] = []
        self.spaceremain = 0
        self.maxlinesize = 0
        self.leftotal = 1
        self.rightotal = 1
        self.needbreak = False
        self.commentmode = False
        self.commentfill = ""
        self.scanqueue = CircularQueue(3 * 100, int)
        self.tokqueue = CircularQueue(3 * 100, TokenSplit)
        # Mirrors the C++ PRETTY_DEBUG compile-time switch: delimiter ids are
        # tracked and validated only when the check is enabled.
        self.pretty_debug = False
        self.checkid: list[int] = []
        self.reset_defaults_pretty_print()

    # -- internals -------------------------------------------------------
    def reset_defaults_pretty_print(self) -> None:
        self.set_max_line_size(100)

    def expand(self) -> None:
        max = self.tokqueue.get_max()
        left = self.tokqueue.bottomref()
        self.tokqueue.expand(200)
        for i in range(max):
            self.scanqueue.set_ref(
                i, (self.scanqueue.ref(i) + max - left) % max
            )
        self.scanqueue.expand(200)

    def overflow(self) -> None:
        half = self.maxlinesize // 2
        for i in range(len(self.indentstack) - 1, -1, -1):
            if self.indentstack[i] < half:
                self.indentstack[i] = half
            else:
                break
        if self.indentstack:
            newspaceremain = self.indentstack[-1]
        else:
            newspaceremain = self.maxlinesize
        if newspaceremain == self.spaceremain:
            return
        if self.commentmode and (
            newspaceremain == self.spaceremain + len(self.commentfill)
        ):
            return
        self.spaceremain = newspaceremain
        self.lowlevel.tag_line(self.maxlinesize - self.spaceremain)
        if self.commentmode and len(self.commentfill) != 0:
            self.lowlevel.print(self.commentfill, SyntaxHighlight.comment_color)
            self.spaceremain -= len(self.commentfill)

    def _print_token(self, tok: TokenSplit) -> None:
        val = 0
        cls = tok.get_class()
        debug = self.pretty_debug
        if cls == PrintClass.ignore:
            tok.do_print(self.lowlevel)
        elif cls == PrintClass.begin_indent:
            val = self.indentstack[-1] - tok.get_indent_bump()
            self.indentstack.append(val)
            if debug:
                self.checkid.append(tok.get_count())
        elif cls == PrintClass.begin_comment:
            self.commentmode = True
            # fallthru, treat as a group begin
            tok.do_print(self.lowlevel)
            self.indentstack.append(self.spaceremain)
            if debug:
                self.checkid.append(tok.get_count())
        elif cls == PrintClass.begin:
            tok.do_print(self.lowlevel)
            self.indentstack.append(self.spaceremain)
            if debug:
                self.checkid.append(tok.get_count())
        elif cls == PrintClass.end_indent:
            if not self.indentstack:
                raise EmitError("indent error")
            if debug:
                if not self.checkid or self.checkid[-1] != tok.get_count():
                    raise EmitError("mismatch1")
                self.checkid.pop()
                if not self.indentstack:
                    raise EmitError("Empty indent stack")
            self.indentstack.pop()
        elif cls == PrintClass.end_comment:
            self.commentmode = False
            # fallthru, treat as a group end
            tok.do_print(self.lowlevel)
            if debug:
                if not self.checkid or self.checkid[-1] != tok.get_count():
                    raise EmitError("mismatch2")
                self.checkid.pop()
                if not self.indentstack:
                    raise EmitError("indent error")
            self.indentstack.pop()
        elif cls == PrintClass.end:
            tok.do_print(self.lowlevel)
            if debug:
                if not self.checkid or self.checkid[-1] != tok.get_count():
                    raise EmitError("mismatch2")
                self.checkid.pop()
                if not self.indentstack:
                    raise EmitError("indent error")
            self.indentstack.pop()
        elif cls == PrintClass.tokenstring:
            if tok.get_size() > self.spaceremain:
                self.overflow()
            tok.do_print(self.lowlevel)
            self.spaceremain -= tok.get_size()
        elif cls == PrintClass.tokenbreak:
            if tok.get_size() > self.spaceremain:
                if tok.get_tag() == TagType.line_t:  # Absolute indent
                    self.spaceremain = self.maxlinesize - tok.get_indent_bump()
                else:  # relative indent
                    val = self.indentstack[-1] - tok.get_indent_bump()
                    # If creating a line break doesn't save that much
                    # don't do the line break
                    if (tok.get_num_spaces() <= self.spaceremain) and (
                        val - self.spaceremain < 10
                    ):
                        self.lowlevel.spaces(tok.get_num_spaces())
                        self.spaceremain -= tok.get_num_spaces()
                        return
                    self.indentstack[-1] = val
                    self.spaceremain = val
                self.lowlevel.tag_line(self.maxlinesize - self.spaceremain)
                if self.commentmode and len(self.commentfill) != 0:
                    self.lowlevel.print(self.commentfill, SyntaxHighlight.comment_color)
                    self.spaceremain -= len(self.commentfill)
            else:
                self.lowlevel.spaces(tok.get_num_spaces())
                self.spaceremain -= tok.get_num_spaces()

    def advanceleft(self) -> None:
        l = self.tokqueue.bottom().get_size()
        while l >= 0:
            tok = self.tokqueue.bottom()
            self._print_token(tok)
            cls = tok.get_class()
            if cls == PrintClass.tokenbreak:
                self.leftotal += tok.get_num_spaces()
            elif cls == PrintClass.tokenstring:
                self.leftotal += l
            self.tokqueue.popbottom()
            if self.tokqueue.empty():
                break
            l = self.tokqueue.bottom().get_size()

    def scan(self) -> None:
        if self.tokqueue.empty():
            self.expand()
        tok = self.tokqueue.top()
        cls = tok.get_class()
        if cls in (PrintClass.begin_comment, PrintClass.begin):
            if self.scanqueue.empty():
                self.leftotal = self.rightotal = 1
            tok.set_size(-self.rightotal)
            self.scanqueue.push(self.tokqueue.topref())
        elif cls in (PrintClass.end_comment, PrintClass.end):
            tok.set_size(0)
            if not self.scanqueue.empty():
                ref = self.tokqueue.ref(self.scanqueue.pop())
                ref.set_size(ref.get_size() + self.rightotal)
                if (
                    ref.get_class() == PrintClass.tokenbreak
                    and not self.scanqueue.empty()
                ):
                    ref2 = self.tokqueue.ref(self.scanqueue.pop())
                    ref2.set_size(ref2.get_size() + self.rightotal)
                if self.scanqueue.empty():
                    self.advanceleft()
        elif cls == PrintClass.tokenbreak:
            if self.scanqueue.empty():
                self.leftotal = self.rightotal = 1
            else:
                ref = self.tokqueue.ref(self.scanqueue.top())
                if ref.get_class() == PrintClass.tokenbreak:
                    self.scanqueue.pop()
                    ref.set_size(ref.get_size() + self.rightotal)
            tok.set_size(-self.rightotal)
            self.scanqueue.push(self.tokqueue.topref())
            self.rightotal += tok.get_num_spaces()
        elif cls in (
            PrintClass.begin_indent,
            PrintClass.end_indent,
            PrintClass.ignore,
        ):
            tok.set_size(0)
        elif cls == PrintClass.tokenstring:
            if not self.scanqueue.empty():
                self.rightotal += tok.get_size()
                while self.rightotal - self.leftotal > self.spaceremain:
                    ref = self.tokqueue.ref(self.scanqueue.popbottom())
                    ref.set_size(999999)
                    self.advanceleft()
                    if self.scanqueue.empty():
                        break

    def checkstart(self) -> None:
        if self.needbreak:
            tok = self.tokqueue.push()
            tok.spaces(0, 0)
            self.scan()
        self.needbreak = False

    def checkstring(self) -> None:
        if self.needbreak:
            tok = self.tokqueue.push()
            tok.spaces(0, 0)
            self.scan()
        self.needbreak = True

    def checkend(self) -> None:
        if not self.needbreak:
            tok = self.tokqueue.push()
            tok.print(Emit.EMPTY_STRING, SyntaxHighlight.no_color)
            self.scan()
        self.needbreak = True

    def checkbreak(self) -> None:
        if not self.needbreak:
            tok = self.tokqueue.push()
            tok.print(Emit.EMPTY_STRING, SyntaxHighlight.no_color)
            self.scan()
        self.needbreak = False

    def _push(self) -> TokenSplit:
        return self.tokqueue.push()

    # -- Emit API --------------------------------------------------------
    def begin_document(self) -> int:
        self.checkstart()
        tok = self._push()
        id = tok.begin_document()
        self.scan()
        return id

    def end_document(self, id: int) -> None:
        self.checkend()
        tok = self._push()
        tok.end_document(id)
        self.scan()

    def begin_function(self, fd: Any) -> int:
        self.checkstart()
        tok = self._push()
        id = tok.begin_function(fd)
        self.scan()
        return id

    def end_function(self, id: int) -> None:
        self.checkend()
        tok = self._push()
        tok.end_function(id)
        self.scan()

    def begin_block(self, bl: Any) -> int:
        tok = self._push()
        id = tok.begin_block(bl)
        self.scan()
        return id

    def end_block(self, id: int) -> None:
        tok = self._push()
        tok.end_block(id)
        self.scan()

    def tag_line(self, indent: Optional[int] = None) -> None:
        self.emit_pending()
        self.checkbreak()
        tok = self._push()
        tok.tag_line(indent)
        self.scan()

    def begin_return_type(self, vn: Any) -> int:
        self.checkstart()
        tok = self._push()
        id = tok.begin_return_type(vn)
        self.scan()
        return id

    def end_return_type(self, id: int) -> None:
        self.checkend()
        tok = self._push()
        tok.end_return_type(id)
        self.scan()

    def begin_var_decl(self, symbol: Any) -> int:
        self.checkstart()
        tok = self._push()
        id = tok.begin_var_decl(symbol)
        self.scan()
        return id

    def end_var_decl(self, id: int) -> None:
        self.checkend()
        tok = self._push()
        tok.end_var_decl(id)
        self.scan()

    def begin_statement(self, op: Any) -> int:
        self.checkstart()
        tok = self._push()
        id = tok.begin_statement(op)
        self.scan()
        return id

    def end_statement(self, id: int) -> None:
        self.checkend()
        tok = self._push()
        tok.end_statement(id)
        self.scan()

    def begin_func_proto(self) -> int:
        self.checkstart()
        tok = self._push()
        id = tok.begin_func_proto()
        self.scan()
        return id

    def end_func_proto(self, id: int) -> None:
        self.checkend()
        tok = self._push()
        tok.end_func_proto(id)
        self.scan()

    def tag_variable(self, name: str, hl: SyntaxHighlight, vn: Any, op: Any) -> None:
        self.checkstring()
        tok = self._push()
        tok.tag_variable(name, hl, vn, op)
        self.scan()

    def tag_op(self, name: str, hl: SyntaxHighlight, op: Any) -> None:
        self.checkstring()
        tok = self._push()
        tok.tag_op(name, hl, op)
        self.scan()

    def tag_func_name(self, name: str, hl: SyntaxHighlight, fd: Any, op: Any) -> None:
        self.checkstring()
        tok = self._push()
        tok.tag_func_name(name, hl, fd, op)
        self.scan()

    def tag_type(self, name: str, hl: SyntaxHighlight, ct: Any) -> None:
        self.checkstring()
        tok = self._push()
        tok.tag_type(name, hl, ct)
        self.scan()

    def tag_field(
        self, name: str, hl: SyntaxHighlight, ct: Any, off: int, op: Any
    ) -> None:
        self.checkstring()
        tok = self._push()
        tok.tag_field(name, hl, ct, off, op)
        self.scan()

    def tag_bit_field(
        self, name: str, hl: SyntaxHighlight, ct: Any, id: int, op: Any
    ) -> None:
        self.checkstring()
        tok = self._push()
        tok.tag_bit_field(name, hl, ct, id, op)
        self.scan()

    def tag_comment(self, name: str, hl: SyntaxHighlight, spc: Any, off: int) -> None:
        self.checkstring()
        tok = self._push()
        tok.tag_comment(name, hl, spc, off)
        self.scan()

    def tag_label(self, name: str, hl: SyntaxHighlight, spc: Any, off: int) -> None:
        self.checkstring()
        tok = self._push()
        tok.tag_label(name, hl, spc, off)
        self.scan()

    def tag_case_label(
        self, name: str, hl: SyntaxHighlight, op: Any, value: int
    ) -> None:
        self.checkstring()
        tok = self._push()
        tok.tag_case_label(name, hl, op, value)
        self.scan()

    def print(self, data: str, hl: SyntaxHighlight = SyntaxHighlight.no_color) -> None:
        self.checkstring()
        tok = self._push()
        tok.print(data, hl)
        self.scan()

    def open_paren(self, paren: str, id: int = 0) -> int:
        id = self.open_group()  # Open paren automatically opens group
        tok = self._push()
        tok.open_paren(paren, id)
        self.scan()
        self.needbreak = True
        return id

    def close_paren(self, paren: str, id: int) -> None:
        self.checkstring()
        tok = self._push()
        tok.close_paren(paren, id)
        self.scan()
        self.close_group(id)

    def open_group(self) -> int:
        self.checkstart()
        tok = self._push()
        id = tok.open_group()
        self.scan()
        return id

    def close_group(self, id: int) -> None:
        self.checkend()
        tok = self._push()
        tok.close_group(id)
        self.scan()

    def start_comment(self) -> int:
        self.checkstart()
        tok = self._push()
        id = tok.start_comment()
        self.scan()
        return id

    def stop_comment(self, id: int) -> None:
        self.checkend()
        tok = self._push()
        tok.stop_comment(id)
        self.scan()

    def clear(self) -> None:
        super().clear()
        self.lowlevel.clear()
        self.indentstack.clear()
        self.checkid.clear()
        self.scanqueue.clear()
        self.tokqueue.clear()
        self.leftotal = 1
        self.rightotal = 1
        self.needbreak = False
        self.commentmode = False
        self.spaceremain = self.maxlinesize

    def spaces(self, num: int, bump: int = 0) -> None:
        self.checkbreak()
        tok = self._push()
        tok.spaces(num, bump)
        self.scan()

    def start_indent(self) -> int:
        tok = self._push()
        id = tok.start_indent(self.indentincrement)
        self.scan()
        return id

    def stop_indent(self, id: int) -> None:
        tok = self._push()
        tok.stop_indent(id)
        self.scan()

    def flush(self) -> None:
        while not self.tokqueue.empty():
            tok = self.tokqueue.popbottom()
            if tok.get_size() < 0:
                raise EmitError("Cannot flush pretty printer. Missing group end")
            self._print_token(tok)
        self.needbreak = False
        if self.pretty_debug:
            if not self.scanqueue.empty():
                raise EmitError("prettyprint scanqueue did not flush")
            if self.indentstack:
                raise EmitError("prettyprint indentstack did not flush")
        self.lowlevel.flush()

    def set_markup(self, val: bool) -> None:
        stream = self.lowlevel.get_output_stream()
        self.lowlevel = EmitMarkup() if val else EmitNoMarkup()
        self.lowlevel.set_output_stream(stream)

    def set_output_stream(self, stream: Optional[TextIO]) -> None:
        self.lowlevel.set_output_stream(stream)

    def get_output_stream(self) -> Optional[TextIO]:
        return self.lowlevel.get_output_stream()

    def set_packed_output(self, val: bool) -> None:
        self.lowlevel.set_packed_output(val)

    def set_max_line_size(self, val: int) -> None:
        if val < 20 or val > 10000:
            raise EmitError("Bad maximum line size")
        self.maxlinesize = val
        self.scanqueue.set_max(3 * val)
        self.tokqueue.set_max(3 * val)
        self.spaceremain = self.maxlinesize
        self.clear()

    def get_max_line_size(self) -> int:
        return self.maxlinesize

    def set_comment_fill(self, fill: str) -> None:
        self.commentfill = fill

    def emits_markup(self) -> bool:
        return self.lowlevel.emits_markup()

    def reset_defaults(self) -> None:
        self.lowlevel.reset_defaults()
        self.reset_defaults_internal()
        self.reset_defaults_pretty_print()
