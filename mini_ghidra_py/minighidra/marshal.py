"""Port of Ghidra's marshaling layer (``marshal.hh``/``marshal.cc``).

Ghidra moves structured data between the decompiler and its clients through a
pair of abstract codecs: :class:`Encoder` writes elements/attributes and
:class:`Decoder` reads them back.  Two concrete formats exist and both are
ported here:

* :class:`XmlEncoder`/``XmlEncode`` -- the human readable XML form used by the
  ``<clang_document>`` markup of the printing engine.
* :class:`XmlDecode` -- reads that XML back into the same element/attribute
  vocabulary (``ElementId``/``AttributeId`` numeric ids).
* :class:`PackedEncoder`/``PackedEncode`` and :class:`PackedDecode` -- Ghidra's
  compact byte format.  Every byte is non-zero; element and attribute records
  are one or two bytes, integers are stored 7 bits per byte and strings are
  UTF-8 with a length prefix (see :class:`PackedFormat`).

The id registries (``ElementId::find``/``AttributeId::find``), the p-code
``OpCode`` enumeration (``opcodes.hh``) and the ``type_metatype`` enumeration
(``metatype.hh``) live here too, exactly as they do in the C++ sources.

Deviations from the C++ code
----------------------------
* There is no ``AddrSpaceManager`` in MiniGhidraPy, so an address space is
  marshaled by name.  :data:`SPACE_INDEX` mirrors the order in which
  ``Architecture`` registers its spaces (``const``, ``unique``, ``ram``,
  ``register``); ``stack``/``join``/``fspec``/``iop``/spacebase spaces use
  Ghidra's *special* space type codes instead of an index.
* ``PackedDecode`` keeps the whole stream in one buffer instead of Ghidra's
  linked list of 1024-byte chunks; the ``Position`` semantics (start of the
  open element, next attribute, end of the attributes) are preserved so
  ``rewind_attributes``/``get_next_attribute_id`` behave identically.
* ``Decoder`` returns space *names* (strings) rather than ``AddrSpace``
  pointers.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, BinaryIO, Iterable, Optional, TextIO
from xml.etree import ElementTree

__all__ = [
    "ATTRIB_BLOCKREF",
    "ATTRIB_CLOSE",
    "ATTRIB_COLOR",
    "ATTRIB_CONTENT",
    "ATTRIB_ID",
    "ATTRIB_INDENT",
    "ATTRIB_NAME",
    "ATTRIB_OFF",
    "ATTRIB_OPEN",
    "ATTRIB_OPREF",
    "ATTRIB_SPACE",
    "ATTRIB_SYMREF",
    "ATTRIB_UNKNOWN",
    "ATTRIB_VARREF",
    "AttributeId",
    "Decoder",
    "DecoderError",
    "ELEM_BITFIELD",
    "ELEM_BLOCK",
    "ELEM_BREAK",
    "ELEM_CLANG_DOCUMENT",
    "ELEM_COMMENT",
    "ELEM_FIELD",
    "ELEM_FUNCNAME",
    "ELEM_FUNCTION",
    "ELEM_FUNCPROTO",
    "ELEM_LABEL",
    "ELEM_OP",
    "ELEM_RETURN_TYPE",
    "ELEM_STATEMENT",
    "ELEM_SYNTAX",
    "ELEM_TYPE",
    "ELEM_UNKNOWN",
    "ELEM_VALUE",
    "ELEM_VARDECL",
    "ELEM_VARIABLE",
    "ElementId",
    "Encoder",
    "MarshalError",
    "OpCode",
    "PackedDecode",
    "PackedEncode",
    "PackedEncoder",
    "PackedFormat",
    "SPACE_INDEX",
    "SPECIAL_SPACE_CODE",
    "TypeMetatype",
    "XmlDecode",
    "XmlEncode",
    "XmlEncoder",
    "get_opcode",
    "get_opname",
    "is_formal_stack_space",
    "iter_elements",
    "lookup_attribute_id",
    "lookup_element_id",
    "metatype_to_string",
    "space_index_of",
    "space_name_of",
    "string_to_metatype",
    "xml_escape",
    "xml_read_bool",
]


class MarshalError(RuntimeError):
    """Raised for marshaling misuse, mirroring C++ ``LowlevelError``."""


class DecoderError(MarshalError):
    """Raised when an incoming stream is malformed (C++ ``DecoderError``)."""


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


# The ids of the placeholders match marshal.cc ("number serves as next open
# index"); they are deliberately not 0, because 0 means "no element/attribute
# left" while these mean "name not registered".
ELEM_UNKNOWN = ElementId("XMLunknown", 292)
ATTRIB_UNKNOWN = AttributeId("XMLunknown", 161)

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

_ELEMENTS: tuple[ElementId, ...] = (
    ELEM_UNKNOWN,
    ELEM_VALUE,
    ELEM_BREAK,
    ELEM_CLANG_DOCUMENT,
    ELEM_FUNCNAME,
    ELEM_FUNCPROTO,
    ELEM_LABEL,
    ELEM_RETURN_TYPE,
    ELEM_STATEMENT,
    ELEM_SYNTAX,
    ELEM_VARDECL,
    ELEM_VARIABLE,
    ELEM_OP,
    ELEM_FIELD,
    ELEM_TYPE,
    ELEM_COMMENT,
    ELEM_BLOCK,
    ELEM_FUNCTION,
    ELEM_BITFIELD,
)

_ATTRIBUTES: tuple[AttributeId, ...] = (
    ATTRIB_UNKNOWN,
    ATTRIB_CONTENT,
    ATTRIB_ID,
    ATTRIB_NAME,
    ATTRIB_SPACE,
    ATTRIB_BLOCKREF,
    ATTRIB_CLOSE,
    ATTRIB_COLOR,
    ATTRIB_INDENT,
    ATTRIB_OFF,
    ATTRIB_OPEN,
    ATTRIB_OPREF,
    ATTRIB_VARREF,
    ATTRIB_SYMREF,
)

_ELEMENT_LOOKUP: dict[str, int] = {item.name: item.id for item in _ELEMENTS}
_ATTRIBUTE_LOOKUP: dict[str, int] = {item.name: item.id for item in _ATTRIBUTES}


def lookup_element_id(name: str, scope: int = 0) -> int:
    """``ElementId::find``: the id of a name, ``ELEM_UNKNOWN`` when unregistered."""
    if scope != 0:  # Reverse look-up is only supported for scope 0.
        return ELEM_UNKNOWN.id
    return _ELEMENT_LOOKUP.get(name, ELEM_UNKNOWN.id)


def lookup_attribute_id(name: str, scope: int = 0) -> int:
    """``AttributeId::find``: the id of a name, ``ATTRIB_UNKNOWN`` if unknown."""
    if scope != 0:
        return ATTRIB_UNKNOWN.id
    return _ATTRIBUTE_LOOKUP.get(name, ATTRIB_UNKNOWN.id)


class TypeMetatype(IntEnum):
    """``type_metatype`` in ``metatype.hh``; the values are part of the format."""

    TYPE_PARTIALUNION = 0
    TYPE_PARTIALSTRUCT = 1
    TYPE_PARTIALENUM = 2
    TYPE_UNION = 3
    TYPE_STRUCT = 4
    TYPE_ENUM_INT = 5
    TYPE_ENUM_UINT = 6
    TYPE_ARRAY = 7
    TYPE_PTRREL = 8
    TYPE_PTR = 9
    TYPE_FLOAT = 10
    TYPE_CODE = 11
    TYPE_BOOL = 12
    TYPE_UINT = 13
    TYPE_INT = 14
    TYPE_UNKNOWN = 15
    TYPE_SPACEBASE = 16
    TYPE_VOID = 17


_METATYPE_NAMES = {
    TypeMetatype.TYPE_VOID: "void",
    TypeMetatype.TYPE_SPACEBASE: "spacebase",
    TypeMetatype.TYPE_UNKNOWN: "unknown",
    TypeMetatype.TYPE_INT: "int",
    TypeMetatype.TYPE_UINT: "uint",
    TypeMetatype.TYPE_BOOL: "bool",
    TypeMetatype.TYPE_CODE: "code",
    TypeMetatype.TYPE_FLOAT: "float",
    TypeMetatype.TYPE_PTR: "ptr",
    TypeMetatype.TYPE_PTRREL: "ptrrel",
    TypeMetatype.TYPE_ARRAY: "array",
    TypeMetatype.TYPE_ENUM_UINT: "enum_uint",
    TypeMetatype.TYPE_ENUM_INT: "enum_int",
    TypeMetatype.TYPE_STRUCT: "struct",
    TypeMetatype.TYPE_UNION: "union",
    TypeMetatype.TYPE_PARTIALENUM: "partenum",
    TypeMetatype.TYPE_PARTIALSTRUCT: "partstruct",
    TypeMetatype.TYPE_PARTIALUNION: "partunion",
}

_METATYPE_IDS = {name: value for value, name in _METATYPE_NAMES.items()}


def metatype_to_string(metatype: Any) -> str:
    """``metatype2string``: the marshaled name of a ``type_metatype``."""
    try:
        return _METATYPE_NAMES[TypeMetatype(int(metatype))]
    except (KeyError, ValueError):
        return _METATYPE_NAMES[TypeMetatype.TYPE_UNKNOWN]


def string_to_metatype(name: str) -> TypeMetatype:
    """``string2metatype``: the ``type_metatype`` encoded by a name."""
    if not name:
        return TypeMetatype.TYPE_UNKNOWN
    # The printer reports plain "enum" for both signed and unsigned enums.
    if name == "enum":
        return TypeMetatype.TYPE_ENUM_INT
    return _METATYPE_IDS.get(name, TypeMetatype.TYPE_UNKNOWN)


class OpCode(IntEnum):
    """``OpCode`` in ``opcodes.hh``; the values are part of the format."""

    CPUI_COPY = 1
    CPUI_LOAD = 2
    CPUI_STORE = 3
    CPUI_BRANCH = 4
    CPUI_CBRANCH = 5
    CPUI_BRANCHIND = 6
    CPUI_CALL = 7
    CPUI_CALLIND = 8
    CPUI_CALLOTHER = 9
    CPUI_RETURN = 10
    CPUI_INT_EQUAL = 11
    CPUI_INT_NOTEQUAL = 12
    CPUI_INT_SLESS = 13
    CPUI_INT_SLESSEQUAL = 14
    CPUI_INT_LESS = 15
    CPUI_INT_LESSEQUAL = 16
    CPUI_INT_ZEXT = 17
    CPUI_INT_SEXT = 18
    CPUI_INT_ADD = 19
    CPUI_INT_SUB = 20
    CPUI_INT_CARRY = 21
    CPUI_INT_SCARRY = 22
    CPUI_INT_SBORROW = 23
    CPUI_INT_2COMP = 24
    CPUI_INT_NEGATE = 25
    CPUI_INT_XOR = 26
    CPUI_INT_AND = 27
    CPUI_INT_OR = 28
    CPUI_INT_LEFT = 29
    CPUI_INT_RIGHT = 30
    CPUI_INT_SRIGHT = 31
    CPUI_INT_MULT = 32
    CPUI_INT_DIV = 33
    CPUI_INT_SDIV = 34
    CPUI_INT_REM = 35
    CPUI_INT_SREM = 36
    CPUI_BOOL_NEGATE = 37
    CPUI_BOOL_XOR = 38
    CPUI_BOOL_AND = 39
    CPUI_BOOL_OR = 40
    CPUI_FLOAT_EQUAL = 41
    CPUI_FLOAT_NOTEQUAL = 42
    CPUI_FLOAT_LESS = 43
    CPUI_FLOAT_LESSEQUAL = 44
    CPUI_FLOAT_NAN = 46
    CPUI_FLOAT_ADD = 47
    CPUI_FLOAT_DIV = 48
    CPUI_FLOAT_MULT = 49
    CPUI_FLOAT_SUB = 50
    CPUI_FLOAT_NEG = 51
    CPUI_FLOAT_ABS = 52
    CPUI_FLOAT_SQRT = 53
    CPUI_FLOAT_INT2FLOAT = 54
    CPUI_FLOAT_FLOAT2FLOAT = 55
    CPUI_FLOAT_TRUNC = 56
    CPUI_FLOAT_CEIL = 57
    CPUI_FLOAT_FLOOR = 58
    CPUI_FLOAT_ROUND = 59
    CPUI_MULTIEQUAL = 60
    CPUI_INDIRECT = 61
    CPUI_PIECE = 62
    CPUI_SUBPIECE = 63
    CPUI_CAST = 64
    CPUI_PTRADD = 65
    CPUI_PTRSUB = 66
    CPUI_SEGMENTOP = 67
    CPUI_CPOOLREF = 68
    CPUI_NEW = 69
    CPUI_INSERT = 70
    CPUI_ZPULL = 71
    CPUI_POPCOUNT = 72
    CPUI_LZCOUNT = 73
    CPUI_SPULL = 74
    CPUI_MAX = 75


_OPCODE_NAMES = {
    member.value: member.name[len("CPUI_"):]
    for member in OpCode
    if member.name.startswith("CPUI_")
}
_OPCODE_IDS = {name: value for value, name in _OPCODE_NAMES.items()}


def get_opname(opc: Any) -> str:
    """``get_opname``: the marshaled name of a p-code ``OpCode``."""
    try:
        return _OPCODE_NAMES[OpCode(int(opc))]
    except (KeyError, ValueError):
        return str(opc)


def get_opcode(name: str) -> int:
    """``get_opcode``: the ``OpCode`` value encoded by a name (0 when unknown)."""
    return _OPCODE_IDS.get(name, 0)


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


def xml_read_bool(text: str) -> bool:
    """``xml_readbool``: only the literal ``true`` counts as true."""
    return text.strip() == "true"


_NUMERIC_RE = re.compile(r"^[+-]?(?:0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*|0)")


def _read_integer_text(text: str) -> int:
    """Read an integer the way ``istringstream >> res`` does with base auto-detect.

    A leading ``0x`` selects hexadecimal and a leading ``0`` octal, anything
    else is decimal.  Trailing junk is ignored and a value that cannot be
    parsed at all reads as 0, matching the C++ extraction semantics.
    """
    match = _NUMERIC_RE.match((text or "").strip())
    if match is None:
        return 0
    literal = match.group(0)
    negative = literal.startswith("-")
    digits = literal.lstrip("+-")
    try:
        if digits[:2].lower() == "0x":
            value = int(digits, 16)
        elif (
            len(digits) > 1
            and digits[0] == "0"
            and all(character in "01234567" for character in digits[1:])
        ):
            value = int(digits, 8)
        else:
            value = int(digits, 10)
    except ValueError:  # pragma: no cover - defensive
        return 0
    return -value if negative else value


# ---------------------------------------------------------------------------
# Address spaces
# ---------------------------------------------------------------------------

#: Index of every space MiniGhidraPy models, in ``Architecture`` creation order.
SPACE_INDEX: dict[str, int] = {"const": 0, "unique": 1, "ram": 2, "register": 3}

#: Ghidra's ``PackedFormat::SPECIALSPACE_*`` codes.
SPECIAL_SPACE_CODE: dict[str, int] = {"stack": 0, "join": 1, "fspec": 2, "iop": 3}

_SPACE_NAMES = {index: name for name, index in SPACE_INDEX.items()}
_SPECIAL_SPACE_NAMES = {code: name for name, code in SPECIAL_SPACE_CODE.items()}


def space_name_of(space: Any) -> str:
    """The marshaled name of a space object (``AddrSpace::getName``)."""
    if space is None:
        return ""
    name = getattr(space, "name", None)
    if isinstance(name, str):
        return name
    space_attribute = getattr(space, "space", None)
    if isinstance(space_attribute, str):
        return space_attribute
    return str(space)


def space_index_of(space: Any) -> int:
    """``AddrSpace::getIndex`` for the spaces MiniGhidraPy models."""
    name = space_name_of(space)
    if name not in SPACE_INDEX:
        raise MarshalError(f"Unknown address space name: {name}")
    return SPACE_INDEX[name]


def is_formal_stack_space(space: Any) -> bool:
    """``AddrSpace::isFormalStackSpace``: only ``stack`` plays that role here."""
    return space_name_of(space) == "stack"


# ---------------------------------------------------------------------------
# Abstract codecs
# ---------------------------------------------------------------------------


class Encoder(ABC):
    """``Encoder`` in ``marshal.hh``: write hierarchical data to a stream."""

    @abstractmethod
    def open_element(self, elem_id: ElementId) -> None:
        """Open an element; attributes may be written until content starts."""

    @abstractmethod
    def close_element(self, elem_id: ElementId) -> None:
        """Close the most recently opened element."""

    @abstractmethod
    def write_bool(self, attrib_id: AttributeId, value: bool) -> None:
        """Write a boolean attribute (or the element content)."""

    @abstractmethod
    def write_signed_integer(self, attrib_id: AttributeId, value: int) -> None:
        """Write a signed integer attribute (or the element content)."""

    @abstractmethod
    def write_unsigned_integer(self, attrib_id: AttributeId, value: int) -> None:
        """Write an unsigned integer attribute (or the element content)."""

    @abstractmethod
    def write_string(self, attrib_id: AttributeId, value: str) -> None:
        """Write a string attribute (or the element content)."""

    def write_string_indexed(
        self, attrib_id: AttributeId, index: int, value: str
    ) -> None:
        """Write one attribute of an indexed family (``name1``, ``name2``, ...)."""
        raise MarshalError("write_string_indexed is not supported by this encoder")

    @abstractmethod
    def write_space(self, attrib_id: AttributeId, space: Any) -> None:
        """Write an address space attribute (or the element content)."""

    def write_opcode(self, attrib_id: AttributeId, opc: Any) -> None:
        """Write a p-code ``OpCode`` attribute."""
        raise MarshalError("write_opcode is not supported by this encoder")

    def write_datatype_meta(self, attrib_id: AttributeId, metatype: Any) -> None:
        """Write a ``type_metatype`` attribute."""
        raise MarshalError("write_datatype_meta is not supported by this encoder")


class Decoder(ABC):
    """``Decoder`` in ``marshal.hh``: read hierarchical data from a stream."""

    def __init__(self, space_manager: Any = None) -> None:
        self.space_manager = space_manager

    def ingest_stream(self, stream: Any) -> None:
        """Read a whole stream, preparing the decoder for traversal."""

    @abstractmethod
    def peek_element(self) -> int:
        """Return the id of the next element without opening it (0 if none)."""

    @abstractmethod
    def open_element(self, elem_id: Optional[ElementId] = None) -> int:
        """Open the next element, optionally insisting on a specific id."""

    @abstractmethod
    def close_element(self, id: int) -> None:
        """Close the element opened with the given id."""

    @abstractmethod
    def close_element_skipping(self, id: int) -> None:
        """Close the element with the given id, skipping any content."""

    @abstractmethod
    def rewind_attributes(self) -> None:
        """Restart attribute traversal for the current element."""

    @abstractmethod
    def get_next_attribute_id(self) -> int:
        """Return the id of the next attribute (0 when they are exhausted)."""

    @abstractmethod
    def get_indexed_attribute_id(self, attrib_id: AttributeId) -> int:
        """Reinterpret the current attribute as an indexed form of ``attrib_id``."""

    @abstractmethod
    def read_bool(self, attrib_id: Optional[AttributeId] = None) -> bool:
        """Read a boolean from the current attribute (or a specific one)."""

    @abstractmethod
    def read_signed_integer(self, attrib_id: Optional[AttributeId] = None) -> int:
        """Read a signed integer from the current attribute (or a specific one)."""

    def read_signed_integer_expect_string(
        self, expect: str, expectval: int, attrib_id: Optional[AttributeId] = None
    ) -> int:
        """Read an integer that may be encoded as a specific string."""
        if attrib_id is not None:
            value = self.read_string(attrib_id)
            if value == expect:
                return expectval
            return _read_integer_text(value)
        text = self.read_string()
        if text == expect:
            return expectval
        return _read_integer_text(text)

    @abstractmethod
    def read_unsigned_integer(self, attrib_id: Optional[AttributeId] = None) -> int:
        """Read an unsigned integer from the current attribute."""

    @abstractmethod
    def read_string(self, attrib_id: Optional[AttributeId] = None) -> str:
        """Read a string from the current attribute (or a specific one)."""

    @abstractmethod
    def read_space(self, attrib_id: Optional[AttributeId] = None) -> str:
        """Read an address space (returned as its name)."""

    def read_opcode(self, attrib_id: Optional[AttributeId] = None) -> int:
        """Read a p-code ``OpCode``."""
        raise DecoderError("read_opcode is not supported by this decoder")

    def read_datatype_meta(
        self, attrib_id: Optional[AttributeId] = None
    ) -> TypeMetatype:
        """Read a ``type_metatype``."""
        raise DecoderError("read_datatype_meta is not supported by this decoder")


# ---------------------------------------------------------------------------
# Unpacked XML
# ---------------------------------------------------------------------------


class XmlEncoder(Encoder):
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

    def write_string_indexed(
        self, attrib_id: AttributeId, index: int, value: str
    ) -> None:
        self._write(f" {attrib_id.name}{index + 1:d}")
        self._write('="' + xml_escape(value) + '"')

    def write_space(self, attrib_id: AttributeId, space: Any) -> None:
        name = space_name_of(space)
        if attrib_id == ATTRIB_CONTENT:
            self._content_prefix()
            self._write(xml_escape(name))
            self.tag_status = self.TAG_CONTENT
            return
        self._a_v(attrib_id, name)

    def write_opcode(self, attrib_id: AttributeId, opc: Any) -> None:
        name = get_opname(opc)
        if attrib_id == ATTRIB_CONTENT:
            self._content_prefix()
            self._write(xml_escape(name))
            self.tag_status = self.TAG_CONTENT
            return
        self._a_v(attrib_id, name)

    def write_datatype_meta(self, attrib_id: AttributeId, metatype: Any) -> None:
        name = metatype if isinstance(metatype, str) else metatype_to_string(metatype)
        if attrib_id == ATTRIB_CONTENT:
            self._content_prefix()
            self._write(xml_escape(name))
            self.tag_status = self.TAG_CONTENT
            return
        self._a_v(attrib_id, name)


#: The C++ class name, kept as an alias.
XmlEncode = XmlEncoder


class _XmlPosition:
    """One open XML element plus the index of its next child."""

    __slots__ = ("element", "children", "index")

    def __init__(self, element: Any) -> None:
        self.element = element
        self.children = list(element)
        self.index = 0


class XmlDecode(Decoder):
    """XML decoder, a port of ``XmlDecode`` in ``marshal.cc``."""

    def __init__(self, space_manager: Any = None, root: Any = None, scope: int = 0):
        super().__init__(space_manager)
        self.document: Optional[Any] = None
        self.root_element = root
        self.scope = scope
        self.element_stack: list[_XmlPosition] = []
        self.attribute_index = -1

    # -- traversal -------------------------------------------------------
    def ingest_stream(self, stream: Any) -> None:
        """``xml_tree``: parse the whole document and keep its root."""
        if isinstance(stream, (bytes, bytearray)):
            text = bytes(stream).decode("utf-8")
        elif isinstance(stream, str):
            text = stream
        else:
            text = stream.read()
            if isinstance(text, (bytes, bytearray)):
                text = text.decode("utf-8")
        self.document = ElementTree.fromstring(text)
        self.root_element = self.document

    def peek_element(self) -> int:
        if not self.element_stack:
            if self.root_element is None:
                return 0
            element = self.root_element
        else:
            top = self.element_stack[-1]
            if top.index >= len(top.children):
                return 0
            element = top.children[top.index]
        return lookup_element_id(element.tag, self.scope)

    def open_element(self, elem_id: Optional[ElementId] = None) -> int:
        if not self.element_stack:
            if self.root_element is None:
                if elem_id is not None:
                    raise DecoderError(
                        f"Expecting <{elem_id.name}> but reached end of document"
                    )
                return 0
            element = self.root_element
            self.root_element = None  # Only open once
        else:
            top = self.element_stack[-1]
            if top.index >= len(top.children):
                if elem_id is not None:
                    raise DecoderError(
                        f"Expecting <{elem_id.name}> but no remaining children "
                        "in current element"
                    )
                return 0
            element = top.children[top.index]
            top.index += 1
        if elem_id is not None and element.tag != elem_id.name:
            raise DecoderError(
                f"Expecting <{elem_id.name}> but got <{element.tag}>"
            )
        self.element_stack.append(_XmlPosition(element))
        self.attribute_index = -1
        return elem_id.id if elem_id is not None else lookup_element_id(
            element.tag, self.scope
        )

    def close_element(self, id: int) -> None:
        self._pop(id)

    def close_element_skipping(self, id: int) -> None:
        self._pop(id)

    def _pop(self, id: int) -> None:
        position = self.element_stack.pop()
        if lookup_element_id(position.element.tag, self.scope) != id:
            raise DecoderError(
                f"Trying to close <{position.element.tag}> with mismatching id"
            )
        self.attribute_index = 1000  # No additional attributes can be read

    # -- attributes ------------------------------------------------------
    @staticmethod
    def _attributes(element: Any) -> list[tuple[str, str]]:
        return list(element.attrib.items())

    def _current_element(self) -> Any:
        if not self.element_stack:
            raise DecoderError("No open element")
        return self.element_stack[-1].element

    def rewind_attributes(self) -> None:
        self.attribute_index = -1

    def get_next_attribute_id(self) -> int:
        attributes = self._attributes(self._current_element())
        next_index = self.attribute_index + 1
        if next_index < len(attributes):
            self.attribute_index = next_index
            return lookup_attribute_id(attributes[next_index][0], self.scope)
        return 0

    def get_indexed_attribute_id(self, attrib_id: AttributeId) -> int:
        attributes = self._attributes(self._current_element())
        if self.attribute_index < 0 or self.attribute_index >= len(attributes):
            return ATTRIB_UNKNOWN.id
        name = attributes[self.attribute_index][0]
        base = attrib_id.name
        if not name.startswith(base):
            return ATTRIB_UNKNOWN.id
        suffix = name[len(base):]
        if not suffix.isdigit():
            return ATTRIB_UNKNOWN.id
        value = int(suffix)
        if value == 0:
            raise MarshalError(f"Bad indexed attribute: {base}")
        return attrib_id.id + (value - 1)

    def _find_matching_attribute(self, attrib_name: str) -> str:
        for name, value in self._attributes(self._current_element()):
            if name == attrib_name:
                return value
        raise DecoderError(f"Attribute missing: {attrib_name}")

    def _value(self, attrib_id: Optional[AttributeId]) -> str:
        element = self._current_element()
        if attrib_id is None:
            return self._attributes(element)[self.attribute_index][1]
        if attrib_id == ATTRIB_CONTENT:
            return element.text or ""
        return self._find_matching_attribute(attrib_id.name)

    # -- typed reads -----------------------------------------------------
    def read_bool(self, attrib_id: Optional[AttributeId] = None) -> bool:
        return xml_read_bool(self._value(attrib_id))

    def read_signed_integer(self, attrib_id: Optional[AttributeId] = None) -> int:
        return _read_integer_text(self._value(attrib_id))

    def read_unsigned_integer(self, attrib_id: Optional[AttributeId] = None) -> int:
        return _read_integer_text(self._value(attrib_id))

    def read_string(self, attrib_id: Optional[AttributeId] = None) -> str:
        return self._value(attrib_id)

    def read_space(self, attrib_id: Optional[AttributeId] = None) -> str:
        name = self._value(attrib_id)
        if name not in SPACE_INDEX and name not in SPECIAL_SPACE_CODE:
            raise DecoderError(f"Unknown address space name: {name}")
        return name

    def read_opcode(self, attrib_id: Optional[AttributeId] = None) -> int:
        opc = get_opcode(self._value(attrib_id))
        if opc == 0:
            raise DecoderError("Bad encoded OpCode")
        return opc

    def read_datatype_meta(
        self, attrib_id: Optional[AttributeId] = None
    ) -> TypeMetatype:
        return string_to_metatype(self._value(attrib_id))


# ---------------------------------------------------------------------------
# Packed binary format
# ---------------------------------------------------------------------------


class PackedFormat:
    """Protocol constants for :class:`PackedEncoder`/:class:`PackedDecode`.

    All bytes are non-zero.  ``01xiiiii`` starts an element, ``10xiiiii`` ends
    one and ``11xiiiii`` starts an attribute, where ``iiiii`` are the first five
    bits of the id; ``x=1`` means the next byte (``1iiiiiii``) holds seven more.
    An attribute is followed by a type byte ``ttttllll``: ``tttt`` is the type
    code and ``llll`` the number of 7-bit bytes encoding its integer payload
    (a length code of 0 encodes the value 0 with no following bytes).  Strings
    store their length as that integer, then UTF-8 data.
    """

    HEADER_MASK = 0xC0
    ELEMENT_START = 0x40
    ELEMENT_END = 0x80
    ATTRIBUTE = 0xC0
    HEADEREXTEND_MASK = 0x20
    ELEMENTID_MASK = 0x1F
    RAWDATA_MASK = 0x7F
    RAWDATA_BITSPERBYTE = 7
    RAWDATA_MARKER = 0x80
    TYPECODE_SHIFT = 4
    LENGTHCODE_MASK = 0xF
    TYPECODE_BOOLEAN = 1
    TYPECODE_SIGNEDINT_POSITIVE = 2
    TYPECODE_SIGNEDINT_NEGATIVE = 3
    TYPECODE_UNSIGNEDINT = 4
    TYPECODE_ADDRESSSPACE = 5
    TYPECODE_SPECIALSPACE = 6
    TYPECODE_STRING = 7
    SPECIALSPACE_STACK = 0
    SPECIALSPACE_JOIN = 1
    SPECIALSPACE_FSPEC = 2
    SPECIALSPACE_IOP = 3
    SPECIALSPACE_SPACEBASE = 4


class PackedEncoder(Encoder):
    """Byte-based encoder, a port of ``PackedEncode`` in ``marshal.cc``."""

    def __init__(self, stream: Optional[BinaryIO] = None) -> None:
        self.stream = stream
        self.data = bytearray()

    # -- low level -------------------------------------------------------
    def _put(self, value: int) -> None:
        self.data.append(value & 0xFF)
        if self.stream is not None:
            self.stream.write(bytes((value & 0xFF,)))

    def _put_bytes(self, payload: bytes) -> None:
        self.data.extend(payload)
        if self.stream is not None:
            self.stream.write(payload)

    def get_bytes(self) -> bytes:
        """Everything encoded so far."""
        return bytes(self.data)

    def write_header(self, header: int, id: int) -> None:
        """Write an element or attribute header, extending it when needed."""
        if id > 0x1F:
            header |= PackedFormat.HEADEREXTEND_MASK
            header |= id >> PackedFormat.RAWDATA_BITSPERBYTE
            extend_byte = (id & PackedFormat.RAWDATA_MASK) | PackedFormat.RAWDATA_MARKER
            self._put(header)
            self._put(extend_byte)
        else:
            self._put(header | id)

    def write_integer(self, type_byte: int, val: int) -> None:
        """Write a non-negative integer after its type/length byte."""
        if val < 0:
            raise MarshalError("PackedEncode: negative value in writeInteger")
        if val == 0:
            len_code = 0
            sa = -1
        elif val < 0x8_0000_0000:
            if val < 0x20_0000:
                if val < 0x80:
                    len_code, sa = 1, 0
                elif val < 0x4000:
                    len_code, sa = 2, PackedFormat.RAWDATA_BITSPERBYTE
                else:
                    len_code, sa = 3, 2 * PackedFormat.RAWDATA_BITSPERBYTE
            elif val < 0x1000_0000:
                len_code, sa = 4, 3 * PackedFormat.RAWDATA_BITSPERBYTE
            else:
                len_code, sa = 5, 4 * PackedFormat.RAWDATA_BITSPERBYTE
        elif val < 0x2_0000_0000_0000:
            if val < 0x400_0000_0000:
                len_code, sa = 6, 5 * PackedFormat.RAWDATA_BITSPERBYTE
            else:
                len_code, sa = 7, 6 * PackedFormat.RAWDATA_BITSPERBYTE
        elif val < 0x1_0000_0000_0000_00:
            len_code, sa = 8, 7 * PackedFormat.RAWDATA_BITSPERBYTE
        elif val < 0x8000_0000_0000_0000:
            len_code, sa = 9, 8 * PackedFormat.RAWDATA_BITSPERBYTE
        else:
            len_code, sa = 10, 9 * PackedFormat.RAWDATA_BITSPERBYTE
        self._put(type_byte | len_code)
        while sa >= 0:
            piece = (val >> sa) & PackedFormat.RAWDATA_MASK
            self._put(piece | PackedFormat.RAWDATA_MARKER)
            sa -= PackedFormat.RAWDATA_BITSPERBYTE

    # -- Encoder API -----------------------------------------------------
    def open_element(self, elem_id: ElementId) -> None:
        self.write_header(PackedFormat.ELEMENT_START, elem_id.id)

    def close_element(self, elem_id: ElementId) -> None:
        self.write_header(PackedFormat.ELEMENT_END, elem_id.id)

    def write_bool(self, attrib_id: AttributeId, value: bool) -> None:
        self.write_header(PackedFormat.ATTRIBUTE, attrib_id.id)
        type_byte = PackedFormat.TYPECODE_BOOLEAN << PackedFormat.TYPECODE_SHIFT
        self._put(type_byte | (1 if value else 0))

    def write_signed_integer(self, attrib_id: AttributeId, value: int) -> None:
        self.write_header(PackedFormat.ATTRIBUTE, attrib_id.id)
        if value < 0:
            type_byte = (
                PackedFormat.TYPECODE_SIGNEDINT_NEGATIVE << PackedFormat.TYPECODE_SHIFT
            )
            num = -value
        else:
            type_byte = (
                PackedFormat.TYPECODE_SIGNEDINT_POSITIVE << PackedFormat.TYPECODE_SHIFT
            )
            num = value
        self.write_integer(type_byte, num)

    def write_unsigned_integer(self, attrib_id: AttributeId, value: int) -> None:
        self.write_header(PackedFormat.ATTRIBUTE, attrib_id.id)
        self.write_integer(
            PackedFormat.TYPECODE_UNSIGNEDINT << PackedFormat.TYPECODE_SHIFT, value
        )

    def _write_string(self, attrib_id: AttributeId, value: str) -> None:
        payload = value.encode("utf-8")
        if b"\0" in payload:
            raise MarshalError("PackedEncode: string with null character")
        self.write_header(PackedFormat.ATTRIBUTE, attrib_id.id)
        self.write_integer(
            PackedFormat.TYPECODE_STRING << PackedFormat.TYPECODE_SHIFT, len(payload)
        )
        self._put_bytes(payload)

    def write_string(self, attrib_id: AttributeId, value: str) -> None:
        self._write_string(attrib_id, value)

    def write_string_indexed(
        self, attrib_id: AttributeId, index: int, value: str
    ) -> None:
        self._write_string(AttributeId(attrib_id.name, attrib_id.id + index), value)

    def write_space(self, attrib_id: AttributeId, space: Any) -> None:
        name = space_name_of(space)
        self.write_header(PackedFormat.ATTRIBUTE, attrib_id.id)
        shift = PackedFormat.TYPECODE_SHIFT
        if name in SPECIAL_SPACE_CODE:
            special = PackedFormat.TYPECODE_SPECIALSPACE << shift
            self._put(special | SPECIAL_SPACE_CODE[name])
        elif name == "spacebase":
            # A secondary register-offset space (AddrSpace::isFormalStackSpace
            # is false for it, so Ghidra uses its own special code).
            self._put(
                (PackedFormat.TYPECODE_SPECIALSPACE << shift)
                | PackedFormat.SPECIALSPACE_SPACEBASE
            )
        else:
            self.write_integer(
                PackedFormat.TYPECODE_ADDRESSSPACE << shift, space_index_of(space)
            )

    def write_opcode(self, attrib_id: AttributeId, opc: Any) -> None:
        self.write_header(PackedFormat.ATTRIBUTE, attrib_id.id)
        value = int(opc) if not isinstance(opc, str) else get_opcode(opc)
        self.write_integer(
            PackedFormat.TYPECODE_SIGNEDINT_POSITIVE << PackedFormat.TYPECODE_SHIFT,
            value,
        )

    def write_datatype_meta(self, attrib_id: AttributeId, metatype: Any) -> None:
        self.write_header(PackedFormat.ATTRIBUTE, attrib_id.id)
        value = (
            int(string_to_metatype(metatype))
            if isinstance(metatype, str)
            else int(metatype)
        )
        self.write_integer(
            PackedFormat.TYPECODE_SIGNEDINT_POSITIVE << PackedFormat.TYPECODE_SHIFT,
            value,
        )


#: The C++ class name, kept as an alias.
PackedEncode = PackedEncoder


class _PackedPosition:
    """An iterator into the packed stream (``PackedDecode::Position``)."""

    __slots__ = ("index",)

    def __init__(self, index: int = 0) -> None:
        self.index = index

    def copy(self) -> "_PackedPosition":
        return _PackedPosition(self.index)


class PackedDecode(Decoder):
    """Byte-based decoder, a port of ``PackedDecode`` in ``marshal.cc``."""

    def __init__(self, space_manager: Any = None) -> None:
        super().__init__(space_manager)
        self.stream_data = bytearray()
        self.start_pos = _PackedPosition()
        self.cur_pos = _PackedPosition()
        self.end_pos = _PackedPosition()
        self.attribute_read = True

    # -- ingestion -------------------------------------------------------
    def ingest_stream(self, stream: Any) -> None:
        """Read the encoded bytes and terminate them with an element end."""
        if isinstance(stream, (bytes, bytearray)):
            payload = bytes(stream)
        elif isinstance(stream, str):
            payload = stream.encode("latin-1")
        else:
            payload = stream.read()
            if isinstance(payload, str):
                payload = payload.encode("latin-1")
        if not payload:
            raise DecoderError("Ended ingestion without any input")
        self.stream_data = bytearray(payload)
        self.stream_data.append(PackedFormat.ELEMENT_END)
        self.start_pos = _PackedPosition(0)
        self.cur_pos = _PackedPosition(0)
        self.end_pos = _PackedPosition(0)
        self.attribute_read = True

    # -- low level -------------------------------------------------------
    def _get_byte(self, pos: _PackedPosition) -> int:
        if pos.index >= len(self.stream_data):
            raise DecoderError("Unexpected end of stream")
        return self.stream_data[pos.index]

    def _get_byte_plus1(self, pos: _PackedPosition) -> int:
        if pos.index + 1 >= len(self.stream_data):
            raise DecoderError("Unexpected end of stream")
        return self.stream_data[pos.index + 1]

    def _get_next_byte(self, pos: _PackedPosition) -> int:
        value = self._get_byte(pos)
        pos.index += 1
        return value

    def _advance_position(self, pos: _PackedPosition, skip: int) -> None:
        if pos.index + skip > len(self.stream_data):
            raise DecoderError("Unexpected end of stream")
        pos.index += skip

    def _read_integer(self, length: int) -> int:
        result = 0
        while length > 0:
            result <<= PackedFormat.RAWDATA_BITSPERBYTE
            result |= self._get_next_byte(self.cur_pos) & PackedFormat.RAWDATA_MASK
            length -= 1
        return result

    @staticmethod
    def _read_length_code(type_byte: int) -> int:
        return type_byte & PackedFormat.LENGTHCODE_MASK

    def _read_id(self, header: int, pos: _PackedPosition) -> int:
        """Decode an element/attribute id from a header byte at ``pos``."""
        id = header & PackedFormat.ELEMENTID_MASK
        if (header & PackedFormat.HEADEREXTEND_MASK) != 0:
            id <<= PackedFormat.RAWDATA_BITSPERBYTE
            id |= self._get_byte_plus1(pos) & PackedFormat.RAWDATA_MASK
        return id

    def _consume_id(self, header: int, pos: _PackedPosition) -> int:
        """Decode an id whose header byte has already been consumed."""
        id = header & PackedFormat.ELEMENTID_MASK
        if (header & PackedFormat.HEADEREXTEND_MASK) != 0:
            id <<= PackedFormat.RAWDATA_BITSPERBYTE
            id |= self._get_next_byte(pos) & PackedFormat.RAWDATA_MASK
        return id

    # -- attribute scanning ---------------------------------------------
    def _find_matching_attribute(self, attrib_id: AttributeId) -> None:
        self.cur_pos = self.start_pos.copy()
        while True:
            header = self._get_byte(self.cur_pos)
            if (header & PackedFormat.HEADER_MASK) != PackedFormat.ATTRIBUTE:
                break
            if attrib_id.id == self._read_id(header, self.cur_pos):
                return
            self._skip_attribute()
        raise DecoderError(f"Attribute {attrib_id.name} is not present")

    def _skip_attribute(self) -> None:
        header = self._get_next_byte(self.cur_pos)
        if (header & PackedFormat.HEADEREXTEND_MASK) != 0:
            self._get_next_byte(self.cur_pos)
        type_byte = self._get_next_byte(self.cur_pos)
        self._skip_attribute_remaining(type_byte)

    def _skip_attribute_remaining(self, type_byte: int) -> None:
        attrib_type = type_byte >> PackedFormat.TYPECODE_SHIFT
        if attrib_type in (
            PackedFormat.TYPECODE_BOOLEAN,
            PackedFormat.TYPECODE_SPECIALSPACE,
        ):
            return
        length = self._read_length_code(type_byte)
        if attrib_type == PackedFormat.TYPECODE_STRING:
            length = self._read_integer(length)
        self._advance_position(self.cur_pos, length)

    # -- traversal -------------------------------------------------------
    def peek_element(self) -> int:
        header = self._get_byte(self.end_pos)
        if (header & PackedFormat.HEADER_MASK) != PackedFormat.ELEMENT_START:
            return 0
        return self._read_id(header, self.end_pos)

    def open_element(self, elem_id: Optional[ElementId] = None) -> int:
        id = self._open_element()
        if elem_id is None:
            return id
        if id != elem_id.id:
            if id == 0:
                raise DecoderError(
                    f"Expecting <{elem_id.name}> but did not scan an element"
                )
            raise DecoderError(f"Expecting <{elem_id.name}> but id did not match")
        return id

    def _open_element(self) -> int:
        header = self._get_byte(self.end_pos)
        if (header & PackedFormat.HEADER_MASK) != PackedFormat.ELEMENT_START:
            return 0
        self._get_next_byte(self.end_pos)
        id = self._consume_id(header, self.end_pos)
        self.start_pos = self.end_pos.copy()
        self.cur_pos = self.end_pos.copy()
        header = self._get_byte(self.cur_pos)
        while (header & PackedFormat.HEADER_MASK) == PackedFormat.ATTRIBUTE:
            self._skip_attribute()
            header = self._get_byte(self.cur_pos)
        self.end_pos = self.cur_pos.copy()
        self.cur_pos = self.start_pos.copy()
        self.attribute_read = True
        return id

    def close_element(self, id: int) -> None:
        header = self._get_next_byte(self.end_pos)
        if (header & PackedFormat.HEADER_MASK) != PackedFormat.ELEMENT_END:
            raise DecoderError("Expecting element close")
        close_id = self._consume_id(header, self.end_pos)
        if id != close_id:
            raise DecoderError("Did not see expected closing element")

    def close_element_skipping(self, id: int) -> None:
        id_stack = [id]
        while id_stack:
            header = self._get_byte(self.end_pos) & PackedFormat.HEADER_MASK
            if header == PackedFormat.ELEMENT_END:
                self.close_element(id_stack[-1])
                id_stack.pop()
            elif header == PackedFormat.ELEMENT_START:
                id_stack.append(self._open_element())
            else:
                raise DecoderError("Corrupt stream")

    def rewind_attributes(self) -> None:
        self.cur_pos = self.start_pos.copy()
        self.attribute_read = True

    def get_next_attribute_id(self) -> int:
        if not self.attribute_read:
            self._skip_attribute()
        header = self._get_byte(self.cur_pos)
        if (header & PackedFormat.HEADER_MASK) != PackedFormat.ATTRIBUTE:
            return 0
        id = self._read_id(header, self.cur_pos)
        self.attribute_read = False
        return id

    def get_indexed_attribute_id(self, attrib_id: AttributeId) -> int:
        # PackedDecode never needs to reinterpret an attribute.
        return ATTRIB_UNKNOWN.id

    # -- typed reads -----------------------------------------------------
    def _read_type_byte(self) -> int:
        header = self._get_next_byte(self.cur_pos)
        if (header & PackedFormat.HEADEREXTEND_MASK) != 0:
            self._get_next_byte(self.cur_pos)
        return self._get_next_byte(self.cur_pos)

    def read_bool(self, attrib_id: Optional[AttributeId] = None) -> bool:
        if attrib_id is not None:
            self._find_matching_attribute(attrib_id)
            result = self.read_bool()
            self.cur_pos = self.start_pos.copy()
            return result
        type_byte = self._read_type_byte()
        self.attribute_read = True
        if (type_byte >> PackedFormat.TYPECODE_SHIFT) != PackedFormat.TYPECODE_BOOLEAN:
            raise DecoderError("Expecting boolean attribute")
        return (type_byte & PackedFormat.LENGTHCODE_MASK) != 0

    def read_signed_integer(self, attrib_id: Optional[AttributeId] = None) -> int:
        if attrib_id is not None:
            self._find_matching_attribute(attrib_id)
            result = self.read_signed_integer()
            self.cur_pos = self.start_pos.copy()
            return result
        type_byte = self._read_type_byte()
        type_code = type_byte >> PackedFormat.TYPECODE_SHIFT
        if type_code == PackedFormat.TYPECODE_SIGNEDINT_POSITIVE:
            result = self._read_integer(self._read_length_code(type_byte))
        elif type_code == PackedFormat.TYPECODE_SIGNEDINT_NEGATIVE:
            result = -self._read_integer(self._read_length_code(type_byte))
        else:
            self._skip_attribute_remaining(type_byte)
            self.attribute_read = True
            raise DecoderError("Expecting signed integer attribute")
        self.attribute_read = True
        return result

    def read_unsigned_integer(self, attrib_id: Optional[AttributeId] = None) -> int:
        if attrib_id is not None:
            self._find_matching_attribute(attrib_id)
            result = self.read_unsigned_integer()
            self.cur_pos = self.start_pos.copy()
            return result
        type_byte = self._read_type_byte()
        type_code = type_byte >> PackedFormat.TYPECODE_SHIFT
        if type_code != PackedFormat.TYPECODE_UNSIGNEDINT:
            self._skip_attribute_remaining(type_byte)
            self.attribute_read = True
            raise DecoderError("Expecting unsigned integer attribute")
        result = self._read_integer(self._read_length_code(type_byte))
        self.attribute_read = True
        return result

    def read_string(self, attrib_id: Optional[AttributeId] = None) -> str:
        if attrib_id is not None:
            self._find_matching_attribute(attrib_id)
            result = self.read_string()
            self.cur_pos = self.start_pos.copy()
            return result
        type_byte = self._read_type_byte()
        if (type_byte >> PackedFormat.TYPECODE_SHIFT) != PackedFormat.TYPECODE_STRING:
            self._skip_attribute_remaining(type_byte)
            self.attribute_read = True
            raise DecoderError("Expecting string attribute")
        length = self._read_integer(self._read_length_code(type_byte))
        self.attribute_read = True
        start = self.cur_pos.index
        self._advance_position(self.cur_pos, length)
        return bytes(self.stream_data[start:start + length]).decode("utf-8")

    def read_space(self, attrib_id: Optional[AttributeId] = None) -> str:
        if attrib_id is not None:
            self._find_matching_attribute(attrib_id)
            result = self.read_space()
            self.cur_pos = self.start_pos.copy()
            return result
        type_byte = self._read_type_byte()
        type_code = type_byte >> PackedFormat.TYPECODE_SHIFT
        self.attribute_read = True
        if type_code == PackedFormat.TYPECODE_SPECIALSPACE:
            code = type_byte & PackedFormat.LENGTHCODE_MASK
            if code == PackedFormat.SPECIALSPACE_STACK:
                return "stack"
            if code == PackedFormat.SPECIALSPACE_JOIN:
                return "join"
            if code == PackedFormat.SPECIALSPACE_FSPEC:
                return "fspec"
            if code == PackedFormat.SPECIALSPACE_IOP:
                return "iop"
            return "spacebase"
        if type_code != PackedFormat.TYPECODE_ADDRESSSPACE:
            self._skip_attribute_remaining(type_byte)
            raise DecoderError("Expecting address space attribute")
        index = self._read_integer(self._read_length_code(type_byte))
        if index not in _SPACE_NAMES:
            raise DecoderError(f"Unknown address space index: {index}")
        return _SPACE_NAMES[index]

    def read_opcode(self, attrib_id: Optional[AttributeId] = None) -> int:
        return int(self.read_signed_integer(attrib_id))

    def read_datatype_meta(
        self, attrib_id: Optional[AttributeId] = None
    ) -> TypeMetatype:
        return TypeMetatype(int(self.read_signed_integer(attrib_id)))


def iter_elements(decoder: Decoder) -> Iterable[int]:
    """Yield every element id in the current subtree (helper for tests)."""
    while True:
        elem = decoder.peek_element()
        if elem == 0:
            return
        decoder.open_element()
        yield elem
        for child in iter_elements(decoder):
            yield child
        decoder.close_element(elem)
