"""Tests for the marshaling layer: ``PackedEncode``/``PackedDecode``/``XmlDecode``."""

from __future__ import annotations

import io
import unittest

from minighidra.marshal import (
    ATTRIB_COLOR,
    ATTRIB_CONTENT,
    ATTRIB_ID,
    ATTRIB_INDENT,
    ATTRIB_OFF,
    ATTRIB_OPEN,
    ATTRIB_SPACE,
    ATTRIB_UNKNOWN,
    ELEM_BREAK,
    ELEM_CLANG_DOCUMENT,
    ELEM_COMMENT,
    ELEM_FUNCTION,
    ELEM_SYNTAX,
    ELEM_UNKNOWN,
    ELEM_VARIABLE,
    AttributeId,
    DecoderError,
    ElementId,
    MarshalError,
    OpCode,
    PackedDecode,
    PackedEncoder,
    PackedFormat,
    TypeMetatype,
    XmlDecode,
    XmlEncoder,
    get_opcode,
    get_opname,
    lookup_attribute_id,
    lookup_element_id,
    metatype_to_string,
    space_index_of,
    string_to_metatype,
    xml_escape,
    xml_read_bool,
)


class _Space:
    """Stand-in for ``AddrSpace``: only the name is marshaled here."""

    def __init__(self, name: str) -> None:
        self.name = name


def _encode(write) -> bytes:
    encoder = PackedEncoder()
    write(encoder)
    return encoder.get_bytes()


class PackedHeaderTests(unittest.TestCase):
    def test_short_ids_fit_in_one_byte(self) -> None:
        data = _encode(lambda enc: enc.open_element(ELEM_SYNTAX))
        self.assertEqual(data, bytes([PackedFormat.ELEMENT_START | 24]))
        self.assertEqual(data[0] & PackedFormat.HEADER_MASK, PackedFormat.ELEMENT_START)

    def test_long_ids_use_the_extension_byte(self) -> None:
        bitfield = ElementId("bitfield", 289)
        data = _encode(lambda enc: enc.open_element(bitfield))
        self.assertEqual(len(data), 2)
        self.assertNotEqual(data[0] & PackedFormat.HEADEREXTEND_MASK, 0)
        self.assertNotEqual(data[1] & PackedFormat.RAWDATA_MARKER, 0)
        decoded = (data[0] & PackedFormat.ELEMENTID_MASK) << 7
        decoded |= data[1] & PackedFormat.RAWDATA_MASK
        self.assertEqual(decoded, 289)

    def test_close_and_attribute_headers(self) -> None:
        data = _encode(lambda enc: enc.close_element(ELEM_SYNTAX))
        self.assertEqual(data[0] & PackedFormat.HEADER_MASK, PackedFormat.ELEMENT_END)
        data = _encode(lambda enc: enc.write_bool(ATTRIB_COLOR, True))
        self.assertEqual(data[0] & PackedFormat.HEADER_MASK, PackedFormat.ATTRIBUTE)

    def test_no_byte_is_ever_zero(self) -> None:
        def write(encoder: PackedEncoder) -> None:
            encoder.open_element(ELEM_CLANG_DOCUMENT)
            encoder.open_element(ELEM_VARIABLE)
            encoder.write_bool(ATTRIB_COLOR, False)
            encoder.write_signed_integer(ATTRIB_ID, 0)
            encoder.write_unsigned_integer(ATTRIB_OFF, 0)
            encoder.write_string(ATTRIB_CONTENT, "")
            encoder.write_space(ATTRIB_SPACE, _Space("stack"))
            encoder.close_element(ELEM_VARIABLE)
            encoder.close_element(ELEM_CLANG_DOCUMENT)

        data = _encode(write)
        self.assertTrue(data)
        self.assertNotIn(0, data)


class PackedIntegerTests(unittest.TestCase):
    def round_trip(self, value: int, signed: bool = True) -> int:
        def write(encoder: PackedEncoder) -> None:
            encoder.open_element(ELEM_SYNTAX)
            if signed:
                encoder.write_signed_integer(ATTRIB_ID, value)
            else:
                encoder.write_unsigned_integer(ATTRIB_ID, value)
            encoder.close_element(ELEM_SYNTAX)

        decoder = PackedDecode()
        decoder.ingest_stream(_encode(write))
        decoder.open_element()
        if signed:
            return decoder.read_signed_integer(ATTRIB_ID)
        return decoder.read_unsigned_integer(ATTRIB_ID)

    def test_zero_needs_no_data_byte(self) -> None:
        data = _encode(lambda enc: enc.write_signed_integer(ATTRIB_ID, 0))
        self.assertEqual(len(data), 2)  # header plus type byte, length code 0
        self.assertEqual(data[1] & PackedFormat.LENGTHCODE_MASK, 0)
        self.assertEqual(self.round_trip(0), 0)

    def test_length_codes_follow_the_magnitude(self) -> None:
        expected = {
            1: 1,
            0x7F: 1,
            0x80: 2,
            0x3FFF: 2,
            0x4000: 3,
            0x1FFFFF: 3,
            0x200000: 4,
            0xFFFFFFF: 4,
            0x10000000: 5,
            0x7FFFFFFFF: 5,
            0x800000000: 6,
            0x3FFFFFFFFFF: 6,
            0x40000000000: 7,
            0x1FFFFFFFFFFFF: 7,
            0x2000000000000: 8,
            0xFFFFFFFFFFFFF: 8,
            0x100000000000000: 9,
        }
        for value, length_code in expected.items():
            data = _encode(
                lambda enc, value=value: enc.write_unsigned_integer(ATTRIB_ID, value)
            )
            self.assertEqual(
                data[1] & PackedFormat.LENGTHCODE_MASK,
                length_code,
                f"length code of {value:#x}",
            )
            self.assertEqual(len(data), 2 + length_code)
            self.assertEqual(self.round_trip(value, signed=False), value)

    def test_negative_values_are_stored_negated(self) -> None:
        data = _encode(lambda enc: enc.write_signed_integer(ATTRIB_ID, -5))
        self.assertEqual(
            data[1] >> PackedFormat.TYPECODE_SHIFT,
            PackedFormat.TYPECODE_SIGNEDINT_NEGATIVE,
        )
        self.assertEqual(self.round_trip(-5), -5)
        self.assertEqual(self.round_trip(-(2**40)), -(2**40))

    def test_positive_values_use_the_positive_type_code(self) -> None:
        data = _encode(lambda enc: enc.write_signed_integer(ATTRIB_ID, 5))
        self.assertEqual(
            data[1] >> PackedFormat.TYPECODE_SHIFT,
            PackedFormat.TYPECODE_SIGNEDINT_POSITIVE,
        )

    def test_write_integer_rejects_negatives(self) -> None:
        encoder = PackedEncoder()
        with self.assertRaises(MarshalError):
            encoder.write_integer(PackedFormat.TYPECODE_UNSIGNEDINT << 4, -1)


class PackedValueTests(unittest.TestCase):
    def test_boolean_is_carried_by_the_type_byte(self) -> None:
        for value in (False, True):
            data = _encode(lambda enc, v=value: enc.write_bool(ATTRIB_ID, v))
            self.assertEqual(len(data), 2)  # header plus type byte, no data
            # An id above 0x1f extends the header but never adds data bytes.
            extended = _encode(lambda enc, v=value: enc.write_bool(ATTRIB_COLOR, v))
            self.assertEqual(len(extended), 3)
            self.assertEqual(
                data[1] >> PackedFormat.TYPECODE_SHIFT, PackedFormat.TYPECODE_BOOLEAN
            )
            self.assertEqual(bool(data[1] & PackedFormat.LENGTHCODE_MASK), value)

    def test_strings_are_utf8_with_a_length_prefix(self) -> None:
        data = _encode(lambda enc: enc.write_string(ATTRIB_CONTENT, "ñandú"))
        payload = "ñandú".encode("utf-8")
        self.assertEqual(data[-len(payload):], payload)
        self.assertEqual(
            data[1] >> PackedFormat.TYPECODE_SHIFT, PackedFormat.TYPECODE_STRING
        )

    def test_string_with_a_null_character_is_rejected(self) -> None:
        encoder = PackedEncoder()
        encoder.open_element(ELEM_SYNTAX)
        with self.assertRaises(MarshalError):
            encoder.write_string(ATTRIB_CONTENT, "a\0b")

    def test_indexed_strings_shift_the_attribute_id(self) -> None:
        base = AttributeId("param", 5)

        def write(encoder: PackedEncoder) -> None:
            encoder.open_element(ELEM_SYNTAX)
            encoder.write_string_indexed(base, 0, "first")
            encoder.write_string_indexed(base, 1, "second")
            encoder.close_element(ELEM_SYNTAX)

        decoder = PackedDecode()
        decoder.ingest_stream(_encode(write))
        decoder.open_element()
        self.assertEqual(decoder.get_next_attribute_id(), 5)
        self.assertEqual(decoder.read_string(), "first")
        self.assertEqual(decoder.get_next_attribute_id(), 6)
        self.assertEqual(decoder.read_string(), "second")
        self.assertEqual(decoder.get_next_attribute_id(), 0)

    def test_special_spaces_use_a_single_type_byte(self) -> None:
        for name, code in (
            ("stack", PackedFormat.SPECIALSPACE_STACK),
            ("join", PackedFormat.SPECIALSPACE_JOIN),
            ("fspec", PackedFormat.SPECIALSPACE_FSPEC),
            ("iop", PackedFormat.SPECIALSPACE_IOP),
        ):
            data = _encode(
                lambda enc, nm=name: enc.write_space(ATTRIB_SPACE, _Space(nm))
            )
            self.assertEqual(len(data), 2)
            self.assertEqual(
                data[1] >> PackedFormat.TYPECODE_SHIFT,
                PackedFormat.TYPECODE_SPECIALSPACE,
            )
            self.assertEqual(data[1] & PackedFormat.LENGTHCODE_MASK, code)

    def test_ordinary_spaces_are_marshaled_by_index(self) -> None:
        data = _encode(lambda enc: enc.write_space(ATTRIB_SPACE, _Space("ram")))
        self.assertEqual(
            data[1] >> PackedFormat.TYPECODE_SHIFT, PackedFormat.TYPECODE_ADDRESSSPACE
        )
        self.assertEqual(space_index_of(_Space("ram")), 2)

    def test_unknown_space_names_are_rejected(self) -> None:
        encoder = PackedEncoder()
        encoder.open_element(ELEM_SYNTAX)
        with self.assertRaises(MarshalError):
            encoder.write_space(ATTRIB_SPACE, _Space("nowhere"))

    def test_opcodes_and_metatypes_are_positive_integers(self) -> None:
        data = _encode(lambda enc: enc.write_opcode(ATTRIB_ID, OpCode.CPUI_FLOAT_ADD))
        self.assertEqual(
            data[1] >> PackedFormat.TYPECODE_SHIFT,
            PackedFormat.TYPECODE_SIGNEDINT_POSITIVE,
        )
        data = _encode(
            lambda enc: enc.write_datatype_meta(ATTRIB_ID, TypeMetatype.TYPE_FLOAT)
        )
        self.assertEqual(
            data[1] >> PackedFormat.TYPECODE_SHIFT,
            PackedFormat.TYPECODE_SIGNEDINT_POSITIVE,
        )

    def test_a_binary_stream_receives_the_same_bytes(self) -> None:
        stream = io.BytesIO()
        encoder = PackedEncoder(stream)
        encoder.open_element(ELEM_SYNTAX)
        encoder.write_string(ATTRIB_CONTENT, "x")
        encoder.close_element(ELEM_SYNTAX)
        self.assertEqual(stream.getvalue(), encoder.get_bytes())


class PackedDecodeTests(unittest.TestCase):
    def build(self) -> bytes:
        encoder = PackedEncoder()
        encoder.open_element(ELEM_CLANG_DOCUMENT)
        encoder.open_element(ELEM_FUNCTION)
        encoder.open_element(ELEM_BREAK)
        encoder.write_signed_integer(ATTRIB_INDENT, 4)
        encoder.close_element(ELEM_BREAK)
        encoder.open_element(ELEM_COMMENT)
        encoder.write_unsigned_integer(ATTRIB_COLOR, 1)
        encoder.write_space(ATTRIB_SPACE, _Space("ram"))
        encoder.write_unsigned_integer(ATTRIB_OFF, 0x401000)
        encoder.write_string(ATTRIB_CONTENT, "/* hola */")
        encoder.close_element(ELEM_COMMENT)
        encoder.open_element(ELEM_SYNTAX)
        encoder.write_signed_integer(ATTRIB_OPEN, 7)
        encoder.write_string(ATTRIB_CONTENT, "(")
        encoder.close_element(ELEM_SYNTAX)
        encoder.close_element(ELEM_FUNCTION)
        encoder.close_element(ELEM_CLANG_DOCUMENT)
        return encoder.get_bytes()

    def test_nested_traversal(self) -> None:
        decoder = PackedDecode()
        decoder.ingest_stream(self.build())
        self.assertEqual(decoder.peek_element(), ELEM_CLANG_DOCUMENT.id)
        self.assertEqual(decoder.open_element(), ELEM_CLANG_DOCUMENT.id)
        self.assertEqual(decoder.open_element(), ELEM_FUNCTION.id)
        self.assertEqual(decoder.open_element(), ELEM_BREAK.id)
        self.assertEqual(decoder.read_signed_integer(ATTRIB_INDENT), 4)
        decoder.close_element(ELEM_BREAK.id)
        self.assertEqual(decoder.open_element(), ELEM_COMMENT.id)
        self.assertEqual(decoder.read_space(ATTRIB_SPACE), "ram")
        self.assertEqual(decoder.read_unsigned_integer(ATTRIB_OFF), 0x401000)
        self.assertEqual(decoder.read_string(ATTRIB_CONTENT), "/* hola */")
        decoder.close_element(ELEM_COMMENT.id)
        self.assertEqual(decoder.open_element(ELEM_SYNTAX), ELEM_SYNTAX.id)
        self.assertEqual(decoder.read_signed_integer(ATTRIB_OPEN), 7)
        decoder.close_element(ELEM_SYNTAX.id)
        # Nothing left in the group: peek and open both report the end.
        self.assertEqual(decoder.peek_element(), 0)
        self.assertEqual(decoder.open_element(), 0)
        decoder.close_element(ELEM_FUNCTION.id)
        decoder.close_element(ELEM_CLANG_DOCUMENT.id)

    def test_open_element_checks_the_expected_id(self) -> None:
        decoder = PackedDecode()
        decoder.ingest_stream(self.build())
        decoder.open_element()
        with self.assertRaises(DecoderError):
            decoder.open_element(ELEM_SYNTAX)

    def test_attributes_are_scanned_in_order_and_can_be_rewound(self) -> None:
        decoder = PackedDecode()
        decoder.ingest_stream(self.build())
        decoder.open_element()
        decoder.open_element()
        decoder.open_element()  # <break indent="4">
        decoder.close_element(ELEM_BREAK.id)
        decoder.open_element()  # <comment color space off>content
        seen = []
        while True:
            attrib = decoder.get_next_attribute_id()
            if attrib == 0:
                break
            seen.append(attrib)
        self.assertEqual(
            seen,
            [ATTRIB_COLOR.id, ATTRIB_SPACE.id, ATTRIB_OFF.id, ATTRIB_CONTENT.id],
        )
        decoder.rewind_attributes()
        self.assertEqual(decoder.get_next_attribute_id(), ATTRIB_COLOR.id)
        self.assertEqual(decoder.read_unsigned_integer(), 1)
        # Reading by id restarts from the beginning of the element.
        self.assertEqual(decoder.read_string(ATTRIB_CONTENT), "/* hola */")
        self.assertEqual(decoder.read_unsigned_integer(ATTRIB_OFF), 0x401000)

    def test_indexed_attributes_are_never_reinterpreted(self) -> None:
        decoder = PackedDecode()
        decoder.ingest_stream(self.build())
        decoder.open_element()
        self.assertEqual(
            decoder.get_indexed_attribute_id(ATTRIB_ID), ATTRIB_UNKNOWN.id
        )

    def test_missing_attribute_raises(self) -> None:
        decoder = PackedDecode()
        decoder.ingest_stream(self.build())
        decoder.open_element()
        decoder.open_element()
        decoder.open_element()  # <break indent="4">
        with self.assertRaises(DecoderError):
            decoder.read_string(ATTRIB_CONTENT)

    def test_wrong_type_raises_after_skipping_the_attribute(self) -> None:
        decoder = PackedDecode()
        decoder.ingest_stream(self.build())
        decoder.open_element()
        decoder.open_element()
        decoder.open_element()  # <break indent="4">
        with self.assertRaises(DecoderError):
            decoder.read_string()  # the current attribute is a signed integer
        # The attribute was skipped, so traversal can continue.
        self.assertEqual(decoder.get_next_attribute_id(), 0)

    def test_close_element_skipping_jumps_over_a_subtree(self) -> None:
        decoder = PackedDecode()
        decoder.ingest_stream(self.build())
        decoder.open_element()
        function = decoder.open_element()
        decoder.close_element_skipping(function)
        self.assertEqual(decoder.peek_element(), 0)
        decoder.close_element(ELEM_CLANG_DOCUMENT.id)

    def test_close_element_checks_the_id(self) -> None:
        decoder = PackedDecode()
        decoder.ingest_stream(self.build())
        decoder.open_element()
        with self.assertRaises(DecoderError):
            decoder.close_element(ELEM_SYNTAX.id)

    def test_ingesting_nothing_is_an_error(self) -> None:
        decoder = PackedDecode()
        with self.assertRaises(DecoderError):
            decoder.ingest_stream(b"")


MARKUP = (
    "<clang_document>\n"
    " <function>\n"
    '  <break indent="4"/>\n'
    '  <comment color="0x1" space="ram" off="0x401000">/* hola */</comment>\n'
    '  <syntax open="-3">(</syntax>\n'
    '  <syntax close="0">&lt;&amp;&gt;</syntax>\n'
    '  <vardecl param2="second" param1="first"/>\n'
    " </function>\n"
    "</clang_document>\n"
)


class XmlDecodeTests(unittest.TestCase):
    def decoder(self) -> XmlDecode:
        decoder = XmlDecode()
        decoder.ingest_stream(MARKUP)
        self.assertEqual(decoder.open_element(), ELEM_CLANG_DOCUMENT.id)
        return decoder

    def test_elements_attributes_and_content(self) -> None:
        decoder = self.decoder()
        self.assertEqual(decoder.peek_element(), ELEM_FUNCTION.id)
        self.assertEqual(decoder.open_element(), ELEM_FUNCTION.id)
        self.assertEqual(decoder.open_element(), ELEM_BREAK.id)
        self.assertEqual(decoder.read_signed_integer(ATTRIB_INDENT), 4)
        decoder.close_element(ELEM_BREAK.id)
        self.assertEqual(decoder.open_element(), ELEM_COMMENT.id)
        self.assertEqual(decoder.read_unsigned_integer(ATTRIB_COLOR), 1)
        self.assertEqual(decoder.read_space(ATTRIB_SPACE), "ram")
        self.assertEqual(decoder.read_unsigned_integer(ATTRIB_OFF), 0x401000)
        self.assertEqual(decoder.read_string(ATTRIB_CONTENT), "/* hola */")
        decoder.close_element(ELEM_COMMENT.id)

    def test_signed_values_and_xml_unescaping(self) -> None:
        decoder = self.decoder()
        decoder.open_element()
        decoder.open_element()  # break
        decoder.close_element(ELEM_BREAK.id)
        decoder.open_element()  # comment
        decoder.close_element(ELEM_COMMENT.id)
        self.assertEqual(decoder.open_element(), ELEM_SYNTAX.id)
        self.assertEqual(decoder.read_signed_integer(ATTRIB_OPEN), -3)
        decoder.close_element(ELEM_SYNTAX.id)
        self.assertEqual(decoder.open_element(), ELEM_SYNTAX.id)
        self.assertEqual(decoder.read_string(ATTRIB_CONTENT), "<&>")
        decoder.close_element(ELEM_SYNTAX.id)

    def test_unknown_elements_and_attributes_read_as_zero(self) -> None:
        decoder = XmlDecode()
        decoder.ingest_stream('<clang_document><mystery thing="1"/></clang_document>')
        decoder.open_element()
        self.assertEqual(decoder.open_element(), ELEM_UNKNOWN.id)
        self.assertEqual(decoder.get_next_attribute_id(), ATTRIB_UNKNOWN.id)

    def test_indexed_attribute_names_are_reinterpreted(self) -> None:
        decoder = XmlDecode()
        decoder.ingest_stream(
            '<clang_document><vardecl param1="first" param2="second"/></clang_document>'
        )
        decoder.open_element()
        decoder.open_element()
        base = AttributeId("param", 5)
        # "param1" is not a registered name: it reads as ATTRIB_UNKNOWN and gets
        # reinterpreted as the first member of the indexed family.
        self.assertEqual(decoder.get_next_attribute_id(), ATTRIB_UNKNOWN.id)
        self.assertEqual(decoder.get_indexed_attribute_id(base), 5)
        self.assertEqual(decoder.read_string(), "first")
        self.assertEqual(decoder.get_next_attribute_id(), ATTRIB_UNKNOWN.id)
        self.assertEqual(decoder.get_indexed_attribute_id(base), 6)
        self.assertEqual(decoder.read_string(), "second")
        self.assertEqual(decoder.get_next_attribute_id(), 0)

    def test_a_zero_index_is_a_bad_attribute(self) -> None:
        decoder = XmlDecode()
        decoder.ingest_stream('<clang_document><vardecl param0="x"/></clang_document>')
        decoder.open_element()
        decoder.open_element()
        decoder.get_next_attribute_id()
        with self.assertRaises(MarshalError):
            decoder.get_indexed_attribute_id(AttributeId("param", 5))

    def test_empty_content_and_missing_attributes(self) -> None:
        decoder = self.decoder()
        decoder.open_element()
        decoder.open_element()  # <break indent="4"/>
        # An element without text has empty content (getContent()).
        self.assertEqual(decoder.read_string(ATTRIB_CONTENT), "")
        with self.assertRaises(DecoderError):
            decoder.read_string(AttributeId("nope", 700))

    def test_mismatched_close_raises(self) -> None:
        decoder = self.decoder()
        decoder.open_element()
        with self.assertRaises(DecoderError):
            decoder.close_element(ELEM_SYNTAX.id)

    def test_open_element_checks_the_expected_name(self) -> None:
        decoder = self.decoder()
        with self.assertRaises(DecoderError):
            decoder.open_element(ELEM_SYNTAX)

    def test_exhausted_document_reports_zero(self) -> None:
        decoder = XmlDecode()
        decoder.ingest_stream("<clang_document/>")
        self.assertEqual(decoder.open_element(), ELEM_CLANG_DOCUMENT.id)
        self.assertEqual(decoder.peek_element(), 0)
        self.assertEqual(decoder.open_element(), 0)
        decoder.close_element(ELEM_CLANG_DOCUMENT.id)
        with self.assertRaises(DecoderError):
            decoder.open_element(ELEM_FUNCTION)

    def test_opcode_and_metatype_names(self) -> None:
        decoder = XmlDecode()
        decoder.ingest_stream('<x opcode="FLOAT_ADD" meta="float"/>')
        decoder.open_element()
        self.assertEqual(
            decoder.read_opcode(AttributeId("opcode", 500)), OpCode.CPUI_FLOAT_ADD
        )
        self.assertEqual(
            decoder.read_datatype_meta(AttributeId("meta", 501)),
            TypeMetatype.TYPE_FLOAT,
        )


class XmlEncoderExtraTests(unittest.TestCase):
    def test_indexed_strings_number_the_attribute_name(self) -> None:
        stream = io.StringIO()
        encoder = XmlEncoder(stream)
        encoder.open_element(ELEM_FUNCTION)
        encoder.write_string_indexed(AttributeId("param", 5), 0, "a")
        encoder.write_string_indexed(AttributeId("param", 5), 2, "b")
        encoder.close_element(ELEM_FUNCTION)
        flat = stream.getvalue().replace("\n", "").replace(" ", "")
        self.assertEqual(flat, '<functionparam1="a"param3="b"/>')

    def test_opcodes_and_metatypes_are_written_by_name(self) -> None:
        stream = io.StringIO()
        encoder = XmlEncoder(stream)
        encoder.open_element(ELEM_FUNCTION)
        encoder.write_opcode(AttributeId("code", 500), OpCode.CPUI_MULTIEQUAL)
        encoder.write_datatype_meta(
            AttributeId("meta", 501), TypeMetatype.TYPE_STRUCT
        )
        encoder.close_element(ELEM_FUNCTION)
        text = stream.getvalue()
        self.assertIn('code="MULTIEQUAL"', text)
        self.assertIn('meta="struct"', text)

    def test_content_is_not_wrapped_in_an_attribute(self) -> None:
        stream = io.StringIO()
        encoder = XmlEncoder(stream, do_format=False)
        encoder.open_element(ELEM_SYNTAX)
        encoder.write_string(ATTRIB_CONTENT, "a<b>&c")
        encoder.close_element(ELEM_SYNTAX)
        self.assertEqual(
            stream.getvalue(), "<syntax>a&lt;b&gt;&amp;c</syntax>"
        )


class RegistryTests(unittest.TestCase):
    def test_element_and_attribute_names_resolve(self) -> None:
        self.assertEqual(lookup_element_id("clang_document"), ELEM_CLANG_DOCUMENT.id)
        self.assertEqual(lookup_element_id("variable"), ELEM_VARIABLE.id)
        self.assertEqual(lookup_attribute_id("color"), ATTRIB_COLOR.id)
        self.assertEqual(lookup_attribute_id("XMLcontent"), ATTRIB_CONTENT.id)

    def test_unregistered_names_fall_back_to_the_placeholders(self) -> None:
        self.assertEqual(lookup_element_id("nope"), ELEM_UNKNOWN.id)
        self.assertEqual(lookup_attribute_id("nope"), ATTRIB_UNKNOWN.id)
        self.assertNotEqual(ELEM_UNKNOWN.id, 0)
        self.assertNotEqual(ATTRIB_UNKNOWN.id, 0)

    def test_scoped_lookups_are_not_supported(self) -> None:
        self.assertEqual(lookup_element_id("syntax", 1), ELEM_UNKNOWN.id)
        self.assertEqual(lookup_attribute_id("color", 1), ATTRIB_UNKNOWN.id)

    def test_metatype_names_round_trip(self) -> None:
        for metatype in TypeMetatype:
            name = metatype_to_string(metatype)
            if metatype in (
                TypeMetatype.TYPE_ENUM_INT,
                TypeMetatype.TYPE_ENUM_UINT,
            ):
                continue  # both are spelled "enum_int"/"enum_uint"
            self.assertEqual(string_to_metatype(name), metatype, name)
        self.assertEqual(metatype_to_string(TypeMetatype.TYPE_VOID), "void")
        self.assertEqual(string_to_metatype("enum"), TypeMetatype.TYPE_ENUM_INT)
        self.assertEqual(
            string_to_metatype("nonsense"), TypeMetatype.TYPE_UNKNOWN
        )
        self.assertEqual(
            metatype_to_string("nonsense"), metatype_to_string(TypeMetatype.TYPE_UNKNOWN)
        )

    def test_opcode_names_round_trip(self) -> None:
        self.assertEqual(get_opname(OpCode.CPUI_FLOAT_ADD), "FLOAT_ADD")
        self.assertEqual(get_opcode("FLOAT_ADD"), OpCode.CPUI_FLOAT_ADD)
        self.assertEqual(get_opcode("NOT_AN_OP"), 0)
        self.assertEqual(get_opname(0), "0")


class XmlHelperTests(unittest.TestCase):
    def test_xml_escape_only_touches_special_characters(self) -> None:
        self.assertEqual(xml_escape('<a b="c">&\''), "&lt;a b=&quot;c&quot;&gt;&amp;&apos;")
        self.assertEqual(xml_escape("plain text ?@"), "plain text ?@")

    def test_xml_read_bool_only_accepts_true(self) -> None:
        self.assertTrue(xml_read_bool("true"))
        self.assertTrue(xml_read_bool(" true "))
        self.assertFalse(xml_read_bool("false"))
        self.assertFalse(xml_read_bool("1"))
        self.assertFalse(xml_read_bool(""))

    def test_integer_parsing_follows_the_stream_rules(self) -> None:
        decoder = XmlDecode()
        decoder.ingest_stream(
            '<x dec="-12" hex="0x1f" oct="010" junk="42abc" empty=""/>'
        )
        decoder.open_element()
        self.assertEqual(decoder.read_signed_integer(AttributeId("dec", 1)), -12)
        self.assertEqual(decoder.read_unsigned_integer(AttributeId("hex", 2)), 0x1F)
        self.assertEqual(decoder.read_signed_integer(AttributeId("oct", 3)), 8)
        self.assertEqual(decoder.read_signed_integer(AttributeId("junk", 4)), 42)
        self.assertEqual(decoder.read_signed_integer(AttributeId("empty", 5)), 0)

    def test_unknown_space_names_are_rejected(self) -> None:
        decoder = XmlDecode()
        decoder.ingest_stream('<x space="nowhere"/>')
        decoder.open_element()
        with self.assertRaises(DecoderError):
            decoder.read_space(ATTRIB_SPACE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
