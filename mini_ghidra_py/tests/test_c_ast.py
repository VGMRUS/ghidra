from __future__ import annotations

import unittest
from types import SimpleNamespace

from minighidra.c_ast import (
    CArrayDesignator,
    CArrayModifier,
    CAssignment,
    CAtomicType,
    CBinary,
    CBlock,
    CCall,
    CCharacter,
    CCast,
    CCompoundLiteral,
    CConditional,
    CDeclaration,
    CDeclarator,
    CDoWhile,
    CDesignatedInitializer,
    CFieldDesignator,
    CFunctionModifier,
    CEmitter,
    CFloat,
    CFor,
    CFunction,
    CFunctionPrototype,
    CGlobalDeclaration,
    CPrimitiveType,
    CIdentifier,
    CIf,
    CIndex,
    CInitializerList,
    CInteger,
    CMember,
    CParameter,
    CPointerModifier,
    CRawExpression,
    CReturn,
    CSizeof,
    CSwitch,
    CString,
    CTranslationUnit,
    CTypeDeclaration,
    CTagType,
    CTypeName,
    CTypedefDeclaration,
    CTypedefType,
    CUnary,
    CWhile,
    parse_c_declaration,
    parse_c_parameter,
    parse_c_statements,
    parse_c_type_name,
    parse_c_translation_unit,
)
from minighidra.decompiler import SemanticDecompiler, _Expr
from minighidra.typesys import StorageLocation, TypeDatabase


class CEmitterTests(unittest.TestCase):
    def _ast_renderer(self) -> SemanticDecompiler:
        renderer = object.__new__(SemanticDecompiler)
        renderer.bits = 64
        renderer.analyzer = SimpleNamespace(name_at=lambda _address: None)
        renderer._string_by_address = {}
        renderer._c_emitter = CEmitter()
        return renderer

    def test_expression_nodes_emit_precedence_safe_c(self) -> None:
        emitter = CEmitter()
        expression = CBinary(
            "||",
            CBinary("&&", CIdentifier("ready"), CIdentifier("enabled")),
            CCall(CIdentifier("check"), (CInteger(7), CString('a"b'))),
        )
        self.assertEqual(
            emitter.emit_expression(expression),
            '((ready && enabled) || check(7, "a\\"b"))',
        )

    def test_pointer_dereference_and_postfix_bases_keep_c_precedence(self) -> None:
        emitter = CEmitter()
        dereference = emitter.emit_expression(
            CUnary("*", CCast("uint32_t *", CInteger(0x1000, 16)))
        )
        member = emitter.emit_expression(
            CMember(CUnary("*", CIdentifier("pointer")), "field")
        )
        call = emitter.emit_expression(
            CCall(CUnary("*", CIdentifier("callback")), (CInteger(1),))
        )
        self.assertEqual(dereference, "*(uint32_t *)(0x1000)")
        self.assertEqual(member, "(*pointer).field")
        self.assertEqual(call, "(*callback)(1)")

    def test_compound_literals_emit_designated_initializers(self) -> None:
        literal = CCompoundLiteral(
            "union { uint32_t i; float f; }",
            CInitializerList(
                (CDesignatedInitializer("f", CCast("float", CIdentifier("value"))),)
            ),
        )
        source = CEmitter().emit_expression(CMember(literal, "i"))
        self.assertEqual(
            source,
            "((union { uint32_t i; float f; }){ .f = (float)(value) }).i",
        )

    def test_function_parameters_use_typed_declarators_with_raw_fallbacks(self) -> None:
        pointer = parse_c_parameter("const uint8_t *buffer")
        self.assertIsInstance(pointer, CParameter)
        self.assertEqual(pointer.type_name, "const uint8_t")
        self.assertEqual(pointer.declarator.pointer_depth, 1)

        callback = parse_c_parameter("int (*callback)(uint32_t)")
        self.assertEqual(callback.type_name, "int")
        self.assertEqual(callback.declarator.name, "callback")
        self.assertEqual(
            [type(modifier) for modifier in callback.declarator.modifiers],
            [CPointerModifier, CFunctionModifier],
        )
        function = CFunction(
            "void", "consume", (pointer, callback, "..."), CBlock()
        )
        self.assertEqual(
            CEmitter().emit_function(function).splitlines()[0],
            "void consume(const uint8_t *buffer, int (*callback)(uint32_t), ...) {",
        )

    def test_body_parser_preserves_structured_branches(self) -> None:
        body = parse_c_statements(
            [
                "loc_1000:",
                "  value = input + 1;",
                "  if (value != 0) {",
                "    result = 3;",
                "  } else {",
                "    result = 4;",
                "  }",
                "  return result;",
            ]
        )
        function = CFunction(
            "int",
            "example",
            ("int input",),
            body,
            local_declarations=(CDeclaration("int value;"), CDeclaration("int result;")),
        )
        source = CEmitter().emit(CTranslationUnit((function,)))
        self.assertIn("loc_1000:", source)
        self.assertIn("if ((value != 0)) {", source)
        self.assertIn("} else {", source)
        self.assertIn("return result;", source)

    def test_switch_cases_with_inline_gotos_are_ast_nodes(self) -> None:
        body = parse_c_statements(
            [
                "switch ((uint32_t)selector) {",
                "  case 0x10: goto loc_1010;",
                "  default: goto loc_1020;",
                "}",
            ]
        )
        source = CEmitter().emit_statement(body.statements[0])
        self.assertIn("switch (", source[0])
        self.assertIn("uint32_t", source[0])
        self.assertIn("selector", source[0])
        self.assertIn("case 0x10:", source[1])
        self.assertIn("goto loc_1010;", source[2])
        self.assertIn("default:", source[3])
        self.assertIn("goto loc_1020;", source[4])

    def test_switch_cases_preserve_nested_control_flow(self) -> None:
        body = parse_c_statements(
            [
                "switch (selector) {",
                "  case 0:",
                "    if (ready) {",
                "      while (pending) {",
                "        break;",
                "      }",
                "    } else {",
                "      continue;",
                "    }",
                "  case 1:",
                "    for (uint32_t i = 0; i < 2; ++i) {",
                "      consume(i);",
                "    }",
                "}",
            ]
        )
        switch = body.statements[0]
        self.assertIsInstance(switch, CSwitch)
        nested_if = switch.cases[0].statements[0]
        self.assertIsInstance(nested_if, CIf)
        self.assertIsInstance(nested_if.body.statements[0], CWhile)
        self.assertIsInstance(nested_if.otherwise, CBlock)
        self.assertIsInstance(switch.cases[1].statements[0], CFor)
        emitted = "\\n".join(CEmitter().emit_statement(switch))
        self.assertIn("while (pending)", emitted)
        self.assertIn("for (uint32_t i = 0;", emitted)

    def test_single_line_if_and_return_are_typed_nodes(self) -> None:
        body = parse_c_statements(["if (ready) goto loc_2000;", "return;"])
        self.assertIsInstance(body.statements[0], CIf)
        self.assertIsInstance(body.statements[1], CReturn)

    def test_generated_statement_expressions_lower_to_ast(self) -> None:
        body = parse_c_statements(
            line
            for line in (
                "uint32_t result = 0;",
                "if ((uint32_t)count > 0 && ready) {",
                "  result = choose_value(arg, 2.5f) + values[index].field;",
                "}",
                "return result;",
            )
        )
        declaration, branch, returned = body.statements
        self.assertIsInstance(declaration, CDeclaration)
        self.assertEqual(declaration.type_name, "uint32_t")
        self.assertIsInstance(declaration.declarators[0].initializer, CInteger)
        self.assertIsInstance(branch, CIf)
        self.assertIsInstance(branch.condition, CBinary)
        self.assertEqual(branch.condition.operator, "&&")
        self.assertIsInstance(branch.condition.left, CBinary)
        self.assertIsInstance(branch.condition.left.left, CCast)
        assignment = branch.body.statements[0]
        self.assertIsInstance(assignment, CAssignment)
        self.assertIsInstance(assignment.value, CBinary)
        call = assignment.value.left
        self.assertIsInstance(call, CCall)
        self.assertIsInstance(call.arguments[1], CFloat)
        access = assignment.value.right
        self.assertIsInstance(access, CMember)
        self.assertIsInstance(access.value, CIndex)
        self.assertIsInstance(returned, CReturn)
        self.assertIsInstance(returned.value, CIdentifier)

    def test_decompiler_core_expressions_lower_to_typed_ast(self) -> None:
        renderer = self._ast_renderer()
        load = _Expr(
            "load",
            args=(_Expr("address", ("absolute", 0x1000), bits=64),),
            bits=64,
        )
        loaded = renderer._expression_ast(load)
        self.assertIsInstance(loaded, CUnary)
        self.assertEqual(
            renderer._c_emitter.emit_expression(loaded), "*(uint64_t *)(0x1000)"
        )

        extract = _Expr(
            "extract",
            args=(_Expr("var", "wide_value", bits=64), _Expr("const", 8, bits=8)),
            bits=8,
        )
        extracted = renderer._expression_ast(extract)
        self.assertIsInstance(extracted, CBinary)
        self.assertIn("wide_value >> 8", renderer._c_emitter.emit_expression(extracted))

        concat = _Expr(
            "concat",
            value=32,
            args=(_Expr("var", "high", bits=32), _Expr("var", "low", bits=32)),
            bits=64,
        )
        combined = renderer._expression_ast(concat)
        self.assertIsInstance(combined, CBinary)
        combined_source = renderer._c_emitter.emit_expression(combined)
        self.assertIn("high", combined_source)
        self.assertIn("low", combined_source)
        self.assertIn("<< 32", combined_source)

    def test_decompiler_bitcasts_and_aggregate_values_lower_to_ast(self) -> None:
        renderer = self._ast_renderer()
        fp_to_bits = _Expr(
            "fp_to_bits",
            32,
            args=(_Expr("var", "floating", bits=32),),
            bits=32,
        )
        bitcast = renderer._expression_ast(fp_to_bits)
        self.assertIsInstance(bitcast, CMember)
        self.assertIsInstance(bitcast.value, CCompoundLiteral)
        self.assertEqual(
            renderer._c_emitter.emit_expression(bitcast),
            "((union { uint32_t i; float f; }){ .f = (float)(floating) }).i",
        )

        simd_float = _Expr(
            "simd_float",
            32,
            args=(_Expr("var", "raw_bits", bits=32),),
            bits=80,
        )
        self.assertEqual(
            renderer._c_emitter.emit_expression(renderer._expression_ast(simd_float)),
            "((union { uint32_t i; float f; }){ .i = (uint32_t)(raw_bits) }).f",
        )

        database = TypeDatabase(64)
        pair_type = database.structure(
            "AstPair", (("first", "uint32_t"), ("second", "uint32_t"))
        )
        locations = (
            StorageLocation("register", 4, register="eax", value_offset=0),
            StorageLocation("register", 4, register="edx", value_offset=4),
        )
        aggregate = _Expr(
            "aggregate_value",
            (pair_type, locations),
            args=(
                _Expr("var", "first", bits=32),
                _Expr("var", "second", bits=32),
            ),
            bits=64,
        )
        aggregate_ast = renderer._expression_ast(aggregate)
        self.assertIsInstance(aggregate_ast, CCompoundLiteral)
        self.assertEqual(
            renderer._c_emitter.emit_expression(aggregate_ast),
            "((struct AstPair){ .first = ((uint32_t)(first)), "
            ".second = ((uint32_t)(second)) })",
        )

    def test_decompiler_vector_expressions_lower_to_typed_ast(self) -> None:
        renderer = self._ast_renderer()
        vector = _Expr("var", "vector", bits=128)
        old_destination = _Expr("var", "old_destination", bits=128)
        fp_result = _Expr(
            "fp_op",
            "add",
            args=(_Expr("fp_const", 1.25, bits=80), _Expr("fp_const", 2.5, bits=80)),
            bits=32,
        )
        expressions = {
            "mmx_lanes": _Expr(
                "mmx_lanes",
                ("add", 16),
                args=(_Expr("var", "mmx_left", bits=64), _Expr("var", "mmx_right", bits=64)),
                bits=64,
            ),
            "simd_lanes": _Expr(
                "simd_lanes",
                ("eq", 32),
                args=(vector, _Expr("var", "other_vector", bits=128)),
                bits=128,
            ),
            "mmx_shift": _Expr(
                "mmx_shift",
                ("ashr", 16),
                args=(_Expr("var", "mmx_value", bits=64), _Expr("var", "count", bits=8)),
                bits=64,
            ),
            "simd_fp_binary": _Expr(
                "simd_fp_binary", 32, args=(fp_result, old_destination), bits=128
            ),
            "simd_convert_int_fp": _Expr(
                "simd_convert_int_fp",
                (32, 32),
                args=(_Expr("var", "integer", bits=32), old_destination),
                bits=128,
            ),
            "simd_convert_fp": _Expr(
                "simd_convert_fp",
                (64, 64),
                args=(_Expr("var", "double_value", bits=64), old_destination),
                bits=128,
            ),
            "simd_minmax": _Expr(
                "simd_minmax",
                "min",
                args=(
                    _Expr("var", "left_fp", bits=64),
                    _Expr("var", "right_fp", bits=64),
                    old_destination,
                ),
                bits=128,
            ),
            "simd_cvtdq2pd": _Expr(
                "simd_cvtdq2pd", args=(vector,), bits=128
            ),
            "simd_cvtt_to_int": _Expr(
                "simd_cvtt_to_int",
                (32, 64),
                args=(_Expr("var", "floating_value", bits=64),),
                bits=32,
            ),
            "simd_shuffle_dwords": _Expr(
                "simd_shuffle_dwords", 0x1B, args=(vector,), bits=128
            ),
            "simd_shift_bytes": _Expr(
                "simd_shift_bytes",
                "left",
                args=(vector, _Expr("const", 3, bits=8)),
                bits=128,
            ),
        }
        rendered = {}
        for name, expression in expressions.items():
            with self.subTest(expression=name):
                node = renderer._expression_ast(expression)
                self.assertIsNotNone(node)
                self.assertNotIsInstance(node, CRawExpression)
                rendered[name] = renderer._c_emitter.emit_expression(node)

        self.assertIn("uint16_t", rendered["mmx_lanes"])
        self.assertIn("==", rendered["simd_lanes"])
        self.assertIn(">= 16", rendered["mmx_shift"])
        self.assertIn("union { uint32_t i; float f; }", rendered["simd_fp_binary"])
        self.assertIn("int32_t", rendered["simd_convert_int_fp"])
        self.assertIn("double", rendered["simd_convert_fp"])
        self.assertIn("isnan", rendered["simd_minmax"])
        self.assertIn("int32_t", rendered["simd_cvtdq2pd"])
        self.assertIn("isfinite", rendered["simd_cvtt_to_int"])
        self.assertIn("<< 96", rendered["simd_shuffle_dwords"])
        self.assertIn(">= 16", rendered["simd_shift_bytes"])

    def test_decompiler_x87_and_pcode_expressions_lower_to_typed_ast(self) -> None:
        renderer = self._ast_renderer()
        fscale = _Expr(
            "fp_func",
            "fscale",
            args=(_Expr("var", "left", bits=80), _Expr("var", "right", bits=80)),
            bits=80,
        )
        source = renderer._c_emitter.emit_expression(renderer._expression_ast(fscale))
        self.assertEqual(source, "(left * exp2l(truncl(right)))")

        condition = _Expr(
            "fp_condition",
            "ja",
            args=(_Expr("var", "left", bits=80), _Expr("var", "right", bits=80)),
            bits=1,
        )
        condition_ast = renderer._expression_ast(condition)
        self.assertIsInstance(condition_ast, CBinary)
        condition_source = renderer._c_emitter.emit_expression(condition_ast)
        self.assertIn("isnan(left)", condition_source)
        self.assertIn("left > right", condition_source)

        builtin = _Expr(
            "pcode_builtin",
            "bsr",
            args=(_Expr("var", "value", bits=32),),
            bits=32,
        )
        builtin_ast = renderer._expression_ast(builtin)
        self.assertIsInstance(builtin_ast, CConditional)
        builtin_source = renderer._c_emitter.emit_expression(builtin_ast)
        self.assertIn("__builtin_clzll", builtin_source)
        self.assertIn("63 -", builtin_source)

    def test_ternary_case_label_is_parsed_as_one_expression(self) -> None:
        body = parse_c_statements(
            [
                "switch (selector) {",
                "  case choose ? 1 : 2: goto loc_1000;",
                "}",
            ]
        )
        case = body.statements[0].cases[0]
        self.assertIsInstance(case.value, CConditional)
        self.assertEqual(case.statements[0].target, "loc_1000")

    def test_empty_switch_cases_are_not_dropped(self) -> None:
        body = parse_c_statements(
            [
                "switch (selector) {",
                "  case 0:",
                "  case 1: break;",
                "  default:",
                "}",
            ]
        )
        switch = body.statements[0]
        self.assertEqual(len(switch.cases), 3)
        self.assertEqual(switch.cases[0].statements, ())
        self.assertEqual(len(switch.cases[1].statements), 1)
        self.assertIsNone(switch.cases[2].value)
        self.assertEqual(switch.cases[2].statements, ())

    def test_function_pointer_declarations_have_structured_layers(self) -> None:
        body = parse_c_statements(["int (*callback)(uint32_t);"])
        declaration = body.statements[0]
        self.assertIsInstance(declaration, CDeclaration)
        self.assertEqual(declaration.type_name, "int")
        declarator = declaration.declarators[0]
        self.assertEqual(declarator.name, "callback")
        self.assertEqual(
            [type(modifier) for modifier in declarator.modifiers],
            [CPointerModifier, CFunctionModifier],
        )
        self.assertEqual(
            CEmitter().emit_statement(declaration),
            ["int (*callback)(uint32_t);"],
        )
        cast = parse_c_statements(
            ["return (int (*)(uint32_t)) callback;"]
        ).statements[0].value
        self.assertIsInstance(cast, CCast)
        self.assertIsInstance(cast.type_ast.declarator.modifiers[-1], CFunctionModifier)

    def test_simple_declarations_build_typed_declarator_nodes(self) -> None:
        body = parse_c_statements(
            ["uint8_t first = 1, *cursor = buffer, table[4];"]
        )
        declaration = body.statements[0]
        self.assertIsInstance(declaration, CDeclaration)
        self.assertEqual(declaration.type_name, "uint8_t")
        self.assertEqual(len(declaration.declarators), 3)
        first, cursor, table = declaration.declarators
        self.assertIsInstance(first, CDeclarator)
        self.assertEqual(first.name, "first")
        self.assertIsInstance(first.initializer, CInteger)
        self.assertEqual(cursor.name, "cursor")
        self.assertEqual(cursor.pointer_depth, 1)
        self.assertEqual(table.name, "table")
        self.assertIsInstance(table.array_dimensions[0], CInteger)
        self.assertEqual(
            CEmitter().emit_statement(declaration),
            ["uint8_t first = 1, *cursor = buffer, table[4];"],
        )

    def test_recursive_declarators_preserve_pointer_array_function_binding(self) -> None:
        sources = (
            "int *items[3];",
            "int (*items)[3];",
            "int (*callbacks[2])(const char *, ...);",
            "int *convert(uint32_t value);",
        )
        for source in sources:
            with self.subTest(source=source):
                declaration = parse_c_declaration(source)
                self.assertIsNotNone(declaration.type_name)
                self.assertEqual(
                    CEmitter()._emit_declaration(declaration), source
                )
        callback = parse_c_declaration(sources[2]).declarators[0]
        self.assertEqual(
            [type(modifier) for modifier in callback.modifiers],
            [CArrayModifier, CPointerModifier, CFunctionModifier],
        )
        self.assertTrue(callback.modifiers[-1].variadic)

    def test_typedef_function_pointer_and_function_returning_callback_are_structured(self) -> None:
        unit = parse_c_translation_unit(
            "typedef int (*Callback)(uint32_t);\n"
            "extern int (*callback)(uint32_t);\n"
            "int (*factory(void))(uint32_t);\n"
            "int (*make_callback(uint32_t flags))(uint32_t) { "
            "(void)flags; return callback; }\n"
        )
        self.assertIsInstance(unit.items[0], CTypedefDeclaration)
        self.assertIsInstance(unit.items[1], CGlobalDeclaration)
        self.assertIsInstance(unit.items[2], CFunctionPrototype)
        self.assertIsInstance(unit.items[3], CFunction)
        self.assertIsInstance(
            unit.items[3].declarator.modifiers[0], CFunctionModifier
        )
        self.assertIsInstance(
            unit.items[3].declarator.modifiers[1], CPointerModifier
        )
        emitted = CEmitter().emit(unit)
        self.assertIn("typedef int (*Callback)(uint32_t);", emitted)
        self.assertIn("extern int (*callback)(uint32_t);", emitted)
        self.assertIn("int (*factory(void))(uint32_t);", emitted)
        self.assertIn(
            "int (*make_callback(uint32_t flags))(uint32_t) {", emitted
        )
        self.assertIn("(void)(flags);", emitted)

    def test_type_names_are_structured_primitives_aliases_tags_and_abstract_declarators(
        self,
    ) -> None:
        alias_pointer = parse_c_type_name("const Node *")
        self.assertIsInstance(alias_pointer, CTypeName)
        self.assertIsInstance(alias_pointer.base, CTypedefType)
        self.assertEqual(alias_pointer.base.name, "Node")
        self.assertEqual(alias_pointer.qualifiers, ("const",))
        self.assertEqual(alias_pointer.declarator.pointer_depth, 1)
        declaration = parse_c_declaration("const Node *next;")
        self.assertIsInstance(declaration.type_ast.base, CTypedefType)
        parameter = parse_c_parameter("const Node *next")
        self.assertIsInstance(parameter.type_ast.base, CTypedefType)
        cast = CCast("Node *", CIdentifier("node"))
        self.assertEqual(cast.type_ast.declarator.pointer_depth, 1)

        primitive = parse_c_type_name("unsigned long long")
        self.assertIsInstance(primitive.base, CPrimitiveType)
        self.assertEqual(primitive.base.name, "unsigned long long")

        tagged = parse_c_type_name("union { uint32_t i; float f; }")
        self.assertIsInstance(tagged.base, CTagType)
        self.assertEqual(tagged.base.kind, "union")
        self.assertIsNotNone(tagged.base.definition)

        atomic = parse_c_type_name("_Atomic(uint32_t)")
        self.assertIsInstance(atomic.base, CAtomicType)
        self.assertEqual(
            CEmitter()._emit_type_name("int (*)(void)"), "int (*)(void)"
        )
        array_type = parse_c_type_name("int [4]")
        self.assertEqual(
            CEmitter()._emit_type_name("int [4]", array_type), "int [4]"
        )
        pointer_to_array = parse_c_type_name("int (*)[4]")
        self.assertEqual(
            CEmitter()._emit_type_name("int (*)[4]", pointer_to_array),
            "int (*)[4]",
        )

    def test_nested_initializers_and_array_designators_are_structured(self) -> None:
        source = (
            "struct Matrix matrix[2] = "
            "{ [0].rows[1] = 3, [1] = { .rows = { [0] = 4 } } };"
        )
        declaration = parse_c_declaration(source)
        self.assertIsInstance(declaration.declarators[0].initializer, CInitializerList)
        first = declaration.declarators[0].initializer.entries[0]
        self.assertIsInstance(first, CDesignatedInitializer)
        self.assertEqual(
            [type(designator) for designator in first.designators],
            [CArrayDesignator, CFieldDesignator, CArrayDesignator],
        )
        self.assertEqual(CEmitter()._emit_declaration(declaration), source)

    def test_tagged_type_definitions_structure_bitfields_and_forward_declarations(self) -> None:
        unit = parse_c_translation_unit(
            "struct Flags { unsigned int ready : 1; unsigned int : 0; };\n"
            "union Payload;\n"
        )
        definition, forward = unit.items
        self.assertIsInstance(definition, CTypeDeclaration)
        self.assertEqual(definition.kind, "struct")
        self.assertEqual(definition.members[0].declarators[0].bit_width, CInteger(1))
        self.assertEqual(definition.members[1].declarators[0].name, "")
        self.assertIsInstance(forward, CTypeDeclaration)
        self.assertFalse(forward.definition)
        emitted = CEmitter().emit(unit)
        self.assertIn("unsigned int ready : 1;", emitted)
        self.assertIn("union Payload;", emitted)

    def test_translation_unit_scanner_ignores_delimiters_in_comments_and_strings(self) -> None:
        source = '''int sample(void) {
  /* A brace here must not end the function: } */
  return 1;
}
const char *text = "}";
'''
        unit = parse_c_translation_unit(source)
        self.assertIsInstance(unit.items[0], CFunction)
        self.assertIsInstance(unit.items[1], CGlobalDeclaration)
        emitted = CEmitter().emit(unit)
        self.assertIn("/* A brace here must not end the function: } */", emitted)
        self.assertIn('const char *text = "}";', emitted)

    def test_translation_unit_structures_globals_types_prototypes_and_functions(self) -> None:
        source = """#include <stdint.h>
typedef struct Node { uint32_t value; struct Node *next; } Node;
extern int global_count;
static const uint32_t values[3] = { [0] = 1, [2] = 4 };
enum Mode { MODE_A, MODE_B = 4 };
int add(int a, int b);
int add(int a, int b) { return a + b; }
"""
        unit = parse_c_translation_unit(source)
        self.assertEqual(len(unit.functions), 1)
        self.assertIsInstance(unit.items[1], CTypedefDeclaration)
        self.assertIsInstance(unit.items[1].declaration, CTypeDeclaration)
        self.assertEqual(unit.items[1].declaration.kind, "struct")
        self.assertIsInstance(unit.items[2], CGlobalDeclaration)
        self.assertIsInstance(unit.items[3], CGlobalDeclaration)
        self.assertEqual(unit.items[3].declaration.storage_specifiers, ("static",))
        self.assertIsInstance(unit.items[4], CTypeDeclaration)
        self.assertIsInstance(unit.items[5], CFunctionPrototype)
        self.assertEqual(unit.items[6].name, "add")
        emitted = CEmitter().emit(unit)
        self.assertIn("typedef struct Node {", emitted)
        self.assertIn("extern int global_count;", emitted)
        self.assertIn("[0] = 1, [2] = 4", emitted)
        self.assertIn("int add(int a, int b);", emitted)
        self.assertIn("int add(int a, int b) {", emitted)
        self.assertIn("return (a + b);", emitted)

    def test_for_loop_header_is_structured_when_supported(self) -> None:
        body = parse_c_statements(
            [
                "for (uint32_t index = 0; index < limit; ++index) {",
                "  total += values[index];",
                "}",
            ]
        )
        loop = body.statements[0]
        self.assertIsInstance(loop, CFor)
        self.assertTrue(loop.structured)
        self.assertIsInstance(loop.initializer, CDeclaration)
        self.assertIsInstance(loop.condition, CBinary)
        self.assertIsInstance(loop.iteration, CUnary)
        self.assertEqual(loop.iteration.operator, "++")
        self.assertEqual(
            CEmitter().emit_statement(loop),
            [
                "for (uint32_t index = 0; (index < limit); ++index) {",
                "  total += values[index];",
                "}",
            ],
        )
        compact = parse_c_statements(
            [
                "while(pending) step();",
                "for(;;) tick();",
                "if(ready) continue;",
                "do {",
                "  step();",
                "} while (ready);",
            ]
        )
        self.assertIsInstance(compact.statements[0], CWhile)
        self.assertIsInstance(compact.statements[1], CFor)
        self.assertTrue(compact.statements[1].structured)
        self.assertIsInstance(compact.statements[2], CIf)
        self.assertIsInstance(compact.statements[3], CDoWhile)

    def test_expression_parser_structures_simple_compound_literals(self) -> None:
        body = parse_c_statements(
            ["value = ((union { int i; float f; }){ .i = 1 });"]
        )
        assignment = body.statements[0]
        self.assertIsInstance(assignment, CAssignment)
        self.assertIsInstance(assignment.value, CCompoundLiteral)
        self.assertTrue(assignment.value.type_name.startswith("union {"))
        self.assertIsInstance(
            assignment.value.initializer.entries[0], CDesignatedInitializer
        )
        self.assertEqual(
            CEmitter().emit_statement(assignment),
            ["value = ((union { int i; float f; }){ .i = 1 });"],
        )
        alias_literal = parse_c_statements(
            ["return (Payload){ .bytes = { [0] = 1 } };"]
        ).statements[0].value
        self.assertIsInstance(alias_literal, CCompoundLiteral)
        self.assertIsInstance(alias_literal.type_ast.base, CTypedefType)

    def test_string_character_and_adjacent_literals_use_typed_nodes(self) -> None:
        body = parse_c_statements(
            [
                'text = "Mini" "Ghidra";',
                'wide = L"line\\n";',
                "letter = '\\x41';",
                "wide_letter = U'\\u03bb';",
            ]
        )
        text, wide, letter, wide_letter = body.statements
        self.assertIsInstance(text.value, CString)
        self.assertEqual(text.value.value, "MiniGhidra")
        self.assertEqual(text.value.literal, '"Mini""Ghidra"')
        self.assertIsInstance(wide.value, CString)
        self.assertEqual(wide.value.prefix, "L")
        self.assertIsInstance(letter.value, CCharacter)
        self.assertEqual(letter.value.literal, "'\\x41'")
        self.assertIsInstance(wide_letter.value, CCharacter)
        self.assertEqual(wide_letter.value.prefix, "U")

    def test_sizeof_and_alignof_distinguish_type_and_expression_operands(self) -> None:
        body = parse_c_statements(
            [
                "return sizeof(uint32_t *);",
                "size = sizeof(value);",
                "return _Alignof(MyType);",
                "return sizeof(MyType *);",
            ]
        )
        size_type = body.statements[0].value
        size_expression = body.statements[1].value
        align_type = body.statements[2].value
        typedef_pointer = body.statements[3].value
        self.assertIsInstance(size_type, CSizeof)
        self.assertIsInstance(size_type.operand, CTypeName)
        self.assertEqual(CEmitter().emit_expression(size_type), "sizeof(uint32_t *)")
        self.assertIsInstance(size_expression, CSizeof)
        self.assertIsInstance(size_expression.operand, CIdentifier)
        self.assertIsInstance(align_type, CSizeof)
        self.assertEqual(align_type.operator, "_Alignof")
        self.assertIsInstance(align_type.operand.base, CTypedefType)
        self.assertIsInstance(typedef_pointer.operand.declarator, CDeclarator)

    def test_expression_parser_preserves_unsupported_forms_raw(self) -> None:
        body = parse_c_statements(["value = (MyType)value;"])
        assignment = body.statements[0]
        self.assertIsInstance(assignment, CAssignment)
        self.assertIsInstance(assignment.value, CRawExpression)

    def test_shift_compound_assignment_is_structured(self) -> None:
        body = parse_c_statements(["value >>= 2;"])
        assignment = body.statements[0]
        self.assertIsInstance(assignment, CAssignment)
        self.assertEqual(assignment.operator, ">>=")

    def test_c_octal_literal_keeps_c_spelling(self) -> None:
        body = parse_c_statements(["return 010;"])
        self.assertEqual(CEmitter().emit_statement(body.statements[0]), ["return 010;"])


if __name__ == "__main__":
    unittest.main()
