from __future__ import annotations

import unittest

from minighidra.c_ast import (
    CBinary,
    CCall,
    CCast,
    CConditional,
    CEmitter,
    CIdentifier,
    CInteger,
    CMember,
    CParenthesized,
    CSizeof,
    CUnary,
    parse_c_type_name,
)
from minighidra.printlanguage import (
    OpToken,
    OpTokenKind,
    PcodeSymbolResolver,
    PrintLanguageRpn,
    PrintToken,
    RpnTokenStack,
)
from minighidra.sleigh import SleighPcodeOp, SleighVarnode
from minighidra.ssa import SSAFunction, SSAOperand, SSAOperation, SSALocation, SSAValue


class RpnTokenStackTests(unittest.TestCase):
    def test_binary_operator_advances_through_two_operand_stages(self) -> None:
        stack = RpnTokenStack()
        stack.push_operator(
            OpToken(
                "+",
                stage=2,
                precedence=50,
                associative=True,
                kind=OpTokenKind.BINARY,
                spacing=1,
            )
        )
        stack.push_atom(PrintToken("left", "identifier"))
        self.assertEqual(stack.depth, 1)
        stack.push_atom(PrintToken("right", "identifier"))
        self.assertEqual(stack.depth, 0)
        self.assertEqual(stack.finish(), "left + right")
        self.assertEqual(
            stack.push_trace,
            [
                ("operator", "+"),
                ("atom", "left"),
                ("atom", "right"),
            ],
        )

    def test_finish_rejects_incomplete_operator_frames(self) -> None:
        stack = RpnTokenStack()
        stack.push_operator(
            OpToken("-", stage=2, kind=OpTokenKind.BINARY, name="subtract")
        )
        with self.assertRaisesRegex(ValueError, "subtract"):
            stack.finish()

    def test_rpn_printer_matches_printc_precedence_and_associativity(self) -> None:
        emitter = CEmitter(expression_mode="rpn")
        cases = (
            (
                CBinary(
                    "+",
                    CIdentifier("left"),
                    CBinary("*", CIdentifier("right"), CIdentifier("scale")),
                ),
                "left + right * scale",
            ),
            (
                CBinary(
                    "*",
                    CIdentifier("left"),
                    CBinary("+", CIdentifier("right"), CIdentifier("scale")),
                ),
                "left * (right + scale)",
            ),
            (
                CBinary(
                    "-",
                    CIdentifier("left"),
                    CBinary("-", CIdentifier("right"), CIdentifier("scale")),
                ),
                "left - (right - scale)",
            ),
            (
                CParenthesized(
                    CBinary("+", CIdentifier("left"), CIdentifier("right"))
                ),
                "(left + right)",
            ),
            (CUnary("-", CUnary("-", CIdentifier("value"))), "- -value"),
            (CUnary("+", CUnary("+", CIdentifier("value"))), "+ +value"),
            (CUnary("&", CUnary("&", CIdentifier("value"))), "& &value"),
        )
        for expression, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(emitter.emit_expression(expression), expected)

    def test_rpn_printer_handles_surround_cast_member_call_and_sizeof_tokens(self) -> None:
        emitter = CEmitter(expression_mode="rpn")
        expression = CCall(
            CUnary("*", CIdentifier("callback")),
            (
                CCast("uint32_t", CIdentifier("value")),
                CMember(CIdentifier("record"), "field"),
            ),
        )
        self.assertEqual(
            emitter.emit_expression(expression),
            "(*callback)((uint32_t)value,record.field)",
        )
        self.assertEqual(
            emitter.emit_expression(CSizeof(parse_c_type_name("int (*)(void)"))),
            "sizeof(int (*)(void))",
        )

    def test_rpn_printer_preserves_conditional_right_associativity(self) -> None:
        expression = CConditional(
            CIdentifier("first"),
            CInteger(1),
            CConditional(CIdentifier("second"), CInteger(2), CInteger(3)),
        )
        self.assertEqual(
            CEmitter(expression_mode="rpn").emit_expression(expression),
            "first ? 1 : second ? 2 : 3",
        )

    def test_rpn_printer_exposes_emitted_token_records(self) -> None:
        printer = PrintLanguageRpn(CEmitter(expression_mode="rpn"))
        self.assertEqual(
            printer.emit_expression(CBinary("+", CInteger(1), CInteger(2))),
            "1 + 2",
        )
        self.assertEqual(
            [token.kind for token in printer.last_tokens],
            ["literal", "space", "operator", "space", "literal"],
        )

    def test_direct_pcode_walk_resolves_ssa_inputs_and_keeps_markup(self) -> None:
        pcode = SleighPcodeOp(
            0x401000,
            7,
            "INT_ADD",
            (
                SleighVarnode("register", 0, 4, "EAX"),
                SleighVarnode("register", 4, 4, "EBX"),
            ),
            SleighVarnode("unique", 0x20, 4, "unique[20]"),
        )
        operation = SSAOperation(
            index=0,
            block_id=0,
            instruction_address=0x401000,
            instruction_mnemonic="add",
            sequence=7,
            opcode="INT_ADD",
            inputs=(
                SSAOperand("value", 32, values=(0,)),
                SSAOperand("value", 32, values=(1,)),
            ),
            output_value=2,
            pcode=pcode,
        )
        function = SSAFunction(
            address=0x401000,
            blocks=(),
            values=(
                SSAValue(
                    0, "input", "input_eax", 32,
                    storage=SSALocation("register", 0),
                ),
                SSAValue(
                    1, "input", "input_ebx", 32,
                    storage=SSALocation("register", 4),
                ),
                SSAValue(
                    2, "operation", "int_add", 32,
                    instruction_address=0x401000,
                    operation_sequence=7,
                    opcode="INT_ADD",
                ),
            ),
            phis=(),
            operations=(operation,),
            entry_blocks=(),
            live_in=(),
            live_out=(),
        )
        printer = PrintLanguageRpn(CEmitter(expression_mode="rpn"))
        symbols = PcodeSymbolResolver(value_names={0: "param_1", 1: "param_2"})
        self.assertEqual(
            printer.emit_ssa_value(function, 2, symbols=symbols),
            "param_1 + param_2",
        )
        operator = next(token for token in printer.last_tokens if token.kind == "operator")
        self.assertEqual(operator.metadata.address, 0x401000)
        self.assertEqual(operator.metadata.sequence, 7)
        self.assertEqual(operator.metadata.opcode, "INT_ADD")
        self.assertIn(
            'class="pcode-operator" title="INT_ADD, 0x401000, seq 7"',
            printer.emit_html(),
        )

    def test_direct_pcode_dispatch_handles_indirect_calls_and_memory(self) -> None:
        pointer = SleighVarnode("register", 0, 8, "RAX")
        argument = SleighVarnode("register", 8, 8, "RDI")
        load = SSAOperation(
            index=0,
            block_id=0,
            instruction_address=0x401000,
            instruction_mnemonic="mov",
            sequence=1,
            opcode="LOAD",
            inputs=(
                SSAOperand("constant", 32, constant=0),
                SSAOperand("value", 64, values=(0,)),
            ),
            output_value=2,
            pcode=SleighPcodeOp(
                0x401000,
                1,
                "LOAD",
                (SleighVarnode("const", 0, 4, "0"), pointer),
                SleighVarnode("unique", 0x10, 4, "unique[10]"),
            ),
        )
        indirect_call = SSAOperation(
            index=1,
            block_id=0,
            instruction_address=0x401004,
            instruction_mnemonic="call",
            sequence=2,
            opcode="CALLIND",
            inputs=(
                SSAOperand("value", 64, values=(3,)),
                SSAOperand("value", 64, values=(1,)),
            ),
            pcode=SleighPcodeOp(
                0x401004,
                2,
                "CALLIND",
                (pointer, argument),
                None,
            ),
        )
        store = SSAOperation(
            index=2,
            block_id=0,
            instruction_address=0x401008,
            instruction_mnemonic="mov",
            sequence=3,
            opcode="STORE",
            inputs=(
                SSAOperand("constant", 32, constant=0),
                SSAOperand("value", 64, values=(3,)),
                SSAOperand("constant", 32, constant=7),
            ),
            pcode=SleighPcodeOp(
                0x401008,
                3,
                "STORE",
                (
                    SleighVarnode("const", 0, 4, "0"),
                    pointer,
                    SleighVarnode("const", 7, 4, "7"),
                ),
                None,
            ),
        )
        function = SSAFunction(
            address=0x401000,
            blocks=(),
            values=(
                SSAValue(0, "input", "ptr", 64),
                SSAValue(1, "input", "arg", 64),
                SSAValue(2, "operation", "load", 32),
                SSAValue(3, "input", "callback", 64),
            ),
            phis=(),
            operations=(load, indirect_call, store),
            entry_blocks=(),
            live_in=(),
            live_out=(),
        )
        printer = PrintLanguageRpn(CEmitter(expression_mode="rpn"))
        symbols = PcodeSymbolResolver(
            value_names={0: "ptr", 1: "arg", 3: "callback"}
        )
        self.assertEqual(
            printer.emit_ssa_operation(function, 0, symbols=symbols),
            "*(uint32_t *)ptr",
        )
        self.assertEqual(
            printer.emit_ssa_operation(function, 1, symbols=symbols),
            "(*callback)(arg)",
        )
        self.assertEqual(
            printer.emit_ssa_operation(function, 2, symbols=symbols),
            "*(uint32_t *)callback = 7",
        )

    def test_direct_pcode_dispatch_resolves_calls_and_has_opaque_fallbacks(self) -> None:
        register = SleighVarnode("register", 0, 4, "EAX")
        target = SleighVarnode("ram", 0x402000, 8, "ram[402000]")
        call = SSAOperation(
            index=0,
            block_id=0,
            instruction_address=0x401000,
            instruction_mnemonic="call",
            sequence=3,
            opcode="CALL",
            inputs=(
                SSAOperand("address", 64, address=0x402000),
                SSAOperand("value", 32, values=(0,)),
            ),
            pcode=SleighPcodeOp(0x401000, 3, "CALL", (target, register), None),
        )
        unknown = SSAOperation(
            index=1,
            block_id=0,
            instruction_address=0x401004,
            instruction_mnemonic="custom",
            sequence=0,
            opcode="INT_CUSTOM",
            inputs=(SSAOperand("value", 32, values=(0,)),),
            output_value=1,
            pcode=SleighPcodeOp(
                0x401004,
                0,
                "INT_CUSTOM",
                (register,),
                SleighVarnode("unique", 0x40, 4, "unique[40]"),
            ),
        )
        function = SSAFunction(
            address=0x401000,
            blocks=(),
            values=(
                SSAValue(0, "input", "eax", 32, storage=SSALocation("register", 0)),
                SSAValue(
                    1, "operation", "custom", 32,
                    instruction_address=0x401004,
                    operation_sequence=0,
                    opcode="INT_CUSTOM",
                ),
            ),
            phis=(),
            operations=(call, unknown),
            entry_blocks=(),
            live_in=(),
            live_out=(),
        )
        printer = PrintLanguageRpn(CEmitter(expression_mode="rpn"))
        symbols = PcodeSymbolResolver(
            address_names={0x402000: "target_function"},
            value_names={0: "param_1"},
            function_addresses={0x402000},
        )
        self.assertEqual(
            printer.emit_ssa_operation(function, 0, symbols=symbols),
            "target_function(param_1)",
        )
        self.assertEqual(
            printer.emit_ssa_value(function, 1, symbols=symbols),
            "pcode_int_custom(param_1)",
        )
        self.assertIn("pcode-fallback", printer.emit_html())


if __name__ == "__main__":
    unittest.main()
