"""RPN-based C token emission modeled on Ghidra's PrintLanguage/PrintC.

Ghidra's C renderer does not first build a source-level C expression tree. It
pushes operators and atoms while walking p-code, then resolves precedence and
parentheses with a reverse-polish operator stack. MiniGhidraPy retains a Python
C AST path and also exposes a direct walk from its SSA-linked SLEIGH operations;
both paths feed the same token stack, with explicit generic fallbacks for
untranslated opcodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from html import escape as html_escape
import re
from typing import TYPE_CHECKING, Iterable, Mapping, TypeAlias

from .c_ast import (
    CBinary,
    CCall,
    CCharacter,
    CCast,
    CCompoundLiteral,
    CConditional,
    CExpression,
    CFloat,
    CIdentifier,
    CIndex,
    CInteger,
    CMember,
    CParenthesized,
    CRawExpression,
    CSizeof,
    CString,
    CUnary,
    CTypeName,
)

if TYPE_CHECKING:
    from .c_ast import CEmitter
    from .sleigh import SleighVarnode
    from .ssa import SSAFunction, SSAOperand, SSAOperation, SSAValue


class OpTokenKind(Enum):
    """Operator forms represented by Ghidra's ``OpToken::tokentype``."""

    BINARY = "binary"
    UNARY_PREFIX = "unary_prefix"
    POST_SURROUND = "postsurround"
    PRE_SURROUND = "presurround"
    SPACE = "space"
    HIDDEN_FUNCTION = "hiddenfunction"
    POSTFIX = "postfix"
    CONDITIONAL = "conditional"
    GROUP = "group"


@dataclass(frozen=True, slots=True)
class OpToken:
    """Printing and precedence metadata for one language operator."""

    print1: str
    print2: str = ""
    stage: int = 2
    precedence: int = 0
    associative: bool = False
    kind: OpTokenKind = OpTokenKind.BINARY
    spacing: int = 0
    bump: int = 0
    name: str = ""
    force_parentheses: bool = False

    def __post_init__(self) -> None:
        if self.stage < 1:
            raise ValueError("an operator token must consume at least one stage")
        if self.spacing < 0 or self.bump < 0:
            raise ValueError("operator spacing and bump must be non-negative")


@dataclass(frozen=True, slots=True)
class PrintToken:
    """One output token (the Python counterpart of Ghidra's ``Atom``)."""

    text: str
    kind: str = "syntax"
    metadata: object | None = None


@dataclass(frozen=True, slots=True)
class PcodeTokenMetadata:
    """Source coordinates retained for a p-code-derived output token."""

    address: int | None = None
    sequence: int | None = None
    opcode: str | None = None
    space: str | None = None
    offset: int | None = None
    size: int | None = None
    symbol: str | None = None


@dataclass(frozen=True, slots=True)
class PcodeExpression:
    """A tiny expression node that keeps p-code dispatch separate from C AST."""

    token: OpToken
    children: tuple[PrintToken | PcodeExpression, ...]
    metadata: PcodeTokenMetadata | None = None


@dataclass(slots=True)
class ReversePolish:
    """An active operator entry mirroring ``PrintLanguage::ReversePolish``."""

    token: OpToken
    visited: int = 0
    parenthesized: bool = False
    metadata: PcodeTokenMetadata | None = None


class RpnTokenStack:
    """Incremental precedence-aware token emitter using an operator stack."""

    def __init__(self) -> None:
        self.stack: list[ReversePolish] = []
        self.tokens: list[PrintToken] = []
        self.push_trace: list[tuple[str, str]] = []

    @property
    def depth(self) -> int:
        return len(self.stack)

    @property
    def source(self) -> str:
        return "".join(token.text for token in self.tokens)

    def clear(self) -> None:
        self.stack.clear()
        self.tokens.clear()
        self.push_trace.clear()

    def push_operator(
        self,
        token: OpToken,
        *,
        metadata: PcodeTokenMetadata | None = None,
    ) -> None:
        if self.stack:
            previous = self.stack[-1].token
            self._emit_operator(self.stack[-1])
            parenthesized = token.force_parentheses or self._parentheses(token)
            if (
                not parenthesized
                and previous.kind is OpTokenKind.UNARY_PREFIX
                and token.kind is OpTokenKind.UNARY_PREFIX
                and previous.print1 + token.print1 in {"++", "--", "&&"}
            ):
                self._spaces(1)
        else:
            parenthesized = token.force_parentheses
        if parenthesized:
            self._append("(", "punctuation", metadata)
        self.stack.append(
            ReversePolish(token, parenthesized=parenthesized, metadata=metadata)
        )
        self.push_trace.append(("operator", token.name or token.print1))

    def push_atom(self, atom: PrintToken | str) -> None:
        if isinstance(atom, str):
            atom = PrintToken(atom)
        self.push_trace.append(("atom", atom.text))
        if not self.stack:
            self._emit_atom(atom)
            return

        self._emit_operator(self.stack[-1])
        self._emit_atom(atom)
        while self.stack:
            entry = self.stack[-1]
            entry.visited += 1
            if entry.visited != entry.token.stage:
                break
            self._emit_operator(entry)
            if entry.parenthesized:
                self._append(")", "punctuation", entry.metadata)
            self.stack.pop()

    def finish(self) -> str:
        if self.stack:
            pending = ", ".join(
                entry.token.name or entry.token.print1 or entry.token.kind.value
                for entry in self.stack
            )
            raise ValueError(f"unfinished RPN operator stack: {pending}")
        return self.source

    def _append(
        self,
        text: str,
        kind: str,
        metadata: PcodeTokenMetadata | None = None,
    ) -> None:
        if text:
            self.tokens.append(PrintToken(text, kind, metadata))

    def _spaces(
        self, count: int, metadata: PcodeTokenMetadata | None = None
    ) -> None:
        self._append(" " * count, "space", metadata)

    def _emit_atom(self, atom: PrintToken) -> None:
        if atom.text:
            self.tokens.append(atom)

    def _emit_operator(self, entry: ReversePolish) -> None:
        token = entry.token
        metadata = entry.metadata
        visited = entry.visited
        if token.kind is OpTokenKind.BINARY:
            if visited != 1:
                return
            self._spaces(token.spacing, metadata)
            self._append(token.print1, "operator", metadata)
            self._spaces(token.spacing, metadata)
        elif token.kind is OpTokenKind.UNARY_PREFIX:
            if visited != 0:
                return
            self._append(token.print1, "operator", metadata)
            self._spaces(token.spacing, metadata)
        elif token.kind is OpTokenKind.POST_SURROUND:
            if visited == 0:
                return
            if visited == 1:
                self._spaces(token.spacing, metadata)
                self._append(token.print1, "punctuation", metadata)
                self._spaces(0, metadata)
            else:
                self._append(token.print2, "punctuation", metadata)
        elif token.kind is OpTokenKind.PRE_SURROUND:
            if visited == 0:
                self._append(token.print1, "punctuation", metadata)
            elif visited != 2:
                self._append(token.print2, "punctuation", metadata)
                self._spaces(token.spacing, metadata)
        elif token.kind is OpTokenKind.SPACE:
            if visited == 1:
                self._spaces(token.spacing, metadata)
        elif token.kind is OpTokenKind.HIDDEN_FUNCTION:
            return
        elif token.kind is OpTokenKind.POSTFIX:
            if visited == 1:
                self._spaces(token.spacing, metadata)
                self._append(token.print1, "operator", metadata)
        elif token.kind is OpTokenKind.CONDITIONAL:
            if visited == 1:
                self._spaces(token.spacing, metadata)
                self._append("?", "operator", metadata)
                self._spaces(token.spacing, metadata)
            elif visited == 2:
                self._spaces(token.spacing, metadata)
                self._append(":", "operator", metadata)
                self._spaces(token.spacing, metadata)
        elif token.kind is OpTokenKind.GROUP:
            return

    def _parentheses(self, new_token: OpToken) -> bool:
        top = self.stack[-1]
        current = top.token
        stage = top.visited

        if current.kind in {OpTokenKind.SPACE, OpTokenKind.BINARY}:
            if current.precedence > new_token.precedence:
                return True
            if current.precedence < new_token.precedence:
                return False
            if current.associative and current is new_token:
                return False
            if (
                new_token.kind is OpTokenKind.POST_SURROUND
                and stage == 0
            ):
                return False
            return True
        if current.kind is OpTokenKind.UNARY_PREFIX:
            if current.precedence > new_token.precedence:
                return True
            if current.precedence < new_token.precedence:
                return False
            return new_token.kind not in {
                OpTokenKind.UNARY_PREFIX,
                OpTokenKind.PRE_SURROUND,
            }
        if current.kind is OpTokenKind.POST_SURROUND:
            if stage == 1:
                return False
            if current.precedence > new_token.precedence:
                return True
            if current.precedence < new_token.precedence:
                return False
            return new_token.kind not in {
                OpTokenKind.POST_SURROUND,
                OpTokenKind.BINARY,
            }
        if current.kind is OpTokenKind.PRE_SURROUND:
            if stage == 0:
                return False
            if current.precedence > new_token.precedence:
                return True
            if current.precedence < new_token.precedence:
                return False
            return new_token.kind not in {
                OpTokenKind.UNARY_PREFIX,
                OpTokenKind.PRE_SURROUND,
            }
        if current.kind is OpTokenKind.HIDDEN_FUNCTION:
            if stage == 0 and len(self.stack) > 1:
                previous = self.stack[-2].token
                if previous.kind not in {
                    OpTokenKind.BINARY,
                    OpTokenKind.UNARY_PREFIX,
                }:
                    return False
                if previous.precedence < new_token.precedence:
                    return False
            return True
        if current.kind is OpTokenKind.CONDITIONAL:
            if current.precedence > new_token.precedence:
                return True
            if current.precedence < new_token.precedence:
                return False
            # C's false arm is right-associative; a nested condition is not.
            return not (
                stage == 2 and new_token.kind is OpTokenKind.CONDITIONAL
            )
        if current.kind is OpTokenKind.GROUP:
            return False
        if current.kind is OpTokenKind.POSTFIX:
            if current.precedence > new_token.precedence:
                return True
            if current.precedence < new_token.precedence:
                return False
            return new_token.kind not in {
                OpTokenKind.POST_SURROUND,
                OpTokenKind.BINARY,
            }
        return True


# The values mirror the corresponding OpToken constants in printc.cc.
_ASSIGNMENT_OPERATORS = frozenset(
    {"=", "*=", "/=", "%=", "+=", "-=", "<<=", ">>=", "&=", "|=", "^="}
)
_BINARY_TOKENS: dict[str, OpToken] = {}
for _operator, _precedence, _associative in (
    ("*", 54, True),
    ("/", 54, False),
    ("%", 54, False),
    ("+", 50, True),
    ("-", 50, False),
    ("<<", 46, False),
    (">>", 46, False),
    ("<", 42, False),
    ("<=", 42, False),
    (">", 42, False),
    (">=", 42, False),
    ("==", 38, False),
    ("!=", 38, False),
    ("&", 34, True),
    ("^", 30, True),
    ("|", 26, True),
    ("&&", 22, False),
    ("^^", 20, False),
    ("||", 18, False),
    ("=", 14, False),
    ("*=", 14, False),
    ("/=", 14, False),
    ("%=", 14, False),
    ("+=", 14, False),
    ("-=", 14, False),
    ("<<=", 14, False),
    (">>=", 14, False),
    ("&=", 14, False),
    ("|=", 14, False),
    ("^=", 14, False),
    (",", 2, True),
):
    _BINARY_TOKENS[_operator] = OpToken(
        _operator,
        stage=2,
        precedence=_precedence,
        associative=_associative,
        kind=OpTokenKind.BINARY,
        spacing=0 if _operator == "," else 1,
        bump=5 if _operator in _ASSIGNMENT_OPERATORS else 0,
        name=_operator,
    )

_MEMBER_TOKENS = {
    ".": OpToken(
        ".", stage=2, precedence=66, associative=True, kind=OpTokenKind.BINARY, name="."
    ),
    "->": OpToken(
        "->", stage=2, precedence=66, associative=True, kind=OpTokenKind.BINARY, name="->"
    ),
}

_UNARY_PRECEDENCE = 62
_POSTFIX_PRECEDENCE = 66
_CALL_TOKEN = OpToken(
    "(", ")", 2, 66, False, OpTokenKind.POST_SURROUND, 0, 10, "call"
)
_INDEX_TOKEN = OpToken(
    "[", "]", 2, 66, False, OpTokenKind.POST_SURROUND, 0, 0, "index"
)
_CAST_TOKEN = OpToken(
    "(", ")", 2, 62, False, OpTokenKind.PRE_SURROUND, 0, 0, "cast"
)
_CONDITIONAL_TOKEN = OpToken(
    "?", ":", 3, 16, False, OpTokenKind.CONDITIONAL, 1, 0, "conditional"
)


_PendingExpression: TypeAlias = CExpression | PrintToken | PcodeExpression


class PrintLanguageRpn:
    """RPN token printer for C AST expressions and SSA-linked p-code values.

    ``CEmitter(expression_mode="rpn")`` uses this class for both AST emission
    and direct p-code operation/value rendering, without replacing the AST API.
    """

    def __init__(self, emitter: CEmitter) -> None:
        self.emitter = emitter
        self.stack = RpnTokenStack()
        self._pending: list[_PendingExpression] = []

    @property
    def last_tokens(self) -> tuple[PrintToken, ...]:
        return tuple(self.stack.tokens)

    @property
    def last_push_trace(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.stack.push_trace)

    def emit_expression(self, expression: CExpression) -> str:
        self.stack.clear()
        self._pending.clear()
        self._pending.append(expression)
        self.recurse()
        return self.stack.finish()

    def emit_ssa_value(
        self,
        function: SSAFunction,
        value_id: int,
        *,
        symbols: PcodeSymbolResolver | None = None,
    ) -> str:
        """Print a value by walking its SLEIGH ``PcodeOp``/``Varnode`` graph."""
        builder = _PcodeExpressionBuilder(function, symbols or PcodeSymbolResolver())
        return self._emit_pcode_node(builder.value(value_id))

    def emit_ssa_operand(
        self,
        function: SSAFunction,
        operand: SSAOperand,
        *,
        varnode: SleighVarnode | None = None,
        symbols: PcodeSymbolResolver | None = None,
    ) -> str:
        """Render a whole p-code varnode operand, combining byte SSA slices."""
        builder = _PcodeExpressionBuilder(function, symbols or PcodeSymbolResolver())
        return self._emit_pcode_node(builder.operand(operand, varnode=varnode))

    def emit_ssa_operation(
        self,
        function: SSAFunction,
        operation: int | SSAOperation,
        *,
        symbols: PcodeSymbolResolver | None = None,
        resolve_definitions: bool = True,
    ) -> str:
        """Dispatch one SSA-linked p-code operation, including opaque fallbacks."""
        builder = _PcodeExpressionBuilder(
            function,
            symbols or PcodeSymbolResolver(),
            resolve_definitions=resolve_definitions,
        )
        if isinstance(operation, int):
            operation_index = operation
            operation = next(
                (item for item in function.operations if item.index == operation_index),
                None,
            )
            if operation is None:
                raise KeyError(
                    f"no SSA p-code operation with index {operation_index}"
                )
        return self._emit_pcode_node(builder.operation(operation))

    def _emit_pcode_node(self, expression: PcodeExpression | PrintToken) -> str:
        self.stack.clear()
        self._pending.clear()
        self._pending.append(expression)
        self.recurse()
        return self.stack.finish()

    def emit_html(self) -> str:
        """Render the last token stream with semantic markup classes."""
        chunks: list[str] = []
        for token in self.stack.tokens:
            text = html_escape(token.text, quote=False)
            if token.kind in {"space", "blank"}:
                chunks.append(text)
                continue
            kind = re.sub(r"[^a-z0-9_-]", "-", token.kind.lower()) or "syntax"
            title = ""
            if isinstance(token.metadata, PcodeTokenMetadata):
                details = []
                if token.metadata.opcode:
                    details.append(token.metadata.opcode)
                if token.metadata.address is not None:
                    details.append(f"0x{token.metadata.address:x}")
                if token.metadata.sequence is not None:
                    details.append(f"seq {token.metadata.sequence}")
                if details:
                    title = f' title="{html_escape(", ".join(details), quote=True)}"'
            chunks.append(f'<span class="pcode-{kind}"{title}>{text}</span>')
        return "".join(chunks)

    def recurse(self) -> None:
        """Drain pending expression nodes in stack order, as ``PrintLanguage`` does."""
        while self._pending:
            expression = self._pending.pop()
            if isinstance(expression, PrintToken):
                self.stack.push_atom(expression)
            elif isinstance(expression, PcodeExpression):
                self.stack.push_operator(expression.token, metadata=expression.metadata)
                self._queue_children(*expression.children)
            else:
                self._push_expression(expression)

    def _queue_children(self, *children: _PendingExpression) -> None:
        self._pending.extend(reversed(children))

    def _push_expression(self, expression: CExpression) -> None:
        if isinstance(expression, CBinary):
            token = _BINARY_TOKENS.get(expression.operator)
            if token is None:
                self.stack.push_atom(self._direct_atom(expression, "raw"))
                return
            self.stack.push_operator(token)
            self._queue_children(expression.left, expression.right)
            return
        if isinstance(expression, CConditional):
            self.stack.push_operator(_CONDITIONAL_TOKEN)
            self._queue_children(
                expression.condition, expression.if_true, expression.if_false
            )
            return
        if isinstance(expression, CUnary):
            if expression.postfix:
                token = OpToken(
                    expression.operator,
                    stage=1,
                    precedence=_POSTFIX_PRECEDENCE,
                    kind=OpTokenKind.POSTFIX,
                    name=f"post{expression.operator}",
                )
            else:
                token = OpToken(
                    expression.operator,
                    stage=1,
                    precedence=_UNARY_PRECEDENCE,
                    kind=OpTokenKind.UNARY_PREFIX,
                    name=f"pre{expression.operator}",
                )
            self.stack.push_operator(token)
            self._queue_children(expression.operand)
            return
        if isinstance(expression, CParenthesized):
            token = OpToken(
                "(",
                ")",
                stage=1,
                precedence=100,
                kind=OpTokenKind.GROUP,
                name="explicit-group",
                force_parentheses=True,
            )
            self.stack.push_operator(token)
            self._queue_children(expression.value)
            return
        if isinstance(expression, CCast):
            self.stack.push_operator(_CAST_TOKEN)
            type_name = self.emitter._direct_emitter._emit_type_name(
                expression.type_name, expression.type_ast
            )
            self.stack.push_atom(PrintToken(type_name, "type", expression.type_ast))
            self._queue_children(expression.value)
            return
        if isinstance(expression, CSizeof):
            if isinstance(expression.operand, CTypeName):
                operand = self.emitter._direct_emitter._emit_type_specifier(
                    expression.operand
                )
                pending: _PendingExpression = PrintToken(
                    operand, "type", expression.operand
                )
            else:
                pending = expression.operand
            token = OpToken(
                f"{expression.operator}(",
                ")",
                stage=1,
                precedence=_UNARY_PRECEDENCE,
                kind=OpTokenKind.PRE_SURROUND,
                name=expression.operator,
            )
            self.stack.push_operator(token)
            self._queue_children(pending)
            return
        if isinstance(expression, CMember):
            token = _MEMBER_TOKENS["->" if expression.pointer else "."]
            self.stack.push_operator(token)
            self._queue_children(
                expression.value, PrintToken(expression.member, "field")
            )
            return
        if isinstance(expression, CIndex):
            self.stack.push_operator(_INDEX_TOKEN)
            self._queue_children(expression.value, expression.index)
            return
        if isinstance(expression, CCall):
            self.stack.push_operator(_CALL_TOKEN)
            if expression.arguments:
                arguments = expression.arguments[0]
                for argument in expression.arguments[1:]:
                    arguments = CBinary(",", arguments, argument)
                self._queue_children(expression.function, arguments)
            else:
                self._queue_children(
                    expression.function, PrintToken("", "blank")
                )
            return
        if isinstance(expression, CCompoundLiteral):
            self.stack.push_atom(self._direct_atom(expression, "literal"))
            return
        if isinstance(expression, CIdentifier):
            self.stack.push_atom(PrintToken(expression.name, "identifier"))
            return
        if isinstance(expression, (CInteger, CFloat, CString, CCharacter)):
            self.stack.push_atom(self._direct_atom(expression, "literal"))
            return
        if isinstance(expression, CRawExpression):
            self.stack.push_atom(PrintToken(expression.source, "raw"))
            return
        self.stack.push_atom(self._direct_atom(expression, "raw"))

    def _direct_atom(self, expression: CExpression, kind: str) -> PrintToken:
        return PrintToken(
            self.emitter._direct_emitter._emit_expression_direct(expression), kind
        )


class PcodeSymbolResolver:
    """Resolve p-code storage and addresses to stable source-level token names."""

    _C_KEYWORDS = frozenset(
        {
            "auto", "break", "case", "char", "const", "continue", "default",
            "do", "double", "else", "enum", "extern", "float", "for",
            "goto", "if", "inline", "int", "long", "register", "restrict",
            "return", "short", "signed", "sizeof", "static", "struct",
            "switch", "typedef", "union", "unsigned", "void", "volatile",
            "while", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic",
            "_Imaginary", "_Noreturn", "_Static_assert", "_Thread_local",
        }
    )

    def __init__(
        self,
        *,
        address_names: Mapping[int, str] | None = None,
        register_names: Mapping[tuple[str, int, int], str] | None = None,
        storage_names: Mapping[tuple[str, int], str] | None = None,
        value_names: Mapping[int, str] | None = None,
        function_addresses: Iterable[int] = (),
        raw_register_names: bool = False,
    ) -> None:
        self.address_names = dict(address_names or {})
        self.register_names = dict(register_names or {})
        self.storage_names = dict(storage_names or {})
        self.value_names = dict(value_names or {})
        self.function_addresses = frozenset(function_addresses)
        self.raw_register_names = raw_register_names

    @staticmethod
    def _identifier(name: str, prefix: str = "sym") -> str:
        result = re.sub(r"[^A-Za-z0-9_]", "_", name)
        if not result or result[0].isdigit():
            result = f"{prefix}_{result}"
        if result in PcodeSymbolResolver._C_KEYWORDS:
            result = f"sym_{result}"
        return result

    @staticmethod
    def _with_symbol(
        metadata: PcodeTokenMetadata | None, name: str
    ) -> PcodeTokenMetadata:
        if metadata is None:
            return PcodeTokenMetadata(symbol=name)
        return PcodeTokenMetadata(
            metadata.address,
            metadata.sequence,
            metadata.opcode,
            metadata.space,
            metadata.offset,
            metadata.size,
            name,
        )

    def resolve_address(
        self,
        address: int,
        *,
        function: bool = False,
        metadata: PcodeTokenMetadata | None = None,
    ) -> PrintToken:
        symbol = self.address_names.get(address)
        if symbol:
            kind = "function" if function or address in self.function_addresses else "global"
            name = self._identifier(symbol)
        elif function or address in self.function_addresses:
            kind = "function"
            name = f"FUN_{address:X}"
        else:
            kind = "constant"
            name = f"0x{address:x}"
        return PrintToken(name, kind, self._with_symbol(metadata, name))

    def resolve_varnode(
        self,
        varnode: SleighVarnode,
        *,
        instruction_address: int | None = None,
        metadata: PcodeTokenMetadata | None = None,
        value_id: int | None = None,
        function: bool = False,
    ) -> PrintToken:
        if value_id is not None and value_id in self.value_names:
            name = self._identifier(self.value_names[value_id])
            return PrintToken(name, "variable", self._with_symbol(metadata, name))
        key = (varnode.space, varnode.offset, varnode.size)
        if key in self.register_names:
            name = self._identifier(self.register_names[key])
            return PrintToken(name, "variable", self._with_symbol(metadata, name))
        if varnode.space == "const":
            value = varnode.offset & ((1 << max(varnode.size_bits, 1)) - 1)
            return PrintToken(str(value), "constant", metadata)
        if varnode.space == "ram":
            return self.resolve_address(
                varnode.offset, function=function, metadata=metadata
            )
        storage_name = self._storage_range_name(
            varnode.space, varnode.offset, varnode.size
        )
        if storage_name:
            name = self._identifier(storage_name)
            return PrintToken(name, "variable", self._with_symbol(metadata, name))
        if varnode.space == "register":
            display = varnode.display.lower() or f"reg_{varnode.offset:x}"
            raw_name = display if self.raw_register_names else f"in_{display}"
            name = self._identifier(raw_name)
            return PrintToken(name, "variable", self._with_symbol(metadata, name))
        if varnode.space == "unique":
            scope = instruction_address if instruction_address is not None else 0
            name = f"unique_{scope:X}_{varnode.offset:X}"
            return PrintToken(name, "variable", self._with_symbol(metadata, name))
        name = self._identifier(f"{varnode.space}_{varnode.offset:X}")
        return PrintToken(name, "variable", self._with_symbol(metadata, name))

    def resolve_value(
        self,
        value: SSAValue,
        *,
        metadata: PcodeTokenMetadata | None = None,
    ) -> PrintToken:
        if value.identifier in self.value_names:
            name = self._identifier(self.value_names[value.identifier])
        elif value.kind == "input" and value.storage is not None:
            name = self._storage_name(value.storage.space, value.storage.offset)
            if name is None:
                name = self._identifier(value.name, "value")
        else:
            name = self._identifier(value.name, "value")
        kind = "constant" if value.kind == "constant" else "variable"
        return PrintToken(name, kind, self._with_symbol(metadata, name))

    def resolve_storage_range(
        self,
        space: str,
        offset: int,
        size: int,
        *,
        metadata: PcodeTokenMetadata | None = None,
    ) -> PrintToken | None:
        name = self._storage_range_name(space, offset, size)
        if name is None:
            return None
        name = self._identifier(name)
        return PrintToken(name, "variable", self._with_symbol(metadata, name))

    def _storage_name(self, space: str, offset: int) -> str | None:
        if (space, offset) in self.storage_names:
            return self.storage_names[(space, offset)]
        candidates = [
            (size, name)
            for (candidate_space, start, size), name in self.register_names.items()
            if candidate_space == space and start <= offset < start + size
        ]
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]
        return None

    def _storage_range_name(self, space: str, offset: int, size: int) -> str | None:
        names = [self._storage_name(space, offset + index) for index in range(size)]
        if names and names[0] is not None and all(name == names[0] for name in names):
            return names[0]
        return None


class _PcodeExpressionBuilder:
    """Walk SSA-resolved operands back to their originating SLEIGH p-code ops."""

    _BINARY_OPERATORS = {
        "INT_ADD": "+",
        "INT_SUB": "-",
        "INT_MULT": "*",
        "INT_DIV": "/",
        "INT_SDIV": "/",
        "INT_REM": "%",
        "INT_SREM": "%",
        "INT_AND": "&",
        "INT_OR": "|",
        "INT_XOR": "^",
        "INT_LEFT": "<<",
        "INT_RIGHT": ">>",
        "INT_SRIGHT": ">>",
        "INT_EQUAL": "==",
        "INT_NOTEQUAL": "!=",
        "INT_LESS": "<",
        "INT_LESSEQUAL": "<=",
        "INT_SLESS": "<",
        "INT_SLESSEQUAL": "<=",
        "BOOL_AND": "&&",
        "BOOL_OR": "||",
        "BOOL_XOR": "^^",
        "FLOAT_ADD": "+",
        "FLOAT_SUB": "-",
        "FLOAT_MULT": "*",
        "FLOAT_DIV": "/",
        "FLOAT_EQUAL": "==",
        "FLOAT_NOTEQUAL": "!=",
        "FLOAT_LESS": "<",
        "FLOAT_LESSEQUAL": "<=",
    }
    _UNARY_OPERATORS = {
        "INT_2COMP": "-",
        "INT_NEGATE": "~",
        "BOOL_NEGATE": "!",
        "FLOAT_NEG": "-",
    }
    _FLOAT_FUNCTIONS = {
        "FLOAT_ABS": "fabs",
        "FLOAT_CEIL": "ceil",
        "FLOAT_FLOOR": "floor",
        "FLOAT_ROUND": "round",
        "FLOAT_SQRT": "sqrt",
        "FLOAT_TRUNC": "trunc",
        "FLOAT_NAN": "isnan",
    }

    def __init__(
        self,
        function: SSAFunction,
        symbols: PcodeSymbolResolver,
        *,
        resolve_definitions: bool = True,
    ) -> None:
        self.function = function
        self.symbols = symbols
        self.resolve_definitions = resolve_definitions
        self.values = function.value_by_id
        self.operations_by_output = {
            operation.output_value: operation
            for operation in function.operations
            if operation.output_value is not None
        }
        self.call_operations_by_location = {
            (operation.instruction_address, operation.sequence): operation
            for operation in function.operations
            if operation.opcode in {"CALL", "CALLIND"}
        }
        self.phis_by_output = {
            phi.output_value: phi for phi in function.phis
        }
        self._active: set[int] = set()
        self._memo: dict[int, PrintToken | PcodeExpression] = {}
        self._nodes_built = 0

    def value(self, value_id: int) -> PrintToken | PcodeExpression:
        self._nodes_built += 1
        if (
            value_id in self._active
            or len(self._active) >= 128
            or self._nodes_built > 4096
        ):
            value = self.values[value_id]
            return self.symbols.resolve_value(value)
        if value_id in self._memo:
            return self._memo[value_id]
        value = self.values[value_id]
        metadata = self._value_metadata(value)
        self._active.add(value_id)
        try:
            if value.kind == "slice" and value.source is not None:
                result = self._slice_value(value, metadata)
            elif (
                value.kind == "call_return"
                and value.instruction_address is not None
                and value.operation_sequence is not None
                and (call := self.call_operations_by_location.get(
                    (value.instruction_address, value.operation_sequence)
                )) is not None
            ):
                result = self.operation(call)
            elif value_id in self.phis_by_output:
                result = self._phi_value(value, metadata)
            elif value_id in self.operations_by_output:
                result = self.operation(self.operations_by_output[value_id])
            else:
                result = self.symbols.resolve_value(value, metadata=metadata)
        finally:
            self._active.remove(value_id)
        self._memo[value_id] = result
        return result

    def operand(
        self,
        operand: SSAOperand,
        *,
        varnode: SleighVarnode | None = None,
    ) -> PrintToken | PcodeExpression:
        return self._operand_node(operand, varnode, None)

    def operation(self, operation: SSAOperation) -> PrintToken | PcodeExpression:
        opcode = operation.opcode.upper()
        metadata = self._operation_metadata(operation)
        if opcode == "IMARK":
            return PrintToken("", "blank", metadata)
        pcode = operation.pcode
        input_nodes = tuple(
            self._operand_node(
                operand,
                pcode.inputs[index] if pcode is not None and index < len(pcode.inputs) else None,
                operation,
                call_target=(index == 0 and opcode in {"CALL", "CALLIND"}),
            )
            for index, operand in enumerate(operation.inputs)
        )
        if opcode in self._BINARY_OPERATORS and len(input_nodes) >= 2:
            operator = self._BINARY_OPERATORS[opcode]
            token = _BINARY_TOKENS.get(operator)
            if token is not None:
                return PcodeExpression(token, input_nodes[:2], metadata)
        if opcode in self._UNARY_OPERATORS and input_nodes:
            token = OpToken(
                self._UNARY_OPERATORS[opcode],
                stage=1,
                precedence=_UNARY_PRECEDENCE,
                kind=OpTokenKind.UNARY_PREFIX,
                name=opcode.lower(),
            )
            return PcodeExpression(token, (input_nodes[0],), metadata)
        if opcode in {"COPY", "INDIRECT"} and input_nodes:
            return input_nodes[0]
        if opcode in {
            "INT_ZEXT",
            "INT_SEXT",
            "INT_TRUNC",
            "FLOAT_INT2FLOAT",
            "FLOAT_FLOAT2FLOAT",
            "FLOAT_TRUNC",
        } and input_nodes:
            if opcode == "FLOAT_INT2FLOAT" or opcode == "FLOAT_FLOAT2FLOAT":
                type_name = self._float_type(self._output_bits(operation))
            elif opcode == "FLOAT_TRUNC":
                type_name = self._integer_type(self._output_bits(operation), signed=True)
            else:
                type_name = self._integer_type(
                    self._output_bits(operation), signed=opcode == "INT_SEXT"
                )
            return self._cast(type_name, input_nodes[0], metadata)
        if opcode == "PTRADD" and len(input_nodes) >= 2:
            scale = (
                input_nodes[2]
                if len(input_nodes) > 2
                else PrintToken("1", "constant")
            )
            scaled_index = self._binary("*", input_nodes[1], scale, metadata)
            return self._binary("+", input_nodes[0], scaled_index, metadata)
        if opcode == "PTRSUB" and len(input_nodes) >= 2:
            return self._binary("+", input_nodes[0], input_nodes[1], metadata)
        if opcode == "PIECE" and len(input_nodes) >= 2:
            low_bits = (
                pcode.inputs[1].size_bits
                if pcode is not None and len(pcode.inputs) > 1
                else max(self._output_bits(operation) // 2, 8)
            )
            type_name = self._integer_type(self._output_bits(operation), signed=False)
            high = self._cast(type_name, input_nodes[0], metadata)
            shifted = self._binary(
                "<<", high, PrintToken(str(low_bits), "constant", metadata), metadata
            )
            low = self._cast(type_name, input_nodes[1], metadata)
            return self._binary("|", shifted, low, metadata)
        if opcode == "SUBPIECE" and len(input_nodes) >= 2:
            offset = (
                operation.inputs[1].constant
                if operation.inputs[1].kind == "constant"
                else 0
            )
            value = input_nodes[0]
            if offset:
                value = self._binary(
                    ">>",
                    value,
                    PrintToken(str(offset * 8), "constant", metadata),
                    metadata,
                )
            return self._cast(
                self._integer_type(self._output_bits(operation), signed=False),
                value,
                metadata,
            )
        if opcode == "LOAD" and len(input_nodes) >= 2:
            type_name = self._integer_type(self._output_bits(operation), signed=False)
            pointer = self._cast(f"{type_name} *", input_nodes[1], metadata)
            return PcodeExpression(
                OpToken(
                    "*",
                    stage=1,
                    precedence=_UNARY_PRECEDENCE,
                    kind=OpTokenKind.UNARY_PREFIX,
                    name="load",
                ),
                (pointer,),
                metadata,
            )
        if opcode == "STORE" and len(input_nodes) >= 3:
            bits = pcode.inputs[2].size_bits if pcode is not None else 8
            pointer = self._cast(
                f"{self._integer_type(bits, signed=False)} *", input_nodes[1], metadata
            )
            destination = PcodeExpression(
                OpToken(
                    "*",
                    stage=1,
                    precedence=_UNARY_PRECEDENCE,
                    kind=OpTokenKind.UNARY_PREFIX,
                    name="store",
                ),
                (pointer,),
                metadata,
            )
            return self._binary("=", destination, input_nodes[2], metadata)
        if opcode in {"CALL", "CALLIND"} and input_nodes:
            target = input_nodes[0]
            if opcode == "CALLIND":
                target = PcodeExpression(
                    OpToken(
                        "*",
                        stage=1,
                        precedence=_UNARY_PRECEDENCE,
                        kind=OpTokenKind.UNARY_PREFIX,
                        name="callind",
                    ),
                    (target,),
                    metadata,
                )
            return self._call(target, input_nodes[1:], metadata)
        if opcode == "CALLOTHER":
            userop = pcode.inputs[0].offset if pcode and pcode.inputs else operation.sequence
            name = PrintToken(f"userop_{userop:x}", "fallback", metadata)
            return self._call(name, input_nodes[1:], metadata)
        if opcode in self._FLOAT_FUNCTIONS and input_nodes:
            name = PrintToken(self._FLOAT_FUNCTIONS[opcode], "function", metadata)
            return self._call(name, input_nodes[:1], metadata)
        if opcode in {"POPCOUNT", "LZCOUNT"} and input_nodes:
            name = PrintToken(f"pcode_{opcode.lower()}", "fallback", metadata)
            return self._call(name, input_nodes[:1], metadata)
        return self._fallback(operation, input_nodes, metadata)

    def _operand_node(
        self,
        operand: SSAOperand,
        varnode: SleighVarnode | None,
        operation: SSAOperation | None,
        *,
        call_target: bool = False,
    ) -> PrintToken | PcodeExpression:
        metadata = (
            self._operation_metadata(operation, varnode)
            if operation is not None
            else PcodeTokenMetadata(
                space=varnode.space if varnode is not None else None,
                offset=varnode.offset if varnode is not None else None,
                size=varnode.size if varnode is not None else None,
            )
        )
        if operand.kind == "constant":
            return PrintToken(str(operand.constant or 0), "constant", metadata)
        if operand.kind == "address" and operand.address is not None:
            return self.symbols.resolve_address(
                operand.address, function=call_target, metadata=metadata
            )
        if operand.kind == "memory" and operand.address is not None:
            address = self.symbols.resolve_address(operand.address, metadata=metadata)
            type_name = self._integer_type(operand.size_bits, signed=False)
            pointer = self._cast(f"{type_name} *", address, metadata)
            return PcodeExpression(
                OpToken(
                    "*",
                    stage=1,
                    precedence=_UNARY_PRECEDENCE,
                    kind=OpTokenKind.UNARY_PREFIX,
                    name="memory",
                ),
                (pointer,),
                metadata,
            )
        if not self.resolve_definitions and varnode is not None:
            return self.symbols.resolve_varnode(
                varnode,
                instruction_address=(
                    operation.instruction_address if operation is not None else None
                ),
                metadata=metadata,
                function=call_target,
            )
        if not operand.values:
            if varnode is not None:
                return self.symbols.resolve_varnode(
                    varnode,
                    instruction_address=(
                        operation.instruction_address if operation is not None else None
                    ),
                    metadata=metadata,
                    function=call_target,
                )
            return PrintToken("unknown", "fallback", metadata)
        if varnode is not None and self._is_unmodified_input(operand.values, varnode):
            return self.symbols.resolve_varnode(
                varnode,
                instruction_address=(
                    operation.instruction_address if operation is not None else None
                ),
                metadata=metadata,
                function=call_target,
            )
        storage_range = self._input_storage_range(operand.values)
        if storage_range is not None:
            symbol = self.symbols.resolve_storage_range(
                *storage_range, metadata=metadata
            )
            if symbol is not None:
                return symbol
        call = self._collapsible_call_return(operand.values, operand.size_bits)
        if call is not None:
            return self.operation(call)
        slice_range = self._collapsible_slice_range(
            operand.values, operand.size_bits
        )
        if slice_range is not None:
            source_id, offset_bits = slice_range
            source = self.value(source_id)
            if offset_bits:
                source = self._binary(
                    ">>",
                    source,
                    PrintToken(str(offset_bits), "constant", metadata),
                    metadata,
                )
            if (
                offset_bits
                or self.values[source_id].size_bits != operand.size_bits
            ):
                source = self._cast(
                    self._integer_type(operand.size_bits, signed=False),
                    source,
                    metadata,
                )
            return source
        if len(operand.values) == 1:
            return self.value(operand.values[0])
        if operand.size_bits <= 64 and len(operand.values) * 8 == operand.size_bits:
            return self._compose_bytes(operand.values, operand.size_bits, metadata)
        return self._call(
            PrintToken("pcode_join_values", "fallback", metadata),
            tuple(self.value(value_id) for value_id in operand.values),
            metadata,
        )

    def _input_storage_range(
        self, value_ids: tuple[int, ...]
    ) -> tuple[str, int, int] | None:
        values = [self.values[value_id] for value_id in value_ids]
        if not values or any(value.kind != "input" or value.storage is None for value in values):
            return None
        first = values[0].storage
        assert first is not None
        if any(
            value.storage is None
            or value.storage.space != first.space
            or value.storage.scope_address != first.scope_address
            or value.storage.offset != first.offset + index
            for index, value in enumerate(values)
        ):
            return None
        return first.space, first.offset, len(values)

    def _is_unmodified_input(
        self, value_ids: tuple[int, ...], varnode: SleighVarnode
    ) -> bool:
        if varnode.space in {"const", "ram"}:
            return False
        if len(value_ids) != varnode.size:
            return False
        for index, value_id in enumerate(value_ids):
            value = self.values[value_id]
            if value.kind != "input" or value.storage is None:
                return False
            if (
                value.storage.space != varnode.space
                or value.storage.offset != varnode.offset + index
            ):
                return False
            if varnode.space == "unique" and value.storage.scope_address is not None:
                # A unique-space input is local to the current instruction.
                if value.instruction_address not in {None, value.storage.scope_address}:
                    return False
        return True

    def _collapsible_call_return(
        self, value_ids: tuple[int, ...], size_bits: int
    ) -> SSAOperation | None:
        if not value_ids or len(value_ids) * 8 != size_bits:
            return None
        values = [self.values[value_id] for value_id in value_ids]
        first = values[0]
        if (
            first.kind != "call_return"
            or first.instruction_address is None
            or first.operation_sequence is None
            or first.storage is None
        ):
            return None
        if any(
            value.kind != "call_return"
            or value.instruction_address != first.instruction_address
            or value.operation_sequence != first.operation_sequence
            or value.storage is None
            or value.storage.space != first.storage.space
            or value.storage.offset != first.storage.offset + index
            for index, value in enumerate(values)
        ):
            return None
        return self.call_operations_by_location.get(
            (first.instruction_address, first.operation_sequence)
        )

    def _collapsible_slice_range(
        self, value_ids: tuple[int, ...], size_bits: int
    ) -> tuple[int, int] | None:
        if not value_ids or len(value_ids) * 8 != size_bits:
            return None
        values = [self.values[value_id] for value_id in value_ids]
        source = values[0].source
        if (
            source is None
            or any(value.kind != "slice" or value.source != source for value in values)
        ):
            return None
        source_value = self.values.get(source)
        offset = values[0].offset_bits
        if (
            source_value is None
            or offset < 0
            or offset + size_bits > source_value.size_bits
            or any(
                value.offset_bits != offset + index * 8
                for index, value in enumerate(values)
            )
        ):
            return None
        return source, offset

    def _compose_bytes(
        self,
        value_ids: tuple[int, ...],
        size_bits: int,
        metadata: PcodeTokenMetadata,
    ) -> PrintToken | PcodeExpression:
        if size_bits > 64:
            return self._call(
                PrintToken("pcode_join_bytes", "fallback", metadata),
                tuple(self.value(value_id) for value_id in value_ids),
                metadata,
            )
        target_type = self._integer_type(size_bits, signed=False)
        wide_type = "uint64_t"
        result: PrintToken | PcodeExpression | None = None
        for index, value_id in enumerate(value_ids):
            byte = self._cast(wide_type, self.value(value_id), metadata)
            if index:
                byte = self._binary(
                    "<<",
                    byte,
                    PrintToken(str(index * 8), "constant", metadata),
                    metadata,
                )
            result = byte if result is None else self._binary("|", result, byte, metadata)
        assert result is not None
        return self._cast(target_type, result, metadata)

    def _slice_value(
        self, value: SSAValue, metadata: PcodeTokenMetadata
    ) -> PrintToken | PcodeExpression:
        assert value.source is not None
        source = self.value(value.source)
        if value.offset_bits:
            source = self._binary(
                ">>",
                source,
                PrintToken(str(value.offset_bits), "constant", metadata),
                metadata,
            )
        return self._cast(
            self._integer_type(value.size_bits, signed=False), source, metadata
        )

    def _phi_value(
        self, value: SSAValue, metadata: PcodeTokenMetadata
    ) -> PrintToken | PcodeExpression:
        phi = self.phis_by_output[value.identifier]
        incoming = [self.value(value_id) for _, value_id in phi.incoming]
        if not incoming:
            return self.symbols.resolve_value(value, metadata=metadata)
        return self._call(PrintToken("phi", "function", metadata), tuple(incoming), metadata)

    def _fallback(
        self,
        operation: SSAOperation,
        input_nodes: tuple[PrintToken | PcodeExpression, ...],
        metadata: PcodeTokenMetadata,
    ) -> PrintToken | PcodeExpression:
        if operation.opcode == "IMARK":
            return PrintToken("", "blank", metadata)
        name = self.symbols._identifier(f"pcode_{operation.opcode.lower()}", "pcode")
        return self._call(PrintToken(name, "fallback", metadata), input_nodes, metadata)

    @staticmethod
    def _binary(
        operator: str,
        left: PrintToken | PcodeExpression,
        right: PrintToken | PcodeExpression,
        metadata: PcodeTokenMetadata | None,
    ) -> PcodeExpression:
        return PcodeExpression(_BINARY_TOKENS[operator], (left, right), metadata)

    @staticmethod
    def _cast(
        type_name: str,
        value: PrintToken | PcodeExpression,
        metadata: PcodeTokenMetadata | None,
    ) -> PcodeExpression:
        return PcodeExpression(
            _CAST_TOKEN,
            (PrintToken(type_name, "type", metadata), value),
            metadata,
        )

    @staticmethod
    def _call(
        function: PrintToken | PcodeExpression,
        arguments: tuple[PrintToken | PcodeExpression, ...],
        metadata: PcodeTokenMetadata | None,
    ) -> PcodeExpression:
        if not arguments:
            argument_node: PrintToken | PcodeExpression = PrintToken("", "blank", metadata)
        else:
            argument_node = arguments[0]
            for argument in arguments[1:]:
                argument_node = _PcodeExpressionBuilder._binary(
                    ",", argument_node, argument, metadata
                )
        return PcodeExpression(_CALL_TOKEN, (function, argument_node), metadata)

    def _operation_metadata(
        self,
        operation: SSAOperation,
        varnode: SleighVarnode | None = None,
    ) -> PcodeTokenMetadata:
        return PcodeTokenMetadata(
            address=operation.instruction_address,
            sequence=operation.sequence,
            opcode=operation.opcode,
            space=varnode.space if varnode is not None else None,
            offset=varnode.offset if varnode is not None else None,
            size=varnode.size if varnode is not None else None,
        )

    @staticmethod
    def _value_metadata(value: SSAValue) -> PcodeTokenMetadata | None:
        if value.instruction_address is None and value.opcode is None:
            return None
        return PcodeTokenMetadata(
            address=value.instruction_address,
            sequence=value.operation_sequence,
            opcode=value.opcode,
            space=value.storage.space if value.storage is not None else None,
            offset=value.storage.offset if value.storage is not None else None,
            size=max(value.size_bits // 8, 1),
        )

    def _output_bits(self, operation: SSAOperation) -> int:
        if operation.output_value is not None:
            return max(self.values[operation.output_value].size_bits, 8)
        if operation.pcode is not None and operation.pcode.output is not None:
            return max(operation.pcode.output.size_bits, 8)
        return 32

    @staticmethod
    def _integer_type(bits: int, *, signed: bool) -> str:
        width = next((width for width in (8, 16, 32, 64) if bits <= width), bits)
        if width not in {8, 16, 32, 64}:
            return f"{'' if signed else 'u'}int{width}_t"
        return f"{'int' if signed else 'uint'}{width}_t"

    @staticmethod
    def _float_type(bits: int) -> str:
        if bits <= 32:
            return "float"
        if bits <= 64:
            return "double"
        return "long double"
