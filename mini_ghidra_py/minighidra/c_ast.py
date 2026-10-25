"""A compact C syntax tree and source emitter for MiniGhidraPy output.

The AST keeps declarations, control-flow statements, labels, and expressions
separate until the final emission pass.  ``RawExpression`` and
``RawStatement`` are deliberate escape hatches for architecture-specific
expressions that have already been normalized by the semantic lifter.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, TypeAlias

if TYPE_CHECKING:
    from .printlanguage import PcodeSymbolResolver, PrintLanguageRpn
    from .sleigh import SleighVarnode
    from .ssa import SSAFunction, SSAOperand, SSAOperation


class CTypeSpecifier:
    """Base class for structural C type specifiers."""


@dataclass(frozen=True, slots=True)
class CPrimitiveType(CTypeSpecifier):
    name: str


@dataclass(frozen=True, slots=True)
class CTypedefType(CTypeSpecifier):
    name: str


@dataclass(frozen=True, slots=True)
class CTagType(CTypeSpecifier):
    kind: str
    tag: str | None = None
    definition: CTypeDeclaration | None = None


@dataclass(frozen=True, slots=True)
class CAtomicType(CTypeSpecifier):
    base: CTypeName


@dataclass(frozen=True, slots=True)
class CRawTypeSpecifier(CTypeSpecifier):
    source: str


@dataclass(frozen=True, slots=True)
class CTypeName:
    base: CTypeSpecifier
    qualifiers: tuple[str, ...] = ()
    declarator: CDeclarator | None = None


class CExpression:
    """Base type for C expressions."""


@dataclass(frozen=True, slots=True)
class CIdentifier(CExpression):
    name: str


@dataclass(frozen=True, slots=True)
class CInteger(CExpression):
    value: int
    base: int = 10
    suffix: str = ""


@dataclass(frozen=True, slots=True)
class CFloat(CExpression):
    literal: str


@dataclass(frozen=True, slots=True)
class CString(CExpression):
    value: str
    prefix: str = ""
    literal: str | None = None


@dataclass(frozen=True, slots=True)
class CCharacter(CExpression):
    literal: str
    prefix: str = ""


@dataclass(frozen=True, slots=True)
class CRawExpression(CExpression):
    source: str


@dataclass(frozen=True, slots=True)
class CParenthesized(CExpression):
    value: CExpression


@dataclass(frozen=True, slots=True)
class CFieldDesignator:
    name: str


@dataclass(frozen=True, slots=True)
class CArrayDesignator:
    index: CExpression


CDesignator: TypeAlias = CFieldDesignator | CArrayDesignator


class CInitializerNode:
    """Base type for nested and designated initializer nodes."""


@dataclass(frozen=True, slots=True)
class CDesignatedInitializer(CInitializerNode):
    member: str
    value: CExpression | CInitializerList
    designators: tuple[CDesignator, ...] = ()


CInitializerEntry: TypeAlias = CExpression | CInitializerNode


@dataclass(frozen=True, slots=True)
class CInitializerList(CInitializerNode):
    entries: tuple[CInitializerEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class CCompoundLiteral(CExpression):
    type_name: str
    initializer: CInitializerList
    type_ast: CTypeName | None = None

    def __post_init__(self) -> None:
        if self.type_ast is None:
            object.__setattr__(self, "type_ast", parse_c_type_name(self.type_name))


@dataclass(frozen=True, slots=True)
class CUnary(CExpression):
    operator: str
    operand: CExpression
    postfix: bool = False


@dataclass(frozen=True, slots=True)
class CBinary(CExpression):
    operator: str
    left: CExpression
    right: CExpression


@dataclass(frozen=True, slots=True)
class CConditional(CExpression):
    condition: CExpression
    if_true: CExpression
    if_false: CExpression


@dataclass(frozen=True, slots=True)
class CCall(CExpression):
    function: CExpression
    arguments: tuple[CExpression, ...] = ()


@dataclass(frozen=True, slots=True)
class CCast(CExpression):
    type_name: str
    value: CExpression
    type_ast: CTypeName | None = None

    def __post_init__(self) -> None:
        if self.type_ast is None:
            object.__setattr__(self, "type_ast", parse_c_type_name(self.type_name))


@dataclass(frozen=True, slots=True)
class CSizeof(CExpression):
    operand: CExpression | CTypeName
    operator: str = "sizeof"


@dataclass(frozen=True, slots=True)
class CMember(CExpression):
    value: CExpression
    member: str
    pointer: bool = False


@dataclass(frozen=True, slots=True)
class CIndex(CExpression):
    value: CExpression
    index: CExpression


class CStatement:
    """Base type for C statements."""


@dataclass(frozen=True, slots=True)
class CBlock(CStatement):
    statements: tuple[CStatement, ...] = ()


@dataclass(frozen=True, slots=True)
class CComment(CStatement):
    text: str


@dataclass(frozen=True, slots=True)
class CRawStatement(CStatement):
    source: str


@dataclass(frozen=True, slots=True)
class CExpressionStatement(CStatement):
    expression: CExpression


@dataclass(frozen=True, slots=True)
class CAssignment(CStatement):
    target: CExpression
    value: CExpression
    operator: str = "="


@dataclass(frozen=True, slots=True)
class CDeclarator:
    name: str
    pointer_depth: int = 0
    array_dimensions: tuple[CExpression | None, ...] = ()
    initializer: CExpression | CInitializerList | None = None
    modifiers: tuple["CDeclaratorModifier", ...] = ()
    bit_width: CExpression | None = None


@dataclass(frozen=True, slots=True)
class CParameter:
    source: str
    type_name: str | None = None
    declarator: CDeclarator | None = None
    type_ast: CTypeName | None = None

    def __post_init__(self) -> None:
        if self.type_ast is None and self.type_name:
            object.__setattr__(self, "type_ast", parse_c_type_name(self.type_name))


CFunctionParameter: TypeAlias = str | CParameter


@dataclass(frozen=True, slots=True)
class CPointerModifier:
    qualifiers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CArrayModifier:
    size: CExpression | None = None
    qualifiers: tuple[str, ...] = ()
    is_static: bool = False


@dataclass(frozen=True, slots=True)
class CFunctionModifier:
    parameters: tuple[CFunctionParameter, ...] | None = None
    variadic: bool = False


CDeclaratorModifier: TypeAlias = CPointerModifier | CArrayModifier | CFunctionModifier


@dataclass(frozen=True, slots=True)
class CDeclaration(CStatement):
    source: str
    type_name: str | None = None
    declarators: tuple[CDeclarator, ...] = ()
    storage_specifiers: tuple[str, ...] = ()
    calling_convention: str = ""
    type_ast: CTypeName | None = None

    def __post_init__(self) -> None:
        if self.type_ast is None and self.type_name:
            object.__setattr__(self, "type_ast", parse_c_type_name(self.type_name))


@dataclass(frozen=True, slots=True)
class CGlobalDeclaration:
    declaration: CDeclaration


@dataclass(frozen=True, slots=True)
class CFunctionPrototype:
    return_type: str
    declarator: CDeclarator
    storage_specifiers: tuple[str, ...] = ()
    calling_convention: str = ""
    return_type_ast: CTypeName | None = None

    def __post_init__(self) -> None:
        if self.return_type_ast is None:
            object.__setattr__(
                self, "return_type_ast", parse_c_type_name(self.return_type)
            )


@dataclass(frozen=True, slots=True)
class CEnumerator:
    name: str
    value: CExpression | None = None


@dataclass(frozen=True, slots=True)
class CTypeDeclaration(CStatement):
    kind: str
    tag: str | None = None
    members: tuple[CDeclaration, ...] = ()
    enumerators: tuple[CEnumerator, ...] = ()
    declarators: tuple[CDeclarator, ...] = ()
    storage_specifiers: tuple[str, ...] = ()
    definition: bool = True


@dataclass(frozen=True, slots=True)
class CTypedefDeclaration:
    declaration: CDeclaration | CTypeDeclaration


@dataclass(frozen=True, slots=True)
class CLabel(CStatement):
    name: str


@dataclass(frozen=True, slots=True)
class CGoto(CStatement):
    target: str
    indirect: bool = False


@dataclass(frozen=True, slots=True)
class CReturn(CStatement):
    value: CExpression | None = None


@dataclass(frozen=True, slots=True)
class CBreak(CStatement):
    pass


@dataclass(frozen=True, slots=True)
class CContinue(CStatement):
    pass


@dataclass(frozen=True, slots=True)
class CIf(CStatement):
    condition: CExpression
    body: CStatement
    otherwise: CStatement | None = None
    compact: bool = False


@dataclass(frozen=True, slots=True)
class CWhile(CStatement):
    condition: CExpression
    body: CBlock


@dataclass(frozen=True, slots=True)
class CFor(CStatement):
    header: str
    body: CBlock
    initializer: CStatement | CExpression | None = None
    condition: CExpression | None = None
    iteration: CExpression | None = None
    structured: bool = False


@dataclass(frozen=True, slots=True)
class CDoWhile(CStatement):
    body: CBlock
    condition: CExpression


@dataclass(frozen=True, slots=True)
class CCase(CStatement):
    value: CExpression | None
    statements: tuple[CStatement, ...] = ()


@dataclass(frozen=True, slots=True)
class CSwitch(CStatement):
    expression: CExpression
    cases: tuple[CCase, ...]


@dataclass(frozen=True, slots=True)
class CFunction:
    return_type: str
    name: str
    parameters: tuple[CFunctionParameter, ...]
    body: CBlock
    calling_convention: str = ""
    local_declarations: tuple[CDeclaration, ...] = ()
    leading_comments: tuple[str, ...] = ()
    storage_specifiers: tuple[str, ...] = ()
    return_type_ast: CTypeName | None = None
    declarator: CDeclarator | None = None

    def __post_init__(self) -> None:
        if self.return_type_ast is None:
            object.__setattr__(
                self, "return_type_ast", parse_c_type_name(self.return_type)
            )


@dataclass(frozen=True, slots=True)
class CRawExternalDeclaration:
    source: str


CExternalDeclaration: TypeAlias = (
    CFunction
    | CGlobalDeclaration
    | CFunctionPrototype
    | CTypedefDeclaration
    | CDeclaration
    | CTypeDeclaration
    | CComment
    | CRawExternalDeclaration
)


@dataclass(frozen=True, slots=True)
class CTranslationUnit:
    functions: tuple[CFunction, ...] = ()
    leading_comments: tuple[str, ...] = ()
    items: tuple[CExternalDeclaration, ...] = ()


_CStatementOrBlock: TypeAlias = CStatement | CBlock


class CEmitter:
    """Emit readable C text from C AST nodes."""

    def __init__(
        self, *, indent: str = "  ", expression_mode: str = "direct"
    ) -> None:
        if expression_mode not in {"direct", "rpn"}:
            raise ValueError("expression_mode must be 'direct' or 'rpn'")
        self.indent = indent
        self.expression_mode = expression_mode
        self._direct_emitter = (
            self
            if expression_mode == "direct"
            else CEmitter(indent=indent, expression_mode="direct")
        )
        self._rpn_expression_printer = None

    def emit(self, unit: CTranslationUnit) -> str:
        sections: list[str] = []
        sections.extend(unit.leading_comments)
        if unit.items:
            sections.extend(self.emit_external(item) for item in unit.items)
        else:
            sections.extend(self.emit_function(function) for function in unit.functions)
        return "\n\n".join(section for section in sections if section)

    def emit_external(self, declaration: CExternalDeclaration) -> str:
        if isinstance(declaration, CFunction):
            return self.emit_function(declaration)
        if isinstance(declaration, CGlobalDeclaration):
            return "\n".join(self.emit_statement(declaration.declaration))
        if isinstance(declaration, CFunctionPrototype):
            prototype = CDeclaration(
                "",
                declaration.return_type,
                (declaration.declarator,),
                declaration.storage_specifiers,
                declaration.calling_convention,
                declaration.return_type_ast,
            )
            return "\n".join(self.emit_statement(prototype))
        if isinstance(declaration, CTypedefDeclaration):
            return self.emit_external(declaration.declaration)
        if isinstance(declaration, CTypeDeclaration):
            return "\n".join(self._emit_type_declaration(declaration))
        if isinstance(declaration, CDeclaration):
            return "\n".join(self.emit_statement(declaration))
        if isinstance(declaration, CComment):
            return declaration.text
        if isinstance(declaration, CRawExternalDeclaration):
            return declaration.source.strip()
        raise TypeError(f"Declaración externa no soportada: {type(declaration).__name__}")

    def emit_function(self, function: CFunction) -> str:
        lines = list(function.leading_comments)
        storage = " ".join(function.storage_specifiers)
        storage_prefix = f"{storage} " if storage else ""
        convention = f"{function.calling_convention} " if function.calling_convention else ""
        return_type = self._emit_type_name(
            function.return_type, function.return_type_ast
        )
        if function.declarator is not None:
            declarator = self._emit_declarator(function.declarator)
        else:
            parameters = (
                ", ".join(
                    self._emit_parameter(parameter) for parameter in function.parameters
                )
                if function.parameters
                else "void"
            )
            declarator = f"{function.name}({parameters})"
        lines.append(
            f"{storage_prefix}{return_type} {convention}{declarator} {{"
        )
        body = [*function.local_declarations, *function.body.statements]
        for statement in body:
            lines.extend(self.emit_statement(statement, 1))
        lines.append("}")
        return "\n".join(lines)

    def emit_statement(self, statement: CStatement, depth: int = 0) -> list[str]:
        prefix = self.indent * depth
        if isinstance(statement, CBlock):
            lines = [prefix + "{"]
            for child in statement.statements:
                lines.extend(self.emit_statement(child, depth + 1))
            lines.append(prefix + "}")
            return lines
        if isinstance(statement, CComment):
            return [prefix + statement.text]
        if isinstance(statement, CRawStatement):
            return [prefix + statement.source.lstrip()]
        if isinstance(statement, CTypeDeclaration):
            return self._emit_type_declaration(statement, depth)
        if isinstance(statement, CDeclaration):
            return [prefix + self._emit_declaration(statement)]
        if isinstance(statement, CExpressionStatement):
            return [prefix + self.emit_expression(statement.expression) + ";"]
        if isinstance(statement, CAssignment):
            return [
                prefix
                + self.emit_expression(statement.target)
                + f" {statement.operator} "
                + self.emit_expression(statement.value)
                + ";"
            ]
        if isinstance(statement, CLabel):
            return [prefix + statement.name + ":"]
        if isinstance(statement, CGoto):
            target = self.emit_expression(CRawExpression(statement.target))
            return [prefix + ("goto *" if statement.indirect else "goto ") + target + ";"]
        if isinstance(statement, CReturn):
            if statement.value is None:
                return [prefix + "return;"]
            return [prefix + "return " + self.emit_expression(statement.value) + ";"]
        if isinstance(statement, CBreak):
            return [prefix + "break;"]
        if isinstance(statement, CContinue):
            return [prefix + "continue;"]
        if isinstance(statement, CIf):
            condition = self.emit_expression(statement.condition)
            if statement.compact and statement.otherwise is None:
                body = self.emit_statement(statement.body, 0)[0].strip()
                return [prefix + f"if ({condition}) {body}"]
            lines = [prefix + f"if ({condition}) {{"]
            lines.extend(self._emit_as_block(statement.body, depth + 1))
            if statement.otherwise is not None:
                lines.append(prefix + "} else {")
                lines.extend(self._emit_as_block(statement.otherwise, depth + 1))
            lines.append(prefix + "}")
            return lines
        if isinstance(statement, CWhile):
            lines = [prefix + f"while ({self.emit_expression(statement.condition)}) {{"]
            for child in statement.body.statements:
                lines.extend(self.emit_statement(child, depth + 1))
            lines.append(prefix + "}")
            return lines
        if isinstance(statement, CFor):
            if statement.structured:
                if isinstance(statement.initializer, CDeclaration):
                    initializer = self._emit_declaration(
                        statement.initializer, include_semicolon=False
                    )
                elif isinstance(statement.initializer, CExpression):
                    initializer = self.emit_expression(statement.initializer)
                else:
                    initializer = ""
                condition = (
                    self.emit_expression(statement.condition)
                    if statement.condition is not None
                    else ""
                )
                iteration = (
                    self.emit_expression(statement.iteration)
                    if statement.iteration is not None
                    else ""
                )
                header = f"{initializer}; {condition}; {iteration}"
            else:
                header = statement.header
            lines = [prefix + f"for ({header}) {{"]
            for child in statement.body.statements:
                lines.extend(self.emit_statement(child, depth + 1))
            lines.append(prefix + "}")
            return lines
        if isinstance(statement, CDoWhile):
            lines = [prefix + "do {"]
            for child in statement.body.statements:
                lines.extend(self.emit_statement(child, depth + 1))
            lines.append(
                prefix + f"}} while ({self.emit_expression(statement.condition)});"
            )
            return lines
        if isinstance(statement, CSwitch):
            lines = [prefix + f"switch ({self.emit_expression(statement.expression)}) {{"]
            for case in statement.cases:
                label = (
                    "default:"
                    if case.value is None
                    else f"case {self.emit_expression(case.value)}:"
                )
                lines.append(self.indent * (depth + 1) + label)
                for child in case.statements:
                    lines.extend(self.emit_statement(child, depth + 2))
            lines.append(prefix + "}")
            return lines
        if isinstance(statement, CCase):
            label = (
                "default:"
                if statement.value is None
                else f"case {self.emit_expression(statement.value)}:"
            )
            return [prefix + label]
        raise TypeError(f"Nodo C no soportado: {type(statement).__name__}")

    def emit_expression(self, expression: CExpression) -> str:
        if self.expression_mode == "rpn":
            return self._rpn_printer().emit_expression(expression)
        return self._emit_expression_direct(expression)

    def emit_pcode_value(
        self,
        function: SSAFunction,
        value_id: int,
        *,
        symbols: PcodeSymbolResolver | None = None,
    ) -> str:
        """Emit an SSA value by following its linked SLEIGH p-code definitions."""
        return self._rpn_printer().emit_ssa_value(
            function, value_id, symbols=symbols
        )

    def emit_pcode_operand(
        self,
        function: SSAFunction,
        operand: SSAOperand,
        *,
        varnode: SleighVarnode | None = None,
        symbols: PcodeSymbolResolver | None = None,
    ) -> str:
        """Emit a varnode operand, reassembling its byte-level SSA slices."""
        return self._rpn_printer().emit_ssa_operand(
            function, operand, varnode=varnode, symbols=symbols
        )

    def emit_pcode_operation(
        self,
        function: SSAFunction,
        operation: int | SSAOperation,
        *,
        symbols: PcodeSymbolResolver | None = None,
        resolve_definitions: bool = True,
    ) -> str:
        """Dispatch a p-code operation through the RPN token emitter."""
        return self._rpn_printer().emit_ssa_operation(
            function,
            operation,
            symbols=symbols,
            resolve_definitions=resolve_definitions,
        )

    def emit_token_markup(self) -> str:
        """Render the most recently emitted RPN token stream as semantic HTML."""
        return self._rpn_printer().emit_html()

    def _rpn_printer(self) -> PrintLanguageRpn:
        if self.expression_mode != "rpn":
            raise ValueError("p-code token emission requires expression_mode='rpn'")
        if self._rpn_expression_printer is None:
            from .printlanguage import PrintLanguageRpn

            self._rpn_expression_printer = PrintLanguageRpn(self)
        return self._rpn_expression_printer

    def _emit_expression_direct(self, expression: CExpression) -> str:
        if isinstance(expression, CIdentifier):
            return expression.name
        if isinstance(expression, CInteger):
            if expression.base == 16:
                value = f"{expression.value:#x}"
            elif expression.base == 8:
                value = "0" if expression.value == 0 else f"0{expression.value:o}"
            else:
                value = str(expression.value)
            return value + expression.suffix
        if isinstance(expression, CFloat):
            return expression.literal
        if isinstance(expression, CString):
            if expression.literal is not None:
                return expression.literal
            literal = json.dumps(expression.value, ensure_ascii=True)
            return expression.prefix + literal
        if isinstance(expression, CCharacter):
            return expression.prefix + expression.literal
        if isinstance(expression, CRawExpression):
            return expression.source
        if isinstance(expression, CParenthesized):
            return f"({self.emit_expression(expression.value)})"
        if isinstance(expression, CCompoundLiteral):
            initializer = (
                self._emit_initializer(expression.initializer)
                if expression.initializer.entries
                else "{ 0 }"
            )
            type_name = self._emit_type_name(expression.type_name, expression.type_ast)
            return f"(({type_name}){initializer})"
        if isinstance(expression, CUnary):
            value = self.emit_expression(expression.operand)
            return (
                f"{value}{expression.operator}"
                if expression.postfix
                else f"{expression.operator}{value}"
            )
        if isinstance(expression, CBinary):
            return (
                f"({self.emit_expression(expression.left)} {expression.operator} "
                f"{self.emit_expression(expression.right)})"
            )
        if isinstance(expression, CConditional):
            return (
                f"({self.emit_expression(expression.condition)} ? "
                f"{self.emit_expression(expression.if_true)} : "
                f"{self.emit_expression(expression.if_false)})"
            )
        if isinstance(expression, CCall):
            arguments = ", ".join(self.emit_expression(arg) for arg in expression.arguments)
            function = self.emit_expression(expression.function)
            if isinstance(
                expression.function,
                (CUnary, CBinary, CConditional, CCast),
            ):
                function = f"({function})"
            return f"{function}({arguments})"
        if isinstance(expression, CCast):
            type_name = self._emit_type_name(expression.type_name, expression.type_ast)
            return f"({type_name})({self.emit_expression(expression.value)})"
        if isinstance(expression, CSizeof):
            if isinstance(expression.operand, CTypeName):
                operand = self._emit_type_specifier(expression.operand)
            else:
                operand = self.emit_expression(expression.operand)
            return f"{expression.operator}({operand})"
        if isinstance(expression, CMember):
            operator = "->" if expression.pointer else "."
            value = self.emit_expression(expression.value)
            if isinstance(expression.value, (CUnary, CBinary, CConditional, CCast)):
                value = f"({value})"
            return f"{value}{operator}{expression.member}"
        if isinstance(expression, CIndex):
            value = self.emit_expression(expression.value)
            if isinstance(expression.value, (CUnary, CBinary, CConditional, CCast)):
                value = f"({value})"
            return f"{value}[{self.emit_expression(expression.index)}]"
        raise TypeError(f"Expresión C no soportada: {type(expression).__name__}")

    def _emit_parameter(self, parameter: CFunctionParameter) -> str:
        if isinstance(parameter, str):
            return parameter
        if parameter.type_name and parameter.declarator:
            type_name = self._emit_type_name(parameter.type_name, parameter.type_ast)
            return f"{type_name} {self._emit_declarator(parameter.declarator)}"
        return parameter.source

    def _emit_declaration(
        self, declaration: CDeclaration, *, include_semicolon: bool = True
    ) -> str:
        if declaration.type_name and declaration.declarators:
            type_name = self._emit_type_name(
                declaration.type_name, declaration.type_ast
            )
            prefix = " ".join((*declaration.storage_specifiers, type_name))
            convention = (
                f" {declaration.calling_convention}"
                if declaration.calling_convention
                else ""
            )
            declarators = ", ".join(
                self._emit_declarator(declarator)
                for declarator in declaration.declarators
            )
            source = f"{prefix}{convention} {declarators}"
        else:
            source = declaration.source.strip().removesuffix(";")
        return source + ";" if include_semicolon else source

    def _emit_type_declaration(
        self, declaration: CTypeDeclaration, depth: int = 0
    ) -> list[str]:
        prefix = self.indent * depth
        storage = " ".join(declaration.storage_specifiers)
        storage_prefix = f"{storage} " if storage else ""
        tag = f" {declaration.tag}" if declaration.tag else ""
        head = f"{storage_prefix}{declaration.kind}{tag}"
        if not declaration.definition:
            return [prefix + head + ";"]
        if declaration.kind == "enum":
            entries = []
            for enumerator in declaration.enumerators:
                value = (
                    f" = {self.emit_expression(enumerator.value)}"
                    if enumerator.value is not None
                    else ""
                )
                entries.append(self.indent * (depth + 1) + enumerator.name + value)
            lines = [prefix + head + " {"]
            if entries:
                lines.extend(
                    line + ("," if index + 1 < len(entries) else "")
                    for index, line in enumerate(entries)
                )
            lines.append(prefix + "}" + self._emit_type_declarator_suffix(declaration))
            return lines
        lines = [prefix + head + " {"]
        for member in declaration.members:
            lines.extend(self.emit_statement(member, depth + 1))
        lines.append(prefix + "}" + self._emit_type_declarator_suffix(declaration))
        return lines

    def _emit_type_declarator_suffix(self, declaration: CTypeDeclaration) -> str:
        suffix = ""
        if declaration.declarators:
            suffix = " " + ", ".join(
                self._emit_declarator(declarator)
                for declarator in declaration.declarators
            )
        return suffix + ";"

    def _emit_declarator(self, declarator: CDeclarator) -> str:
        if declarator.modifiers:
            value = declarator.name
            precedence = "atom"
            for modifier in declarator.modifiers:
                if isinstance(modifier, CPointerModifier):
                    qualifiers = " ".join(modifier.qualifiers)
                    value = "*" + (f" {qualifiers} " if qualifiers else "") + value
                    precedence = "pointer"
                elif isinstance(modifier, CArrayModifier):
                    if precedence == "pointer":
                        value = f"({value})"
                    bound = "" if modifier.size is None else self.emit_expression(modifier.size)
                    qualifiers = " ".join(modifier.qualifiers)
                    static = "static " if modifier.is_static else ""
                    suffix = f"[{static}{qualifiers + ' ' if qualifiers else ''}{bound}]"
                    value += suffix
                    precedence = "postfix"
                elif isinstance(modifier, CFunctionModifier):
                    if precedence == "pointer":
                        value = f"({value})"
                    if modifier.parameters is None:
                        parameters = ""
                    else:
                        values = [
                            self._emit_parameter(parameter)
                            for parameter in modifier.parameters
                        ]
                        if modifier.variadic:
                            values.append("...")
                        parameters = ", ".join(values)
                    value += f"({parameters})"
                    precedence = "postfix"
        else:
            value = "*" * declarator.pointer_depth + declarator.name
            for dimension in declarator.array_dimensions:
                bound = "" if dimension is None else self.emit_expression(dimension)
                value += f"[{bound}]"
        if declarator.bit_width is not None:
            separator = ": " if not value else " : "
            value += separator + self.emit_expression(declarator.bit_width)
        if declarator.initializer is not None:
            value += " = " + self._emit_initializer(declarator.initializer)
        return value

    def _emit_initializer(self, initializer: CExpression | CInitializerList) -> str:
        if isinstance(initializer, CExpression):
            return self.emit_expression(initializer)
        entries = []
        for entry in initializer.entries:
            if isinstance(entry, CDesignatedInitializer):
                if entry.designators:
                    designator = "".join(
                        f".{item.name}"
                        if isinstance(item, CFieldDesignator)
                        else f"[{self.emit_expression(item.index)}]"
                        for item in entry.designators
                    )
                else:
                    designator = f".{entry.member}"
                value = self._emit_initializer(entry.value)
                entries.append(f"{designator} = {value}")
            elif isinstance(entry, CInitializerList):
                entries.append(self._emit_initializer(entry))
            else:
                entries.append(self.emit_expression(entry))
        return "{ " + ", ".join(entries) + " }"

    def _emit_type_name(
        self, source: str, type_ast: CTypeName | None = None
    ) -> str:
        parsed = type_ast or parse_c_type_name(source)
        return self._emit_type_specifier(parsed)

    def _emit_type_specifier(self, type_name: CTypeName) -> str:
        base = type_name.base
        if isinstance(base, CPrimitiveType):
            source = base.name
        elif isinstance(base, CTypedefType):
            source = base.name
        elif isinstance(base, CTagType):
            tag = f" {base.tag}" if base.tag else ""
            source = f"{base.kind}{tag}"
            if base.definition is not None:
                definition = base.definition
                if definition.kind == "enum":
                    values = []
                    for item in definition.enumerators:
                        value = (
                            f" = {self.emit_expression(item.value)}"
                            if item.value is not None
                            else ""
                        )
                        values.append(item.name + value)
                    source += " { " + ", ".join(values) + " }"
                else:
                    members = [
                        self._emit_declaration(member)
                        for member in definition.members
                    ]
                    source += " { " + " ".join(members) + " }"
        elif isinstance(base, CAtomicType):
            source = f"_Atomic({self._emit_type_specifier(base.base)})"
        elif isinstance(base, CRawTypeSpecifier):
            source = base.source
        else:
            raise TypeError(f"Especificador de tipo no soportado: {type(base).__name__}")
        qualifiers = " ".join(type_name.qualifiers)
        source = f"{qualifiers} {source}" if qualifiers else source
        if type_name.declarator is not None:
            abstract = self._emit_declarator(type_name.declarator)
            if abstract:
                source += " " + abstract
        return source

    def _emit_as_block(self, statement: CStatement, depth: int) -> list[str]:
        if isinstance(statement, CBlock):
            result: list[str] = []
            for child in statement.statements:
                result.extend(self.emit_statement(child, depth))
            return result
        return self.emit_statement(statement, depth)


def parse_c_statements(lines: Iterable[str]) -> CBlock:
    """Parse the decompiler's small statement subset into structured C AST.

    Expressions not constructed by this module remain ``CRawExpression``
    leaves, preserving the semantic lifter's exact casts and architecture
    intrinsics while moving all statement/bracing emission into ``CEmitter``.
    """
    materialized = [line.rstrip() for line in lines]
    statements, _ = _parse_sequence(materialized, 0, stop_at_brace=False)
    return CBlock(tuple(statements))


def parse_c_type_name(source: str) -> CTypeName:
    """Parse a C type specifier into primitive, typedef, tag, or raw nodes."""
    text = source.strip()
    if not text:
        return CTypeName(CRawTypeSpecifier(text))

    split = _split_declaration_specifiers(text)
    if split is not None:
        storage, base_source, declarator_source, calling_convention = split
        if not storage and not calling_convention and declarator_source:
            base_type = parse_c_type_name(base_source)
            parsed_declarator = _CDeclaratorParser(
                declarator_source, abstract=True
            ).parse()
            if parsed_declarator is not None and not parsed_declarator[0]:
                name, modifiers = parsed_declarator
                declarator = CDeclarator(
                    name=name,
                    pointer_depth=sum(
                        isinstance(modifier, CPointerModifier)
                        for modifier in modifiers
                    ),
                    array_dimensions=tuple(
                        modifier.size
                        for modifier in modifiers
                        if isinstance(modifier, CArrayModifier)
                    ),
                    modifiers=modifiers,
                )
                return CTypeName(base_type.base, base_type.qualifiers, declarator)

    qualifiers: list[str] = []
    cursor = 0
    while True:
        match = re.match(r"\s*([A-Za-z_]\w*)", text[cursor:])
        if match is None or match.group(1) not in _C_TYPE_QUALIFIERS:
            break
        if (
            match.group(1) == "_Atomic"
            and text[cursor + match.end() :].lstrip().startswith("(")
        ):
            break
        qualifiers.append(match.group(1))
        cursor += match.end()
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1

    atomic = re.match(r"_Atomic\s*\(", text[cursor:])
    if atomic is not None:
        open_paren = cursor + atomic.end() - 1
        close = _matching_c_delimiter(text, open_paren, "(", ")")
        if close is not None and not text[close + 1 :].strip():
            inner = parse_c_type_name(text[open_paren + 1 : close])
            return CTypeName(CAtomicType(inner), tuple(qualifiers))

    tag_match = re.match(r"(struct|union|enum)\b", text[cursor:])
    if tag_match is not None:
        kind = tag_match.group(1)
        tag_cursor = cursor + tag_match.end()
        while tag_cursor < len(text) and text[tag_cursor].isspace():
            tag_cursor += 1
        tag_match = re.match(r"[A-Za-z_]\w*", text[tag_cursor:])
        tag = tag_match.group() if tag_match else None
        if tag_match is not None:
            tag_cursor += len(tag)
            while tag_cursor < len(text) and text[tag_cursor].isspace():
                tag_cursor += 1
        definition = None
        if tag_cursor < len(text) and text[tag_cursor] == "{":
            close = _matching_c_delimiter(text, tag_cursor, "{", "}")
            if close is None:
                return CTypeName(CRawTypeSpecifier(text))
            tail = text[close + 1 :].strip()
            if tail:
                tail_words = tail.split()
                if any(word not in _C_TYPE_QUALIFIERS for word in tail_words):
                    return CTypeName(CRawTypeSpecifier(text))
                qualifiers.extend(tail_words)
            tag_source = text[cursor : close + 1]
            definition = _parse_type_declaration(tag_source + ";")
            if definition is None:
                return CTypeName(CRawTypeSpecifier(text))
            return CTypeName(CTagType(kind, tag, definition), tuple(qualifiers))
        tail = text[tag_cursor:].strip()
        if tag is None or any(
            word not in _C_TYPE_QUALIFIERS for word in tail.split()
        ):
            return CTypeName(CRawTypeSpecifier(text))
        qualifiers.extend(tail.split())
        return CTypeName(CTagType(kind, tag), tuple(qualifiers))

    if re.sub(r"[A-Za-z_]\w*|\s+", "", text):
        return CTypeName(CRawTypeSpecifier(text))
    words = re.findall(r"[A-Za-z_]\w*", text)
    type_words = [word for word in words if word not in _C_TYPE_QUALIFIERS]
    qualifiers.extend(word for word in words if word in _C_TYPE_QUALIFIERS)
    if not type_words:
        return CTypeName(CRawTypeSpecifier(text))
    if all(
        word in _C_TYPE_KEYWORDS
        or word.endswith("_t")
        or re.fullmatch(r"(?:u?int|__u?int|undefined)\d+_?t?", word)
        or word in {"__int8", "__int16", "__int32", "__int64"}
        for word in type_words
    ):
        base: CTypeSpecifier = CPrimitiveType(" ".join(type_words))
    elif len(type_words) == 1:
        base = CTypedefType(type_words[0])
    else:
        base = CRawTypeSpecifier(text)
    return CTypeName(base, tuple(dict.fromkeys(qualifiers)))


def parse_c_declaration(source: str) -> CDeclaration:
    """Parse a declaration, retaining the original spelling on unsupported C."""
    text = source.strip()
    if not text.endswith(";"):
        text += ";"
    parts = _parse_c_declaration_parts(text[:-1].strip())
    if parts is None:
        return CDeclaration(text)
    storage, type_name, declarators, calling_convention = parts
    return CDeclaration(text, type_name, declarators, storage, calling_convention)


def parse_c_parameter(source: str) -> CParameter:
    """Parse a parameter with recursive pointer/array/function declarators."""
    text = source.strip()
    parts = _parse_c_declaration_parts(text)
    if parts is None:
        specifiers = _split_declaration_specifiers(text)
        if specifiers is not None:
            storage, type_name, remainder, _calling_convention = specifiers
            if not storage and not remainder:
                return CParameter(text, type_name)
        return CParameter(text)
    storage, type_name, declarators, _calling_convention = parts
    if storage or len(declarators) != 1 or declarators[0].initializer is not None:
        return CParameter(text)
    return CParameter(text, type_name, declarators[0])


def parse_c_translation_unit(source: str) -> CTranslationUnit:
    """Parse common file-scope C declarations and definitions conservatively.

    Globals, typedefs, tagged types, prototypes, and ordinary function
    definitions become typed nodes. Extensions or declarations outside the
    supported grammar are retained as raw external declarations.
    """
    items: list[CExternalDeclaration] = []
    for chunk in _split_translation_unit(source):
        stripped = chunk.strip()
        if not stripped:
            continue
        if stripped.startswith(("//", "/*")):
            items.append(CComment(stripped))
            continue
        if stripped.startswith("#"):
            items.append(CRawExternalDeclaration(stripped))
            continue
        function = _parse_function_definition(stripped)
        if function is not None:
            items.append(function)
            continue
        type_declaration = _parse_type_declaration(stripped)
        if type_declaration is not None:
            if "typedef" in type_declaration.storage_specifiers:
                items.append(CTypedefDeclaration(type_declaration))
            else:
                items.append(type_declaration)
            continue
        declaration = parse_c_declaration(stripped)
        if declaration.type_name is None or not declaration.declarators:
            items.append(CRawExternalDeclaration(stripped))
        elif "typedef" in declaration.storage_specifiers:
            items.append(CTypedefDeclaration(declaration))
        elif (
            len(declaration.declarators) == 1
            and declaration.declarators[0].modifiers
            and isinstance(declaration.declarators[0].modifiers[0], CFunctionModifier)
        ):
            items.append(
                CFunctionPrototype(
                    declaration.type_name,
                    declaration.declarators[0],
                    declaration.storage_specifiers,
                    declaration.calling_convention,
                )
            )
        else:
            items.append(CGlobalDeclaration(declaration))
    functions = tuple(item for item in items if isinstance(item, CFunction))
    return CTranslationUnit(functions=functions, items=tuple(items))


def _split_translation_unit(source: str) -> list[str]:
    chunks: list[str] = []
    position = 0
    length = len(source)
    while position < length:
        while position < length and source[position].isspace():
            position += 1
        if position >= length:
            break
        if source.startswith("//", position):
            end = source.find("\n", position)
            if end < 0:
                end = length
            chunks.append(source[position:end])
            position = end
            continue
        if source.startswith("/*", position):
            end = source.find("*/", position + 2)
            end = length if end < 0 else end + 2
            chunks.append(source[position:end])
            position = end
            continue
        if source[position] == "#":
            end = source.find("\n", position)
            if end < 0:
                end = length
            chunks.append(source[position:end])
            position = end
            continue

        start = position
        index = position
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        quote: str | None = None
        escaped = False
        while index < length:
            if quote is not None:
                character = source[index]
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                index += 1
                continue
            if source.startswith("//", index):
                newline = source.find("\n", index)
                index = length if newline < 0 else newline
                continue
            if source.startswith("/*", index):
                end = source.find("*/", index + 2)
                index = length if end < 0 else end + 2
                continue
            character = source[index]
            if character in {"'", '"'}:
                quote = character
            elif character == "(":
                paren_depth += 1
            elif character == ")" and paren_depth:
                paren_depth -= 1
            elif character == "[":
                bracket_depth += 1
            elif character == "]" and bracket_depth:
                bracket_depth -= 1
            elif character == "{" and paren_depth == 0 and bracket_depth == 0:
                prefix = source[start:index].strip()
                if brace_depth == 0 and _looks_like_function_definition(prefix):
                    close = _matching_c_delimiter(source, index, "{", "}")
                    if close is None:
                        chunks.append(source[start:].strip())
                        return chunks
                    chunks.append(source[start : close + 1].strip())
                    position = close + 1
                    break
                brace_depth += 1
            elif character == "}" and brace_depth:
                brace_depth -= 1
            elif character == ";" and not paren_depth and not bracket_depth and not brace_depth:
                chunks.append(source[start : index + 1].strip())
                position = index + 1
                break
            index += 1
        else:
            if source[start:].strip():
                chunks.append(source[start:].strip())
            break
        if position == start:
            # A closing brace without a corresponding open brace or malformed
            # declaration: consume to the next semicolon as a raw unit.
            end = source.find(";", start)
            if end < 0:
                chunks.append(source[start:].strip())
                break
            chunks.append(source[start : end + 1].strip())
            position = end + 1
    return chunks


def _looks_like_function_definition(prefix: str) -> bool:
    if not prefix or _find_assignment(prefix) is not None:
        return False
    parts = _parse_c_declaration_parts(prefix)
    if parts is None:
        return False
    _storage, _type_name, declarators, _calling_convention = parts
    if len(declarators) != 1 or not declarators[0].name:
        return False
    modifiers = declarators[0].modifiers
    return bool(modifiers and isinstance(modifiers[0], CFunctionModifier))


def _parse_function_definition(source: str) -> CFunction | None:
    open_brace = _find_top_level_character(source, "{")
    if open_brace is None or not source.rstrip().endswith("}"):
        return None
    header = source[:open_brace].strip()
    parts = _parse_c_declaration_parts(header)
    if parts is None:
        return None
    storage, return_type, declarators, calling_convention = parts
    if len(declarators) != 1:
        return None
    declarator = declarators[0]
    if (
        not declarator.name
        or not declarator.modifiers
        or not isinstance(declarator.modifiers[0], CFunctionModifier)
    ):
        return None
    modifier = declarator.modifiers[0]
    parameters = modifier.parameters or ()
    if (
        len(parameters) == 1
        and isinstance(parameters[0], CParameter)
        and parameters[0].source == "void"
    ):
        parameters = ()
    body_source = source[open_brace + 1 : source.rfind("}")]
    body_lines = body_source.splitlines()
    if len(body_lines) <= 1:
        body_lines = _split_simple_c_body(body_source)
    return CFunction(
        return_type=return_type,
        name=declarator.name,
        parameters=tuple(parameters),
        body=parse_c_statements(body_lines),
        calling_convention=calling_convention,
        storage_specifiers=storage,
        declarator=declarator,
    )


def _split_simple_c_body(source: str) -> list[str]:
    lines: list[str] = []
    start = 0
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(source):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "(":
            paren_depth += 1
        elif character == ")" and paren_depth:
            paren_depth -= 1
        elif character == "[":
            bracket_depth += 1
        elif character == "]" and bracket_depth:
            bracket_depth -= 1
        elif character == "{" and not paren_depth and not bracket_depth:
            brace_depth += 1
            segment = source[start:index].strip()
            if segment:
                lines.append(segment + " {")
            else:
                lines.append("{")
            start = index + 1
        elif character == "}" and brace_depth:
            segment = source[start:index].strip()
            if segment:
                lines.append(segment)
            lines.append("}")
            brace_depth -= 1
            start = index + 1
        elif character == ";" and not paren_depth and not bracket_depth and not brace_depth:
            lines.append(source[start : index + 1].strip())
            start = index + 1
    if source[start:].strip():
        lines.append(source[start:].strip())
    return lines


def _parse_type_declaration(source: str) -> CTypeDeclaration | None:
    text = source.strip().removesuffix(";").strip()
    storage: list[str] = []
    cursor = 0
    while True:
        match = re.match(r"\s*([A-Za-z_]\w*)", text[cursor:])
        if match is None or match.group(1) not in {
            "typedef", "static", "extern", "register", "_Thread_local"
        }:
            break
        storage.append(match.group(1))
        cursor += match.end()
    match = re.match(r"\s*(struct|union|enum)\b", text[cursor:])
    if match is None:
        return None
    kind = match.group(1)
    cursor += match.end()
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    tag_match = re.match(r"[A-Za-z_]\w*", text[cursor:])
    tag = tag_match.group() if tag_match else None
    if tag_match:
        cursor += len(tag)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        if tag is not None and not text[cursor:].strip():
            return CTypeDeclaration(
                kind, tag, storage_specifiers=tuple(storage), definition=False
            )
        return None
    close = _matching_c_delimiter(text, cursor, "{", "}")
    if close is None:
        return None
    body = text[cursor + 1 : close].strip()
    suffix = text[close + 1 :].strip()
    declarators: list[CDeclarator] = []
    if suffix:
        declarators = [
            declarator
            for item in _split_top_level_commas(suffix)
            if (declarator := _parse_declarator_item(item)) is not None
        ]
        if len(declarators) != len(_split_top_level_commas(suffix)):
            return None
    if kind == "enum":
        enumerators: list[CEnumerator] = []
        for item in _split_top_level_commas(body) if body else ():
            assignment = _find_assignment(item)
            if assignment is None:
                name, value = item.strip(), None
            else:
                name, operator, value_source = assignment
                if operator != "=" or not value_source:
                    return None
                value = _parse_expression(value_source)
            if not re.fullmatch(r"[A-Za-z_]\w*", name):
                return None
            enumerators.append(CEnumerator(name, value))
        return CTypeDeclaration(
            kind,
            tag,
            enumerators=tuple(enumerators),
            declarators=tuple(declarators),
            storage_specifiers=tuple(storage),
        )
    members: list[CDeclaration] = []
    for member_source in _split_top_level_semicolons(body) if body else ():
        if not member_source:
            continue
        member = parse_c_declaration(member_source + ";")
        if member.type_name is None and not member.declarators:
            return None
        members.append(member)
    return CTypeDeclaration(
        kind,
        tag,
        members=tuple(members),
        declarators=tuple(declarators),
        storage_specifiers=tuple(storage),
    )


def _parse_sequence(
    lines: list[str], start: int, *, stop_at_brace: bool
) -> tuple[list[CStatement], int]:
    result: list[CStatement] = []
    index = start
    while index < len(lines):
        text = lines[index].strip()
        if not text:
            index += 1
            continue
        if text == "}" or text.startswith("} while "):
            if stop_at_brace:
                return result, index + (text == "}")
            result.append(CRawStatement(text))
            index += 1
            continue
        if text in {"else {", "} else {"}:
            if stop_at_brace:
                return result, index
            result.append(CRawStatement(text))
            index += 1
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*:", text):
            result.append(CLabel(text[:-1]))
            index += 1
            continue
        structured = _parse_structured_statement(lines, index)
        if structured is not None:
            statement, index = structured
            result.append(statement)
            continue
        result.append(_parse_simple_statement(text))
        index += 1
    return result, index


def _parse_structured_statement(
    lines: list[str], index: int
) -> tuple[CStatement, int] | None:
    text = lines[index].strip()
    if_header = _control_header(text, "if")
    if if_header is not None:
        condition, suffix = if_header
        if suffix == "{":
            body, next_index = _parse_sequence(lines, index + 1, stop_at_brace=True)
            otherwise: CStatement | None = None
            if next_index < len(lines) and lines[next_index].strip() in {
                "else {", "} else {"
            }:
                else_body, next_index = _parse_sequence(
                    lines, next_index + 1, stop_at_brace=True
                )
                otherwise = CBlock(tuple(else_body))
            return (
                CIf(_parse_expression(condition), CBlock(tuple(body)), otherwise),
                next_index,
            )
        if suffix:
            body = _parse_simple_statement(suffix)
            return CIf(_parse_expression(condition), body, compact=True), index + 1

    while_header = _control_header(text, "while")
    if while_header is not None:
        condition, suffix = while_header
        if suffix == "{":
            body, next_index = _parse_sequence(lines, index + 1, stop_at_brace=True)
        elif suffix:
            body = [_parse_simple_statement(suffix)]
            next_index = index + 1
        else:
            return None
        return CWhile(_parse_expression(condition), CBlock(tuple(body))), next_index

    for_header = _control_header(text, "for")
    if for_header is not None:
        header, suffix = for_header
        if suffix == "{":
            body, next_index = _parse_sequence(lines, index + 1, stop_at_brace=True)
        elif suffix:
            body = [_parse_simple_statement(suffix)]
            next_index = index + 1
        else:
            return None
        parts = _split_top_level_semicolons(header)
        if len(parts) != 3:
            return CFor(header, CBlock(tuple(body))), next_index
        initializer = _parse_for_initializer(parts[0])
        condition = _parse_expression(parts[1]) if parts[1] else None
        iteration = _parse_expression(parts[2]) if parts[2] else None
        return (
            CFor(
                header,
                CBlock(tuple(body)),
                initializer,
                condition,
                iteration,
                True,
            ),
            next_index,
        )

    if re.fullmatch(r"do\s*\{", text):
        body, next_index = _parse_sequence(lines, index + 1, stop_at_brace=True)
        condition = "true"
        if next_index < len(lines):
            do_while_line = lines[next_index].strip()
            if do_while_line.startswith("}"):
                do_while_line = do_while_line[1:].strip()
            do_while = _control_header(do_while_line, "while")
            if do_while is not None:
                condition_source, suffix = do_while
                if suffix == ";":
                    condition = condition_source
                    next_index += 1
        return CDoWhile(CBlock(tuple(body)), _parse_expression(condition)), next_index

    switch_header = _control_header(text, "switch")
    if switch_header is not None and switch_header[1] == "{":
        cases, next_index = _parse_switch_cases(lines, index + 1)
        return CSwitch(_parse_expression(switch_header[0]), tuple(cases)), next_index
    return None


def _parse_case_label(text: str) -> tuple[CExpression | None, str] | None:
    if text.startswith("default:"):
        return None, text[len("default:") :].strip()
    if not text.startswith("case "):
        return None
    source = text[len("case ") :]
    colon = _top_level_case_colon(source)
    if colon is None:
        return None
    return _parse_expression(source[:colon].strip()), source[colon + 1 :].strip()


def _top_level_case_colon(source: str) -> int | None:
    depths = {"(": 0, "[": 0, "{": 0}
    pairs = {")": "(", "]": "[", "}": "{"}
    ternary_depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(source):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character in depths:
            depths[character] += 1
        elif character in pairs and depths[pairs[character]]:
            depths[pairs[character]] -= 1
        elif not any(depths.values()):
            if character == "?":
                ternary_depth += 1
            elif character == ":":
                if ternary_depth:
                    ternary_depth -= 1
                else:
                    return index
    return None


def _parse_switch_cases(
    lines: list[str], start: int
) -> tuple[list[CCase], int]:
    cases: list[CCase] = []
    current_value: CExpression | None = None
    current_statements: list[CStatement] = []
    has_current_case = False
    index = start
    while index < len(lines):
        text = lines[index].strip()
        if text == "}":
            if has_current_case:
                cases.append(CCase(current_value, tuple(current_statements)))
            return cases, index + 1
        label = _parse_case_label(text)
        if label is not None:
            value, inline = label
            if has_current_case:
                cases.append(CCase(current_value, tuple(current_statements)))
            current_value = value
            current_statements = []
            has_current_case = True
            if inline:
                current_statements.append(_parse_simple_statement(inline))
            index += 1
            continue
        structured = _parse_structured_statement(lines, index)
        if structured is not None:
            statement, index = structured
            current_statements.append(statement)
            continue
        current_statements.append(_parse_simple_statement(text))
        index += 1
    if has_current_case:
        cases.append(CCase(current_value, tuple(current_statements)))
    return cases, index


def _parse_simple_statement(text: str) -> CStatement:
    if text.startswith("/*") or text.startswith("//"):
        return CComment(text)
    if text == "break;":
        return CBreak()
    if text == "continue;":
        return CContinue()
    if text == "return;":
        return CReturn()
    if text.startswith("return ") and text.endswith(";"):
        return CReturn(_parse_expression(text[len("return ") : -1].strip()))
    jump = re.fullmatch(r"goto\s+(\*?)(.*);", text)
    if jump:
        return CGoto(jump.group(2).strip(), bool(jump.group(1)))
    if text.endswith(";"):
        source = text[:-1].strip()
        assignment = _find_assignment(source)
        if _looks_like_declaration(source) or (
            assignment is not None
            and assignment[1] == "="
            and _looks_like_declaration(assignment[0])
        ):
            return parse_c_declaration(source + ";")
        if assignment is not None:
            left, operator, right = assignment
            return CAssignment(
                _parse_expression(left), _parse_expression(right), operator
            )
        return CExpressionStatement(_parse_expression(source))
    return CRawStatement(text)


def _find_assignment(source: str) -> tuple[str, str, str] | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(source):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}" and depth:
            depth -= 1
        elif character == "=" and depth == 0:
            previous = source[index - 1] if index else ""
            following = source[index + 1] if index + 1 < len(source) else ""
            if source[max(0, index - 2) : index] in {">>", "<<"}:
                operator = source[index - 2 : index] + "="
                return source[: index - 2].strip(), operator, source[index + 1 :].strip()
            if previous in "=!<>" or following == "=":
                continue
            operator = "="
            if previous in "+-*/%&|^":
                operator = previous + "="
                return source[: index - 1].strip(), operator, source[index + 1 :].strip()
            return source[:index].strip(), operator, source[index + 1 :].strip()
    return None


def _looks_like_declaration(source: str) -> bool:
    if _parse_c_declaration_parts(source) is not None:
        return True
    return bool(
        re.match(
            r"^(?:[A-Za-z_]\w*\s+)+\(\s*\*\s*[A-Za-z_]\w*\s*\)\s*\(",
            source,
        )
    )


def _parse_c_declaration_parts(
    source: str,
) -> tuple[tuple[str, ...], str, tuple[CDeclarator, ...], str] | None:
    split = _split_declaration_specifiers(source)
    if split is None:
        return None
    storage, type_name, declarator_source, calling_convention = split
    if not declarator_source:
        return None
    declarators = []
    for item in _split_top_level_commas(declarator_source):
        declarator = _parse_declarator_item(item)
        if declarator is None:
            return None
        declarators.append(declarator)
    if not declarators:
        return None
    return storage, type_name, tuple(declarators), calling_convention


def _split_declaration_specifiers(
    source: str,
) -> tuple[tuple[str, ...], str, str, str] | None:
    storage_classes = {
        "auto", "extern", "register", "static", "typedef", "_Thread_local"
    }
    calling_conventions = {
        "__cdecl", "__stdcall", "__fastcall", "__thiscall", "__vectorcall"
    }
    storage: list[str] = []
    type_tokens: list[str] = []
    calling_convention = ""
    cursor = 0
    has_type = False
    while cursor < len(source):
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        if cursor >= len(source):
            break
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", source[cursor:])
        if match is None:
            break
        word = match.group()
        end = cursor + len(word)
        if word in storage_classes:
            storage.append(word)
            cursor = end
            continue
        if word in calling_conventions:
            if calling_convention:
                return None
            calling_convention = word
            cursor = end
            continue
        if word in {"struct", "union", "enum"}:
            type_start = cursor
            cursor = end
            while cursor < len(source) and source[cursor].isspace():
                cursor += 1
            tag = re.match(r"[A-Za-z_][A-Za-z0-9_]*", source[cursor:])
            if tag is not None:
                cursor += len(tag.group())
                while cursor < len(source) and source[cursor].isspace():
                    cursor += 1
            if cursor < len(source) and source[cursor] == "{":
                close = _matching_c_delimiter(source, cursor, "{", "}")
                if close is None:
                    return None
                cursor = close + 1
            elif tag is None:
                return None
            type_tokens.append(source[type_start:cursor].strip())
            has_type = True
            continue
        if word == "_Atomic" and end < len(source) and source[end:].lstrip().startswith("("):
            open_paren = source.find("(", end)
            close = _matching_delimiter(source, open_paren, "(", ")")
            if close is None:
                return None
            type_tokens.append(source[cursor : close + 1].strip())
            cursor = close + 1
            has_type = True
            continue
        if word in _C_TYPE_QUALIFIERS:
            type_tokens.append(word)
            cursor = end
            continue
        if word in _C_TYPE_KEYWORDS or word.endswith("_t") or re.fullmatch(
            r"(?:u?int|__u?int|undefined)\d+_?t?", word
        ) or word in {"__int8", "__int16", "__int32", "__int64"}:
            type_tokens.append(word)
            has_type = True
            cursor = end
            continue
        if not has_type:
            # An otherwise unknown identifier is a typedef-name in this
            # conservative grammar; a later identifier starts the declarator.
            type_tokens.append(word)
            has_type = True
            cursor = end
            continue
        break
    if not has_type:
        return None
    type_name = " ".join(type_tokens).strip()
    return tuple(storage), type_name, source[cursor:].strip(), calling_convention


def _parse_declarator_item(source: str) -> CDeclarator | None:
    item = source.strip()
    if not item:
        return None
    assignment = _find_assignment(item)
    if assignment is not None:
        left, operator, initializer_source = assignment
        if operator != "=" or not initializer_source:
            return None
    else:
        left, initializer_source = item, None
    bit_width = None
    colon = _find_top_level_character(left, ":")
    if colon is not None:
        bit_width_source = left[colon + 1 :].strip()
        if not bit_width_source:
            return None
        bit_width = _parse_expression(bit_width_source)
        left = left[:colon].strip()
    parser = _CDeclaratorParser(left)
    parsed = parser.parse()
    if parsed is None:
        # Unnamed bit-fields have no direct declarator.
        if bit_width is None or left:
            return None
        name, modifiers = "", ()
    else:
        name, modifiers = parsed
    initializer = (
        _parse_initializer(initializer_source)
        if initializer_source is not None
        else None
    )
    pointers = sum(isinstance(modifier, CPointerModifier) for modifier in modifiers)
    dimensions = tuple(
        modifier.size
        for modifier in modifiers
        if isinstance(modifier, CArrayModifier)
    )
    return CDeclarator(
        name,
        pointers,
        dimensions,
        initializer,
        modifiers,
        bit_width,
    )


def _parse_initializer(source: str) -> CExpression | CInitializerList:
    tokens = _tokenize_c_expression(source)
    if tokens and tokens[0].text == "{":
        parser = _CExpressionParser(tokens)
        initializer = parser._parse_initializer_list()
        if initializer is not None and parser.position == len(tokens):
            return initializer
    return _parse_expression(source)


def _find_top_level_character(source: str, character: str) -> int | None:
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    quote: str | None = None
    escaped = False
    for index, current in enumerate(source):
        if quote is not None:
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == quote:
                quote = None
            continue
        if current in {"'", '"'}:
            quote = current
        elif current == character and not any(depths.values()):
            return index
        elif current in depths:
            depths[current] += 1
        elif current in closing and depths[closing[current]]:
            depths[closing[current]] -= 1
    return None


class _CDeclaratorParser:
    """Parse C declarators as identifier-outward pointer/array/function layers."""

    def __init__(self, source: str, *, abstract: bool = False) -> None:
        self.source = source
        self.position = 0
        self.abstract = abstract

    def parse(self) -> tuple[str, tuple[CDeclaratorModifier, ...]] | None:
        parsed = self._parse_declarator()
        self._skip_space()
        if parsed is None or self.position != len(self.source):
            return None
        return parsed

    def _parse_declarator(
        self,
    ) -> tuple[str, tuple[CDeclaratorModifier, ...]] | None:
        pointers: list[CPointerModifier] = []
        self._skip_space()
        while self._accept("*"):
            qualifiers: list[str] = []
            while True:
                self._skip_space()
                word = self._peek_word()
                if word not in _C_TYPE_QUALIFIERS:
                    break
                qualifiers.append(word)
                self.position += len(word)
            pointers.append(CPointerModifier(tuple(qualifiers)))
            self._skip_space()

        if self.position < len(self.source) and self.source[self.position] == "(":
            opening = self.position
            close = _matching_delimiter(self.source, opening, "(", ")")
            if close is None:
                return None
            inner_source = self.source[opening + 1 : close]
            inner_tokens = _tokenize_c_expression(inner_source)
            abstract_function_suffix = self.abstract and (
                not inner_source.strip()
                or (
                    inner_tokens is not None
                    and _is_type_token_sequence(inner_tokens)
                )
            )
            if abstract_function_suffix:
                name = ""
                modifiers = ()
            else:
                inner = _CDeclaratorParser(
                    inner_source, abstract=self.abstract
                ).parse()
                if inner is None:
                    return None
                name, modifiers = inner
                self.position = close + 1
        else:
            identifier = self._peek_word()
            if identifier is not None:
                name = identifier
                self.position += len(identifier)
                modifiers = ()
            elif pointers or (
                self.abstract
                and self.position < len(self.source)
                and self.source[self.position] == "["
            ):
                # Abstract declarators such as ``int *``, ``int [4]``, and
                # ``int (*)(int)`` have no identifier at this level.
                name = ""
                modifiers = ()
            else:
                return None

        result = list(modifiers)
        while True:
            self._skip_space()
            if self.position >= len(self.source):
                break
            opening = self.source[self.position]
            if opening == "[":
                close = _matching_delimiter(self.source, self.position, "[", "]")
                if close is None:
                    return None
                contents = self.source[self.position + 1 : close].strip()
                is_static = False
                qualifiers: list[str] = []
                words = contents.split()
                while words and (words[0] == "static" or words[0] in _C_TYPE_QUALIFIERS):
                    word = words.pop(0)
                    if word == "static":
                        is_static = True
                    else:
                        qualifiers.append(word)
                bound_source = " ".join(words)
                size = _parse_expression(bound_source) if bound_source else None
                result.append(CArrayModifier(size, tuple(qualifiers), is_static))
                self.position = close + 1
                continue
            if opening == "(":
                close = _matching_delimiter(self.source, self.position, "(", ")")
                if close is None:
                    return None
                contents = self.source[self.position + 1 : close].strip()
                if not contents:
                    parameters = None
                    variadic = False
                else:
                    parameter_sources = _split_top_level_commas(contents)
                    variadic = bool(parameter_sources and parameter_sources[-1] == "...")
                    if variadic:
                        parameter_sources.pop()
                    parameters = tuple(
                        parse_c_parameter(parameter)
                        for parameter in parameter_sources
                    )
                    if len(parameters) == 1 and parameters[0].source == "void":
                        parameters = (parameters[0],)
                result.append(CFunctionModifier(parameters, variadic))
                self.position = close + 1
                continue
            break
        result.extend(reversed(pointers))
        return name, tuple(result)

    def _skip_space(self) -> None:
        while self.position < len(self.source) and self.source[self.position].isspace():
            self.position += 1

    def _peek_word(self) -> str | None:
        if self.position >= len(self.source):
            return None
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", self.source[self.position :])
        return match.group() if match else None

    def _accept(self, text: str) -> bool:
        self._skip_space()
        if self.source.startswith(text, self.position):
            self.position += len(text)
            return True
        return False


def _parse_for_initializer(source: str) -> CStatement | CExpression | None:
    initializer = source.strip()
    if not initializer:
        return None
    assignment = _find_assignment(initializer)
    if _looks_like_declaration(initializer) or (
        assignment is not None
        and assignment[1] == "="
        and _looks_like_declaration(assignment[0])
    ):
        return parse_c_declaration(initializer + ";")
    return _parse_expression(initializer)


def _split_top_level_commas(source: str) -> list[str]:
    return _split_top_level_delimiter(source, ",")


def _split_top_level_semicolons(source: str) -> list[str]:
    return _split_top_level_delimiter(source, ";")


def _split_top_level_delimiter(source: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    quote: str | None = None
    escaped = False
    for index, character in enumerate(source):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character in depths:
            depths[character] += 1
        elif character in closing and depths[closing[character]]:
            depths[closing[character]] -= 1
        elif character == delimiter and not any(depths.values()):
            parts.append(source[start:index].strip())
            start = index + 1
    parts.append(source[start:].strip())
    return parts


def _control_header(source: str, keyword: str) -> tuple[str, str] | None:
    match = re.match(rf"\s*{re.escape(keyword)}\s*\(", source)
    if match is None:
        return None
    open_paren = match.end() - 1
    close_paren = _matching_c_delimiter(source, open_paren, "(", ")")
    if close_paren is None:
        return None
    condition = source[open_paren + 1 : close_paren]
    suffix = source[close_paren + 1 :].strip()
    return condition, suffix


def _control_condition(source: str, keyword: str) -> str:
    header = _control_header(source, keyword)
    return header[0] if header is not None else "true"


def _matching_c_delimiter(
    source: str, start: int, opening: str, closing: str
) -> int | None:
    """Find a C delimiter while ignoring strings, character literals, and comments."""
    depth = 0
    quote: str | None = None
    escaped = False
    index = start
    while index < len(source):
        if quote is not None:
            character = source[index]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            index += 1
            continue
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            index = len(source) if end < 0 else end + 2
            continue
        character = source[index]
        if character in {"'", '"'}:
            quote = character
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _matching_delimiter(source: str, start: int, opening: str, closing: str) -> int | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(source)):
        character = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


@dataclass(frozen=True, slots=True)
class _CExpressionToken:
    kind: str
    text: str


_C_EXPRESSION_TOKEN_RE = re.compile(
    r"""
    (?P<space>\s+)
    |(?P<float>
        0[xX](?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)[pP][+-]?[0-9]+[fFlL]?
        |(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?[fFlL]?
        |[0-9]+[eE][+-]?[0-9]+[fFlL]?
    )
    |(?P<integer>
        0[xX][0-9a-fA-F]+[uUlL]*
        |0[bB][01]+[uUlL]*
        |0[0-7]*[uUlL]*
        |[1-9][0-9]*[uUlL]*
    )
    |(?P<string>(?:u8|u|U|L)?"(?:\\.|[^"\\])*")
    |(?P<char>(?:u8|u|U|L)?'(?:\\.|[^'\\])*')
    |(?P<identifier>[A-Za-z_][A-Za-z0-9_]*)
    |(?P<operator>
        >>>=|<<=|\+\+|--|->|\+=|-=|\*=|/=|%=|&=|\|=|\^=|==|!=|<=|>=|&&|\|\||<<|>>|\.\.\.
        |[()\[\]{}.,;?:+*/%&|^~!<>=-]
    )
    """,
    re.VERBOSE,
)

_C_BINARY_PRECEDENCE = {
    ",": 1,
    "=": 2,
    "+=": 2,
    "-=": 2,
    "*=": 2,
    "/=": 2,
    "%=": 2,
    "<<=": 2,
    ">>=": 2,
    "&=": 2,
    "^=": 2,
    "|=": 2,
    "||": 4,
    "&&": 5,
    "|": 6,
    "^": 7,
    "&": 8,
    "==": 9,
    "!=": 9,
    "<": 10,
    "<=": 10,
    ">": 10,
    ">=": 10,
    "<<": 11,
    ">>": 11,
    "+": 12,
    "-": 12,
    "*": 13,
    "/": 13,
    "%": 13,
}
_C_RIGHT_ASSOCIATIVE = frozenset(
    {"=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "^=", "|="}
)
_C_PREFIX_OPERATORS = frozenset({"+", "-", "!", "~", "*", "&", "++", "--"})
_C_TYPE_QUALIFIERS = frozenset({"const", "volatile", "restrict", "_Atomic"})
_C_TYPE_KEYWORDS = frozenset(
    {
        "void",
        "char",
        "short",
        "int",
        "long",
        "float",
        "double",
        "signed",
        "unsigned",
        "_Bool",
        "bool",
        "size_t",
        "ssize_t",
        "ptrdiff_t",
        "intptr_t",
        "uintptr_t",
        "__int128",
        "__uint128_t",
        "byte",
        "word",
        "dword",
        "qword",
    }
)


def _tokenize_c_expression(source: str) -> list[_CExpressionToken] | None:
    tokens: list[_CExpressionToken] = []
    position = 0
    while position < len(source):
        match = _C_EXPRESSION_TOKEN_RE.match(source, position)
        if match is None:
            return None
        kind = match.lastgroup
        text = match.group()
        position = match.end()
        if kind != "space":
            tokens.append(_CExpressionToken(kind or "operator", text))
    return tokens


def _decode_c_string_token(source: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r'(?P<prefix>u8|u|U|L)?(?P<literal>"(?:\\.|[^"\\])*")',
        source,
    )
    if match is None:
        return None
    literal = match.group("literal")
    try:
        value = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        value = literal[1:-1]
    if not isinstance(value, str):
        value = literal[1:-1]
    return match.group("prefix") or "", value


def _decode_c_character_token(source: str) -> tuple[str, str] | None:
    prefix = ""
    for candidate in ("u8", "u", "U", "L"):
        if source.startswith(candidate):
            prefix = candidate
            break
    literal = source[len(prefix) :]
    if len(literal) < 2 or not literal.startswith("'") or not literal.endswith("'"):
        return None
    return prefix, literal


def _is_type_token_sequence(tokens: list[_CExpressionToken]) -> bool:
    if not tokens:
        return False
    words = [token.text for token in tokens if token.kind == "identifier"]
    if len(words) + sum(token.text == "*" for token in tokens) != len(tokens):
        return False
    qualifiers = set(_C_TYPE_QUALIFIERS)
    has_type = False
    expect_tag = False
    for word in words:
        if expect_tag:
            expect_tag = False
            has_type = True
            continue
        if word in {"struct", "union", "enum"}:
            expect_tag = True
            continue
        if word in qualifiers:
            continue
        if word in _C_TYPE_KEYWORDS or re.fullmatch(
            r"(?:u?int|__u?int|undefined)\d+_?t?", word
        ):
            has_type = True
            continue
        if word.endswith("_t") or word in {"__int64", "__int32", "__int16", "__int8"}:
            has_type = True
            continue
        return False
    return has_type and not expect_tag


def _format_c_type_tokens(tokens: list[_CExpressionToken]) -> str:
    source = " ".join(token.text for token in tokens)
    source = re.sub(r"\s+([;,\)\]])", r"\1", source)
    source = re.sub(r"\{\s+", "{ ", source)
    source = re.sub(r"\(\s+", "(", source)
    source = re.sub(r"\[\s+", "[", source)
    return source.strip()


def _type_name_from_tokens(
    tokens: list[_CExpressionToken], *, allow_unknown_typedef: bool
) -> CTypeName | None:
    if not tokens:
        return None
    source = _format_c_type_tokens(tokens)
    type_name = parse_c_type_name(source)
    if isinstance(type_name.base, CRawTypeSpecifier):
        return None
    if (
        not allow_unknown_typedef
        and isinstance(type_name.base, CTypedefType)
        and type_name.declarator is None
        and not type_name.base.name.endswith("_t")
        and not _is_type_token_sequence(tokens)
    ):
        return None
    return type_name


def _compound_literal_type_name(tokens: list[_CExpressionToken]) -> str | None:
    if _type_name_from_tokens(tokens, allow_unknown_typedef=True) is not None:
        return _format_c_type_tokens(tokens)
    index = 0
    while index < len(tokens) and tokens[index].text in _C_TYPE_QUALIFIERS:
        index += 1
    if index >= len(tokens) or tokens[index].text not in {"struct", "union", "enum"}:
        return None
    index += 1
    if index < len(tokens) and tokens[index].kind == "identifier":
        index += 1
    if index >= len(tokens) or tokens[index].text != "{":
        return None
    depth = 0
    for token_index in range(index, len(tokens)):
        text = tokens[token_index].text
        if text == "{":
            depth += 1
        elif text == "}":
            depth -= 1
            if depth == 0 and token_index != len(tokens) - 1:
                return None
    if depth != 0:
        return None
    return _format_c_type_tokens(tokens)


class _CExpressionParser:
    def __init__(self, tokens: list[_CExpressionToken]) -> None:
        self.tokens = tokens
        self.position = 0

    def parse(self) -> CExpression | None:
        if not self.tokens:
            return None
        expression = self._parse_expression(1)
        if expression is None or self.position != len(self.tokens):
            return None
        return expression

    def _peek(self, text: str | None = None) -> _CExpressionToken | None:
        if self.position >= len(self.tokens):
            return None
        token = self.tokens[self.position]
        if text is not None and token.text != text:
            return None
        return token

    def _accept(self, text: str) -> bool:
        if self._peek(text) is None:
            return False
        self.position += 1
        return True

    def _parse_expression(self, minimum_precedence: int) -> CExpression | None:
        left = self._parse_prefix()
        if left is None:
            return None
        left = self._parse_postfix(left)
        if left is None:
            return None

        while self.position < len(self.tokens):
            token = self._peek()
            if token is None:
                break
            if token.text == "?":
                precedence = 3
                if precedence < minimum_precedence:
                    break
                self.position += 1
                if_true = self._parse_expression(1)
                if if_true is None or not self._accept(":"):
                    return None
                if_false = self._parse_expression(precedence)
                if if_false is None:
                    return None
                left = CConditional(left, if_true, if_false)
                continue

            precedence = _C_BINARY_PRECEDENCE.get(token.text)
            if precedence is None or precedence < minimum_precedence:
                break
            self.position += 1
            next_minimum = (
                precedence
                if token.text in _C_RIGHT_ASSOCIATIVE
                else precedence + 1
            )
            right = self._parse_expression(next_minimum)
            if right is None:
                return None
            left = CBinary(token.text, left, right)
        return left

    def _parse_prefix(self) -> CExpression | None:
        token = self._peek()
        if token is None:
            return None
        if token.kind == "identifier" and token.text in {
            "sizeof", "_Alignof", "alignof"
        }:
            operator = token.text
            self.position += 1
            is_alignof = operator != "sizeof"
            if self._accept("("):
                close = self._matching_parenthesis(self.position - 1)
                if close is None:
                    return None
                type_tokens = self.tokens[self.position : close]
                type_name = _type_name_from_tokens(
                    type_tokens, allow_unknown_typedef=is_alignof
                )
                if type_name is not None:
                    self.position = close + 1
                    return CSizeof(type_name, operator)
                if is_alignof:
                    return None
                operand = self._parse_expression(1)
                if operand is None or not self._accept(")"):
                    return None
                return CSizeof(operand, operator)
            if is_alignof:
                return None
            operand = self._parse_expression(14)
            return CSizeof(operand, operator) if operand is not None else None
        if token.text in _C_PREFIX_OPERATORS:
            self.position += 1
            operand = self._parse_expression(14)
            return CUnary(token.text, operand) if operand is not None else None
        if token.text == "(":
            close = self._matching_parenthesis(self.position)
            if close is None:
                return None
            type_tokens = self.tokens[self.position + 1 : close]
            compound_type = _compound_literal_type_name(type_tokens)
            if (
                close + 1 < len(self.tokens)
                and self.tokens[close + 1].text == "{"
                and compound_type is not None
            ):
                self.position = close + 1
                initializer = self._parse_initializer_list()
                if initializer is None:
                    return None
                return CCompoundLiteral(compound_type, initializer)
            cast_type = _type_name_from_tokens(
                type_tokens, allow_unknown_typedef=False
            )
            if close + 1 < len(self.tokens) and cast_type is not None:
                type_name = _format_c_type_tokens(type_tokens)
                self.position = close + 1
                value = self._parse_expression(14)
                return CCast(type_name, value, cast_type) if value is not None else None
            self.position += 1
            inner = self._parse_expression(1)
            if inner is None or not self._accept(")"):
                return None
            return inner
        if token.kind == "identifier":
            self.position += 1
            return CIdentifier(token.text)
        if token.kind == "string":
            self.position += 1
            decoded = _decode_c_string_token(token.text)
            if decoded is None:
                return CRawExpression(token.text)
            prefix, value = decoded
            return CString(value, prefix, token.text)
        if token.kind == "char":
            self.position += 1
            decoded = _decode_c_character_token(token.text)
            if decoded is None:
                return CRawExpression(token.text)
            prefix, literal = decoded
            return CCharacter(literal, prefix)
        if token.kind == "float":
            self.position += 1
            return CFloat(token.text)
        if token.kind == "integer":
            self.position += 1
            suffix_match = re.search(r"[uUlL]+$", token.text)
            suffix = suffix_match.group() if suffix_match else ""
            digits = token.text[: -len(suffix)] if suffix else token.text
            if digits.lower().startswith("0b"):
                return CRawExpression(token.text)
            if digits.lower().startswith("0x"):
                base = 16
            elif len(digits) > 1 and digits.startswith("0"):
                base = 8
            else:
                base = 10
            try:
                value = int(digits, base)
            except ValueError:
                return None
            return CInteger(value, base, suffix)
        return None

    def _parse_initializer_list(self) -> CInitializerList | None:
        if not self._accept("{"):
            return None
        entries: list[CInitializerEntry] = []
        if self._accept("}"):
            return CInitializerList()
        while True:
            designators: list[CDesignator] = []
            while True:
                if self._accept("."):
                    member = self._peek()
                    if member is None or member.kind != "identifier":
                        return None
                    self.position += 1
                    designators.append(CFieldDesignator(member.text))
                elif self._accept("["):
                    index = self._parse_expression(1)
                    if index is None or not self._accept("]"):
                        return None
                    designators.append(CArrayDesignator(index))
                else:
                    break
            if designators:
                if not self._accept("="):
                    return None
                value = self._parse_initializer_value()
                if value is None:
                    return None
                first_member = next(
                    (
                        item.name
                        for item in designators
                        if isinstance(item, CFieldDesignator)
                    ),
                    "",
                )
                entries.append(
                    CDesignatedInitializer(first_member, value, tuple(designators))
                )
            else:
                value = self._parse_initializer_value()
                if value is None:
                    return None
                entries.append(value)
            if self._accept("}"):
                break
            if not self._accept(","):
                return None
            if self._accept("}"):
                break
        return CInitializerList(tuple(entries))

    def _parse_initializer_value(self) -> CExpression | CInitializerList | None:
        if self._peek("{") is not None:
            return self._parse_initializer_list()
        return self._parse_expression(2)

    def _parse_postfix(self, expression: CExpression) -> CExpression | None:
        while self.position < len(self.tokens):
            token = self._peek()
            if token is not None and token.kind == "string" and isinstance(expression, CString):
                decoded = _decode_c_string_token(token.text)
                if decoded is None:
                    return None
                prefix, value = decoded
                literal = expression.literal
                if literal is None:
                    literal = expression.prefix + json.dumps(
                        expression.value, ensure_ascii=True
                    )
                expression = CString(
                    expression.value + value,
                    expression.prefix or prefix,
                    literal + token.text,
                )
                self.position += 1
                continue
            if self._accept("("):
                arguments: list[CExpression] = []
                if not self._accept(")"):
                    while True:
                        argument = self._parse_expression(2)
                        if argument is None:
                            return None
                        arguments.append(argument)
                        if self._accept(")"):
                            break
                        if not self._accept(","):
                            return None
                expression = CCall(expression, tuple(arguments))
                continue
            if self._accept("["):
                index = self._parse_expression(1)
                if index is None or not self._accept("]"):
                    return None
                expression = CIndex(expression, index)
                continue
            if self._accept(".") or self._accept("->"):
                operator = self.tokens[self.position - 1].text
                member = self._peek()
                if member is None or member.kind != "identifier":
                    return None
                self.position += 1
                expression = CMember(expression, member.text, operator == "->")
                continue
            if self._accept("++"):
                expression = CUnary("++", expression, postfix=True)
                continue
            if self._accept("--"):
                expression = CUnary("--", expression, postfix=True)
                continue
            break
        return expression

    def _matching_parenthesis(self, start: int) -> int | None:
        depth = 0
        for index in range(start, len(self.tokens)):
            if self.tokens[index].text == "(":
                depth += 1
            elif self.tokens[index].text == ")":
                depth -= 1
                if depth == 0:
                    return index
        return None


def _parse_expression(source: str) -> CExpression:
    """Parse the supported C expression subset, retaining unknown syntax raw."""
    tokens = _tokenize_c_expression(source)
    if tokens is None:
        return CRawExpression(source)
    parsed = _CExpressionParser(tokens).parse()
    return parsed if parsed is not None else CRawExpression(source)
