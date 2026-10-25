"""Structural bridge between MiniGhidraPy's AST backend and Ghidra's printers.

Ghidra's ``PrintC`` walks ``Funcdata``: a symbol table, a ``TypeFactory``, the
collapsed ``FlowBlock`` structure and the p-code operations of every basic
block.  MiniGhidraPy's decompiler backend keeps its own expression AST
(``_Expr``) and renders text itself, so this module builds the equivalent
document from what the backend already recovers:

* :class:`PrintType`/:class:`PrintSymbol`/:class:`PrintScope` resolve every
  identifier and data-type through the analyzer's :class:`~minighidra.typesys.TypeDatabase`
  and the recovered :class:`~minighidra.model.FunctionSignature` (parameters
  keep their ABI locations, locals keep their inferred width, globals resolve
  through ``Analyzer.name_at``).
* :class:`PrintExpr` is the normalized AST node the printer pushes on its RPN
  stack.  Expressions the backend renders as composite C text (x87, MMX, SSE
  lane arithmetic, ...) are carried through as *opaque* nodes holding that text,
  so nothing is lost when the printer cannot decompose them.
* :class:`PrintStatement`/:class:`PrintTerminator` give the printer statements
  and branches per basic block, associated with the instruction address (the
  role ``PcodeOp::getTime``/``getAddr`` play in Ghidra's markup).
* :class:`PrintableFunction` ties it together with the collapsed control-flow
  structure produced by :mod:`minighidra.block` and :mod:`minighidra.collapse`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

from .emit import SyntaxHighlight
from .model import BasicBlock, Function, FunctionParameter, FunctionSignature
from .printlanguage import Comment, CommentType
from .typesys import CType, StorageLocation, TypeDatabase, TypeKind


class DisplayFormat(IntEnum):
    """``Symbol`` display formats used by ``push_integer``."""

    none = 0
    force_hex = 1
    force_dec = 2
    force_oct = 3
    force_bin = 4
    force_char = 5


class SymbolCategory(IntEnum):
    """``Symbol::symbol_category``."""

    no_category = -1
    function_parameter = 0
    equate = 1


# Ghidra's type_metatype names, used by the printer's type rules.
METATYPE_VOID = "void"
METATYPE_UNKNOWN = "unknown"
METATYPE_INT = "int"
METATYPE_UINT = "uint"
METATYPE_BOOL = "bool"
METATYPE_FLOAT = "float"
METATYPE_PTR = "ptr"
METATYPE_ARRAY = "array"
METATYPE_STRUCT = "struct"
METATYPE_UNION = "union"
METATYPE_ENUM = "enum"
METATYPE_CODE = "code"


@dataclass(frozen=True, slots=True)
class PrintAddress:
    """An address in a named space (``Address``/``AddrSpace`` in Ghidra)."""

    space: str
    offset: int
    size: int = 0

    @property
    def name(self) -> str:
        return self.space

    @property
    def shortcut(self) -> str:
        """``AddrSpace::getShortcut`` for the spaces MiniGhidraPy models."""
        return {
            "ram": "r",
            "register": "%",
            "stack": "s",
            "const": "#",
            "unique": "u",
        }.get(self.space, "x")

    @property
    def address_size(self) -> int:
        return 8 if self.offset > 0xFFFFFFFF else 4

    def print_raw(self) -> str:
        """``AddrSpace::printRaw``: '0x' plus zero padded offset."""
        size = self.address_size
        if size > 4:
            if (self.offset >> 32) == 0:
                size = 4
            elif (self.offset >> 48) == 0:
                size = 6
        return f"0x{self.offset:0{2 * size}x}"

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.space}:{self.print_raw()}"


@dataclass(frozen=True)
class PrintScope:
    """A symbol scope (``Scope`` in Ghidra)."""

    name: str
    is_global: bool = False
    parent: Optional["PrintScope"] = None
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            object.__setattr__(self, "display_name", self.name)


@dataclass(frozen=True)
class PrintField:
    """A component of a structured data-type (``TypeField``)."""

    name: str
    ctype: "PrintType"
    offset: int
    size: int
    ident: int


@dataclass(frozen=True)
class PrintType:
    """A resolved data-type with Ghidra's printing semantics."""

    ctype: Optional[CType]
    name: str
    metatype: str
    size: int
    signed: bool = False
    char_print: bool = False
    enum: bool = False
    display_format: int = DisplayFormat.none.value
    unsized_id: int = 0
    pointer_to: Optional["PrintType"] = None
    base: Optional["PrintType"] = None
    count: int = 0
    fields: tuple[PrintField, ...] = ()
    parameters: tuple["PrintType", ...] = ()
    variadic: bool = False
    core: bool = False

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def needs_resolution(self) -> bool:
        return self.metatype == METATYPE_UNION

    def is_pointer_to_array(self) -> bool:
        return (
            self.metatype == METATYPE_PTR
            and self.pointer_to is not None
            and self.pointer_to.metatype == METATYPE_ARRAY
        )


@dataclass
class PrintSymbol:
    """A resolved symbol (``Symbol`` in Ghidra)."""

    name: str
    ctype: PrintType
    scope: PrintScope
    category: int = SymbolCategory.no_category.value
    symbol_id: int = 0
    address: Optional[PrintAddress] = None
    locations: tuple[StorageLocation, ...] = ()
    display_format: int = DisplayFormat.none.value
    name_locked: bool = False
    volatile: bool = False
    implicit: bool = False
    this_pointer: bool = False
    merge_problems: bool = False

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def size(self) -> int:
        return self.ctype.size


@dataclass
class PrintOp:
    """Markup reference for a statement/operator (``PcodeOp`` role)."""

    op_time: int
    address: Optional[PrintAddress] = None
    kind: str = ""

    @property
    def addr(self) -> Optional[PrintAddress]:
        return self.address


@dataclass
class PrintExpr:
    """Normalized AST node consumed by :class:`~minighidra.printc.PrintC`."""

    kind: str
    args: tuple["PrintExpr", ...] = ()
    bits: int = 0
    ctype: Optional[PrintType] = None
    symbol: Optional[PrintSymbol] = None
    symbol_offset: int = -1
    int_value: int = 0
    float_value: Optional[float] = None
    name: str = ""
    text: Optional[str] = None
    address: Optional[PrintAddress] = None
    op: Optional[PrintOp] = None
    create_index: int = 0
    slot: int = -1
    display_format: int = DisplayFormat.none.value
    signed: bool = False

    @property
    def size(self) -> int:
        return max(1, self.bits // 8)


@dataclass
class PrintCall:
    """A recovered call site."""

    op: PrintOp
    target_name: str
    target_address: Optional[int]
    arguments: tuple[PrintExpr, ...]
    result: Optional[PrintExpr]
    result_symbol: Optional[PrintSymbol]
    no_return: bool
    indirect: bool
    target_expr: Optional[PrintExpr] = None
    prototype: Optional["PrintPrototype"] = None


@dataclass
class PrintStatement:
    """One statement inside a basic block."""

    kind: str
    op: PrintOp
    text: str = ""
    out_symbol: Optional[PrintSymbol] = None
    out_name: str = ""
    expression: Optional[PrintExpr] = None
    values: tuple[PrintExpr, ...] = ()
    call: Optional[PrintCall] = None
    comment: str = ""
    decl_type: Optional[PrintType] = None
    decl_name: str = ""
    label: str = ""


@dataclass
class PrintTerminator:
    """How a basic block transfers control."""

    kind: str
    op: PrintOp
    condition: Optional[PrintExpr] = None
    true_target: Optional[int] = None
    fallthrough: Optional[int] = None
    targets: tuple[int, ...] = ()
    switch_value: Optional[PrintExpr] = None
    switch_cases: tuple[tuple[int, int], ...] = ()
    indirect: Optional[PrintExpr] = None
    tail_call: Optional[str] = None
    tail_call_address: Optional[int] = None
    tail_call_no_return: bool = False
    return_value: Optional[PrintExpr] = None
    text: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrintParameter:
    """A formal parameter with its resolved type and ABI locations."""

    symbol: PrintSymbol
    ctype: PrintType
    locations: tuple[StorageLocation, ...] = ()
    implicit: bool = False
    this_pointer: bool = False


@dataclass(frozen=True)
class PrintPrototype:
    """``FuncProto`` equivalent."""

    output_type: PrintType
    parameters: tuple[PrintParameter, ...]
    variadic: bool = False
    model_name: str = ""
    print_model_in_decl: bool = False
    model_unknown: bool = False
    hidden_return: Optional[PrintParameter] = None

    def num_params(self) -> int:
        return len(self.parameters)

    def is_dotdotdot(self) -> bool:
        return self.variadic


@dataclass
class PrintableFunction:
    """``Funcdata`` equivalent: everything needed to emit one function."""

    function: Function
    name: str
    address: PrintAddress
    prototype: PrintPrototype
    scope: PrintScope
    symbols: dict[str, PrintSymbol]
    local_symbols: tuple[PrintSymbol, ...]
    statements_by_block: dict[int, tuple[PrintStatement, ...]]
    terminators: dict[int, PrintTerminator]
    structure: Any
    blocks: tuple[BasicBlock, ...]
    comments: dict[int, tuple[Comment, ...]]
    labels: dict[int, str]
    header_comments: tuple[Comment, ...]
    artifacts: Any
    resolver: "TypeResolver"
    symbol_table: "SymbolTable"
    is_thunk: bool = False
    text: str = ""

    @property
    def display_name(self) -> str:
        return self.name

    def entry_address(self) -> PrintAddress:
        return self.address


class TypeResolver:
    """Resolve C types (and synthetic widths) into :class:`PrintType`."""

    def __init__(self, database: TypeDatabase, bits: int) -> None:
        self.database = database
        self.bits = bits
        self._cache: dict[Any, PrintType] = {}
        self._ids: dict[str, int] = {}
        self._next_id = 1
        self.void = self.from_spec("void")

    def _type_id(self, name: str) -> int:
        if not name:
            return 0
        if name not in self._ids:
            self._ids[name] = self._next_id
            self._next_id += 1
        return self._ids[name]

    @staticmethod
    def metatype_of(ctype: CType) -> str:
        base = ctype.unwrapped() if ctype is not None else None
        kind = base.kind if base is not None else TypeKind.UNKNOWN
        if kind == TypeKind.VOID:
            return METATYPE_VOID
        if kind == TypeKind.UNKNOWN:
            return METATYPE_UNKNOWN
        if kind == TypeKind.INTEGER:
            return METATYPE_INT if base.signed else METATYPE_UINT
        if kind == TypeKind.BOOL:
            return METATYPE_BOOL
        if kind == TypeKind.FLOAT:
            return METATYPE_FLOAT
        if kind == TypeKind.POINTER:
            return METATYPE_PTR
        if kind == TypeKind.ARRAY:
            return METATYPE_ARRAY
        if kind == TypeKind.STRUCT:
            return METATYPE_STRUCT
        if kind == TypeKind.UNION:
            return METATYPE_UNION
        if kind == TypeKind.ENUM:
            return METATYPE_ENUM
        if kind == TypeKind.FUNCTION:
            return METATYPE_CODE
        return METATYPE_UNKNOWN

    def from_spec(self, spec: Any) -> PrintType:
        """Resolve a type name/:class:`CType` into a :class:`PrintType`."""
        if isinstance(spec, PrintType):
            return spec
        key = spec if isinstance(spec, str) else id(spec)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            ctype = self.database.resolve(spec)
        except (KeyError, ValueError, TypeError):
            ctype = self.database.get(f"uint{self.bits}_t")
        result = self._build(ctype)
        self._cache[key] = result
        return result

    def _build(self, ctype: CType) -> PrintType:
        metatype = self.metatype_of(ctype)
        base = ctype.unwrapped()
        size = int(ctype.size or base.size or 0)
        signed = bool(base.signed) if base.signed is not None else False
        char_print = base.kind == TypeKind.INTEGER and size == 1 and ctype.name in {
            "char",
            "signed char",
            "unsigned char",
        }
        pointer_to = None
        base_type = None
        count = 0
        fields: tuple[PrintField, ...] = ()
        parameters: tuple[PrintType, ...] = ()
        variadic = bool(ctype.variadic)
        if ctype.kind == TypeKind.POINTER and ctype.target is not None:
            pointer_to = self.from_spec(ctype.target)
        elif ctype.kind == TypeKind.ARRAY and ctype.target is not None:
            base_type = self.from_spec(ctype.target)
            count = int(ctype.count or 0)
        elif ctype.kind == TypeKind.FUNCTION:
            base_type = (
                self.from_spec(ctype.target)
                if ctype.target is not None
                else self.void
            )
            parameters = tuple(self.from_spec(item) for item in ctype.parameters)
        elif ctype.kind in {TypeKind.STRUCT, TypeKind.UNION}:
            fields = tuple(
                PrintField(
                    name=item.name,
                    ctype=self.from_spec(item.type),
                    offset=int(item.offset),
                    size=int(item.type.size or 0),
                    ident=index,
                )
                for index, item in enumerate(ctype.fields)
            )
        return PrintType(
            ctype=ctype,
            name=ctype.name,
            metatype=metatype,
            size=size,
            signed=signed,
            char_print=char_print,
            enum=ctype.kind == TypeKind.ENUM,
            unsized_id=self._type_id(ctype.name),
            pointer_to=pointer_to,
            base=base_type,
            count=count,
            fields=fields,
            parameters=parameters,
            variadic=variadic,
            core=ctype.name in _CORE_TYPE_NAMES,
        )

    def integer(self, bits: int, signed: bool = False) -> PrintType:
        width = bits if bits in (8, 16, 32, 64, 128) else _round_width(bits)
        if width > 64:
            name = "__int128_t" if signed else "__uint128_t"
        else:
            name = f"{'int' if signed else 'uint'}{width}_t"
        try:
            return self.from_spec(name)
        except (KeyError, ValueError):  # pragma: no cover - defensive
            return self.from_spec(f"uint{self.bits}_t")

    def floating(self, bits: int) -> PrintType:
        name = "float" if bits <= 32 else ("double" if bits <= 64 else "long double")
        return self.from_spec(name)

    def pointer(self, target: PrintType) -> PrintType:
        key = ("pointer", id(target))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        spec = f"{target.name} *"
        try:
            # The database may not name composite targets (pointer to array,
            # pointer to function); build the pointer type directly then.
            ctype = self.database.resolve(spec)
        except (KeyError, ValueError, TypeError):
            ctype = CType(
                name=spec,
                kind=TypeKind.POINTER,
                size=self.bits // 8,
                alignment=self.bits // 8,
                signed=False,
                target=target.ctype,
            )
        result = self._build(ctype)
        self._cache[key] = result
        return result


_CORE_TYPE_NAMES = frozenset(
    {
        "void",
        "bool",
        "char",
        "signed char",
        "unsigned char",
        "short",
        "unsigned short",
        "int",
        "unsigned int",
        "long",
        "unsigned long",
        "long long",
        "unsigned long long",
        "float",
        "double",
        "long double",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "__int128_t",
        "__uint128_t",
    }
)


def _round_width(bits: int) -> int:
    for width in (8, 16, 32, 64, 128):
        if bits <= width:
            return width
    return 128


class SymbolTable:
    """Resolve identifier names to symbols with types and highlighting."""

    def __init__(
        self,
        analyzer: Any,
        resolver: TypeResolver,
        bits: int,
        function: Function,
    ) -> None:
        self.analyzer = analyzer
        self.resolver = resolver
        self.bits = bits
        self.function = function
        self.global_scope = PrintScope(name="global", is_global=True)
        self.local_scope = PrintScope(
            name=function.name or f"FUN_{function.address:08x}",
            is_global=False,
            parent=self.global_scope,
        )
        self.symbols: dict[str, PrintSymbol] = {}
        self._next_id = 1
        self._global_names = set(getattr(analyzer, "_name_by_address", {}).values())

    def _register(self, symbol: PrintSymbol) -> PrintSymbol:
        symbol.symbol_id = self._next_id
        self._next_id += 1
        self.symbols[symbol.name] = symbol
        return symbol

    def declare_parameter(
        self,
        parameter: FunctionParameter,
        ctype: PrintType,
        *,
        implicit: bool = False,
        this_pointer: bool = False,
    ) -> PrintSymbol:
        name = _c_identifier(parameter.name)
        symbol = self._register(
            PrintSymbol(
                name=name,
                ctype=ctype,
                scope=self.local_scope,
                category=SymbolCategory.function_parameter.value,
                locations=tuple(parameter.locations),
                implicit=implicit or parameter.implicit,
                this_pointer=this_pointer,
                display_format=ctype.display_format,
            )
        )
        return symbol

    def declare_local(self, name: str, type_spec: Any) -> PrintSymbol:
        existing = self.symbols.get(name)
        if existing is not None:
            return existing
        return self._register(
            PrintSymbol(
                name=name,
                ctype=self.resolver.from_spec(type_spec),
                scope=self.local_scope,
            )
        )

    def lookup(self, name: str, bits: int = 0) -> Optional[PrintSymbol]:
        symbol = self.symbols.get(name)
        if symbol is not None:
            return symbol
        if name in self._global_names:
            ctype = self.resolver.integer(bits or self.bits)
            return self._register(
                PrintSymbol(
                    name=name,
                    ctype=ctype,
                    scope=self.global_scope,
                )
            )
        return None

    def anonymous(self, name: str, bits: int = 0) -> PrintSymbol:
        """Create (and cache) an undeclared local symbol."""
        existing = self.symbols.get(name)
        if existing is not None:
            return existing
        return self._register(
            PrintSymbol(
                name=name,
                ctype=self.resolver.integer(bits or self.bits),
                scope=self.local_scope,
            )
        )

    def global_at(self, address: int) -> Optional[PrintSymbol]:
        name = self.analyzer.name_at(address) if self.analyzer is not None else None
        if not name:
            return None
        identifier = _c_identifier(name)
        existing = self.symbols.get(identifier)
        if existing is not None:
            return existing
        return self._register(
            PrintSymbol(
                name=identifier,
                ctype=self.resolver.integer(self.bits),
                scope=self.global_scope,
            )
        )


def get_symbol_highlight(symbol: PrintSymbol) -> SyntaxHighlight:
    """``PrintC::getSymbolHighlight``."""
    if symbol.volatile:
        return SyntaxHighlight.special_color
    if symbol.scope.is_global:
        return SyntaxHighlight.global_color
    if symbol.category == SymbolCategory.function_parameter.value:
        return SyntaxHighlight.param_color
    if symbol.category == SymbolCategory.equate.value:
        return SyntaxHighlight.const_color
    return SyntaxHighlight.var_color


def _c_identifier(value: Optional[str]) -> str:
    if not value:
        return "unknown"
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not identifier or identifier[0].isdigit():
        identifier = "sym_" + identifier
    return identifier


# ---------------------------------------------------------------------------
# Expression conversion
# ---------------------------------------------------------------------------

_BINARY_OPS = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "imul": "mul",
    "udiv": "udiv",
    "sdiv": "sdiv",
    "urem": "urem",
    "srem": "srem",
    "and": "and",
    "or": "or",
    "xor": "xor",
    "shl": "shl",
    "lshr": "lshr",
    "ashr": "ashr",
    "eq": "eq",
    "ne": "ne",
    "ult": "ult",
    "ule": "ule",
    "ugt": "ugt",
    "uge": "uge",
    "slt": "slt",
    "sle": "sle",
    "sgt": "sgt",
    "sge": "sge",
    "land": "land",
    "lor": "lor",
}

_SIGNED_COMPARISONS = frozenset({"slt", "sle", "sgt", "sge"})
_UNSIGNED_COMPARISONS = frozenset({"ult", "ule", "ugt", "uge"})


class ExpressionConverter:
    """Convert backend ``_Expr`` nodes into normalized :class:`PrintExpr`."""

    def __init__(
        self,
        resolver: TypeResolver,
        symbols: SymbolTable,
        bits: int,
        catalog: dict[str, Any],
    ) -> None:
        self.resolver = resolver
        self.symbols = symbols
        self.bits = bits
        self.catalog = catalog
        self._reverse: dict[int, str] = {}
        for text, expr in catalog.items():
            self._reverse.setdefault(id(expr), text)
        self._memo: dict[int, PrintExpr] = {}
        self._next_index = 1

    def opaque_text(self, expr: Any) -> str:
        text = self._reverse.get(id(expr))
        if text is not None:
            return text
        return f"unknown_expr_{getattr(expr, 'kind', 'node')}"

    def convert(self, expr: Any, op: Optional[PrintOp] = None) -> Optional[PrintExpr]:
        if expr is None:
            return None
        key = id(expr)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        result = self._convert(expr, op)
        self._memo[key] = result
        return result

    def _convert(self, expr: Any, op: Optional[PrintOp]) -> PrintExpr:
        kind = str(getattr(expr, "kind", ""))
        value = getattr(expr, "value", None)
        args = tuple(getattr(expr, "args", ()) or ())
        bits = int(getattr(expr, "bits", 0) or 0)
        index = self._next_index
        self._next_index += 1
        common = {
            "bits": bits,
            "op": op,
            "create_index": index,
        }

        if kind == "const":
            signed = False
            int_value = int(value or 0)
            return PrintExpr(
                kind="const",
                int_value=int_value,
                ctype=self.resolver.integer(bits or self.bits, signed),
                signed=signed,
                **common,
            )
        if kind == "fp_const":
            number = float(value or 0.0)
            return PrintExpr(
                kind="fp_const",
                float_value=number,
                ctype=self.resolver.floating(bits or 80),
                **common,
            )
        if kind == "var":
            name = _c_identifier(str(value))
            symbol = self.symbols.lookup(name, bits) or self.symbols.anonymous(
                name, bits
            )
            return PrintExpr(
                kind="var",
                symbol=symbol,
                ctype=symbol.ctype,
                name=name,
                **common,
            )
        if kind == "op":
            operator = str(value)
            converted = tuple(self.convert(arg, op) for arg in args)
            signed = operator in _SIGNED_COMPARISONS or operator in {"sdiv", "srem", "ashr"}
            width = max([arg.bits for arg in converted] + [bits, self.bits])
            ctype = self.resolver.integer(1 if _is_boolean_op(operator) else width, False)
            return PrintExpr(
                kind="op",
                args=converted,
                name=_BINARY_OPS.get(operator, operator),
                ctype=ctype,
                signed=signed,
                **common,
            )
        if kind == "unary":
            operator = str(value)
            operand = self.convert(args[0], op) if args else None
            return PrintExpr(
                kind="unary",
                args=(operand,) if operand is not None else (),
                name=operator,
                ctype=operand.ctype if operand is not None else None,
                **common,
            )
        if kind in {"zext", "sext", "trunc"}:
            operand = self.convert(args[0], op) if args else None
            signed = kind == "sext"
            return PrintExpr(
                kind=kind,
                args=(operand,) if operand is not None else (),
                ctype=self.resolver.integer(bits or self.bits, signed),
                signed=signed,
                **common,
            )
        if kind == "load":
            pointer = self.convert(args[0], op) if args else None
            return PrintExpr(
                kind="load",
                args=(pointer,) if pointer is not None else (),
                ctype=self.resolver.integer(bits or self.bits, False),
                **common,
            )
        if kind in {"typed_load", "aggregate_load"}:
            pointer = self.convert(args[0], op) if args else None
            ctype = value if isinstance(value, CType) else None
            return PrintExpr(
                kind="load_typed",
                args=(pointer,) if pointer is not None else (),
                ctype=self.resolver.from_spec(ctype) if ctype is not None else None,
                **common,
            )
        if kind == "select":
            converted = tuple(self.convert(arg, op) for arg in args)
            return PrintExpr(
                kind="select",
                args=converted,
                ctype=converted[1].ctype if len(converted) > 1 else None,
                **common,
            )
        if kind == "address":
            address_kind = value[0] if isinstance(value, tuple) else "ram"
            raw = value[1] if isinstance(value, tuple) else value
            if address_kind == "stack":
                name = _c_identifier(str(raw))
                symbol = self.symbols.lookup(name) or self.symbols.anonymous(name)
                return PrintExpr(
                    kind="address_of",
                    args=(
                        PrintExpr(
                            kind="var",
                            symbol=symbol,
                            ctype=symbol.ctype,
                            name=name,
                            bits=bits,
                            op=op,
                            create_index=index,
                        ),
                    ),
                    ctype=self.resolver.pointer(symbol.ctype),
                    **common,
                )
            address = int(raw or 0)
            symbol = self.symbols.global_at(address)
            string_text = _string_at(self.symbols.analyzer, address)
            if string_text is not None:
                return PrintExpr(
                    kind="string",
                    name=string_text,
                    address=PrintAddress("ram", address),
                    ctype=self.resolver.pointer(self.resolver.from_spec("char")),
                    **common,
                )
            if symbol is not None:
                return PrintExpr(
                    kind="address_of",
                    args=(
                        PrintExpr(
                            kind="var",
                            symbol=symbol,
                            ctype=symbol.ctype,
                            name=symbol.name,
                            bits=self.bits,
                            op=op,
                            create_index=index,
                        ),
                    ),
                    address=PrintAddress("ram", address),
                    ctype=self.resolver.pointer(symbol.ctype),
                    **common,
                )
            target = self.symbols.analyzer.functions.get(address) if self.symbols.analyzer else None
            if target is not None:
                return PrintExpr(
                    kind="function_address",
                    name=_c_identifier(target.name),
                    address=PrintAddress("ram", address),
                    ctype=self.resolver.pointer(self.resolver.from_spec("void")),
                    **common,
                )
            return PrintExpr(
                kind="const",
                int_value=address,
                address=PrintAddress("ram", address),
                ctype=self.resolver.integer(self.bits, False),
                **common,
            )
        if kind == "stack_pointer":
            delta = int(value or 0)
            base_name = "in_rsp" if self.bits == 64 else "in_esp"
            base = self.symbols.lookup(base_name) or self.symbols.anonymous(
                base_name, self.bits
            )
            base_node = PrintExpr(
                kind="var",
                symbol=base,
                ctype=base.ctype,
                name=base_name,
                bits=self.bits,
                op=op,
                create_index=index,
            )
            if delta == 0:
                return base_node
            return PrintExpr(
                kind="op",
                args=(
                    base_node,
                    PrintExpr(
                        kind="const",
                        int_value=abs(delta),
                        bits=self.bits,
                        ctype=self.resolver.integer(self.bits, False),
                        op=op,
                        create_index=index,
                    ),
                ),
                name="add" if delta > 0 else "sub",
                ctype=self.resolver.integer(self.bits, False),
                **common,
            )
        if kind == "extract":
            converted = tuple(self.convert(arg, op) for arg in args)
            return PrintExpr(
                kind="extract",
                args=converted,
                ctype=self.resolver.integer(bits or self.bits, False),
                **common,
            )
        if kind == "segment_address":
            converted = tuple(self.convert(arg, op) for arg in args)
            return PrintExpr(
                kind="segment",
                args=converted,
                name=str(value),
                ctype=self.resolver.integer(bits or self.bits, False),
                **common,
            )
        if kind == "phi":
            converted = tuple(self.convert(arg, op) for arg in args)
            return PrintExpr(
                kind="phi",
                args=converted,
                ctype=converted[0].ctype if converted else None,
                **common,
            )
        # Everything else (x87, MMX/SSE composites, builtins, aggregates, ...)
        # keeps the backend's own C rendering as an opaque token.
        return PrintExpr(
            kind="opaque",
            text=self.opaque_text(expr),
            ctype=self.resolver.integer(bits or self.bits, False),
            **common,
        )


def _is_boolean_op(operator: str) -> bool:
    return operator in {
        "eq",
        "ne",
        "ult",
        "ule",
        "ugt",
        "uge",
        "slt",
        "sle",
        "sgt",
        "sge",
        "land",
        "lor",
    }


def _string_at(analyzer: Any, address: int) -> Optional[str]:
    strings = getattr(analyzer, "strings", ()) or ()
    for entry in strings:
        if entry.address == address:
            return entry.text
    return None


# ---------------------------------------------------------------------------
# Statement conversion
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"^(?P<label>[A-Za-z_][A-Za-z0-9_]*):$")
_COMMENT_RE = re.compile(r"^/\*(?P<text>.*)\*/$")
_RETURN_RE = re.compile(r"^return(?: (?P<value>.+?))?;$")
_IF_GOTO_RE = re.compile(
    r"^if \((?P<cond>.+)\) goto (?P<label>[A-Za-z_][A-Za-z0-9_]*);$"
)
_GOTO_RE = re.compile(r"^goto (?P<label>.+);$")
_PHI_RE = re.compile(r"^(?P<dst>[A-Za-z_][A-Za-z0-9_]*) = phi\((?P<args>.*)\);$")
_DECL_RE = re.compile(
    r"^(?P<type>[A-Za-z_][A-Za-z0-9_]*(?:\s*\*)*(?:\s+[A-Za-z_][A-Za-z0-9_]*)?)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*) = (?P<rhs>.+);$"
)
_ASSIGN_RE = re.compile(
    r"^(?P<dst>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*"
    r"|->[A-Za-z_][A-Za-z0-9_]*|\[[^\]]*\])*) = (?P<rhs>.+);$"
)


def split_arguments(text: str) -> list[str]:
    """Split a comma separated argument list, honouring nesting."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


class DocumentBuilder:
    """Build a :class:`PrintableFunction` from backend artifacts."""

    def __init__(self, artifacts: Any) -> None:
        self.artifacts = artifacts
        self.analyzer = artifacts.analyzer
        self.bits = int(artifacts.bits)
        database = getattr(self.analyzer, "type_database", None) or TypeDatabase(
            self.bits, getattr(getattr(self.analyzer, "image", None), "format", "")
        )
        self.resolver = TypeResolver(database, self.bits)
        self.symbols = SymbolTable(
            self.analyzer, self.resolver, self.bits, artifacts.function
        )
        self.converter = ExpressionConverter(
            self.resolver, self.symbols, self.bits, artifacts.render_catalog
        )
        self._op_time = 0
        self._statements: dict[int, list[PrintStatement]] = {}
        self._terminators: dict[int, PrintTerminator] = {}
        self._pending_calls: dict[int, list[Any]] = {
            address: list(records) for address, records in artifacts.calls.items()
        }

    # -- helpers ---------------------------------------------------------
    def _next_op(self, address: Optional[int], kind: str = "") -> PrintOp:
        self._op_time += 1
        return PrintOp(
            op_time=self._op_time,
            address=PrintAddress("ram", address) if address is not None else None,
            kind=kind,
        )

    def _expr(self, node: Any, op: PrintOp) -> Optional[PrintExpr]:
        converted = self.converter.convert(node, op)
        if converted is not None:
            converted.op = op
        return converted

    def _expr_from_text(self, text: str, op: PrintOp) -> PrintExpr:
        node = self.artifacts.render_catalog.get(text)
        if node is not None:
            converted = self.converter.convert(node, op)
            if converted is not None:
                converted.op = op
                return converted
        return PrintExpr(
            kind="opaque",
            text=text,
            op=op,
            ctype=self.resolver.integer(self.bits, False),
        )

    # -- prototype -------------------------------------------------------
    def build_prototype(self) -> PrintPrototype:
        signature: Optional[FunctionSignature] = self.artifacts.signature
        parameters: list[PrintParameter] = []
        if signature is not None:
            for parameter in signature.parameters:
                ctype = self.resolver.from_spec(
                    parameter.data_type
                    if parameter.data_type is not None
                    else parameter.type_name
                )
                symbol = self.symbols.declare_parameter(parameter, ctype)
                parameters.append(
                    PrintParameter(
                        symbol=symbol,
                        ctype=ctype,
                        locations=tuple(parameter.locations),
                        implicit=bool(parameter.implicit),
                    )
                )
            return_type: Any = (
                signature.return_data_type
                if signature.return_data_type is not None
                else signature.return_type
            )
            output = self.resolver.from_spec(return_type)
            variadic = bool(signature.variadic)
            model_name = signature.calling_convention or ""
            hidden: Optional[PrintParameter] = None
            if signature.hidden_return_location is not None:
                pointer = self.resolver.pointer(output)
                hidden_symbol = self.symbols.declare_parameter(
                    FunctionParameter("__hidden_return", f"{output.name} *"),
                    pointer,
                    implicit=True,
                )
                hidden = PrintParameter(
                    symbol=hidden_symbol,
                    ctype=pointer,
                    locations=(signature.hidden_return_location,),
                    implicit=True,
                )
        else:
            output = self.resolver.from_spec(self.artifacts.return_type or "void")
            variadic = False
            model_name = ""
            hidden = None
            for declaration in self.artifacts.parameters:
                ctype = self.resolver.from_spec(
                    declaration.data_type
                    if declaration.data_type is not None
                    else declaration.type_name
                )
                symbol = self.symbols.declare_parameter(declaration, ctype)
                parameters.append(
                    PrintParameter(
                        symbol=symbol,
                        ctype=ctype,
                        locations=tuple(declaration.locations),
                        implicit=bool(declaration.implicit),
                    )
                )
        convention = self.artifacts.calling_convention
        print_model = bool(getattr(convention, "print_name", False))
        return PrintPrototype(
            output_type=output,
            parameters=tuple(parameters),
            variadic=variadic,
            model_name=getattr(convention, "name", model_name) or model_name,
            print_model_in_decl=print_model,
            model_unknown=False,
            hidden_return=hidden,
        )

    def build_locals(self) -> tuple[PrintSymbol, ...]:
        locals_: list[PrintSymbol] = []
        for name, type_name in self.artifacts.local_declarations:
            locals_.append(self.symbols.declare_local(name, type_name))
        return tuple(locals_)

    # -- statements ------------------------------------------------------
    def build_statements(self) -> None:
        for start, records in self.artifacts.statements_by_block.items():
            converted: list[PrintStatement] = []
            for record in records:
                converted.append(self._statement(record))
            self._statements[start] = converted

    def _statement(self, record: Any) -> PrintStatement:
        text = record.text
        address = record.address
        op = self._next_op(address, "statement")

        label = _LABEL_RE.match(text)
        if label is not None:
            return PrintStatement(
                kind="label", op=op, text=text, label=label.group("label")
            )
        comment = _COMMENT_RE.match(text)
        if comment is not None and "=" not in text:
            return PrintStatement(
                kind="comment", op=op, text=text, comment=comment.group("text").strip()
            )

        call = self._match_call(text, address)
        if call is not None:
            return call

        returned = _RETURN_RE.match(text)
        if returned is not None:
            value = returned.group("value")
            expression = (
                self._expr_from_text(value, op) if value is not None else None
            )
            return PrintStatement(
                kind="return", op=op, text=text, expression=expression
            )

        if_goto = _IF_GOTO_RE.match(text)
        if if_goto is not None:
            return PrintStatement(
                kind="if_goto",
                op=op,
                text=text,
                expression=self._expr_from_text(if_goto.group("cond"), op),
                label=if_goto.group("label"),
            )

        goto = _GOTO_RE.match(text)
        if goto is not None:
            return PrintStatement(
                kind="goto", op=op, text=text, label=goto.group("label").strip()
            )

        phi = _PHI_RE.match(text)
        if phi is not None:
            name = phi.group("dst")
            symbol = self.symbols.lookup(name) or self.symbols.anonymous(name)
            values = tuple(
                self._expr_from_text(part, op)
                for part in split_arguments(phi.group("args"))
            )
            return PrintStatement(
                kind="phi",
                op=op,
                text=text,
                out_symbol=symbol,
                out_name=name,
                values=values,
            )

        declaration = _DECL_RE.match(text)
        if declaration is not None:
            type_name = declaration.group("type").strip()
            name = declaration.group("name")
            ctype = self.resolver.from_spec(type_name)
            self.symbols.declare_local(name, ctype.name)
            return PrintStatement(
                kind="decl",
                op=op,
                text=text,
                decl_type=ctype,
                decl_name=name,
                expression=self._expr_from_text(declaration.group("rhs"), op),
            )

        assignment = _ASSIGN_RE.match(text)
        if assignment is not None:
            name = assignment.group("dst")
            symbol = self.symbols.lookup(name) or self.symbols.anonymous(name)
            return PrintStatement(
                kind="assign",
                op=op,
                text=text,
                out_symbol=symbol,
                out_name=name,
                expression=self._expr_from_text(assignment.group("rhs"), op),
            )

        return PrintStatement(kind="raw", op=op, text=text)

    def _match_call(self, text: str, address: Optional[int]) -> Optional[PrintStatement]:
        if address is None:
            return None
        pending = self._pending_calls.get(address)
        if not pending:
            return None
        record = pending[0]
        result_name = (
            str(record.result.value) if record.result is not None else None
        )
        assigned = result_name is not None and text.startswith(f"{result_name} = ")
        bare = text.startswith(f"{record.target_text}(")
        if not (assigned or bare):
            return None
        pending.pop(0)
        op = self._next_op(address, "call")
        arguments = tuple(
            converted
            for converted in (
                self.converter.convert(argument, op) for argument in record.arguments
            )
            if converted is not None
        )
        result_symbol = None
        result_expr = None
        if assigned and result_name is not None:
            result_symbol = self.symbols.lookup(result_name) or self.symbols.anonymous(
                result_name, self.bits
            )
            result_expr = PrintExpr(
                kind="var",
                symbol=result_symbol,
                ctype=result_symbol.ctype,
                name=result_name,
                bits=self.bits,
                op=op,
            )
        target_expr = (
            self.converter.convert(record.target_expr, op)
            if record.indirect and record.target_expr is not None
            else None
        )
        call = PrintCall(
            op=op,
            target_name=record.target_text,
            target_address=record.target_address,
            arguments=arguments,
            result=result_expr,
            result_symbol=result_symbol,
            no_return=bool(record.no_return),
            indirect=bool(record.indirect),
            target_expr=target_expr,
        )
        return PrintStatement(kind="call", op=op, text=text, call=call)

    # -- terminators -----------------------------------------------------
    def build_terminators(self) -> None:
        for start, record in self.artifacts.terminators.items():
            op = self._next_op(start, "terminator")
            self._terminators[start] = PrintTerminator(
                kind=record.kind,
                op=op,
                condition=(
                    self._expr(record.condition, op)
                    if record.condition is not None
                    else None
                ),
                true_target=record.true_target,
                fallthrough=record.fallthrough,
                targets=tuple(record.targets),
                switch_value=(
                    self._expr(record.switch_value, op)
                    if record.switch_value is not None
                    else None
                ),
                switch_cases=tuple(record.switch_cases),
                indirect=(
                    self._expr(record.indirect, op)
                    if record.indirect is not None
                    else None
                ),
                tail_call=record.tail_call,
                tail_call_address=record.tail_call_address,
                tail_call_no_return=record.tail_call_no_return,
                text=tuple(record.text),
            )

    # -- comments and labels --------------------------------------------
    def build_comments(self) -> tuple[dict[int, tuple[Comment, ...]], dict[int, str]]:
        comments: dict[int, tuple[Comment, ...]] = {}
        state = getattr(self.analyzer, "project_state", None)
        raw_comments = getattr(state, "comments", {}) or {}
        for address in sorted(raw_comments):
            comments[address] = (
                Comment(
                    type=CommentType.user2.value,
                    address=PrintAddress("ram", address),
                    text=str(raw_comments[address]),
                ),
            )
        labels = {
            int(address): str(name)
            for address, name in (getattr(state, "labels", {}) or {}).items()
        }
        return comments, labels

    def header_comments(self) -> tuple[Comment, ...]:
        address = PrintAddress("ram", self.artifacts.function.address)
        notes = [
            "Experimental semantic pseudocode: symbolic propagation, structured branches, and CFG goto fallback.",
            "Types, prototypes, and unsupported instruction effects remain conservative guesses;",
            "this is not a byte-for-byte or feature-complete replacement for Ghidra.",
        ]
        return (
            Comment(
                type=CommentType.warningheader.value,
                address=address,
                text="\n".join(" " + note if index else note for index, note in enumerate(notes)),
            ),
        )

    # -- document --------------------------------------------------------
    def build(self) -> PrintableFunction:
        prototype = self.build_prototype()
        locals_ = self.build_locals()
        self.build_statements()
        self.build_terminators()
        comments, labels = self.build_comments()
        function = self.artifacts.function
        return PrintableFunction(
            function=function,
            name=_c_identifier(function.name),
            address=PrintAddress("ram", function.address),
            prototype=prototype,
            scope=self.symbols.local_scope,
            symbols=self.symbols.symbols,
            local_symbols=locals_,
            statements_by_block={
                start: tuple(records) for start, records in self._statements.items()
            },
            terminators=dict(self._terminators),
            structure=self.artifacts.structure,
            blocks=tuple(self.artifacts.blocks),
            comments=comments,
            labels=labels,
            header_comments=self.header_comments(),
            artifacts=self.artifacts,
            resolver=self.resolver,
            symbol_table=self.symbols,
            is_thunk=bool(self.artifacts.is_thunk),
            text=self.artifacts.text,
        )


def build_printable_function(artifacts: Any) -> PrintableFunction:
    """Turn :class:`~minighidra.decompiler.DecompileArtifacts` into a document."""
    return DocumentBuilder(artifacts).build()
