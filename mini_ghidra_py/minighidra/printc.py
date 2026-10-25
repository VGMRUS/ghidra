"""Port of Ghidra's c-language back-end (``printc.hh``/``printc.cc``).

:class:`PrintC` reproduces the C printer of the Ghidra decompiler on top of the
:class:`~minighidra.printlanguage.PrintLanguage` RPN machinery:

* the operator token table (precedences, associativity, spacing, line-break
  bumps and the negation pairs) is taken verbatim from ``printc.cc``;
* ``docFunction`` emits a *complete* function: header comments, prototype
  (return type, calling convention, name, parameters), local variable
  declarations and the whole body, all through the original ``Emit`` markup
  (``<function>``, ``<funcproto>``, ``<return_type>``, ``<vardecl>``,
  ``<block>``, ``<statement>``, ``<op>``, ``<variable>``, ``<datatype>``, ...);
* control-flow structures (``if``/``else``, ``while``, ``do/while``, infinite
  loops, ``switch``, ``goto``/labels, ``break``) follow ``emitBlock*``,
  including the ``PendingBrace`` trick that produces ``else if`` syntax;
* constants follow ``push_integer``/``push_float``/``pushCharConstant``/
  ``pushEnumConstant`` with Ghidra's ``mostNaturalBase`` heuristics, display
  formats and character escaping rules;
* data-type declarations follow ``pushTypeStart``/``pushTypeEnd`` so pointers,
  arrays and function pointers get proper C declarator syntax.

The one structural difference with Ghidra is deliberate: the decompiler backend
of MiniGhidraPy still runs on its expression AST, so the ``op*`` methods receive
normalized AST nodes (:class:`~minighidra.printdoc.PrintExpr`) instead of
``PcodeOp`` objects, and the block structure comes from
:mod:`minighidra.block`/:mod:`minighidra.collapse` instead of ``Funcdata``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional

from .block import (
    BlockBasic,
    BlockCondition,
    BlockDoWhile,
    BlockGraph,
    BlockGoto,
    BlockIf,
    BlockIfElse,
    BlockIfGoto,
    BlockIfNoExit,
    BlockInfLoop,
    BlockList,
    BlockMultiGoto,
    BlockSwitch,
    BlockWhileDo,
    FlowBlock,
)
from .emit import BraceStyle, Emit, EmitError, PendPrint, SyntaxHighlight
from .printdoc import (
    DisplayFormat,
    PrintableFunction,
    PrintAddress,
    PrintCall,
    PrintExpr,
    PrintOp,
    PrintParameter,
    PrintPrototype,
    PrintStatement,
    PrintSymbol,
    PrintTerminator,
    PrintType,
    SymbolCategory,
    TypeResolver,
    get_symbol_highlight,
    METATYPE_ARRAY,
    METATYPE_BOOL,
    METATYPE_CODE,
    METATYPE_ENUM,
    METATYPE_FLOAT,
    METATYPE_INT,
    METATYPE_PTR,
    METATYPE_STRUCT,
    METATYPE_UINT,
    METATYPE_UNION,
    METATYPE_UNKNOWN,
    METATYPE_VOID,
)
from .printlanguage import (
    Atom,
    Comment,
    CommentType,
    Modifier,
    NamespaceStrategy,
    OpToken,
    OpTokenType,
    PrintLanguage,
    PrintLanguageCapability,
    ReversePolish,
    TagType,
)

# Ghidra's FlowBlock goto types (block.hh).
F_GOTO_GOTO = 1
F_BREAK_GOTO = 2
F_CONTINUE_GOTO = 4


class CastStrategyC:
    """Minimal ``CastStrategy``/``CastStrategyC`` equivalent.

    Ghidra decides whether an explicit cast is needed by comparing propagated
    high-level types.  MiniGhidraPy's AST already carries the width/signedness
    of every operation, so the strategy answers from the normalized node types
    and, for extension operators, keeps the cast the backend materialises.
    """

    def __init__(self, resolver: Optional[TypeResolver] = None) -> None:
        self.resolver = resolver

    def set_type_factory(self, resolver: TypeResolver) -> None:
        self.resolver = resolver

    @staticmethod
    def is_zext_cast(outtype: Optional[PrintType], intype: Optional[PrintType]) -> bool:
        if outtype is None or intype is None:
            return False
        return outtype.metatype in (METATYPE_UINT, METATYPE_INT) and outtype.size > intype.size

    @staticmethod
    def is_sext_cast(outtype: Optional[PrintType], intype: Optional[PrintType]) -> bool:
        if outtype is None or intype is None:
            return False
        return outtype.metatype in (METATYPE_UINT, METATYPE_INT) and outtype.size > intype.size

    @staticmethod
    def is_subpiece_cast(
        outtype: Optional[PrintType], intype: Optional[PrintType], offset: int
    ) -> bool:
        if outtype is None or intype is None or offset != 0:
            return False
        return outtype.size < intype.size

    @staticmethod
    def is_extension_cast_implied(node: Any, read_op: Any) -> bool:
        # The AST backend materialises every extension as an explicit cast, so
        # there is no implied extension to hide.
        return False

    @staticmethod
    def cares_about_char_representation(vn: Any, op: Any) -> bool:
        return False


class CommentSorter:
    """Container/organizer for comments in the current function."""

    def __init__(self) -> None:
        self.doc: Optional[PrintableFunction] = None
        self.mask = 0
        self._queue: list[Comment] = []
        self._pending: list[Comment] = []
        self._headers: tuple[Comment, ...] = ()
        self._unplaced: tuple[Comment, ...] = ()

    def setup_function_list(
        self,
        mask: int,
        doc: PrintableFunction,
        comments: dict[int, tuple[Comment, ...]],
        option_unplaced: bool,
    ) -> None:
        self.doc = doc
        self.mask = mask
        self._headers = tuple(doc.header_comments)
        self._unplaced = () if not option_unplaced else ()
        self._pending = []
        self._queue = []

    def setup_block_list(self, bb: Any) -> None:
        doc = self.doc
        if doc is None:
            return
        start = getattr(bb, "start_addr", None)
        if start is None:
            return
        end = start
        for block in doc.blocks:
            if block.start == start:
                end = block.end
                break
        for address in sorted(doc.comments):
            if start <= address <= end:
                self._pending.extend(doc.comments[address])

    def setup_op_list(self, inst: Any) -> None:
        address = getattr(getattr(inst, "op", None), "address", None)
        if address is None:
            self._queue = list(self._pending)
            self._pending = []
            return
        offset = address.offset
        matching = [item for item in self._pending if item.address.offset == offset]
        rest = [item for item in self._pending if item.address.offset != offset]
        self._queue = matching
        self._pending = rest

    def setup_header(self, kind: int) -> None:
        if kind == 0:  # header_basic
            self._queue = list(self._headers)
        else:
            self._queue = list(self._unplaced)

    def has_next(self) -> bool:
        return bool(self._queue)

    def get_next(self) -> Comment:
        return self._queue.pop(0)

    def remaining(self) -> list[Comment]:
        return list(self._pending)


class PendingBrace(PendPrint):
    """Print commands for an open brace that can still be cancelled."""

    def __init__(self, style: BraceStyle) -> None:
        self.indent_id = -1
        self.style = style

    def get_indent_id(self) -> int:
        return self.indent_id

    def callback(self, emit: Emit) -> None:
        self.indent_id = emit.open_brace_indent(PrintC.OPEN_CURLY, self.style)


class PrintC(PrintLanguage):
    """The c-language token emitter."""

    # Operator tokens for expressions
    #                    token #in prec assoc   optype       space bump
    hidden = OpToken("", "", 1, 70, False, OpTokenType.hiddenfunction, 0, 0)
    scope = OpToken("::", "", 2, 70, True, OpTokenType.binary, 0, 0)
    object_member = OpToken(".", "", 2, 66, True, OpTokenType.binary, 0, 0)
    pointer_member = OpToken("->", "", 2, 66, True, OpTokenType.binary, 0, 0)
    subscript = OpToken("[", "]", 2, 66, False, OpTokenType.postsurround, 0, 0)
    function_call = OpToken("(", ")", 2, 66, False, OpTokenType.postsurround, 0, 10)
    bitwise_not = OpToken("~", "", 1, 62, False, OpTokenType.unary_prefix, 0, 0)
    boolean_not = OpToken("!", "", 1, 62, False, OpTokenType.unary_prefix, 0, 0)
    unary_minus = OpToken("-", "", 1, 62, False, OpTokenType.unary_prefix, 0, 0)
    unary_plus = OpToken("+", "", 1, 62, False, OpTokenType.unary_prefix, 0, 0)
    addressof = OpToken("&", "", 1, 62, False, OpTokenType.unary_prefix, 0, 0)
    dereference = OpToken("*", "", 1, 62, False, OpTokenType.unary_prefix, 0, 0)
    typecast = OpToken("(", ")", 2, 62, False, OpTokenType.presurround, 0, 0)
    multiply = OpToken("*", "", 2, 54, True, OpTokenType.binary, 1, 0)
    divide = OpToken("/", "", 2, 54, False, OpTokenType.binary, 1, 0)
    modulo = OpToken("%", "", 2, 54, False, OpTokenType.binary, 1, 0)
    binary_plus = OpToken("+", "", 2, 50, True, OpTokenType.binary, 1, 0)
    binary_minus = OpToken("-", "", 2, 50, False, OpTokenType.binary, 1, 0)
    shift_left = OpToken("<<", "", 2, 46, False, OpTokenType.binary, 1, 0)
    shift_right = OpToken(">>", "", 2, 46, False, OpTokenType.binary, 1, 0)
    shift_sright = OpToken(">>", "", 2, 46, False, OpTokenType.binary, 1, 0)
    less_than = OpToken("<", "", 2, 42, False, OpTokenType.binary, 1, 0)
    less_equal = OpToken("<=", "", 2, 42, False, OpTokenType.binary, 1, 0)
    greater_than = OpToken(">", "", 2, 42, False, OpTokenType.binary, 1, 0)
    greater_equal = OpToken(">=", "", 2, 42, False, OpTokenType.binary, 1, 0)
    equal = OpToken("==", "", 2, 38, False, OpTokenType.binary, 1, 0)
    not_equal = OpToken("!=", "", 2, 38, False, OpTokenType.binary, 1, 0)
    bitwise_and = OpToken("&", "", 2, 34, True, OpTokenType.binary, 1, 0)
    bitwise_xor = OpToken("^", "", 2, 30, True, OpTokenType.binary, 1, 0)
    bitwise_or = OpToken("|", "", 2, 26, True, OpTokenType.binary, 1, 0)
    boolean_and = OpToken("&&", "", 2, 22, False, OpTokenType.binary, 1, 0)
    boolean_xor = OpToken("^^", "", 2, 20, False, OpTokenType.binary, 1, 0)
    boolean_or = OpToken("||", "", 2, 18, False, OpTokenType.binary, 1, 0)
    assignment = OpToken("=", "", 2, 14, False, OpTokenType.binary, 1, 5)
    comma = OpToken(",", "", 2, 2, True, OpTokenType.binary, 0, 0)
    new_op = OpToken("", "", 2, 62, False, OpTokenType.space, 1, 0)

    # Inplace assignment operators
    multequal = OpToken("*=", "", 2, 14, False, OpTokenType.binary, 1, 5)
    divequal = OpToken("/=", "", 2, 14, False, OpTokenType.binary, 1, 5)
    remequal = OpToken("%=", "", 2, 14, False, OpTokenType.binary, 1, 5)
    plusequal = OpToken("+=", "", 2, 14, False, OpTokenType.binary, 1, 5)
    minusequal = OpToken("-=", "", 2, 14, False, OpTokenType.binary, 1, 5)
    leftequal = OpToken("<<=", "", 2, 14, False, OpTokenType.binary, 1, 5)
    rightequal = OpToken(">>=", "", 2, 14, False, OpTokenType.binary, 1, 5)
    andequal = OpToken("&=", "", 2, 14, False, OpTokenType.binary, 1, 5)
    orequal = OpToken("|=", "", 2, 14, False, OpTokenType.binary, 1, 5)
    xorequal = OpToken("^=", "", 2, 14, False, OpTokenType.binary, 1, 5)

    # Extension: MiniGhidraPy's AST keeps conditional selects, which Ghidra
    # structures away (CondExe) before printing.  Two tokens model C's right
    # associative ``a ? b : c``: the inner "?" binds tighter than the outer ":"
    # so nesting in the false branch needs no parentheses, while a nested
    # ternary in the true branch is parenthesised, as C requires.
    ternary_then = OpToken("?", "", 2, 17, False, OpTokenType.binary, 1, 0)
    ternary_else = OpToken(":", "", 2, 16, True, OpTokenType.binary, 1, 0)

    # Operator tokens for type expressions
    type_expr_space = OpToken("", "", 2, 10, False, OpTokenType.space, 1, 0)
    type_expr_nospace = OpToken("", "", 2, 10, False, OpTokenType.space, 0, 0)
    ptr_expr = OpToken("*", "", 1, 62, False, OpTokenType.unary_prefix, 0, 0)
    array_expr = OpToken("[", "]", 2, 66, False, OpTokenType.postsurround, 1, 0)
    enum_cat = OpToken("|", "", 2, 26, True, OpTokenType.binary, 0, 0)

    EMPTY_STRING = ""
    OPEN_CURLY = "{"
    CLOSE_CURLY = "}"
    SEMICOLON = ";"
    COLON = ":"
    EQUALSIGN = "="
    COMMA = ","
    DOTDOTDOT = "..."
    KEYWORD_VOID = "void"
    KEYWORD_TRUE = "true"
    KEYWORD_FALSE = "false"
    KEYWORD_IF = "if"
    KEYWORD_ELSE = "else"
    KEYWORD_DO = "do"
    KEYWORD_WHILE = "while"
    KEYWORD_FOR = "for"
    KEYWORD_GOTO = "goto"
    KEYWORD_BREAK = "break"
    KEYWORD_CONTINUE = "continue"
    KEYWORD_CASE = "case"
    KEYWORD_SWITCH = "switch"
    KEYWORD_DEFAULT = "default"
    KEYWORD_RETURN = "return"
    KEYWORD_NEW = "new"
    typePointerRelToken = "ADJ"

    # AST operator name -> C token, mirroring the p-code op* dispatch.
    _BINARY_TOKENS = {
        "add": "binary_plus",
        "sub": "binary_minus",
        "mul": "multiply",
        "udiv": "divide",
        "sdiv": "divide",
        "urem": "modulo",
        "srem": "modulo",
        "and": "bitwise_and",
        "or": "bitwise_or",
        "xor": "bitwise_xor",
        "shl": "shift_left",
        "lshr": "shift_right",
        "ashr": "shift_sright",
        "eq": "equal",
        "ne": "not_equal",
        "ult": "less_than",
        "ule": "less_equal",
        "ugt": "greater_than",
        "uge": "greater_equal",
        "slt": "less_than",
        "sle": "less_equal",
        "sgt": "greater_than",
        "sge": "greater_equal",
        "land": "boolean_and",
        "lor": "boolean_or",
    }
    _NEGATABLE = {
        "less_than": "greater_equal",
        "less_equal": "greater_than",
        "greater_than": "less_equal",
        "greater_equal": "less_than",
        "equal": "not_equal",
        "not_equal": "equal",
    }
    _UNSIGNED_COMPARISONS = frozenset({"ult", "ule", "ugt", "uge"})
    _SIGNED_COMPARISONS = frozenset({"slt", "sle", "sgt", "sge"})
    _UNARY_TOKENS = {
        "neg": "unary_minus",
        "bitnot": "bitwise_not",
        "boolnot": "boolean_not",
    }

    def __init__(self, arch: Any = None, nm: str = "c-language") -> None:
        super().__init__(arch, nm)
        self.null_token = "NULL"
        self.size_suffix = "L"
        self.commsorter = CommentSorter()
        self.doc: Optional[PrintableFunction] = None
        self.resolver: Optional[TypeResolver] = None
        self._visited_blocks: set[int] = set()
        self._label_targets: set[int] = set()

        # Set the flip tokens
        PrintC.less_than.negate = PrintC.greater_equal
        PrintC.less_equal.negate = PrintC.greater_than
        PrintC.greater_than.negate = PrintC.less_equal
        PrintC.greater_equal.negate = PrintC.less_than
        PrintC.equal.negate = PrintC.not_equal
        PrintC.not_equal.negate = PrintC.equal

        self.option_NULL = False
        self.option_inplace_ops = False
        self.option_convention = True
        self.option_nocasts = False
        self.option_unplaced = False
        self.option_hide_exts = True
        self.option_brace_func = BraceStyle.skip_line
        self.option_brace_ifelse = BraceStyle.same_line
        self.option_brace_loop = BraceStyle.same_line
        self.option_brace_switch = BraceStyle.same_line
        self.cast_strategy = CastStrategyC()
        # "goto" only labels blocks that a printed goto actually references
        # (Ghidra's isUnstructuredTarget); "join" labels every join block;
        # "none" prints no labels at all.
        self.label_mode = "goto"
        self._recording_labels: Optional[set] = None
        self.reset_defaults_print_c()

    # -- options ---------------------------------------------------------
    def set_null_printing(self, val: bool) -> None:
        self.option_NULL = val

    def set_inplace_ops(self, val: bool) -> None:
        self.option_inplace_ops = val

    def set_convention(self, val: bool) -> None:
        self.option_convention = val

    def set_no_cast_printing(self, val: bool) -> None:
        self.option_nocasts = val

    def set_c_style_comments(self) -> None:
        self.set_comment_delimeter("/* ", " */", False)

    def set_c_plus_plus_style_comments(self) -> None:
        self.set_comment_delimeter("// ", "", True)

    def set_display_unplaced(self, val: bool) -> None:
        self.option_unplaced = val

    def set_hide_implied_exts(self, val: bool) -> None:
        self.option_hide_exts = val

    def set_brace_format_function(self, style: BraceStyle) -> None:
        self.option_brace_func = style

    def set_brace_format_if_else(self, style: BraceStyle) -> None:
        self.option_brace_ifelse = style

    def set_brace_format_loop(self, style: BraceStyle) -> None:
        self.option_brace_loop = style

    def set_brace_format_switch(self, style: BraceStyle) -> None:
        self.option_brace_switch = style

    def set_label_mode(self, mode: str) -> None:
        if mode not in ("goto", "join", "none"):
            raise EmitError('Unknown label mode. Use "goto", "join" or "none"')
        self.label_mode = mode

    def reset_defaults_print_c(self) -> None:
        self.option_convention = True
        self.option_hide_exts = True
        self.option_inplace_ops = False
        self.option_nocasts = False
        self.option_NULL = False
        self.option_unplaced = False
        self.option_brace_func = BraceStyle.skip_line
        self.option_brace_ifelse = BraceStyle.same_line
        self.option_brace_loop = BraceStyle.same_line
        self.option_brace_switch = BraceStyle.same_line
        self.set_c_style_comments()

    def reset_defaults(self) -> None:
        super().reset_defaults()
        self.reset_defaults_print_c()

    def initialize_from_architecture(self) -> None:
        resolver = self.resolver
        if resolver is not None:
            self.cast_strategy.set_type_factory(resolver)
        long_size = _type_size(self.arch, "long", 0)
        int_size = _type_size(self.arch, "int", 0)
        if long_size and int_size and long_size == int_size:
            self.size_suffix = "LL"
        elif long_size and int_size:
            self.size_suffix = "L"
        else:
            bits = getattr(getattr(self.arch, "image", None), "bits", 0)
            self.size_suffix = "LL" if bits in (0, 32) else "L"

    def adjust_type_operators(self) -> None:
        PrintC.scope.print1 = "::"
        PrintC.shift_right.print1 = ">>"

    def set_comment_style(self, nm: str) -> None:
        if nm == "c" or (len(nm) >= 2 and nm[0] == "/" and nm[1] == "*"):
            self.set_c_style_comments()
        elif nm == "cplusplus" or (len(nm) >= 2 and nm[0] == "/" and nm[1] == "/"):
            self.set_c_plus_plus_style_comments()
        else:
            raise EmitError('Unknown comment style. Use "c" or "cplusplus"')

    def get_scope_delimiter(self) -> str:
        return PrintC.scope.print1

    # -- document entry points -------------------------------------------
    def attach_document(self, doc: PrintableFunction) -> None:
        """Bind the printable document (``Funcdata`` role) to the printer."""
        self.doc = doc
        self.resolver = doc.resolver
        self.initialize_from_architecture()
        self._visited_blocks = set()
        self._label_targets = _collect_label_targets(doc)

    def doc_function(self, fd: PrintableFunction) -> None:
        """Emit the declaration and body of a complete function."""
        if self.doc is not fd:
            self.attach_document(fd)
        modsave = self.mods
        try:
            self.commsorter.setup_function_list(
                self.instr_comment_type | self.head_comment_type,
                fd,
                fd.comments,
                self.option_unplaced,
            )
            id1 = self.emit.begin_function(fd)
            self.emit_comment_func_header(fd)
            self.emit.tag_line()
            self.emit_function_declaration(fd)
            id = self.emit.open_brace_indent(self.OPEN_CURLY, self.option_brace_func)
            self._emit_function_body(fd)
            self.pop_scope()
            self.emit.close_brace_indent(self.CLOSE_CURLY, id)
            self.emit.tag_line()
            self.emit.end_function(id1)
            self.emit.flush()
            self.mods = modsave
        except EmitError:
            self.clear()
            raise

    def _emit_function_body(self, fd: PrintableFunction) -> None:
        """Emit local declarations plus the whole (structured) body."""
        if self.label_mode == "goto":
            self._label_targets = _probe_goto_targets(self, fd)
        self.emit_local_var_decls(fd)
        if fd.is_thunk:
            self.emit_thunk_body(fd)
        elif self.is_set(Modifier.flat.value):
            self.emit_flat_blocks(fd)
        elif fd.structure is not None:
            self.emit_block_graph(fd.structure)
        else:
            self.emit_flat_blocks(fd)
        self.emit_uncovered_blocks(fd)

    def doc_all_globals(self) -> None:
        doc = self.doc
        if doc is None:
            raise EmitError("docAllGlobals requiere un documento asociado")
        id = self.emit.begin_document()
        for symbol in sorted(doc.symbols.values(), key=lambda item: item.name):
            if symbol.scope.is_global:
                self.emit_var_decl_statement(symbol)
        self.emit.tag_line()
        self.emit.end_document(id)
        self.emit.flush()

    def doc_single_global(self, sym: PrintSymbol) -> None:
        id = self.emit.begin_document()
        self.emit_var_decl_statement(sym)
        self.emit.tag_line()
        self.emit.end_document(id)
        self.emit.flush()

    def doc_type_definitions(self, typegrp: Any) -> None:
        types = getattr(typegrp, "types", None)
        if types is None:
            return
        for ctype in types():
            print_type = self.resolver.from_spec(ctype) if self.resolver else None
            if print_type is None or print_type.core:
                continue
            if print_type.metatype in (METATYPE_STRUCT, METATYPE_ENUM, METATYPE_UNION):
                self.emit_type_definition(print_type)

    # -- declarations ----------------------------------------------------
    def build_type_stack(self, ct: PrintType) -> list[PrintType]:
        """Lay out the components of a data-type declaration, bottom up."""
        typestack: list[PrintType] = []
        seen: set[int] = set()
        while True:
            typestack.append(ct)
            if id(ct) in seen:
                break
            seen.add(id(ct))
            if ct.metatype == METATYPE_PTR and ct.pointer_to is not None:
                ct = ct.pointer_to
            elif ct.metatype == METATYPE_ARRAY and ct.base is not None:
                ct = ct.base
            elif ct.metatype == METATYPE_CODE and ct.base is not None:
                ct = ct.base
            else:
                break
        return typestack

    def push_prototype_inputs(self, proto: PrintPrototype) -> None:
        """Push the comma separated parameter declarations of a prototype."""
        sz = proto.num_params()
        if sz == 0 and not proto.is_dotdotdot():
            self.push_atom(
                Atom(self.KEYWORD_VOID, TagType.syntax, SyntaxHighlight.keyword_color)
            )
        else:
            for _ in range(sz - 1):
                self.push_op(PrintC.comma, None)
            if proto.is_dotdotdot() and sz != 0:
                self.push_op(PrintC.comma, None)
            for param in proto.parameters:
                self.push_type_start(param.ctype, True)
                self.push_atom(Atom(self.EMPTY_STRING, TagType.blanktoken, SyntaxHighlight.no_color))
                self.push_type_end(param.ctype)
            if proto.is_dotdotdot():
                if sz != 0:
                    self.push_atom(
                        Atom(self.DOTDOTDOT, TagType.syntax, SyntaxHighlight.no_color)
                    )
                else:
                    self.push_atom(
                        Atom(self.EMPTY_STRING, TagType.blanktoken, SyntaxHighlight.no_color)
                    )

    def push_symbol_scope(self, symbol: PrintSymbol) -> None:
        """Push the namespace elements needed to resolve a symbol."""
        if self.namespc_strategy == NamespaceStrategy.MINIMAL_NAMESPACES:
            scopedepth = 1 if symbol.scope.is_global and self.curscope is not None else 0
        elif self.namespc_strategy == NamespaceStrategy.ALL_NAMESPACES:
            scopedepth = 0 if symbol.scope is self.curscope else 1
        else:
            scopedepth = 0
        if scopedepth != 0:
            for _ in range(scopedepth):
                self.push_op(PrintC.scope, None)
            self.push_atom(
                Atom(
                    symbol.scope.display_name,
                    TagType.syntax,
                    SyntaxHighlight.global_color,
                )
            )

    def emit_symbol_scope(self, symbol: PrintSymbol) -> None:
        if self.namespc_strategy == NamespaceStrategy.MINIMAL_NAMESPACES:
            scopedepth = 1 if symbol.scope.is_global and self.curscope is not None else 0
        elif self.namespc_strategy == NamespaceStrategy.ALL_NAMESPACES:
            scopedepth = 0 if symbol.scope is self.curscope else 1
        else:
            scopedepth = 0
        if scopedepth != 0:
            self.emit.print(symbol.scope.display_name, SyntaxHighlight.global_color)
            self.emit.print(PrintC.scope.print1, SyntaxHighlight.no_color)

    def push_type_start(self, ct: PrintType, noident: bool) -> None:
        """Push a data-type declaration up to (but not including) the identifier."""
        typestack = self.build_type_stack(ct)
        base = typestack[-1]
        if noident and len(typestack) == 1:
            tok = PrintC.type_expr_nospace
        else:
            tok = PrintC.type_expr_space
        if not base.name:
            nm = self.generic_type_name(base)
            self.push_op(tok, None)
            self.push_atom(Atom(nm, TagType.typetoken, SyntaxHighlight.type_color, datatype=base))
        else:
            self.push_op(tok, None)
            self.push_atom(
                Atom(
                    base.display_name,
                    TagType.typetoken,
                    SyntaxHighlight.type_color,
                    datatype=base,
                )
            )
        for index in range(len(typestack) - 2, -1, -1):
            item = typestack[index]
            if item.metatype == METATYPE_PTR:
                self.push_op(PrintC.ptr_expr, None)
            elif item.metatype == METATYPE_ARRAY:
                self.push_op(PrintC.array_expr, None)
            elif item.metatype == METATYPE_CODE:
                self.push_op(PrintC.function_call, None)
            else:
                self.clear()
                raise EmitError("Bad type expression")

    def push_type_end(self, ct: PrintType) -> None:
        """Push array subscripts and function parameters after the identifier."""
        self.push_mod()
        self.set_mod(Modifier.force_dec.value)
        seen: set[int] = set()
        while True:
            if id(ct) in seen:
                break
            seen.add(id(ct))
            if ct.metatype == METATYPE_PTR and ct.pointer_to is not None:
                ct = ct.pointer_to
            elif ct.metatype == METATYPE_ARRAY and ct.base is not None:
                count = ct.count
                ct = ct.base
                self.push_integer(
                    count, 4, False, TagType.syntax, None, None, 0
                )
            elif ct.metatype == METATYPE_CODE:
                self.push_type_parameters(ct)
                if ct.base is None:
                    break
                ct = ct.base
            else:
                break
        self.pop_mod()

    def push_type_parameters(self, ct: PrintType) -> None:
        """Push the parameter list of a function data-type (no identifiers)."""
        params = tuple(ct.parameters)
        variadic = bool(ct.variadic)
        if not params and not variadic:
            self.push_atom(
                Atom(self.EMPTY_STRING, TagType.blanktoken, SyntaxHighlight.no_color)
            )
            return
        if not params:
            self.push_atom(Atom(self.DOTDOTDOT, TagType.syntax, SyntaxHighlight.no_color))
            return
        for _ in range(len(params) - 1):
            self.push_op(PrintC.comma, None)
        if variadic:
            self.push_op(PrintC.comma, None)
        for param in params:
            self.push_type_start(param, True)
            self.push_atom(
                Atom(self.EMPTY_STRING, TagType.blanktoken, SyntaxHighlight.no_color)
            )
            self.push_type_end(param)
        if variadic:
            self.push_atom(
                Atom(self.DOTDOTDOT, TagType.syntax, SyntaxHighlight.no_color)
            )

    def push_type(self, ct: PrintType) -> None:
        self.push_type_start(ct, True)
        self.push_atom(Atom(self.EMPTY_STRING, TagType.blanktoken, SyntaxHighlight.no_color))
        self.push_type_end(ct)

    def generic_type_name(self, ct: PrintType) -> str:
        if ct.metatype == METATYPE_INT:
            return f"unkint{ct.size}"
        if ct.metatype == METATYPE_UINT:
            return f"unkuint{ct.size}"
        if ct.metatype == METATYPE_UNKNOWN:
            return f"unkbyte{ct.size}"
        if ct.metatype == METATYPE_FLOAT:
            return f"unkfloat{ct.size}"
        return "BADTYPE"

    def generic_function_name(self, addr: PrintAddress) -> str:
        return f"func_{addr.offset:x}"

    # -- constants -------------------------------------------------------
    def push_integer(
        self,
        val: int,
        sz: int,
        sign: bool,
        tag: TagType,
        vn: Any,
        op: Any,
        display_format: int,
    ) -> None:
        """Push an integer constant, choosing hex/dec/oct/char/bin as Ghidra does."""
        print_negsign = False
        force_unsigned_token = False
        force_sized_token = False
        symbol = getattr(vn, "symbol", None) if vn is not None else None
        if symbol is not None:
            if symbol.name_locked and symbol.category == SymbolCategory.equate.value:
                if self.push_equate(val, sz, symbol, vn, op):
                    return
            if symbol.display_format:
                display_format = symbol.display_format
        if sign and display_format != DisplayFormat.force_char.value:
            mask = _calc_mask(sz)
            flip = (val ^ mask) & mask
            print_negsign = flip < val
            if print_negsign:
                val = (flip + 1) & mask
            force_unsigned_token = False

        if display_format != 0:
            pass  # Format is forced by the Symbol or data-type
        elif (self.mods & Modifier.force_hex.value) != 0:
            display_format = DisplayFormat.force_hex.value
        elif val <= 10 or (self.mods & Modifier.force_dec.value):
            display_format = DisplayFormat.force_dec.value
        else:
            display_format = (
                DisplayFormat.force_hex.value
                if PrintLanguage.most_natural_base(val) == 16
                else DisplayFormat.force_dec.value
            )

        text = []
        if print_negsign:
            text.append("-")
        if display_format == DisplayFormat.force_hex.value:
            text.append(f"0x{val:x}")
        elif display_format == DisplayFormat.force_dec.value:
            text.append(f"{val:d}")
        elif display_format == DisplayFormat.force_oct.value:
            text.append(f"0{val:o}")
        elif display_format == DisplayFormat.force_char.value:
            if self.do_emit_wide_char_prefix() and sz > 1:
                text.append("L")
            text.append("'")
            if sz == 1 and val >= 0x80:
                text.append(_char_hex_escape(val))
            else:
                text.append(self._unicode_text(val))
            text.append("'")
        else:  # force_bin
            text.append("0b")
            text.append(_format_binary_text(val))
        if force_unsigned_token:
            text.append("U")
        if force_sized_token:
            text.append(self.size_suffix)
        self.push_atom(
            Atom(
                "".join(text),
                tag,
                SyntaxHighlight.const_color,
                op=op,
                varnode=vn,
                int_value=val,
            )
        )

    def push_float(
        self, val: float, sz: int, tag: TagType, vn: Any, op: Any
    ) -> None:
        """Push a floating-point constant token."""
        if math.isnan(val):
            token = "-NAN" if math.copysign(1.0, val) < 0 else "NAN"
        elif math.isinf(val):
            token = "-INFINITY" if val < 0 else "INFINITY"
        elif (self.mods & Modifier.force_scinote.value) != 0:
            token = f"{val:e}"
        else:
            token = repr(float(val))
            if "." not in token and "e" not in token:
                token += ".0"
        if sz > 8:
            token += "L"
        self.push_atom(
            Atom(token, tag, SyntaxHighlight.const_color, op=op, varnode=vn)
        )

    def _unicode_text(self, onechar: int) -> str:
        import io

        out = io.StringIO()
        self.print_unicode(out, onechar)
        return out.getvalue()

    def print_unicode(self, out, onechar: int) -> None:
        if PrintLanguage.unicode_needs_escape(onechar):
            escapes = {
                0: "\\0",
                7: "\\a",
                8: "\\b",
                9: "\\t",
                10: "\\n",
                11: "\\v",
                12: "\\f",
                13: "\\r",
                92: "\\\\",
                ord('"'): '\\"',
                ord("'"): "\\'",
            }
            escape = escapes.get(onechar)
            if escape is not None:
                out.write(escape)
                return
            out.write(_char_hex_escape(onechar))
            return
        out.write(chr(onechar))

    def do_emit_wide_char_prefix(self) -> bool:
        return True

    def push_bool_constant(
        self, val: int, ct: PrintType, tag: TagType, vn: Any, op: Any
    ) -> None:
        keyword = self.KEYWORD_TRUE if val != 0 else self.KEYWORD_FALSE
        self.push_atom(
            Atom(
                keyword,
                tag,
                SyntaxHighlight.const_color,
                op=op,
                varnode=vn,
                int_value=val,
            )
        )

    def push_char_constant(
        self, val: int, ct: PrintType, tag: TagType, vn: Any, op: Any
    ) -> None:
        display_format = DisplayFormat.none.value
        is_signed = ct.metatype == METATYPE_INT
        symbol = getattr(vn, "symbol", None) if vn is not None else None
        if symbol is not None:
            if symbol.name_locked and symbol.category == SymbolCategory.equate.value:
                if self.push_equate(val, ct.size, symbol, vn, op):
                    return
            display_format = symbol.display_format
        if display_format != 0 and display_format != DisplayFormat.force_char.value:
            if not self.cast_strategy.cares_about_char_representation(vn, op):
                self.push_integer(
                    val, ct.size, is_signed, tag, vn, op, display_format
                )
                return
        if ct.size == 1 and val >= 0x80:
            if display_format not in (
                DisplayFormat.force_hex.value,
                DisplayFormat.force_char.value,
            ):
                self.push_integer(val, 1, is_signed, tag, vn, op, display_format)
                return
            display_format = DisplayFormat.force_hex.value
        text = []
        if self.do_emit_wide_char_prefix() and ct.size > 1:
            text.append("L")
        text.append("'")
        if display_format == DisplayFormat.force_hex.value:
            text.append(_char_hex_escape(val))
        else:
            text.append(self._unicode_text(val))
        text.append("'")
        self.push_atom(
            Atom(
                "".join(text),
                tag,
                SyntaxHighlight.const_color,
                op=op,
                varnode=vn,
                int_value=val,
            )
        )

    def push_enum_constant(
        self, val: int, ct: PrintType, tag: TagType, vn: Any, op: Any
    ) -> None:
        matches = _enum_matches(ct, val)
        if matches:
            for index in range(len(matches) - 1):
                self.push_op(PrintC.enum_cat, op)
            for name in matches:
                self.push_atom(
                    Atom(
                        name,
                        tag,
                        SyntaxHighlight.const_color,
                        op=op,
                        varnode=vn,
                        int_value=val,
                    )
                )
        else:
            self.push_integer(val, ct.size, False, tag, vn, op, ct.display_format)

    def push_ptr_code_constant(
        self, val: int, ct: PrintType, vn: Any, op: Any
    ) -> bool:
        doc = self.doc
        if doc is None:
            return False
        target = doc.artifacts.analyzer.functions.get(val) if doc.artifacts else None
        if target is None:
            return False
        name = target.name or self.generic_function_name(PrintAddress("ram", val))
        self.push_atom(
            Atom(name, TagType.functoken, SyntaxHighlight.funcname_color, op=op)
        )
        return True

    def push_ptr_char_constant(
        self, val: int, ct: PrintType, vn: Any, op: Any
    ) -> bool:
        doc = self.doc
        if doc is None or val == 0:
            return False
        text = _string_at_address(doc, val)
        if text is None:
            return False
        self.push_atom(
            Atom(
                _quote_string(text),
                TagType.vartoken,
                SyntaxHighlight.const_color,
                op=op,
                varnode=vn,
            )
        )
        return True

    def push_constant(
        self,
        val: int,
        ct: Optional[PrintType],
        tag: TagType,
        vn: Any,
        op: Any,
        display_format: int,
    ) -> None:
        """Push a constant, dispatching on its data-type as ``PrintC`` does."""
        resolver = self.resolver
        if ct is None and resolver is not None:
            ct = resolver.integer(4, False)
        metatype = ct.metatype if ct is not None else METATYPE_UNKNOWN
        if metatype in (METATYPE_UINT, METATYPE_INT):
            if ct.char_print:
                self.push_char_constant(val, ct, tag, vn, op)
            elif ct.enum:
                self.push_enum_constant(val, ct, tag, vn, op)
            else:
                self.push_integer(
                    val,
                    ct.size,
                    metatype == METATYPE_INT,
                    tag,
                    vn,
                    op,
                    display_format,
                )
            return
        if metatype == METATYPE_UNKNOWN:
            self.push_integer(val, ct.size, False, tag, vn, op, display_format)
            return
        if metatype == METATYPE_BOOL:
            self.push_bool_constant(val, ct, tag, vn, op)
            return
        if metatype == METATYPE_VOID:
            self.clear()
            raise EmitError("Cannot have a constant of type void")
        if metatype == METATYPE_PTR:
            if self.option_NULL and val == 0:
                self.push_atom(
                    Atom(self.null_token, TagType.vartoken, SyntaxHighlight.var_color, op=op, varnode=vn)
                )
                return
            subtype = ct.pointer_to
            if subtype is not None and subtype.char_print:
                if self.push_ptr_char_constant(val, ct, vn, op):
                    return
            elif subtype is not None and subtype.metatype == METATYPE_CODE:
                if self.push_ptr_code_constant(val, ct, vn, op):
                    return
            if self.push_ptr_code_constant(val, ct, vn, op):
                return
        elif metatype == METATYPE_FLOAT:
            self.push_float(_float_from_bits(val, ct.size), ct.size, tag, vn, op)
            return
        # Default printing
        if not self.option_nocasts:
            self.push_op(PrintC.typecast, op)
            self.push_type(ct)
        self.push_mod()
        if not self.is_set(Modifier.force_dec.value):
            self.set_mod(Modifier.force_hex.value)
        self.push_integer(val, ct.size, False, tag, vn, op, display_format)
        self.pop_mod()

    def push_equate(self, val: int, sz: int, sym: Any, vn: Any, op: Any) -> bool:
        """Equate symbols are not modelled by the AST backend."""
        return False

    # -- symbols ---------------------------------------------------------
    def get_symbol_highlight(self, sym: PrintSymbol) -> SyntaxHighlight:
        return get_symbol_highlight(sym)

    def push_symbol(self, sym: PrintSymbol, vn: Any, op: Any) -> None:
        token_color = self.get_symbol_highlight(sym)
        self.push_symbol_scope(sym)
        self.push_atom(
            Atom(sym.display_name, TagType.vartoken, token_color, op=op, varnode=vn)
        )

    def push_annotation(self, vn: Any, op: Any) -> None:
        address = getattr(vn, "address", None)
        name = address.print_raw() if address is not None else "annotation"
        self.push_atom(
            Atom(name, TagType.vartoken, SyntaxHighlight.special_color, op=op, varnode=vn)
        )

    def push_unnamed_location(self, addr: Any, size: int, vn: Any, op: Any) -> None:
        if addr is None:
            name = getattr(vn, "name", "") or "unknown"
        else:
            name = f"{addr.space}_{addr.print_raw()}"
        self.push_atom(
            Atom(name, TagType.vartoken, SyntaxHighlight.var_color, op=op, varnode=vn)
        )

    def push_partial_symbol(
        self,
        sym: PrintSymbol,
        off: int,
        sz: int,
        vn: Any,
        op: Any,
        slot: int,
        allow_cast: bool,
    ) -> None:
        """Emit member syntax for a partial (field) reference to a symbol.

        Simplified subset of Ghidra's ``pushPartialSymbol``: it walks named
        fields (and array indices) bottom up so parentheses come out right, and
        falls back to the plain symbol when the layout does not resolve.  Union
        resolution and truncation casts are not modelled because the AST backend
        does not produce partial symbol references for them.
        """
        ct = sym.ctype
        entries: list[tuple[OpToken, str, PrintType, int]] = []
        depth = 0
        while (off != 0 or sz != ct.size) and depth < 16:
            depth += 1
            field = _find_field(ct, off)
            if field is None:
                break
            token = (
                PrintC.pointer_member
                if ct.metatype == METATYPE_PTR
                else PrintC.object_member
            )
            entries.append((token, field.name, ct, field.offset))
            ct = field.ctype
            off -= field.offset
            if ct.metatype == METATYPE_ARRAY and ct.base is not None:
                index = off // max(1, ct.base.size)
                off -= index * max(1, ct.base.size)
                entries.append((PrintC.subscript, str(index), ct, 0))
                ct = ct.base
            if sz == ct.size and off == 0:
                break
        for token, name, parent, offset in reversed(entries):
            self.push_op(token, op)
        self.push_atom(
            Atom(sym.display_name, TagType.vartoken, self.get_symbol_highlight(sym), op=op, varnode=vn)
        )
        for token, name, parent, offset in entries:
            if token is PrintC.subscript:
                self.push_atom(
                    Atom(name, TagType.syntax, SyntaxHighlight.no_color, op=op)
                )
            else:
                self.push_atom(
                    Atom(
                        name,
                        TagType.fieldtoken,
                        SyntaxHighlight.no_color,
                        op=op,
                        datatype=parent,
                        offset=offset,
                    )
                )

    def push_mismatch_symbol(
        self, sym: PrintSymbol, off: int, sz: int, vn: Any, op: Any
    ) -> None:
        """Emit a cast when a variable overlaps but is not contained by a symbol."""
        if off != 0:
            self.push_op(PrintC.subscript, None)
        self.push_atom(
            Atom(sym.display_name, TagType.vartoken, self.get_symbol_highlight(sym), op=op, varnode=vn)
        )
        if off != 0:
            self.push_integer(off, 4, False, TagType.syntax, None, None, DisplayFormat.force_dec.value)
        resolver = self.resolver
        if resolver is not None and not self.option_nocasts:
            self.push_op(PrintC.typecast, op)
            self.push_type(resolver.integer(sz * 8, False))

    def push_implied_field(self, vn: Any, op: Any) -> None:
        self.push_node_explicit(vn, op)

    # -- AST driven expression printing ----------------------------------
    def push_ast_node(self, node: PrintExpr, op: Any) -> None:
        """Dispatch an AST node the way ``TypeOp::push`` dispatches a PcodeOp."""
        kind = node.kind
        handler = _AST_DISPATCH.get(kind)
        if handler is None:
            self.push_atom(
                Atom(
                    node.text or f"unknown_expr_{kind}",
                    TagType.syntax,
                    SyntaxHighlight.no_color,
                    op=op,
                )
            )
            return
        handler(self, node, op)

    def node_type_read_facing(self, node: PrintExpr, op: Any) -> Optional[PrintType]:
        return node.ctype

    def op_copy(self, node: PrintExpr, op: Any) -> None:
        self.push_node(node.args[0], node, self.mods)

    def op_binary_token(self, token: OpToken, node: PrintExpr, op: Any) -> None:
        # PrintLanguage.op_binary honours the negatetoken modifier and pushes
        # the inputs in reverse order, exactly like Ghidra's opBinary.
        self.op_binary(token, node)

    def op_unary_token(self, token: OpToken, node: PrintExpr, op: Any) -> None:
        self.push_op(token, node)
        self.push_node(node.args[0], node, self.mods)

    def op_arithmetic(self, node: PrintExpr, op: Any) -> None:
        token = getattr(PrintC, self._BINARY_TOKENS[node.name])
        self.op_binary_token(token, node, op)

    def op_comparison(self, node: PrintExpr, op: Any) -> None:
        """Signed/unsigned comparisons keep the backend's explicit casts."""
        token = getattr(PrintC, self._BINARY_TOKENS[node.name])
        resolver = self.resolver
        args = node.args
        width = max([arg.bits for arg in args] + [node.bits, 32])
        signed = node.name in self._SIGNED_COMPARISONS
        if resolver is None or not signed and node.name not in self._UNSIGNED_COMPARISONS:
            self.op_binary_token(token, node, op)
            return
        ctype = resolver.integer(width, signed)
        if self.is_set(Modifier.negatetoken.value):
            token = token.negate
            self.unset_mod(Modifier.negatetoken.value)
            if token is None:
                raise EmitError("Could not find fliptoken")
        self.push_op(token, node)
        # Implied nodes are queued in reverse order (see PrintLanguage.push_node)
        # so that args[0] is printed first, exactly like Ghidra's opBinary.
        self.push_node(_cast_node(ctype, args[1], node), node, self.mods)
        self.push_node(_cast_node(ctype, args[0], node), node, self.mods)

    def op_unary(self, node: PrintExpr, op: Any) -> None:
        if node.name == "boolnot":
            # BOOL_NEGATE gets its own handler, as in PrintC.
            self.op_bool_negate(node, op)
            return
        token = getattr(PrintC, self._UNARY_TOKENS.get(node.name, "unary_plus"))
        self.op_unary_token(token, node, op)

    def op_bool_negate(self, node: PrintExpr, op: Any) -> None:
        if self.is_set(Modifier.negatetoken.value):
            self.unset_mod(Modifier.negatetoken.value)
            self.push_node(node.args[0], node, self.mods)
        elif self.check_print_negation(node.args[0]):
            self.push_node(
                node.args[0], node, self.mods | Modifier.negatetoken.value
            )
        else:
            self.op_unary_token(PrintC.boolean_not, node, op)

    def check_print_negation(self, node: Optional[PrintExpr]) -> bool:
        if node is None or node.kind != "op":
            return False
        return node.name in self._NEGATABLE_VALUES

    _NEGATABLE_VALUES = frozenset(
        {"ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge", "eq", "ne"}
    )

    def op_func(self, node: PrintExpr, op: Any, name: Optional[str] = None) -> None:
        """Push a functional expression whose name is the operator name."""
        self.push_op(PrintC.function_call, node)
        nm = name if name is not None else node.name
        self.push_atom(Atom(nm, TagType.optoken, SyntaxHighlight.no_color, op=node))
        args = node.args
        if args:
            for _ in range(len(args) - 1):
                self.push_op(PrintC.comma, node)
            for index in range(len(args) - 1, -1, -1):
                self.push_node(args[index], node, self.mods)
        else:
            self.push_atom(
                Atom(self.EMPTY_STRING, TagType.blanktoken, SyntaxHighlight.no_color)
            )

    def op_type_cast(self, node: PrintExpr, op: Any) -> None:
        if not self.option_nocasts and node.ctype is not None:
            self.push_op(PrintC.typecast, node)
            self.push_type(node.ctype)
        self.push_node(node.args[0], node, self.mods)

    def push_type_cast(self, ctype: PrintType, node: Any, op: Any) -> None:
        """Push an explicit cast around an already built expression node."""
        if self.option_nocasts or ctype is None:
            self.push_node(node, op, self.mods)
            return
        self.push_op(PrintC.typecast, op)
        self.push_type(ctype)
        self.push_node(node, op, self.mods)

    def op_hidden_func(self, node: PrintExpr, op: Any) -> None:
        self.push_op(PrintC.hidden, node)
        self.push_node(node.args[0], node, self.mods)

    def op_int_zext(self, node: PrintExpr, op: Any) -> None:
        self._extension_cast(node, op, signed=False)

    def op_int_sext(self, node: PrintExpr, op: Any) -> None:
        self._extension_cast(node, op, signed=True)

    def _extension_cast(self, node: PrintExpr, op: Any, *, signed: bool) -> None:
        resolver = self.resolver
        operand = node.args[0] if node.args else None
        if resolver is None or operand is None:
            self.op_func(node, op, "zext" if not signed else "sext")
            return
        outer = node.ctype or resolver.integer(node.bits, signed)
        inner = resolver.integer(operand.bits or outer.size * 8, signed)
        if self.option_hide_exts and self.cast_strategy.is_extension_cast_implied(
            node, op
        ):
            self.op_hidden_func(node, op)
            return
        if self.option_nocasts:
            self.push_node(operand, node, self.mods)
            return
        self.push_op(PrintC.typecast, node)
        self.push_type(outer)
        self.push_type_cast(inner, operand, node)

    def op_subpiece(self, node: PrintExpr, op: Any) -> None:
        """SUBPIECE/truncation: the backend prints it as an unsigned cast."""
        resolver = self.resolver
        operand = node.args[0] if node.args else None
        if resolver is None or operand is None:
            self.op_func(node, op, "subpiece")
            return
        ctype = node.ctype or resolver.integer(node.bits, False)
        if self.option_nocasts:
            self.push_node(operand, node, self.mods)
            return
        self.push_op(PrintC.typecast, node)
        self.push_type(ctype)
        self.push_node(operand, node, self.mods)

    def op_extract(self, node: PrintExpr, op: Any) -> None:
        """Bit-range extraction: ``(v >> off) & mask`` (offset 0 is a cast)."""
        resolver = self.resolver
        if resolver is None or len(node.args) < 2:
            self.op_func(node, op, "extract")
            return
        value, offset = node.args
        shift = offset.int_value if offset.kind == "const" else None
        ctype = node.ctype or resolver.integer(node.bits, False)
        if shift == 0:
            if self.option_nocasts:
                self.push_node(value, node, self.mods)
                return
            self.push_op(PrintC.typecast, node)
            self.push_type(ctype)
            self.push_node(value, node, self.mods)
            return
        mask = (1 << node.bits) - 1
        shifted = PrintExpr(
            kind="op",
            args=(value, offset),
            name="lshr",
            bits=node.bits,
            ctype=ctype,
            op=op,
        )
        mask_node = PrintExpr(
            kind="const",
            int_value=mask,
            bits=node.bits,
            ctype=ctype,
            op=op,
        )
        self.push_op(PrintC.bitwise_and, node)
        self.push_node(mask_node, node, self.mods)
        self.push_node(shifted, node, self.mods)

    def op_load(self, node: PrintExpr, op: Any) -> None:
        """LOAD: ``*(type *)(ptr)`` (Ghidra's dereference plus a typed pointer)."""
        resolver = self.resolver
        self.push_op(PrintC.dereference, node)
        if resolver is not None and not self.option_nocasts:
            target = node.ctype or resolver.integer(node.bits, False)
            self.push_op(PrintC.typecast, node)
            self.push_type(resolver.pointer(target))
        self.push_node(node.args[0], node, self.mods)

    def op_store(self, node: PrintExpr, op: Any) -> None:
        self.push_op(PrintC.assignment, node)
        self.push_op(PrintC.dereference, node)
        resolver = self.resolver
        if resolver is not None and not self.option_nocasts:
            self.push_op(PrintC.typecast, node)
            self.push_type(resolver.pointer(node.ctype or resolver.integer(node.bits)))
        args = node.args
        self.push_node(args[1], node, self.mods)
        self.push_node(args[0], node, self.mods)

    def op_address_of(self, node: PrintExpr, op: Any) -> None:
        self.push_op(PrintC.addressof, node)
        self.push_node(node.args[0], node, self.mods)

    def op_string(self, node: PrintExpr, op: Any) -> None:
        self.push_atom(
            Atom(
                _quote_string(node.name),
                TagType.vartoken,
                SyntaxHighlight.const_color,
                op=op,
                varnode=node,
            )
        )

    def op_function_address(self, node: PrintExpr, op: Any) -> None:
        self.push_atom(
            Atom(node.name, TagType.functoken, SyntaxHighlight.funcname_color, op=op)
        )

    def op_segment(self, node: PrintExpr, op: Any) -> None:
        """SEGMENTOP: ``seg_base + (offset)`` as the backend renders it."""
        base_name = f"{node.name}_base"
        doc = self.doc
        symbol = None
        if doc is not None:
            symbol = doc.symbols.get(base_name)
            if symbol is None:
                symbol = doc.symbol_table.anonymous(base_name, doc.artifacts.bits)
        base = PrintExpr(
            kind="var",
            symbol=symbol,
            name=base_name,
            ctype=symbol.ctype if symbol is not None else None,
            bits=symbol.ctype.size * 8 if symbol is not None else 32,
            op=op,
        )
        self.push_op(PrintC.binary_plus, node)
        self.push_node(node.args[0], node, self.mods)
        self.push_node(base, node, self.mods)

    def op_multiequal(self, node: PrintExpr, op: Any) -> None:
        """MULTIEQUAL: the AST keeps phi nodes, printed as ``phi(a,b)``."""
        self.op_func(node, op, "phi")

    def op_select(self, node: PrintExpr, op: Any) -> None:
        """Conditional select pushed through the RPN stack as ``a ? b : c``."""
        if len(node.args) != 3:
            self.op_func(node, op, "select")
            return
        condition, yes, no = node.args
        self.push_op(PrintC.ternary_else, node)
        self.push_op(PrintC.ternary_then, node)
        # Implied nodes are queued in reverse order (see PrintLanguage.push_node)
        self.push_node(no, node, self.mods)
        self.push_node(yes, node, self.mods)
        self.push_node(condition, node, self.mods)

    def op_opaque(self, node: PrintExpr, op: Any) -> None:
        self.push_atom(
            Atom(
                node.text or "unknown_expr",
                TagType.syntax,
                SyntaxHighlight.no_color,
                op=op,
            )
        )

    def op_constant(self, node: PrintExpr, op: Any) -> None:
        self.push_constant(
            node.int_value,
            node.ctype,
            TagType.vartoken,
            node,
            op,
            node.display_format,
        )

    def op_float_constant(self, node: PrintExpr, op: Any) -> None:
        value = node.float_value if node.float_value is not None else 0.0
        size = node.ctype.size if node.ctype is not None else max(1, node.bits // 8)
        self.push_float(value, size, TagType.vartoken, node, op)

    def op_var(self, node: PrintExpr, op: Any) -> None:
        self.push_symbol_detail(node, op, True)

    # -- statements ------------------------------------------------------
    def emit_expression(self, stmt: PrintStatement) -> None:
        """Emit one statement's expression through the RPN stack."""
        kind = stmt.kind
        op = stmt.op
        if kind == "assign":
            self.push_op(PrintC.assignment, op)
            self._push_out_symbol(stmt, op)
            if stmt.expression is not None:
                self.push_node(stmt.expression, op, self.mods)
            self.recurse()
            return
        if kind == "call":
            self.emit_call_statement(stmt)
            return
        if kind == "return":
            self.emit.tag_op(self.KEYWORD_RETURN, SyntaxHighlight.keyword_color, op)
            if stmt.expression is not None:
                self.emit.spaces(1)
                self.push_node(stmt.expression, op, self.mods)
                self.recurse()
            return
        if kind == "phi":
            self.push_op(PrintC.assignment, op)
            self._push_out_symbol(stmt, op)
            self.push_op(PrintC.function_call, op)
            self.push_atom(
                Atom("phi", TagType.optoken, SyntaxHighlight.no_color, op=op)
            )
            values = stmt.values
            if values:
                for _ in range(len(values) - 1):
                    self.push_op(PrintC.comma, op)
                for index in range(len(values) - 1, -1, -1):
                    self.push_node(values[index], op, self.mods)
            else:
                self.push_atom(
                    Atom(self.EMPTY_STRING, TagType.blanktoken, SyntaxHighlight.no_color)
                )
            self.recurse()
            return
        if kind == "decl":
            ctype = stmt.decl_type
            if ctype is not None:
                self.push_type_start(ctype, False)
                self.push_atom(
                    Atom(
                        stmt.decl_name,
                        TagType.vartoken,
                        SyntaxHighlight.var_color,
                        op=op,
                    )
                )
                self.push_type_end(ctype)
                self.recurse()
            else:
                self.emit.print(stmt.decl_name, SyntaxHighlight.var_color)
            if stmt.expression is not None:
                self.emit.spaces(1)
                self.emit.tag_op(self.EQUALSIGN, SyntaxHighlight.no_color, op)
                self.emit.spaces(1)
                self.push_node(stmt.expression, op, self.mods)
                self.recurse()
            return
        if kind == "goto":
            self.emit_goto_label(stmt.label, op)
            return
        if kind == "if_goto":
            self.emit.tag_op(self.KEYWORD_IF, SyntaxHighlight.keyword_color, op)
            self.emit.spaces(1)
            id = self.emit.open_paren(self.OPEN_PAREN)
            if stmt.expression is not None:
                self.push_node(stmt.expression, op, self.mods)
                self.recurse()
            self.emit.close_paren(self.CLOSE_PAREN, id)
            self.emit.spaces(1)
            self.emit_goto_label(stmt.label, op)
            return
        if kind == "label":
            self.emit.tag_label(stmt.label, SyntaxHighlight.no_color, None, 0)
            self.emit.print(self.COLON)
            return
        if kind == "comment":
            self.emit_line_comment(
                -1,
                Comment(
                    type=CommentType.user2.value,
                    address=op.address or PrintAddress("ram", 0),
                    text=stmt.comment,
                ),
            )
            return
        # Anything the bridge could not re-associate with the AST keeps its
        # rendered text as plain syntax.
        if stmt.text:
            self.emit.print(stmt.text, SyntaxHighlight.no_color)

    def _push_out_symbol(self, stmt: PrintStatement, op: PrintOp) -> None:
        symbol = stmt.out_symbol
        if symbol is None:
            self.emit.print(stmt.out_name, SyntaxHighlight.var_color)
            return
        node = PrintExpr(
            kind="var",
            symbol=symbol,
            ctype=symbol.ctype,
            name=symbol.name,
            bits=symbol.ctype.size * 8,
            op=op,
        )
        self.push_symbol_detail(node, op, False)

    def emit_call_statement(self, stmt: PrintStatement) -> None:
        call = stmt.call
        if call is None:
            self.emit.print(stmt.text, SyntaxHighlight.no_color)
            return
        op = call.op
        if call.result_symbol is not None:
            self.push_op(PrintC.assignment, op)
            self.push_symbol(call.result_symbol, None, op)
        self.push_op(PrintC.function_call, op)
        self._push_call_target(call, op)
        args = call.arguments
        if args:
            for _ in range(len(args) - 1):
                self.push_op(PrintC.comma, op)
            for index in range(len(args) - 1, -1, -1):
                self.push_node(args[index], op, self.mods)
        else:
            self.push_atom(
                Atom(self.EMPTY_STRING, TagType.blanktoken, SyntaxHighlight.no_color)
            )
        self.recurse()
        if call.no_return:
            self.emit.spaces(1)
            self.emit_comment_text("no retorna", op.address)

    def _push_call_target(self, call: PrintCall, op: PrintOp) -> None:
        if call.indirect and call.target_expr is not None:
            self.push_op(PrintC.dereference, op)
            self.push_node(call.target_expr, op, self.mods)
            return
        name = call.target_name
        if call.target_address is not None:
            doc = self.doc
            target = (
                doc.artifacts.analyzer.functions.get(call.target_address)
                if doc is not None and doc.artifacts is not None
                else None
            )
            if target is not None:
                name = target.name or name
        self.push_atom(
            Atom(name, TagType.functoken, SyntaxHighlight.funcname_color, op=op)
        )

    def emit_comment_text(self, text: str, address: Optional[PrintAddress]) -> None:
        self.emit_line_comment(
            -1,
            Comment(
                type=CommentType.user2.value,
                address=address or PrintAddress("ram", 0),
                text=text,
            ),
        )

    def emit_goto_label(self, label: str, op: PrintOp) -> None:
        self.emit.print(self.KEYWORD_GOTO, SyntaxHighlight.keyword_color)
        self.emit.spaces(1)
        if label.startswith("*"):
            self.emit.print(label, SyntaxHighlight.no_color)
        else:
            self.emit.tag_label(label, SyntaxHighlight.no_color, None, 0)

    def emit_statement(self, stmt: PrintStatement) -> None:
        id = self.emit.begin_statement(stmt.op)
        self.emit_expression(stmt)
        self.emit.end_statement(id)
        if not self.is_set(Modifier.comma_separate.value) and stmt.kind not in (
            "comment",
            "label",
        ):
            self.emit.print(self.SEMICOLON)

    def emit_var_decl(self, sym: PrintSymbol) -> None:
        id = self.emit.begin_var_decl(sym)
        self.push_type_start(sym.ctype, False)
        self.push_symbol(sym, None, None)
        self.push_type_end(sym.ctype)
        self.recurse()
        self.emit.end_var_decl(id)

    def emit_var_decl_statement(self, sym: PrintSymbol) -> None:
        self.emit.tag_line()
        self.emit_var_decl(sym)
        self.emit.print(self.SEMICOLON)

    def emit_local_var_decls(self, fd: PrintableFunction) -> None:
        notempty = False
        for sym in fd.local_symbols:
            notempty = True
            self.emit_var_decl_statement(sym)
        if notempty:
            self.emit.tag_line()

    def emit_prototype_output(self, proto: PrintPrototype, fd: PrintableFunction) -> None:
        id = self.emit.begin_return_type(None)
        self.push_type(proto.output_type)
        self.recurse()
        self.emit.end_return_type(id)

    def emit_prototype_inputs(self, proto: PrintPrototype) -> None:
        visible = [
            param for param in proto.parameters if not _is_hidden_parameter(param)
        ]
        if not visible:
            self.emit.print(self.KEYWORD_VOID, SyntaxHighlight.keyword_color)
        else:
            print_comma = False
            for param in visible:
                if print_comma:
                    self.emit.print(self.COMMA)
                if self.is_set(Modifier.hide_thisparam.value) and param.this_pointer:
                    continue
                print_comma = True
                self.emit_var_decl(param.symbol)
        if proto.is_dotdotdot():
            if visible:
                self.emit.print(self.COMMA)
            self.emit.print(self.DOTDOTDOT)

    def emit_function_declaration(self, fd: PrintableFunction) -> None:
        proto = fd.prototype
        id = self.emit.begin_func_proto()
        self.emit_prototype_output(proto, fd)
        self.emit.spaces(1)
        if self.option_convention and proto.print_model_in_decl and proto.model_name:
            highlight = (
                SyntaxHighlight.error_color
                if proto.model_unknown
                else SyntaxHighlight.keyword_color
            )
            self.emit.print(proto.model_name, highlight)
            self.emit.spaces(1)
        id1 = self.emit.open_group()
        self.emit.tag_func_name(
            fd.display_name, SyntaxHighlight.funcname_color, fd, None
        )
        self.emit.spaces(PrintC.function_call.spacing, PrintC.function_call.bump)
        id2 = self.emit.open_paren(self.OPEN_PAREN)
        self.emit.spaces(0, PrintC.function_call.bump)
        self.push_scope(fd.scope)
        self.emit_prototype_inputs(proto)
        self.emit.close_paren(self.CLOSE_PAREN, id2)
        self.emit.close_group(id1)
        self.emit.end_func_proto(id)

    # -- control flow ----------------------------------------------------
    def emit_block(self, bl: FlowBlock, boolean_flip: bool = False) -> None:
        """``FlowBlock::emit``: dispatch on the structured block type."""
        block_type = getattr(bl, "block_type", "")
        if isinstance(bl, BlockBasic):
            self.emit_block_basic(bl, boolean_flip=boolean_flip)
        elif isinstance(bl, (BlockGraph,)):
            self.emit_block_graph(bl)
        elif isinstance(bl, BlockList):
            self.emit_block_ls(bl)
        elif isinstance(bl, BlockIfGoto):
            self.emit_block_if(bl)
        elif isinstance(bl, BlockIfElse):
            self.emit_block_if(bl)
        elif isinstance(bl, BlockIfNoExit):
            self.emit_block_if(bl)
        elif isinstance(bl, BlockIf):
            self.emit_block_if(bl)
        elif isinstance(bl, BlockWhileDo):
            self.emit_block_while_do(bl)
        elif isinstance(bl, BlockDoWhile):
            self.emit_block_do_while(bl)
        elif isinstance(bl, BlockInfLoop):
            self.emit_block_inf_loop(bl)
        elif isinstance(bl, BlockSwitch):
            self.emit_block_switch(bl)
        elif isinstance(bl, BlockCondition):
            self.emit_block_condition(bl)
        elif isinstance(bl, BlockMultiGoto):
            self.emit_block_multi_goto(bl)
        elif isinstance(bl, BlockGoto):
            self.emit_block_goto(bl)
        else:  # pragma: no cover - defensive
            self.emit_comment_text(
                f"unsupported block structure: {block_type}", None
            )

    def emit_block_graph(self, bl: Any) -> None:
        for item in _sub_blocks(bl):
            id = self.emit.begin_block(item)
            self.emit_block(item)
            self.emit.end_block(id)

    def emit_flat_blocks(self, fd: PrintableFunction) -> None:
        """Emit every basic block in address order (Ghidra's ``flat`` mode)."""
        self.push_mod()
        self.set_mod(Modifier.flat.value)
        for block in fd.blocks:
            basic = _basic_for_address(fd, block.start)
            id = self.emit.begin_block(basic)
            self.emit_block_basic(basic)
            self.emit.end_block(id)
        self.pop_mod()

    def emit_thunk_body(self, fd: PrintableFunction) -> None:
        """A tail thunk forwards to its target; the backend keeps it as a goto."""
        target = fd.function.thunk_target
        self.emit.tag_line()
        id = self.emit.begin_statement(None)
        self.emit.print(self.KEYWORD_GOTO, SyntaxHighlight.keyword_color)
        self.emit.spaces(1)
        name = fd.function.thunk_target_name
        if name:
            self.emit.tag_func_name(name, SyntaxHighlight.funcname_color, None, None)
        elif target is not None:
            self.emit_label_for_address(target)
        else:
            self.emit.print("indirect_target", SyntaxHighlight.no_color)
        self.emit.print(self.SEMICOLON)
        self.emit.end_statement(id)

    def emit_uncovered_blocks(self, fd: PrintableFunction) -> None:
        """Emit blocks the (partial) structurer left out, so nothing is lost."""
        missing = [
            block.start
            for block in fd.blocks
            if block.start not in self._visited_blocks
        ]
        if not missing:
            return
        self.push_mod()
        self.unset_mod(Modifier.no_branch.value | Modifier.only_branch.value)
        for start in missing:
            basic = _basic_for_address(fd, start)
            id = self.emit.begin_block(basic)
            self.emit_block_basic(basic)
            self.emit.end_block(id)
        self.pop_mod()

    def emit_block_basic(self, bb: Any, boolean_flip: bool = False) -> None:
        doc = self.doc
        if doc is None:
            return
        start = getattr(bb, "start_addr", None)
        if start is not None:
            self._visited_blocks.add(start)
        self.commsorter.setup_block_list(bb)
        self.emit_label_statement(bb)
        term = doc.terminators.get(start) if start is not None else None
        if self.is_set(Modifier.only_branch.value):
            if term is not None and term.kind in ("cbranch", "cbranch_indirect"):
                self.emit_condition(term, bb, boolean_flip=boolean_flip)
            return
        separator = False
        for stmt in doc.statements_by_block.get(start, ()):
            if separator:
                if self.is_set(Modifier.comma_separate.value):
                    self.emit.print(self.COMMA)
                    self.emit.spaces(1)
                else:
                    self.emit_comment_group(stmt)
                    self.emit.tag_line()
            elif not self.is_set(Modifier.comma_separate.value):
                self.emit_comment_group(stmt)
                self.emit.tag_line()
            self.emit_statement(stmt)
            separator = True
        if (
            term is not None
            and not self.is_set(Modifier.no_branch.value)
            and _terminator_prints(term)
        ):
            # The branch is one more element of the block: Ghidra separates it
            # with a comma inside conditions and with a line break everywhere
            # else (statements always start on their own line).
            if self.is_set(Modifier.comma_separate.value):
                if separator:
                    self.emit.print(self.COMMA)
                    self.emit.spaces(1)
            else:
                self.emit.tag_line()
            self.emit_terminator(term, bb, boolean_flip=boolean_flip)
            separator = True
        self.emit_comment_group(None)

    def emit_condition(
        self,
        term: PrintTerminator,
        bb: Any,
        keyword_if: bool = False,
        boolean_flip: bool = False,
    ) -> None:
        """``opCbranch``: print the branching condition.

        The AST backend records the condition already oriented as "true means
        jump", so Ghidra's ``isBooleanFlip``/``isFallthruTrue`` handling has no
        counterpart here and the negation path is not needed.
        """
        yesif = keyword_if or self.is_set(Modifier.flat.value)
        yesparen = not self.is_set(Modifier.comma_separate.value)
        m = self.mods
        if yesif:
            self.emit.tag_op(self.KEYWORD_IF, SyntaxHighlight.keyword_color, term.op)
            self.emit.spaces(1)
        id = (
            self.emit.open_paren(self.OPEN_PAREN)
            if yesparen
            else self.emit.open_group()
        )
        if boolean_flip:
            # Ghidra's isBooleanFlip/isFallthruTrue path: the structured body is
            # the fall-through edge, so the printed condition is negated.  When
            # the top operator can be flipped, negatetoken does it instead.
            if self.check_print_negation(term.condition):
                m |= Modifier.negatetoken.value
                boolean_flip = False
        if boolean_flip:
            self.push_op(PrintC.boolean_not, term.op)
        if term.condition is not None:
            self.push_node(term.condition, term.op, m)
        self.recurse()
        if yesparen:
            self.emit.close_paren(self.CLOSE_PAREN, id)
        else:
            self.emit.close_group(id)
        if yesif and keyword_if:
            self.emit.spaces(1)
            self.emit.print(self.KEYWORD_GOTO, SyntaxHighlight.keyword_color)
            self.emit.spaces(1)
            self.emit_label_for_address(term.true_target)
            self.emit.print(self.SEMICOLON)

    def emit_terminator(
        self, term: PrintTerminator, bb: Any, boolean_flip: bool = False
    ) -> None:
        """Emit the branch/return/switch statements closing a basic block.

        ``boolean_flip`` only applies when the condition is being printed as a
        structured loop/if condition (``comma_separate``/``only_branch``); an
        unstructured ``if (cond) goto`` keeps the branch condition as recorded.
        """
        kind = term.kind
        if kind in ("return", "noreturn_call", "none"):
            for text in term.text:
                self.emit.print(text, SyntaxHighlight.comment_color)
            return
        if kind == "cbranch":
            if self.is_set(Modifier.comma_separate.value):
                # Inside a while(...)/if(...) header: print the bare condition,
                # as opCbranch does when comma_separate is active.
                self.emit_condition(term, bb, boolean_flip=boolean_flip)
                return
            id = self.emit.begin_statement(term.op)
            self.emit_condition(term, bb, keyword_if=True)
            self.emit.end_statement(id)
            if term.fallthrough is not None:
                self.emit.tag_line()
                id = self.emit.begin_statement(term.op)
                self.emit_goto_statement(bb, term.fallthrough, F_GOTO_GOTO)
                self.emit.end_statement(id)
            return
        if kind == "cbranch_indirect":
            for text in term.text:
                self.emit.print(text, SyntaxHighlight.comment_color)
            return
        if kind == "switch":
            self.emit_branch_indirect(term)
            return
        if kind == "tail_call":
            id = self.emit.begin_statement(term.op)
            if term.tail_call_no_return:
                self.push_op(PrintC.function_call, term.op)
                self.push_atom(
                    Atom(
                        term.tail_call or "",
                        TagType.functoken,
                        SyntaxHighlight.funcname_color,
                        op=term.op,
                    )
                )
                self.push_atom(
                    Atom(self.EMPTY_STRING, TagType.blanktoken, SyntaxHighlight.no_color)
                )
                self.recurse()
                self.emit.print(self.SEMICOLON)
                self.emit_comment_text("tail jump; no retorna", term.op.address)
            else:
                self.emit.tag_op(self.KEYWORD_RETURN, SyntaxHighlight.keyword_color, term.op)
                self.emit.spaces(1)
                self.push_op(PrintC.function_call, term.op)
                self.push_atom(
                    Atom(
                        term.tail_call or "",
                        TagType.functoken,
                        SyntaxHighlight.funcname_color,
                        op=term.op,
                    )
                )
                self.push_atom(
                    Atom(
                        "/* forwarded arguments, unknown prototype */",
                        TagType.syntax,
                        SyntaxHighlight.comment_color,
                        op=term.op,
                    )
                )
                self.recurse()
                self.emit.print(self.SEMICOLON)
            self.emit.end_statement(id)
            return
        if kind == "branch_indirect":
            id = self.emit.begin_statement(term.op)
            self.emit.print(self.KEYWORD_GOTO, SyntaxHighlight.keyword_color)
            self.emit.spaces(1)
            self.push_op(PrintC.dereference, term.op)
            if term.indirect is not None:
                self.push_node(term.indirect, term.op, self.mods)
            self.recurse()
            self.emit.print(self.SEMICOLON)
            self.emit.end_statement(id)
            return
        # branch / fallthrough
        if kind in ("branch", "fallthrough") and term.true_target is not None:
            id = self.emit.begin_statement(term.op)
            self.emit_goto_statement(bb, term.true_target, F_GOTO_GOTO)
            self.emit.end_statement(id)
            return
        for text in term.text:
            self.emit.print(text, SyntaxHighlight.no_color)

    def emit_branch_indirect(self, term: PrintTerminator) -> None:
        """``opBranchind`` plus the case labels of a recovered jump table."""
        id = self.emit.begin_statement(term.op)
        self.emit.tag_op(self.KEYWORD_SWITCH, SyntaxHighlight.keyword_color, term.op)
        paren = self.emit.open_paren(self.OPEN_PAREN)
        if term.switch_value is not None:
            resolver = self.resolver
            if resolver is not None and not self.option_nocasts:
                self.push_op(PrintC.typecast, term.op)
                self.push_type(resolver.integer(32, False))
            self.push_node(term.switch_value, term.op, self.mods)
        self.recurse()
        self.emit.close_paren(self.CLOSE_PAREN, paren)
        self.emit.open_brace(self.OPEN_CURLY, self.option_brace_switch)
        indent = self.emit.start_indent()
        for value, target in term.switch_cases:
            self.emit.tag_line()
            self.emit.print(self.KEYWORD_CASE, SyntaxHighlight.keyword_color)
            self.emit.spaces(1)
            self.push_constant(
                value,
                self.resolver.integer(32, False) if self.resolver else None,
                TagType.casetoken,
                None,
                term.op,
                0,
            )
            self.recurse()
            self.emit.print(self.COLON)
            self.emit.spaces(1)
            self.emit.print(self.KEYWORD_GOTO, SyntaxHighlight.keyword_color)
            self.emit.spaces(1)
            self.emit_label_for_address(target)
            self.emit.print(self.SEMICOLON)
        self.emit.tag_line()
        self.emit.stop_indent(indent)
        self.emit.print(self.CLOSE_CURLY)
        self.emit.end_statement(id)

    def emit_goto_statement(self, bl: Any, target: Any, type: int) -> None:
        """Emit ``goto``/``break``/``continue`` for an unstructured branch.

        ``target`` is either a basic-block address or a structured block, whose
        front leaf provides the label (``FlowBlock::getFrontLeaf``).
        """
        if type == F_BREAK_GOTO:
            self.emit.print(self.KEYWORD_BREAK, SyntaxHighlight.keyword_color)
        elif type == F_CONTINUE_GOTO:
            self.emit.print(self.KEYWORD_CONTINUE, SyntaxHighlight.keyword_color)
        else:
            self.emit.print(self.KEYWORD_GOTO, SyntaxHighlight.keyword_color)
            self.emit.spaces(1)
            if target is None or isinstance(target, int):
                self.emit_label_for_address(target)
            else:
                self.emit_label_for_address(
                    getattr(front_leaf(target), "start_addr", None)
                )
        self.emit.print(self.SEMICOLON)

    def emit_label_for_address(
        self, address: Optional[int], is_goto: bool = True
    ) -> None:
        doc = self.doc
        if is_goto and self._recording_labels is not None and address is not None:
            self._recording_labels.add(address)
        if address is None or doc is None:
            self.emit.print("unknown_label", SyntaxHighlight.no_color)
            return
        label = doc.labels.get(address)
        if label is not None:
            self.emit.tag_label(label, SyntaxHighlight.no_color, PrintAddress("ram", address), address)
            return
        raw = PrintAddress("ram", address)
        self.emit.tag_label(
            f"code_{raw.shortcut}{raw.print_raw()}",
            SyntaxHighlight.no_color,
            raw,
            address,
        )

    def emit_label(self, bl: Any) -> None:
        leaf = front_leaf(bl)
        address = getattr(leaf, "start_addr", None)
        if address is not None and self.doc is not None:
            label = self.doc.labels.get(address)
            raw = PrintAddress("ram", address)
            self.emit.tag_label(
                label if label is not None else f"code_{raw.shortcut}{raw.print_raw()}",
                SyntaxHighlight.no_color,
                raw,
                address,
            )
            return
        self.emit_label_for_address(address, is_goto=False)

    def emit_label_statement(self, bl: Any) -> None:
        if self.is_set(Modifier.only_branch.value):
            return
        address = getattr(bl, "start_addr", None)
        if address is None:
            return
        if self.label_mode == "none":
            needs_label = False
        elif self.is_set(Modifier.flat.value):
            needs_label = True
        else:
            needs_label = address in self._label_targets
        if not needs_label:
            return
        self.emit.tag_line(0)
        self.emit_label(bl)
        self.emit.print(self.COLON)

    def emit_any_label_statement(self, bl: Any) -> None:
        leaf = front_leaf(bl)
        if leaf is None:
            return
        self.emit_label_statement(leaf)

    def emit_block_copy(self, bl: Any) -> None:
        self.emit_any_label_statement(bl)
        children = _sub_blocks(bl)
        if children:
            self.emit_block(children[0])

    def emit_block_goto(self, bl: Any) -> None:
        self.push_mod()
        self.set_mod(Modifier.no_branch.value)
        children = _sub_blocks(bl)
        for child in children:
            self.emit_block(child)
        self.pop_mod()
        target = getattr(bl, "gototarget", None)
        if target is not None:
            self.emit.tag_line()
            id = self.emit.begin_statement(None)
            self.emit_goto_statement(bl, getattr(target, "start_addr", None), F_GOTO_GOTO)
            self.emit.end_statement(id)

    def emit_block_multi_goto(self, bl: Any) -> None:
        self.emit_block_goto(bl)

    def falls_through(self, first: Any, second: Any) -> bool:
        """Does control flow naturally from ``first`` into ``second``?

        ``FlowBlock::nextInFlow`` equivalent: the last basic block of ``first``
        must have ``second``'s entry block as its only successor.
        """
        doc = self.doc
        leaf = _last_leaf(first)
        target = front_leaf(second)
        if doc is None or leaf is None or target is None:
            return False
        start = getattr(leaf, "start_addr", None)
        entry = getattr(target, "start_addr", None)
        if start is None or entry is None:
            return False
        for block in doc.blocks:
            if block.start == start:
                return tuple(block.successors) == (entry,)
        return False

    def emit_block_ls(self, bl: Any) -> None:
        children = _sub_blocks(bl)
        if self.is_set(Modifier.only_branch.value):
            if children:
                self.emit_block(children[-1])
            return
        if not children:
            return
        index = 0
        subbl = children[index]
        index += 1
        id1 = self.emit.begin_block(subbl)
        if index == len(children):
            self.emit_block(subbl)
            self.emit.end_block(id1)
            return
        self.push_mod()
        if not self.is_set(Modifier.flat.value):
            self.set_mod(Modifier.no_branch.value)
        if not self.falls_through(subbl, children[index]):
            self.push_mod()
            self.set_mod(Modifier.nofallthru.value)
            self.unset_mod(Modifier.no_branch.value)
            self.emit_block(subbl)
            self.pop_mod()
        else:
            self.emit_block(subbl)
        self.emit.end_block(id1)

        while index < len(children) - 1:
            subbl = children[index]
            index += 1
            id2 = self.emit.begin_block(subbl)
            if not self.falls_through(subbl, children[index]):
                self.push_mod()
                self.set_mod(Modifier.nofallthru.value)
                self.unset_mod(Modifier.no_branch.value)
                self.emit_block(subbl)
                self.pop_mod()
            else:
                self.emit_block(subbl)
            self.emit.end_block(id2)
        self.pop_mod()
        subbl = children[index]
        id3 = self.emit.begin_block(subbl)
        self.emit_block(subbl)
        self.emit.end_block(id3)

    def emit_block_condition(self, bl: Any) -> None:
        children = (bl.cond1, bl.cond2)
        if self.is_set(Modifier.no_branch.value):
            id = self.emit.begin_block(children[0])
            self.emit_block(children[0])
            self.emit.end_block(id)
            return
        if self.is_set(Modifier.only_branch.value) or self.is_set(
            Modifier.comma_separate.value
        ):
            id = self.emit.open_paren(self.OPEN_PAREN)
            self.emit_block(children[0])
            self.push_mod()
            self.unset_mod(Modifier.only_branch.value)
            self.set_mod(Modifier.comma_separate.value)
            pol = ReversePolish(
                tok=(
                    PrintC.boolean_and
                    if getattr(bl, "op", "&&") == "&&"
                    else PrintC.boolean_or
                ),
                visited=1,
                op=None,
            )
            self.emit_op(pol)
            id2 = self.emit.open_paren(self.OPEN_PAREN)
            self.emit_block(children[1])
            self.emit.close_paren(self.CLOSE_PAREN, id2)
            self.pop_mod()
            self.emit.close_paren(self.CLOSE_PAREN, id)

    def emit_block_if(self, bl: Any) -> None:
        pending_brace = PendingBrace(self.option_brace_ifelse)
        if self.is_set(Modifier.pending_brace.value):
            self.emit.set_pending_print(pending_brace)

        # An if block never prints its final branch.
        self.push_mod()
        self.unset_mod(
            Modifier.no_branch.value
            | Modifier.only_branch.value
            | Modifier.pending_brace.value
        )

        cond_block = _condition_block(bl)
        self.push_mod()
        self.set_mod(Modifier.no_branch.value)
        self.emit_block(cond_block)
        self.pop_mod()
        self.emit_comment_block_tree(cond_block)
        if self.emit.has_pending_print(pending_brace):
            # A brace was issued but did not emit: keep "else if" syntax.
            self.emit.cancel_pending_print()
            self.emit.spaces(1)
        else:
            self.emit.tag_line()

        self.emit.tag_op(self.KEYWORD_IF, SyntaxHighlight.keyword_color, None)
        self.emit.spaces(1)
        self.push_mod()
        self.set_mod(Modifier.only_branch.value)
        self.emit_block(cond_block, boolean_flip=_if_condition_flipped(self, bl))
        self.pop_mod()

        block_type = getattr(bl, "block_type", "if")
        if block_type == "ifgoto":
            self.emit.spaces(1)
            id = self.emit.begin_statement(None)
            self.emit_goto_statement(
                cond_block, getattr(bl, "gototarget", None), F_GOTO_GOTO
            )
            self.emit.end_statement(id)
        else:
            self.set_mod(Modifier.no_branch.value)
            id = self.emit.open_brace_indent(self.OPEN_CURLY, self.option_brace_ifelse)
            true_block = getattr(bl, "true_block", None)
            if true_block is not None:
                id1 = self.emit.begin_block(true_block)
                self.emit_block(true_block)
                self.emit.end_block(id1)
            self.emit.close_brace_indent(self.CLOSE_CURLY, id)
            if block_type == "ifelse":
                self.emit.tag_line()
                self.emit.print(self.KEYWORD_ELSE, SyntaxHighlight.keyword_color)
                else_block = getattr(bl, "false_block", None)
                if else_block is not None:
                    else_type = getattr(else_block, "block_type", "")
                    if else_type in ("if", "ifelse", "ifgoto"):
                        # Attempt to merge the "else" and "if" syntax
                        self.set_mod(Modifier.pending_brace.value)
                        id2 = self.emit.begin_block(else_block)
                        self.emit_block(else_block)
                        self.emit.end_block(id2)
                    else:
                        id2 = self.emit.open_brace_indent(
                            self.OPEN_CURLY, self.option_brace_ifelse
                        )
                        id3 = self.emit.begin_block(else_block)
                        self.emit_block(else_block)
                        self.emit.end_block(id3)
                        self.emit.close_brace_indent(self.CLOSE_CURLY, id2)
            elif block_type == "ifnoexit":
                follow_block = getattr(bl, "false_block", None)
                if follow_block is not None:
                    id2 = self.emit.begin_block(follow_block)
                    self.emit_block(follow_block)
                    self.emit.end_block(id2)
        self.pop_mod()
        if pending_brace.get_indent_id() >= 0:
            self.emit.close_brace_indent(self.CLOSE_CURLY, pending_brace.get_indent_id())

    def emit_block_while_do(self, bl: Any) -> None:
        # while/do blocks never print their final branch.
        self.push_mod()
        self.unset_mod(Modifier.no_branch.value | Modifier.only_branch.value)
        self.emit_any_label_statement(bl)
        cond_block = bl.cond_block
        self.emit_comment_block_tree(cond_block)
        self.emit.tag_line()
        self.emit.tag_op(self.KEYWORD_WHILE, SyntaxHighlight.keyword_color, None)
        self.emit.spaces(1)
        id1 = self.emit.open_paren(self.OPEN_PAREN)
        self.push_mod()
        self.set_mod(Modifier.comma_separate.value)
        self.emit_block(
            cond_block,
            boolean_flip=_loop_condition_flipped(
                self, cond_block, getattr(front_leaf(bl.body_block), "start_addr", None)
            ),
        )
        self.pop_mod()
        self.emit.close_paren(self.CLOSE_PAREN, id1)
        indent = self.emit.open_brace_indent(self.OPEN_CURLY, self.option_brace_loop)
        self.set_mod(Modifier.no_branch.value)
        id2 = self.emit.begin_block(bl.body_block)
        self.emit_block(bl.body_block)
        self.emit.end_block(id2)
        self.emit.close_brace_indent(self.CLOSE_CURLY, indent)
        self.pop_mod()

    def emit_block_do_while(self, bl: Any) -> None:
        self.push_mod()
        self.unset_mod(Modifier.no_branch.value | Modifier.only_branch.value)
        self.emit_any_label_statement(bl)
        self.emit.tag_line()
        self.emit.print(self.KEYWORD_DO, SyntaxHighlight.keyword_color)
        id = self.emit.open_brace_indent(self.OPEN_CURLY, self.option_brace_loop)
        self.push_mod()
        id2 = self.emit.begin_block(bl.body_block)
        self.set_mod(Modifier.no_branch.value)
        self.emit_block(bl.body_block)
        self.emit.end_block(id2)
        self.pop_mod()
        self.emit.close_brace_indent(self.CLOSE_CURLY, id)
        self.emit.spaces(1)
        self.emit.tag_op(self.KEYWORD_WHILE, SyntaxHighlight.keyword_color, None)
        self.emit.spaces(1)
        self.set_mod(Modifier.only_branch.value)
        self.emit_block(
            bl.body_block,
            boolean_flip=_loop_condition_flipped(
                self,
                _last_leaf(bl.body_block),
                getattr(front_leaf(bl.body_block), "start_addr", None),
            ),
        )
        self.emit.print(self.SEMICOLON)
        self.pop_mod()

    def emit_block_inf_loop(self, bl: Any) -> None:
        self.push_mod()
        self.unset_mod(Modifier.no_branch.value | Modifier.only_branch.value)
        self.emit_any_label_statement(bl)
        self.emit.tag_line()
        self.emit.print(self.KEYWORD_DO, SyntaxHighlight.keyword_color)
        id = self.emit.open_brace_indent(self.OPEN_CURLY, self.option_brace_loop)
        id1 = self.emit.begin_block(bl.body_block)
        self.emit_block(bl.body_block)
        self.emit.end_block(id1)
        self.emit.close_brace_indent(self.CLOSE_CURLY, id)
        self.emit.spaces(1)
        self.emit.tag_op(self.KEYWORD_WHILE, SyntaxHighlight.keyword_color, None)
        id2 = self.emit.open_paren(self.OPEN_PAREN)
        self.emit.spaces(1)
        self.emit.print(self.KEYWORD_TRUE, SyntaxHighlight.const_color)
        self.emit.spaces(1)
        self.emit.close_paren(self.CLOSE_PAREN, id2)
        self.emit.print(self.SEMICOLON)
        self.pop_mod()

    def emit_block_switch(self, bl: Any) -> None:
        self.push_mod()
        self.unset_mod(Modifier.no_branch.value | Modifier.only_branch.value)
        self.push_mod()
        self.set_mod(Modifier.no_branch.value)
        cases = list(getattr(bl, "cases", ()) or ())
        if cases:
            self.emit_block(cases[0])
        self.pop_mod()
        self.emit.tag_line()
        self.push_mod()
        self.set_mod(Modifier.only_branch.value | Modifier.comma_separate.value)
        if cases:
            self.emit_block(cases[0])
        self.pop_mod()
        self.emit.open_brace(self.OPEN_CURLY, self.option_brace_switch)
        values_by_target = _switch_values_by_target(self, cases[0] if cases else None)
        for index, case in enumerate(cases[1:], start=1):
            self.emit_switch_case(index, bl, case, values_by_target)
            id = self.emit.start_indent()
            id2 = self.emit.begin_block(case)
            self.emit_block(case)
            self.emit.end_block(id2)
            self.emit.stop_indent(id)
        self.emit.tag_line()
        self.emit.print(self.CLOSE_CURLY)
        self.pop_mod()

    def emit_switch_case(
        self,
        casenum: int,
        switchbl: Any,
        case: Any,
        values_by_target: dict[int, list[int]],
    ) -> None:
        """``emitSwitchCase``: emit the labels of one case block."""
        address = getattr(front_leaf(case), "start_addr", None)
        values = values_by_target.get(address, []) if address is not None else []
        ctype = self.resolver.integer(32, False) if self.resolver else None
        if not values:
            self.emit.tag_line()
            self.emit.tag_case_label(
                self.KEYWORD_DEFAULT, SyntaxHighlight.keyword_color, None, 0
            )
            self.emit.print(self.COLON)
            return
        for value in values:
            self.emit.tag_line()
            self.emit.print(self.KEYWORD_CASE, SyntaxHighlight.keyword_color)
            self.emit.spaces(1)
            self.push_constant(value, ctype, TagType.casetoken, None, None, 0)
            self.recurse()
            self.emit.print(self.COLON)

    # -- comments --------------------------------------------------------
    def emit_comment_group(self, inst: Any) -> None:
        self.commsorter.setup_op_list(inst)
        while self.commsorter.has_next():
            comm = self.commsorter.get_next()
            if comm.emitted:
                continue
            if (self.instr_comment_type & comm.type) == 0:
                continue
            self.emit_line_comment(-1, comm)

    def emit_comment_block_tree(self, bl: Any) -> None:
        if bl is None:
            return
        if isinstance(bl, BlockBasic):
            self.commsorter.setup_block_list(bl)
            self.emit_comment_group(None)
            return
        for child in _sub_blocks(bl):
            self.emit_comment_block_tree(child)

    def emit_comment_func_header(self, fd: PrintableFunction) -> None:
        extralinebreak = False
        self.commsorter.setup_header(0)
        while self.commsorter.has_next():
            comm = self.commsorter.get_next()
            if comm.emitted:
                continue
            if (self.head_comment_type & comm.type) == 0:
                continue
            self.emit_line_comment(0, comm)
            extralinebreak = True
        if self.option_nocasts:
            if extralinebreak:
                self.emit.tag_line()
            warning = Comment(
                type=CommentType.warningheader.value,
                address=fd.address,
                text="DISPLAY WARNING: Type casts are NOT being printed",
            )
            self.emit_line_comment(0, warning)
            extralinebreak = True
        if extralinebreak:
            self.emit.tag_line()

    # -- type definitions ------------------------------------------------
    def emit_type_definition(self, ct: PrintType) -> None:
        if ct.metatype == METATYPE_STRUCT or ct.metatype == METATYPE_UNION:
            self.emit_struct_definition(ct)
        elif ct.metatype == METATYPE_ENUM:
            self.emit_enum_definition(ct)
        else:
            self.clear()
            raise EmitError("Unsupported typedef")

    def emit_struct_definition(self, ct: PrintType) -> None:
        if not ct.name:
            self.clear()
            raise EmitError("Trying to save unnamed structure")
        keyword = "typedef struct" if ct.metatype == METATYPE_STRUCT else "typedef union"
        self.emit.tag_line()
        self.emit.print(keyword, SyntaxHighlight.keyword_color)
        id = self.emit.open_brace_indent(self.OPEN_CURLY, BraceStyle.same_line)
        self.emit.tag_line()
        fields = list(ct.fields)
        for index, item in enumerate(fields):
            self.push_type_start(item.ctype, False)
            self.push_atom(Atom(item.name, TagType.syntax, SyntaxHighlight.var_color))
            self.push_type_end(item.ctype)
            self.recurse()
            if index != len(fields) - 1:
                self.emit.print(self.COMMA)
                self.emit.tag_line()
        self.emit.close_brace_indent(self.CLOSE_CURLY, id)
        self.emit.spaces(1)
        self.emit.print(_bare_type_name(ct))
        self.emit.print(self.SEMICOLON)

    def emit_enum_definition(self, ct: PrintType) -> None:
        if not ct.name:
            self.clear()
            raise EmitError("Trying to save unnamed enumeration")
        self.push_mod()
        sign = ct.metatype == METATYPE_INT
        self.emit.tag_line()
        self.emit.print("typedef enum", SyntaxHighlight.keyword_color)
        id = self.emit.open_brace_indent(self.OPEN_CURLY, BraceStyle.same_line)
        entries = _enum_entries(ct)
        self.emit.tag_line()
        for index, (value, name) in enumerate(entries):
            self.emit.print(name, SyntaxHighlight.const_color)
            self.emit.spaces(1)
            self.emit.print(self.EQUALSIGN, SyntaxHighlight.no_color)
            self.emit.spaces(1)
            self.push_integer(value, ct.size, sign, TagType.syntax, None, None, 0)
            self.recurse()
            self.emit.print(self.SEMICOLON)
            if index != len(entries) - 1:
                self.emit.tag_line()
        self.pop_mod()
        self.emit.close_brace_indent(self.CLOSE_CURLY, id)
        self.emit.spaces(1)
        self.emit.print(_bare_type_name(ct))
        self.emit.print(self.SEMICOLON)


_AST_DISPATCH = {
    "address_of": PrintC.op_address_of,
    "cast": PrintC.op_type_cast,
    "const": PrintC.op_constant,
    "extract": PrintC.op_extract,
    "fp_const": PrintC.op_float_constant,
    "function_address": PrintC.op_function_address,
    "load": PrintC.op_load,
    "load_typed": PrintC.op_load,
    "opaque": PrintC.op_opaque,
    "op": None,  # filled in below (needs the operator name)
    "phi": PrintC.op_multiequal,
    "segment": PrintC.op_segment,
    "select": PrintC.op_select,
    "sext": PrintC.op_int_sext,
    "store": PrintC.op_store,
    "string": PrintC.op_string,
    "trunc": PrintC.op_subpiece,
    "unary": PrintC.op_unary,
    "var": PrintC.op_var,
    "zext": PrintC.op_int_zext,
}


def _dispatch_ast(printer: PrintC, node: PrintExpr, op: Any) -> None:
    """Route a normalized AST node to the matching C printing rule."""
    kind = node.kind
    if kind == "op":
        name = node.name
        if name in PrintC._UNSIGNED_COMPARISONS or name in PrintC._SIGNED_COMPARISONS:
            printer.op_comparison(node, op)
        elif name == "land" or name == "lor":
            printer.op_binary_token(getattr(PrintC, PrintC._BINARY_TOKENS[name]), node, op)
        elif name in PrintC._BINARY_TOKENS:
            printer.op_arithmetic(node, op)
        else:
            printer.op_func(node, op, name)
        return
    handler = _AST_DISPATCH.get(kind)
    if handler is None:
        printer.op_opaque(node, op)
        return
    handler(printer, node, op)


_AST_DISPATCH["op"] = _dispatch_ast


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _calc_mask(sz: int) -> int:
    return (1 << (sz * 8)) - 1


def _char_hex_escape(val: int) -> str:
    if val < 256:
        return f"\\x{val:02x}"
    if val < 65536:
        return f"\\x{val:04x}"
    return f"\\x{val:08x}"


def _format_binary_text(val: int) -> str:
    import io

    out = io.StringIO()
    PrintLanguage.format_binary(out, val)
    return out.getvalue()


def _float_from_bits(val: int, size: int) -> float:
    import struct

    try:
        if size == 4:
            return struct.unpack("<f", struct.pack("<I", val & 0xFFFFFFFF))[0]
        if size == 8:
            return struct.unpack("<d", struct.pack("<Q", val & 0xFFFFFFFFFFFFFFFF))[0]
    except struct.error:  # pragma: no cover - defensive
        return float(val)
    return float(val)


def _quote_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=True)


def _string_at_address(doc: PrintableFunction, address: int) -> Optional[str]:
    artifacts = doc.artifacts
    analyzer = getattr(artifacts, "analyzer", None)
    for entry in getattr(analyzer, "strings", ()) or ():
        if entry.address == address:
            return entry.text
    return None


def _type_size(arch: Any, name: str, default: int) -> int:
    database = getattr(arch, "type_database", None)
    if database is None:
        return default
    try:
        ctype = database.get(name)
    except (KeyError, ValueError):
        return default
    return int(ctype.size or default)


def _probe_goto_targets(printer: "PrintC", fd: PrintableFunction) -> set:
    """Run the body once against a throwaway emitter to learn the goto targets.

    Ghidra knows which blocks are unstructured goto targets from the flags the
    structurer sets on each ``FlowBlock``.  MiniGhidraPy's ``collapse`` port does
    not track them, so the printer discovers the same information by emitting the
    body once and recording every address a printed ``goto`` refers to.
    """
    import io

    probe = PrintC(None)
    probe.set_output_stream(io.StringIO())
    probe.set_markup(False)
    probe.attach_document(fd)
    probe.label_mode = "none"
    probe._label_targets = set()
    probe._recording_labels = set()
    probe.push_mod()
    id1 = probe.emit.begin_function(fd)
    probe.push_scope(fd.scope)
    id2 = probe.emit.open_brace_indent(probe.OPEN_CURLY, probe.option_brace_func)
    probe._emit_function_body(fd)
    probe.emit.close_brace_indent(probe.CLOSE_CURLY, id2)
    probe.pop_scope()
    probe.emit.end_function(id1)
    probe.emit.flush()
    probe.pop_mod()
    targets = set(probe._recording_labels or ())
    for comments in fd.comments.values():
        for comment in comments:
            comment.set_emitted(False)
    for comment in fd.header_comments:
        comment.set_emitted(False)
    return targets


def _cast_node(ctype: PrintType, node: PrintExpr, op: Any) -> PrintExpr:
    """Wrap an expression in an explicit cast node (the AST's CAST operator)."""
    if ctype is None:
        return node
    return PrintExpr(
        kind="cast",
        args=(node,),
        bits=ctype.size * 8,
        ctype=ctype,
        op=op,
    )


def _terminator_prints(term: PrintTerminator) -> bool:
    """Does this terminator emit any token? (returns/halts emit nothing)."""
    if term.kind in ("return", "noreturn_call", "none"):
        return bool(term.text)
    return True


def _loop_condition_flipped(
    printer: "PrintC", cond_block: Any, repeat_target: Optional[int]
) -> bool:
    """Is the loop entered/repeated through the fall-through edge?

    ``while``/``do`` bodies run while the printed condition is \e true, so when
    the structurer attached the body to the fall-through edge the condition has
    to be negated (Ghidra's ``isFallthruTrue``/``isBooleanFlip``).
    """
    doc = printer.doc
    start = getattr(front_leaf(cond_block), "start_addr", None)
    if doc is None or start is None or repeat_target is None:
        return False
    term = doc.terminators.get(start)
    if term is None or term.kind not in ("cbranch", "cbranch_indirect"):
        return False
    if term.true_target == repeat_target:
        return False
    return term.fallthrough == repeat_target


def _if_condition_flipped(printer: "PrintC", bl: Any) -> bool:
    """Is the structured if-body the fall-through edge of the condition block?"""
    doc = printer.doc
    cond_block = _condition_block(bl)
    body = getattr(bl, "true_block", None)
    if doc is None or cond_block is None or body is None:
        return False
    start = getattr(front_leaf(cond_block), "start_addr", None)
    entry = getattr(front_leaf(body), "start_addr", None)
    if start is None or entry is None:
        return False
    term = doc.terminators.get(start)
    if term is None or term.kind not in ("cbranch", "cbranch_indirect"):
        return False
    if term.true_target == entry:
        return False
    return term.fallthrough == entry


def _bare_type_name(ct: PrintType) -> str:
    """The tag of an aggregate type, without its C keyword prefix."""
    name = ct.display_name
    for prefix in ("struct ", "union ", "enum "):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _switch_values_by_target(
    printer: "PrintC", switch_block: Any
) -> dict[int, list[int]]:
    """Map each case block entry address to its recovered jump-table values."""
    values: dict[int, list[int]] = {}
    doc = printer.doc
    address = getattr(switch_block, "start_addr", None)
    if doc is None or address is None:
        return values
    term = doc.terminators.get(address)
    if term is None:
        return values
    for value, target in term.switch_cases:
        values.setdefault(target, []).append(value)
    return values


def _enum_matches(ct: PrintType, val: int) -> list[str]:
    entries = _enum_entries(ct)
    return [name for value, name in entries if value == val]


def _enum_entries(ct: PrintType) -> list[tuple[int, str]]:
    ctype = ct.ctype
    names = getattr(ctype, "enum_names", None)
    if isinstance(names, dict):
        return sorted((int(value), str(name)) for name, value in names.items())
    return []


def _find_field(ct: PrintType, off: int) -> Any:
    target = ct.pointer_to if ct.metatype == METATYPE_PTR else ct
    if target is None:
        return None
    for item in target.fields:
        if item.offset <= off < item.offset + max(1, item.size):
            return item
    return None


def _is_hidden_parameter(param: PrintParameter) -> bool:
    return bool(param.implicit) and param.symbol.name == "__hidden_return"


def _sub_blocks(bl: Any) -> list[Any]:
    """Child blocks of a structured node (``FlowBlock::subBlock``)."""
    if isinstance(bl, BlockBasic):
        return []
    if isinstance(bl, (BlockList, BlockGraph)):
        return list(getattr(bl, "blocks", ()) or ())
    if isinstance(bl, BlockWhileDo):
        return [bl.cond_block, bl.body_block]
    if isinstance(bl, (BlockDoWhile, BlockInfLoop)):
        return [bl.body_block]
    if isinstance(bl, BlockIfNoExit):
        children = [bl.cond_block, bl.true_block]
        if bl.false_block is not None:
            children.append(bl.false_block)
        return children
    if isinstance(bl, BlockIf):
        children = [bl.cond_block]
        if getattr(bl, "true_block", None) is not None:
            children.append(bl.true_block)
        if getattr(bl, "false_block", None) is not None:
            children.append(bl.false_block)
        return children
    if isinstance(bl, BlockCondition):
        return [bl.cond1, bl.cond2]
    if isinstance(bl, BlockSwitch):
        return list(getattr(bl, "cases", ()) or ())
    if isinstance(bl, (BlockGoto, BlockMultiGoto)):
        return []
    return list(getattr(bl, "blocks", ()) or ())


def _condition_block(bl: Any) -> Any:
    return getattr(bl, "cond_block", None) or getattr(bl, "cond1", None) or bl


def front_leaf(bl: Any) -> Any:
    """First basic block of a structured node (``FlowBlock::getFrontLeaf``)."""
    seen: set[int] = set()
    current = bl
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, BlockBasic):
            return current
        children = _sub_blocks(current)
        if not children:
            return None
        current = children[0]
    return None


def _basic_for_address(fd: PrintableFunction, address: int) -> Any:
    basic = BlockBasic(address)
    basic.index = 0
    return basic


def _last_leaf(bl: Any) -> Any:
    seen: set[int] = set()
    current = bl
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, BlockBasic):
            return current
        children = _sub_blocks(current)
        if not children:
            return None
        current = children[-1]
    return None


def _collect_label_targets(doc: PrintableFunction) -> set[int]:
    """Addresses that need an explicit label in structured output."""
    targets: set[int] = set()
    predecessors: dict[int, int] = {}
    for block in doc.blocks:
        for successor in block.successors:
            predecessors[successor] = predecessors.get(successor, 0) + 1
    for address, count in predecessors.items():
        if count > 1:
            targets.add(address)
    return targets


PrintLanguageCapability("c-language", PrintC, isdefault=True)
