"""Port of Ghidra's ``printlanguage.hh``/``printlanguage.cc`` base printer.

:class:`PrintLanguage` owns the pieces every high-level language back-end needs:

* the Reverse Polish Notation (RPN) stack that turns a data-flow/expression tree
  into a correctly parenthesised token stream (:meth:`PrintLanguage.push_op`,
  :meth:`PrintLanguage.push_atom`, :meth:`PrintLanguage.recurse`,
  :meth:`PrintLanguage.parentheses`, :meth:`PrintLanguage.emit_op`),
* :class:`OpToken`, the description of a language operator (printed characters,
  stage, precedence, associativity, spacing, line-break bump and the token that
  represents its negation),
* the printing *modification* stack (``force_hex``, ``comma_separate``,
  ``no_branch``, ``flat``, ``negatetoken``, ...) and the symbol *scope* stack,
* comment emission (:meth:`PrintLanguage.emit_line_comment`) including the
  ``{@...}`` annotation tokens and the comment fill characters,
* integer/float formatting helpers (:meth:`PrintLanguage.most_natural_base`,
  :meth:`PrintLanguage.format_binary`) and unicode escaping rules.

Ghidra walks ``Varnode``/``PcodeOp`` data-flow.  MiniGhidraPy's decompiler
backend keeps its own expression AST, so the pending-node queue holds AST nodes
instead of ``Varnode`` pointers and :meth:`PrintLanguage.recurse` dispatches on
``node.kind`` through :meth:`PrintLanguage.push_ast_node` /
:meth:`PrintLanguage.push_node_explicit`, which play the roles of
``TypeOp::push`` and ``pushVnExplicit``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

from .emit import (
    Emit,
    EmitError,
    EmitPrettyPrint,
    SyntaxHighlight,
)


class Modifier(IntEnum):
    """``PrintLanguage::modifiers`` -- context sensitive printing modifiers."""

    force_hex = 1
    force_dec = 2
    bestfit = 4
    force_scinote = 8
    force_pointer = 0x10
    print_load_value = 0x20
    print_store_value = 0x40
    no_branch = 0x80
    only_branch = 0x100
    comma_separate = 0x200
    flat = 0x400
    falsebranch = 0x800
    nofallthru = 0x1000
    negatetoken = 0x2000
    hide_thisparam = 0x4000
    pending_brace = 0x8000


class TagType(IntEnum):
    """``PrintLanguage::tagtype`` -- the possible kinds of :class:`Atom`."""

    syntax = 0
    vartoken = 1
    functoken = 2
    optoken = 3
    typetoken = 4
    fieldtoken = 5
    bitfieldtoken = 6
    casetoken = 7
    blanktoken = 8


class NamespaceStrategy(IntEnum):
    """``PrintLanguage::namespace_strategy``."""

    MINIMAL_NAMESPACES = 0
    NO_NAMESPACES = 1
    ALL_NAMESPACES = 2


class OpTokenType(IntEnum):
    """``OpToken::tokentype``."""

    binary = 0
    unary_prefix = 1
    postsurround = 2
    presurround = 3
    space = 4
    hiddenfunction = 5


class CommentType(IntEnum):
    """Subset of Ghidra's ``Comment::comment_type`` used by the printers."""

    warning = 1
    warningheader = 2
    user1 = 4
    user2 = 8
    user3 = 0x10
    header = 0x20
    body = warning | user1 | user2 | user3


@dataclass
class OpToken:
    """A token representing an operator in the high-level language."""

    print1: str
    print2: str = ""
    stage: int = 0
    precedence: int = 0
    associative: bool = False
    type: OpTokenType = OpTokenType.binary
    spacing: int = 0
    bump: int = 0
    negate: Optional["OpToken"] = None


@dataclass
class Atom:
    """A single non-operator token emitted by the decompiler."""

    name: str
    type: TagType
    highlight: SyntaxHighlight
    op: Any = None
    varnode: Any = None
    function: Any = None
    datatype: Any = None
    offset: int = 0
    int_value: int = 0


@dataclass
class ReversePolish:
    """An entry on the reverse polish notation (RPN) stack."""

    tok: OpToken
    visited: int = 0
    paren: bool = False
    op: Any = None
    id: int = 0
    id2: int = 0


@dataclass
class NodePending:
    """A pending data-flow node waiting to be placed on the RPN stack."""

    node: Any
    op: Any
    nodemod: int


@dataclass(frozen=True)
class Comment:
    """A comment attached to an address, mirroring Ghidra's ``Comment``."""

    type: int
    address: Any
    text: str
    emitted: bool = False

    def set_emitted(self, val: bool) -> None:
        object.__setattr__(self, "emitted", val)


def mostsigbit_set(val: int) -> int:
    """Index of the most significant set bit, or -1 (``mostsigbit_set``)."""
    if val == 0:
        return -1
    return val.bit_length() - 1


class PrintLanguageCapability:
    """Registry entry for a high-level language back-end."""

    thelist: list["PrintLanguageCapability"] = []

    def __init__(self, name: str, factory, isdefault: bool = False) -> None:
        self.name = name
        self.factory = factory
        self.isdefault = isdefault
        self.initialize()

    def initialize(self) -> None:
        if self.isdefault:
            PrintLanguageCapability.thelist.insert(0, self)
        else:
            PrintLanguageCapability.thelist.append(self)

    def build_language(self, arch: Any) -> "PrintLanguage":
        return self.factory(arch, self.name)

    @classmethod
    def get_default(cls) -> "PrintLanguageCapability":
        if not cls.thelist:
            raise EmitError("No print languages registered")
        return cls.thelist[0]

    @classmethod
    def find_capability(cls, name: str) -> Optional["PrintLanguageCapability"]:
        for plc in cls.thelist:
            if plc.name == name:
                return plc
        return None


class PrintLanguage(ABC):
    """Base class API for emitting a high-level language."""

    OPEN_PAREN = "("
    CLOSE_PAREN = ")"

    # Re-exported so callers can write PrintLanguage.force_hex like the C++ enum.
    force_hex = Modifier.force_hex.value
    force_dec = Modifier.force_dec.value
    bestfit = Modifier.bestfit.value
    force_scinote = Modifier.force_scinote.value
    force_pointer = Modifier.force_pointer.value
    print_load_value = Modifier.print_load_value.value
    print_store_value = Modifier.print_store_value.value
    no_branch = Modifier.no_branch.value
    only_branch = Modifier.only_branch.value
    comma_separate = Modifier.comma_separate.value
    flat = Modifier.flat.value
    falsebranch = Modifier.falsebranch.value
    nofallthru = Modifier.nofallthru.value
    negatetoken = Modifier.negatetoken.value
    hide_thisparam = Modifier.hide_thisparam.value
    pending_brace = Modifier.pending_brace.value

    syntax = TagType.syntax
    vartoken = TagType.vartoken
    functoken = TagType.functoken
    optoken = TagType.optoken
    typetoken = TagType.typetoken
    fieldtoken = TagType.fieldtoken
    bitfieldtoken = TagType.bitfieldtoken
    casetoken = TagType.casetoken
    blanktoken = TagType.blanktoken

    MINIMAL_NAMESPACES = NamespaceStrategy.MINIMAL_NAMESPACES
    NO_NAMESPACES = NamespaceStrategy.NO_NAMESPACES
    ALL_NAMESPACES = NamespaceStrategy.ALL_NAMESPACES

    def __init__(self, arch: Any, name: str) -> None:
        self.arch = arch
        self.cast_strategy: Any = None
        self.name = name
        self.curscope: Any = None
        self.emit: Emit = EmitPrettyPrint()
        self.modstack: list[int] = []
        self.scopestack: list[Any] = []
        self.revpol: list[ReversePolish] = []
        self.nodepend: list[NodePending] = []
        self.pending = 0
        self.reset_defaults_internal()

    # -- configuration ---------------------------------------------------
    def set_line_comment_indent(self, val: int) -> None:
        if val < 0 or val >= self.emit.get_max_line_size():
            raise EmitError("Bad comment indent value")
        self.line_commentindent = val

    def set_comment_delimeter(
        self, start: str, stop: str, usecommentfill: bool
    ) -> None:
        self.commentstart = start
        self.commentend = stop
        if usecommentfill:
            self.emit.set_comment_fill(start)
        else:
            self.emit.set_comment_fill(" " * len(start))

    def set_output_stream(self, stream) -> None:
        self.emit.set_output_stream(stream)

    def get_output_stream(self):
        return self.emit.get_output_stream()

    def set_max_line_size(self, val: int) -> None:
        self.emit.set_max_line_size(val)

    def get_max_line_size(self) -> int:
        return self.emit.get_max_line_size()

    def set_indent_increment(self, val: int) -> None:
        self.emit.set_indent_increment(val)

    def get_indent_increment(self) -> int:
        return self.emit.get_indent_increment()

    def get_line_comment_indent(self) -> int:
        return self.line_commentindent

    def get_instruction_comment(self) -> int:
        return self.instr_comment_type

    def set_instruction_comment(self, val: int) -> None:
        self.instr_comment_type = val

    def get_header_comment(self) -> int:
        return self.head_comment_type

    def set_header_comment(self, val: int) -> None:
        self.head_comment_type = val

    def set_namespace_strategy(self, strat: NamespaceStrategy) -> None:
        self.namespc_strategy = strat

    def emits_markup(self) -> bool:
        return self.emit.emits_markup()

    def set_markup(self, val: bool) -> None:
        self.emit.set_markup(val)

    def set_packed_output(self, val: bool) -> None:
        self.emit.set_packed_output(val)

    def set_flat(self, val: bool) -> None:
        if val:
            self.mods |= Modifier.flat.value
        else:
            self.mods &= ~Modifier.flat.value

    # -- printing context ------------------------------------------------
    def is_set(self, m: int) -> bool:
        return (self.mods & m) != 0

    def push_scope(self, sc: Any) -> None:
        self.scopestack.append(sc)
        self.curscope = sc

    def pop_scope(self) -> None:
        self.scopestack.pop()
        self.curscope = self.scopestack[-1] if self.scopestack else None

    def push_mod(self) -> None:
        self.modstack.append(self.mods)

    def pop_mod(self) -> None:
        self.mods = self.modstack[-1]
        self.modstack.pop()

    def set_mod(self, m: int) -> None:
        self.mods |= m

    def unset_mod(self, m: int) -> None:
        self.mods &= ~m

    def is_stack_empty(self) -> bool:
        return not self.nodepend and not self.revpol

    def is_mod_stack_empty(self) -> bool:
        return not self.modstack

    # -- RPN machinery ---------------------------------------------------
    def push_op(self, tok: OpToken, op: Any = None) -> None:
        """Push an operator token onto the RPN stack."""
        if self.pending < len(self.nodepend):
            self.recurse()

        if not self.revpol:
            paren = False
            id = self.emit.open_group()
        else:
            self.emit_op(self.revpol[-1])
            paren = self.parentheses(tok)
            if paren:
                id = self.emit.open_paren(self.OPEN_PAREN)
            else:
                id = self.emit.open_group()
        self.revpol.append(
            ReversePolish(tok=tok, visited=0, paren=paren, op=op, id=id, id2=0)
        )

    def push_atom(self, atom: Atom) -> None:
        """Push a single token (an Atom) onto the RPN stack."""
        if self.pending < len(self.nodepend):
            self.recurse()

        if not self.revpol:
            self.emit_atom(atom)
        else:
            self.emit_op(self.revpol[-1])
            self.emit_atom(atom)
            while self.revpol:
                top = self.revpol[-1]
                top.visited += 1
                if top.visited == top.tok.stage:
                    self.emit_op(self.revpol[-1])
                    if self.revpol[-1].paren:
                        self.emit.close_paren(self.CLOSE_PAREN, self.revpol[-1].id)
                    else:
                        self.emit.close_group(self.revpol[-1].id)
                    self.revpol.pop()
                else:
                    break

    def push_node(self, node: Any, op: Any, m: int) -> None:
        """Queue an expression node (``pushVn``).

        Inputs of a single operator must be queued in reverse order, exactly as
        the C++ code documents for ``pushVn``.
        """
        self.nodepend.append(NodePending(node, op, m))

    def push_node_explicit(self, node: Any, op: Any) -> None:
        """Push a leaf node (``pushVnExplicit``)."""
        if node is None:
            self.push_atom(Atom("", TagType.blanktoken, SyntaxHighlight.no_color))
            return
        if is_annotation(node):
            self.push_annotation(node, op)
            return
        if is_constant(node):
            ct = self.node_type_read_facing(node, op)
            self.push_constant(
                constant_value(node),
                ct,
                TagType.vartoken,
                node,
                op,
                display_format(ct),
            )
            return
        self.push_symbol_detail(node, op, True)

    def push_symbol_detail(self, node: Any, op: Any, is_read: bool) -> None:
        """Push tokens for a node known to match (part of) a symbol."""
        symbol = node_symbol(node)
        if symbol is None:
            self.push_unnamed_location(
                node_address(node), node_size(node), node, op
            )
            return
        symboloff = node_symbol_offset(node)
        if symboloff == -1:
            if not symbol_needs_resolution(symbol):
                self.push_symbol(symbol, node, op)
                return
            symboloff = 0
        size = node_size(node)
        if symboloff + size <= symbol_size(symbol):
            inslot = node_slot(node, op) if is_read else -1
            self.push_partial_symbol(symbol, symboloff, size, node, op, inslot, is_read)
        else:
            self.push_mismatch_symbol(symbol, symboloff, size, node, op)

    def parentheses(self, op2: OpToken) -> bool:
        """Decide whether ``op2`` needs parentheses inside the current token."""
        top = self.revpol[-1]
        top_token = top.tok
        stage = top.visited
        if top_token.type in (OpTokenType.space, OpTokenType.binary):
            if top_token.precedence > op2.precedence:
                return True
            if top_token.precedence < op2.precedence:
                return False
            if top_token.associative and top_token is op2:
                return False
            if op2.type == OpTokenType.postsurround and stage == 0:
                return False
            return True
        if top_token.type == OpTokenType.unary_prefix:
            if top_token.precedence > op2.precedence:
                return True
            if top_token.precedence < op2.precedence:
                return False
            if op2.type in (OpTokenType.unary_prefix, OpTokenType.presurround):
                return False
            return True
        if top_token.type == OpTokenType.postsurround:
            if stage == 1:
                return False  # Inside the surround
            if top_token.precedence > op2.precedence:
                return True
            if top_token.precedence < op2.precedence:
                return False
            if op2.type in (OpTokenType.postsurround, OpTokenType.binary):
                return False
            return True
        if top_token.type == OpTokenType.presurround:
            if stage == 0:
                return False  # Inside the surround
            if top_token.precedence > op2.precedence:
                return True
            if top_token.precedence < op2.precedence:
                return False
            if op2.type in (OpTokenType.unary_prefix, OpTokenType.presurround):
                return False
            return True
        if top_token.type == OpTokenType.hiddenfunction:
            if stage == 0 and len(self.revpol) > 1:
                prev_token = self.revpol[-2].tok
                if prev_token.type not in (
                    OpTokenType.binary,
                    OpTokenType.unary_prefix,
                ):
                    return False
                if prev_token.precedence < op2.precedence:
                    return False
            return True
        return True

    def emit_op(self, entry: ReversePolish) -> None:
        """Send an operator token from the RPN stack to the emitter."""
        tok = entry.tok
        if tok.type == OpTokenType.binary:
            if entry.visited != 1:
                return
            self.emit.spaces(tok.spacing, tok.bump)
            self.emit.tag_op(tok.print1, SyntaxHighlight.no_color, entry.op)
            self.emit.spaces(tok.spacing, tok.bump)
        elif tok.type == OpTokenType.unary_prefix:
            if entry.visited != 0:
                return
            self.emit.tag_op(tok.print1, SyntaxHighlight.no_color, entry.op)
            self.emit.spaces(tok.spacing, tok.bump)
        elif tok.type == OpTokenType.postsurround:
            if entry.visited == 0:
                return
            if entry.visited == 1:
                self.emit.spaces(tok.spacing, tok.bump)
                entry.id2 = self.emit.open_paren(tok.print1)
                self.emit.spaces(0, tok.bump)
            else:
                self.emit.close_paren(tok.print2, entry.id2)
        elif tok.type == OpTokenType.presurround:
            if entry.visited == 2:
                return
            if entry.visited == 0:
                entry.id2 = self.emit.open_paren(tok.print1)
            else:
                self.emit.close_paren(tok.print2, entry.id2)
                self.emit.spaces(tok.spacing, tok.bump)
        elif tok.type == OpTokenType.space:
            if entry.visited != 1:
                return
            self.emit.spaces(tok.spacing, tok.bump)
        # hiddenfunction never directly prints anything

    def emit_atom(self, atom: Atom) -> None:
        """Send the given Atom to the low-level emitter with its markup."""
        if atom.type == TagType.syntax:
            self.emit.print(atom.name, atom.highlight)
        elif atom.type == TagType.vartoken:
            self.emit.tag_variable(
                atom.name, atom.highlight, atom.varnode, atom.op
            )
        elif atom.type == TagType.functoken:
            self.emit.tag_func_name(
                atom.name, atom.highlight, atom.function, atom.op
            )
        elif atom.type == TagType.optoken:
            self.emit.tag_op(atom.name, atom.highlight, atom.op)
        elif atom.type == TagType.typetoken:
            self.emit.tag_type(atom.name, atom.highlight, atom.datatype)
        elif atom.type == TagType.fieldtoken:
            self.emit.tag_field(
                atom.name, atom.highlight, atom.datatype, atom.offset, atom.op
            )
        elif atom.type == TagType.bitfieldtoken:
            self.emit.tag_bit_field(
                atom.name, atom.highlight, atom.datatype, atom.offset, atom.op
            )
        elif atom.type == TagType.casetoken:
            self.emit.tag_case_label(
                atom.name, atom.highlight, atom.op, atom.int_value
            )
        # blanktoken prints nothing

    @staticmethod
    def unicode_needs_escape(codepoint: int) -> bool:
        """Separate clearly printable codepoints from escaped ones."""
        if codepoint < 0x20:  # C0 Control characters
            return True
        if codepoint < 0x7F:  # Printable ASCII
            if codepoint in (92, ord('"'), ord("'")):
                return True
            return False
        if codepoint < 0x100:
            if codepoint > 0xA0:  # Printable codepoints A1-FF
                return False
            return True  # Delete + C1 Control characters
        if codepoint >= 0x2FA20:  # Up to last currently defined language
            return True
        if codepoint < 0x2000:
            if 0x180B <= codepoint <= 0x180E:
                return True  # Mongolian separators
            if codepoint == 0x61C:
                return True  # arabic letter mark
            if codepoint == 0x1680:
                return True  # ogham space mark
            return False
        if codepoint < 0x3000:
            if codepoint < 0x2010:
                return True  # white space and separators
            if 0x2028 <= codepoint <= 0x202F:
                return True
            if codepoint in (0x205F, 0x2060):
                return True
            if 0x2066 <= codepoint <= 0x206F:
                return True  # bidirectional markers
            return False
        if codepoint < 0xE000:
            if codepoint == 0x3000:
                return True  # ideographic space
            if codepoint >= 0xD7FC:
                return True  # unassigned + surrogates
            return False
        if codepoint < 0xF900:
            return True  # private use
        if 0xFE00 <= codepoint <= 0xFE0F:
            return True  # variation selectors
        if codepoint == 0xFEFF:
            return True  # zero width non-breaking space
        if 0xFFF0 <= codepoint <= 0xFFFF:
            if codepoint in (0xFFFC, 0xFFFD):
                return False
            return True  # interlinear specials
        return False

    def escape_character_data(
        self,
        out,
        buf: bytes,
        count: int,
        charsize: int,
        bigend: bool,
    ) -> bool:
        """Emit a byte buffer as escaped unicode characters."""
        i = 0
        codepoint = 0
        while i < count:
            codepoint, skip = _get_codepoint(buf, i, count, charsize, bigend)
            if codepoint == 0 or codepoint == -1:
                break
            self.print_unicode(out, codepoint)
            i += skip
        return codepoint == 0

    def recurse(self) -> None:
        """Emit any complete sub-expressions still on the RPN stack."""
        modsave = self.mods
        last_pending = self.pending
        self.pending = len(self.nodepend)
        while last_pending < self.pending:
            entry = self.nodepend[-1]
            node = entry.node
            op = entry.op
            self.mods = entry.nodemod
            self.nodepend.pop()
            self.pending -= 1
            if is_implied(node):
                if has_implied_field(node):
                    self.push_implied_field(node, op)
                else:
                    self.push_ast_node(node, op)
            else:
                self.push_node_explicit(node, op)
            self.pending = len(self.nodepend)
        self.mods = modsave

    def op_binary(self, tok: OpToken, node: Any) -> None:
        """Push a binary operator plus both of its input expressions."""
        if self.is_set(Modifier.negatetoken.value):
            tok = tok.negate
            self.unset_mod(Modifier.negatetoken.value)
            if tok is None:
                raise EmitError("Could not find fliptoken")
        self.push_op(tok, node)
        args = node_args(node)
        # implied nodes are pushed in reverse order for efficiency
        self.push_node(args[1], node, self.mods)
        self.push_node(args[0], node, self.mods)

    def op_unary(self, tok: OpToken, node: Any) -> None:
        """Push a unary operator plus its input expression."""
        self.push_op(tok, node)
        self.push_node(node_args(node)[0], node, self.mods)

    def get_pending(self) -> int:
        return self.pending

    def reset_defaults_internal(self) -> None:
        self.mods = 0
        self.head_comment_type = CommentType.header | CommentType.warningheader
        self.line_commentindent = 20
        self.namespc_strategy = NamespaceStrategy.MINIMAL_NAMESPACES
        self.instr_comment_type = CommentType.user2 | CommentType.warning

    def emit_line_comment(self, indent: int, comm: Comment) -> None:
        """Emit one comment line using the language's delimiters."""
        text = comm.text
        address = comm.address
        spc = address_space(address)
        off = address_offset(address)
        if indent < 0:
            indent = self.line_commentindent
        self.emit.tag_line(indent)
        id = self.emit.start_comment()
        # The comment delimiters are not printed as comment tags, so that they
        # won't get filled.
        self.emit.tag_comment(self.commentstart, SyntaxHighlight.comment_color, spc, off)
        pos = 0
        while pos < len(text):
            tok = text[pos]
            pos += 1
            if tok in (" ", "\t"):
                count = 1
                while pos < len(text):
                    tok = text[pos]
                    if tok not in (" ", "\t"):
                        break
                    count += 1
                    pos += 1
                self.emit.spaces(count)
            elif tok == "\n":
                self.emit.tag_line()
            elif tok == "\r":
                continue
            elif tok == "{" and pos < len(text) and text[pos] == "@":
                # Comment annotation: treat as a single token
                count = 1
                while pos < len(text):
                    tok = text[pos]
                    count += 1
                    pos += 1
                    if tok == "}":
                        break
                annote = text[pos - count : pos]
                self.emit.tag_comment(
                    annote, SyntaxHighlight.comment_color, spc, off
                )
            else:
                count = 1
                while pos < len(text):
                    tok = text[pos]
                    if tok.isspace():
                        break
                    count += 1
                    pos += 1
                sub = text[pos - count : pos]
                self.emit.tag_comment(sub, SyntaxHighlight.comment_color, spc, off)
        if len(self.commentend) != 0:
            self.emit.tag_comment(
                self.commentend, SyntaxHighlight.comment_color, spc, off
            )
        self.emit.stop_comment(id)
        comm.set_emitted(True)

    def reset_defaults(self) -> None:
        self.emit.reset_defaults()
        self.reset_defaults_internal()

    def clear(self) -> None:
        self.emit.clear()
        if self.modstack:
            self.mods = self.modstack[0]
            self.modstack.clear()
        self.scopestack.clear()
        self.curscope = None
        self.revpol.clear()
        self.pending = 0
        self.nodepend.clear()

    def set_integer_format(self, nm: str) -> None:
        """Select the default integer format: "hex", "dec" or "best"."""
        if nm[:3] == "hex":
            mod = Modifier.force_hex.value
        elif nm[:3] == "dec":
            mod = Modifier.force_dec.value
        elif nm[:4] == "best":
            mod = 0
        else:
            raise EmitError("Unknown integer format option: " + nm)
        self.mods &= ~(Modifier.force_hex.value | Modifier.force_dec.value)
        self.mods |= mod

    def unnamed_field(self, off: int, size: int) -> str:
        return f"_{off:d}_{size:d}_"

    @staticmethod
    def most_natural_base(val: int) -> int:
        """Count '0'/'9' digits base 10 and '0'/'f' digits base 16."""
        countdec = 0
        tmp = val
        if tmp == 0:
            return 10
        setdig = tmp % 10
        if setdig in (0, 9):
            countdec += 1
            tmp //= 10
            while tmp != 0:
                dig = tmp % 10
                if dig == setdig:
                    countdec += 1
                else:
                    break
                tmp //= 10
        if countdec == 0:
            return 16
        if countdec == 1:
            if tmp > 1 or setdig == 9:
                return 16
        elif countdec == 2:
            if tmp > 10:
                return 16
        elif countdec in (3, 4):
            if tmp > 100:
                return 16
        else:
            if tmp > 1000:
                return 16

        counthex = 0
        tmp = val
        setdig = tmp & 0xF
        if setdig in (0, 0xF):
            counthex += 1
            tmp >>= 4
            while tmp != 0:
                dig = tmp & 0xF
                if dig == setdig:
                    counthex += 1
                else:
                    break
                tmp >>= 4
        return 10 if countdec > counthex else 16

    @staticmethod
    def format_binary(out, val: int) -> None:
        """Print a string of '0'/'1' characters representing the value."""
        pos = mostsigbit_set(val)
        if pos < 0:
            out.write("0")
            return
        if pos <= 7:
            pos = 7
        elif pos <= 15:
            pos = 15
        elif pos <= 31:
            pos = 31
        else:
            pos = 63
        mask = 1 << pos
        while mask != 0:
            out.write("1" if (mask & val) != 0 else "0")
            mask >>= 1

    # -- abstract API ----------------------------------------------------
    @abstractmethod
    def initialize_from_architecture(self) -> None:
        ...

    @abstractmethod
    def adjust_type_operators(self) -> None:
        ...

    @abstractmethod
    def set_comment_style(self, nm: str) -> None:
        ...

    @abstractmethod
    def doc_type_definitions(self, typegrp: Any) -> None:
        ...

    @abstractmethod
    def doc_all_globals(self) -> None:
        ...

    @abstractmethod
    def doc_single_global(self, sym: Any) -> None:
        ...

    @abstractmethod
    def doc_function(self, fd: Any) -> None:
        ...

    @abstractmethod
    def print_unicode(self, out, onechar: int) -> None:
        ...

    @abstractmethod
    def push_type(self, ct: Any) -> None:
        ...

    @abstractmethod
    def push_constant(
        self,
        val: int,
        ct: Any,
        tag: TagType,
        vn: Any,
        op: Any,
        display_format: int,
    ) -> None:
        ...

    @abstractmethod
    def push_equate(self, val: int, sz: int, sym: Any, vn: Any, op: Any) -> bool:
        ...

    @abstractmethod
    def push_annotation(self, vn: Any, op: Any) -> None:
        ...

    @abstractmethod
    def push_symbol(self, sym: Any, vn: Any, op: Any) -> None:
        ...

    @abstractmethod
    def push_unnamed_location(self, addr: Any, size: int, vn: Any, op: Any) -> None:
        ...

    @abstractmethod
    def push_partial_symbol(
        self, sym: Any, off: int, sz: int, vn: Any, op: Any, slot: int, allow_cast: bool
    ) -> None:
        ...

    @abstractmethod
    def push_mismatch_symbol(
        self, sym: Any, off: int, sz: int, vn: Any, op: Any
    ) -> None:
        ...

    @abstractmethod
    def push_implied_field(self, vn: Any, op: Any) -> None:
        ...

    @abstractmethod
    def push_ast_node(self, node: Any, op: Any) -> None:
        """Push a compound AST node (``TypeOp::push`` in Ghidra)."""

    @abstractmethod
    def node_type_read_facing(self, node: Any, op: Any) -> Any:
        ...

    @abstractmethod
    def get_scope_delimiter(self) -> str:
        ...

    # Block emission (``emitBlock*`` in Ghidra); each derived printer walks the
    # structured control-flow graph produced by ``block.py``/``collapse.py``.
    @abstractmethod
    def emit_block_basic(self, bb: Any) -> None:
        ...

    @abstractmethod
    def emit_block_graph(self, bl: Any) -> None:
        ...

    @abstractmethod
    def emit_block_copy(self, bl: Any) -> None:
        ...

    @abstractmethod
    def emit_block_goto(self, bl: Any) -> None:
        ...

    @abstractmethod
    def emit_block_ls(self, bl: Any) -> None:
        ...

    @abstractmethod
    def emit_block_condition(self, bl: Any) -> None:
        ...

    @abstractmethod
    def emit_block_if(self, bl: Any) -> None:
        ...

    @abstractmethod
    def emit_block_while_do(self, bl: Any) -> None:
        ...

    @abstractmethod
    def emit_block_do_while(self, bl: Any) -> None:
        ...

    @abstractmethod
    def emit_block_inf_loop(self, bl: Any) -> None:
        ...

    @abstractmethod
    def emit_block_switch(self, bl: Any) -> None:
        ...


# ---------------------------------------------------------------------------
# AST node accessors.  The decompiler backend's ``_Expr`` nodes are the
# "Varnode/PcodeOp" of MiniGhidraPy; these helpers keep PrintLanguage free of
# direct imports of the (private) backend types while reproducing the C++
# predicates that drive printing.
# ---------------------------------------------------------------------------


def node_kind(node: Any) -> str:
    return str(getattr(node, "kind", ""))


def node_value(node: Any) -> Any:
    return getattr(node, "value", None)


def node_args(node: Any) -> tuple:
    return tuple(getattr(node, "args", ()) or ())


def node_bits(node: Any) -> int:
    return int(getattr(node, "bits", 0) or 0)


def node_size(node: Any) -> int:
    """Byte size of the value the node holds (``Varnode::getSize``)."""
    return max(1, node_bits(node) // 8)


def is_implied(node: Any) -> bool:
    """Every normalized AST node prints as the expression producing its value.

    Ghidra separates \e implied from \e explicit Varnodes.  The AST backend
    hands the printer whole expression nodes, so anything carrying a ``kind`` is
    dispatched through :meth:`PrintLanguage.push_ast_node` (the role of
    ``TypeOp::push``); only nodes pushed explicitly as symbol leaves go through
    :meth:`PrintLanguage.push_node_explicit`.
    """
    return bool(node_kind(node))


def has_implied_field(node: Any) -> bool:
    """Union field selection needs an explicit member token."""
    return node_kind(node) == "implied_field"


def is_annotation(node: Any) -> bool:
    return bool(getattr(node, "annotation", False))


def is_constant(node: Any) -> bool:
    return node_kind(node) in {"const", "fp_const", "imm"}


def constant_value(node: Any) -> int:
    direct = getattr(node, "int_value", None)
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
    value = node_value(node)
    if isinstance(value, float):
        import struct

        bits = node_bits(node)
        if bits == 32:
            return struct.unpack("<I", struct.pack("<f", value))[0]
        if bits == 64:
            return struct.unpack("<Q", struct.pack("<d", value))[0]
        return int(value)
    return int(value or 0)


def node_symbol(node: Any) -> Any:
    return getattr(node, "symbol", None)


def node_symbol_offset(node: Any) -> int:
    value = getattr(node, "symbol_offset", None)
    return -1 if value is None else int(value)


def node_address(node: Any) -> Any:
    return getattr(node, "address", None)


def node_slot(node: Any, op: Any) -> int:
    value = getattr(node, "slot", None)
    return -1 if value is None else int(value)


def symbol_size(symbol: Any) -> int:
    ctype = getattr(symbol, "ctype", None)
    size = getattr(ctype, "size", None) if ctype is not None else None
    if size is None:
        size = getattr(symbol, "size", 0)
    return int(size or 0)


def symbol_needs_resolution(symbol: Any) -> bool:
    ctype = getattr(symbol, "ctype", None)
    if ctype is None:
        return False
    return bool(getattr(ctype, "needs_resolution", False))


def address_space(address: Any) -> Any:
    if address is None:
        return "ram"
    return getattr(address, "space", address)


def address_offset(address: Any) -> int:
    if address is None:
        return 0
    offset = getattr(address, "offset", None)
    return int(address if offset is None else offset)


def display_format(ctype: Any) -> int:
    value = getattr(ctype, "display_format", None)
    return 0 if value is None else int(value)




def _get_codepoint(
    buf: bytes, i: int, count: int, charsize: int, bigend: bool
) -> tuple[int, int]:
    """Decode one codepoint (``StringManager::getCodepoint``) for UTF8/16/32."""
    remaining = count - i
    if remaining <= 0:
        return -1, charsize
    if charsize == 1:
        byte = buf[i]
        if byte < 0x80:
            return byte, 1
        if 0xC0 <= byte < 0xE0 and remaining >= 2:
            return ((byte & 0x1F) << 6) | (buf[i + 1] & 0x3F), 2
        if 0xE0 <= byte < 0xF0 and remaining >= 3:
            return (
                ((byte & 0x0F) << 12)
                | ((buf[i + 1] & 0x3F) << 6)
                | (buf[i + 2] & 0x3F),
                3,
            )
        if byte >= 0xF0 and remaining >= 4:
            return (
                ((byte & 0x07) << 18)
                | ((buf[i + 1] & 0x3F) << 12)
                | ((buf[i + 2] & 0x3F) << 6)
                | (buf[i + 3] & 0x3F),
                4,
            )
        return -1, 1
    if remaining < charsize:
        return -1, charsize
    raw = buf[i : i + charsize]
    value = int.from_bytes(raw, "big" if bigend else "little")
    if charsize == 2 and 0xD800 <= value <= 0xDBFF and remaining >= 4:
        low = int.from_bytes(buf[i + 2 : i + 4], "big" if bigend else "little")
        if 0xDC00 <= low <= 0xDFFF:
            return 0x10000 + ((value - 0xD800) << 10) + (low - 0xDC00), 4
    return value, charsize
