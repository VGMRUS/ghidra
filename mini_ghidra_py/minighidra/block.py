"""Modelo de bloques de control: puerto de ``block.hh``/``block.cc`` de Ghidra.

Traducción del modelo de aristas
--------------------------------
Ghidra guarda las aristas en ``vector<BlockEdge>`` donde cada ``BlockEdge`` tiene
tres campos (``label``, ``point``, ``reverse_index``).  Aquí se traducen a listas
paralelas por bloque:

    ``in_edges``  / ``in_labels``  / ``in_rev``    (``intothis``)
    ``out_edges`` / ``out_labels`` / ``out_rev``   (``outofthis``)

``in_edges[i]``/``out_edges[i]`` siguen devolviendo el ``FlowBlock`` del otro
extremo (API que ya consumen ``printc``, ``condexe``, ``rule``, ``decompiler``),
mientras que las etiquetas viven en ``*_labels`` y ``*_rev`` guarda el índice de
la arista gemela.  Todas las mutaciones pasan por los métodos ``_add_in_edge``,
``_remove_out_edge``, ``_replace_out_edge``, ... que mantienen las cuatro listas
coherentes, igual que ``FlowBlock::addInEdge``/``halfDeleteInEdge``/etc.

Desviaciones documentadas (ver ``PORTING_NOTES.md``)
---------------------------------------------------
* Las hojas son ``BlockBasic``: no existe la capa ``BlockCopy`` de Ghidra.  Por
  eso ``BlockGraph.build_copy`` aplica sobre las propias hojas el etiquetado que
  Ghidra hace en ``Funcdata::structureReset`` (árbol de recubrimiento, aristas
  irreducibles, bucles y dominadores) antes de la estructuración.
* ``negate_condition`` no puede reescribir el ``CBRANCH`` (los ``FlowBlock`` del
  puerto no llevan p-code): sólo intercambia las aristas de salida, que es el
  efecto estructural, y devuelve ``False`` (sin cambio de dataflow).
* ``BlockBasic.is_complex`` devuelve ``False``: sin lista de p-code no se puede
  contar sentencias, así que nunca se usa la sintaxis de desbordamiento de
  ``while`` (``f_whiledo_overflow``).
* ``new_block_switch`` no ejecuta ``grabCaseBasic`` (no hay ``JumpTable``); los
  casos y sus etiquetas los reconstruye el impresor a partir del análisis.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

# ---------------------------------------------------------------------------
# FlowBlock::block_flags (block.hh)
# ---------------------------------------------------------------------------
F_GOTO_GOTO = 1  # (Block ends in) non-structured branch
F_BREAK_GOTO = 2  # Block ends with a break;
F_CONTINUE_GOTO = 4  # Block ends with a continue;
F_SWITCH_OUT = 0x10  # Output is decided by switch
F_UNSTRUCTURED_TARG = 0x20  # Block is destination of unstructured goto
F_MARK = 0x80  # Generic way to mark a block
F_MARK2 = 0x100  # A secondary mark
F_ENTRY_POINT = 0x200  # Official entry point of the function
F_INTERIOR_GOTOOUT = 0x400  # Unstructured jump out of interior
F_INTERIOR_GOTOIN = 0x800  # Target of unstructured jump to its interior
F_LABEL_BUMPUP = 0x1000  # Any label printed higher up in hierarchy
F_DONOTHING_LOOP = 0x2000  # Block does nothing in infinite loop (halt)
F_DEAD = 0x4000  # Block is in process of being deleted
F_WHILEDO_OVERFLOW = 0x8000  # whiledo conditional too big for while(cond)
F_FLIP_PATH = 0x10000  # Out edges have been flipped since last trace
F_JOINED_BLOCK = 0x20000  # Block is a merged form of original basic blocks
F_DUPLICATE_BLOCK = 0x40000  # Duplicated version of an original basic block
F_DELAYED_DONOTHING = 0x80000  # Potential do-nothing block, removal delayed
F_FINAL_TRANSFORM = 0x100000  # Has the final transform been run

# ---------------------------------------------------------------------------
# FlowBlock::edge_flags (block.hh)
# ---------------------------------------------------------------------------
F_GOTO_EDGE = 1  # Edge is unstructured
F_LOOP_EDGE = 2  # Edge completes a loop
F_DEFAULTSWITCH_EDGE = 4  # Default edge from switchblock
F_IRREDUCIBLE = 8  # Edge which must be removed to make graph reducible
F_TREE_EDGE = 0x10  # An edge in the spanning tree
F_FORWARD_EDGE = 0x20  # Edge that jumps forward in the spanning tree
F_CROSS_EDGE = 0x40  # Edge that crosses subtrees in the spanning tree
F_BACK_EDGE = 0x80  # Back edge defining a loop (reducible graph)
F_LOOP_EXIT_EDGE = 0x100  # Edge exits the body of a loop
F_IMMED_COPY = 0x200  # Copy propagation has happened across the edge

# Máscara que ``isLoopDAGIn/Out`` de Ghidra descarta (block.hh:354-358).
LOOP_DAG_MASK = F_IRREDUCIBLE | F_BACK_EDGE | F_LOOP_EXIT_EDGE | F_GOTO_EDGE
# Máscara de ``isDecisionIn/Out`` (block.hh).
DECISION_MASK = F_IRREDUCIBLE | F_BACK_EDGE | F_GOTO_EDGE
# Máscara de ``isGotoIn/Out`` (block.hh).
GOTO_MASK = F_IRREDUCIBLE | F_GOTO_EDGE


class StructureError(RuntimeError):
    """La estructuración no pudo completarse (``LowlevelError`` de Ghidra)."""


class FlowBlock:
    """Nodo de la jerarquía de control (``FlowBlock`` de Ghidra)."""

    def __init__(self, block_type: str = "plain") -> None:
        self.block_type = block_type
        self.parent: Optional["FlowBlock"] = None
        self.immed_dom: Optional["FlowBlock"] = None
        self.copymap: "FlowBlock" = self
        # vector<BlockEdge> intothis -> tres listas paralelas
        self.in_edges: list["FlowBlock"] = []
        self.in_labels: list[int] = []
        self.in_rev: list[int] = []
        # vector<BlockEdge> outofthis -> tres listas paralelas
        self.out_edges: list["FlowBlock"] = []
        self.out_labels: list[int] = []
        self.out_rev: list[int] = []
        self.flags = 0
        self.index = -1  # reverse post order
        self.visit_count = 0
        self.numdesc = 0
        self.leaf_count = 0
        self.structure_depth = 0

    # -- gestión de aristas (block.cc:77-243) ------------------------------
    def _add_in_edge(self, b: "FlowBlock", lab: int) -> None:
        """``FlowBlock::addInEdge``: añade una arista entrante desde ``b``."""
        ourrev = len(b.out_edges)
        brev = len(self.in_edges)
        self.in_edges.append(b)
        self.in_labels.append(lab)
        self.in_rev.append(ourrev)
        b.out_edges.append(self)
        b.out_labels.append(lab)
        b.out_rev.append(brev)

    def _half_delete_in_edge(self, slot: int) -> None:
        last = len(self.in_edges) - 1
        while slot < last:
            point = self.in_edges[slot + 1]
            rev = self.in_rev[slot + 1]
            self.in_edges[slot] = point
            self.in_labels[slot] = self.in_labels[slot + 1]
            self.in_rev[slot] = rev
            point.out_rev[rev] -= 1
            slot += 1
        self.in_edges.pop()
        self.in_labels.pop()
        self.in_rev.pop()

    def _half_delete_out_edge(self, slot: int) -> None:
        last = len(self.out_edges) - 1
        while slot < last:
            point = self.out_edges[slot + 1]
            rev = self.out_rev[slot + 1]
            self.out_edges[slot] = point
            self.out_labels[slot] = self.out_labels[slot + 1]
            self.out_rev[slot] = rev
            point.in_rev[rev] -= 1
            slot += 1
        self.out_edges.pop()
        self.out_labels.pop()
        self.out_rev.pop()

    def _remove_in_edge(self, slot: int) -> None:
        b = self.in_edges[slot]
        rev = self.in_rev[slot]
        self._half_delete_in_edge(slot)
        b._half_delete_out_edge(rev)

    def _remove_out_edge(self, slot: int) -> None:
        b = self.out_edges[slot]
        rev = self.out_rev[slot]
        self._half_delete_out_edge(slot)
        b._half_delete_in_edge(rev)

    def _replace_in_edge(self, num: int, b: "FlowBlock") -> None:
        oldb = self.in_edges[num]
        oldb._half_delete_out_edge(self.in_rev[num])
        self.in_edges[num] = b
        self.in_rev[num] = len(b.out_edges)
        b.out_edges.append(self)
        b.out_labels.append(self.in_labels[num])
        b.out_rev.append(num)

    def _replace_out_edge(self, num: int, b: "FlowBlock") -> None:
        oldb = self.out_edges[num]
        oldb._half_delete_in_edge(self.out_rev[num])
        self.out_edges[num] = b
        self.out_rev[num] = len(b.in_edges)
        b.in_edges.append(self)
        b.in_labels.append(self.out_labels[num])
        b.in_rev.append(num)

    def _swap_edges(self) -> None:
        """``FlowBlock::swapEdges``: intercambia las dos aristas de salida."""
        if len(self.out_edges) != 2:
            raise StructureError("Swapping edges for block that doesn't have two edges")
        self.out_edges[0], self.out_edges[1] = self.out_edges[1], self.out_edges[0]
        self.out_labels[0], self.out_labels[1] = self.out_labels[1], self.out_labels[0]
        self.out_rev[0], self.out_rev[1] = self.out_rev[1], self.out_rev[0]
        bl = self.out_edges[0]
        bl.in_rev[self.out_rev[0]] = 0
        bl = self.out_edges[1]
        bl.in_rev[self.out_rev[1]] = 1
        self.flags ^= F_FLIP_PATH

    def _set_out_edge_flag(self, i: int, lab: int) -> None:
        bbout = self.out_edges[i]
        self.out_labels[i] |= lab
        bbout.in_labels[self.out_rev[i]] |= lab

    def _clear_out_edge_flag(self, i: int, lab: int) -> None:
        bbout = self.out_edges[i]
        self.out_labels[i] &= ~lab
        bbout.in_labels[self.out_rev[i]] &= ~lab

    def _set_flag(self, fl: int) -> None:
        self.flags |= fl

    def _clear_flag(self, fl: int) -> None:
        self.flags &= ~fl

    def _clear_all_flags(self) -> None:
        self.flags = 0

    def _replace_using_map(self) -> None:
        """``FlowBlock::replaceUsingMap``: reescribe extremos vía ``copymap``."""
        self.in_edges = [point.copymap for point in self.in_edges]
        self.out_edges = [point.copymap for point in self.out_edges]
        if self.immed_dom is not None:
            self.immed_dom = self.immed_dom.copymap

    # -- deduplicado de aristas (block.cc:450-547) -------------------------
    def _eliminate_in_dups(self, bl: "FlowBlock") -> None:
        indval = -1
        i = 0
        while i < len(self.in_edges):
            if self.in_edges[i] is bl:
                if indval == -1:
                    indval = i
                    i += 1
                else:
                    self.in_labels[indval] |= self.in_labels[i]
                    rev = self.in_rev[i]
                    self._half_delete_in_edge(i)
                    bl._half_delete_out_edge(rev)
            else:
                i += 1

    def _eliminate_out_dups(self, bl: "FlowBlock") -> None:
        indval = -1
        i = 0
        while i < len(self.out_edges):
            if self.out_edges[i] is bl:
                if indval == -1:
                    indval = i
                    i += 1
                else:
                    self.out_labels[indval] |= self.out_labels[i]
                    rev = self.out_rev[i]
                    self._half_delete_out_edge(i)
                    bl._half_delete_in_edge(rev)
            else:
                i += 1

    @staticmethod
    def _find_dups(ref_edges: list["FlowBlock"], duplist: list["FlowBlock"]) -> None:
        for point in ref_edges:
            if point.flags & F_MARK2:
                continue
            if point.flags & F_MARK:
                duplist.append(point)
                point.flags |= F_MARK2
            else:
                point.flags |= F_MARK
        for point in ref_edges:
            point.flags &= ~(F_MARK | F_MARK2)

    def _dedup(self) -> None:
        duplist: list[FlowBlock] = []
        FlowBlock._find_dups(self.in_edges, duplist)
        for bl in duplist:
            self._eliminate_in_dups(bl)
        duplist = []
        FlowBlock._find_dups(self.out_edges, duplist)
        for bl in duplist:
            self._eliminate_out_dups(bl)

    # -- API pública de aristas (block.hh) ---------------------------------
    def size_out(self) -> int:
        return len(self.out_edges)

    def size_in(self) -> int:
        return len(self.in_edges)

    def get_out(self, i: int) -> "FlowBlock":
        return self.out_edges[i]

    def get_in(self, i: int) -> "FlowBlock":
        return self.in_edges[i]

    def get_out_rev_index(self, i: int) -> int:
        return self.out_rev[i]

    def get_in_rev_index(self, i: int) -> int:
        return self.in_rev[i]

    def get_false_out(self) -> "FlowBlock":
        return self.out_edges[0]

    def get_true_out(self) -> "FlowBlock":
        return self.out_edges[1]

    def get_in_index(self, bl: "FlowBlock") -> int:
        for i, point in enumerate(self.in_edges):
            if point is bl:
                return i
        return -1

    def get_out_index(self, bl: "FlowBlock") -> int:
        for i, point in enumerate(self.out_edges):
            if point is bl:
                return i
        return -1

    def get_index(self) -> int:
        return self.index

    def get_parent(self) -> Optional["FlowBlock"]:
        return self.parent

    def get_immed_dom(self) -> Optional["FlowBlock"]:
        return self.immed_dom

    def get_copy_map(self) -> "FlowBlock":
        return self.copymap

    def get_flags(self) -> int:
        return self.flags

    def get_type(self) -> str:
        return self.block_type

    def sub_block(self, i: int) -> Optional["FlowBlock"]:
        return None

    def get_start(self) -> Any:
        return None

    def get_stop(self) -> Any:
        return None

    def get_exit_leaf(self) -> Optional["FlowBlock"]:
        return None

    def get_basic_count(self) -> int:
        return self.leaf_count

    def get_structure_depth(self) -> int:
        return self.structure_depth

    def set_visit_count(self, i: int) -> None:
        self.visit_count = i

    def get_visit_count(self) -> int:
        return self.visit_count

    def is_mark(self) -> bool:
        return (self.flags & F_MARK) != 0

    def set_mark(self) -> None:
        self.flags |= F_MARK

    def clear_mark(self) -> None:
        self.flags &= ~F_MARK

    def set_donothing_loop(self) -> None:
        self.flags |= F_DONOTHING_LOOP

    def set_dead(self) -> None:
        self.flags |= F_DEAD

    def is_dead(self) -> bool:
        return (self.flags & F_DEAD) != 0

    def is_donothing_loop(self) -> bool:
        return (self.flags & F_DONOTHING_LOOP) != 0

    def is_entry_point(self) -> bool:
        return (self.flags & F_ENTRY_POINT) != 0

    def is_switch_out(self) -> bool:
        return (self.flags & F_SWITCH_OUT) != 0

    def is_unstructured_target(self) -> bool:
        return (self.flags & F_UNSTRUCTURED_TARG) != 0

    def is_interior_goto_target(self) -> bool:
        return (self.flags & F_INTERIOR_GOTOIN) != 0

    def has_interior_goto(self) -> bool:
        return (self.flags & F_INTERIOR_GOTOOUT) != 0

    def is_label_bump_up(self) -> bool:
        return (self.flags & F_LABEL_BUMPUP) != 0

    def get_flip_path(self) -> bool:
        return (self.flags & F_FLIP_PATH) != 0

    def mark_label_bump_up(self, bump: bool) -> None:
        if bump:
            self.flags |= F_LABEL_BUMPUP

    def mark_unstructured(self) -> None:
        """``FlowBlock::markUnstructured`` (sólo las estructuras lo reimplementan)."""

    def is_complex(self) -> bool:
        """``FlowBlock::isComplex``: ¿demasiado complejo para ser condición?"""
        return True

    # -- etiquetas de arista (block.hh) ------------------------------------
    def set_loop_exit(self, i: int) -> None:
        self._set_out_edge_flag(i, F_LOOP_EXIT_EDGE)

    def clear_loop_exit(self, i: int) -> None:
        self._clear_out_edge_flag(i, F_LOOP_EXIT_EDGE)

    def set_back_edge(self, i: int) -> None:
        self._set_out_edge_flag(i, F_BACK_EDGE)

    def set_immed_copy_edge(self, i: int) -> None:
        self._set_out_edge_flag(i, F_IMMED_COPY)

    def has_immed_copy_edge(self, i: int) -> bool:
        return (self.out_labels[i] & F_IMMED_COPY) != 0

    def set_goto_branch(self, i: int) -> None:
        """``FlowBlock::setGotoBranch``: punto de entrada para marcar un goto."""
        if 0 <= i < len(self.out_edges):
            self._set_out_edge_flag(i, F_GOTO_EDGE)
        else:
            raise StructureError("Could not find block edge to mark unstructured")
        self.flags |= F_INTERIOR_GOTOOUT
        self.out_edges[i].flags |= F_INTERIOR_GOTOIN

    def set_default_switch(self, pos: int) -> None:
        """``FlowBlock::setDefaultSwitch``: un único default por switch."""
        for i in range(len(self.out_edges)):
            if self.is_default_branch(i):
                self._clear_out_edge_flag(i, F_DEFAULTSWITCH_EDGE)
        self._set_out_edge_flag(pos, F_DEFAULTSWITCH_EDGE)

    def is_default_branch(self, i: int) -> bool:
        return (self.out_labels[i] & F_DEFAULTSWITCH_EDGE) != 0

    def is_goto_branch(self, i: int) -> bool:
        """Alias del puerto para ``isGotoOut`` (usado por ``printc``/``condexe``)."""
        return self.is_goto_out(i)

    def is_goto_out(self, i: int) -> bool:
        return (self.out_labels[i] & GOTO_MASK) != 0

    def is_goto_in(self, i: int) -> bool:
        return (self.in_labels[i] & GOTO_MASK) != 0

    def is_loop_out(self, i: int) -> bool:
        return (self.out_labels[i] & F_LOOP_EDGE) != 0

    def is_loop_in(self, i: int) -> bool:
        return (self.in_labels[i] & F_LOOP_EDGE) != 0

    def has_loop_in(self) -> bool:
        return any(label & F_LOOP_EDGE for label in self.in_labels)

    def has_loop_out(self) -> bool:
        return any(label & F_LOOP_EDGE for label in self.out_labels)

    def is_tree_edge_in(self, i: int) -> bool:
        return (self.in_labels[i] & F_TREE_EDGE) != 0

    def is_back_edge_in(self, i: int) -> bool:
        return (self.in_labels[i] & F_BACK_EDGE) != 0

    def is_back_edge_out(self, i: int) -> bool:
        return (self.out_labels[i] & F_BACK_EDGE) != 0

    def is_irreducible_out(self, i: int) -> bool:
        return (self.out_labels[i] & F_IRREDUCIBLE) != 0

    def is_irreducible_in(self, i: int) -> bool:
        return (self.in_labels[i] & F_IRREDUCIBLE) != 0

    def is_decision_out(self, i: int) -> bool:
        return (self.out_labels[i] & DECISION_MASK) == 0

    def is_decision_in(self, i: int) -> bool:
        return (self.in_labels[i] & DECISION_MASK) == 0

    def is_loop_dag_out(self, i: int) -> bool:
        """``FlowBlock::isLoopDAGOut`` (block.hh:354)."""
        return (self.out_labels[i] & LOOP_DAG_MASK) == 0

    def is_loop_dag_in(self, i: int) -> bool:
        """``FlowBlock::isLoopDAGIn`` (block.hh:357)."""
        return (self.in_labels[i] & LOOP_DAG_MASK) == 0

    def is_jump_target(self) -> bool:
        for point in self.in_edges:
            if point.index != self.index - 1:
                return True
        return False

    # -- dominancia y condiciones ------------------------------------------
    def dominates(self, sub_block: Optional["FlowBlock"]) -> bool:
        """``FlowBlock::dominates`` (block.cc:390)."""
        while sub_block is not None and self.index <= sub_block.index:
            if sub_block is self:
                return True
            sub_block = sub_block.immed_dom
        return False

    def restricted_by_conditional(self, cond: "FlowBlock") -> bool:
        """``FlowBlock::restrictedByConditional`` (block.cc:409)."""
        if len(self.in_edges) == 1:
            return True
        if self.immed_dom is not cond:
            return False
        seen_cond = False
        for in_block in self.in_edges:
            cur: Optional[FlowBlock] = in_block
            if cur is cond:
                if seen_cond:
                    return False
                seen_cond = True
                continue
            while cur is not self:
                if cur is cond:
                    return False
                if cur is None:
                    break
                cur = cur.immed_dom
        return True

    def get_front_leaf(self) -> Optional["FlowBlock"]:
        # Ghidra desciende hasta un BlockCopy; en el puerto las hojas son
        # BlockBasic, así que la bajada se detiene en cualquiera de los dos.
        bl: Optional[FlowBlock] = self
        while bl is not None and bl.get_type() not in ("copy", "basic"):
            bl = bl.sub_block(0)
        return bl

    def calc_depth(self, leaf: Optional["FlowBlock"]) -> int:
        depth = 0
        while leaf is not self:
            if leaf is None:
                return -1
            leaf = leaf.parent
            depth += 1
        return depth

    def negate_condition(self, toporbottom: bool = True) -> bool:
        """``FlowBlock::negateCondition``: invierte el orden de las salidas.

        Ghidra además reescribe el ``CBRANCH`` del bloque; los ``FlowBlock`` de
        MiniGhidraPy no llevan p-code, así que sólo se aplica el efecto
        estructural y se devuelve ``False`` (ningún cambio de dataflow).
        """
        if not toporbottom:
            return False
        self._swap_edges()
        return False

    # -- estáticos (block.cc:745-870) --------------------------------------
    @staticmethod
    def find_common_block(
        bl1: Optional["FlowBlock"], bl2: Optional["FlowBlock"]
    ) -> Optional["FlowBlock"]:
        """``FlowBlock::findCommonBlock(bl1,bl2)``."""
        common: Optional[FlowBlock] = None
        b1 = bl1
        b2 = bl2
        while True:
            if b2 is None:
                while b1 is not None:
                    if b1.is_mark():
                        common = b1
                        break
                    b1 = b1.immed_dom
                break
            if b1 is None:
                while b2 is not None:
                    if b2.is_mark():
                        common = b2
                        break
                    b2 = b2.immed_dom
                break
            if b1.is_mark():
                common = b1
                break
            b1.set_mark()
            if b2.is_mark():
                common = b2
                break
            b2.set_mark()
            b1 = b1.immed_dom
            b2 = b2.immed_dom
        cur = bl1
        while cur is not None:
            if not cur.is_mark():
                break
            cur.clear_mark()
            cur = cur.immed_dom
        cur = bl2
        while cur is not None:
            if not cur.is_mark():
                break
            cur.clear_mark()
            cur = cur.immed_dom
        return common

    @staticmethod
    def find_common_block_set(block_set: Sequence["FlowBlock"]) -> "FlowBlock":
        """``FlowBlock::findCommonBlock(blockSet)``."""
        marked_set: list[FlowBlock] = []
        res = block_set[0]
        best_index = res.index
        bl: Optional[FlowBlock] = res
        while True:
            assert bl is not None
            bl.set_mark()
            marked_set.append(bl)
            bl = bl.immed_dom
            if bl is None:
                break
        for i in range(1, len(block_set)):
            if best_index == 0:
                break
            bl = block_set[i]
            while not bl.is_mark():
                bl.set_mark()
                marked_set.append(bl)
                bl = bl.immed_dom
                if bl is None:
                    break
            if bl is None:
                continue
            if bl.index < best_index:
                res = bl
                best_index = res.index
        for marked in marked_set:
            marked.clear_mark()
        return res

    @staticmethod
    def find_condition(
        bl1: "FlowBlock", edge1: int, bl2: "FlowBlock", edge2: int
    ) -> tuple[Optional["FlowBlock"], int]:
        """``FlowBlock::findCondition``; devuelve ``(cond, slot1)``."""
        slot1 = 0
        cond = bl1.get_in(edge1)
        while len(cond.out_edges) != 2:
            if len(cond.out_edges) != 1:
                return None, 0
            bl1 = cond
            edge1 = 0
            cond = bl1.get_in(0)
        while cond is not bl2.get_in(edge2):
            bl2 = bl2.get_in(edge2)
            if len(bl2.out_edges) != 1:
                return None, 0
            edge2 = 0
        slot1 = bl1.get_in_rev_index(edge1)
        return cond, slot1

    @staticmethod
    def compare_block_index(bl1: "FlowBlock", bl2: "FlowBlock") -> bool:
        return bl1.index < bl2.index

    # -- compatibilidad con el API anterior del puerto ---------------------
    def add_out_edge(self, target: "FlowBlock") -> None:
        """Añade ``self -> target`` sin etiquetas (``BlockGraph::addEdge``)."""
        target._add_in_edge(self, 0)

    def remove_out_edge(self, index: int) -> None:
        self._remove_out_edge(index)

    def check_edges(self) -> None:
        """``FlowBlock::checkEdges`` (BLOCKCONSISTENT_DEBUG): consistencia."""
        for i, point in enumerate(self.in_edges):
            rev = self.in_rev[i]
            if rev >= len(point.out_edges):
                raise StructureError("Not enough outofthis blocks")
            if point.out_edges[rev] is not self:
                raise StructureError("Intothis edge mismatch")
            if point.out_rev[rev] != i:
                raise StructureError("Intothis index mismatch")
            if point.out_labels[rev] != self.in_labels[i]:
                raise StructureError("Intothis label mismatch")
        for i, point in enumerate(self.out_edges):
            rev = self.out_rev[i]
            if rev >= len(point.in_edges):
                raise StructureError("Not enough intothis blocks")
            if point.in_edges[rev] is not self:
                raise StructureError("Outofthis edge mismatch")
            if point.in_rev[rev] != i:
                raise StructureError("Outofthis index mismatch")
            if point.in_labels[rev] != self.out_labels[i]:
                raise StructureError("Outofthis label mismatch")


class BlockBasic(FlowBlock):
    """Bloque básico (``BlockBasic`` de Ghidra)."""

    def __init__(self, start_addr: Any = None) -> None:
        super().__init__("basic")
        self.start_addr = start_addr
        # Número de sentencias del bloque; sin p-code no se puede calcular como
        # hace BlockBasic::isComplex, se usa sólo como documentación del límite.
        self.statement_count = 1

    def get_start(self) -> Any:
        return self.start_addr

    def get_stop(self) -> Any:
        return self.start_addr

    def get_exit_leaf(self) -> Optional[FlowBlock]:
        return self

    def is_complex(self) -> bool:
        # Desviación: sin lista de p-code no hay sintaxis de desbordamiento.
        return False


class _BlockContainer(FlowBlock):
    """Contenedor de componentes (el ``list`` de un ``BlockGraph`` de Ghidra).

    Todas las estructuras (``BlockIf``, ``BlockWhileDo``, ...) son contenedores:
    guardan sus componentes en ``blocks`` y mantienen el ``parent`` de cada uno,
    exactamente igual que ``BlockGraph::addBlock``/``identifyInternal``.
    """

    def __init__(self, block_type: str) -> None:
        super().__init__(block_type)
        self.blocks: list[FlowBlock] = []

    def get_size(self) -> int:
        return len(self.blocks)

    def get_block(self, i: int) -> FlowBlock:
        return self.blocks[i]

    def sub_block(self, i: int) -> Optional[FlowBlock]:
        if 0 <= i < len(self.blocks):
            return self.blocks[i]
        return None

    def add_block(self, bl: FlowBlock) -> None:
        """``BlockGraph::addBlock``."""
        minimum = bl.index
        if not self.blocks:
            self.index = minimum
        elif minimum < self.index:
            self.index = minimum
        bl.parent = self
        self.blocks.append(bl)
        self.leaf_count += bl.get_basic_count()
        depth = bl.get_structure_depth() + 1
        if depth > self.structure_depth:
            self.structure_depth = depth

    def self_identify(self) -> None:
        """``BlockGraph::selfIdentify``: hereda las aristas que cruzan la frontera.

        Las etiquetas de las aristas se conservan porque se usa
        ``_replace_out_edge``/``_replace_in_edge`` (como en Ghidra), que copian
        el ``label`` de la arista original.
        """
        if not self.blocks:
            return
        for mybl in self.blocks:
            i = 0
            while i < len(mybl.in_edges):
                otherbl = mybl.in_edges[i]
                if otherbl.parent is self:
                    i += 1
                else:
                    for j in range(len(otherbl.out_edges)):
                        if otherbl.out_edges[j] is mybl:
                            otherbl._replace_out_edge(j, self)
            i = 0
            while i < len(mybl.out_edges):
                otherbl = mybl.out_edges[i]
                if otherbl.parent is self:
                    i += 1
                else:
                    for j in range(len(otherbl.in_edges)):
                        if otherbl.in_edges[j] is mybl:
                            otherbl._replace_in_edge(j, self)
                    if mybl.is_switch_out():
                        self._set_flag(F_SWITCH_OUT)
        self._dedup()

    def force_output_num(self, i: int) -> None:
        """``BlockGraph::forceOutputNum``: crea aristas hacia sí mismo."""
        while len(self.out_edges) < i:
            self._add_in_edge(self, F_LOOP_EDGE | F_BACK_EDGE)

    def force_false_edge(self, out0: Optional[FlowBlock]) -> None:
        """``BlockGraph::forceFalseEdge``."""
        if len(self.out_edges) != 2:
            raise StructureError("Can only preserve binary condition")
        target = out0
        if target is not None and target.parent is self:
            target = self
        if self.out_edges[0] is not target:
            self._swap_edges()
        if self.out_edges[0] is not target:
            raise StructureError("Unable to preserve condition")

    def get_start(self) -> Any:
        leaf = self.get_front_leaf()
        return leaf.get_start() if leaf is not None else None

    def get_exit_leaf(self) -> Optional[FlowBlock]:
        return None


class BlockList(_BlockContainer):
    """Secuencia de bloques (``BlockList``)."""

    def __init__(self, blocks: Optional[list[FlowBlock]] = None) -> None:
        super().__init__("list")
        if blocks:
            self.blocks = list(blocks)

    def get_exit_leaf(self) -> Optional[FlowBlock]:
        if not self.blocks:
            return None
        return self.blocks[-1].get_exit_leaf()


class BlockCondition(_BlockContainer):
    """Condición compuesta AND/OR (``BlockCondition``)."""

    def __init__(
        self, cond1: Optional[FlowBlock] = None, cond2: Optional[FlowBlock] = None,
        op: str = "&&",
    ) -> None:
        super().__init__("condition")
        self.op = op
        if cond1 is not None and cond2 is not None:
            self.blocks = [cond1, cond2]

    @property
    def cond1(self) -> FlowBlock:
        return self.blocks[0]

    @property
    def cond2(self) -> FlowBlock:
        return self.blocks[1]

    def get_opcode(self) -> str:
        return self.op

    def negate_condition(self, toporbottom: bool = True) -> bool:
        if not toporbottom:
            return False
        self.op = "||" if self.op == "&&" else "&&"
        self._swap_edges()
        return False

    def is_complex(self) -> bool:
        return self.blocks[0].is_complex() if self.blocks else True

    def get_exit_leaf(self) -> Optional[FlowBlock]:
        return self.blocks[-1].get_exit_leaf() if self.blocks else None


class BlockIf(_BlockContainer):
    """``if`` sin ``else`` (``BlockIf``): componentes ``[cond, true]``."""

    def __init__(
        self, cond_block: Optional[FlowBlock] = None,
        true_block: Optional[FlowBlock] = None,
    ) -> None:
        super().__init__("if")
        if cond_block is not None:
            self.blocks = [cond_block]
            if true_block is not None:
                self.blocks.append(true_block)

    @property
    def cond_block(self) -> FlowBlock:
        return self.blocks[0]

    @property
    def true_block(self) -> Optional[FlowBlock]:
        return self.blocks[1] if len(self.blocks) > 1 else None

    @property
    def false_block(self) -> Optional[FlowBlock]:
        return None

    def get_exit_leaf(self) -> Optional[FlowBlock]:
        return self.blocks[0].get_exit_leaf() if self.blocks else None


class BlockIfElse(BlockIf):
    """``if/else`` (``BlockIfElse``): componentes ``[cond, true, false]``."""

    def __init__(
        self,
        cond_block: Optional[FlowBlock] = None,
        true_block: Optional[FlowBlock] = None,
        false_block: Optional[FlowBlock] = None,
    ) -> None:
        super().__init__(cond_block, true_block)
        self.block_type = "ifelse"
        if false_block is not None:
            self.blocks.append(false_block)

    @property
    def false_block(self) -> Optional[FlowBlock]:  # type: ignore[override]
        return self.blocks[2] if len(self.blocks) > 2 else None


class BlockIfNoExit(_BlockContainer):
    """``if`` cuyo cuerpo no tiene salida (``BlockIfNoExit``)."""

    def __init__(
        self,
        cond_block: Optional[FlowBlock] = None,
        true_block: Optional[FlowBlock] = None,
        false_block: Optional[FlowBlock] = None,
    ) -> None:
        super().__init__("ifnoexit")
        if cond_block is not None:
            self.blocks = [cond_block]
            if true_block is not None:
                self.blocks.append(true_block)
            if false_block is not None:
                self.blocks.append(false_block)

    @property
    def cond_block(self) -> FlowBlock:
        return self.blocks[0]

    @property
    def true_block(self) -> Optional[FlowBlock]:
        return self.blocks[1] if len(self.blocks) > 1 else None

    @property
    def false_block(self) -> Optional[FlowBlock]:
        return self.blocks[2] if len(self.blocks) > 2 else None


class BlockIfGoto(BlockIf):
    """``if`` cuya rama verdadera es un goto (``BlockIfGoto``)."""

    def __init__(
        self,
        cond_block: Optional[FlowBlock] = None,
        target: Optional[FlowBlock] = None,
    ) -> None:
        super().__init__(cond_block)
        self.block_type = "ifgoto"
        self.gototype = F_GOTO_GOTO
        self.gototarget = target if target is not None else cond_block

    def get_goto_target(self) -> Optional[FlowBlock]:
        return self.gototarget

    def get_goto_type(self) -> int:
        return self.gototype

    def mark_unstructured(self) -> None:
        if self.gototarget is not None:
            self.gototarget._set_flag(F_UNSTRUCTURED_TARG)


class BlockWhileDo(_BlockContainer):
    """``while``/``for`` (``BlockWhileDo``): componentes ``[cond, body]``."""

    def __init__(
        self, cond_block: Optional[FlowBlock] = None,
        body_block: Optional[FlowBlock] = None,
    ) -> None:
        super().__init__("whiledo")
        if cond_block is not None:
            self.blocks = [cond_block]
            if body_block is not None:
                self.blocks.append(body_block)

    @property
    def cond_block(self) -> FlowBlock:
        return self.blocks[0]

    @property
    def body_block(self) -> FlowBlock:
        return self.blocks[1] if len(self.blocks) > 1 else self.blocks[0]

    def has_overflow_syntax(self) -> bool:
        return (self.flags & F_WHILEDO_OVERFLOW) != 0

    def set_overflow_syntax(self) -> None:
        self._set_flag(F_WHILEDO_OVERFLOW)

    def mark_label_bump_up(self, bump: bool) -> None:
        if bump:
            self._set_flag(F_LABEL_BUMPUP)
        elif self.blocks:
            self.blocks[0].mark_label_bump_up(False)


class BlockDoWhile(_BlockContainer):
    """``do/while`` (``BlockDoWhile``): un único componente condición+cuerpo."""

    def __init__(self, body_block: Optional[FlowBlock] = None) -> None:
        super().__init__("dowhile")
        if body_block is not None:
            self.blocks = [body_block]

    @property
    def body_block(self) -> FlowBlock:
        return self.blocks[0]

    def mark_label_bump_up(self, bump: bool) -> None:
        if bump:
            self._set_flag(F_LABEL_BUMPUP)
        elif self.blocks:
            self.blocks[0].mark_label_bump_up(False)


class BlockInfLoop(_BlockContainer):
    """Bucle sin salida (``BlockInfLoop``)."""

    def __init__(self, body_block: Optional[FlowBlock] = None) -> None:
        super().__init__("infloop")
        if body_block is not None:
            self.blocks = [body_block]

    @property
    def body_block(self) -> FlowBlock:
        return self.blocks[0]


class BlockSwitch(_BlockContainer):
    """``switch`` (``BlockSwitch``): ``cases[0]`` es el bloque raíz."""

    def __init__(
        self, cases: Optional[list[FlowBlock]] = None, has_exit: bool = False
    ) -> None:
        super().__init__("switch")
        self.has_exit = has_exit
        if cases:
            self.blocks = list(cases)

    @property
    def cases(self) -> list[FlowBlock]:
        return self.blocks

    def get_switch_block(self) -> FlowBlock:
        return self.blocks[0]

    def get_num_case_blocks(self) -> int:
        return len(self.blocks)

    def get_case_block(self, i: int) -> FlowBlock:
        return self.blocks[i]

    def mark_unstructured(self) -> None:
        for case in self.blocks[1:]:
            case._set_flag(F_UNSTRUCTURED_TARG)


class BlockGoto(_BlockContainer):
    """Bloque terminado en goto (``BlockGoto``)."""

    def __init__(self, target: Optional[FlowBlock] = None) -> None:
        super().__init__("goto")
        self.gototarget = target
        self.gototype = F_GOTO_GOTO

    def get_goto_target(self) -> Optional[FlowBlock]:
        return self.gototarget

    def get_goto_type(self) -> int:
        return self.gototype

    def goto_prints(self) -> bool:
        return True

    def mark_unstructured(self) -> None:
        if self.gototarget is not None:
            self.gototarget._set_flag(F_UNSTRUCTURED_TARG)

    def get_exit_leaf(self) -> Optional[FlowBlock]:
        return self.blocks[0].get_exit_leaf() if self.blocks else None


class BlockMultiGoto(BlockGoto):
    """Bloque con varias salidas no estructuradas (``BlockMultiGoto``)."""

    def __init__(self, target: Optional[FlowBlock] = None) -> None:
        # Ghidra: BlockMultiGoto(FlowBlock *bl) { defaultswitch = false; } -- el
        # bloque envuelto llega por identifyInternal, el ctor lo ignora.
        self.gotoedges: list[FlowBlock] = []
        self.defaultswitch = False
        _BlockContainer.__init__(self, "multigoto")

    def set_default_goto(self) -> None:
        self.defaultswitch = True

    def has_default_goto(self) -> bool:
        return self.defaultswitch

    def add_edge(self, bl: FlowBlock) -> None:
        """``BlockMultiGoto::addEdge``: registra un destino no estructurado."""
        self.gotoedges.append(bl)

    def num_gotos(self) -> int:
        return len(self.gotoedges)

    def get_goto(self, i: int) -> FlowBlock:
        return self.gotoedges[i]

    @property
    def gototarget(self) -> Optional[FlowBlock]:  # type: ignore[override]
        return self.gotoedges[0] if self.gotoedges else None


class BlockGraph(_BlockContainer):
    """Grafo de bloques (``BlockGraph`` de Ghidra)."""

    def __init__(self) -> None:
        super().__init__("graph")
        self.edge_size = 0

    # -- utilidades de grafo (block.cc:871-1250) ---------------------------
    def identify_internal(self, ident: _BlockContainer, nodes: list[FlowBlock]) -> None:
        """``BlockGraph::identifyInternal``: mueve ``nodes`` dentro de ``ident``."""
        for node in nodes:
            node.set_mark()
            ident.add_block(node)
            ident.flags |= node.flags & (F_INTERIOR_GOTOOUT | F_INTERIOR_GOTOIN)
        newlist = []
        for bl in self.blocks:
            if not bl.is_mark():
                newlist.append(bl)
            else:
                bl.clear_mark()
        self.blocks = newlist
        ident.self_identify()

    def clear_edge_flags(self, fl: int) -> None:
        """``BlockGraph::clearEdgeFlags`` sobre todos los bloques del grafo."""
        mask = ~fl
        for bl in self.blocks:
            for i in range(len(bl.in_labels)):
                bl.in_labels[i] &= mask
                other = bl.in_edges[i]
                other.out_labels[bl.in_rev[i]] &= mask
            for i in range(len(bl.out_labels)):
                bl.out_labels[i] &= mask
                other = bl.out_edges[i]
                other.in_labels[bl.out_rev[i]] &= mask

    def create_virtual_root(self, rootlist: list[FlowBlock]) -> FlowBlock:
        """``BlockGraph::createVirtualRoot``."""
        newroot = FlowBlock("plain")
        for root in rootlist:
            root._add_in_edge(newroot, 0)
        return newroot

    def find_spanning_tree(
        self, preorder: list[FlowBlock], rootlist: list[FlowBlock]
    ) -> None:
        """``BlockGraph::findSpanningTree`` (Tarjan).

        Etiqueta aristas de árbol/directas/cruzadas/vuelta, calcula ``numdesc``,
        asigna ``index`` en orden post-inverso y reordena ``blocks``.
        """
        if not self.blocks:
            return
        rpostorder: list[Optional[FlowBlock]] = [None] * len(self.blocks)
        state: list[FlowBlock] = []
        istate: list[int] = []
        origrootpos = 0
        for tmpbl in self.blocks:
            tmpbl.index = -1
            tmpbl.visit_count = -1
            tmpbl.copymap = tmpbl
            if len(tmpbl.in_edges) == 0:
                rootlist.append(tmpbl)
        if len(rootlist) > 1:
            rootlist[-1], rootlist[0] = rootlist[0], rootlist[-1]
        elif len(rootlist) == 0:
            rootlist.append(self.blocks[0])
        origrootpos = len(rootlist) - 1

        for repeat in range(2):
            extraroots = False
            rpostcount = len(self.blocks)
            rootindex = 0
            self.clear_edge_flags(
                F_IRREDUCIBLE | F_TREE_EDGE | F_FORWARD_EDGE | F_CROSS_EDGE
                | F_BACK_EDGE | F_LOOP_EDGE | F_LOOP_EXIT_EDGE
            )
            while len(preorder) < len(self.blocks):
                startbl: Optional[FlowBlock] = None
                while rootindex < len(rootlist):
                    startbl = rootlist[rootindex]
                    rootindex += 1
                    if startbl.visit_count == -1:
                        break
                    for i in range(rootindex, len(rootlist)):
                        rootlist[i - 1] = rootlist[i]
                    rootlist.pop()
                    rootindex -= 1
                    startbl = None
                if startbl is None:
                    extraroots = True
                    startbl = self.blocks[0]
                    for candidate in self.blocks:
                        if candidate.visit_count == -1:
                            startbl = candidate
                            break
                    rootlist.append(startbl)
                    rootindex += 1

                state.append(startbl)
                istate.append(0)
                startbl.visit_count = len(preorder)
                preorder.append(startbl)
                startbl.numdesc = 1

                while state:
                    curbl = state[-1]
                    if len(curbl.out_edges) <= istate[-1]:
                        state.pop()
                        istate.pop()
                        rpostcount -= 1
                        curbl.index = rpostcount
                        rpostorder[rpostcount] = curbl
                        if state:
                            state[-1].numdesc += curbl.numdesc
                    else:
                        edgenum = istate[-1]
                        istate[-1] += 1
                        if curbl.is_irreducible_out(edgenum):
                            continue
                        childbl = curbl.get_out(edgenum)
                        if childbl.visit_count == -1:
                            curbl._set_out_edge_flag(edgenum, F_TREE_EDGE)
                            state.append(childbl)
                            istate.append(0)
                            childbl.visit_count = len(preorder)
                            preorder.append(childbl)
                            childbl.numdesc = 1
                        elif childbl.index == -1:
                            curbl._set_out_edge_flag(edgenum, F_BACK_EDGE | F_LOOP_EDGE)
                        elif curbl.visit_count < childbl.visit_count:
                            curbl._set_out_edge_flag(edgenum, F_FORWARD_EDGE)
                        else:
                            curbl._set_out_edge_flag(edgenum, F_CROSS_EDGE)
            if not extraroots:
                break
            if repeat == 1:
                raise StructureError("Could not generate spanning tree")
            rootlist[-1], rootlist[origrootpos] = (
                rootlist[origrootpos], rootlist[-1]
            )
            for tmpbl in self.blocks:
                tmpbl.index = -1
                tmpbl.visit_count = -1
                tmpbl.copymap = tmpbl
            preorder.clear()
            state.clear()
            istate.clear()

        if len(rootlist) > 1:
            rootlist[-1], rootlist[0] = rootlist[0], rootlist[-1]

        self.blocks = [bl for bl in rpostorder if bl is not None]

    def find_irreducible(
        self, preorder: list[FlowBlock], irreduciblecount: int = 0
    ) -> tuple[bool, int]:
        """``BlockGraph::findIrreducible``; devuelve ``(needrebuild, count)``."""
        reachunder: list[FlowBlock] = []
        needrebuild = False
        xi = len(preorder) - 1
        while xi >= 0:
            x = preorder[xi]
            xi -= 1
            for i in range(len(x.in_edges)):
                if not x.is_back_edge_in(i):
                    continue
                y = x.get_in(i)
                if y is x:
                    continue
                reachunder.append(y.copymap)
                y.copymap.set_mark()
            q = 0
            while q < len(reachunder):
                t = reachunder[q]
                q += 1
                for i in range(len(t.in_edges)):
                    if t.is_irreducible_in(i):
                        continue
                    y = t.get_in(i)
                    yprime = y.copymap
                    if (x.visit_count > yprime.visit_count) or (
                        x.visit_count + x.numdesc <= yprime.visit_count
                    ):
                        irreduciblecount += 1
                        edgeout = t.get_in_rev_index(i)
                        y._set_out_edge_flag(edgeout, F_IRREDUCIBLE)
                        if t.is_tree_edge_in(i):
                            needrebuild = True
                        else:
                            y._clear_out_edge_flag(edgeout, F_CROSS_EDGE | F_FORWARD_EDGE)
                    elif (not yprime.is_mark()) and (yprime is not x):
                        reachunder.append(yprime)
                        yprime.set_mark()
            for s in reachunder:
                s.clear_mark()
                s.copymap = x
            reachunder.clear()
        return needrebuild, irreduciblecount

    def calc_loop(self) -> None:
        """``BlockGraph::calcLoop``: marca aristas de bucle restantes."""
        if not self.blocks:
            return
        path: list[FlowBlock] = [self.blocks[0]]
        state: list[int] = [0]
        self.blocks[0]._set_flag(F_MARK | F_MARK2)
        while path:
            bl = path[-1]
            i = state[-1]
            if i >= len(bl.out_edges):
                bl._clear_flag(F_MARK2)
                path.pop()
                state.pop()
            else:
                state[-1] += 1
                if bl.is_loop_out(i):
                    continue
                nextbl = bl.get_out(i)
                if nextbl.flags & F_MARK2:
                    self.add_loop_edge(bl, i)
                elif not (nextbl.flags & F_MARK):
                    nextbl._set_flag(F_MARK | F_MARK2)
                    path.append(nextbl)
                    state.append(0)
        for bl in self.blocks:
            bl._clear_flag(F_MARK | F_MARK2)

    def structure_loops(self, rootlist: Optional[list[FlowBlock]] = None) -> list[FlowBlock]:
        """``BlockGraph::structureLoops``; devuelve la lista de raíces."""
        if rootlist is None:
            rootlist = []
        preorder: list[FlowBlock] = []
        irreduciblecount = 0
        while True:
            self.find_spanning_tree(preorder, rootlist)
            needrebuild, irreduciblecount = self.find_irreducible(
                preorder, irreduciblecount
            )
            if not needrebuild:
                break
            self.clear_edge_flags(
                F_TREE_EDGE | F_FORWARD_EDGE | F_CROSS_EDGE | F_BACK_EDGE | F_LOOP_EDGE
            )
            preorder.clear()
            rootlist.clear()
        if irreduciblecount > 0:
            self.calc_loop()
        return rootlist

    def swap_blocks(self, i: int, j: int) -> None:
        self.blocks[i], self.blocks[j] = self.blocks[j], self.blocks[i]

    def mark_copy_block(self, bl: FlowBlock, fl: int) -> None:
        leaf = bl.get_front_leaf()
        if leaf is not None:
            leaf._set_flag(fl)

    def clear_visit_count(self) -> None:
        for bl in self.blocks:
            bl.visit_count = 0

    # -- aristas a nivel de grafo (block.cc:1444-1560) ---------------------
    def add_edge(self, begin: FlowBlock, end: FlowBlock) -> None:
        end._add_in_edge(begin, 0)

    def add_loop_edge(self, begin: FlowBlock, outindex: int) -> None:
        begin._set_out_edge_flag(outindex, F_LOOP_EDGE)

    def remove_edge(self, begin: FlowBlock, end: FlowBlock) -> None:
        for i, point in enumerate(end.in_edges):
            if point is begin:
                end._remove_in_edge(i)
                return

    def switch_edge(
        self, in_: FlowBlock, outbefore: FlowBlock, outafter: FlowBlock
    ) -> None:
        for i, point in enumerate(in_.out_edges):
            if point is outbefore:
                in_._replace_out_edge(i, outafter)

    def move_out_edge(self, blold: FlowBlock, slot: int, blnew: FlowBlock) -> None:
        outbl = blold.get_out(slot)
        i = blold.get_out_rev_index(slot)
        outbl._replace_in_edge(i, blnew)

    def remove_block(self, bl: FlowBlock) -> None:
        while len(bl.in_edges) > 0:
            self.remove_edge(bl.get_in(0), bl)
        while len(bl.out_edges) > 0:
            self.remove_edge(bl, bl.get_out(0))
        for i, candidate in enumerate(self.blocks):
            if candidate is bl:
                del self.blocks[i]
                break

    # -- dominadores (block.cc:1972-2060) ----------------------------------
    def calc_forward_dominator(self, rootlist: list[FlowBlock]) -> None:
        """``BlockGraph::calcForwardDominator`` (Cooper, Harvey y Kennedy).

        Asume que ``blocks`` está en orden post-inverso y que ``index`` lo
        refleja (lo establece ``find_spanning_tree``).
        """
        if not self.blocks:
            return
        numnodes = len(self.blocks) - 1
        postorder: list[FlowBlock] = [None] * len(self.blocks)  # type: ignore[list-item]
        for i, bl in enumerate(self.blocks):
            bl.immed_dom = None
            postorder[numnodes - i] = bl
        virtualroot: Optional[FlowBlock] = None
        if len(rootlist) > 1:
            virtualroot = self.create_virtual_root(rootlist)
            postorder.append(virtualroot)

        b = postorder[-1]
        if len(b.in_edges) != 0:
            if len(rootlist) != 1 or rootlist[0] is not b:
                raise StructureError("Problems finding root node of graph")
            virtualroot = self.create_virtual_root(rootlist)
            postorder.append(virtualroot)
            b = virtualroot
        b.immed_dom = b
        for i in range(len(b.out_edges)):
            b.get_out(i).immed_dom = b
        changed = True
        new_idom: Optional[FlowBlock] = None
        while changed:
            changed = False
            for i in range(len(postorder) - 2, -1, -1):
                b = postorder[i]
                if b.immed_dom is postorder[-1]:
                    continue
                first_proc = -1
                for j in range(len(b.in_edges)):
                    new_idom = b.get_in(j)
                    if new_idom.immed_dom is not None:
                        first_proc = j
                        break
                if first_proc == -1:
                    continue
                for j in range(first_proc + 1, len(b.in_edges)):
                    rho = b.get_in(j)
                    if rho.immed_dom is not None:
                        assert new_idom is not None
                        finger1 = numnodes - rho.index
                        finger2 = numnodes - new_idom.index
                        while finger1 != finger2:
                            while finger1 < finger2:
                                dom = postorder[finger1].immed_dom
                                if dom is None:
                                    break
                                finger1 = numnodes - dom.index
                            while finger2 < finger1:
                                dom = postorder[finger2].immed_dom
                                if dom is None:
                                    break
                                finger2 = numnodes - dom.index
                        new_idom = postorder[finger1]
                if b.immed_dom is not new_idom:
                    b.immed_dom = new_idom
                    changed = True

        if virtualroot is not None:
            for i in range(len(postorder)):
                if postorder[i].immed_dom is virtualroot:
                    postorder[i].immed_dom = None
            while len(virtualroot.out_edges) > 0:
                virtualroot._remove_out_edge(len(virtualroot.out_edges) - 1)
        else:
            postorder[-1].immed_dom = None

    def build_dom_tree(self) -> list[list[FlowBlock]]:
        """``BlockGraph::buildDomTree``: hijos por índice de dominador."""
        child: list[list[FlowBlock]] = [[] for _ in range(len(self.blocks) + 1)]
        for bl in self.blocks:
            if bl.immed_dom is not None:
                child[self.blocks.index(bl.immed_dom)].append(bl)
            else:
                child[len(self.blocks)].append(bl)
        return child

    # -- construcción de estructuras (block.cc:1664-1943) ------------------
    def new_block(self) -> FlowBlock:
        ret = FlowBlock("plain")
        self.add_block(ret)
        return ret

    def new_block_basic(self, start_addr: Any = None) -> BlockBasic:
        ret = BlockBasic(start_addr)
        self.add_block(ret)
        return ret

    def new_block_goto(self, bl: FlowBlock) -> BlockGoto:
        """``BlockGraph::newBlockGoto``."""
        ret = BlockGoto(bl.get_out(0))
        self.identify_internal(ret, [bl])
        self.add_block(ret)
        ret.force_output_num(1)
        self.remove_edge(ret, ret.get_out(0))
        return ret

    def new_block_multigoto(self, bl: FlowBlock, outedge: int) -> BlockMultiGoto:
        """``BlockGraph::newBlockMultiGoto``."""
        targetbl = bl.get_out(outedge)
        isdefaultedge = bl.is_default_branch(outedge)
        if bl.get_type() == "multigoto":
            ret = bl  # type: ignore[assignment]
            ret.add_edge(targetbl)
            self.remove_edge(ret, targetbl)
            if isdefaultedge:
                ret.set_default_goto()
            return ret
        ret = BlockMultiGoto(bl)
        orig_size_out = bl.size_out()
        self.identify_internal(ret, [bl])
        self.add_block(ret)
        ret.add_edge(targetbl)
        if targetbl is not bl:
            if ret.size_out() != orig_size_out:
                ret.force_output_num(ret.size_out() + 1)
            self.remove_edge(ret, targetbl)
        if isdefaultedge:
            ret.set_default_goto()
        return ret

    def new_block_list(self, nodes: list[FlowBlock]) -> BlockList:
        """``BlockGraph::newBlockList``."""
        out0 = None
        outforce = nodes[-1].size_out()
        if outforce == 2:
            out0 = nodes[-1].get_out(0)
        ret = BlockList()
        self.identify_internal(ret, nodes)
        self.add_block(ret)
        ret.force_output_num(outforce)
        if ret.size_out() == 2:
            ret.force_false_edge(out0)
        return ret

    def new_block_condition(self, b1: FlowBlock, b2: FlowBlock, op: Optional[str] = None) -> BlockCondition:
        """``BlockGraph::newBlockCondition`` (el operador se deriva de las aristas)."""
        out0 = b2.get_out(0)
        derived_op = "||" if b1.get_false_out() is b2 else "&&"
        ret = BlockCondition(op=op if op is not None else derived_op)
        self.identify_internal(ret, [b1, b2])
        self.add_block(ret)
        ret.force_output_num(2)
        ret.force_false_edge(out0)
        return ret

    def new_block_if_goto(self, cond: FlowBlock) -> BlockIfGoto:
        """``BlockGraph::newBlockIfGoto``."""
        if not cond.is_goto_out(1):
            raise StructureError("Building ifgoto where true branch is not the goto")
        out0 = cond.get_out(0)
        ret = BlockIfGoto(None, cond.get_out(1))
        self.identify_internal(ret, [cond])
        self.add_block(ret)
        ret.force_output_num(2)
        ret.force_false_edge(out0)
        self.remove_edge(ret, ret.get_true_out())
        return ret

    def new_block_if(self, cond: FlowBlock, tc: FlowBlock) -> BlockIf:
        ret = BlockIf()
        self.identify_internal(ret, [cond, tc])
        self.add_block(ret)
        ret.force_output_num(1)
        return ret

    def new_block_if_else(self, cond: FlowBlock, tc: FlowBlock, fc: FlowBlock) -> BlockIfElse:
        ret = BlockIfElse()
        self.identify_internal(ret, [cond, tc, fc])
        self.add_block(ret)
        ret.force_output_num(1)
        return ret

    def new_block_if_no_exit(
        self, cond: FlowBlock, tc: FlowBlock, fc: Optional[FlowBlock] = None
    ) -> BlockIfNoExit:
        nodes = [cond, tc]
        if fc is not None:
            nodes.append(fc)
        ret = BlockIfNoExit()
        self.identify_internal(ret, nodes)
        self.add_block(ret)
        return ret

    def new_block_while_do(self, cond: FlowBlock, cl: FlowBlock) -> BlockWhileDo:
        ret = BlockWhileDo()
        self.identify_internal(ret, [cond, cl])
        self.add_block(ret)
        ret.force_output_num(1)
        return ret

    def new_block_do_while(self, condcl: FlowBlock) -> BlockDoWhile:
        ret = BlockDoWhile()
        self.identify_internal(ret, [condcl])
        self.add_block(ret)
        ret.force_output_num(1)
        return ret

    def new_block_inf_loop(self, body: FlowBlock) -> BlockInfLoop:
        ret = BlockInfLoop()
        self.identify_internal(ret, [body])
        self.add_block(ret)
        return ret

    def new_block_switch(self, cs: list[FlowBlock], has_exit: bool) -> BlockSwitch:
        """``BlockGraph::newBlockSwitch`` (sin ``grabCaseBasic``: no hay JumpTable)."""
        ret = BlockSwitch(has_exit=has_exit)
        self.identify_internal(ret, cs)
        self.add_block(ret)
        if has_exit:
            ret.force_output_num(1)
        ret._clear_flag(F_SWITCH_OUT)
        return ret

    # -- copia del grafo (block.cc:1943-1971) ------------------------------
    def build_copy(self, blocks_data: Any) -> None:
        """Construye el grafo a partir de bloques planos o de otro ``BlockGraph``.

        Además aplica el etiquetado que Ghidra realiza en
        ``Funcdata::structureReset`` (``structureLoops`` + ``calcForwardDominator``)
        sobre las hojas, de modo que la estructuración dispone de índices en
        orden post-inverso, aristas de vuelta/irreducibles y dominadores.
        """
        self.blocks = []
        if isinstance(blocks_data, BlockGraph):
            self._build_copy_graph(blocks_data)
        else:
            self._build_copy_raw(blocks_data)
        self.edge_size = sum(len(bl.out_edges) for bl in self.blocks)
        rootlist = self.structure_loops()
        self.calc_forward_dominator(rootlist)
        self.clear_visit_count()
        if rootlist:
            rootlist[0]._set_flag(F_ENTRY_POINT)
        self.edge_size = sum(len(bl.out_edges) for bl in self.blocks)

    def _build_copy_raw(self, blocks_data: Sequence[Any]) -> None:
        block_map: dict[Any, BlockBasic] = {}
        for raw in blocks_data:
            start = getattr(raw, "start", None)
            bb = BlockBasic(start)
            bb.name = f"block_{hex(start)}" if isinstance(start, int) else ""
            instructions = getattr(raw, "instructions", None)
            if instructions is not None:
                bb.statement_count = max(1, len(instructions))
            bb.leaf_count = 1
            self.blocks.append(bb)
            bb.parent = self
            block_map[start] = bb
        for i, bl in enumerate(self.blocks):
            bl.index = i
        for raw in blocks_data:
            bb = block_map[getattr(raw, "start", None)]
            wired: set[int] = set()
            for succ in self._ordered_successors(raw):
                target = block_map.get(succ)
                if target is None or id(target) in wired:
                    # El CFG de Ghidra no tiene aristas paralelas entre dos
                    # bloques basicos: se deduplican por seguridad (una tabla
                    # de saltos puede repetir destino).
                    continue
                wired.add(id(target))
                self.add_edge(bb, target)
        for raw in blocks_data:
            # Ghidra activa f_switch_out cuando el bloque termina en un
            # BRANCHIND (tabla de saltos); aquí el equivalente es un salto
            # indirecto con destinos calculados, o simplemente más de dos
            # salidas, que en un CFG real sólo puede venir de un switch.
            bb = block_map[getattr(raw, "start", None)]
            instructions = getattr(raw, "instructions", None)
            last = instructions[-1] if instructions else None
            computed = bool(getattr(last, "computed_targets", ()) or ())
            if computed or bb.size_out() > 2:
                bb._set_flag(F_SWITCH_OUT)

    @staticmethod
    def _ordered_successors(raw: Any) -> list[Any]:
        """Sucesores en la convención de Ghidra: ``outofthis[0]`` = rama falsa.

        ``Analyzer._build_blocks`` genera los sucesores de un salto condicional
        como ``(objetivo_del_salto, caida)``; Ghidra, en cambio, documenta que
        "the first block in outofthis should be the result of the condition
        being false" (block.hh).  Aquí se reordena ese par para que
        ``get_false_out()``/``get_true_out()`` sean correctos.
        """
        succs = list(getattr(raw, "successors", ()) or ())
        instructions = getattr(raw, "instructions", None)
        if len(succs) == 2 and instructions:
            last = instructions[-1]
            if getattr(last, "is_conditional_jump", False):
                target = getattr(last, "branch_target", None)
                fallthrough = getattr(last, "next_address", None)
                if (
                    target is not None
                    and fallthrough is not None
                    and target != fallthrough
                    and target in succs
                    and fallthrough in succs
                ):
                    return [fallthrough, target]
        return succs

    def _build_copy_graph(self, graph: "BlockGraph") -> None:
        for bl in graph.blocks:
            copy = BlockBasic(bl.get_start())
            copy.in_edges = list(bl.in_edges)
            copy.in_labels = list(bl.in_labels)
            copy.in_rev = list(bl.in_rev)
            copy.out_edges = list(bl.out_edges)
            copy.out_labels = list(bl.out_labels)
            copy.out_rev = list(bl.out_rev)
            copy.immed_dom = bl.immed_dom
            copy.index = bl.index
            copy.numdesc = bl.numdesc
            copy.flags |= bl.flags
            copy.leaf_count = bl.get_basic_count()
            copy.statement_count = getattr(bl, "statement_count", 1)
            if len(copy.out_edges) > 2:
                copy.flags |= F_SWITCH_OUT
            bl.copymap = copy
            self.blocks.append(copy)
            copy.parent = self
        for copy in self.blocks:
            copy._replace_using_map()
