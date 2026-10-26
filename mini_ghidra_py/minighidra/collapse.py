"""Puerto de ``LoopBody`` y ``CollapseStructure`` (``blockaction.cc`` de Ghidra).

Correspondencia con el C++:

    LoopBody::extendToContainer   -> blockaction.cc:46
    LoopBody::update              -> :94
    LoopBody::findBase            -> :119
    LoopBody::extend              -> :150
    LoopBody::findExit            -> :182
    LoopBody::orderTails          -> :245
    LoopBody::labelExitEdges      -> :270
    LoopBody::labelContainments   -> :327
    LoopBody::emitLikelyEdges     -> :364
    LoopBody::setExitMarks        -> :416
    LoopBody::clearExitMarks      -> :430
    LoopBody::mergeIdenticalHeads -> :446
    LoopBody::compare_ends        -> :473
    LoopBody::compare_head        -> :489
    LoopBody::find                -> :1021
    LoopBody::clearMarks          -> :1039
    CollapseStructure::onlyReachableFromRoot -> :1052
    CollapseStructure::markExitsAsGotos      -> :1083
    CollapseStructure::clipExtraRoots        -> :1108
    CollapseStructure::labelLoops            -> :1126
    CollapseStructure::orderLoopBodies       -> :1148
    CollapseStructure::updateLoopBody        -> :1193
    CollapseStructure::selectGoto            -> :1260
    CollapseStructure::ruleBlock*            -> :1284-1731
    CollapseStructure::collapseInternal      -> :1780
    CollapseStructure::collapseConditions    -> :1848
    CollapseStructure::collapseAll           -> :1875

Desviaciones documentadas (ver ``PORTING_NOTES.md``):

* Ghidra usa ``list<...>::iterator`` para ``likelyiter`` y ``loopbodyiter``; aquí
  son índices enteros sobre las listas equivalentes.
* ``negateCondition`` no reescribe el ``CBRANCH`` (los ``FlowBlock`` del puerto
  no llevan p-code): sólo invierte el orden de las aristas y no contabiliza
  cambios de dataflow, así que ``getChangeCount()`` refleja sólo los cambios que
  sí existen en este modelo.
* ``ruleBlockSwitch`` no construye ``CaseOrder``/``JumpTable`` (no hay tabla de
  saltos en el modelo de bloques); el impresor reconstruye las etiquetas.
* ``collapseAll`` y ``collapseInternal`` tienen un presupuesto de pasos: si se
  agota lanzan ``StructureError`` en lugar de colgar.  Ghidra lanza
  ``LowlevelError`` cuando ``selectGoto`` no encuentra salida; aquí ese caso
  también se convierte en ``StructureError`` para que el decompilador pueda
  volver a la impresión plana con ``goto``.
"""

from __future__ import annotations

from typing import Optional

from .block import (
    BlockGraph,
    FlowBlock,
    StructureError,
)
from .tracedag import FloatingEdge, TraceDAG, push_budget

# Presupuesto de pasos de los bucles de estructuración (defensa anti-cuelgue).
_STEPS_PER_BLOCK = 64
_STEPS_MIN = 4096


def _budget(graph: BlockGraph) -> int:
    edges = sum(len(bl.out_edges) for bl in graph.blocks)
    return _STEPS_MIN + _STEPS_PER_BLOCK * (len(graph.blocks) + edges)


class LoopBody:
    """Cuerpo de bucle según Tarjan: ``head`` + una o más ``tails``."""

    def __init__(self, h: Optional[FlowBlock]) -> None:
        self.head: Optional[FlowBlock] = h
        self.tails: list[FlowBlock] = []
        self.depth = 0
        self.uniquecount = 0
        self.exitblock: Optional[FlowBlock] = None
        self.exitedges: list[FloatingEdge] = []
        self.immed_container: Optional[LoopBody] = None

    def get_head(self) -> Optional[FlowBlock]:
        return self.head

    def get_exit_block(self) -> Optional[FlowBlock]:
        return self.exitblock

    def add_tail(self, bl: FlowBlock) -> None:
        self.tails.append(bl)

    def __lt__(self, op2: "LoopBody") -> bool:
        # Ghidra: operator< ordena por profundidad decreciente.
        return self.depth > op2.depth

    def extend_to_container(self, container: "LoopBody", body: list[FlowBlock]) -> None:
        """``LoopBody::extendToContainer`` (blockaction.cc:46)."""
        head = self.head
        i = 0
        if container.head is not None and not container.head.is_mark():
            container.head.set_mark()
            body.append(container.head)
            i = 1
        for tail in container.tails:
            if not tail.is_mark():
                tail.set_mark()
                body.append(tail)
        if head is not None and container.head is not None and head is not container.head:
            for k in range(head.size_in()):
                if head.is_goto_in(k):
                    continue
                bl = head.get_in(k)
                if bl.is_mark():
                    continue
                bl.set_mark()
                body.append(bl)
        while i < len(body):
            curblock = body[i]
            i += 1
            for k in range(curblock.size_in()):
                if curblock.is_goto_in(k):
                    continue
                bl = curblock.get_in(k)
                if bl.is_mark():
                    continue
                bl.set_mark()
                body.append(bl)

    def update(self, graph: FlowBlock) -> Optional[FlowBlock]:
        """``LoopBody::update`` (blockaction.cc:94)."""
        head = self.head
        if head is None:
            return None
        while head.parent is not graph:
            head = head.parent
            if head is None:
                raise StructureError("LoopBody.update lost the graph")
        self.head = head
        for i, bottom in enumerate(self.tails):
            while bottom.parent is not graph:
                bottom = bottom.parent
                if bottom is None:
                    raise StructureError("LoopBody.update lost the graph")
            self.tails[i] = bottom
            if bottom is not head:
                return bottom
        for i in range(head.size_out() - 1, -1, -1):
            if head.get_out(i) is head:
                return head
        return None

    def find_base(self, body: list[FlowBlock]) -> None:
        """``LoopBody::findBase`` (blockaction.cc:119)."""
        head = self.head
        assert head is not None
        head.set_mark()
        body.append(head)
        for tail in self.tails:
            if not tail.is_mark():
                tail.set_mark()
                body.append(tail)
        self.uniquecount = len(body)
        i = 1
        while i < len(body):
            curblock = body[i]
            i += 1
            for k in range(curblock.size_in()):
                if curblock.is_goto_in(k):
                    continue
                bl = curblock.get_in(k)
                if bl.is_mark():
                    continue
                bl.set_mark()
                body.append(bl)

    def extend(self, body: list[FlowBlock]) -> None:
        """``LoopBody::extend`` (blockaction.cc:150)."""
        trial: list[FlowBlock] = []
        i = 0
        while i < len(body):
            bl = body[i]
            i += 1
            for j in range(bl.size_out()):
                if bl.is_goto_out(j):
                    continue
                curbl = bl.get_out(j)
                if curbl.is_mark():
                    continue
                if curbl is self.exitblock:
                    continue
                count = curbl.get_visit_count()
                if count == 0:
                    trial.append(curbl)
                count += 1
                curbl.set_visit_count(count)
                if count == curbl.size_in():
                    curbl.set_mark()
                    body.append(curbl)
        for bl in trial:
            bl.set_visit_count(0)

    def find_exit(self, body: list[FlowBlock]) -> None:
        """``LoopBody::findExit`` (blockaction.cc:182)."""
        trialexit: list[FlowBlock] = []
        for tail in self.tails:
            for i in range(tail.size_out()):
                if tail.is_goto_out(i):
                    continue
                curbl = tail.get_out(i)
                if not curbl.is_mark():
                    if self.immed_container is None:
                        self.exitblock = curbl
                        return
                    trialexit.append(curbl)

        for i, bl in enumerate(body):
            if 0 < i < self.uniquecount:
                continue  # Descarta las tails (ya procesadas)
            for j in range(bl.size_out()):
                if bl.is_goto_out(j):
                    continue
                curbl = bl.get_out(j)
                if not curbl.is_mark():
                    if self.immed_container is None:
                        self.exitblock = curbl
                        return
                    trialexit.append(curbl)

        self.exitblock = None
        if not trialexit:
            return
        if self.immed_container is not None:
            extension: list[FlowBlock] = []
            self.extend_to_container(self.immed_container, extension)
            for bl in trialexit:
                if bl.is_mark():
                    self.exitblock = bl
                    break
            LoopBody.clear_marks(extension)

    def order_tails(self) -> None:
        """``LoopBody::orderTails`` (blockaction.cc:245)."""
        if len(self.tails) <= 1:
            return
        if self.exitblock is None:
            return
        prefindex = 0
        trial: Optional[FlowBlock] = None
        while prefindex < len(self.tails):
            trial = self.tails[prefindex]
            j = 0
            sizeout = trial.size_out()
            while j < sizeout:
                if trial.get_out(j) is self.exitblock:
                    break
                j += 1
            if j < sizeout:
                break
            prefindex += 1
        if prefindex >= len(self.tails):
            return
        if prefindex == 0:
            return
        self.tails[prefindex] = self.tails[0]
        self.tails[0] = trial

    def label_exit_edges(self, body: list[FlowBlock]) -> None:
        """``LoopBody::labelExitEdges`` (blockaction.cc:270)."""
        toexitblock: list[FlowBlock] = []
        for i in range(self.uniquecount, len(body)):
            curblock = body[i]
            for k in range(curblock.size_out()):
                if curblock.is_goto_out(k):
                    continue
                bl = curblock.get_out(k)
                if bl is self.exitblock:
                    toexitblock.append(curblock)
                    continue
                if not bl.is_mark():
                    self.exitedges.append(FloatingEdge(curblock, bl))
        head = self.head
        if head is not None:
            for k in range(head.size_out()):
                if head.is_goto_out(k):
                    continue
                bl = head.get_out(k)
                if bl is self.exitblock:
                    toexitblock.append(head)
                    continue
                if not bl.is_mark():
                    self.exitedges.append(FloatingEdge(head, bl))
        for i in range(len(self.tails) - 1, -1, -1):
            curblock = self.tails[i]
            if curblock is head:
                continue
            for k in range(curblock.size_out()):
                if curblock.is_goto_out(k):
                    continue
                bl = curblock.get_out(k)
                if bl is self.exitblock:
                    toexitblock.append(curblock)
                    continue
                if not bl.is_mark():
                    self.exitedges.append(FloatingEdge(curblock, bl))
        for bl in toexitblock:
            self.exitedges.append(FloatingEdge(bl, self.exitblock))

    def label_containments(
        self, body: list[FlowBlock], looporder: list["LoopBody"]
    ) -> None:
        """``LoopBody::labelContainments`` (blockaction.cc:327)."""
        containlist: list[LoopBody] = []
        for curblock in body:
            if curblock is not self.head:
                subloop = LoopBody.find(curblock, looporder)
                if subloop is not None:
                    containlist.append(subloop)
                    subloop.depth += 1
        for lb in containlist:
            if lb.immed_container is None or lb.immed_container.depth < self.depth:
                lb.immed_container = self

    def emit_likely_edges(self, likely: list[FloatingEdge], graph: FlowBlock) -> None:
        """``LoopBody::emitLikelyEdges`` (blockaction.cc:364)."""
        head = self.head
        assert head is not None
        while head.parent is not graph:
            head = head.parent
            if head is None:
                raise StructureError("emitLikelyEdges lost the graph")
        self.head = head
        if self.exitblock is not None:
            exitblock: Optional[FlowBlock] = self.exitblock
            while exitblock is not None and exitblock.parent is not graph:
                exitblock = exitblock.parent
            self.exitblock = exitblock
        for i, tail in enumerate(self.tails):
            while tail.parent is not graph:
                tail = tail.parent
                if tail is None:
                    raise StructureError("emitLikelyEdges lost the graph")
            self.tails[i] = tail
            if tail is self.exitblock:
                self.exitblock = None
        holdin: Optional[FlowBlock] = None
        holdout: Optional[FlowBlock] = None
        for pos, edge in enumerate(self.exitedges):
            startbl, outedge = edge.get_current_edge(graph)
            if startbl is None:
                continue
            outbl = startbl.get_out(outedge)
            if pos == len(self.exitedges) - 1:
                if outbl is self.exitblock:
                    holdin = startbl
                    holdout = outbl
                    break
            likely.append(FloatingEdge(startbl, outbl))
        for i in range(len(self.tails) - 1, -1, -1):
            if holdin is not None and i == 0:
                likely.append(FloatingEdge(holdin, holdout))
            tail = self.tails[i]
            for j in range(tail.size_out()):
                bl = tail.get_out(j)
                if bl is head:
                    likely.append(FloatingEdge(tail, head))

    def set_exit_marks(self, graph: FlowBlock) -> None:
        """``LoopBody::setExitMarks`` (blockaction.cc:416)."""
        for edge in self.exitedges:
            inloop, outedge = edge.get_current_edge(graph)
            if inloop is not None:
                inloop.set_loop_exit(outedge)

    def clear_exit_marks(self, graph: FlowBlock) -> None:
        """``LoopBody::clearExitMarks`` (blockaction.cc:430)."""
        for edge in self.exitedges:
            inloop, outedge = edge.get_current_edge(graph)
            if inloop is not None:
                inloop.clear_loop_exit(outedge)

    # -- estáticos ---------------------------------------------------------
    @staticmethod
    def merge_identical_heads(looporder: list["LoopBody"]) -> None:
        """``LoopBody::mergeIdenticalHeads`` (blockaction.cc:446)."""
        if not looporder:
            return
        i = 0
        j = i + 1
        curbody = looporder[i]
        while j < len(looporder):
            nextbody = looporder[j]
            j += 1
            if nextbody.head is curbody.head:
                curbody.add_tail(nextbody.tails[0])
                nextbody.head = None  # Marcado como subsumido
            else:
                i += 1
                looporder[i] = nextbody
                curbody = nextbody
        i += 1
        del looporder[i:]

    @staticmethod
    def compare_ends(a: "LoopBody", b: "LoopBody") -> bool:
        """``LoopBody::compare_ends`` (blockaction.cc:473)."""
        aindex = a.head.index
        bindex = b.head.index
        if aindex != bindex:
            return aindex < bindex
        return a.tails[0].index < b.tails[0].index

    @staticmethod
    def compare_head(a: "LoopBody", looptop: FlowBlock) -> int:
        """``LoopBody::compare_head`` (blockaction.cc:489)."""
        aindex = a.head.index if a.head is not None else -1
        bindex = looptop.index
        if aindex != bindex:
            return -1 if aindex < bindex else 1
        return 0

    @staticmethod
    def find(looptop: FlowBlock, looporder: list["LoopBody"]) -> Optional["LoopBody"]:
        """``LoopBody::find`` (blockaction.cc:1021): búsqueda binaria."""
        minimum = 0
        maximum = len(looporder) - 1
        while minimum <= maximum:
            mid = (minimum + maximum) // 2
            comp = LoopBody.compare_head(looporder[mid], looptop)
            if comp == 0:
                return looporder[mid]
            if comp < 0:
                minimum = mid + 1
            else:
                maximum = mid - 1
        return None

    @staticmethod
    def clear_marks(body: list[FlowBlock]) -> None:
        for bl in body:
            bl.clear_mark()


class CollapseStructure:
    """Construye la estructura de código a partir del grafo (``CollapseStructure``)."""

    def __init__(self, graph: BlockGraph) -> None:
        self.graph = graph
        self.finaltrace = False
        self.likelylistfull = False
        self.likelygoto: list[FloatingEdge] = []
        self.likelyiter = 0
        self.loopbody: list[LoopBody] = []
        self.loopbodyiter = 0
        self.dataflow_changecount = 0
        # Compatibilidad con el nombre antiguo del puerto.
        self.change_count = 0

    def get_change_count(self) -> int:
        return self.dataflow_changecount

    # -- bucles ------------------------------------------------------------
    def label_loops(self, looporder: list[LoopBody]) -> None:
        """``CollapseStructure::labelLoops`` (blockaction.cc:1126)."""
        for i in range(self.graph.get_size()):
            bl = self.graph.get_block(i)
            for j in range(bl.size_in()):
                if bl.is_back_edge_in(j):
                    loopbottom = bl.get_in(j)
                    curbody = LoopBody(bl)
                    curbody.add_tail(loopbottom)
                    self.loopbody.append(curbody)
                    looporder.append(curbody)
        looporder.sort(key=lambda lb: (lb.head.index, lb.tails[0].index))

    def order_loop_bodies(self) -> None:
        """``CollapseStructure::orderLoopBodies`` (blockaction.cc:1148)."""
        looporder: list[LoopBody] = []
        self.label_loops(looporder)
        if self.loopbody:
            oldsize = len(looporder)
            LoopBody.merge_identical_heads(looporder)
            if oldsize != len(looporder):
                self.loopbody = [
                    lb for lb in self.loopbody if lb.get_head() is not None
                ]
            for lb in self.loopbody:
                body: list[FlowBlock] = []
                lb.find_base(body)
                lb.label_containments(body, looporder)
                LoopBody.clear_marks(body)
            # loopbody.sort() con operator< (profundidad decreciente, estable)
            self.loopbody = sorted(self.loopbody, key=lambda item: -item.depth)
            for lb in self.loopbody:
                body = []
                lb.find_base(body)
                lb.find_exit(body)
                lb.order_tails()
                lb.extend(body)
                lb.label_exit_edges(body)
                LoopBody.clear_marks(body)
        self.likelylistfull = False
        self.loopbodyiter = 0

    def update_loop_body(self) -> bool:
        """``CollapseStructure::updateLoopBody`` (blockaction.cc:1193)."""
        if self.finaltrace:
            return False
        loopbottom: Optional[FlowBlock] = None
        looptop: Optional[FlowBlock] = None
        while self.loopbodyiter < len(self.loopbody):
            cur_body = self.loopbody[self.loopbodyiter]
            loopbottom = cur_body.update(self.graph)
            if loopbottom is not None:
                looptop = cur_body.get_head()
                if loopbottom is looptop:
                    # Un único nodo que bucle sobre sí mismo: probable switch.
                    self.likelygoto.clear()
                    self.likelygoto.append(FloatingEdge(looptop, looptop))
                    self.likelyiter = 0
                    self.likelylistfull = True
                    return True
                if not self.likelylistfull or self.likelyiter != len(self.likelygoto):
                    break  # El bucle sigue existiendo
            self.loopbodyiter += 1
            self.likelylistfull = False
            loopbottom = None
        if self.likelylistfull and self.likelyiter != len(self.likelygoto):
            return True

        self.likelygoto.clear()
        tracer = TraceDAG(self.likelygoto, max_iterations=push_budget(self.graph))
        if loopbottom is not None:
            tracer.add_root(looptop)
            tracer.set_finish_block(loopbottom)
            self.loopbody[self.loopbodyiter].set_exit_marks(self.graph)
        else:
            for i in range(self.graph.get_size()):
                bl = self.graph.get_block(i)
                if bl.size_in() == 0:
                    tracer.add_root(bl)
        tracer.initialize()
        tracer.push_branches()
        self.likelylistfull = True
        if loopbottom is not None:
            self.loopbody[self.loopbodyiter].emit_likely_edges(
                self.likelygoto, self.graph
            )
            self.loopbody[self.loopbodyiter].clear_exit_marks(self.graph)
        elif not self.likelygoto:
            self.finaltrace = True
            return False
        self.likelyiter = 0
        return True

    def select_goto(self) -> Optional[FlowBlock]:
        """``CollapseStructure::selectGoto`` (blockaction.cc:1260)."""
        while self.update_loop_body():
            while self.likelyiter != len(self.likelygoto):
                edge = self.likelygoto[self.likelyiter]
                startbl, outedge = edge.get_current_edge(self.graph)
                self.likelyiter += 1
                if startbl is not None:
                    startbl.set_goto_branch(outedge)
                    return startbl
        if not self.clip_extra_roots():
            raise StructureError("Could not finish collapsing block structure")
        return None

    # -- raíces extra ------------------------------------------------------
    def only_reachable_from_root(
        self, root: FlowBlock, body: list[FlowBlock]
    ) -> None:
        """``CollapseStructure::onlyReachableFromRoot`` (blockaction.cc:1052)."""
        trial: list[FlowBlock] = []
        i = 0
        root.set_mark()
        body.append(root)
        while i < len(body):
            bl = body[i]
            i += 1
            for j in range(bl.size_out()):
                curbl = bl.get_out(j)
                if curbl.is_mark():
                    continue
                count = curbl.get_visit_count()
                if count == 0:
                    trial.append(curbl)
                count += 1
                curbl.set_visit_count(count)
                if count == curbl.size_in():
                    curbl.set_mark()
                    body.append(curbl)
        for bl in trial:
            bl.set_visit_count(0)

    def mark_exits_as_gotos(self, body: list[FlowBlock]) -> int:
        """``CollapseStructure::markExitsAsGotos`` (blockaction.cc:1083)."""
        changecount = 0
        for bl in body:
            for j in range(bl.size_out()):
                curbl = bl.get_out(j)
                if not curbl.is_mark():
                    bl.set_goto_branch(j)
                    changecount += 1
        return changecount

    def clip_extra_roots(self) -> bool:
        """``CollapseStructure::clipExtraRoots`` (blockaction.cc:1108)."""
        for i in range(1, self.graph.get_size()):
            bl = self.graph.get_block(i)
            if bl.size_in() != 0:
                continue
            body: list[FlowBlock] = []
            self.only_reachable_from_root(bl, body)
            count = self.mark_exits_as_gotos(body)
            LoopBody.clear_marks(body)
            if count != 0:
                return True
        return False

    # -- reglas ------------------------------------------------------------
    def rule_block_goto(self, bl: FlowBlock) -> bool:
        """``CollapseStructure::ruleBlockGoto`` (blockaction.cc:1443)."""
        sizeout = bl.size_out()
        for i in range(sizeout):
            if bl.is_goto_out(i):
                if bl.is_switch_out():
                    self.graph.new_block_multigoto(bl, i)
                    return True
                if sizeout == 2:
                    if not bl.is_goto_out(1):
                        if bl.negate_condition(True):
                            self.dataflow_changecount += 1
                    self.graph.new_block_if_goto(bl)
                    return True
                if sizeout == 1:
                    self.graph.new_block_goto(bl)
                    return True
        return False

    def rule_block_cat(self, bl: FlowBlock) -> bool:
        """``CollapseStructure::ruleBlockCat`` (blockaction.cc:1284)."""
        if bl.size_out() != 1:
            return False
        if bl.is_switch_out():
            return False
        if bl.size_in() == 1 and bl.get_in(0).size_out() == 1:
            return False  # Debe ser el inicio de la cadena
        outblock = bl.get_out(0)
        if outblock is bl:
            return False
        if outblock.size_in() != 1:
            return False
        if not bl.is_decision_out(0):
            return False
        if outblock.is_switch_out():
            return False

        nodes = [bl, outblock]
        while outblock.size_out() == 1:
            outbl2 = outblock.get_out(0)
            if outbl2 is bl:
                break
            if outbl2.size_in() != 1:
                break
            if not outblock.is_decision_out(0):
                break
            if outbl2.is_switch_out():
                break
            outblock = outbl2
            nodes.append(outblock)

        self.graph.new_block_list(nodes)
        return True

    def rule_block_or(self, bl: FlowBlock) -> bool:
        """``CollapseStructure::ruleBlockOr`` (blockaction.cc:1317)."""
        if bl.size_out() != 2:
            return False
        if bl.is_goto_out(0):
            return False
        if bl.is_goto_out(1):
            return False
        if bl.is_switch_out():
            return False
        for i in range(2):
            orblock = bl.get_out(i)
            if orblock is bl:
                continue
            if orblock.size_in() != 1:
                continue
            if orblock.size_out() != 2:
                continue
            if orblock.is_interior_goto_target():
                continue
            if orblock.is_switch_out():
                continue
            if bl.is_back_edge_out(i):
                continue
            if orblock.is_complex():
                continue
            clauseblock = bl.get_out(1 - i)
            if clauseblock is bl:
                continue
            if clauseblock is orblock:
                continue
            j = 0
            while j < 2:
                if clauseblock is orblock.get_out(j):
                    break
                j += 1
            if j == 2:
                continue
            if orblock.get_out(1 - j) is bl:
                continue
            if i == 1:
                if bl.negate_condition(True):
                    self.dataflow_changecount += 1
            if j == 0:
                if orblock.negate_condition(True):
                    self.dataflow_changecount += 1
            self.graph.new_block_condition(bl, orblock)
            return True
        return False

    def rule_block_proper_if(self, bl: FlowBlock) -> bool:
        """``CollapseStructure::ruleBlockProperIf`` (blockaction.cc:1377)."""
        if bl.size_out() != 2:
            return False
        if bl.is_switch_out():
            return False
        if bl.get_out(0) is bl:
            return False
        if bl.get_out(1) is bl:
            return False
        if bl.is_goto_out(0):
            return False
        if bl.is_goto_out(1):
            return False
        for i in range(2):
            clauseblock = bl.get_out(i)
            if clauseblock.size_in() != 1:
                continue
            if clauseblock.size_out() != 1:
                continue
            if clauseblock.is_switch_out():
                continue
            if not bl.is_decision_out(i):
                continue
            if clauseblock.is_goto_out(0):
                continue
            outblock = clauseblock.get_out(0)
            if outblock is not bl.get_out(1 - i):
                continue
            if i == 0:
                if bl.negate_condition(True):
                    self.dataflow_changecount += 1
            self.graph.new_block_if(bl, clauseblock)
            return True
        return False

    def rule_block_if_else(self, bl: FlowBlock) -> bool:
        """``CollapseStructure::ruleBlockIfElse`` (blockaction.cc:1410)."""
        if bl.size_out() != 2:
            return False
        if bl.is_switch_out():
            return False
        if not bl.is_decision_out(0):
            return False
        if not bl.is_decision_out(1):
            return False
        tc = bl.get_true_out()
        fc = bl.get_false_out()
        if tc.size_in() != 1:
            return False
        if fc.size_in() != 1:
            return False
        if tc.size_out() != 1:
            return False
        if fc.size_out() != 1:
            return False
        outblock = tc.get_out(0)
        if outblock is bl:
            return False
        if outblock is not fc.get_out(0):
            return False
        if tc.is_switch_out():
            return False
        if fc.is_switch_out():
            return False
        if tc.is_goto_out(0):
            return False
        if fc.is_goto_out(0):
            return False
        self.graph.new_block_if_else(bl, tc, fc)
        return True

    def rule_block_if_no_exit(self, bl: FlowBlock) -> bool:
        """``CollapseStructure::ruleBlockIfNoExit`` (blockaction.cc:1478)."""
        if bl.size_out() != 2:
            return False
        if bl.is_switch_out():
            return False
        if bl.get_out(0) is bl:
            return False
        if bl.get_out(1) is bl:
            return False
        if bl.is_goto_out(0):
            return False
        if bl.is_goto_out(1):
            return False
        for i in range(2):
            clauseblock = bl.get_out(i)
            if clauseblock.size_in() != 1:
                continue
            if clauseblock.size_out() != 0:
                continue
            if clauseblock.is_switch_out():
                continue
            if not bl.is_decision_out(i):
                continue
            if i == 0:
                otherblock = bl.get_out(1)
                if (
                    otherblock.size_in() == 1
                    and otherblock.size_out() == 0
                    and not otherblock.is_switch_out()
                    and bl.is_decision_out(1)
                ):
                    if bl.negate_condition(True):
                        self.dataflow_changecount += 1
                    self.graph.new_block_if_no_exit(bl, clauseblock, otherblock)
                    return True
            if i == 0:
                if bl.negate_condition(True):
                    self.dataflow_changecount += 1
            self.graph.new_block_if(bl, clauseblock)
            return True
        return False

    def rule_block_while_do(self, bl: FlowBlock) -> bool:
        """``CollapseStructure::ruleBlockWhileDo`` (blockaction.cc:1525)."""
        if bl.size_out() != 2:
            return False
        if bl.is_switch_out():
            return False
        if bl.get_out(0) is bl:
            return False
        if bl.get_out(1) is bl:
            return False
        if bl.is_interior_goto_target():
            return False
        if bl.is_goto_out(0):
            return False
        if bl.is_goto_out(1):
            return False
        for i in range(2):
            clauseblock = bl.get_out(i)
            if clauseblock.size_in() != 1:
                continue
            if clauseblock.size_out() != 1:
                continue
            if clauseblock.is_switch_out():
                continue
            if clauseblock.get_out(0) is not bl:
                continue
            overflow = bl.is_complex()
            if (i == 0) != overflow:
                if bl.negate_condition(True):
                    self.dataflow_changecount += 1
            newbl = self.graph.new_block_while_do(bl, clauseblock)
            if overflow:
                newbl.set_overflow_syntax()
            return True
        return False

    def rule_block_do_while(self, bl: FlowBlock) -> bool:
        """``CollapseStructure::ruleBlockDoWhile`` (blockaction.cc:1566)."""
        if bl.size_out() != 2:
            return False
        if bl.is_switch_out():
            return False
        if bl.is_goto_out(0):
            return False
        if bl.is_goto_out(1):
            return False
        for i in range(2):
            if bl.get_out(i) is not bl:
                continue
            if i == 0:
                if bl.negate_condition(True):
                    self.dataflow_changecount += 1
            self.graph.new_block_do_while(bl)
            return True
        return False

    def rule_block_inf_loop(self, bl: FlowBlock) -> bool:
        """``CollapseStructure::ruleBlockInfLoop`` (blockaction.cc:1590)."""
        if bl.size_out() != 1:
            return False
        if bl.is_goto_out(0):
            return False
        if bl.get_out(0) is not bl:
            return False
        self.graph.new_block_inf_loop(bl)
        return True

    def check_switch_skips(
        self, switchbl: FlowBlock, exitblock: Optional[FlowBlock]
    ) -> bool:
        """``CollapseStructure::checkSwitchSkips`` (blockaction.cc:1613)."""
        if exitblock is None:
            return True
        sizeout = switchbl.size_out()
        defaultnottoexit = False
        anyskiptoexit = False
        for edgenum in range(sizeout):
            if switchbl.get_out(edgenum) is exitblock:
                if not switchbl.is_default_branch(edgenum):
                    anyskiptoexit = True
            else:
                if switchbl.is_default_branch(edgenum):
                    defaultnottoexit = True
        if not anyskiptoexit:
            return True
        if not defaultnottoexit and switchbl.get_type() == "multigoto":
            if switchbl.has_default_goto():  # type: ignore[attr-defined]
                defaultnottoexit = True
        if not defaultnottoexit:
            return True
        for edgenum in range(sizeout):
            if switchbl.get_out(edgenum) is exitblock:
                if not switchbl.is_default_branch(edgenum):
                    switchbl.set_goto_branch(edgenum)
        return False

    def rule_block_switch(self, bl: FlowBlock) -> bool:
        """``CollapseStructure::ruleBlockSwitch`` (blockaction.cc:1657)."""
        if not bl.is_switch_out():
            return False
        exitblock: Optional[FlowBlock] = None
        sizeout = bl.size_out()
        for i in range(sizeout):
            curbl = bl.get_out(i)
            if curbl is bl:
                exitblock = curbl
                break
            if curbl.size_out() > 1:
                exitblock = curbl
                break
            if curbl.size_in() > 1:
                exitblock = curbl
                break
        if exitblock is None:
            for i in range(sizeout):
                curbl = bl.get_out(i)
                if curbl.size_in() > 0 and curbl.is_goto_in(0):
                    return False
                if curbl.is_switch_out():
                    return False
                if curbl.size_out() == 1:
                    if curbl.is_goto_out(0):
                        return False
                    if exitblock is not None:
                        if exitblock is not curbl.get_out(0):
                            return False
                    else:
                        exitblock = curbl.get_out(0)
        else:
            for i in range(exitblock.size_in()):
                if exitblock.is_goto_in(i):
                    return False
            for i in range(exitblock.size_out()):
                if exitblock.is_goto_out(i):
                    return False
            for i in range(sizeout):
                curbl = bl.get_out(i)
                if curbl is exitblock:
                    continue
                if curbl.size_in() > 1:
                    return False
                if curbl.size_in() > 0 and curbl.is_goto_in(0):
                    return False
                if curbl.size_out() > 1:
                    return False
                if curbl.size_out() == 1:
                    if curbl.is_goto_out(0):
                        return False
                    if curbl.get_out(0) is not exitblock:
                        return False
                if curbl.is_switch_out():
                    return False

        if not self.check_switch_skips(bl, exitblock):
            return True

        cases = [bl]
        for i in range(sizeout):
            curbl = bl.get_out(i)
            if curbl is exitblock:
                continue
            cases.append(curbl)
        self.graph.new_block_switch(cases, exitblock is not None)
        return True

    def rule_case_fallthru(self, bl: FlowBlock) -> bool:
        """``CollapseStructure::ruleCaseFallthru`` (blockaction.cc:1737)."""
        if not bl.is_switch_out():
            return False
        sizeout = bl.size_out()
        nonfallthru = 0
        fallthru: list[FlowBlock] = []
        for i in range(sizeout):
            curbl = bl.get_out(i)
            if curbl is bl:
                return False
            if curbl.size_in() > 2 or curbl.size_out() > 1:
                nonfallthru += 1
            elif curbl.size_out() == 1:
                target = curbl.get_out(0)
                if target.size_in() == 2 and target.size_out() <= 1:
                    inslot = curbl.get_out_rev_index(0)
                    if target.get_in(1 - inslot) is bl:
                        fallthru.append(curbl)
            if nonfallthru > 1:
                return False
        if not fallthru:
            return False
        for curbl in fallthru:
            curbl.set_goto_branch(0)
        return True

    # -- bucles principales ------------------------------------------------
    def collapse_internal(self, targetbl: Optional[FlowBlock]) -> int:
        """``CollapseStructure::collapseInternal`` (blockaction.cc:1780)."""
        steps = 0
        budget = _budget(self.graph)
        isolated_count = 0
        while True:
            while True:
                change = False
                index = 0
                isolated_count = 0
                while index < self.graph.get_size():
                    if targetbl is None:
                        bl = self.graph.get_block(index)
                        index += 1
                    else:
                        bl = targetbl
                        change = True
                        targetbl = None
                        index = self.graph.get_size()
                    if bl.size_in() == 0 and bl.size_out() == 0:
                        isolated_count += 1
                        continue
                    steps += 1
                    if steps > budget:
                        raise StructureError("collapseInternal exceeded its budget")
                    if self.rule_block_goto(bl):
                        change = True
                        continue
                    if self.rule_block_cat(bl):
                        change = True
                        continue
                    if self.rule_block_proper_if(bl):
                        change = True
                        continue
                    if self.rule_block_if_else(bl):
                        change = True
                        continue
                    if self.rule_block_while_do(bl):
                        change = True
                        continue
                    if self.rule_block_do_while(bl):
                        change = True
                        continue
                    if self.rule_block_inf_loop(bl):
                        change = True
                        continue
                    if self.rule_block_switch(bl):
                        change = True
                        continue
                if not change:
                    break
            # IfNoExit y CaseFallthru sólo se aplican cuando nada más encaja.
            fullchange = False
            index = 0
            while index < self.graph.get_size():
                bl = self.graph.get_block(index)
                index += 1
                steps += 1
                if steps > budget:
                    raise StructureError("collapseInternal exceeded its budget")
                if self.rule_block_if_no_exit(bl):
                    fullchange = True
                    break
                if self.rule_case_fallthru(bl):
                    fullchange = True
                    break
            if not fullchange:
                break
        return isolated_count

    def collapse_conditions(self) -> None:
        """``CollapseStructure::collapseConditions`` (blockaction.cc:1848)."""
        while True:
            change = False
            i = 0
            while i < self.graph.get_size():
                if self.rule_block_or(self.graph.get_block(i)):
                    change = True
                i += 1
            if not change:
                break

    def collapse_all(self) -> None:
        """``CollapseStructure::collapseAll`` (blockaction.cc:1875)."""
        self.finaltrace = False
        self.graph.clear_visit_count()
        self.order_loop_bodies()
        self.collapse_conditions()
        budget = _budget(self.graph)
        steps = 0
        isolated_count = self.collapse_internal(None)
        while isolated_count < self.graph.get_size():
            steps += 1
            if steps > budget:
                raise StructureError("collapseAll exceeded its budget")
            targetbl = self.select_goto()
            isolated_count = self.collapse_internal(targetbl)
        self.change_count = self.dataflow_changecount
