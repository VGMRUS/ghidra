"""Puerto de ``TraceDAG`` (``blockaction.hh``/``blockaction.cc`` de Ghidra).

``TraceDAG`` recorre el grafo de control como un DAG y, cuando ya no puede
avanzar, elige la arista más probable de ser no estructurada (``selectBadEdge``)
y la registra en la lista ``likelygoto`` que consume ``CollapseStructure``.

Correspondencia con el C++ (blockaction.cc):

    FloatingEdge::getCurrentEdge          -> :27
    TraceDAG::BranchPoint::createTraces   -> :499
    BranchPoint::markPath / distance      -> :509 / :524
    BlockTrace (constructores)            -> :586 / :603
    BadEdgeScore::compareFinal / operator<-> :617 / :635
    TraceDAG::removeTrace                 -> :656
    processExitConflict                   -> :694
    selectBadEdge                         -> :730
    insertActive / removeActive           -> :786 / :798
    checkOpen / openBranch                -> :810 / :839
    checkRetirement / retireBranch        -> :866 / :900
    clearVisitCount                       -> :940
    initialize / pushBranches             -> :967 / :983

Desviaciones documentadas (ver ``PORTING_NOTES.md``):

* Ghidra guarda ``list<BlockTrace *>::iterator``; aquí ``activeiter`` es el
  índice dentro de ``activetrace`` y se renumera tras cada borrado (las listas
  de Ghidra no invalidan iteradores, las de Python sí).
* ``pushBranches`` añade un presupuesto de iteraciones: si se agota lanza
  ``StructureError`` en vez de colgar (Ghidra confiaría en el algoritmo).
"""

from __future__ import annotations

from typing import Any, Optional

from minighidra.block import FlowBlock, StructureError

# Presupuesto defensivo de iteraciones del bucle principal de pushBranches.
#
# Ghidra no lo necesita: confía en que el subgrafo de aristas "loop DAG"
# (las que no son f_back_edge|f_irreducible|f_loop_exit_edge|f_goto_edge) es
# acíclico.  Al colapsar nodos una arista de vuelta puede quedar *dentro* de un
# bloque y desaparecer de la frontera, de modo que en grafos irreducibles puede
# aparecer un ciclo de aristas DAG y el trazo rebota entre dos bloques para
# siempre (openBranch -> checkRetirement -> openBranch ...).  El presupuesto es
# proporcional al tamaño del grafo: en código real se usan ~2 iteraciones por
# bloque+arista, así que queda margen de sobra y los casos patológicos fallan
# en milisegundos con StructureError en vez de colgar.
MAX_PUSH_ITERATIONS = 2_000_000
_PUSH_STEPS_MIN = 4096
_PUSH_STEPS_PER_ELEMENT = 256


def push_budget(graph: Any) -> int:
    """Iteraciones máximas de ``pushBranches`` para un grafo dado."""
    try:
        blocks = list(graph.blocks)
    except AttributeError:
        return MAX_PUSH_ITERATIONS
    edges = sum(len(bl.out_edges) for bl in blocks)
    return _PUSH_STEPS_MIN + _PUSH_STEPS_PER_ELEMENT * (len(blocks) + edges)


class FloatingEdge:
    """Arista cuyos extremos pueden colapsarse (``FloatingEdge`` de Ghidra).

    ``top`` es el bloque de origen y ``bottom`` el de destino, igual que en el
    C++ (``FloatingEdge(FlowBlock *t, FlowBlock *b)``).
    """

    __slots__ = ("top", "bottom")

    def __init__(self, top: Optional[FlowBlock], bottom: Optional[FlowBlock]) -> None:
        self.top = top
        self.bottom = bottom

    def get_top(self) -> Optional[FlowBlock]:
        return self.top

    def get_bottom(self) -> Optional[FlowBlock]:
        return self.bottom

    def get_current_edge(self, graph: FlowBlock) -> tuple[Optional[FlowBlock], int]:
        """``FloatingEdge::getCurrentEdge``: devuelve ``(top, outedge)``.

        Sube por la jerarquía de colapsos hasta el grafo actual; si los dos
        extremos han quedado dentro del mismo componente devuelve ``(None, -1)``.
        """
        top = self.top
        bottom = self.bottom
        while top is not None and top.parent is not graph:
            top = top.parent
        while bottom is not None and bottom.parent is not graph:
            bottom = bottom.parent
        self.top = top
        self.bottom = bottom
        if top is None or bottom is None:
            return None, -1
        outedge = top.get_out_index(bottom)
        if outedge < 0:
            return None, -1
        return top, outedge


class BranchPoint:
    """Nodo del grafo con varias salidas en el DAG (``TraceDAG::BranchPoint``)."""

    __slots__ = ("parent", "pathout", "top", "paths", "depth", "ismark")

    def __init__(self, parenttrace: Optional["BlockTrace"] = None) -> None:
        self.parent: Optional[BranchPoint] = None
        self.pathout = -1
        self.top: Optional[FlowBlock] = None
        self.paths: list["BlockTrace"] = []
        self.depth = 0
        self.ismark = False
        if parenttrace is not None:
            self.parent = parenttrace.top
            self.depth = self.parent.depth + 1
            self.pathout = parenttrace.pathout
            self.top = parenttrace.destnode
            self.create_traces()

    def create_traces(self) -> None:
        """``BranchPoint::createTraces`` (blockaction.cc:499)."""
        top = self.top
        if top is None:
            return
        for i in range(top.size_out()):
            if not top.is_loop_dag_out(i):
                continue
            self.paths.append(BlockTrace(self, len(self.paths), i))

    def mark_path(self) -> None:
        """``BranchPoint::markPath``: marca el camino hasta la raíz."""
        cur: Optional[BranchPoint] = self
        while cur is not None:
            cur.ismark = not cur.ismark
            cur = cur.parent

    def distance(self, op2: "BranchPoint") -> int:
        """``BranchPoint::distance``: distancia al ancestro común."""
        cur: Optional[BranchPoint] = op2
        while cur is not None:
            if cur.ismark:
                return (self.depth - cur.depth) + (op2.depth - cur.depth)
            cur = cur.parent
        return self.depth + op2.depth + 1

    def get_path_start(self, i: int) -> Optional[FlowBlock]:
        """``BranchPoint::getPathStart``."""
        top = self.top
        if top is None:
            return None
        res = 0
        for j in range(top.size_out()):
            if not top.is_loop_dag_out(j):
                continue
            if res == i:
                return top.get_out(j)
            res += 1
        return None


class BlockTrace:
    """Trazo de un único camino saliendo de un ``BranchPoint``."""

    F_ACTIVE = 1
    F_TERMINAL = 2

    __slots__ = (
        "flags", "top", "pathout", "bottom", "destnode", "edgelump",
        "activeiter", "derivedbp",
    )

    def __init__(
        self,
        t: BranchPoint,
        po: int,
        eo_or_bl: Any,
        is_root: bool = False,
    ) -> None:
        self.flags = 0
        self.top = t
        self.pathout = po
        self.edgelump = 1
        self.activeiter = -1
        self.derivedbp: Optional[BranchPoint] = None
        if not is_root:
            # BlockTrace(BranchPoint *t,int4 po,int4 eo)
            self.bottom = t.top
            self.destnode = self.bottom.get_out(eo_or_bl) if self.bottom else None
        else:
            # BlockTrace(BranchPoint *root,int4 po,FlowBlock *bl)
            self.bottom = None
            self.destnode = eo_or_bl

    def is_active(self) -> bool:
        return (self.flags & BlockTrace.F_ACTIVE) != 0

    def is_terminal(self) -> bool:
        return (self.flags & BlockTrace.F_TERMINAL) != 0


class BadEdgeScore:
    """Puntuación de un ``BlockTrace`` como candidata a arista no estructurada."""

    __slots__ = ("exitproto", "trace", "distance", "terminal", "siblingedge")

    def __init__(self, trace: Optional[BlockTrace] = None) -> None:
        self.trace: Optional[BlockTrace] = trace
        self.exitproto: Optional[FlowBlock] = None
        self.distance = -1
        self.terminal = 0
        self.siblingedge = 0
        if trace is not None:
            self.exitproto = trace.destnode
            # Ghidra: terminal = (destnode->sizeOut()==0) ? 1 : 0
            destnode = trace.destnode
            self.terminal = 1 if destnode is None or destnode.size_out() == 0 else 0

    def compare_final(self, op2: "BadEdgeScore") -> bool:
        """``BadEdgeScore::compareFinal``: ¿es ``self`` MENOS probable que ``op2``?"""
        if self.siblingedge != op2.siblingedge:
            return op2.siblingedge < self.siblingedge
        if self.terminal != op2.terminal:
            return self.terminal < op2.terminal
        if self.distance != op2.distance:
            return self.distance < op2.distance
        assert self.trace is not None and op2.trace is not None
        return self.trace.top.depth < op2.trace.top.depth

    def group_key(self) -> tuple[int, int, int]:
        """Clave equivalente a ``BadEdgeScore::operator<`` (blockaction.cc:635)."""
        thisind = self.exitproto.index if self.exitproto is not None else -1
        assert self.trace is not None
        tmpbl = self.trace.top.top
        topind = tmpbl.index if tmpbl is not None else -1
        return (thisind, topind, self.trace.pathout)

    def __lt__(self, op2: "BadEdgeScore") -> bool:
        return self.group_key() < op2.group_key()


class TraceDAG:
    """Selección de aristas no estructuradas sobre un DAG (``TraceDAG``)."""

    def __init__(
        self,
        likelygoto: list[FloatingEdge],
        max_iterations: Optional[int] = None,
    ) -> None:
        self.likelygoto = likelygoto
        self.max_iterations = (
            MAX_PUSH_ITERATIONS if max_iterations is None else max_iterations
        )
        self.rootlist: list[FlowBlock] = []
        self.branchlist: list[BranchPoint] = []
        self.activecount = 0
        self.missedactivecount = 0
        self.activetrace: list[BlockTrace] = []
        # Iterador actual: índice en activetrace; len(activetrace) == end()
        self.current_activeiter = 0
        self.finishblock: Optional[FlowBlock] = None

    def add_root(self, root: FlowBlock) -> None:
        self.rootlist.append(root)

    def set_finish_block(self, bl: Optional[FlowBlock]) -> None:
        self.finishblock = bl

    # -- lista de trazos activos -------------------------------------------
    def insert_active(self, trace: BlockTrace) -> None:
        """``TraceDAG::insertActive`` (blockaction.cc:786)."""
        self.activetrace.append(trace)
        trace.activeiter = len(self.activetrace) - 1
        trace.flags |= BlockTrace.F_ACTIVE
        self.activecount += 1

    def remove_active(self, trace: BlockTrace) -> None:
        """``TraceDAG::removeActive`` (blockaction.cc:798)."""
        pos = trace.activeiter
        if pos < 0 or pos >= len(self.activetrace) or self.activetrace[pos] is not trace:
            # Desviación defensiva: Ghidra borra por iterador; si el índice se ha
            # desincronizado se busca el trazo para no corromper la lista.
            pos = -1
            for i, candidate in enumerate(self.activetrace):
                if candidate is trace:
                    pos = i
                    break
            if pos < 0:
                trace.flags &= ~BlockTrace.F_ACTIVE
                return
        del self.activetrace[pos]
        for other in self.activetrace[pos:]:
            other.activeiter -= 1
        trace.activeiter = -1
        trace.flags &= ~BlockTrace.F_ACTIVE
        self.activecount -= 1

    # -- manipulación de trazos --------------------------------------------
    def remove_trace(self, trace: BlockTrace) -> None:
        """``TraceDAG::removeTrace`` (blockaction.cc:656)."""
        # La arista pasa a tratarse como un goto y se ignoran sus visitas.
        self.likelygoto.append(FloatingEdge(trace.bottom, trace.destnode))
        if trace.destnode is not None:
            trace.destnode.set_visit_count(
                trace.destnode.get_visit_count() + trace.edgelump
            )

        parentbp = trace.top
        if trace.bottom is not parentbp.top:
            # El trazo ya pasó la raíz del BranchPoint: se trata como terminal.
            trace.flags |= BlockTrace.F_TERMINAL
            trace.bottom = None
            trace.destnode = None
            trace.edgelump = 0
            return

        self.remove_active(trace)
        size = len(parentbp.paths)
        for i in range(trace.pathout + 1, size):
            movedtrace = parentbp.paths[i]
            movedtrace.pathout -= 1
            derivedbp = movedtrace.derivedbp
            if derivedbp is not None:
                derivedbp.pathout -= 1
            parentbp.paths[i - 1] = movedtrace
        parentbp.paths.pop()

    def process_exit_conflict(
        self, badedgelist: list[BadEdgeScore], start: int, end: int
    ) -> None:
        """``TraceDAG::processExitConflict`` (blockaction.cc:694)."""
        while start != end:
            iter_ = start + 1
            startbp = badedgelist[start].trace.top
            if iter_ != end:
                startbp.mark_path()
                while iter_ != end:
                    other = badedgelist[iter_]
                    if startbp is other.trace.top:
                        badedgelist[start].siblingedge += 1
                        other.siblingedge += 1
                    dist = startbp.distance(other.trace.top)
                    if badedgelist[start].distance == -1 or badedgelist[start].distance > dist:
                        badedgelist[start].distance = dist
                    if other.distance == -1 or other.distance > dist:
                        other.distance = dist
                    iter_ += 1
                startbp.mark_path()
            start += 1

    def select_bad_edge(self) -> BlockTrace:
        """``TraceDAG::selectBadEdge`` (blockaction.cc:730)."""
        badedgelist: list[BadEdgeScore] = []
        for trace in self.activetrace:
            if trace.is_terminal():
                continue
            if trace.top.top is None and trace.bottom is None:
                continue  # Nunca se eliminan aristas virtuales
            badedgelist.append(BadEdgeScore(trace))
        if not badedgelist:
            raise StructureError("selectBadEdge without candidate traces")
        badedgelist.sort()

        # Agrupa los trazos que salen al MISMO bloque de salida (identidad),
        # como hace el C++ recorriendo la lista ya ordenada.
        curbl = badedgelist[0].exitproto
        startiter = 0
        samenodecount = 1
        index = 1
        while index != len(badedgelist):
            score = badedgelist[index]
            if curbl is score.exitproto:
                samenodecount += 1
                index += 1
            else:
                if samenodecount > 1:
                    self.process_exit_conflict(badedgelist, startiter, index)
                curbl = score.exitproto
                startiter = index
                samenodecount = 1
                index += 1
        if samenodecount > 1:
            self.process_exit_conflict(badedgelist, startiter, index)

        maxiter = 0
        for index in range(1, len(badedgelist)):
            if badedgelist[maxiter].compare_final(badedgelist[index]):
                maxiter = index
        return badedgelist[maxiter].trace

    def check_open(self, trace: BlockTrace) -> bool:
        """``TraceDAG::checkOpen`` (blockaction.cc:810)."""
        if trace.is_terminal():
            return False
        isroot = False
        if trace.top.depth == 0:
            if trace.bottom is None:
                return True
            isroot = True

        bl = trace.destnode
        if bl is None:
            return False
        if bl is self.finishblock and not isroot:
            return False
        ignore = trace.edgelump + bl.get_visit_count()
        count = 0
        for i in range(bl.size_in()):
            if bl.is_loop_dag_in(i):
                count += 1
                if count > ignore:
                    return False
        return True

    def open_branch(self, parent: BlockTrace) -> int:
        """``TraceDAG::openBranch`` (blockaction.cc:839); devuelve el iterador."""
        newbranch = BranchPoint(parent)
        parent.derivedbp = newbranch
        if not newbranch.paths:
            parent.derivedbp = None
            parent.flags |= BlockTrace.F_TERMINAL
            parent.bottom = None
            parent.destnode = None
            parent.edgelump = 0
            return parent.activeiter
        self.remove_active(parent)
        self.branchlist.append(newbranch)
        for path in newbranch.paths:
            self.insert_active(path)
        return newbranch.paths[0].activeiter

    def check_retirement(
        self, trace: BlockTrace
    ) -> tuple[bool, Optional[FlowBlock]]:
        """``TraceDAG::checkRetirement`` (blockaction.cc:866)."""
        if trace.pathout != 0:
            return False, None
        bp = trace.top
        if bp.depth == 0:
            for curtrace in bp.paths:
                if not curtrace.is_active():
                    return False, None
                if not curtrace.is_terminal():
                    return False, None
            return True, None
        outblock: Optional[FlowBlock] = None
        for curtrace in bp.paths:
            if not curtrace.is_active():
                return False, None
            if curtrace.is_terminal():
                continue
            if outblock is curtrace.destnode:
                continue
            if outblock is not None:
                return False, None
            outblock = curtrace.destnode
        return True, outblock

    def retire_branch(
        self, bp: BranchPoint, exitblock: Optional[FlowBlock]
    ) -> int:
        """``TraceDAG::retireBranch`` (blockaction.cc:900); devuelve el iterador."""
        edgeout_bl: Optional[FlowBlock] = None
        edgelump_sum = 0
        for curtrace in bp.paths:
            if not curtrace.is_terminal():
                edgelump_sum += curtrace.edgelump
                if edgeout_bl is None:
                    edgeout_bl = curtrace.bottom
            self.remove_active(curtrace)
        if bp.depth == 0:
            return 0  # activetrace.begin()

        if bp.parent is not None:
            parenttrace = bp.parent.paths[bp.pathout]
            parenttrace.derivedbp = None
            if edgeout_bl is None:
                parenttrace.flags |= BlockTrace.F_TERMINAL
                parenttrace.bottom = None
                parenttrace.destnode = None
                parenttrace.edgelump = 0
            else:
                parenttrace.bottom = edgeout_bl
                parenttrace.destnode = exitblock
                parenttrace.edgelump = edgelump_sum
            self.insert_active(parenttrace)
            return parenttrace.activeiter
        return 0

    def clear_visit_count(self) -> None:
        """``TraceDAG::clearVisitCount`` (blockaction.cc:940).

        El campo ``visitcount`` sólo se modifica en ``removeTrace``, donde se
        incrementa el del ``destnode`` de la arista, que es el ``bottom`` del
        ``FloatingEdge``; por eso se limpia ``getBottom()``.
        """
        for edge in self.likelygoto:
            bottom = edge.get_bottom()
            if bottom is not None:
                bottom.set_visit_count(0)

    def initialize(self) -> None:
        """``TraceDAG::initialize`` (blockaction.cc:967)."""
        root_branch = BranchPoint()
        self.branchlist.append(root_branch)
        for root in self.rootlist:
            newtrace = BlockTrace(root_branch, len(root_branch.paths), root, is_root=True)
            root_branch.paths.append(newtrace)
            self.insert_active(newtrace)

    def push_branches(self) -> None:
        """``TraceDAG::pushBranches`` (blockaction.cc:983)."""
        self.current_activeiter = 0
        self.missedactivecount = 0
        iterations = 0
        while self.activecount > 0:
            iterations += 1
            if iterations > self.max_iterations:
                raise StructureError("TraceDAG.pushBranches exceeded its budget")
            if self.current_activeiter >= len(self.activetrace):
                self.current_activeiter = 0
            curtrace = self.activetrace[self.current_activeiter]
            if self.missedactivecount >= self.activecount:
                badtrace = self.select_bad_edge()
                self.remove_trace(badtrace)
                self.current_activeiter = 0
                self.missedactivecount = 0
                continue
            can_retire, exitblock = self.check_retirement(curtrace)
            if can_retire:
                self.current_activeiter = self.retire_branch(curtrace.top, exitblock)
                self.missedactivecount = 0
                continue
            if self.check_open(curtrace):
                self.current_activeiter = self.open_branch(curtrace)
                self.missedactivecount = 0
                continue
            self.missedactivecount += 1
            self.current_activeiter += 1
        self.clear_visit_count()


class TraceDAGAlias:  # pragma: no cover - compatibilidad con nombres antiguos
    """Alias históricos usados por código anterior del puerto."""

    BlockTrace = BlockTrace
    BranchPoint = BranchPoint
    BadEdgeScore = BadEdgeScore
