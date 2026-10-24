from typing import List, Tuple, Optional, Any
from minighidra.block import FlowBlock, BlockGraph

class FloatingEdge:
    def __init__(self, bottom: FlowBlock, destnode: FlowBlock):
        self.bottom = bottom
        self.destnode = destnode

    def get_bottom(self) -> FlowBlock:
        return self.bottom
        
    def get_destnode(self) -> FlowBlock:
        return self.destnode

class BlockTrace:
    pass

class BranchPoint:
    def __init__(self, parenttrace: Optional[BlockTrace] = None):
        self.parent: Optional[BranchPoint] = None
        self.pathout: int = -1
        self.top: Optional[FlowBlock] = None
        self.paths: List[BlockTrace] = []
        self.depth: int = 0
        self.ismark: bool = False
        
        if parenttrace:
            self.parent = parenttrace.top
            self.depth = self.parent.depth + 1
            self.pathout = parenttrace.pathout
            self.top = parenttrace.destnode
            self.create_traces()

    def create_traces(self) -> None:
        if not self.top:
            return
        sizeout = len(self.top.out_edges)
        for i in range(sizeout):
            # is_loop_dag_out logic simplified or ported?
            if not self.top.is_loop_dag_out(i):
                continue
            self.paths.append(BlockTrace(self, len(self.paths), i))

    def mark_path(self) -> None:
        cur = self
        while cur:
            cur.ismark = not cur.ismark
            cur = cur.parent

    def distance(self, op2: 'BranchPoint') -> int:
        cur = op2
        while cur:
            if cur.ismark:
                return (self.depth - cur.depth) + (op2.depth - cur.depth)
            cur = cur.parent
        return self.depth + op2.depth + 1

class BlockTrace:
    def __init__(self, t: BranchPoint, po: int, eo_or_bl: Any, is_root: bool = False):
        self.flags: int = 0
        self.top: BranchPoint = t
        self.pathout: int = po
        self.edgelump: int = 1
        self.derivedbp: Optional[BranchPoint] = None
        
        if not is_root:
            self.bottom = self.top.top
            self.destnode = self.bottom.out_edges[eo_or_bl] if self.bottom else None
        else:
            self.bottom = None
            self.destnode = eo_or_bl

    @property
    def f_active(self) -> int: return 1
    @property
    def f_terminal(self) -> int: return 2

    def is_active(self) -> bool:
        return (self.flags & self.f_active) != 0

    def is_terminal(self) -> bool:
        return (self.flags & self.f_terminal) != 0

class BadEdgeScore:
    def __init__(self, trace: BlockTrace):
        self.trace = trace
        self.exitproto = trace.destnode
        self.distance = -1
        self.terminal = 1 if not trace.destnode or len(trace.destnode.out_edges) == 0 else 0
        self.siblingedge = 0

    def compare_final(self, op2: 'BadEdgeScore') -> bool:
        if self.siblingedge != op2.siblingedge:
            return op2.siblingedge < self.siblingedge
        if self.terminal != op2.terminal:
            return self.terminal < op2.terminal
        if self.distance != op2.distance:
            return self.distance < op2.distance
        return self.trace.top.depth < op2.trace.top.depth

    def compare_group(self, op2: 'BadEdgeScore') -> bool:
        thisind = self.exitproto.index if self.exitproto else -1
        op2ind = op2.exitproto.index if op2.exitproto else -1
        if thisind != op2ind:
            return thisind < op2ind
            
        tmpbl = self.trace.top.top
        thisind = tmpbl.index if tmpbl else -1
        tmpbl = op2.trace.top.top
        op2ind = tmpbl.index if tmpbl else -1
        if thisind != op2ind:
            return thisind < op2ind
            
        return self.trace.pathout < op2.trace.pathout

class TraceDAG:
    def __init__(self, likelygoto: List[FloatingEdge]):
        self.likelygoto = likelygoto
        self.rootlist: List[FlowBlock] = []
        self.branchlist: List[BranchPoint] = []
        self.activecount: int = 0
        self.missedactivecount: int = 0
        self.activetrace: List[BlockTrace] = []
        self.finishblock: Optional[FlowBlock] = None
        self.current_activeiter: int = 0

    def add_root(self, root: FlowBlock) -> None:
        self.rootlist.append(root)

    def set_finish_block(self, bl: FlowBlock) -> None:
        self.finishblock = bl

    def initialize(self) -> None:
        root_branch = BranchPoint()
        self.branchlist.append(root_branch)
        for i, root in enumerate(self.rootlist):
            newtrace = BlockTrace(root_branch, len(root_branch.paths), root, is_root=True)
            root_branch.paths.append(newtrace)
            self.insert_active(newtrace)

    def insert_active(self, trace: BlockTrace) -> None:
        self.activetrace.append(trace)
        trace.flags |= 1 # f_active
        self.activecount += 1

    def remove_active(self, trace: BlockTrace) -> None:
        if trace in self.activetrace:
            self.activetrace.remove(trace)
        trace.flags &= ~1
        self.activecount -= 1

    def remove_trace(self, trace: BlockTrace) -> None:
        if trace.bottom and trace.destnode:
            self.likelygoto.append(FloatingEdge(trace.bottom, trace.destnode))
            if hasattr(trace.destnode, 'visit_count'):
                trace.destnode.visit_count += trace.edgelump

        parentbp = trace.top
        if trace.bottom != parentbp.top:
            trace.flags |= 2 # f_terminal
            trace.bottom = None
            trace.destnode = None
            trace.edgelump = 0
            return
            
        self.remove_active(trace)
        idx = trace.pathout
        parentbp.paths.pop(idx)
        for i in range(idx, len(parentbp.paths)):
            parentbp.paths[i].pathout -= 1
            if parentbp.paths[i].derivedbp:
                parentbp.paths[i].derivedbp.pathout -= 1

    def select_bad_edge(self) -> BlockTrace:
        badedgelist: List[BadEdgeScore] = []
        for trace in self.activetrace:
            if trace.is_terminal(): continue
            if not trace.top.top and not trace.bottom: continue
            
            badedgelist.append(BadEdgeScore(trace))
            
        import functools
        def cmp_group(a, b): return -1 if a.compare_group(b) else (1 if b.compare_group(a) else 0)
        badedgelist.sort(key=functools.cmp_to_key(cmp_group))
        
        # Process exit conflicts
        start = 0
        while start < len(badedgelist):
            end = start + 1
            while end < len(badedgelist) and cmp_group(badedgelist[start], badedgelist[end]) == 0:
                end += 1
            self._process_exit_conflict(badedgelist, start, end)
            start = end
            
        best = badedgelist[0]
        for score in badedgelist[1:]:
            if score.compare_final(best):
                best = score
        return best.trace

    def _process_exit_conflict(self, badedgelist: List[BadEdgeScore], start: int, end: int) -> None:
        for i in range(start, end):
            score_i = badedgelist[i]
            startbp = score_i.trace.top
            startbp.mark_path()
            for j in range(i+1, end):
                score_j = badedgelist[j]
                if startbp == score_j.trace.top:
                    score_i.siblingedge += 1
                    score_j.siblingedge += 1
                dist = startbp.distance(score_j.trace.top)
                if score_i.distance == -1 or score_i.distance > dist:
                    score_i.distance = dist
                if score_j.distance == -1 or score_j.distance > dist:
                    score_j.distance = dist
            startbp.mark_path()

    def check_open(self, trace: BlockTrace) -> bool:
        if trace.is_terminal(): return False
        isroot = False
        if trace.top.depth == 0:
            if not trace.bottom: return True
            isroot = True
            
        bl = trace.destnode
        if bl == self.finishblock and not isroot:
            return False
            
        ignore = trace.edgelump + getattr(bl, 'visit_count', 0)
        count = 0
        for i in range(len(bl.in_edges)):
            if bl.is_loop_dag_in(i):
                count += 1
                if count > ignore:
                    return False
        return True

    def open_branch(self, parent: BlockTrace) -> None:
        newbranch = BranchPoint(parent)
        parent.derivedbp = newbranch
        if not newbranch.paths:
            parent.derivedbp = None
            parent.flags |= 2
            parent.bottom = None
            parent.destnode = None
            parent.edgelump = 0
            return
            
        self.remove_active(parent)
        self.branchlist.append(newbranch)
        for p in newbranch.paths:
            self.insert_active(p)
        self.current_activeiter = self.activetrace.index(newbranch.paths[0])

    def check_retirement(self, trace: BlockTrace) -> Tuple[bool, Optional[FlowBlock]]:
        if trace.pathout != 0: return False, None
        bp = trace.top
        if bp.depth == 0:
            for p in bp.paths:
                if not p.is_active() or not p.is_terminal(): return False, None
            return True, None
            
        outblock = None
        for p in bp.paths:
            if not p.is_active(): return False, None
            if p.is_terminal(): continue
            if outblock == p.destnode: continue
            if outblock is not None: return False, None
            outblock = p.destnode
        return True, outblock

    def retire_branch(self, bp: BranchPoint, exitblock: Optional[FlowBlock]) -> None:
        edgeout_bl = None
        edgelump_sum = 0
        
        for p in bp.paths:
            if not p.is_terminal():
                edgelump_sum += p.edgelump
                if not edgeout_bl: edgeout_bl = p.bottom
            self.remove_active(p)
            
        if bp.depth == 0:
            self.current_activeiter = 0
            return
            
        if bp.parent:
            parenttrace = bp.parent.paths[bp.pathout]
            parenttrace.derivedbp = None
            if not edgeout_bl:
                parenttrace.flags |= 2
                parenttrace.bottom = None
                parenttrace.destnode = None
                parenttrace.edgelump = 0
            else:
                parenttrace.bottom = edgeout_bl
                parenttrace.destnode = exitblock
                parenttrace.edgelump = edgelump_sum
            self.insert_active(parenttrace)
            self.current_activeiter = self.activetrace.index(parenttrace)

    def clear_visit_count(self) -> None:
        for edge in self.likelygoto:
            if hasattr(edge.bottom, 'visit_count'):
                edge.bottom.visit_count = 0

    def push_branches(self) -> None:
        self.current_activeiter = 0
        self.missedactivecount = 0
        
        while self.activecount > 0:
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
                self.retire_branch(curtrace.top, exitblock)
                self.missedactivecount = 0
                continue
                
            if self.check_open(curtrace):
                self.open_branch(curtrace)
                self.missedactivecount = 0
                continue
                
            self.missedactivecount += 1
            self.current_activeiter += 1
            
        self.clear_visit_count()
