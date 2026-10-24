from __future__ import annotations
from typing import Any
from .block import FlowBlock, BlockGraph


class LoopBody:
    def __init__(self, h: 'FlowBlock') -> None:
        self.head = h
        self.tails: list['FlowBlock'] = []
        self.depth: int = 0
        self.uniquecount: int = 0
        self.exitblock: 'FlowBlock' | None = None
        
    def add_tail(self, bl: 'FlowBlock') -> None:
        self.tails.append(bl)

    
    def find_base(self, body: list['FlowBlock']) -> None:
        self.head.mark = 1
        body.append(self.head)
        for tail in self.tails:
            if tail.mark == 0:
                tail.mark = 1
                body.append(tail)
                
        self.uniquecount = len(body)
        i = 1
        while i < len(body):
            curblock = body[i]
            i += 1
            for k in range(len(curblock.in_edges)):
                if curblock.is_goto_branch(k): # actually we need to check if the edge from parent is goto
                    continue
                # in minighidra we check if parent's out edge to curblock is goto
                parent = curblock.in_edges[k]
                if parent.mark == 1:
                    continue
                # simplified goto check: if any of parent's out edges targeting curblock are not goto, we trace
                is_goto = True
                for out_idx in range(len(parent.out_edges)):
                    if parent.out_edges[out_idx] == curblock and not parent.is_goto_branch(out_idx):
                        is_goto = False
                        break
                if is_goto:
                    continue
                    
                parent.mark = 1
                body.append(parent)
                
    
    def find(looptop: 'FlowBlock', looporder: list['LoopBody']) -> 'LoopBody | None':
        for lb in looporder:
            if lb.head == looptop:
                return lb
        return None

    def label_containments(self, body: list['FlowBlock'], looporder: list['LoopBody']) -> None:
        containlist = []
        for curblock in body:
            if curblock != self.head:
                subloop = LoopBody.find(curblock, looporder)
                if subloop is not None:
                    containlist.append(subloop)
                    subloop.depth += 1
                    
        for lb in containlist:
            if lb.immed_container is None or lb.immed_container.depth < self.depth:
                lb.immed_container = self

    def clear_marks(body: list['FlowBlock']) -> None:
        for b in body:
            b.mark = 0

    def merge_identical_heads(looporder: list['LoopBody']) -> None:
        if not looporder:
            return
        
        i = 0
        while i < len(looporder) - 1:
            if looporder[i].head == looporder[i+1].head:
                looporder[i].tails.extend(looporder[i+1].tails)
                looporder.pop(i+1)
            else:
                i += 1
                
    def compare_ends(a: 'LoopBody', b: 'LoopBody') -> int:
        if a.head.index != b.head.index:
            return a.head.index - b.head.index
        # comparing tails is slightly complex, skip strict sorting for now if identical head
        return 0

class CollapseStructure:
    """Recovers hierarchical control flow from flat blocks."""

    def __init__(self, graph: BlockGraph) -> None:
        self.graph = graph
        self.change_count = 0
        self.dataflow_changecount = 0

    def collapse_all(self) -> None:
        """
        The core of the control-flow structuring algorithm.
        Repeatedly applies rules to collapse block graph nodes.
        """
        self.order_loop_bodies()
        self.collapse_conditions()

        isolated_count = self._collapse_internal(None)
        while isolated_count < len(self.graph.blocks):
            targetbl = self.select_goto()
            if targetbl is None:
                break
            isolated_count = self._collapse_internal(targetbl)

    
    def label_loops(self, looporder: list[LoopBody]) -> None:
        self.loopbody = [] # Keep track of all bodies
        
        for bl in self.graph.blocks:
            for j in range(len(bl.in_edges)):
                if bl.is_back_edge_in(j):
                    loopbottom = bl.in_edges[j]
                    curbody = LoopBody(bl)
                    curbody.add_tail(loopbottom)
                    self.loopbody.append(curbody)
                    looporder.append(curbody)
                    
        # sorting by head index to merge later
        looporder.sort(key=lambda x: x.head.index)

    def order_loop_bodies(self) -> None:
        self.graph.calc_forward_dominator([self.graph.blocks[0]] if self.graph.blocks else [])
        
        looporder = []
        self.label_loops(looporder)
        if self.loopbody:
            oldsize = len(looporder)
            LoopBody.merge_identical_heads(looporder)
            
            # Remove subsumed bodies
            if oldsize != len(looporder):
                self.loopbody = [lb for lb in self.loopbody if lb in looporder]
                
            # Rest of loop ordering relies on deep graph logic, omitted for now
            
            for lb in self.loopbody:
                body = []
                lb.find_base(body)
                lb.label_containments(body, looporder)
                LoopBody.clear_marks(body)
                
            self.loopbody.sort(key=lambda lb: lb.depth, reverse=True)

        

    def collapse_conditions(self) -> None:
        change = True
        while change:
            change = False
            for i in range(len(self.graph.blocks)):
                if i < len(self.graph.blocks) and self.rule_block_or(self.graph.blocks[i]):
                    change = True
                    break

    def select_goto(self) -> FlowBlock | None:
        return None

    def update_loop_body(self) -> bool:
        return False

    def _collapse_internal(self, targetbl: FlowBlock | None) -> int:
        isolated_count = 0
        fullchange = False

        while True:
            change = True
            while change:
                change = False
                index = 0
                isolated_count = 0

                while index < len(self.graph.blocks):
                    if targetbl is None:
                        bl = self.graph.blocks[index]
                        index += 1
                    else:
                        bl = targetbl
                        change = True
                        targetbl = None
                        index = len(self.graph.blocks)

                    if len(bl.in_edges) == 0 and len(bl.out_edges) == 0:
                        isolated_count += 1
                        continue

                    # Try each rule on the block
                    if self.rule_block_goto(bl):
                        change = True
                        continue
                    if self.rule_block_or(bl):
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
                    if self.rule_case_fallthru(bl):
                        fullchange = True
                        break
                    if self.rule_block_do_while(bl):
                        change = True
                        continue
                    if self.rule_block_inf_loop(bl):
                        change = True
                        continue
                    if self.rule_block_if_no_exit(bl):
                        change = True
                        continue
                    if self.rule_block_switch(bl):
                        change = True
                        continue

                if change:
                    fullchange = True

            if not self.update_loop_body():
                break

        if fullchange:
            self.dataflow_changecount += 1

        return isolated_count

    def rule_block_goto(self, bl: FlowBlock) -> bool:
        sizeout = len(bl.out_edges)
        for i in range(sizeout):
            if bl.is_goto_branch(i):
                if bl.block_type == "switch_out":
                    self.graph.new_block_multigoto(bl, i)
                    return True
                if sizeout == 2:
                    if not bl.is_goto_branch(1):
                        # True branch must be goto
                        # In full ghidra: if bl.negateCondition(True): dataflow_changecount += 1
                        pass
                    self.graph.new_block_if_goto(bl)
                    return True
                if sizeout == 1:
                    self.graph.new_block_goto(bl)
                    return True
        return False

    def rule_block_or(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 2: return False
        if bl.is_goto_branch(0) or bl.is_goto_branch(1): return False
        if bl.block_type == "switch_out": return False
        
        for i in range(2):
            orblock = bl.out_edges[i]
            if orblock == bl: continue
            if len(orblock.in_edges) != 1: continue
            if len(orblock.out_edges) != 2: continue
            if orblock.block_type == "switch_out": continue
            
            clauseblock = bl.out_edges[1 - i]
            if clauseblock == bl: continue
            if clauseblock == orblock: continue
            
            j = 0
            match = False
            for j in range(2):
                if clauseblock == orblock.out_edges[j]:
                    match = True
                    break
                    
            if not match: continue
            if orblock.out_edges[1 - j] == bl: continue
            
            self.graph.new_block_condition(bl, orblock, "&&" if i == 0 else "||")
            return True
            
        return False

    def rule_block_cat(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 1: return False
        if bl.block_type == "switch_out": return False
        if len(bl.in_edges) == 1 and len(bl.in_edges[0].out_edges) == 1:
            return False # Must be start of chain
            
        outblock = bl.out_edges[0]
        if outblock == bl: return False
        if len(outblock.in_edges) != 1: return False
        # In ghidra: if not bl.isDecisionOut(0): return False
        if outblock.block_type == "switch_out": return False
        
        nodes = [bl, outblock]
        
        while len(outblock.out_edges) == 1:
            outbl2 = outblock.out_edges[0]
            if outbl2 == bl: break
            if len(outbl2.in_edges) != 1: break
            if outbl2.block_type == "switch_out": break
            outblock = outbl2
            nodes.append(outblock)
            
        self.graph.new_block_list(nodes)
        return True

    def rule_block_proper_if(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 2: return False
        if bl.block_type == "switch_out": return False
        if bl.out_edges[0] == bl or bl.out_edges[1] == bl: return False
        if bl.is_goto_branch(0) or bl.is_goto_branch(1): return False

        for i in range(2):
            clauseblock = bl.out_edges[i]
            if len(clauseblock.in_edges) != 1: continue
            if len(clauseblock.out_edges) != 1: continue
            if getattr(clauseblock, "block_type", "") == "switch_out": continue
            if clauseblock.is_goto_branch(0): continue
            
            outblock = clauseblock.out_edges[0]
            if outblock != bl.out_edges[1 - i]: continue

            self.graph.new_block_if(bl, clauseblock)
            return True
            
        return False

    def rule_block_if_else(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 2: return False
        if bl.block_type == "switch_out": return False
        
        tc = bl.out_edges[0]
        fc = bl.out_edges[1]
        
        if len(tc.in_edges) != 1: return False
        if len(fc.in_edges) != 1: return False
        if len(tc.out_edges) != 1: return False
        if len(fc.out_edges) != 1: return False
        
        outblock = tc.out_edges[0]
        if outblock == bl: return False
        if outblock != fc.out_edges[0]: return False
        
        if tc.block_type == "switch_out": return False
        if fc.block_type == "switch_out": return False
        if tc.is_goto_branch(0): return False
        if fc.is_goto_branch(0): return False
        
        self.graph.new_block_if_else(bl, tc, fc)
        return True

    def rule_block_do_while(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 2: return False
        if bl.block_type == "switch_out": return False
        if bl.is_goto_branch(0) or bl.is_goto_branch(1): return False
        
        for i in range(2):
            if bl.out_edges[i] != bl: continue
            
            self.graph.new_block_do_while(bl)
            return True
        return False

    def rule_block_inf_loop(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 1: return False
        if bl.out_edges[0] != bl: return False
        self.graph.new_block_inf_loop(bl)
        return True

    def rule_block_if_no_exit(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 2: return False
        if bl.block_type == "switch_out": return False
        if bl.out_edges[0] == bl or bl.out_edges[1] == bl: return False
        if bl.is_goto_branch(0) or bl.is_goto_branch(1): return False
        
        for i in range(2):
            clauseblock = bl.out_edges[i]
            if len(clauseblock.in_edges) != 1: continue
            if len(clauseblock.out_edges) != 0: continue
            if clauseblock.block_type == "switch_out": continue
            
            if i == 0:
                otherblock = bl.out_edges[1]
                if len(otherblock.in_edges) == 1 and len(otherblock.out_edges) == 0 and otherblock.block_type != "switch_out":
                    self.graph.new_block_if_no_exit(bl, clauseblock, otherblock)
                    return True
                    
            self.graph.new_block_if(bl, clauseblock)
            return True
            
        return False

    def rule_block_switch(self, bl: FlowBlock) -> bool:
        if bl.block_type != "switch_out" and len(bl.out_edges) <= 2: return False
        
        exitblock = None
        sizeout = len(bl.out_edges)
        
        for i in range(sizeout):
            curbl = bl.out_edges[i]
            if curbl == bl:
                exitblock = curbl
                break
            if len(curbl.out_edges) > 1:
                exitblock = curbl
                break
            if len(curbl.in_edges) > 1:
                exitblock = curbl
                break
                
        if exitblock is None:
            for i in range(sizeout):
                curbl = bl.out_edges[i]
                if curbl.block_type == "switch": return False
                if len(curbl.out_edges) == 1:
                    if exitblock is not None:
                        if exitblock != curbl.out_edges[0]: return False
                    else:
                        exitblock = curbl.out_edges[0]
        else:
            for i in range(sizeout):
                curbl = bl.out_edges[i]
                if curbl == exitblock: continue
                if len(curbl.in_edges) > 1: return False
                if len(curbl.out_edges) > 1: return False
                if len(curbl.out_edges) == 1:
                    if curbl.out_edges[0] != exitblock: return False
                if curbl.block_type == "switch": return False
                
        if not self._check_switch_skips(bl, exitblock):
            return True
            
        cases = [bl]
        for i in range(sizeout):
            curbl = bl.out_edges[i]
            if curbl == exitblock: continue
            cases.append(curbl)
            
        self.graph.new_block_switch(cases, exitblock is not None)
        return True

    
    def rule_block_while_do(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 2: return False
        if bl.block_type == "switch_out": return False
        if bl.out_edges[0] == bl or bl.out_edges[1] == bl: return False
        if bl.is_goto_branch(0) or bl.is_goto_branch(1): return False

        for i in range(2):
            clauseblock = bl.out_edges[i]
            if len(clauseblock.in_edges) != 1: continue
            if len(clauseblock.out_edges) != 1: continue
            if clauseblock.block_type == "switch_out": continue
            if clauseblock.out_edges[0] != bl: continue

            self.graph.new_block_while_do(bl, clauseblock)
            return True
            
        return False




    def rule_case_fallthru(self, bl: FlowBlock) -> bool:
        if bl.block_type != "switch_out": return False
        sizeout = len(bl.out_edges)
        nonfallthru = 0
        fallthru = []
        
        for i in range(sizeout):
            curbl = bl.out_edges[i]
            if curbl == bl: return False
            if len(curbl.in_edges) > 2 or len(curbl.out_edges) > 1:
                nonfallthru += 1
            elif len(curbl.out_edges) == 1:
                target = curbl.out_edges[0]
                if len(target.in_edges) == 2 and len(target.out_edges) <= 1:
                    inslot = target.in_edges.index(curbl)
                    if target.in_edges[1 - inslot] == bl:
                        fallthru.append(curbl)
            
            if nonfallthru > 1: return False
            
        if not fallthru: return False
        
        for curbl in fallthru:
            curbl.set_goto_branch(0)
            
        return True


    def _check_switch_skips(self, switchbl: FlowBlock, exitblock: FlowBlock | None) -> bool:
        if exitblock is None:
            return True
            
        sizeout = len(switchbl.out_edges)
        defaultnottoexit = False
        anyskiptoexit = False
        
        for edgenum in range(sizeout):
            if switchbl.out_edges[edgenum] == exitblock:
                if not switchbl.is_default_branch(edgenum):
                    anyskiptoexit = True
            else:
                if switchbl.is_default_branch(edgenum):
                    defaultnottoexit = True
                    
        if not anyskiptoexit:
            return True
            
        if not defaultnottoexit and getattr(switchbl, "block_type", "") == "multigoto":
            if getattr(switchbl, "has_default_goto", lambda: False)():
                defaultnottoexit = True
                
        if not defaultnottoexit:
            return True
            
        for edgenum in range(sizeout):
            if switchbl.out_edges[edgenum] == exitblock:
                if not switchbl.is_default_branch(edgenum):
                    switchbl.set_goto_branch(edgenum)
                    
        return False

    def get_change_count(self) -> int:
        return self.change_count
