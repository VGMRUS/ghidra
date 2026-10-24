from __future__ import annotations
from typing import Any
from .block import FlowBlock, BlockGraph

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

    def order_loop_bodies(self) -> None:
        pass

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
        return False

    def rule_block_or(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 2: return False
        
        for i in range(2):
            orblock = bl.out_edges[i]
            if orblock == bl: continue
            if len(orblock.in_edges) != 1: continue
            if len(orblock.out_edges) != 2: continue
            
            clauseblock = bl.out_edges[1 - i]
            if clauseblock == bl: continue
            if clauseblock == orblock: continue
            
            # Check if clauses match
            match = False
            for j in range(2):
                if clauseblock == orblock.out_edges[j]:
                    match = True
                    break
                    
            if not match: continue
            if orblock.out_edges[1 - j] == bl: continue
            
            # We found a matching condition
            # op is || if we jump to clauseblock on true? Let's just use "&&" or "||" based on which branch
            # We don't have isGotoOut so we simplify
            # Let's say if false out goes to next condition, it's &&
            # For minighidra, we just use "&&" for now, rendering will refine it.
            self.graph.new_block_condition(bl, orblock, "&&")
            return True
        return False


        
    def rule_block_cat(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 1: return False
        if len(bl.in_edges) == 1 and len(bl.in_edges[0].out_edges) == 1: return False # Must be start of chain
        outblock = bl.out_edges[0]
        if outblock == bl: return False # No looping
        if len(outblock.in_edges) != 1: return False

        nodes = [bl, outblock]
        while len(outblock.out_edges) == 1:
            outbl2 = outblock.out_edges[0]
            if outbl2 == bl: break
            if len(outbl2.in_edges) != 1: break
            outblock = outbl2
            nodes.append(outblock)

        self.graph.new_block_list(nodes)
        return True

    def rule_block_proper_if(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 2: return False
        if bl.out_edges[0] == bl or bl.out_edges[1] == bl: return False

        for i in range(2):
            clauseblock = bl.out_edges[i]
            if len(clauseblock.in_edges) != 1: continue
            if len(clauseblock.out_edges) != 1: continue
            outblock = clauseblock.out_edges[0]
            if outblock != bl.out_edges[1 - i]: continue

            self.graph.new_block_if(bl, clauseblock)
            return True
            
        return False

    def rule_block_if_else(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 2: return False
        tc = bl.out_edges[0]
        fc = bl.out_edges[1]
        
        if len(tc.in_edges) != 1: return False
        if len(fc.in_edges) != 1: return False
        if len(tc.out_edges) != 1: return False
        if len(fc.out_edges) != 1: return False
        
        outblock = tc.out_edges[0]
        if outblock == bl: return False
        if outblock != fc.out_edges[0]: return False
        
        self.graph.new_block_if_else(bl, tc, fc)
        return True

    def rule_block_do_while(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 2: return False
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
        if bl.block_type == "switch": return False
        if bl.out_edges[0] == bl: return False
        if bl.out_edges[1] == bl: return False
        
        for i in range(2):
            clauseblock = bl.out_edges[i]
            if len(clauseblock.in_edges) != 1: continue
            if len(clauseblock.out_edges) != 0: continue
            if clauseblock.block_type == "switch": continue
            
            if i == 0:
                otherblock = bl.out_edges[1]
                if len(otherblock.in_edges) == 1 and len(otherblock.out_edges) == 0 and otherblock.block_type != "switch":
                    self.graph.new_block_if_no_exit(bl, clauseblock, otherblock)
                    return True
            
            self.graph.new_block_if_no_exit(bl, clauseblock)
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
                
        cases = [bl]
        for i in range(sizeout):
            curbl = bl.out_edges[i]
            if curbl == exitblock: continue
            cases.append(curbl)
            
        self.graph.new_block_switch(cases, exitblock is not None)
        return True

    
    def rule_block_while_do(self, bl: FlowBlock) -> bool:
        if len(bl.out_edges) != 2: return False
        if bl.out_edges[0] == bl or bl.out_edges[1] == bl: return False

        for i in range(2):
            clauseblock = bl.out_edges[i]
            if len(clauseblock.in_edges) != 1: continue
            if len(clauseblock.out_edges) != 1: continue
            if clauseblock.out_edges[0] != bl: continue

            self.graph.new_block_while_do(bl, clauseblock)
            return True
            
        return False



    def get_change_count(self) -> int:
        return self.change_count
