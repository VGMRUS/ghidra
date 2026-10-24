from __future__ import annotations
from typing import Any

class FlowBlock:
    """Base control-flow structure node."""
    def __init__(self, block_type: str) -> None:
        self.block_type = block_type
        self.in_edges: list[FlowBlock] = []
        self.out_edges: list[FlowBlock] = []
        self.out_edge_flags: dict[int, int] = {}
        self.flags = 0

    def is_default_branch(self, i: int) -> bool:
        return self.out_edge_flags.get(i, 0) & 1 != 0

    def set_default_switch(self, i: int) -> None:
        self.out_edge_flags[i] = self.out_edge_flags.get(i, 0) | 1

    def is_goto_branch(self, i: int) -> bool:
        return self.out_edge_flags.get(i, 0) & 2 != 0

    def set_goto_branch(self, i: int) -> None:
        self.out_edge_flags[i] = self.out_edge_flags.get(i, 0) | 2


    def get_type(self) -> str:
        return self.block_type

    def add_out_edge(self, target: FlowBlock) -> None:
        self.out_edges.append(target)
        target.in_edges.append(self)

    def remove_out_edge(self, index: int) -> None:
        target = self.out_edges[index]
        self.out_edges.pop(index)
        target.in_edges.remove(self)
        
        # Shift out_edge_flags
        new_flags = {}
        for k, v in self.out_edge_flags.items():
            if k < index:
                new_flags[k] = v
            elif k > index:
                new_flags[k - 1] = v
        self.out_edge_flags = new_flags

class BlockBasic(FlowBlock):
    def __init__(self, start_addr: int) -> None:
        super().__init__("basic")
        self.start_addr = start_addr

class BlockList(FlowBlock):
    def __init__(self, blocks: list[FlowBlock]) -> None:
        super().__init__("list")
        self.blocks = blocks

class BlockIf(FlowBlock):
    def __init__(self, cond_block: FlowBlock, true_block: FlowBlock) -> None:
        super().__init__("if")
        self.cond_block = cond_block
        self.true_block = true_block
        self.false_block: FlowBlock | None = None

class BlockIfElse(BlockIf):
    def __init__(self, cond_block: FlowBlock, true_block: FlowBlock, false_block: FlowBlock) -> None:
        super().__init__(cond_block, true_block)
        self.block_type = "ifelse"
        self.false_block = false_block

class BlockWhileDo(FlowBlock):
    def __init__(self, cond_block: FlowBlock, body_block: FlowBlock) -> None:
        super().__init__("whiledo")
        self.cond_block = cond_block
        self.body_block = body_block

class BlockDoWhile(FlowBlock):
    def __init__(self, body_block: FlowBlock) -> None:
        super().__init__("dowhile")
        self.body_block = body_block

class BlockInfLoop(FlowBlock):
    def __init__(self, body_block: FlowBlock) -> None:
        super().__init__("infloop")
        self.body_block = body_block

class BlockSwitch(FlowBlock):
    def __init__(self, cases: list[FlowBlock], has_exit: bool) -> None:
        super().__init__("switch")
        self.cases = cases
        self.has_exit = has_exit

class BlockIfNoExit(FlowBlock):
    def __init__(self, cond_block: FlowBlock, true_block: FlowBlock, false_block: FlowBlock | None = None) -> None:
        super().__init__("ifnoexit")
        self.cond_block = cond_block
        self.true_block = true_block
        self.false_block = false_block

class BlockCondition(FlowBlock):
    def __init__(self, cond1: FlowBlock, cond2: FlowBlock, op: str) -> None:
        super().__init__("condition")
        self.cond1 = cond1
        self.cond2 = cond2
        self.op = op


class BlockGoto(FlowBlock):
    def __init__(self, target: FlowBlock) -> None:
        super().__init__("goto")
        self.gototarget = target

class BlockMultiGoto(FlowBlock):
    def __init__(self, target: FlowBlock) -> None:
        super().__init__("multigoto")
        self.gotoedges: list[FlowBlock] = []

class BlockIfGoto(BlockIf):
    def __init__(self, cond: FlowBlock, target: FlowBlock) -> None:
        super().__init__(cond, None)
        self.block_type = "ifgoto"
        self.gototarget = target

class BlockGraph(FlowBlock):
    def __init__(self) -> None:
        super().__init__("graph")
        self.blocks: list[FlowBlock] = []

    def build_copy(self, blocks_data: list[Any]) -> None:
        """Build the BlockGraph from the raw SSA blocks."""
        self.blocks = []
        block_map = {}
        for b in blocks_data:
            bb = BlockBasic(b.start)
            bb.name = f"block_{hex(b.start)}"
            self.blocks.append(bb)
            block_map[b.start] = bb
            
        for b in blocks_data:
            bb = block_map[b.start]
            for succ in b.successors:
                if succ in block_map:
                    bb.add_out_edge(block_map[succ])

    def new_block_list(self, nodes: list[FlowBlock]) -> BlockList:
        new_list = BlockList(nodes)
        
        first = nodes[0]
        last = nodes[-1]
        
        # Inputs to first -> new_list
        for in_block in list(first.in_edges):
            idx = in_block.out_edges.index(first)
            in_block.out_edges[idx] = new_list
            new_list.in_edges.append(in_block)
        first.in_edges.clear()
        
        # Outputs from last -> new_list
        for out_block in list(last.out_edges):
            idx = last.out_edges.index(out_block)
            last.out_edges[idx] = new_list  # Wait, list.out_edges doesn't matter for last, we need to clear it and add to new_list
            # Actually better:
            out_idx = out_block.in_edges.index(last)
            out_block.in_edges[out_idx] = new_list
            new_list.out_edges.append(out_block)
        last.out_edges.clear()
        
        # Update blocks
        idx = self.blocks.index(first)
        self.blocks[idx] = new_list
        for node in nodes[1:]:
            if node in self.blocks:
                self.blocks.remove(node)
                
        return new_list

    def new_block_if(self, cond: FlowBlock, body: FlowBlock) -> BlockIf:
        new_if = BlockIf(cond, body)
        
        # Inputs to cond -> new_if
        for in_block in list(cond.in_edges):
            idx = in_block.out_edges.index(cond)
            in_block.out_edges[idx] = new_if
            new_if.in_edges.append(in_block)
        cond.in_edges.clear()
        
        # Outputs:
        # cond points to body and outblock
        # body points to outblock
        outblock = body.out_edges[0]
        
        # Remove cond -> outblock
        cond.out_edges.remove(outblock)
        outblock.in_edges.remove(cond)
        
        # Remove body -> outblock
        body.out_edges.remove(outblock)
        outblock.in_edges.remove(body)
        
        # new_if -> outblock
        new_if.out_edges.append(outblock)
        outblock.in_edges.append(new_if)
        
        idx = self.blocks.index(cond)
        self.blocks[idx] = new_if
        if body in self.blocks:
            self.blocks.remove(body)
            
        return new_if

    def new_block_while_do(self, cond: FlowBlock, body: FlowBlock) -> BlockWhileDo:
        new_loop = BlockWhileDo(cond, body)
        
        # Inputs to cond (excluding body) -> new_loop
        for in_block in list(cond.in_edges):
            if in_block == body:
                continue
            idx = in_block.out_edges.index(cond)
            in_block.out_edges[idx] = new_loop
            new_loop.in_edges.append(in_block)
            cond.in_edges.remove(in_block)
            
        # Outputs: cond has outblock
        # One of cond's out_edges is body, the other is outblock
        outblock = cond.out_edges[0] if cond.out_edges[0] != body else cond.out_edges[1]
        
        # Remove cond -> outblock
        cond.out_edges.remove(outblock)
        outblock.in_edges.remove(cond)
        
        # new_loop -> outblock
        new_loop.out_edges.append(outblock)
        outblock.in_edges.append(new_loop)
        
        idx = self.blocks.index(cond)
        self.blocks[idx] = new_loop
        if body in self.blocks:
            self.blocks.remove(body)
            
        return new_loop

    def new_block_if_else(self, cond: FlowBlock, true_body: FlowBlock, false_body: FlowBlock) -> BlockIfElse:
        new_if = BlockIfElse(cond, true_body, false_body)
        
        for in_block in list(cond.in_edges):
            idx = in_block.out_edges.index(cond)
            in_block.out_edges[idx] = new_if
            new_if.in_edges.append(in_block)
        cond.in_edges.clear()
        
        outblock = true_body.out_edges[0]
        
        true_body.out_edges.remove(outblock)
        outblock.in_edges.remove(true_body)
        
        false_body.out_edges.remove(outblock)
        outblock.in_edges.remove(false_body)
        
        new_if.out_edges.append(outblock)
        outblock.in_edges.append(new_if)
        
        idx = self.blocks.index(cond)
        self.blocks[idx] = new_if
        if true_body in self.blocks: self.blocks.remove(true_body)
        if false_body in self.blocks: self.blocks.remove(false_body)
            
        return new_if

    def new_block_do_while(self, body: FlowBlock) -> BlockDoWhile:
        new_loop = BlockDoWhile(body)
        
        for in_block in list(body.in_edges):
            if in_block == body: continue
            idx = in_block.out_edges.index(body)
            in_block.out_edges[idx] = new_loop
            new_loop.in_edges.append(in_block)
            body.in_edges.remove(in_block)
            
        outblock = body.out_edges[0] if body.out_edges[0] != body else body.out_edges[1]
        
        body.out_edges.remove(outblock)
        outblock.in_edges.remove(body)
        
        new_loop.out_edges.append(outblock)
        outblock.in_edges.append(new_loop)
        
        idx = self.blocks.index(body)
        self.blocks[idx] = new_loop
        return new_loop

    def new_block_inf_loop(self, body: FlowBlock) -> FlowBlock:
        new_loop = BlockInfLoop(body)
        
        for in_block in list(body.in_edges):
            if in_block == body: continue
            idx = in_block.out_edges.index(body)
            in_block.out_edges[idx] = new_loop
            new_loop.in_edges.append(in_block)
            body.in_edges.remove(in_block)
            
        idx = self.blocks.index(body)
        self.blocks[idx] = new_loop
        return new_loop

    def new_block_switch(self, cases: list[FlowBlock], has_exit: bool) -> FlowBlock:
        new_switch = BlockSwitch(cases, has_exit)
        cond = cases[0]
        
        for in_block in list(cond.in_edges):
            idx = in_block.out_edges.index(cond)
            in_block.out_edges[idx] = new_switch
            new_switch.in_edges.append(in_block)
        cond.in_edges.clear()
        
        exit_block = None
        for c in cases:
            for out in list(c.out_edges):
                if out not in cases:
                    exit_block = out
                    c.out_edges.remove(out)
                    out.in_edges.remove(c)
        
        if exit_block:
            new_switch.out_edges.append(exit_block)
            exit_block.in_edges.append(new_switch)
            
        idx = self.blocks.index(cond)
        self.blocks[idx] = new_switch
        for c in cases[1:]:
            if c in self.blocks:
                self.blocks.remove(c)
        return new_switch
        

    def new_block_goto(self, bl: FlowBlock) -> BlockGoto:
        new_goto = BlockGoto(bl.out_edges[0])
        # A BlockGoto replaces bl, but we just link inputs
        for in_block in list(bl.in_edges):
            idx = in_block.out_edges.index(bl)
            in_block.out_edges[idx] = new_goto
            new_goto.in_edges.append(in_block)
        bl.in_edges.clear()
        
        # Link outputs
        for i, out_block in enumerate(list(bl.out_edges)):
            idx = out_block.in_edges.index(bl)
            out_block.in_edges[idx] = new_goto
            new_goto.out_edges.append(out_block)
        bl.out_edges.clear()
        
        self.blocks.remove(bl)
        self.blocks.append(new_goto)
        return new_goto

    def new_block_multigoto(self, bl: FlowBlock, outedge: int) -> BlockMultiGoto:
        if bl.block_type == "multigoto":
            new_goto = bl
        else:
            new_goto = BlockMultiGoto(bl)
            # Link inputs
            for in_block in list(bl.in_edges):
                idx = in_block.out_edges.index(bl)
                in_block.out_edges[idx] = new_goto
                new_goto.in_edges.append(in_block)
            bl.in_edges.clear()
            
            # Link outputs
            for out_block in list(bl.out_edges):
                idx = out_block.in_edges.index(bl)
                out_block.in_edges[idx] = new_goto
                new_goto.out_edges.append(out_block)
            bl.out_edges.clear()
            
            self.blocks.remove(bl)
            self.blocks.append(new_goto)
            
        new_goto.gotoedges.append(new_goto.out_edges[outedge])
        return new_goto

    def new_block_if_goto(self, bl: FlowBlock) -> BlockIfGoto:
        # In Ghidra, IfGoto replaces the branch that is goto
        # Here we just create BlockIfGoto and pass it
        target = bl.out_edges[1] if bl.is_goto_branch(1) else bl.out_edges[0]
        new_ifgoto = BlockIfGoto(bl, target)
        
        for in_block in list(bl.in_edges):
            idx = in_block.out_edges.index(bl)
            in_block.out_edges[idx] = new_ifgoto
            new_ifgoto.in_edges.append(in_block)
        bl.in_edges.clear()
        
        for out_block in list(bl.out_edges):
            idx = out_block.in_edges.index(bl)
            out_block.in_edges[idx] = new_ifgoto
            new_ifgoto.out_edges.append(out_block)
        bl.out_edges.clear()
        
        self.blocks.remove(bl)
        self.blocks.append(new_ifgoto)
        return new_ifgoto

    def new_block_condition(self, b1: FlowBlock, b2: FlowBlock, op: str) -> BlockCondition:
        new_cond = BlockCondition(b1, b2, op)
        
        # In a condition b1 -> b2. We replace b1 with new_cond.
        # But both b1 and b2 are part of the condition.
        for in_block in list(b1.in_edges):
            idx = in_block.out_edges.index(b1)
            in_block.out_edges[idx] = new_cond
            new_cond.in_edges.append(in_block)
        b1.in_edges.clear()
        
        # Output edges are the outputs of b2, plus the output of b1 that doesn't go to b2
        for out in list(b2.out_edges):
            idx = out.in_edges.index(b2)
            out.in_edges[idx] = new_cond
            new_cond.out_edges.append(out)
        b2.out_edges.clear()
        
        for out in list(b1.out_edges):
            if out == b2: continue
            idx = out.in_edges.index(b1)
            out.in_edges[idx] = new_cond
            new_cond.out_edges.append(out)
        
        idx = self.blocks.index(b1)
        self.blocks[idx] = new_cond
        if b2 in self.blocks: self.blocks.remove(b2)
        
        return new_cond

    def new_block_if_no_exit(self, cond: FlowBlock, true_block: FlowBlock, false_block: FlowBlock | None = None) -> FlowBlock:
        new_if = BlockIfNoExit(cond, true_block, false_block)
        
        for in_block in list(cond.in_edges):
            idx = in_block.out_edges.index(cond)
            in_block.out_edges[idx] = new_if
            new_if.in_edges.append(in_block)
        cond.in_edges.clear()
        
        # If there is a false block, it has no exit. If there isn't, the cond's other out edge is the outblock.
        outblock = None
        if false_block is None:
            outblock = cond.out_edges[1] if cond.out_edges[0] == true_block else cond.out_edges[0]
            cond.out_edges.remove(outblock)
            outblock.in_edges.remove(cond)
            new_if.out_edges.append(outblock)
            outblock.in_edges.append(new_if)
        else:
            # both true and false block have no exits. new_if has no out edges
            pass
            
        idx = self.blocks.index(cond)
        self.blocks[idx] = new_if
        if true_block in self.blocks: self.blocks.remove(true_block)
        if false_block and false_block in self.blocks: self.blocks.remove(false_block)
        return new_if

class BlockCondition(FlowBlock):
    def __init__(self, cond1: FlowBlock, cond2: FlowBlock, op: str) -> None:
        super().__init__("condition")
        self.cond1 = cond1
        self.cond2 = cond2
        self.op = op # "&&" or "||"
