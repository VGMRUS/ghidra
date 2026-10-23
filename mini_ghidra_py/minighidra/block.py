from __future__ import annotations
from typing import Any

class FlowBlock:
    """Base control-flow structure node."""
    def __init__(self, block_type: str) -> None:
        self.block_type = block_type
        self.in_edges: list[FlowBlock] = []
        self.out_edges: list[FlowBlock] = []
        self.flags = 0

    def get_type(self) -> str:
        return self.block_type

    def add_out_edge(self, target: FlowBlock) -> None:
        self.out_edges.append(target)
        target.in_edges.append(self)

    def remove_out_edge(self, index: int) -> None:
        target = self.out_edges[index]
        self.out_edges.pop(index)
        target.in_edges.remove(self)

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

class BlockGraph(FlowBlock):
    def __init__(self) -> None:
        super().__init__("graph")
        self.blocks: list[FlowBlock] = []

    def build_copy(self, basic_blocks: list[Any]) -> None:
        pass

    def replace_block(self, original: FlowBlock, new_block: FlowBlock) -> None:
        """Splice new_block into the graph replacing original, maintaining edges."""
        idx = self.blocks.index(original)
        self.blocks[idx] = new_block
        
        # We need to remap inputs that were pointing to original
        for in_block in original.in_edges:
            try:
                out_idx = in_block.out_edges.index(original)
                in_block.out_edges[out_idx] = new_block
                new_block.in_edges.append(in_block)
            except ValueError:
                pass
        
        # Outputs that original was pointing to
        for out_block in original.out_edges:
            try:
                in_idx = out_block.in_edges.index(original)
                out_block.in_edges[in_idx] = new_block
                new_block.out_edges.append(out_block)
            except ValueError:
                pass
                
        # Sub-blocks are removed from graph.blocks since they are now encapsulated
        
    def new_block_list(self, nodes: list[FlowBlock]) -> BlockList:
        new_list = BlockList(nodes)
        self.replace_block(nodes[0], new_list)
        # For the internal nodes, we remap the final output edge to be the new_list's output
        last_block = nodes[-1]
        for out in last_block.out_edges:
            try:
                idx = out.in_edges.index(last_block)
                out.in_edges[idx] = new_list
                if new_list not in new_list.out_edges:
                    new_list.out_edges.append(out)
            except ValueError:
                pass
                
        for node in nodes[1:]:
            if node in self.blocks:
                self.blocks.remove(node)
        return new_list

    def new_block_if(self, cond: FlowBlock, body: FlowBlock) -> BlockIf:
        new_if = BlockIf(cond, body)
        self.replace_block(cond, new_if)
        
        if body in self.blocks:
            self.blocks.remove(body)
            
        # The output of body (and the false branch of cond) now becomes the output of new_if
        # This is a simplification.
        return new_if

    def new_block_while_do(self, cond: FlowBlock, body: FlowBlock) -> BlockWhileDo:
        new_loop = BlockWhileDo(cond, body)
        self.replace_block(cond, new_loop)
        
        if body in self.blocks:
            self.blocks.remove(body)
            
        return new_loop

