from __future__ import annotations

class FlowBlock:
    """Base control-flow structure node."""
    def __init__(self, block_type: str) -> None:
        self.block_type = block_type
        self.in_edges: list[FlowBlock] = []
        self.out_edges: list[FlowBlock] = []
        self.flags = 0

    def get_type(self) -> str:
        return self.block_type

class BlockBasic(FlowBlock):
    def __init__(self, start_addr: int) -> None:
        super().__init__("basic")
        self.start_addr = start_addr

class BlockList(FlowBlock):
    def __init__(self) -> None:
        super().__init__("list")
        self.blocks: list[FlowBlock] = []

class BlockIf(FlowBlock):
    def __init__(self) -> None:
        super().__init__("if")
        self.cond_block: FlowBlock | None = None
        self.true_block: FlowBlock | None = None
        self.false_block: FlowBlock | None = None

class BlockWhileDo(FlowBlock):
    def __init__(self) -> None:
        super().__init__("whiledo")
        self.cond_block: FlowBlock | None = None
        self.body_block: FlowBlock | None = None

class BlockDoWhile(FlowBlock):
    def __init__(self) -> None:
        super().__init__("dowhile")
        self.body_block: FlowBlock | None = None

class BlockGraph(FlowBlock):
    def __init__(self) -> None:
        super().__init__("graph")
        self.blocks: list[FlowBlock] = []

    def build_copy(self, basic_blocks: list[Any]) -> None:
        pass
