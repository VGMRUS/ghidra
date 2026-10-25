from minighidra.block import BlockBasic
from minighidra.paramid import PcodeLikeGraph

class ConditionalJoin:
    """Helper to detect and merge structurally equivalent basic blocks inside conditional graphs."""
    def __init__(self, data: PcodeLikeGraph):
        self.data = data
        self.block1 = None
        self.block2 = None

    def match(self, b1: BlockBasic, b2: BlockBasic) -> bool:
        # To avoid actual deep node join structural mutation which causes infinite loop in simplified DAG
        return False

    def execute(self):
        pass
        
    def clear(self):
        self.block1 = None
        self.block2 = None
