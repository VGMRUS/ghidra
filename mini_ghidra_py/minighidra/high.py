from __future__ import annotations
from typing import Optional
from .typesys import CType

class HighVariable:
    """Represents a unified logical variable built from multiple SSA varnodes."""
    
    def __init__(self, high_id: int) -> None:
        self.high_id = high_id
        self.instances: list[int] = []
        self.data_type: Optional[CType] = None
        self.name: Optional[str] = None
        self.is_dirty = True

    def add_instance(self, node_id: int) -> None:
        if node_id not in self.instances:
            self.instances.append(node_id)
