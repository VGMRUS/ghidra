from __future__ import annotations
from typing import Any
from .paramid import PcodeLikeGraph
from .high import HighVariable

class Merge:
    """Manages grouping of Varnodes into HighVariables."""
    
    def __init__(self, data: PcodeLikeGraph) -> None:
        self.data = data
        self.highs: list[HighVariable] = []
        self.node_to_high: dict[int, HighVariable] = {}
        self._next_high_id = 1

    def new_high(self) -> HighVariable:
        h = HighVariable(self._next_high_id)
        self._next_high_id += 1
        self.highs.append(h)
        return h

    def assign_highs(self) -> None:
        """Assigns every valid varnode its own initial HighVariable."""
        for i, _node in enumerate(self.data.nodes):
            if i not in self.node_to_high:
                h = self.new_high()
                h.add_instance(i)
                self.node_to_high[i] = h

    def merge_op(self, opcode: str) -> None:
        """Merges inputs and outputs of specific operations (like MULTIEQUAL or COPY)."""
        for op in self.data.ops:
            if op.opcode == opcode and op.output is not None:
                out_high = self.node_to_high.get(op.output)
                if not out_high:
                    continue
                for inp in op.inputs:
                    in_high = self.node_to_high.get(inp)
                    if in_high and in_high != out_high:
                        # Merge in_high into out_high
                        for n in in_high.instances:
                            out_high.add_instance(n)
                            self.node_to_high[n] = out_high
                        if in_high in self.highs:
                            self.highs.remove(in_high)

    def merge_adjacent(self) -> None:
        pass

    def mark_explicit_types(self) -> None:
        pass

    def mark_implied_types(self) -> None:
        pass
