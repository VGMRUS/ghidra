from __future__ import annotations
from .paramid import PcodeLikeGraph
from .action import Action
from .merge import Merge

class ActionNameVars(Action):
    """Assigns concrete names to HighVariables using database recommendations."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "namevars")

    def apply(self, data: PcodeLikeGraph) -> int:
        if hasattr(data, "merge"):
            merge: Merge = getattr(data, "merge")
            for high in merge.highs:
                if high.name is None:
                    high.name = f"var_{high.high_id}"
                    self.count += 1
        return 0

class ActionDynamicSymbols(Action):
    """Links explicitly named dynamic symbols to variables in the graph."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "dynamicsymbols")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionRestructureVarnode(Action):
    """Re-synchronizes varnodes with symbol scope."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "restructurevarnode")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionMappedLocalSync(Action):
    """Synchronizes mapped locals with their underlying varnodes."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "mappedlocalsync")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionMapGlobals(Action):
    """Maps global variables based on references discovered during analysis."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "mapglobals")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionSpacebase(Action):
    """Establishes spacebase references (like stack or register tracking)."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "spacebase")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        SPACEBASE_NAMES = {"RSP", "ESP", "SP", "RBP", "EBP", "BP"}
        
        for vn, node in enumerate(data.nodes):
            if not node: continue
            name_upper = (node.name or "").upper()
            
            # If the node represents a known spacebase or spacebase_in
            if any(name_upper == base or name_upper.startswith(base + "_IN") for base in SPACEBASE_NAMES):
                object.__setattr__(node, "is_spacebase", True)
                
                def_op = data.get_def(vn)
                if def_op is not None and def_op.opcode == "INT_ADD":
                    uses = data.get_uses(vn)
                    if len(uses) > 1:
                        # Split uses: clone the ADD operation for each extra use
                        for use_op in uses[1:]:
                            new_vn = data._new_temporary_node(f"{node.name}_split", node.size_bits)
                            object.__setattr__(data.nodes[new_vn], "is_spacebase", True)
                            
                            new_add = PcodeOp()
                            new_add.opcode = "INT_ADD"
                            new_add.inputs = tuple(def_op.inputs)
                            new_add.address = def_op.address
                            new_add.output = new_vn
                            data._append_operation(new_add)
                            
                            # Update the use
                            idx = list(use_op.inputs).index(vn)
                            data.op_set_input(use_op, new_vn, idx)
                        
                        self.count += 1
                        
        return 0


class ActionStackPtrFlow(Action):
    """Analyzes stack pointer flow and aligns stack usage across the function."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "stackptrflow")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionUnjustifiedParams(Action):
    """Adjusts unjustified input parameters (e.g. unaligned parameters) to their proper containers."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "unjustifiedparams")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionHideShadow(Action):
    """Hides redundant variables overshadowed by others during merging."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "hideshadow")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionDynamicMapping(Action):
    """Resolves dynamically mapped symbols from hash tracking (like local variables discovered late)."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "dynamicmapping")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0
