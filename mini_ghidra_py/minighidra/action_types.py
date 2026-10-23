from __future__ import annotations
from .paramid import PcodeLikeGraph
from .action import Action

class ActionHeritage(Action):
    """Links uses to definitions and constructs Single Static Assignment (SSA) form."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "heritage")

    def apply(self, data: PcodeLikeGraph) -> int:
        # In MiniGhidra, `PcodeLikeGraph.from_ssa` already consumes a pre-heritaged 
        # `SSAFunction` for ParamID and control flow. This action serves as the 
        # architectural bridge in the Action framework pipeline.
        self.count = 0
        return 0

class ActionDeadCode(Action):
    """Eliminates p-code operations whose results are never consumed."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "deadcode")

    def apply(self, data: PcodeLikeGraph) -> int:
        # MiniGhidra prunes unread sinks implicitly in the fallback symbolic renderer,
        # but a formal dead code action would iteratively remove PcodeOps without 
        # valid read references or side-effects.
        self.count = 0
        return 0

class ActionInferTypes(Action):
    """Infers missing/ambiguous types for operations in the graph."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "infertypes")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if not hasattr(data, "merge"):
            return 0
        
        merge = getattr(data, "merge")
        
        # Iterative type propagation algorithm (Forward and Backward)
        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            iterations += 1
            
            for op in data.ops:
                # 1. Forward propagation (from inputs to output)
                if op.output is not None:
                    out_high = merge.node_to_high.get(op.output)
                    if out_high and out_high.data_type is None:
                        # Attempt to infer from inputs
                        for inp in op.inputs:
                            if inp < 0: continue
                            in_high = merge.node_to_high.get(inp)
                            if in_high and in_high.data_type is not None:
                                # Simplified: inherit type directly or apply operation-specific typing
                                if op.opcode in ("COPY", "MULTIEQUAL", "INT_ADD", "INT_SUB"):
                                    out_high.data_type = in_high.data_type
                                    out_high.is_dirty = True
                                    changed = True
                                    self.count += 1
                                    break
                
                # 2. Backward propagation (from output to inputs)
                if op.output is not None:
                    out_high = merge.node_to_high.get(op.output)
                    if out_high and out_high.data_type is not None:
                        for inp in op.inputs:
                            if inp < 0: continue
                            in_high = merge.node_to_high.get(inp)
                            if in_high and in_high.data_type is None:
                                if op.opcode in ("COPY", "MULTIEQUAL"):
                                    in_high.data_type = out_high.data_type
                                    in_high.is_dirty = True
                                    changed = True
                                    self.count += 1

            # Mark all dirty highs as clean
            for high in merge.highs:
                if high.is_dirty:
                    high.is_dirty = False
                    
        return 0

class ActionSetCasts(Action):
    """Inserts explicit cast operations into the AST where types clash."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "setcasts")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionPrototypeTypes(Action):
    """Assigns or enforces the prototype structure for standard library sub-calls."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "prototypetypes")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionDefaultParams(Action):
    """Sets a default prototype model for calls that lack explicit specs."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "defaultparams")

    def apply(self, data: PcodeLikeGraph) -> int:
        # In minighidra, call_specs inherently default via calling_convention mapping 
        # before they hit actions, so this acts as a formalization step.
        self.count = 0
        return 0

class ActionPrototypeWarnings(Action):
    """Adds warnings for prototypes that aren't modeled properly."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "prototypewarnings")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionStop(Action):
    """Terminates the action pipeline."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "stop")

    def apply(self, data: PcodeLikeGraph) -> int:
        return -1
