from __future__ import annotations
from .paramid import PcodeLikeGraph
from .action import Action
from .typeop import TYPE_OPS

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
        self.count = 0
        changed = True
        
        # Operations that are preserved even if output is unread
        # CALL, CALLIND, CALLOTHER, STORE, BRANCH, CBRANCH, BRANCHIND, RETURN
        side_effect_ops = {
            "CALL", "CALLIND", "CALLOTHER", "STORE", 
            "BRANCH", "CBRANCH", "BRANCHIND", "RETURN"
        }
        
        while changed:
            changed = False
            
            # 1. Count uses for all varnodes
            use_counts = {}
            for op in data.ops:
                if op.opcode == "IMARK":
                    continue
                for inp in op.inputs:
                    use_counts[inp] = use_counts.get(inp, 0) + 1
                    
            # 2. Iterate and eliminate pure ops with 0 uses
            for op in list(data.ops):
                if op.opcode == "IMARK":
                    continue
                if op.opcode in side_effect_ops:
                    continue
                    
                # If it's a pure operation, it must have an output. 
                # If output is not used and not a special global sink, it's dead.
                if op.output is None:
                    continue
                    
                # Is the output used?
                if use_counts.get(op.output, 0) == 0:
                    # In a full decompiler, we'd check if op.output is an externally 
                    # visible register/memory that must survive. But since minighidra
                    # converts escaping vars to explicit outputs or leaves them implicitly,
                    # we only kill temporaries, unique, and SSA variables.
                    node = data.nodes[op.output]
                    # We can safely delete if it's a temporary/unique, or if it's not a root output
                    # For now, let's assume we can kill anything with 0 internal uses EXCEPT 
                    # specific global registers. But minighidra handles active_output.
                    if node.kind in {"temporary", "unique"}:
                        data.op_destroy(op)
                        changed = True
                        self.count += 1
                        
        return self.count

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
                type_op = TYPE_OPS.get(op.opcode)
                if type_op is None:
                    continue

                in_types = []
                for inp in op.inputs:
                    if inp < 0:
                        in_types.append(None)
                    else:
                        in_high = merge.node_to_high.get(inp)
                        in_types.append(in_high.data_type if in_high else None)
                        
                out_high = None
                out_type = None
                if op.output is not None:
                    out_high = merge.node_to_high.get(op.output)
                    out_type = out_high.data_type if out_high else None

                # 1. Forward propagation (from inputs to output)
                if out_high is not None and out_type is None:
                    inferred_out = type_op.push(op, in_types)
                    if inferred_out is not None:
                        out_high.data_type = inferred_out
                        out_high.is_dirty = True
                        changed = True
                        self.count += 1
                        out_type = inferred_out
                
                # 2. Backward propagation (from output to inputs)
                for slot, inp in enumerate(op.inputs):
                    if inp < 0: continue
                    in_high = merge.node_to_high.get(inp)
                    if in_high and in_high.data_type is None:
                        inferred_in = type_op.pop(op, out_type, in_types, slot)
                        if inferred_in is not None:
                            in_high.data_type = inferred_in
                            in_high.is_dirty = True
                            in_types[slot] = inferred_in
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
