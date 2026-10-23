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
    """Infers missing/ambiguous types for operations in the graph. (Mock port)"""

    def __init__(self, group: str) -> None:
        super().__init__(group, "infertypes")

    def apply(self, data: PcodeLikeGraph) -> int:
        # MiniGhidra uses a Python-native symbolic evaluator in `decompiler.py` 
        # for semantic inference rather than migrating the whole C++ type propagation 
        # algorithm. We will wire this stub for architectural fidelity.
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
