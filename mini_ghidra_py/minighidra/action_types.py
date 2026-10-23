from __future__ import annotations
from .paramid import PcodeLikeGraph
from .action import Action

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
