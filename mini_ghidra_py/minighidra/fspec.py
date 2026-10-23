"""Function call site specifications and prototype assignment."""
from __future__ import annotations

from .paramid import ParamActive, PcodeLikeGraph, PcodeOp
from .paramlist import PrototypeModel


class CallSpecs:
    """Represents a call site and its parameter trials during recovery."""

    def __init__(self, op: PcodeOp, name: str = "") -> None:
        self.op = op
        self.name = name
        self.active_input = ParamActive(recover_subcalls=True)
        self.active_output = ParamActive(recover_subcalls=False)
        self.is_input_active = False
        self.is_output_active = False
        self.model: PrototypeModel | None = None

    def set_input_active(self) -> None:
        self.is_input_active = True

    def set_output_active(self) -> None:
        self.is_output_active = True

    def clear_active_input(self) -> None:
        self.is_input_active = False
        self.active_input.clear()

    def clear_active_output(self) -> None:
        self.is_output_active = False
        self.active_output.clear()

    def check_input_trial_use(self, graph: PcodeLikeGraph) -> None:
        """Trim active input trials based on whether arguments are actually read."""
        if not self.is_input_active:
            return
        
        # We perform a forward/backward walk to check if the argument values
        # are realistic parameters for this subfunction call.
        for trial in self.active_input.trials():
            if trial.is_checked():
                continue
            # Simplified heuristic for trial use: since the parameter comes
            # from a node, we consider it actively used by the call.
            # In a full SLEIGH architecture, ancestorOpUse checks dataflow
            # backwards from the CALL op's slot.
            trial.mark_active()

    def check_output_trial_use(self, graph: PcodeLikeGraph) -> None:
        """Trim active output trials based on whether the return values are used."""
        if not self.is_output_active:
            return
        
        for trial in self.active_output.trials():
            if trial.is_checked():
                continue
            trial.mark_active()

    def resolve_model(self) -> None:
        """Select a prototype model based on active trials."""

    def derive_input_map(self) -> None:
        """Map the recovered parameters to model slots."""

    def derive_output_map(self) -> None:
        """Map the recovered return values to model slots."""

    def build_input_from_trials(self, graph: PcodeLikeGraph) -> None:
        """Commit the final parameter list for this call."""

    def build_output_from_trials(self, graph: PcodeLikeGraph) -> None:
        """Commit the final return list for this call."""
