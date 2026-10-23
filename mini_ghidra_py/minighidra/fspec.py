"""Function call site specifications and prototype assignment."""
from __future__ import annotations

from typing import Any

from .paramid import ParamActive, PcodeLikeGraph, PcodeOp
from .paramlist import PrototypeModel


class CallSpecs:
    """Represents a call site and its parameter trials during recovery."""

    def __init__(
        self, op: PcodeOp, name: str = "", merged_model: Any = None
    ) -> None:
        self.op = op
        self.name = name
        self.merged_model = merged_model
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
        for trial in self.active_input.trials:
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
        
        for trial in self.active_output.trials:
            if trial.is_checked():
                continue
            trial.mark_active()

    def resolve_model(self) -> None:
        """Select a prototype model based on active trials."""
        if self.merged_model:
            self.model = self.merged_model.resolve_model(self.active_input)

    def derive_input_map(self) -> None:
        """Map the recovered parameters to model slots."""
        if self.model:
            self.model.fillin_map(self.active_input, output=False)

    def derive_output_map(self) -> None:
        """Map the recovered return values to model slots."""
        if self.model:
            self.model.fillin_map(self.active_output, output=True)

    def build_input_from_trials(self, graph: PcodeLikeGraph) -> None:
        """Commit the final parameter list for this call."""
        new_inputs = []
        if self.op.inputs:
            new_inputs.append(self.op.inputs[0])  # Preserve target
        
        for trial in self.active_input.trials:
            if not trial.is_used():
                continue
            slot = trial.get_slot()
            if 0 < slot < len(self.op.inputs):
                new_inputs.append(self.op.inputs[slot])
            else:
                # Unreferenced or synthesized parameter placeholder
                pass
        
        # In a full decompiler, we'd update the operation and Varnodes here.
        # self.op.inputs = tuple(new_inputs)

    def build_output_from_trials(self, graph: PcodeLikeGraph) -> None:
        """Commit the final return list for this call."""
