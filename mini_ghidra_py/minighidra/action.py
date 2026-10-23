"""Universal decompiler actions and rules pipeline."""
from __future__ import annotations

from .paramid import PcodeLikeGraph


class Action:
    """Base class for an analysis pass over the function's p-code graph."""

    def __init__(self, group: str, name: str) -> None:
        self.group = group
        self.name = name
        self.count = 0

    def apply(self, data: PcodeLikeGraph) -> int:
        raise NotImplementedError


class ActionActiveParam(Action):
    """Determine which parameters are actively read by subfunctions."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "activeparam")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        for call_spec in data.call_specs:
            if not call_spec.is_input_active:
                continue

            active_input = call_spec.active_input
            trimmable = (active_input.num_passes() > 0) or call_spec.op.opcode != "CALLIND"

            if not active_input.is_fully_checked():
                call_spec.check_input_trial_use(data)

            active_input.finish_pass()
            if active_input.num_passes() > active_input.max_pass():
                active_input.mark_fully_checked()
            else:
                self.count += 1

            if trimmable and active_input.is_fully_checked():
                if active_input.needs_final_check():
                    # call_spec.final_input_check()
                    pass
                call_spec.resolve_model()
                call_spec.derive_input_map()
                call_spec.build_input_from_trials(data)
                call_spec.clear_active_input()
                self.count += 1

        return 0


class ActionActiveReturn(Action):
    """Determine which subfunction return values are actively used."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "activereturn")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        for call_spec in data.call_specs:
            if not call_spec.is_output_active:
                continue

            call_spec.check_output_trial_use(data)
            call_spec.derive_output_map()
            call_spec.build_output_from_trials(data)
            call_spec.clear_active_output()
            self.count += 1

        return 0


class ActionReturnRecovery(Action):
    """Determine which return trials for the current function are actively set."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "returnrecovery")

    def apply(self, data: PcodeLikeGraph) -> int:
        active = getattr(data, "active_output", None)
        if not active:
            return 0

        self.count = 0
        returns = [op for op in data.ops if op.opcode == "RETURN"]

        for op in returns:
            for trial in active.trials():
                if trial.is_checked():
                    continue
                # For each RETURN op, we would trace backwards to see if the value
                # returned in this trial slot is actually defined intentionally.
                # In minighidra, we delegate to measure_output.
                trial.mark_active()
                self.count += 1

        active.finish_pass()
        if active.num_passes() > active.max_pass():
            active.mark_fully_checked()

        if active.is_fully_checked():
            # deriveOutputMap and buildReturnOutput would happen here.
            # data.clear_active_output()
            self.count += 1

        return 0
