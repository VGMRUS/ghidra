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


class ActionGroup(Action):
    """An action that groups multiple sub-actions and applies them sequentially."""

    def __init__(self, group: str, name: str) -> None:
        super().__init__(group, name)
        self.actions: list[Action] = []

    def add_action(self, action: Action) -> None:
        self.actions.append(action)

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        for action in self.actions:
            res = action.apply(data)
            if res > 0:
                self.count += res
        return 0


class ActionDatabase:
    """Registry and orchestrator of decompiler actions."""

    def __init__(self) -> None:
        self.groups: dict[str, ActionGroup] = {}
        self.current_action: ActionGroup | None = None

    def register_action(self, name: str, action: ActionGroup) -> None:
        self.groups[name] = action

    def universal_action(self) -> None:
        """Configure the universal action group with the standard passes."""
        from .action_types import (
            ActionDeadCode,
            ActionDefaultParams,
            ActionHeritage,
            ActionInferTypes,
            ActionPrototypeTypes,
        )
        from .action_varnode import ActionVarnodeProps, ActionDirectWrite
        from .action_controlflow import ActionUnreachable, ActionForceGoto, ActionRestrictLocal
        from .blockaction import ActionBlockStructure, ActionFinalStructure, ActionConditionalConst, ActionSwitchNorm

        from .rule import (
            ActionPool,
            RuleOrConsume, RuleNegateIdentity, RuleShiftBitops, RuleRightShiftAnd, RuleEquality, RuleIntLessEqualRewrite, RuleHighOrderAnd, RuleZextCommute, RuleZextShiftZext, RuleAndDistribute, RuleLessOne, RuleAndCommute, RuleAndPiece, RuleAndZext, RuleAndCompare, RuleDoubleArithShift, RuleConcatShift, RuleLeftRight, RuleShiftCompare, RuleShiftPiece, RuleZextEliminate, RuleSlessToLess, RuleZextSless, RuleBitUndistribute, RuleLogic2Bool, RuleNotDistribute, RuleBoolNegate, RuleSubCommute, RuleConcatCommute, RuleConcatZero, RuleConcatZext, RuleMultNegOne, Rule2Comp2Sub, RuleBooleanUndistribute, RuleBooleanDedup, RuleEqual2Zero, RuleSignShift, RuleTestSign, RuleSubZext, RuleDumptyHump, RuleHumptyOr, RuleHumptyDumpty, RuleBooleanNegate, RuleXorSwap, RuleOrMask, RuleTrivialShift, RuleLessNotEqual, RuleNegateNegate, RuleOrCompare, RuleShiftSub, RuleSubRight, RuleAddUnsigned, RuleSubExtComm, RuleSubNormal, RuleCarryElim, RuleScarry, RuleSborrow, RuleDivChain, RuleSignForm, RuleSignNearMult, RuleModOpt, RuleSignMod2nOpt, RuleSignMod2Opt, RuleSignMod2nOpt2, RulePositiveDiv, RuleDivTermAdd, RuleDivOpt, RuleSignDiv2, RuleDivTermAdd2, RuleConcatLeftShift, RuleExpandLoad, RuleExtensionPush, RuleStructOffset0, RulePtrsubCharConstant, RuleRightShiftSub, RuleConditionalMove, RulePullsubIndirect, RulePushMulti, RulePushPtr, RulePtrsubUndo, RulePtraddUndo, RulePiecePathology, RulePieceStructure, RulePopcountBoolXor, RulePtrArith, RulePtrFlow, RuleRangeMeld, RuleSegment, RuleSelectCse, RuleCollectTerms, RuleEarlyRemoval, RuleFloatCast, RuleIgnoreNan, RuleUnsigned2Float, RuleInt2FloatCollapse, RuleFuncPtrEncoding, RuleLzcountShiftBool, RuleFloatSign, RuleFloatSignCleanup, RuleFloatRange, RuleTransformCpool, RuleTrivialArith, RuleXorCollapse, RuleAddMultCollapse, RulePiece2Zext, RulePiece2Sext, RuleTermOrder, RuleBxor2NotEqual, RuleAndOrLump, RuleOrCollapse, RuleAndMask, RuleLogicIdentity, RuleDoubleLoad, RulePullsubMulti, RuleSub2Add, RuleTrivialBool, RuleLessEqual, RuleCondNegate, RuleDoubleShift, RuleIdentityEl, RuleSubCancel, RuleDoubleSub, RuleShift2Mult, RulePropagateCopy, Rule2Comp2Mult, RuleShiftAnd, RuleLess2Zero, RuleLessEqual2Zero, RuleEqual2Constant, RuleSLess2Zero
        )

        universal = ActionGroup("universal", "universal")
        self.register_action("universal", universal)

        # Standard decompilation phases ported so far:
        # 0. SSA Heritage and Cleanup
        universal.add_action(ActionUnreachable("base"))
        universal.add_action(ActionVarnodeProps("base"))
        universal.add_action(ActionHeritage("base"))
        
        # 0.1 AST Simplification Pool
        actprop = ActionPool("oppool1", "oppool1")
        actprop.add_rule(RuleTrivialArith("analysis"))
        actprop.add_rule(RuleXorCollapse("analysis"))
        actprop.add_rule(RuleAddMultCollapse("analysis"))
        actprop.add_rule(RuleSub2Add("analysis"))
        actprop.add_rule(RulePiece2Zext("analysis"))
        actprop.add_rule(RulePiece2Sext("analysis"))
        actprop.add_rule(RuleTermOrder("analysis"))
        actprop.add_rule(RuleBxor2NotEqual("analysis"))
        actprop.add_rule(RuleAndOrLump("analysis"))
        actprop.add_rule(RuleOrCollapse("analysis"))
        actprop.add_rule(RuleAndMask("analysis"))
        actprop.add_rule(RuleLogicIdentity("analysis"))
        actprop.add_rule(RuleDoubleLoad("analysis"))
        actprop.add_rule(RulePullsubMulti("analysis"))
        actprop.add_rule(RuleLessOne("analysis"))
        actprop.add_rule(RuleZextEliminate("analysis"))
        actprop.add_rule(RuleSlessToLess("analysis"))
        universal.add_action(actprop)

        universal.add_action(ActionForceGoto("blockrecovery"))
        universal.add_action(ActionDirectWrite("protorecovery_a", True))
        universal.add_action(ActionDirectWrite("protorecovery_b", False))
        
        # 1. Parameter discovery and Call site trimming
        universal.add_action(ActionDefaultParams("base"))
        universal.add_action(ActionActiveParam("protorecovery"))
        universal.add_action(ActionReturnRecovery("protorecovery"))
        universal.add_action(ActionRestrictLocal("localrecovery"))
        
        universal.add_action(ActionDeadCode("deadcode"))

        # 2. Block structure and flow
        universal.add_action(ActionConditionalConst("analysis"))
        universal.add_action(ActionSwitchNorm("switchnorm"))
        universal.add_action(ActionBlockStructure("blockrecovery"))

        # 3. HighVariable Merging (converting SSA Varnodes to logical HighVariables)
        from .action_merge import (
            ActionAssignHigh, ActionMergeRequired, ActionMarkExplicit,
            ActionMarkImplied, ActionMergeMultiEntry, ActionMergeCopy,
            ActionMergeAdjacent, ActionMergeType
        )
        universal.add_action(ActionAssignHigh("merge"))
        universal.add_action(ActionMergeRequired("merge"))
        universal.add_action(ActionMarkExplicit("merge"))
        universal.add_action(ActionMarkImplied("merge"))
        universal.add_action(ActionMergeMultiEntry("merge"))
        universal.add_action(ActionMergeCopy("merge"))
        universal.add_action(ActionMergeAdjacent("merge"))
        universal.add_action(ActionMergeType("merge"))

        # 4. Basic Prototype and Type Enforcement
        universal.add_action(ActionActiveReturn("protorecovery"))
        universal.add_action(ActionPrototypeTypes("base"))
        universal.add_action(ActionInferTypes("base"))

        universal.add_action(ActionFinalStructure("blockrecovery"))

        # 5. Variable Naming and Dynamic Symbols
        from .action_name import ActionNameVars, ActionDynamicSymbols
        from .action_types import ActionSetCasts, ActionPrototypeWarnings, ActionStop

        universal.add_action(ActionDynamicSymbols("dynamic"))
        universal.add_action(ActionNameVars("merge"))
        universal.add_action(ActionSetCasts("casts"))
        universal.add_action(ActionPrototypeWarnings("protorecovery"))
        universal.add_action(ActionStop("base"))

        self.current_action = universal


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
            for trial in active.trials:
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
