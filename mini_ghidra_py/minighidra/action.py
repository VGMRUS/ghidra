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



class ActionConditionalExe(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "conditionalexe")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionConstantPtr(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "constantptr")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionConstbase(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "constbase")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionCopyMarker(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "copymarker")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionCse(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "cse")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionDeindirect(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "deindirect")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionDeterminedBranch(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "determinedbranch")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionDoNothing(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "donothing")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionDominantCopy(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "dominantcopy")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionDynamicMapping(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "dynamicmapping")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionExtraPopSetup(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "extrapopsetup")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionFuncLink(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "funclink")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionFuncLinkOutOnly(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "funclinkoutonly")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionHideShadow(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "hideshadow")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionInputPrototype(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "inputprototype")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionInternalStorage(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "internalstorage")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionLaneDivide(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "lanedivide")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionLateDoNothing(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "latedonothing")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionLikelyTrash(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "likelytrash")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionMapGlobals(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "mapglobals")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionMappedLocalSync(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "mappedlocalsync")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionMarkIndirectOnly(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "markindirectonly")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionMergeMultiEntry(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "mergemultientry")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionMultiCse(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "multicse")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionNodeJoin(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "nodejoin")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionNonzeroMask(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "nonzeromask")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionNormalizeBranches(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "normalizebranches")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionNormalizeSetup(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "normalizesetup")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionOutputPrototype(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "outputprototype")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionParamDouble(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "paramdouble")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionParamShiftStart(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "paramshiftstart")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionParamShiftStop(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "paramshiftstop")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionPreferComplement(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "prefercomplement")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionRedundBranch(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "redundbranch")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionRestructureVarnode(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "restructurevarnode")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionReturnSplit(Action):
    """This action splits return blocks to avoid overlapping return conditions."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "returnsplit")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if not hasattr(data, "structure") or not hasattr(data, "ssa_function"):
            return 0
            
        returns = [op for op in data.ops if op.opcode == "RETURN" and op.output is None]
        for retop in returns:
            bb = next((b for b in data.ssa_function.blocks if b.start <= retop.addr < b.start + b.size), None)
            if not bb or len(bb.in_edges) <= 1: continue
            
            # Simple check if splittable
            # In Python, we just check if it only contains COPY and RETURN and MULTIEQUAL
            # but since MULTIEQUAL is mostly implicit, we just check ops.
            ops_in_b = [op for op in data.ops if op.addr >= bb.start and op.addr < bb.start + bb.size]
            splittable = True
            for op in ops_in_b:
                if op.opcode not in ("COPY", "RETURN", "MULTIEQUAL", "IMARK"):
                    splittable = False
                    break
            if not splittable: continue
            
            # Split off the first incoming edge to a clone block (for simplicity)
            # In a full implementation, we'd rebuild the BB logic. Here we just count it.
            # self.count += 1
            # data.node_split(...) -> disabled due to architecture complexity of BB rebuilding
        
        return self.count

class ActionSegmentize(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "segmentize")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionShadowVar(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "shadowvar")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionSpacebase(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "spacebase")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionStackPtrFlow(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "stackptrflow")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionStart(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "start")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionStartCleanUp(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "startcleanup")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionStartTypes(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "starttypes")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionStructureTransform(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "structuretransform")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
        return 0

class ActionUnjustifiedParams(Action):
    """This action is currently stubbed/disabled in the Python port."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "unjustifiedparams")
        self.disabled = True
    def apply(self, data: PcodeLikeGraph) -> int:
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
        from .rule import ActionPool, Rule2Comp2Mult, Rule2Comp2Sub, RuleAddMultCollapse, RuleAddUnsigned, RuleAliasUpdate, RuleAndCommute, RuleAndCompare, RuleAndDistribute, RuleAndMask, RuleAndOrLump, RuleAndPiece, RuleAndStructure, RuleAndZext, RuleBitFieldIn, RuleBitFieldLoad, RuleBitFieldOut, RuleBitFieldStore, RuleBitUndistribute, RuleBoolNegate, RuleBoolZext, RuleBooleanDedup, RuleBooleanNegate, RuleBooleanUndistribute, RuleBxor2NotEqual, RuleCarryElim, RuleCollapseConstants, RuleCollectTerms, RuleConcatCommute, RuleConcatLeftShift, RuleConcatShift, RuleConcatZero, RuleConcatZext, RuleCondNegate, RuleConditionalMove, RuleDivChain, RuleDivOpt, RuleDivTermAdd, RuleDivTermAdd2, RuleDoubleArithShift, RuleDoubleIn, RuleDoubleLoad, RuleDoubleOut, RuleDoubleShift, RuleDoubleStore, RuleDoubleSub, RuleDumptyHump, RuleDumptyHumpLate, RuleEarlyRemoval, RuleEqual2Constant, RuleEqual2Zero, RuleExpandLoad, RuleExtensionPush, RuleFloatCast, RuleFloatRange, RuleFloatSign, RuleFloatSignCleanup, RuleFuncPtrEncoding, RuleHighOrderAnd, RuleHumptyDumpty, RuleHumptyOr, RuleIdentityEl, RuleIgnoreNan, RuleIndirectConcat, RuleInsertAbsorb, RuleInt2FloatCollapse, RuleIntLessEqual, RuleLeftRight, RuleLess2Zero, RuleLessEqual, RuleLessEqual2Zero, RuleLessNotEqual, RuleLessOne, RuleLoadVarnode, RuleLogic2Bool, RuleLzcountShiftBool, RuleModOpt, RuleMultNegOne, RuleMultiCollapse, RuleNegateIdentity, RuleNegateNegate, RuleNotDistribute, RuleOrCollapse, RuleOrCompare, RuleOrConsume, RuleOrMask, RuleOrPredicate, RulePiece2Sext, RulePiece2Zext, RulePiecePathology, RulePieceStructure, RulePopcountBoolXor, RulePositiveDiv, RulePropagateCopy, RulePtrArith, RulePtrFlow, RulePtraddUndo, RulePtrsubCharConstant, RulePtrsubUndo, RulePullAbsorb, RulePullsubIndirect, RulePullsubMulti, RulePushMulti, RulePushPtr, RuleRangeMeld, RuleRightShiftAnd, RuleSLess2Zero, RuleSborrow, RuleScarry, RuleSegment, RuleSelectCse, RuleShift2Mult, RuleShiftAnd, RuleShiftBitops, RuleShiftCompare, RuleShiftPiece, RuleShiftSub, RuleSignDiv2, RuleSignForm, RuleSignForm2, RuleSignMod2Opt, RuleSignMod2nOpt, RuleSignMod2nOpt2, RuleSignNearMult, RuleSignShift, RuleSlessToLess, RuleSplitCopy, RuleSplitFlow, RuleSplitLoad, RuleSplitStore, RuleStoreVarnode, RuleStringCopy, RuleStringStore, RuleStructOffset0, RuleSub2Add, RuleSubCancel, RuleSubCommute, RuleSubExtComm, RuleSubNormal, RuleSubRight, RuleSubZext, RuleSubfloatConvert, RuleSubvarAnd, RuleSubvarCompZero, RuleSubvarSext, RuleSubvarShift, RuleSubvarSubpiece, RuleSubvarZext, RuleSwitchSingle, RuleTermOrder, RuleTestSign, RuleThreeWayCompare, RuleTransformCpool, RuleTrivialArith, RuleTrivialBool, RuleTrivialShift, RuleUnsigned2Float, RuleXorCollapse, RuleXorSwap, RuleZextCommute, RuleZextEliminate, RuleZextShiftZext, RuleZextSless
        from .action_types import (ActionDeadCode, ActionDefaultParams, ActionPrototypeWarnings, ActionHeritage, ActionStop, ActionInferTypes, ActionSetCasts, ActionPrototypeTypes)
        from .action_name import (ActionNameVars, ActionDynamicSymbols)
        from .blockaction import (ActionSwitchNorm, ActionBlockStructure, ActionConditionalConst, ActionFinalStructure)
        from .action_merge import (ActionMergeType, ActionMergeRequired, ActionMarkExplicit, ActionMergeCopy, ActionMergeAdjacent, ActionMarkImplied, ActionAssignHigh)
        from .action_controlflow import (ActionNormalizeBranches, ActionUnreachable, ActionForceGoto, ActionRestrictLocal)
        from .action_varnode import (ActionDirectWrite, ActionVarnodeProps)

        # 0.1 AST Simplification Pool
        actprop = ActionPool('actprop', 'actprop')
        actprop.add_rule(RuleEarlyRemoval('deadcode'))
        actprop.add_rule(RuleTermOrder('analysis'))
        actprop.add_rule(RuleSelectCse('analysis'))
        actprop.add_rule(RuleCollectTerms('analysis'))
        actprop.add_rule(RulePullsubMulti('analysis'))
        actprop.add_rule(RulePullsubIndirect('analysis'))
        actprop.add_rule(RulePushMulti('nodejoin'))
        actprop.add_rule(RuleSborrow('analysis'))
        actprop.add_rule(RuleScarry('analysis'))
        actprop.add_rule(RuleIntLessEqual('analysis'))
        actprop.add_rule(RuleTrivialArith('analysis'))
        actprop.add_rule(RuleTrivialBool('analysis'))
        actprop.add_rule(RuleTrivialShift('analysis'))
        actprop.add_rule(RuleSignShift('analysis'))
        actprop.add_rule(RuleTestSign('analysis'))
        actprop.add_rule(RuleIdentityEl('analysis'))
        actprop.add_rule(RuleOrMask('analysis'))
        actprop.add_rule(RuleAndMask('analysis'))
        actprop.add_rule(RuleOrConsume('analysis'))
        actprop.add_rule(RuleOrCollapse('analysis'))
        actprop.add_rule(RuleAndOrLump('analysis'))
        actprop.add_rule(RuleShiftBitops('analysis'))
        actprop.add_rule(RuleRightShiftAnd('analysis'))
        actprop.add_rule(RuleNotDistribute('analysis'))
        actprop.add_rule(RuleHighOrderAnd('analysis'))
        actprop.add_rule(RuleAndDistribute('analysis'))
        actprop.add_rule(RuleAndCommute('analysis'))
        actprop.add_rule(RuleAndPiece('analysis'))
        actprop.add_rule(RuleAndZext('analysis'))
        actprop.add_rule(RuleAndCompare('analysis'))
        actprop.add_rule(RuleDoubleSub('analysis'))
        actprop.add_rule(RuleDoubleShift('analysis'))
        actprop.add_rule(RuleDoubleArithShift('analysis'))
        actprop.add_rule(RuleConcatShift('analysis'))
        actprop.add_rule(RuleLeftRight('analysis'))
        actprop.add_rule(RuleShiftCompare('analysis'))
        actprop.add_rule(RuleShift2Mult('analysis'))
        actprop.add_rule(RuleShiftPiece('analysis'))
        actprop.add_rule(RuleMultiCollapse('analysis'))
        actprop.add_rule(RuleAliasUpdate('analysis'))
        actprop.add_rule(Rule2Comp2Mult('analysis'))
        actprop.add_rule(RuleSub2Add('analysis'))
        actprop.add_rule(RuleCarryElim('analysis'))
        actprop.add_rule(RuleBxor2NotEqual('analysis'))
        actprop.add_rule(RuleLess2Zero('analysis'))
        actprop.add_rule(RuleLessEqual2Zero('analysis'))
        actprop.add_rule(RuleSLess2Zero('analysis'))
        actprop.add_rule(RuleEqual2Zero('analysis'))
        actprop.add_rule(RuleEqual2Constant('analysis'))
        actprop.add_rule(RuleThreeWayCompare('analysis'))
        actprop.add_rule(RuleXorCollapse('analysis'))
        actprop.add_rule(RuleAddMultCollapse('analysis'))
        actprop.add_rule(RuleCollapseConstants('analysis'))
        actprop.add_rule(RuleTransformCpool('analysis'))
        actprop.add_rule(RulePropagateCopy('analysis'))
        actprop.add_rule(RuleZextEliminate('analysis'))
        actprop.add_rule(RuleSlessToLess('analysis'))
        actprop.add_rule(RuleZextSless('analysis'))
        actprop.add_rule(RuleBitUndistribute('analysis'))
        actprop.add_rule(RuleBooleanUndistribute('analysis'))
        actprop.add_rule(RuleBooleanDedup('analysis'))
        actprop.add_rule(RuleBoolZext('analysis'))
        actprop.add_rule(RuleBooleanNegate('analysis'))
        actprop.add_rule(RuleLogic2Bool('analysis'))
        actprop.add_rule(RuleSubExtComm('analysis'))
        actprop.add_rule(RuleSubCommute('analysis'))
        actprop.add_rule(RuleConcatCommute('analysis'))
        actprop.add_rule(RuleConcatZext('analysis'))
        actprop.add_rule(RuleZextCommute('analysis'))
        actprop.add_rule(RuleZextShiftZext('analysis'))
        actprop.add_rule(RuleShiftAnd('analysis'))
        actprop.add_rule(RuleConcatZero('analysis'))
        actprop.add_rule(RuleConcatLeftShift('analysis'))
        actprop.add_rule(RuleSubZext('analysis'))
        actprop.add_rule(RuleSubCancel('analysis'))
        actprop.add_rule(RuleShiftSub('analysis'))
        actprop.add_rule(RuleHumptyDumpty('analysis'))
        actprop.add_rule(RuleDumptyHump('analysis'))
        actprop.add_rule(RuleHumptyOr('analysis'))
        actprop.add_rule(RuleNegateIdentity('analysis'))
        actprop.add_rule(RuleSubNormal('analysis'))
        actprop.add_rule(RulePositiveDiv('analysis'))
        actprop.add_rule(RuleDivTermAdd('analysis'))
        actprop.add_rule(RuleDivTermAdd2('analysis'))
        actprop.add_rule(RuleDivOpt('analysis'))
        actprop.add_rule(RuleSignForm('analysis'))
        actprop.add_rule(RuleSignForm2('analysis'))
        actprop.add_rule(RuleSignDiv2('analysis'))
        actprop.add_rule(RuleDivChain('analysis'))
        actprop.add_rule(RuleSignNearMult('analysis'))
        actprop.add_rule(RuleModOpt('analysis'))
        actprop.add_rule(RuleSignMod2nOpt('analysis'))
        actprop.add_rule(RuleSignMod2nOpt2('analysis'))
        actprop.add_rule(RuleSignMod2Opt('analysis'))
        actprop.add_rule(RuleSwitchSingle('analysis'))
        actprop.add_rule(RuleCondNegate('analysis'))
        actprop.add_rule(RuleBoolNegate('analysis'))
        actprop.add_rule(RuleLessEqual('analysis'))
        actprop.add_rule(RuleLessNotEqual('analysis'))
        actprop.add_rule(RuleLessOne('analysis'))
        actprop.add_rule(RuleRangeMeld('analysis'))
        actprop.add_rule(RuleFloatRange('analysis'))
        actprop.add_rule(RulePiece2Zext('analysis'))
        actprop.add_rule(RulePiece2Sext('analysis'))
        actprop.add_rule(RulePopcountBoolXor('analysis'))
        actprop.add_rule(RuleXorSwap('analysis'))
        actprop.add_rule(RuleLzcountShiftBool('analysis'))
        actprop.add_rule(RuleFloatSign('analysis'))
        actprop.add_rule(RuleOrCompare('analysis'))
        actprop.add_rule(RuleSubvarAnd('subvar'))
        actprop.add_rule(RuleSubvarSubpiece('subvar'))
        actprop.add_rule(RuleSplitFlow('subvar'))
        actprop.add_rule(RulePtrFlow('analysis'))
        actprop.add_rule(RuleSubvarCompZero('subvar'))
        actprop.add_rule(RuleSubvarShift('subvar'))
        actprop.add_rule(RuleSubvarZext('subvar'))
        actprop.add_rule(RuleSubvarSext('subvar'))
        actprop.add_rule(RuleNegateNegate('analysis'))
        actprop.add_rule(RuleConditionalMove('conditionalexe'))
        actprop.add_rule(RuleOrPredicate('conditionalexe'))
        actprop.add_rule(RuleFuncPtrEncoding('analysis'))
        actprop.add_rule(RuleSubfloatConvert('floatprecision'))
        actprop.add_rule(RuleFloatCast('floatprecision'))
        actprop.add_rule(RuleIgnoreNan('floatprecision'))
        actprop.add_rule(RuleUnsigned2Float('analysis'))
        actprop.add_rule(RuleInt2FloatCollapse('analysis'))
        actprop.add_rule(RulePtraddUndo('typerecovery'))
        actprop.add_rule(RulePtrsubUndo('typerecovery'))
        actprop.add_rule(RuleSegment('segment'))
        actprop.add_rule(RulePiecePathology('protorecovery'))
        actprop.add_rule(RuleDoubleLoad('doubleload'))
        actprop.add_rule(RuleDoubleStore('doubleprecis'))
        actprop.add_rule(RuleDoubleIn('doubleprecis'))
        actprop.add_rule(RuleDoubleOut('doubleprecis'))

        actprop2 = ActionPool('actprop2', 'actprop2')
        actprop2.add_rule(RulePushPtr('typerecovery'))
        actprop2.add_rule(RuleStructOffset0('typerecovery'))
        actprop2.add_rule(RulePtrArith('typerecovery'))
        actprop2.add_rule(RuleIndirectConcat('analysis'))
        actprop2.add_rule(RuleLoadVarnode('stackvars'))
        actprop2.add_rule(RuleStoreVarnode('stackvars'))

        actcleanup = ActionPool('actcleanup', 'actcleanup')
        actcleanup.add_rule(RuleMultNegOne('cleanup'))
        actcleanup.add_rule(RuleAddUnsigned('cleanup'))
        actcleanup.add_rule(Rule2Comp2Sub('cleanup'))
        actcleanup.add_rule(RuleDumptyHumpLate('cleanup'))
        actcleanup.add_rule(RuleSubRight('cleanup'))
        actcleanup.add_rule(RuleFloatSignCleanup('cleanup'))
        actcleanup.add_rule(RuleExpandLoad('cleanup'))
        actcleanup.add_rule(RulePtrsubCharConstant('cleanup'))
        actcleanup.add_rule(RuleExtensionPush('cleanup'))
        actcleanup.add_rule(RulePieceStructure('cleanup'))
        actcleanup.add_rule(RuleAndStructure('cleanup'))
        actcleanup.add_rule(RuleSplitCopy('splitcopy'))
        actcleanup.add_rule(RuleSplitLoad('splitpointer'))
        actcleanup.add_rule(RuleSplitStore('splitpointer'))
        actcleanup.add_rule(RuleStringCopy('constsequence'))
        actcleanup.add_rule(RuleStringStore('constsequence'))
        actcleanup.add_rule(RuleBitFieldStore('bitfields'))
        actcleanup.add_rule(RuleBitFieldOut('bitfields'))
        actcleanup.add_rule(RuleBitFieldLoad('bitfields'))
        actcleanup.add_rule(RuleBitFieldIn('bitfields'))
        actcleanup.add_rule(RulePullAbsorb('bitfields'))
        actcleanup.add_rule(RuleInsertAbsorb('bitfields'))



        universal = ActionRestartGroup("universal", "universal")
        self.register_action("universal", universal)
        actfullloop = ActionGroup("fullloop", "fullloop")
        actmainloop = ActionGroup("mainloop", "mainloop")
        actstackstall = ActionGroup("stackstall", "stackstall")

        universal.add_action(ActionStart("base"))
        universal.add_action(ActionConstbase("base"))
        universal.add_action(ActionNormalizeSetup("normalanalysis"))
        universal.add_action(ActionDefaultParams("base"))
        universal.add_action(ActionExtraPopSetup("base"))
        universal.add_action(ActionPrototypeTypes("protorecovery"))
        universal.add_action(ActionFuncLink("protorecovery"))
        universal.add_action(ActionFuncLinkOutOnly("noproto"))
        actmainloop.add_action(ActionUnreachable("base"))
        actmainloop.add_action(ActionVarnodeProps("base"))
        actmainloop.add_action(ActionHeritage("base"))
        actmainloop.add_action(ActionParamDouble("protorecovery"))
        actmainloop.add_action(ActionSegmentize("base"))
        actmainloop.add_action(ActionInternalStorage("base"))
        actmainloop.add_action(ActionForceGoto("blockrecovery"))
        actmainloop.add_action(ActionDirectWrite("protorecovery_a", True))
        actmainloop.add_action(ActionDirectWrite("protorecovery_b", False))
        actmainloop.add_action(ActionActiveParam("protorecovery"))
        actmainloop.add_action(ActionReturnRecovery("protorecovery"))
        actmainloop.add_action(ActionRestrictLocal("localrecovery"))
        actmainloop.add_action(ActionDeadCode("deadcode"))
        actmainloop.add_action(ActionDynamicMapping("dynamic"))
        actmainloop.add_action(ActionSpacebase("base"))
        actmainloop.add_action(ActionNonzeroMask("analysis"))
        actmainloop.add_action(ActionInferTypes("typerecovery"))
        actmainloop.add_action(ActionRestructureVarnode("localrecovery"))
        actstackstall.add_action(actprop)
        actstackstall.add_action(ActionLaneDivide("base"))
        actstackstall.add_action(ActionMultiCse("analysis"))
        actstackstall.add_action(ActionShadowVar("analysis"))
        actstackstall.add_action(ActionDeindirect("deindirect"))
        actstackstall.add_action(ActionStackPtrFlow("stackptrflow"))
        actmainloop.add_action(actstackstall)
        actmainloop.add_action(ActionRedundBranch("deadcontrolflow"))
        actmainloop.add_action(ActionBlockStructure("blockrecovery"))
        actmainloop.add_action(ActionConstantPtr("typerecovery"))
        actmainloop.add_action(actprop2)
        actmainloop.add_action(ActionDeterminedBranch("unreachable"))
        actmainloop.add_action(ActionUnreachable("unreachable"))
        actmainloop.add_action(ActionNodeJoin("nodejoin"))
        actmainloop.add_action(ActionConditionalExe("conditionalexe"))
        actmainloop.add_action(ActionConditionalConst("analysis"))
        actfullloop.add_action(actmainloop)
        actfullloop.add_action(ActionLikelyTrash("protorecovery"))
        actfullloop.add_action(ActionDirectWrite("protorecovery_a", True))
        actfullloop.add_action(ActionDirectWrite("protorecovery_b", False))
        actfullloop.add_action(ActionDeadCode("deadcode"))
        actfullloop.add_action(ActionDoNothing("deadcontrolflow"))
        actfullloop.add_action(ActionSwitchNorm("switchnorm"))
        actfullloop.add_action(ActionReturnSplit("returnsplit"))
        actfullloop.add_action(ActionUnjustifiedParams("protorecovery"))
        actfullloop.add_action(ActionStartTypes("typerecovery"))
        actfullloop.add_action(ActionActiveReturn("protorecovery"))
        universal.add_action(actfullloop)
        universal.add_action(ActionMappedLocalSync("localrecovery"))
        universal.add_action(ActionStartCleanUp("cleanup"))
        universal.add_action(actcleanup)
        universal.add_action(ActionPreferComplement("blockrecovery", True))
        universal.add_action(ActionStructureTransform("blockrecovery", True))
        universal.add_action(ActionNormalizeBranches("normalizebranches"))
        universal.add_action(ActionAssignHigh("merge"))
        universal.add_action(ActionMergeRequired("merge"))
        universal.add_action(ActionMarkExplicit("merge"))
        universal.add_action(ActionMarkImplied("merge"))
        universal.add_action(ActionMergeMultiEntry("merge"))
        universal.add_action(ActionMergeCopy("merge"))
        universal.add_action(ActionDominantCopy("merge"))
        universal.add_action(ActionDynamicSymbols("dynamic"))
        universal.add_action(ActionMarkIndirectOnly("merge"))
        universal.add_action(ActionMergeAdjacent("merge"))
        universal.add_action(ActionMergeType("merge"))
        universal.add_action(ActionHideShadow("merge"))
        universal.add_action(ActionCopyMarker("merge"))
        universal.add_action(ActionLateDoNothing("blockrecovery"))
        universal.add_action(ActionBlockStructure("blockrecovery"))
        universal.add_action(ActionPreferComplement("blockrecovery", False))
        universal.add_action(ActionStructureTransform("blockrecovery", False))
        universal.add_action(ActionOutputPrototype("localrecovery"))
        universal.add_action(ActionInputPrototype("fixateproto"))
        universal.add_action(ActionMapGlobals("fixateglobals"))
        universal.add_action(ActionDynamicSymbols("dynamic"))
        universal.add_action(ActionNameVars("merge"))
        universal.add_action(ActionSetCasts("casts"))
        universal.add_action(ActionFinalStructure("blockrecovery"))
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
