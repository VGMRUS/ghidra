from __future__ import annotations
from typing import Optional
from .paramid import PcodeLikeGraph, PcodeOp
from .action import Action

class Rule:
    """Base class for simplification rules applied to PcodeOps."""

    def __init__(self, group: str, name: str) -> None:
        self.group = group
        self.name = name
        self.count_tests = 0
        self.count_apply = 0
        self.disabled = False

    def get_op_list(self) -> tuple[str, ...]:
        """Return the list of P-code operations this rule applies to."""
        raise NotImplementedError

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        """Attempt to apply this rule to a specific PcodeOp.
        
        Returns > 0 if a change was made, 0 otherwise.
        """
        raise NotImplementedError

class ActionPool(Action):
    """A collection of Rules that are applied iteratively to a function's p-code ops."""

    def __init__(self, group: str, name: str) -> None:
        super().__init__(group, name)
        self.all_rules: list[Rule] = []
        self.rules_by_op: dict[str, list[Rule]] = {}

    def add_rule(self, rule: Rule) -> None:
        self.all_rules.append(rule)
        for opcode in rule.get_op_list():
            if opcode not in self.rules_by_op:
                self.rules_by_op[opcode] = []
            self.rules_by_op[opcode].append(rule)

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        ops = list(data.ops)
        for op in ops:
            if not op.opcode in self.rules_by_op:
                continue
            
            rules = self.rules_by_op[op.opcode]
            for rule in rules:
                if rule.disabled:
                    continue
                rule.count_tests += 1
                res = rule.apply_op(op, data)
                if res > 0:
                    rule.count_apply += 1
                    self.count += res
                    break
        return 0

class RuleTrivialArith(Rule):
    """Simplifies arithmetic with identical inputs (e.g. V ^ V => 0)."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "trivialarith")

    def get_op_list(self) -> tuple[str, ...]:
        return (
            "INT_NOTEQUAL", "INT_SLESS", "INT_LESS", "BOOL_XOR", "FLOAT_NOTEQUAL", "FLOAT_LESS",
            "INT_EQUAL", "INT_SLESSEQUAL", "INT_LESSEQUAL", "FLOAT_EQUAL", "FLOAT_LESSEQUAL",
            "INT_XOR", "INT_SUB", "BOOL_AND", "BOOL_OR", "INT_AND", "INT_OR"
        )

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2:
            return 0
        in0 = op.inputs[0]
        in1 = op.inputs[1]
        
        if in0 != in1:
            return 0
            
        opcode = op.opcode
        out_vn = op.output
        if out_vn is None:
            return 0
        size = data.nodes[out_vn].size_bits
        
        if opcode in ("INT_NOTEQUAL", "INT_SLESS", "INT_LESS", "BOOL_XOR", "FLOAT_NOTEQUAL", "FLOAT_LESS"):
            const_vn = data.new_constant(1, 0)
        elif opcode in ("INT_EQUAL", "INT_SLESSEQUAL", "INT_LESSEQUAL", "FLOAT_EQUAL", "FLOAT_LESSEQUAL"):
            const_vn = data.new_constant(1, 1)
        elif opcode in ("INT_XOR", "INT_SUB"):
            const_vn = data.new_constant(size, 0)
        elif opcode in ("BOOL_AND", "BOOL_OR", "INT_AND", "INT_OR"):
            # Identity: replace op with COPY
            data.op_set_opcode(op, "COPY")
            # op_set_opcode creates a new op in data.ops, but we also want to remove 2nd input
            # Actually just op_set_input handles it, or we just leave the 2nd input since COPY ignores it
            # But let's be clean
            return 1
        else:
            return 0
            
        data.op_set_opcode(op, "COPY")
        data.op_set_input(op, const_vn, 0)
        return 1

class RuleXorCollapse(Rule):
    """Collapses consecutive XORs with constants: (V ^ c) ^ d => V ^ (c ^ d)."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "xorcollapse")

    def get_op_list(self) -> tuple[str, ...]:
        return ("INT_XOR",)

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        if not data.is_constant(op.inputs[1]): return 0
        
        xorop = data.get_def(op.inputs[0])
        if not xorop or xorop.opcode != "INT_XOR": return 0
        
        coeff1 = data.get_constant_value(op.inputs[1])
        xorvn = xorop.inputs[1]
        
        if not data.is_constant(xorvn):
            if coeff1 != 0: return 0
            # V ^ 0 => V
            data.op_set_input(op, xorvn, 1)
            data.op_set_input(op, xorop.inputs[0], 0)
            return 1
            
        coeff2 = data.get_constant_value(xorvn)
        if coeff2 == 0: return 0
        
        size = data.nodes[op.inputs[1]].size_bits
        const_vn = data.new_constant(size, coeff1 ^ coeff2)
        
        data.op_set_input(op, const_vn, 1)
        data.op_set_input(op, xorop.inputs[0], 0)
        return 1

class RuleAddMultCollapse(Rule):
    """Collapse constants in additive or multiplicative expressions."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "addmultcollapse")

    def get_op_list(self) -> tuple[str, ...]:
        return ("INT_ADD", "INT_MULT")

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        if not data.is_constant(op.inputs[1]): return 0
        
        subop = data.get_def(op.inputs[0])
        if not subop or subop.opcode != op.opcode: return 0
        if not data.is_constant(subop.inputs[1]): return 0
        
        val1 = data.get_constant_value(op.inputs[1])
        val2 = data.get_constant_value(subop.inputs[1])
        
        if op.opcode == "INT_ADD":
            new_val = val1 + val2
        else:
            new_val = val1 * val2
            
        size = data.nodes[op.inputs[1]].size_bits
        mask = (1 << size) - 1
        new_val &= mask
        
        const_vn = data.new_constant(size, new_val)
        data.op_set_input(op, const_vn, 1)
        data.op_set_input(op, subop.inputs[0], 0)
        return 1

class RuleSub2Add(Rule):
    """Convert SUB to ADD: V - c => V + (-c)."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "sub2add")

    def get_op_list(self) -> tuple[str, ...]:
        return ("INT_SUB",)

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        if not data.is_constant(op.inputs[1]): return 0
        
        val = data.get_constant_value(op.inputs[1])
        size = data.nodes[op.inputs[1]].size_bits
        mask = (1 << size) - 1
        
        new_val = (-val) & mask
        const_vn = data.new_constant(size, new_val)
        
        op = data.op_set_opcode(op, "INT_ADD")
        data.op_set_input(op, const_vn, 1)
        return 1

