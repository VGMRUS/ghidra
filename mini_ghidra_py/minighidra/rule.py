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
        changed = True
        while changed:
            changed = False
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
                        changed = True
                        break
        return self.count

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

class RulePiece2Zext(Rule):
    """Concatenation with 0 becomes an extension: V = PIECE(0, W) => V = ZEXT(W)."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "piece2zext")

    def get_op_list(self) -> tuple[str, ...]:
        return ("PIECE",)

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        const_vn = op.inputs[0] # Most significant bits in Ghidra PIECE
        if not data.is_constant(const_vn): return 0
        if data.get_constant_value(const_vn) != 0: return 0
        
        op = data.op_remove_input(op, 0)
        data.op_set_opcode(op, "INT_ZEXT")
        return 1

class RulePiece2Sext(Rule):
    """Concatenation with sign bits becomes an extension: PIECE(V s>> #0x1f, V) => SEXT(V)."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "piece2sext")

    def get_op_list(self) -> tuple[str, ...]:
        return ("PIECE",)

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        shiftout = op.inputs[0]
        shiftop = data.get_def(shiftout)
        if not shiftop or shiftop.opcode != "INT_SRIGHT": return 0
        if not data.is_constant(shiftop.inputs[1]): return 0
        
        n = data.get_constant_value(shiftop.inputs[1])
        x = shiftop.inputs[0]
        if x != op.inputs[1]: return 0
        
        size = data.nodes[x].size_bits
        if n != size - 1: return 0
        
        op = data.op_remove_input(op, 0)
        data.op_set_opcode(op, "INT_SEXT")
        return 1

class RuleTermOrder(Rule):
    """Order the inputs to commutative operations so constants come second."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "termorder")

    def get_op_list(self) -> tuple[str, ...]:
        return (
            "INT_ADD", "INT_MULT", "INT_AND", "INT_OR", "INT_XOR", 
            "INT_EQUAL", "INT_NOTEQUAL", "FLOAT_ADD", "FLOAT_MULT", 
            "FLOAT_EQUAL", "FLOAT_NOTEQUAL", "BOOL_AND", "BOOL_OR", "BOOL_XOR"
        )

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        if data.is_constant(op.inputs[0]) and not data.is_constant(op.inputs[1]):
            # Swap inputs
            in0, in1 = op.inputs[0], op.inputs[1]
            op = data.op_set_input(op, in1, 0)
            op = data.op_set_input(op, in0, 1)
            return 1
        return 0

class RuleBxor2NotEqual(Rule):
    """Convert BOOL_XOR to INT_NOTEQUAL."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "bxor2notequal")

    def get_op_list(self) -> tuple[str, ...]:
        return ("BOOL_XOR",)

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        data.op_set_opcode(op, "INT_NOTEQUAL")
        return 1

class RuleAndOrLump(Rule):
    """Collapse constants in logical expressions: (V & c) & d => V & (c & d)."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "andorlump")

    def get_op_list(self) -> tuple[str, ...]:
        return ("INT_AND", "INT_OR", "INT_XOR")

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        if not data.is_constant(op.inputs[1]): return 0
        
        vn1 = op.inputs[0]
        op2 = data.get_def(vn1)
        if not op2 or op2.opcode != op.opcode: return 0
        if not data.is_constant(op2.inputs[1]): return 0
        
        basevn = op2.inputs[0]
        val1 = data.get_constant_value(op.inputs[1])
        val2 = data.get_constant_value(op2.inputs[1])
        
        if op.opcode == "INT_AND":
            val = val1 & val2
        elif op.opcode == "INT_OR":
            val = val1 | val2
        else:
            val = val1 ^ val2
            
        size = data.nodes[op.inputs[1]].size_bits
        const_vn = data.new_constant(size, val)
        
        op = data.op_set_input(op, basevn, 0)
        op = data.op_set_input(op, const_vn, 1)
        return 1

class RuleOrCollapse(Rule):
    """Simplify INT_OR with full mask: V = W | 0xffff => V = 0xffff. 
       Also handles general V | C => C if V_nzmask | C == C."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "orcollapse")

    def get_op_list(self) -> tuple[str, ...]:
        return ("INT_OR",)

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        if not data.is_constant(op.inputs[1]): return 0
        
        mask = data.get_nzmask(op.inputs[0])
        val = data.get_constant_value(op.inputs[1])
        
        if (mask | val) != val: return 0
        
        # Result is just the constant value
        op = data.op_remove_input(op, 0)
        data.op_set_opcode(op, "COPY")
        return 1

class RuleAndMask(Rule):
    """Simplify INT_AND with 0 or full mask, taking nzmask into account."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "andmask")

    def get_op_list(self) -> tuple[str, ...]:
        return ("INT_AND",)

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        
        mask1 = data.get_nzmask(op.inputs[0])
        mask2 = data.get_nzmask(op.inputs[1])
        andmask = mask1 & mask2
        
        if andmask == 0:
            # Result is 0
            size = data.nodes[op.output].size_bits if op.output is not None else data.nodes[op.inputs[0]].size_bits
            const_vn = data.new_constant(size, 0)
            op = data.op_remove_input(op, 1)
            op = data.op_set_input(op, const_vn, 0)
            data.op_set_opcode(op, "COPY")
            return 1
        elif andmask == mask1:
            if not data.is_constant(op.inputs[1]): return 0
            # Result is input(0)
            op = data.op_remove_input(op, 1)
            data.op_set_opcode(op, "COPY")
            return 1
            
        return 0

class RuleLessOne(Rule):
    """Simplify x < 1 => x == 0, and x <= 0 => x == 0."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "lessone")

    def get_op_list(self) -> tuple[str, ...]:
        return ("INT_LESS", "INT_LESSEQUAL")

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        if not data.is_constant(op.inputs[1]): return 0
        
        val = data.get_constant_value(op.inputs[1])
        if op.opcode == "INT_LESS" and val != 1: return 0
        if op.opcode == "INT_LESSEQUAL" and val != 0: return 0
        
        data.op_set_opcode(op, "INT_EQUAL")
        if val != 0:
            sz = data.nodes[op.inputs[1]].size_bits
            const_vn = data.new_constant(sz, 0)
            data.op_set_input(op, const_vn, 1)
        return 1

class RuleLogicIdentity(Rule):
    """Simplify V & (~V) => 0, V | (~V) => -1, V ^ (~V) => -1."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "logicidentity")

    def get_op_list(self) -> tuple[str, ...]:
        return ("INT_AND", "INT_OR", "INT_XOR")

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        in0 = op.inputs[0]
        in1 = op.inputs[1]
        
        def_op0 = data.get_def(in0)
        def_op1 = data.get_def(in1)
        
        match = False
        if def_op0 and def_op0.opcode == "INT_NEGATE" and def_op0.inputs[0] == in1:
            match = True
        elif def_op1 and def_op1.opcode == "INT_NEGATE" and def_op1.inputs[0] == in0:
            match = True
            
        if not match:
            return 0
            
        size = data.nodes[op.output].size_bits if op.output is not None else data.nodes[in0].size_bits
        mask = (1 << size) - 1
        
        val = 0 if op.opcode == "INT_AND" else mask
        const_vn = data.new_constant(size, val)
        
        op = data.op_remove_input(op, 1)
        op = data.op_set_input(op, const_vn, 0)
        data.op_set_opcode(op, "COPY")
        return 1

class RuleDoubleLoad(Rule):
    """Collapse contiguous loads: V = PIECE(LOAD(ptr+4), LOAD(ptr)) => V = LOAD(ptr)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "doubleload")

    def get_op_list(self) -> tuple[str, ...]:
        return ("PIECE",)

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        piece0 = op.inputs[0] # MSB
        piece1 = op.inputs[1] # LSB
        
        load0 = data.get_def(piece0)
        load1 = data.get_def(piece1)
        
        if not load1 or load1.opcode != "LOAD": return 0
        
        offset = 0
        if load0 and load0.opcode == "SUBPIECE":
            if not data.is_constant(load0.inputs[1]) or data.get_constant_value(load0.inputs[1]) != 0:
                return 0
            vn0 = load0.inputs[0]
            offset = data.nodes[vn0].size_bits - data.nodes[piece0].size_bits
            load0 = data.get_def(vn0)
            
        if not load0 or load0.opcode != "LOAD": return 0
        
        ptr1 = load1.inputs[0]
        ptr0 = load0.inputs[0]
        
        # Check if ptr0 == ptr1 + size(piece1)
        add_op = data.get_def(ptr0)
        if not add_op or add_op.opcode != "INT_ADD": return 0
        
        if add_op.inputs[0] == ptr1 and data.is_constant(add_op.inputs[1]):
            const_add = data.get_constant_value(add_op.inputs[1])
        elif add_op.inputs[1] == ptr1 and data.is_constant(add_op.inputs[0]):
            const_add = data.get_constant_value(add_op.inputs[0])
        else:
            return 0
            
        sz1_bytes = data.nodes[piece1].size_bits // 8
        if const_add != sz1_bytes: return 0
        
        # Collapse! Replace PIECE with LOAD from ptr1
        op = data.op_remove_input(op, 1)
        op = data.op_set_input(op, ptr1, 0)
        data.op_set_opcode(op, "LOAD")
        return 1

class RulePullsubMulti(Rule):
    """Pull SUBPIECE up through a MULTIEQUAL (Phi):
       V = SUBPIECE(MULTIEQUAL(A, B), c) => V = MULTIEQUAL(SUBPIECE(A, c), SUBPIECE(B, c))
    """
    def __init__(self, group: str) -> None:
        super().__init__(group, "pullsubmulti")

    def get_op_list(self) -> tuple[str, ...]:
        return ("SUBPIECE",)

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        vn = op.inputs[0]
        mult = data.get_def(vn)
        if not mult or mult.opcode != "MULTIEQUAL": return 0
        
        # Check if there are other uses of this MULTIEQUAL. 
        # For simplicity, if we pull it up, we duplicate the SUBPIECE. 
        # This is safe and eliminates the wide phi node if it becomes dead.
        out_size = data.nodes[op.output].size_bits if op.output is not None else 8
        c_vn = op.inputs[1]
        
        new_inputs = []
        for i, branch_vn in enumerate(mult.inputs):
            # Create a new SUBPIECE for each branch
            new_vn = data._new_temporary_node(f"pullsub_{i}", out_size)
            new_op = PcodeOp(
                "SUBPIECE",
                (branch_vn, c_vn),
                new_vn,
                address=op.address,
                mnemonic="SUBPIECE"
            )
            data._set_definition(new_op)
            new_inputs.append(new_vn)
            
        # Replace the original SUBPIECE with a MULTIEQUAL of the new truncated inputs
        op = data.op_remove_input(op, 1) # remove c_vn
        # Now op has 1 input. We need to replace all inputs with new_inputs
        new_op_dict = op.__dict__.copy()
        new_op_dict["inputs"] = tuple(new_inputs)
        new_op_dict["opcode"] = "MULTIEQUAL"
        new_op = object.__new__(PcodeOp)
        new_op.__dict__.update(new_op_dict)
        
        try:
            idx = data.ops.index(op)
            data.ops[idx] = new_op
        except ValueError:
            pass
        
        data.clear_nzmask_cache()
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

class RuleZextEliminate(Rule):
    """Simplify ZEXT(V) == C => V == C if C fits in V."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "zexteliminate")

    def get_op_list(self) -> tuple[str, ...]:
        return (
            "INT_EQUAL", "INT_NOTEQUAL", "INT_LESS", "INT_LESSEQUAL", 
            "INT_SLESS", "INT_SLESSEQUAL"
        )

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        in0 = op.inputs[0]
        in1 = op.inputs[1]
        
        zext_slot = -1
        other_slot = -1
        
        def_op0 = data.get_def(in0)
        def_op1 = data.get_def(in1)
        
        if def_op1 and def_op1.opcode == "INT_ZEXT":
            zext_slot = 1
            other_slot = 0
            zext_op = def_op1
        elif def_op0 and def_op0.opcode == "INT_ZEXT":
            zext_slot = 0
            other_slot = 1
            zext_op = def_op0
        else:
            return 0
            
        other_vn = op.inputs[other_slot]
        if not data.is_constant(other_vn): return 0
        
        small_size = data.nodes[zext_op.inputs[0]].size_bits
        val = data.get_constant_value(other_vn)
        
        # Check if constant fits in small_size
        if (val >> small_size) == 0:
            new_vn = data.new_constant(small_size, val)
            op = data.op_set_input(op, zext_op.inputs[0], zext_slot)
            op = data.op_set_input(op, new_vn, other_slot)
            return 1
            
        return 0

class RuleSlessToLess(Rule):
    """Convert INT_SLESS to INT_LESS when comparing positive values."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "slesstoless")

    def get_op_list(self) -> tuple[str, ...]:
        return ("INT_SLESS", "INT_SLESSEQUAL")

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        sz0 = data.nodes[op.inputs[0]].size_bits
        mask0 = data.get_nzmask(op.inputs[0])
        mask1 = data.get_nzmask(op.inputs[1])
        
        sign_bit = 1 << (sz0 - 1)
        if (mask0 & sign_bit) == 0 and (mask1 & sign_bit) == 0:
            new_opcode = "INT_LESS" if op.opcode == "INT_SLESS" else "INT_LESSEQUAL"
            data.op_set_opcode(op, new_opcode)
            return 1
        return 0

