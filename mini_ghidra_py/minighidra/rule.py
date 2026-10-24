from __future__ import annotations
from typing import Optional
from .paramid import PcodeLikeGraph, PcodeOp, PcodeVarnode
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


class RuleOrConsume(Rule):
    """Simplify OR with unconsumed input: V = A | B => V = B if nzm(A) & consume(V) == 0"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "orconsume")

    def get_op_list(self) -> list[str]:
        return ["INT_OR", "INT_XOR"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        out_node = data.nodes[op.output] if op.output is not None else None
        if not out_node: return 0
        
        consume = data.get_consume(op.output)
        
        mask0 = data.get_nzmask(op.inputs[0])
        if (consume & mask0) == 0:
            data.op_remove_input(op, 0)
            data.op_set_opcode(op, "COPY")
            return 1
            
        mask1 = data.get_nzmask(op.inputs[1])
        if (consume & mask1) == 0:
            data.op_remove_input(op, 1)
            data.op_set_opcode(op, "COPY")
            return 1
            
        return 0

class RuleNegateIdentity(Rule):
    """Apply INT_NEGATE identities: V & ~V => #0, V | ~V -> #-1"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "negateidentity")

    def get_op_list(self) -> list[str]:
        return ["INT_NEGATE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 1: return 0
        in_vn = op.inputs[0]
        out_vn = op.output
        if out_vn is None: return 0
        
        count = 0
        for logic_op in data.ops:
            if logic_op.opcode not in ("INT_AND", "INT_OR", "INT_XOR"):
                continue
            
            slot = -1
            if len(logic_op.inputs) == 2:
                if logic_op.inputs[0] == out_vn:
                    slot = 0
                elif logic_op.inputs[1] == out_vn:
                    slot = 1
                    
            if slot == -1: continue
            
            if logic_op.inputs[1 - slot] != in_vn:
                continue
                
            value = 0
            if logic_op.opcode != "INT_AND":
                value = (1 << data.nodes[in_vn].size_bits) - 1
                
            cnode = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnode, hex(value), data.nodes[in_vn].size_bits, "const"))
            
            data.op_set_input(logic_op, cnode, 0)
            data.op_remove_input(logic_op, 1)
            data.op_set_opcode(logic_op, "COPY")
            count += 1
            
        return count


class RuleShiftBitops(Rule):
    """Shifting away all non-zero bits of one-side of a logical/arithmetic op."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "shiftbitops")

    def get_op_list(self) -> list[str]:
        return ["INT_LEFT", "INT_RIGHT", "SUBPIECE", "INT_MULT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        constvn = data.nodes[op.inputs[1]]
        if constvn.kind != "const": return 0
        
        vn = op.inputs[0]
        bitop = None
        for cand in data.ops:
            if cand.output == vn:
                bitop = cand
                break
        if not bitop: return 0
        
        sa = 0
        leftshift = False
        
        if op.opcode == "INT_LEFT":
            sa = int(constvn.name, 16)
            leftshift = True
        elif op.opcode == "INT_RIGHT":
            sa = int(constvn.name, 16)
            leftshift = False
        elif op.opcode == "SUBPIECE":
            sa = int(constvn.name, 16) * 8
            leftshift = False
        elif op.opcode == "INT_MULT":
            val = int(constvn.name, 16)
            # Find least significant bit
            if val == 0: return 0
            sa = (val & -val).bit_length() - 1
            if sa == -1: return 0
            leftshift = True
            
        if bitop.opcode not in ("INT_AND", "INT_OR", "INT_XOR", "INT_MULT", "INT_ADD"):
            return 0
            
        if bitop.opcode in ("INT_MULT", "INT_ADD") and not leftshift:
            return 0
            
        i = 0
        while i < len(bitop.inputs):
            nzm = data.get_nzmask(bitop.inputs[i])
            out_sz = data.nodes[op.output].size_bits if op.output is not None else data.nodes[vn].size_bits
            mask = (1 << out_sz) - 1
            
            if leftshift:
                nzm = (nzm << sa)
            else:
                nzm = (nzm >> sa)
                
            if (nzm & mask) == 0:
                break
            i += 1
            
        if i == len(bitop.inputs): return 0
        
        if bitop.opcode in ("INT_MULT", "INT_AND"):
            out_sz = data.nodes[op.output].size_bits if op.output is not None else data.nodes[vn].size_bits
            cnode = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnode, hex(0), out_sz, "const"))
            data.op_set_input(op, cnode, 0)
        elif bitop.opcode in ("INT_ADD", "INT_XOR", "INT_OR"):
            other_vn = bitop.inputs[1 - i]
            data.op_set_input(op, other_vn, 0)
            
        return 1

class RuleRightShiftAnd(Rule):
    """Simplify INT_RIGHT and INT_SRIGHT ops where an INT_AND mask becomes unnecessary."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "rightshiftand")

    def get_op_list(self) -> list[str]:
        return ["INT_RIGHT", "INT_SRIGHT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        const_node = data.nodes[op.inputs[1]]
        if const_node.kind != "const": return 0
        
        in_vn = op.inputs[0]
        and_op = None
        for cand in data.ops:
            if cand.output == in_vn:
                and_op = cand
                break
        if not and_op or and_op.opcode != "INT_AND": return 0
        
        if len(and_op.inputs) != 2: return 0
        mask_node = data.nodes[and_op.inputs[1]]
        if mask_node.kind != "const": return 0
        
        sa = int(const_node.name, 16)
        mask = int(mask_node.name, 16) >> sa
        
        root_vn = and_op.inputs[0]
        root_sz = data.nodes[root_vn].size_bits
        full = ((1 << root_sz) - 1) >> sa
        
        if full != mask: return 0
        
        data.op_set_input(op, root_vn, 0)
        return 1


class RuleEquality(Rule):
    """Collapse INT_EQUAL and INT_NOTEQUAL: f(V,W) == f(V,W) => true"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "equality")

    def get_op_list(self) -> list[str]:
        return ["INT_EQUAL", "INT_NOTEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        if op.inputs[0] != op.inputs[1]: return 0
        
        data.op_set_opcode(op, "COPY")
        data.op_remove_input(op, 1)
        
        cnode = len(data.nodes)
        val = 1 if op.opcode == "INT_EQUAL" else 0
        data.nodes.append(PcodeVarnode(cnode, hex(val), 1, "const"))
        
        data.op_set_input(op, cnode, 0)
        return 1

class RuleIntLessEqualRewrite(Rule):
    """Convert LESSEQUAL to LESS: V <= c => V < (c+1)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "intlessequal")

    def get_op_list(self) -> list[str]:
        return ["INT_LESSEQUAL", "INT_SLESSEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        c_node = data.nodes[op.inputs[1]]
        if c_node.kind != "const": return 0
        
        val = int(c_node.name, 16)
        mask = (1 << c_node.size_bits) - 1
        
        is_signed = (op.opcode == "INT_SLESSEQUAL")
        
        if is_signed:
            max_val = mask >> 1
            if val == max_val: return 0  # Overflow
        else:
            if val == mask: return 0  # Overflow
            
        cnode_new = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnode_new, hex(val + 1), c_node.size_bits, "const"))
        
        data.op_set_input(op, cnode_new, 1)
        if is_signed:
            data.op_set_opcode(op, "INT_SLESS")
        else:
            data.op_set_opcode(op, "INT_LESS")
            
        return 1


class RuleHighOrderAnd(Rule):
    """Simplify INT_AND when applied to aligned INT_ADD: (V + c) & 0xfff0 => V + (c & 0xfff0)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "highorderand")

    def get_op_list(self) -> list[str]:
        return ["INT_AND"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        in0 = op.inputs[0]
        addop = None
        for cand in data.ops:
            if cand.output == in0:
                addop = cand
                break
                
        if not addop or addop.opcode != "INT_ADD": return 0
        if len(addop.inputs) != 2: return 0
        
        val = int(cnode.name, 16)
        sz = cnode.size_bits
        mask = (1 << sz) - 1
        
        # Check that cvn1 is of the form 11110000
        if (((val - 1) | val) & mask) != mask:
            return 0
            
        cnode2 = data.nodes[addop.inputs[1]]
        if cnode2.kind == "const":
            xalign = addop.inputs[0]
            mask1 = data.get_nzmask(xalign)
            
            if (mask1 & val) != mask1: return 0
            
            data.op_set_opcode(op, "INT_ADD")
            data.op_set_input(op, xalign, 0)
            
            new_val = val & int(cnode2.name, 16)
            cnode_new = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnode_new, hex(new_val), sz, "const"))
            
            data.op_set_input(op, cnode_new, 1)
            return 1
        else:
            # Complex check for two adds
            count_descends = 0
            for test_op in data.ops:
                if in0 in test_op.inputs:
                    count_descends += 1
            if count_descends > 1: return 0
            
            for i in range(2):
                zerovn = addop.inputs[i]
                mask2 = data.get_nzmask(zerovn)
                if (mask2 & val) != mask2: continue
                
                nonzerovn = addop.inputs[1-i]
                addop2 = None
                for cand in data.ops:
                    if cand.output == nonzerovn:
                        addop2 = cand
                        break
                if not addop2 or addop2.opcode != "INT_ADD": continue
                
                count_descends_nz = 0
                for test_op in data.ops:
                    if nonzerovn in test_op.inputs:
                        count_descends_nz += 1
                if count_descends_nz > 1: continue
                
                if len(addop2.inputs) != 2: continue
                cnode3 = data.nodes[addop2.inputs[1]]
                if cnode3.kind != "const": continue
                
                xalign2 = addop2.inputs[0]
                mask3 = data.get_nzmask(xalign2)
                if (mask3 & val) != mask3: continue
                
                new_val = val & int(cnode3.name, 16)
                cnode_new = len(data.nodes)
                data.nodes.append(PcodeVarnode(cnode_new, hex(new_val), sz, "const"))
                
                data.op_set_input(addop2, cnode_new, 1)
                
                data.op_remove_input(op, 1)
                data.op_set_opcode(op, "COPY")
                return 1
                
        return 0


class RuleZextCommute(Rule):
    """Commute INT_ZEXT with INT_RIGHT: zext(V) >> W => zext(V >> W)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "zextcommute")

    def get_op_list(self) -> list[str]:
        return ["INT_RIGHT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        zextvn = op.inputs[0]
        zextop = None
        for cand in data.ops:
            if cand.output == zextvn:
                zextop = cand
                break
                
        if not zextop or zextop.opcode != "INT_ZEXT": return 0
        if len(zextop.inputs) != 1: return 0
        
        zextin = zextop.inputs[0]
        savn = op.inputs[1]
        
        new_op = PcodeOp(op.addr, "INT_RIGHT")
        new_out = len(data.nodes)
        data.nodes.append(PcodeVarnode(new_out, None, data.nodes[zextin].size_bits, "unique"))
        
        new_op.output = new_out
        new_op.inputs = [zextin, savn]
        data.ops.append(new_op)
        
        data.op_remove_input(op, 1)
        data.op_set_input(op, new_out, 0)
        data.op_set_opcode(op, "INT_ZEXT")
        return 1

class RuleZextShiftZext(Rule):
    """Simplify multiple INT_ZEXT operations: zext( zext(V) << c ) => zext(V) << c"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "zextshiftzext")

    def get_op_list(self) -> list[str]:
        return ["INT_ZEXT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 1: return 0
        invn = op.inputs[0]
        
        shiftop = None
        for cand in data.ops:
            if cand.output == invn:
                shiftop = cand
                break
                
        if not shiftop: return 0
        
        if shiftop.opcode == "INT_ZEXT":
            # ZEXT(ZEXT(a)) -> ZEXT(a)
            vn = shiftop.inputs[0]
            
            # Count descends
            count = 0
            for cand in data.ops:
                if invn in cand.inputs:
                    count += 1
            if count > 1: return 0
            
            data.op_set_input(op, vn, 0)
            return 1
            
        if shiftop.opcode != "INT_LEFT": return 0
        if len(shiftop.inputs) != 2: return 0
        
        cnode = data.nodes[shiftop.inputs[1]]
        if cnode.kind != "const": return 0
        
        zext2op = None
        for cand in data.ops:
            if cand.output == shiftop.inputs[0]:
                zext2op = cand
                break
                
        if not zext2op or zext2op.opcode != "INT_ZEXT": return 0
        rootvn = zext2op.inputs[0]
        
        sa = int(cnode.name, 16)
        sz_diff = data.nodes[zext2op.output].size_bits - data.nodes[rootvn].size_bits
        if sa > sz_diff: return 0
        
        new_op = PcodeOp(op.addr, "INT_ZEXT")
        new_out = len(data.nodes)
        out_sz = data.nodes[op.output].size_bits if op.output is not None else data.nodes[invn].size_bits
        data.nodes.append(PcodeVarnode(new_out, None, out_sz, "unique"))
        
        new_op.output = new_out
        new_op.inputs = [rootvn]
        data.ops.append(new_op)
        
        data.op_set_opcode(op, "INT_LEFT")
        data.op_set_input(op, new_out, 0)
        
        cnode2 = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnode2, hex(sa), cnode.size_bits, "const"))
        data.op_set_input(op, cnode2, 1)
        return 1

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


class RuleEquality(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "equality")

    def get_op_list(self) -> list[str]:
        return ["INT_EQUAL", "INT_NOTEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        # V == V -> 1, V != V -> 0
        if len(op.inputs) != 2:
            return 0
        if op.inputs[0] == op.inputs[1]:
            val = 1 if op.opcode == "INT_EQUAL" else 0
            # We don't have a newConstant method that injects constants gracefully right here,
            # but we can resolve this by adding a constant node
            const_node = len(data.nodes)
            data.nodes.append(PcodeVarnode(len(data.nodes), hex(val), 1, "const"))
            
            data.op_set_opcode(op, "COPY")
            data.op_remove_input(op, 1)
            data.op_set_input(op, const_node, 0)
            return 1
        return 0

class RuleTrivialBool(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "trivialbool")

    def get_op_list(self) -> list[str]:
        return ["BOOL_XOR", "BOOL_AND", "BOOL_OR"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2:
            return 0
            
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const":
            return 0
            
        val = int(cnode.name, 16)
        opc = ""
        vn_idx = -1
        
        if op.opcode == "BOOL_XOR":
            vn_idx = op.inputs[0]
            opc = "BOOL_NEGATE" if val == 1 else "COPY"
        elif op.opcode == "BOOL_AND":
            opc = "COPY"
            if val == 1:
                vn_idx = op.inputs[0]
            else:
                # copy false
                const_node = len(data.nodes)
                data.nodes.append(PcodeVarnode(len(data.nodes), hex(0), 1, "const"))
                vn_idx = const_node
        elif op.opcode == "BOOL_OR":
            opc = "COPY"
            if val == 1:
                const_node = len(data.nodes)
                data.nodes.append(PcodeVarnode(len(data.nodes), hex(1), 1, "const"))
                vn_idx = const_node
            else:
                vn_idx = op.inputs[0]
        else:
            return 0
            
        data.op_remove_input(op, 1)
        data.op_set_opcode(op, opc)
        data.op_set_input(op, vn_idx, 0)
        return 1


class RuleLessEqual(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "lessequal")

    def get_op_list(self) -> list[str]:
        return ["INT_LESSEQUAL", "INT_SLESSEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2:
            return 0
        node = data.nodes[op.inputs[1]]
        if node.kind != "const":
            return 0
            
        val = int(node.name, 16)
        # Check max size (we can't easily increment 0xFFFFFFFF)
        size = node.size_bits
        mask = (1 << size) - 1
        
        # Don't translate if it wraps around
        if op.opcode == "INT_LESSEQUAL" and val == mask:
            return 0
        if op.opcode == "INT_SLESSEQUAL" and val == (1 << (size - 1)) - 1:
            return 0
            
        new_val = val + 1
        new_opc = "INT_LESS" if op.opcode == "INT_LESSEQUAL" else "INT_SLESS"
        
        const_node = len(data.nodes)
        data.nodes.append(PcodeVarnode(const_node, hex(new_val), size, "const"))
        
        data.op_set_opcode(op, new_opc)
        data.op_remove_input(op, 1)
        data.op_set_input(op, const_node, 1)
        return 1


class RuleNotDistribute(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "notdistribute")

    def get_op_list(self) -> list[str]:
        return ["BOOL_NEGATE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 1:
            return 0
        
        in_node = data.nodes[op.inputs[0]]
        # Need to find the operation defining in_node
        def_op = None
        for cand_op in data.ops:
            if cand_op.output == op.inputs[0]:
                def_op = cand_op
                break
                
        if def_op is None:
            return 0
            
        # Is the intermediate variable consumed by anyone else?
        # A simple check: if more than 1 use, we can't distribute without duplicating the operation.
        uses = sum(1 for o in data.ops if op.inputs[0] in o.inputs)
        if uses > 1:
            return 0
            
        if def_op.opcode == "INT_EQUAL":
            data.op_set_opcode(def_op, "INT_NOTEQUAL")
        elif def_op.opcode == "INT_NOTEQUAL":
            data.op_set_opcode(def_op, "INT_EQUAL")
        elif def_op.opcode == "BOOL_NEGATE":
            data.op_set_opcode(def_op, "COPY")
        else:
            return 0
            
        # Bypass the BOOL_NEGATE op
        data.op_set_opcode(op, "COPY")
        return 1

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


class RuleZextSless(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "zextsless")

    def get_op_list(self) -> list[str]:
        return ["INT_SLESS", "INT_SLESSEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2:
            return 0
            
        zext_idx = -1
        const_idx = -1
        
        for i in range(2):
            inode = data.nodes[op.inputs[i]]
            if inode.kind == "const":
                const_idx = i
                
        if const_idx == -1:
            return 0
            
        zext_idx = 1 - const_idx
        
        def_op = None
        for cand_op in data.ops:
            if cand_op.output == op.inputs[zext_idx]:
                def_op = cand_op
                break
                
        if def_op is None or def_op.opcode != "INT_ZEXT":
            return 0
            
        in_node = data.nodes[def_op.inputs[0]]
        small_size = in_node.size_bits
        cnode = data.nodes[op.inputs[const_idx]]
        val = int(cnode.name, 16)
        
        # Is zero extension unnecessary, sign bit must also be 0
        if (val >> (small_size - 1)) != 0:
            return 0
            
        new_cnode = len(data.nodes)
        data.nodes.append(PcodeVarnode(new_cnode, hex(val), small_size, "const"))
        
        new_opcode = "INT_LESS" if op.opcode == "INT_SLESS" else "INT_LESSEQUAL"
        data.op_set_opcode(op, new_opcode)
        data.op_set_input(op, def_op.inputs[0], zext_idx)
        data.op_set_input(op, new_cnode, const_idx)
        return 1

class RuleCondNegate(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "condnegate")

    def get_op_list(self) -> list[str]:
        return ["CBRANCH"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2:
            return 0
            
        # Is it a boolean flip?
        # In minighidra, we don't naturally encode boolean flip in the instruction 
        # structure directly on CBRANCH in the same way. The Sleigh translator handles flip during translation.
        # But if we did have it, we could negate the condition.
        return 0
        
class RuleBoolNegate(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "boolnegate")

    def get_op_list(self) -> list[str]:
        return ["BOOL_NEGATE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 1: return 0
        
        def_op = None
        for cand_op in data.ops:
            if cand_op.output == op.inputs[0]:
                def_op = cand_op
                break
                
        if def_op is None: return 0
        
        # Uses > 1 prevents this
        uses = sum(1 for o in data.ops if op.inputs[0] in o.inputs)
        if uses > 1: return 0
        
        new_opc = None
        if def_op.opcode == "BOOL_NEGATE":
            new_opc = "COPY"
        elif def_op.opcode == "INT_EQUAL":
            new_opc = "INT_NOTEQUAL"
        elif def_op.opcode == "INT_NOTEQUAL":
            new_opc = "INT_EQUAL"
        elif def_op.opcode == "INT_LESS":
            new_opc = "INT_LESSEQUAL"
            # Swap operands
        elif def_op.opcode == "INT_LESSEQUAL":
            new_opc = "INT_LESS"
            # Swap operands
        elif def_op.opcode == "INT_SLESS":
            new_opc = "INT_SLESSEQUAL"
            # Swap operands
        elif def_op.opcode == "INT_SLESSEQUAL":
            new_opc = "INT_SLESS"
            # Swap operands
            
        if new_opc:
            if def_op.opcode in ["INT_LESS", "INT_LESSEQUAL", "INT_SLESS", "INT_SLESSEQUAL"]:
                # Swap inputs
                tmp = def_op.inputs[0]
                data.op_set_input(def_op, def_op.inputs[1], 0)
                data.op_set_input(def_op, tmp, 1)
            data.op_set_opcode(def_op, new_opc)
            data.op_set_opcode(op, "COPY")
            return 1
            
        return 0

class RuleDoubleShift(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "doubleshift")

    def get_op_list(self) -> list[str]:
        return ["INT_LEFT", "INT_RIGHT", "INT_MULT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        node2 = data.nodes[op.inputs[1]]
        if node2.kind != "const": return 0
        
        # Get defining operation
        def_op = None
        for cand_op in data.ops:
            if cand_op.output == op.inputs[0]:
                def_op = cand_op
                break
                
        if def_op is None: return 0
        
        opc2 = def_op.opcode
        if opc2 not in ["INT_LEFT", "INT_RIGHT", "INT_MULT"]: return 0
        
        if len(def_op.inputs) != 2: return 0
        node1 = data.nodes[def_op.inputs[1]]
        if node1.kind != "const": return 0
        
        opc1 = op.opcode
        
        def is_power_of_two(n):
            return (n != 0) and (n & (n-1) == 0)
            
        def leastsigbit_set(n):
            if n == 0: return -1
            return (n & -n).bit_length() - 1

        val1 = int(node2.name, 16)
        if opc1 == "INT_MULT":
            if not is_power_of_two(val1): return 0
            sa1 = leastsigbit_set(val1)
            opc1 = "INT_LEFT"
        else:
            sa1 = val1
            
        val2 = int(node1.name, 16)
        if opc2 == "INT_MULT":
            if not is_power_of_two(val2): return 0
            sa2 = leastsigbit_set(val2)
            opc2 = "INT_LEFT"
        else:
            sa2 = val2
            
        if opc1 == opc2: # Shifts in the same direction
            size = data.nodes[op.inputs[0]].size_bits
            if sa1 + sa2 < size:
                new_cnode = len(data.nodes)
                data.nodes.append(PcodeVarnode(new_cnode, hex(sa1 + sa2), node1.size_bits, "const"))
                data.op_set_opcode(op, opc1)
                data.op_set_input(op, def_op.inputs[0], 0)
                data.op_set_input(op, new_cnode, 1)
                return 1
        return 0

class RuleIdentityEl(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "identityel")

    def get_op_list(self) -> list[str]:
        return ["INT_ADD", "INT_XOR", "INT_OR", "BOOL_XOR", "BOOL_OR", "INT_MULT", "INT_LEFT", "INT_RIGHT", "INT_SRIGHT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        node = data.nodes[op.inputs[1]]
        if node.kind != "const": return 0
        
        val = int(node.name, 16)
        if val == 0 and op.opcode != "INT_MULT":
            data.op_set_opcode(op, "COPY")
            data.op_remove_input(op, 1)
            return 1
            
        if op.opcode == "INT_MULT":
            if val == 1:
                data.op_set_opcode(op, "COPY")
                data.op_remove_input(op, 1)
                return 1
            if val == 0:
                data.op_set_opcode(op, "COPY")
                data.op_remove_input(op, 0)
                return 1
                
        return 0

class RuleSubCancel(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "subcancel")

    def get_op_list(self) -> list[str]:
        return ["SUBPIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        node = data.nodes[op.inputs[1]]
        if node.kind != "const": return 0
        
        offset = int(node.name, 16)
        out_node = data.nodes[op.output]
        out_size = out_node.size_bits
        
        def_op = None
        for cand_op in data.ops:
            if cand_op.output == op.inputs[0]:
                def_op = cand_op
                break
                
        if def_op is None: return 0
        
        opc = def_op.opcode
        if opc not in ["INT_ZEXT", "INT_SEXT", "INT_AND"]: return 0
        
        if opc == "INT_AND":
            if offset == 0:
                node_and_c = data.nodes[def_op.inputs[1]]
                if node_and_c.kind == "const":
                    mask = int(node_and_c.name, 16)
                    if mask == ((1 << out_size) - 1):
                        data.op_set_input(op, def_op.inputs[0], 0)
                        return 1
            return 0
            
        far_in_node = data.nodes[def_op.inputs[0]]
        far_in_size = far_in_node.size_bits
        
        if offset == 0:
            if far_in_size == out_size:
                data.op_set_opcode(op, "COPY")
                data.op_remove_input(op, 1)
                data.op_set_input(op, def_op.inputs[0], 0)
                return 1
            if out_size < far_in_size:
                data.op_set_input(op, def_op.inputs[0], 0)
                return 1
            if out_size > far_in_size:
                data.op_set_opcode(op, opc)
                data.op_remove_input(op, 1)
                data.op_set_input(op, def_op.inputs[0], 0)
                return 1
        elif offset >= far_in_size // 8:
            if opc == "INT_ZEXT":
                cnode = len(data.nodes)
                data.nodes.append(PcodeVarnode(cnode, hex(0), out_size, "const"))
                data.op_set_opcode(op, "COPY")
                data.op_remove_input(op, 1)
                data.op_set_input(op, cnode, 0)
                return 1
        return 0

class RuleDoubleSub(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "doublesub")

    def get_op_list(self) -> list[str]:
        return ["SUBPIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        
        def_op = None
        for cand_op in data.ops:
            if cand_op.output == op.inputs[0]:
                def_op = cand_op
                break
                
        if def_op is None or def_op.opcode != "SUBPIECE": return 0
        
        cnode1 = data.nodes[op.inputs[1]]
        cnode2 = data.nodes[def_op.inputs[1]]
        
        if cnode1.kind != "const" or cnode2.kind != "const": return 0
        
        offset1 = int(cnode1.name, 16)
        offset2 = int(cnode2.name, 16)
        
        new_cnode = len(data.nodes)
        data.nodes.append(PcodeVarnode(new_cnode, hex(offset1 + offset2), cnode1.size_bits, "const"))
        
        data.op_set_input(op, def_op.inputs[0], 0)
        data.op_set_input(op, new_cnode, 1)
        return 1

class RuleShift2Mult(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "shift2mult")

    def get_op_list(self) -> list[str]:
        return ["INT_LEFT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        val = int(cnode.name, 16)
        if val >= 32: return 0
        
        # Check if it interacts with arithmetic (just check def and uses)
        flag = False
        
        def_op = None
        for cand_op in data.ops:
            if cand_op.output == op.inputs[0]:
                def_op = cand_op
                break
                
        if def_op and def_op.opcode in ["INT_ADD", "INT_SUB", "INT_MULT"]:
            flag = True
        else:
            for cand_op in data.ops:
                if op.output in cand_op.inputs:
                    if cand_op.opcode in ["INT_ADD", "INT_SUB", "INT_MULT"]:
                        flag = True
                        break
                        
        if not flag: return 0
        
        new_cnode = len(data.nodes)
        data.nodes.append(PcodeVarnode(new_cnode, hex(1 << val), cnode.size_bits, "const"))
        data.op_set_opcode(op, "INT_MULT")
        data.op_set_input(op, new_cnode, 1)
        return 1

class RulePropagateCopy(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "propagatecopy")

    def get_op_list(self) -> list[str]:
        return ["COPY"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        vn = op.output
        invn = op.inputs[0]
        if vn == invn: return 0
        if invn < 0: return 0
        
        count = 0
        for other_op in data.ops:
            if other_op == op: continue
            for slot, in_vn in enumerate(other_op.inputs):
                if vn == in_vn:
                    # In minighidra, we just freely propagate unless there's some restriction
                    # The graph is still somewhat simple
                    data.op_set_input(other_op, invn, slot)
                    count += 1
        return count

class Rule2Comp2Mult(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "2comp2mult")

    def get_op_list(self) -> list[str]:
        return ["INT_2COMP"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 1: return 0
        in_size = data.nodes[op.inputs[0]].size_bits
        
        mask = (1 << in_size) - 1
        cnode = len(data.nodes)
        # -1 in two's complement for the given size
        val = mask
        data.nodes.append(PcodeVarnode(cnode, hex(val), in_size, "const"))
        
        data.op_set_opcode(op, "INT_MULT")
        data.op_set_input(op, cnode, 1)
        return 1

class RuleShiftAnd(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "shiftand")

    def get_op_list(self) -> list[str]:
        return ["INT_LEFT", "INT_RIGHT", "INT_MULT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cvn = data.nodes[op.inputs[1]]
        if cvn.kind != "const": return 0
        
        def_op = None
        for cand_op in data.ops:
            if cand_op.output == op.inputs[0]:
                def_op = cand_op
                break
                
        if def_op is None or def_op.opcode != "INT_AND": return 0
        
        if len(def_op.inputs) != 2: return 0
        maskvn = data.nodes[def_op.inputs[1]]
        if maskvn.kind != "const": return 0
        
        mask = int(maskvn.name, 16)
        invn = def_op.inputs[0]
        
        opc = op.opcode
        
        def leastsigbit_set(n):
            if n == 0: return -1
            return (n & -n).bit_length() - 1

        if opc in ["INT_RIGHT", "INT_LEFT"]:
            sa = int(cvn.name, 16)
        else:
            sa = leastsigbit_set(int(cvn.name, 16))
            if sa <= 0: return 0
            if (1 << sa) != int(cvn.name, 16): return 0
            opc = "INT_LEFT"
            
        nzm = data.get_nzmask(invn)
        size = data.nodes[invn].size_bits
        fullmask = (1 << size) - 1
        
        if opc == "INT_RIGHT":
            nzm >>= sa
            mask >>= sa
        else:
            nzm <<= sa
            mask <<= sa
            nzm &= fullmask
            mask &= fullmask
            
        if (mask & nzm) != nzm: return 0
        data.op_set_input(op, invn, 0)
        return 1

class RuleLess2Zero(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "less2zero")

    def get_op_list(self) -> list[str]:
        return ["INT_LESS"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        lvn = data.nodes[op.inputs[0]]
        rvn = data.nodes[op.inputs[1]]
        
        if lvn.kind == "const":
            val = int(lvn.name, 16)
            mask = (1 << lvn.size_bits) - 1
            if val == 0:
                data.op_set_opcode(op, "INT_NOTEQUAL")
                return 1
            elif val == mask:
                cnode = len(data.nodes)
                data.nodes.append(PcodeVarnode(cnode, hex(0), 1, "const"))
                data.op_set_opcode(op, "COPY")
                data.op_remove_input(op, 1)
                data.op_set_input(op, cnode, 0)
                return 1
        elif rvn.kind == "const":
            val = int(rvn.name, 16)
            mask = (1 << rvn.size_bits) - 1
            if val == 0:
                cnode = len(data.nodes)
                data.nodes.append(PcodeVarnode(cnode, hex(0), 1, "const"))
                data.op_set_opcode(op, "COPY")
                data.op_remove_input(op, 1)
                data.op_set_input(op, cnode, 0)
                return 1
            elif val == mask:
                data.op_set_opcode(op, "INT_NOTEQUAL")
                return 1
        return 0

class RuleLessEqual2Zero(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "lessequal2zero")

    def get_op_list(self) -> list[str]:
        return ["INT_LESSEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        lvn = data.nodes[op.inputs[0]]
        rvn = data.nodes[op.inputs[1]]
        
        if lvn.kind == "const":
            val = int(lvn.name, 16)
            mask = (1 << lvn.size_bits) - 1
            if val == 0:
                cnode = len(data.nodes)
                data.nodes.append(PcodeVarnode(cnode, hex(1), 1, "const"))
                data.op_set_opcode(op, "COPY")
                data.op_remove_input(op, 1)
                data.op_set_input(op, cnode, 0)
                return 1
            elif val == mask:
                data.op_set_opcode(op, "INT_EQUAL")
                return 1
        elif rvn.kind == "const":
            val = int(rvn.name, 16)
            mask = (1 << rvn.size_bits) - 1
            if val == 0:
                data.op_set_opcode(op, "INT_EQUAL")
                return 1
            elif val == mask:
                cnode = len(data.nodes)
                data.nodes.append(PcodeVarnode(cnode, hex(1), 1, "const"))
                data.op_set_opcode(op, "COPY")
                data.op_remove_input(op, 1)
                data.op_set_input(op, cnode, 0)
                return 1
        return 0

class RuleEqual2Constant(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "equal2constant")

    def get_op_list(self) -> list[str]:
        return ["INT_EQUAL", "INT_NOTEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        
        cvn_idx = op.inputs[1]
        lhs_idx = op.inputs[0]
        
        cvn = data.nodes[cvn_idx]
        if cvn.kind != "const": return 0
        
        lhs = data.nodes[lhs_idx]
        def_op = None
        for cand in data.ops:
            if cand.output == lhs_idx:
                def_op = cand
                break
                
        if def_op is None: return 0
        
        opc = def_op.opcode
        newconst = 0
        a = -1
        mask = (1 << cvn.size_bits) - 1
        
        if opc == "INT_ADD":
            if len(def_op.inputs) != 2: return 0
            otherconst = data.nodes[def_op.inputs[1]]
            if otherconst.kind != "const": return 0
            newconst = int(cvn.name, 16) - int(otherconst.name, 16)
            newconst &= mask
            a = def_op.inputs[0]
        elif opc == "INT_MULT":
            if len(def_op.inputs) != 2: return 0
            otherconst = data.nodes[def_op.inputs[1]]
            if otherconst.kind != "const": return 0
            # Multiply by -1
            othermask = (1 << otherconst.size_bits) - 1
            if int(otherconst.name, 16) != othermask: return 0
            newconst = int(cvn.name, 16)
            newconst = (-newconst) & othermask
            a = def_op.inputs[0]
        elif opc == "INT_NEGATE":
            newconst = int(cvn.name, 16)
            newconst = (~newconst) & mask
            a = def_op.inputs[0]
        else:
            return 0
            
        uses = 0
        for cand in data.ops:
            if cand == op: continue
            if lhs_idx in cand.inputs:
                if cand.opcode not in ["INT_EQUAL", "INT_NOTEQUAL"]: return 0
                if data.nodes[cand.inputs[1]].kind != "const": return 0
                uses += 1
                
        new_cnode = len(data.nodes)
        data.nodes.append(PcodeVarnode(new_cnode, hex(newconst), data.nodes[a].size_bits, "const"))
        data.op_set_input(op, a, 0)
        data.op_set_input(op, new_cnode, 1)
        return 1

class RuleSLess2Zero(Rule):
    def __init__(self, group: str) -> None:
        super().__init__(group, "sless2zero")

    def get_op_list(self) -> list[str]:
        return ["INT_SLESS"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        lvn = data.nodes[op.inputs[0]]
        rvn_idx = op.inputs[1]
        
        if lvn.kind == "const":
            if int(lvn.name, 16) == (1 << lvn.size_bits) - 1: # -1
                def_op = None
                for cand in data.ops:
                    if cand.output == rvn_idx:
                        def_op = cand
                        break
                if not def_op: return 0
                
                feedOpCode = def_op.opcode
                if feedOpCode == "SUBPIECE":
                    avn_idx = def_op.inputs[0]
                    if data.nodes[avn_idx].size_bits > 64: return 0
                    cnode = data.nodes[def_op.inputs[1]]
                    if cnode.kind == "const":
                        # cnode is in bytes, so size_bits of rvn + offset * 8 == avn size_bits
                        # Wait, minighidra SUBPIECE offset is just bytes.
                        offset_bytes = int(cnode.name, 16)
                        if data.nodes[rvn_idx].size_bits + offset_bytes * 8 == data.nodes[avn_idx].size_bits:
                            mask = (1 << data.nodes[avn_idx].size_bits) - 1
                            cnode_new = len(data.nodes)
                            data.nodes.append(PcodeVarnode(cnode_new, hex(mask), data.nodes[avn_idx].size_bits, "const"))
                            
                            data.op_set_input(op, avn_idx, 1)
                            data.op_set_input(op, cnode_new, 0)
                            return 1
                elif feedOpCode == "INT_NEGATE":
                    avn_idx = def_op.inputs[0]
                    data.op_set_input(op, avn_idx, 0)
                    cnode_new = len(data.nodes)
                    data.nodes.append(PcodeVarnode(cnode_new, hex(0), data.nodes[avn_idx].size_bits, "const"))
                    data.op_set_input(op, cnode_new, 1)
                    return 1
        return 0
