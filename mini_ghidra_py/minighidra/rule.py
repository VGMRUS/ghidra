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


class RuleAndDistribute(Rule):
    """Distribute INT_AND through INT_OR if result is simpler: A & (B | C) => (A & B) | (A & C)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "anddistribute")

    def get_op_list(self) -> list[str]:
        return ["INT_AND"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        
        sz = data.nodes[op.output].size_bits if op.output is not None else data.nodes[op.inputs[0]].size_bits
        fullmask = (1 << sz) - 1
        
        for i in range(2):
            othervn = op.inputs[1-i]
            orvn = op.inputs[i]
            
            orop = None
            for cand in data.ops:
                if cand.output == orvn:
                    orop = cand
                    break
                    
            if not orop or orop.opcode != "INT_OR": continue
            if len(orop.inputs) != 2: continue
            
            othermask = data.get_nzmask(othervn)
            if othermask == 0: continue
            if othermask == fullmask: continue
            
            ormask1 = data.get_nzmask(orop.inputs[0])
            if (ormask1 & othermask) == 0: break
            
            ormask2 = data.get_nzmask(orop.inputs[1])
            if (ormask2 & othermask) == 0: break
            
            if data.nodes[othervn].kind == "const":
                if (ormask1 & othermask) == ormask1: break
                if (ormask2 & othermask) == ormask2: break
        else:
            return 0
            
        newop1 = PcodeOp(op.addr, "INT_AND", len(data.nodes), (orop.inputs[0], othervn))
        data.nodes.append(PcodeVarnode(newop1.output, None, sz, "unique"))
        data.ops.append(newop1)
        
        newop2 = PcodeOp(op.addr, "INT_AND", len(data.nodes), (orop.inputs[1], othervn))
        data.nodes.append(PcodeVarnode(newop2.output, None, sz, "unique"))
        data.ops.append(newop2)
        
        data.op_set_input(op, newop1.output, 0)
        data.op_set_input(op, newop2.output, 1)
        data.op_set_opcode(op, "INT_OR")
        return 1

class RuleLessOne(Rule):
    """Transform INT_LESS of 0 or 1: V < 1 => V == 0, V <= 0 => V == 0"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "lessone")

    def get_op_list(self) -> list[str]:
        return ["INT_LESS", "INT_LESSEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        val = int(cnode.name, 16)
        if op.opcode == "INT_LESS" and val != 1: return 0
        if op.opcode == "INT_LESSEQUAL" and val != 0: return 0
        
        data.op_set_opcode(op, "INT_EQUAL")
        if val != 0:
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, hex(0), cnode.size_bits, "const"))
            data.op_set_input(op, cnew, 1)
        return 1


class RuleAndCommute(Rule):
    """Commute INT_AND with INT_LEFT and INT_RIGHT: (V << W) & d => (V & (W >> c)) << c"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "andcommute")

    def get_op_list(self) -> list[str]:
        return ["INT_AND"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        sz = data.nodes[op.output].size_bits if op.output is not None else data.nodes[op.inputs[0]].size_bits
        fullmask = (1 << sz) - 1
        
        for i in range(2):
            shiftvn = op.inputs[i]
            shiftop = None
            for cand in data.ops:
                if cand.output == shiftvn:
                    shiftop = cand
                    break
            
            if not shiftop: continue
            opc = shiftop.opcode
            if opc not in ("INT_LEFT", "INT_RIGHT"): continue
            if len(shiftop.inputs) != 2: continue
            
            savn = shiftop.inputs[1]
            cnode = data.nodes[savn]
            if cnode.kind != "const": continue
            sa = int(cnode.name, 16)
            
            othervn = op.inputs[1-i]
            othermask = data.get_nzmask(othervn)
            
            if opc == "INT_RIGHT":
                if (fullmask >> sa) == othermask: continue
                othermask = (othermask << sa) & fullmask
            else:
                if ((fullmask << sa) & fullmask) == othermask: continue
                othermask = (othermask >> sa)
                
            if othermask == 0: continue
            if othermask == fullmask: continue
            
            orvn = shiftop.inputs[0]
            if opc == "INT_LEFT" and data.nodes[othervn].kind == "const":
                count = sum(1 for cand in data.ops if shiftvn in cand.inputs)
                if count == 1:
                    break
                    
            orop = None
            for cand in data.ops:
                if cand.output == orvn:
                    orop = cand
                    break
            if not orop: continue
            
            if orop.opcode == "INT_OR" and len(orop.inputs) == 2:
                ormask1 = data.get_nzmask(orop.inputs[0])
                if (ormask1 & othermask) == 0: break
                ormask2 = data.get_nzmask(orop.inputs[1])
                if (ormask2 & othermask) == 0: break
                if data.nodes[othervn].kind == "const":
                    if (ormask1 & othermask) == ormask1: break
                    if (ormask2 & othermask) == ormask2: break
            elif orop.opcode == "PIECE" and len(orop.inputs) == 2:
                ormask1 = data.get_nzmask(orop.inputs[1])
                if (ormask1 & othermask) == 0: break
                ormask2 = data.get_nzmask(orop.inputs[0])
                low_sz = data.nodes[orop.inputs[1]].size_bits
                ormask2 <<= (low_sz * 8)
                if (ormask2 & othermask) == 0: break
        else:
            return 0
            
        newop1 = PcodeOp(op.addr, "INT_RIGHT" if opc == "INT_LEFT" else "INT_LEFT", len(data.nodes), (othervn, savn))
        data.nodes.append(PcodeVarnode(newop1.output, None, sz, "unique"))
        data.ops.append(newop1)
        
        newop2 = PcodeOp(op.addr, "INT_AND", len(data.nodes), (orvn, newop1.output))
        data.nodes.append(PcodeVarnode(newop2.output, None, sz, "unique"))
        data.ops.append(newop2)
        
        data.op_set_input(op, newop2.output, 0)
        data.op_set_input(op, savn, 1)
        data.op_set_opcode(op, opc)
        return 1

class RuleAndPiece(Rule):
    """Convert PIECE to INT_ZEXT where appropriate: V & concat(W,X) => zext(X)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "andpiece")

    def get_op_list(self) -> list[str]:
        return ["INT_AND"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        sz = data.nodes[op.output].size_bits if op.output is not None else data.nodes[op.inputs[0]].size_bits
        
        for i in range(2):
            piecevn = op.inputs[i]
            pieceop = None
            for cand in data.ops:
                if cand.output == piecevn:
                    pieceop = cand
                    break
            if not pieceop or pieceop.opcode != "PIECE" or len(pieceop.inputs) != 2: continue
            
            othervn = op.inputs[1-i]
            othermask = data.get_nzmask(othervn)
            
            if othermask == ((1 << sz) - 1): continue
            if othermask == 0: continue
            
            highvn = pieceop.inputs[0]
            lowvn = pieceop.inputs[1]
            maskhigh = data.get_nzmask(highvn)
            masklow = data.get_nzmask(lowvn)
            low_sz = data.nodes[lowvn].size_bits
            
            if (maskhigh & (othermask >> (low_sz * 8))) == 0:
                if maskhigh == 0 and data.nodes[highvn].kind == "const": continue
                opc = "INT_ZEXT"
                break
            elif (masklow & othermask) == 0:
                if data.nodes[lowvn].kind == "const": continue
                opc = "PIECE"
                break
        else:
            return 0
            
        if opc == "INT_ZEXT":
            newop = PcodeOp(op.addr, opc, len(data.nodes), (lowvn,))
            data.nodes.append(PcodeVarnode(newop.output, None, sz, "unique"))
        else:
            newc = len(data.nodes)
            data.nodes.append(PcodeVarnode(newc, hex(0), low_sz, "const"))
            newop = PcodeOp(op.addr, opc, len(data.nodes), (highvn, newc))
            data.nodes.append(PcodeVarnode(newop.output, None, sz, "unique"))
            
        data.ops.append(newop)
        data.op_set_input(op, newop.output, i)
        return 1

class RuleAndZext(Rule):
    """Convert INT_AND to INT_ZEXT where appropriate: sext(X) & 0xffff => zext(X)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "andzext")

    def get_op_list(self) -> list[str]:
        return ["INT_AND"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        in0 = op.inputs[0]
        otherop = None
        for cand in data.ops:
            if cand.output == in0:
                otherop = cand
                break
        if not otherop: return 0
        
        if otherop.opcode == "INT_SEXT":
            rootvn = otherop.inputs[0]
        elif otherop.opcode == "PIECE":
            if len(otherop.inputs) != 2: return 0
            rootvn = otherop.inputs[1]
        else:
            return 0
            
        mask = (1 << data.nodes[rootvn].size_bits) - 1
        if mask != int(cnode.name, 16): return 0
        
        data.op_set_opcode(op, "INT_ZEXT")
        data.op_remove_input(op, 1)
        data.op_set_input(op, rootvn, 0)
        return 1


class RuleAndCompare(Rule):
    """Simplify INT_ZEXT and SUBPIECE in masked comparison: zext(V) & c == 0 => V & (c & mask) == 0"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "andcompare")

    def get_op_list(self) -> list[str]:
        return ["INT_EQUAL", "INT_NOTEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const" or int(cnode.name, 16) != 0: return 0
        
        andvn = op.inputs[0]
        andop = None
        for cand in data.ops:
            if cand.output == andvn:
                andop = cand
                break
        if not andop or andop.opcode != "INT_AND": return 0
        if len(andop.inputs) != 2: return 0
        
        andcnode = data.nodes[andop.inputs[1]]
        if andcnode.kind != "const": return 0
        
        subvn = andop.inputs[0]
        subop = None
        for cand in data.ops:
            if cand.output == subvn:
                subop = cand
                break
        if not subop: return 0
        
        if subop.opcode == "SUBPIECE" and len(subop.inputs) == 2:
            basevn = subop.inputs[0]
            baseconst = int(andcnode.name, 16)
            shift_cnode = data.nodes[subop.inputs[1]]
            if shift_cnode.kind != "const": return 0
            andconst = baseconst << (int(shift_cnode.name, 16) * 8)
        elif subop.opcode == "INT_ZEXT" and len(subop.inputs) == 1:
            basevn = subop.inputs[0]
            baseconst = int(andcnode.name, 16)
            mask = (1 << data.nodes[basevn].size_bits) - 1
            andconst = baseconst & mask
        else:
            return 0
            
        fullmask = (1 << data.nodes[andvn].size_bits) - 1
        if baseconst == fullmask: return 0
        
        basesz = data.nodes[basevn].size_bits
        constvn = len(data.nodes)
        data.nodes.append(PcodeVarnode(constvn, hex(andconst), basesz, "const"))
        
        newop = PcodeOp(andop.addr, "INT_AND", len(data.nodes), (basevn, constvn))
        data.nodes.append(PcodeVarnode(newop.output, None, basesz, "unique"))
        data.ops.append(newop)
        
        data.op_set_input(op, newop.output, 0)
        cnode2 = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnode2, hex(0), basesz, "const"))
        data.op_set_input(op, cnode2, 1)
        return 1


class RuleDoubleArithShift(Rule):
    """Simplify two sequential INT_SRIGHT: (x s>> c) s>> d => x s>> saturate(c + d)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "doublearithshift")

    def get_op_list(self) -> list[str]:
        return ["INT_SRIGHT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        constD = data.nodes[op.inputs[1]]
        if constD.kind != "const": return 0
        
        shiftin = op.inputs[0]
        shift2op = None
        for cand in data.ops:
            if cand.output == shiftin:
                shift2op = cand
                break
        
        if not shift2op or shift2op.opcode != "INT_SRIGHT": return 0
        constC = data.nodes[shift2op.inputs[1]]
        if constC.kind != "const": return 0
        
        invn = shift2op.inputs[0]
        max_shift = (data.nodes[op.output].size_bits if op.output is not None else data.nodes[shiftin].size_bits) * 8 - 1
        
        sa = int(constC.name, 16) + int(constD.name, 16)
        if sa <= 0: return 0
        if sa > max_shift:
            sa = max_shift
            
        data.op_set_input(op, invn, 0)
        cnode = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnode, hex(sa), constD.size_bits, "const"))
        data.op_set_input(op, cnode, 1)
        return 1

class RuleConcatShift(Rule):
    """Simplify INT_RIGHT canceling PIECE: concat(V,W) >> c => zext(V)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "concatshift")

    def get_op_list(self) -> list[str]:
        return ["INT_RIGHT", "INT_SRIGHT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        shiftin = op.inputs[0]
        concatop = None
        for cand in data.ops:
            if cand.output == shiftin:
                concatop = cand
                break
                
        if not concatop or concatop.opcode != "PIECE": return 0
        if len(concatop.inputs) != 2: return 0
        
        sa = int(cnode.name, 16)
        leastsize = data.nodes[concatop.inputs[1]].size_bits * 8
        if sa < leastsize: return 0
        
        mainin = concatop.inputs[0]
        sa -= leastsize
        extcode = "INT_ZEXT" if op.opcode == "INT_RIGHT" else "INT_SEXT"
        
        if sa == 0:
            data.op_remove_input(op, 1)
            data.op_set_opcode(op, extcode)
            data.op_set_input(op, mainin, 0)
        else:
            extop = PcodeOp(op.addr, extcode, len(data.nodes), (mainin,))
            data.nodes.append(PcodeVarnode(extop.output, None, data.nodes[shiftin].size_bits, "unique"))
            data.ops.append(extop)
            
            data.op_set_input(op, extop.output, 0)
            cnode2 = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnode2, hex(sa), cnode.size_bits, "const"))
            data.op_set_input(op, cnode2, 1)
            
        return 1

class RuleLeftRight(Rule):
    """Transform canceling INT_RIGHT or INT_SRIGHT of INT_LEFT"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "leftright")

    def get_op_list(self) -> list[str]:
        return ["INT_RIGHT", "INT_SRIGHT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        shiftin = op.inputs[0]
        leftshift = None
        for cand in data.ops:
            if cand.output == shiftin:
                leftshift = cand
                break
                
        if not leftshift or leftshift.opcode != "INT_LEFT": return 0
        if len(leftshift.inputs) != 2: return 0
        leftcnode = data.nodes[leftshift.inputs[1]]
        if leftcnode.kind != "const": return 0
        
        sa = int(cnode.name, 16)
        if int(leftcnode.name, 16) != sa: return 0
        if (sa & 7) != 0: return 0
        
        isa = sa >> 3
        tsz = data.nodes[shiftin].size_bits - isa
        if tsz not in (1, 2, 4, 8): return 0
        
        count = sum(1 for cand in data.ops if shiftin in cand.inputs)
        if count != 1: return 0
        
        # Subpiece extraction
        data.op_set_opcode(leftshift, "SUBPIECE")
        cnew = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew, hex(0), leftcnode.size_bits, "const"))
        data.op_set_input(leftshift, cnew, 1)
        
        # Update leftshift output size
        data.nodes[leftshift.output].size_bits = tsz
        
        data.op_remove_input(op, 1)
        data.op_set_opcode(op, "INT_SEXT" if op.opcode == "INT_SRIGHT" else "INT_ZEXT")
        return 1


class RuleShiftCompare(Rule):
    """Transform shifts in comparisons: V >> c == d => V == (d << c)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "shiftcompare")

    def get_op_list(self) -> list[str]:
        return ["INT_EQUAL", "INT_NOTEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        constvn = data.nodes[op.inputs[1]]
        if constvn.kind != "const": return 0
        
        shiftvn = op.inputs[0]
        shiftop = None
        for cand in data.ops:
            if cand.output == shiftvn:
                shiftop = cand
                break
        if not shiftop: return 0
        
        opc = shiftop.opcode
        isleft = False
        sa = 0
        
        if opc == "INT_LEFT":
            isleft = True
            if len(shiftop.inputs) != 2: return 0
            savn = data.nodes[shiftop.inputs[1]]
            if savn.kind != "const": return 0
            sa = int(savn.name, 16)
        elif opc == "INT_RIGHT":
            isleft = False
            if len(shiftop.inputs) != 2: return 0
            savn = data.nodes[shiftop.inputs[1]]
            if savn.kind != "const": return 0
            sa = int(savn.name, 16)
            
            count = sum(1 for cand in data.ops if shiftvn in cand.inputs)
            if count != 1: return 0
        elif opc == "INT_MULT":
            isleft = True
            if len(shiftop.inputs) != 2: return 0
            savn = data.nodes[shiftop.inputs[1]]
            if savn.kind != "const": return 0
            val = int(savn.name, 16)
            if val == 0 or (val & (val - 1)) != 0: return 0 # power of 2
            sa = (val & -val).bit_length() - 1
        elif opc == "INT_DIV":
            isleft = False
            if len(shiftop.inputs) != 2: return 0
            savn = data.nodes[shiftop.inputs[1]]
            if savn.kind != "const": return 0
            val = int(savn.name, 16)
            if val == 0 or (val & (val - 1)) != 0: return 0
            sa = (val & -val).bit_length() - 1
            
            count = sum(1 for cand in data.ops if shiftvn in cand.inputs)
            if count != 1: return 0
        else:
            return 0
            
        if sa == 0: return 0
        
        mainvn = shiftop.inputs[0]
        constval = int(constvn.name, 16)
        nzmask = data.get_nzmask(mainvn)
        shiftvn_sz = data.nodes[shiftvn].size_bits
        
        if isleft:
            newconst = constval >> sa
            if (newconst << sa) != constval: return 0
            
            tmp = (nzmask << sa) & ((1 << shiftvn_sz) - 1)
            if (tmp >> sa) != nzmask:
                count = sum(1 for cand in data.ops if shiftvn in cand.inputs)
                if count != 1: return 0
                
                sa2 = shiftvn_sz * 8 - sa
                tmp2 = (1 << sa2) - 1
                cnew = len(data.nodes)
                data.nodes.append(PcodeVarnode(cnew, hex(tmp2), constvn.size_bits, "const"))
                
                newop = PcodeOp(op.addr, "INT_AND", len(data.nodes), (mainvn, cnew))
                data.nodes.append(PcodeVarnode(newop.output, None, constvn.size_bits, "unique"))
                data.ops.append(newop)
                
                data.op_set_input(op, newop.output, 0)
                cnew2 = len(data.nodes)
                data.nodes.append(PcodeVarnode(cnew2, hex(newconst), constvn.size_bits, "const"))
                data.op_set_input(op, cnew2, 1)
                return 1
        else:
            if ((nzmask >> sa) << sa) != nzmask: return 0
            newconst = (constval << sa) & ((1 << shiftvn_sz) - 1)
            if (newconst >> sa) != constval: return 0
            
        cnew3 = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew3, hex(newconst), constvn.size_bits, "const"))
        data.op_set_input(op, mainvn, 0)
        data.op_set_input(op, cnew3, 1)
        return 1

class RuleShiftPiece(Rule):
    """Simplify combinations of shifts and pieces."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "shiftpiece")

    def get_op_list(self) -> list[str]:
        return ["PIECE", "INT_OR"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        vn1 = op.inputs[0]
        vn2 = op.inputs[1]
        
        shiftop = None
        for cand in data.ops:
            if cand.output == vn1:
                shiftop = cand
                break
        
        zextloop = None
        for cand in data.ops:
            if cand.output == vn2:
                zextloop = cand
                break
                
        if not shiftop or not zextloop: return 0
        
        def multPowerOf2(sop):
            if sop.opcode == "INT_LEFT" and len(sop.inputs) == 2:
                c = data.nodes[sop.inputs[1]]
                if c.kind == "const": return int(c.name, 16)
            elif sop.opcode == "INT_MULT" and len(sop.inputs) == 2:
                c = data.nodes[sop.inputs[1]]
                if c.kind == "const":
                    val = int(c.name, 16)
                    if val != 0 and (val & (val - 1)) == 0:
                        return (val & -val).bit_length() - 1
            return -1
            
        sa = multPowerOf2(shiftop)
        if sa < 0:
            sa = multPowerOf2(zextloop)
            if sa < 0: return 0
            shiftop, zextloop = zextloop, shiftop
            
        if len(shiftop.inputs) == 0: return 0
        vn1 = shiftop.inputs[0]
        
        zexthiop = None
        for cand in data.ops:
            if cand.output == vn1:
                zexthiop = cand
                break
        
        if not zexthiop or zexthiop.opcode not in ("INT_ZEXT", "INT_SEXT"): return 0
        if len(zexthiop.inputs) == 0: return 0
        vn1 = zexthiop.inputs[0]
        if data.nodes[vn1].kind == "const" and data.nodes[vn1].size_bits < 8: return 0
        
        concatsize = sa + 8 * data.nodes[vn1].size_bits
        out_sz = data.nodes[op.output].size_bits if op.output is not None else data.nodes[vn2].size_bits
        if out_sz * 8 < concatsize: return 0
        
        if zextloop.opcode != "INT_ZEXT":
            # the IDIV CDQ case handled in original source
            # complex, skipping for now
            return 0
            
        if len(zextloop.inputs) == 0: return 0
        vn2 = zextloop.inputs[0]
        if sa != 8 * data.nodes[vn2].size_bits: return 0
        
        if concatsize == out_sz * 8:
            data.op_set_opcode(op, "PIECE")
            data.op_set_input(op, vn1, 0)
            data.op_set_input(op, vn2, 1)
        else:
            newop = PcodeOp(op.addr, "PIECE", len(data.nodes), (vn1, vn2))
            data.nodes.append(PcodeVarnode(newop.output, None, concatsize // 8, "unique"))
            data.ops.append(newop)
            
            data.op_set_opcode(op, zexthiop.opcode)
            data.op_remove_input(op, 1)
            data.op_set_input(op, newop.output, 0)
            
        return 1


class RuleZextEliminate(Rule):
    """Eliminate unnecessary INT_ZEXT in comparisons or other ops: zext(V) == c => V == c_small"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "zexteliminate")

    def get_op_list(self) -> list[str]:
        return ["INT_EQUAL", "INT_NOTEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        
        vn1, vn2 = op.inputs[0], op.inputs[1]
        zextslot = 0
        otherslot = 1
        
        zext_op = None
        for cand in data.ops:
            if cand.output == vn2:
                if cand.opcode == "INT_ZEXT":
                    zext_op = cand
                    vn1, vn2 = vn2, vn1
                    zextslot, otherslot = 1, 0
                break
                
        if not zext_op:
            for cand in data.ops:
                if cand.output == vn1:
                    if cand.opcode == "INT_ZEXT":
                        zext_op = cand
                    break
                    
        if not zext_op: return 0
        
        cnode = data.nodes[vn2]
        if cnode.kind != "const": return 0
        
        count = sum(1 for cand in data.ops if vn1 in cand.inputs)
        if count != 1: return 0
        
        zext_in = zext_op.inputs[0]
        smallsize = data.nodes[zext_in].size_bits
        
        val = int(cnode.name, 16)
        if (val >> (8 * smallsize)) == 0:
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, hex(val), smallsize, "const"))
            data.op_set_input(op, zext_in, zextslot)
            data.op_set_input(op, cnew, otherslot)
            return 1
            
        return 0

class RuleSlessToLess(Rule):
    """Convert INT_SLESS to INT_LESS when comparing positive values."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "slesstoless")

    def get_op_list(self) -> list[str]:
        return ["INT_SLESS", "INT_SLESSEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        
        vn0 = op.inputs[0]
        vn1 = op.inputs[1]
        
        sz = data.nodes[vn0].size_bits
        nzm0 = data.get_nzmask(vn0)
        if (nzm0 & (1 << (sz * 8 - 1))) != 0: return 0
        
        nzm1 = data.get_nzmask(vn1)
        if (nzm1 & (1 << (sz * 8 - 1))) != 0: return 0
        
        data.op_set_opcode(op, "INT_LESS" if op.opcode == "INT_SLESS" else "INT_LESSEQUAL")
        return 1

class RuleZextSless(Rule):
    """Transform INT_ZEXT and INT_SLESS: zext(V) s< c => V < c"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "zextsless")

    def get_op_list(self) -> list[str]:
        return ["INT_SLESS", "INT_SLESSEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        
        vn1, vn2 = op.inputs[0], op.inputs[1]
        zextslot = 0
        otherslot = 1
        
        zext_op = None
        for cand in data.ops:
            if cand.output == vn2:
                if cand.opcode == "INT_ZEXT":
                    zext_op = cand
                    vn1, vn2 = vn2, vn1
                    zextslot, otherslot = 1, 0
                break
                
        if not zext_op:
            for cand in data.ops:
                if cand.output == vn1:
                    if cand.opcode == "INT_ZEXT":
                        zext_op = cand
                    break
                    
        if not zext_op: return 0
        
        cnode = data.nodes[vn2]
        if cnode.kind != "const": return 0
        
        zext_in = zext_op.inputs[0]
        smallsize = data.nodes[zext_in].size_bits
        
        val = int(cnode.name, 16)
        if (val >> (8 * smallsize - 1)) != 0: return 0
        
        cnew = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew, hex(val), smallsize, "const"))
        
        data.op_set_input(op, zext_in, zextslot)
        data.op_set_input(op, cnew, otherslot)
        data.op_set_opcode(op, "INT_LESS" if op.opcode == "INT_SLESS" else "INT_LESSEQUAL")
        return 1


class RuleBitUndistribute(Rule):
    """Undo distributed operations through INT_AND, INT_OR, and INT_XOR
    - zext(V) & zext(W) => zext(V & W)
    - (V >> X) | (W >> X) => (V | W) >> X
    """
    def __init__(self, group: str) -> None:
        super().__init__(group, "bitundistribute")

    def get_op_list(self) -> list[str]:
        return ["INT_AND", "INT_OR", "INT_XOR"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        vn1, vn2 = op.inputs[0], op.inputs[1]
        
        op1 = None
        for cand in data.ops:
            if cand.output == vn1:
                op1 = cand
                break
        if not op1: return 0
        
        op2 = None
        for cand in data.ops:
            if cand.output == vn2:
                op2 = cand
                break
        if not op2: return 0
        
        opc = op1.opcode
        if op2.opcode != opc: return 0
        
        vnextra = -1
        if opc in ("INT_ZEXT", "INT_SEXT"):
            if len(op1.inputs) != 1 or len(op2.inputs) != 1: return 0
            in1, in2 = op1.inputs[0], op2.inputs[0]
            if data.nodes[in1].size_bits != data.nodes[in2].size_bits: return 0
        elif opc in ("INT_LEFT", "INT_RIGHT", "INT_SRIGHT"):
            if len(op1.inputs) != 2 or len(op2.inputs) != 2: return 0
            in1, in2 = op1.inputs[1], op2.inputs[1]
            cnode1, cnode2 = data.nodes[in1], data.nodes[in2]
            
            if cnode1.kind == "const" and cnode2.kind == "const":
                if cnode1.name != cnode2.name: return 0
                vnextra = len(data.nodes)
                data.nodes.append(PcodeVarnode(vnextra, cnode1.name, cnode1.size_bits, "const"))
            elif in1 != in2:
                return 0
            else:
                vnextra = in1
                
            in1, in2 = op1.inputs[0], op2.inputs[0]
        else:
            return 0
            
        newext = PcodeOp(op.addr, op.opcode, len(data.nodes), (in1, in2))
        data.nodes.append(PcodeVarnode(newext.output, None, data.nodes[in1].size_bits, "unique"))
        data.ops.append(newext)
        
        data.op_set_opcode(op, opc)
        data.op_set_input(op, newext.output, 0)
        
        if opc in ("INT_ZEXT", "INT_SEXT"):
            data.op_remove_input(op, 1)
        else:
            data.op_set_input(op, vnextra, 1)
            
        return 1


class RuleLogic2Bool(Rule):
    """Convert bitwise logical operations into boolean operations: V & W => V && W"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "logic2bool")

    def get_op_list(self) -> list[str]:
        return ["INT_AND", "INT_OR", "INT_XOR"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        
        boolVn = op.inputs[0]
        if not data.is_boolean_value(boolVn): return 0
        
        in1 = op.inputs[1]
        cnode = data.nodes[in1]
        if cnode.kind == "const":
            if int(cnode.name, 16) > 1: return 0
        elif not data.is_boolean_value(in1):
            return 0
            
        if op.opcode == "INT_AND":
            data.op_set_opcode(op, "BOOL_AND")
        elif op.opcode == "INT_OR":
            data.op_set_opcode(op, "BOOL_OR")
        elif op.opcode == "INT_XOR":
            data.op_set_opcode(op, "BOOL_XOR")
        return 1


class RuleNotDistribute(Rule):
    """Distribute BOOL_NEGATE: !(V && W) => !V || !W"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "notdistribute")

    def get_op_list(self) -> list[str]:
        return ["BOOL_NEGATE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 1: return 0
        compop = None
        for cand in data.ops:
            if cand.output == op.inputs[0]:
                compop = cand
                break
        
        if not compop: return 0
        
        if compop.opcode == "BOOL_AND":
            opc = "BOOL_OR"
        elif compop.opcode == "BOOL_OR":
            opc = "BOOL_AND"
        else:
            return 0
            
        newneg1 = PcodeOp(op.addr, "BOOL_NEGATE", len(data.nodes), (compop.inputs[0],))
        data.nodes.append(PcodeVarnode(newneg1.output, None, 1, "unique"))
        data.ops.append(newneg1)
        
        newneg2 = PcodeOp(op.addr, "BOOL_NEGATE", len(data.nodes), (compop.inputs[1],))
        data.nodes.append(PcodeVarnode(newneg2.output, None, 1, "unique"))
        data.ops.append(newneg2)
        
        data.op_set_opcode(op, opc)
        data.op_set_input(op, newneg1.output, 0)
        
        if len(op.inputs) == 1:
            op.inputs = (op.inputs[0], newneg2.output)
        else:
            data.op_set_input(op, newneg2.output, 1)
        return 1


class RuleBoolNegate(Rule):
    """Push BOOL_NEGATE into the boolean operation creating it if possible: !(A < B) => A >= B"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "boolnegate")

    def get_op_list(self) -> list[str]:
        return ["BOOL_NEGATE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 1: return 0
        vn = op.inputs[0]
        
        flip_op = None
        for cand in data.ops:
            if cand.output == vn:
                flip_op = cand
                break
        if not flip_op: return 0
        
        # ALL descendants must be negates
        for cand in data.ops:
            if vn in cand.inputs and cand.opcode != "BOOL_NEGATE":
                return 0
                
        def get_booleanflip(opc):
            flips = {
                "INT_EQUAL": ("INT_NOTEQUAL", False),
                "INT_NOTEQUAL": ("INT_EQUAL", False),
                "INT_LESS": ("INT_LESSEQUAL", True),
                "INT_LESSEQUAL": ("INT_LESS", True),
                "INT_SLESS": ("INT_SLESSEQUAL", True),
                "INT_SLESSEQUAL": ("INT_SLESS", True),
                "FLOAT_EQUAL": ("FLOAT_NOTEQUAL", False),
                "FLOAT_NOTEQUAL": ("FLOAT_EQUAL", False),
                "FLOAT_LESS": ("FLOAT_LESSEQUAL", True),
                "FLOAT_LESSEQUAL": ("FLOAT_LESS", True),
            }
            return flips.get(opc, (None, False))
            
        opc, flipyes = get_booleanflip(flip_op.opcode)
        if not opc: return 0
        
        data.op_set_opcode(flip_op, opc)
        if flipyes and len(flip_op.inputs) == 2:
            flip_op.inputs = (flip_op.inputs[1], flip_op.inputs[0])
            
        for cand in data.ops:
            if vn in cand.inputs and cand.opcode == "BOOL_NEGATE":
                data.op_set_opcode(cand, "COPY")
                
        return 1


class RuleSubCommute(Rule):
    """Commute SUBPIECE with INT_ADD, INT_MULT, bitwise ops, etc."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "subcommute")

    def get_op_list(self) -> list[str]:
        return ["SUBPIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        base = op.inputs[0]
        offset = int(data.nodes[op.inputs[1]].name, 16)
        outvn = op.output
        if outvn is None: return 0
        
        longform = None
        for cand in data.ops:
            if cand.output == base:
                longform = cand
                break
        if not longform: return 0
        
        j = -1
        opc = longform.opcode
        
        if opc == "INT_LEFT":
            j = 1
            if offset != 0: return 0
            if len(longform.inputs) != 2: return 0
            in0 = longform.inputs[0]
            def0 = next((cand for cand in data.ops if cand.output == in0), None)
            if not def0 or def0.opcode not in ("INT_ZEXT", "PIECE"): return 0
        elif opc in ("INT_REM", "INT_DIV"):
            if offset != 0: return 0
            if len(longform.inputs) != 2: return 0
            in0 = longform.inputs[0]
            zext0 = next((cand for cand in data.ops if cand.output == in0), None)
            if not zext0 or zext0.opcode != "INT_ZEXT": return 0
            in1 = longform.inputs[1]
            zext1 = next((cand for cand in data.ops if cand.output == in1), None)
            
            if zext1 and zext1.opcode == "INT_ZEXT":
                pass # Commute
            elif data.nodes[in1].kind == "const" and data.nodes[zext0.inputs[0]].size_bits <= data.nodes[outvn].size_bits:
                val = int(data.nodes[in1].name, 16)
                smallval = val & ((1 << data.nodes[outvn].size_bits) - 1)
                if val != smallval: return 0
            else:
                return 0
        elif opc in ("INT_SREM", "INT_SDIV"):
            if offset != 0: return 0
            if len(longform.inputs) != 2: return 0
            in0 = longform.inputs[0]
            sext0 = next((cand for cand in data.ops if cand.output == in0), None)
            if not sext0 or sext0.opcode != "INT_SEXT": return 0
            in1 = longform.inputs[1]
            sext1 = next((cand for cand in data.ops if cand.output == in1), None)
            
            if sext1 and sext1.opcode == "INT_SEXT":
                pass
            elif data.nodes[in1].kind == "const" and data.nodes[sext0.inputs[0]].size_bits <= data.nodes[outvn].size_bits:
                val = int(data.nodes[in1].name, 16)
                smallval = val & ((1 << data.nodes[outvn].size_bits) - 1)
                sz_out = data.nodes[outvn].size_bits
                sz_in = data.nodes[in0].size_bits
                # sign extend smallval to insize
                sign_bit = (smallval >> (sz_out * 8 - 1)) & 1
                if sign_bit:
                    smallval = smallval | (((1 << (sz_in * 8 - sz_out * 8)) - 1) << (sz_out * 8))
                if val != smallval: return 0
            else:
                return 0
        elif opc in ("INT_ADD", "INT_MULT"):
            if offset != 0: return 0
        elif opc in ("INT_NEGATE", "INT_XOR", "INT_AND", "INT_OR"):
            pass
        else:
            return 0
            
        count = sum(1 for cand in data.ops if base in cand.inputs)
        if count != 1: return 0
        
        # We skip check for RuleSubZext overlap to keep logic simpler for now
        
        new_inputs = []
        for i, vn in enumerate(longform.inputs):
            if i != j:
                newsub = PcodeOp(op.addr, "SUBPIECE", len(data.nodes), (vn, op.inputs[1]))
                outsz = data.nodes[outvn].size_bits
                data.nodes.append(PcodeVarnode(newsub.output, None, outsz, "unique"))
                data.ops.append(newsub)
                new_inputs.append(newsub.output)
            else:
                new_inputs.append(vn)
                
        longform.inputs = tuple(new_inputs)
        
        data.op_remove_input(op, 1)
        data.op_set_opcode(op, "COPY")
        data.op_set_input(op, longform.output, 0)
        return 1


class RuleConcatCommute(Rule):
    """Commute PIECE with INT_AND, INT_OR, and INT_XOR"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "concatcommute")

    def get_op_list(self) -> list[str]:
        return ["PIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        out_sz = data.nodes[op.output].size_bits if op.output is not None else 0
        if out_sz > 8: return 0  # Precision limit for constants in ghidra usually
        
        for i in range(2):
            vn = op.inputs[i]
            logicop = None
            for cand in data.ops:
                if cand.output == vn:
                    logicop = cand
                    break
            if not logicop: continue
            
            opc = logicop.opcode
            if opc not in ("INT_OR", "INT_XOR", "INT_AND"): continue
            if len(logicop.inputs) != 2: continue
            
            cnode = data.nodes[logicop.inputs[1]]
            if cnode.kind != "const": continue
            val = int(cnode.name, 16)
            
            if opc in ("INT_OR", "INT_XOR"):
                if i == 0:
                    hi = logicop.inputs[0]
                    lo = op.inputs[1]
                    lo_sz = data.nodes[lo].size_bits
                    val <<= 8 * lo_sz
                else:
                    hi = op.inputs[0]
                    lo = logicop.inputs[0]
            else: # INT_AND
                if i == 0:
                    hi = logicop.inputs[0]
                    lo = op.inputs[1]
                    lo_sz = data.nodes[lo].size_bits
                    val <<= 8 * lo_sz
                    val |= (1 << (8 * lo_sz)) - 1
                else:
                    hi = op.inputs[0]
                    lo = logicop.inputs[0]
                    hi_sz = data.nodes[hi].size_bits
                    val |= ((1 << (8 * hi_sz)) - 1) << (8 * data.nodes[lo].size_bits)
                    
            newconcat = PcodeOp(op.addr, "PIECE", len(data.nodes), (hi, lo))
            data.nodes.append(PcodeVarnode(newconcat.output, None, out_sz, "unique"))
            data.ops.append(newconcat)
            
            data.op_set_opcode(op, opc)
            data.op_set_input(op, newconcat.output, 0)
            
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, hex(val), out_sz, "const"))
            data.op_set_input(op, cnew, 1)
            return 1
            
        return 0

class RuleConcatZero(Rule):
    """PIECE(V, #0) => zext(V) << sa"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "concatzero")

    def get_op_list(self) -> list[str]:
        return ["PIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        if int(cnode.name, 16) != 0: return 0
        
        sa = 8 * cnode.size_bits
        highvn = op.inputs[0]
        if data.nodes[highvn].kind == "const" and int(data.nodes[highvn].name, 16) == 0:
            return 0
            
        out_sz = data.nodes[op.output].size_bits if op.output is not None else 0
        newop = PcodeOp(op.addr, "INT_ZEXT", len(data.nodes), (highvn,))
        data.nodes.append(PcodeVarnode(newop.output, None, out_sz, "unique"))
        data.ops.append(newop)
        
        data.op_set_opcode(op, "INT_LEFT")
        data.op_set_input(op, newop.output, 0)
        
        cnew = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew, hex(sa), cnode.size_bits, "const"))
        data.op_set_input(op, cnew, 1)
        return 1

class RuleConcatZext(Rule):
    """Commute PIECE with INT_ZEXT: concat(zext(V),W) => zext(concat(V,W))"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "concatzext")

    def get_op_list(self) -> list[str]:
        return ["PIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        hi = op.inputs[0]
        
        zextop = None
        for cand in data.ops:
            if cand.output == hi:
                zextop = cand
                break
        if not zextop or zextop.opcode != "INT_ZEXT": return 0
        if len(zextop.inputs) != 1: return 0
        
        hi_in = zextop.inputs[0]
        lo = op.inputs[1]
        
        hi_in_sz = data.nodes[hi_in].size_bits
        lo_sz = data.nodes[lo].size_bits
        
        newconcat = PcodeOp(op.addr, "PIECE", len(data.nodes), (hi_in, lo))
        data.nodes.append(PcodeVarnode(newconcat.output, None, hi_in_sz + lo_sz, "unique"))
        data.ops.append(newconcat)
        
        data.op_remove_input(op, 1)
        data.op_set_input(op, newconcat.output, 0)
        data.op_set_opcode(op, "INT_ZEXT")
        return 1


class RuleMultNegOne(Rule):
    """Cleanup: Convert INT_2COMP from INT_MULT: V * -1 => -V"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "multnegone")

    def get_op_list(self) -> list[str]:
        return ["INT_MULT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        mask = (1 << cnode.size_bits) - 1
        if int(cnode.name, 16) != mask: return 0
        
        data.op_set_opcode(op, "INT_2COMP")
        data.op_remove_input(op, 1)
        return 1

class Rule2Comp2Sub(Rule):
    """Cleanup: Convert INT_ADD back to INT_SUB: V + -W ==> V - W"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "2comp2sub")

    def get_op_list(self) -> list[str]:
        return ["INT_2COMP"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 1: return 0
        
        outvn = op.output
        if outvn is None: return 0
        
        addop = None
        count = 0
        for cand in data.ops:
            if outvn in cand.inputs:
                addop = cand
                count += 1
                
        if count != 1 or not addop or addop.opcode != "INT_ADD": return 0
        if len(addop.inputs) != 2: return 0
        
        if addop.inputs[0] == outvn:
            data.op_set_input(addop, addop.inputs[1], 0)
            
        data.op_set_input(addop, op.inputs[0], 1)
        data.op_set_opcode(addop, "INT_SUB")
        data.op_destroy(op)
        return 1


class RuleBooleanUndistribute(Rule):
    """Undo distributed BOOL_AND through INT_NOTEQUAL"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "booleanundistribute")

    def get_op_list(self) -> list[str]:
        return ["INT_EQUAL", "INT_NOTEQUAL"]

    def _is_match(self, leftVn: int, rightVn: int, data: PcodeLikeGraph) -> tuple[bool, bool]:
        if leftVn == rightVn:
            return True, False
            
        def is_negate_of(v1, v2):
            op = next((cand for cand in data.ops if cand.output == v1), None)
            if op and op.opcode == "BOOL_NEGATE" and len(op.inputs) == 1 and op.inputs[0] == v2:
                return True
            return False
            
        if is_negate_of(leftVn, rightVn) or is_negate_of(rightVn, leftVn):
            return True, True
            
        return False, False

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        vn0, vn1 = op.inputs[0], op.inputs[1]
        
        op0 = next((cand for cand in data.ops if cand.output == vn0), None)
        if not op0 or op0.opcode not in ("BOOL_AND", "BOOL_OR"): return 0
        
        op1 = next((cand for cand in data.ops if cand.output == vn1), None)
        if not op1 or op1.opcode not in ("BOOL_AND", "BOOL_OR"): return 0
        
        if len(op0.inputs) != 2 or len(op1.inputs) != 2: return 0
        ins = [op0.inputs[0], op0.inputs[1], op1.inputs[0], op1.inputs[1]]
        
        isflipped = [False, False, False, False]
        centralEqual = (op.opcode == "INT_EQUAL")
        
        if op0.opcode == "BOOL_OR":
            isflipped[0] = not isflipped[0]
            isflipped[1] = not isflipped[1]
            centralEqual = not centralEqual
            
        if op1.opcode == "BOOL_OR":
            isflipped[2] = not isflipped[2]
            isflipped[3] = not isflipped[3]
            centralEqual = not centralEqual
            
        leftSlot = -1
        rightSlot = -1
        
        matched, flip = self._is_match(ins[0], ins[2], data)
        if matched:
            leftSlot, rightSlot = 0, 2
            if flip: isflipped[2] = not isflipped[2]
        else:
            matched, flip = self._is_match(ins[0], ins[3], data)
            if matched:
                leftSlot, rightSlot = 0, 3
                if flip: isflipped[3] = not isflipped[3]
            else:
                matched, flip = self._is_match(ins[1], ins[2], data)
                if matched:
                    leftSlot, rightSlot = 1, 2
                    if flip: isflipped[2] = not isflipped[2]
                else:
                    matched, flip = self._is_match(ins[1], ins[3], data)
                    if matched:
                        leftSlot, rightSlot = 1, 3
                        if flip: isflipped[3] = not isflipped[3]
                        
        if leftSlot == -1: return 0
        if isflipped[leftSlot] != isflipped[rightSlot]: return 0
        
        if centralEqual:
            combineOpc = "BOOL_OR"
            isflipped[leftSlot] = not isflipped[leftSlot]
        else:
            combineOpc = "BOOL_AND"
            
        finalA = ins[leftSlot]
        if isflipped[leftSlot]:
            neg_op = PcodeOp(op.addr, "BOOL_NEGATE", len(data.nodes), (finalA,))
            data.nodes.append(PcodeVarnode(neg_op.output, None, 1, "unique"))
            data.ops.append(neg_op)
            finalA = neg_op.output
            
        if isflipped[1 - leftSlot]: centralEqual = not centralEqual
        if isflipped[5 - rightSlot]: centralEqual = not centralEqual
        
        finalB = ins[1 - leftSlot]
        finalC = ins[5 - rightSlot]
        
        eqOp = PcodeOp(op.addr, "INT_EQUAL" if centralEqual else "INT_NOTEQUAL", len(data.nodes), (finalB, finalC))
        data.nodes.append(PcodeVarnode(eqOp.output, None, 1, "unique"))
        data.ops.append(eqOp)
        
        data.op_set_opcode(op, combineOpc)
        data.op_set_input(op, eqOp.output, 1)
        data.op_set_input(op, finalA, 0)
        return 1

class RuleBooleanDedup(Rule):
    """Remove duplicate clauses in boolean expressions"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "booleandedup")

    def get_op_list(self) -> list[str]:
        return ["BOOL_AND", "BOOL_OR"]

    def _is_match(self, leftVn: int, rightVn: int, data: PcodeLikeGraph) -> tuple[bool, bool]:
        if leftVn == rightVn:
            return True, False
            
        def is_negate_of(v1, v2):
            op = next((cand for cand in data.ops if cand.output == v1), None)
            if op and op.opcode == "BOOL_NEGATE" and len(op.inputs) == 1 and op.inputs[0] == v2:
                return True
            return False
            
        if is_negate_of(leftVn, rightVn) or is_negate_of(rightVn, leftVn):
            return True, True
            
        return False, False

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        vn0, vn1 = op.inputs[0], op.inputs[1]
        
        op0 = next((cand for cand in data.ops if cand.output == vn0), None)
        if not op0 or op0.opcode not in ("BOOL_AND", "BOOL_OR"): return 0
        
        op1 = next((cand for cand in data.ops if cand.output == vn1), None)
        if not op1 or op1.opcode not in ("BOOL_AND", "BOOL_OR"): return 0
        
        if len(op0.inputs) != 2 or len(op1.inputs) != 2: return 0
        ins = [op0.inputs[0], op0.inputs[1], op1.inputs[0], op1.inputs[1]]
        
        matched, flip = self._is_match(ins[0], ins[2], data)
        if matched:
            leftA, rightA, leftO, rightO = ins[0], ins[2], ins[1], ins[3]
        else:
            matched, flip = self._is_match(ins[0], ins[3], data)
            if matched:
                leftA, rightA, leftO, rightO = ins[0], ins[3], ins[1], ins[2]
            else:
                matched, flip = self._is_match(ins[1], ins[2], data)
                if matched:
                    leftA, rightA, leftO, rightO = ins[1], ins[2], ins[0], ins[3]
                else:
                    matched, flip = self._is_match(ins[1], ins[3], data)
                    if matched:
                        leftA, rightA, leftO, rightO = ins[1], ins[3], ins[0], ins[2]
                    else:
                        return 0
                        
        centralOpc = op.opcode
        opc0, opc1 = op0.opcode, op1.opcode
        
        if flip:
            if centralOpc == "BOOL_AND" and opc0 == "BOOL_AND" and opc1 == "BOOL_AND":
                data.op_set_opcode(op, "COPY")
                data.op_remove_input(op, 1)
                cnew = len(data.nodes)
                data.nodes.append(PcodeVarnode(cnew, hex(0), 1, "const"))
                data.op_set_input(op, cnew, 0)
                return 1
            if centralOpc == "BOOL_OR" and opc0 == "BOOL_OR" and opc1 == "BOOL_OR":
                data.op_set_opcode(op, "COPY")
                data.op_remove_input(op, 1)
                cnew = len(data.nodes)
                data.nodes.append(PcodeVarnode(cnew, hex(1), 1, "const"))
                data.op_set_input(op, cnew, 0)
                return 1
            if centralOpc == "BOOL_OR" and opc0 != opc1:
                finalA = leftA if opc0 == "BOOL_OR" else rightA
                finalOpc = "BOOL_OR"
                bcOpc = "BOOL_OR"
            else:
                return 0
        else:
            if centralOpc == opc0 and centralOpc == opc1:
                finalA = leftA
                finalOpc = centralOpc
                bcOpc = centralOpc
            elif centralOpc != opc0 and centralOpc != opc1:
                finalA = leftA
                finalOpc = centralOpc
                bcOpc = centralOpc
            elif centralOpc == "BOOL_OR" and opc0 != opc1:
                finalA = leftA
                finalOpc = "BOOL_OR"
                bcOpc = "BOOL_AND"
            else:
                return 0
                
        newop = PcodeOp(op.addr, bcOpc, len(data.nodes), (leftO, rightO))
        data.nodes.append(PcodeVarnode(newop.output, None, 1, "unique"))
        data.ops.append(newop)
        
        data.op_set_opcode(op, finalOpc)
        data.op_set_input(op, finalA, 0)
        data.op_set_input(op, newop.output, 1)
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
