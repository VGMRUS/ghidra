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

class RuleIntLessEqual(Rule):
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


class RuleEqual2Zero(Rule):
    """Simplify INT_EQUAL/INT_NOTEQUAL with zero: A + -B == 0 => A == B"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "equal2zero")

    def get_op_list(self) -> list[str]:
        return ["INT_EQUAL", "INT_NOTEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        
        vn, addvn = op.inputs[0], op.inputs[1]
        cnode = data.nodes[vn]
        if cnode.kind == "const" and int(cnode.name, 16) == 0:
            pass
        else:
            addvn, vn = vn, addvn
            cnode = data.nodes[vn]
            if cnode.kind != "const" or int(cnode.name, 16) != 0: return 0
            
        addop = None
        for cand in data.ops:
            if cand.output == addvn:
                addop = cand
                break
                
        if not addop or addop.opcode != "INT_ADD" or len(addop.inputs) != 2: return 0
        
        # Check all uses of addvn are boolean (which in minighidra just means comparisons)
        for cand in data.ops:
            if addvn in cand.inputs:
                if cand.opcode not in ("INT_EQUAL", "INT_NOTEQUAL", "INT_LESS", "INT_LESSEQUAL", "INT_SLESS", "INT_SLESSEQUAL"):
                    return 0
                    
        in0, in1 = addop.inputs[0], addop.inputs[1]
        in1node = data.nodes[in1]
        
        if in1node.kind == "const":
            val = int(in1node.name, 16)
            unneg_val = (-val) & ((1 << in1node.size_bits) - 1)
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, hex(unneg_val), in1node.size_bits, "const"))
            
            data.op_set_input(op, in0, 0)
            data.op_set_input(op, cnew, 1)
            return 1
        else:
            def is_negate_mult(v):
                defop = next((cand for cand in data.ops if cand.output == v), None)
                if defop and defop.opcode == "INT_MULT" and len(defop.inputs) == 2:
                    cn = data.nodes[defop.inputs[1]]
                    if cn.kind == "const" and int(cn.name, 16) == ((1 << cn.size_bits) - 1):
                        return defop.inputs[0]
                return None
                
            unneg0 = is_negate_mult(in0)
            unneg1 = is_negate_mult(in1)
            
            if unneg0 is not None:
                data.op_set_input(op, in1, 0)
                data.op_set_input(op, unneg0, 1)
                return 1
            elif unneg1 is not None:
                data.op_set_input(op, in0, 0)
                data.op_set_input(op, unneg1, 1)
                return 1
                
        return 0


class RuleSignShift(Rule):
    """Convert V >> 31 to (V s>> 31) * -1 when used in ADD, MULT, EQUAL, NOTEQUAL"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "signshift")

    def get_op_list(self) -> list[str]:
        return ["INT_RIGHT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        inVn = op.inputs[0]
        sz = data.nodes[inVn].size_bits
        if int(cnode.name, 16) != 8 * sz - 1: return 0
        
        doConversion = False
        outVn = op.output
        if outVn is None: return 0
        
        for cand in data.ops:
            if outVn in cand.inputs:
                if cand.opcode in ("INT_EQUAL", "INT_NOTEQUAL"):
                    if data.nodes[cand.inputs[1]].kind == "const":
                        doConversion = True
                        break
                elif cand.opcode in ("INT_ADD", "INT_MULT"):
                    doConversion = True
                    break
                    
        if not doConversion: return 0
        
        shiftOp = PcodeOp(op.addr, "INT_SRIGHT", len(data.nodes), (inVn, op.inputs[1]))
        data.nodes.append(PcodeVarnode(shiftOp.output, None, sz, "unique"))
        data.ops.append(shiftOp)
        
        data.op_set_input(op, shiftOp.output, 0)
        cnew = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew, hex((1 << sz) - 1), sz, "const"))
        data.op_set_input(op, cnew, 1)
        data.op_set_opcode(op, "INT_MULT")
        return 1

class RuleTestSign(Rule):
    """Convert sign-bit test to signed comparison: (V s>> 31) != 0 => V s< 0"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "testsign")

    def get_op_list(self) -> list[str]:
        return ["INT_SRIGHT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        inVn = op.inputs[0]
        sz = data.nodes[inVn].size_bits
        if int(cnode.name, 16) != 8 * sz - 1: return 0
        
        outVn = op.output
        if outVn is None: return 0
        
        compareOps = []
        for cand in data.ops:
            if cand.opcode in ("INT_EQUAL", "INT_NOTEQUAL") and cand.inputs[0] == outVn:
                if data.nodes[cand.inputs[1]].kind == "const":
                    compareOps.append(cand)
                    
        if not compareOps: return 0
        
        resultCode = 0
        for compareOp in compareOps:
            compSize = data.nodes[compareOp.inputs[0]].size_bits
            offset = int(data.nodes[compareOp.inputs[1]].name, 16)
            
            sgn = 0
            if offset == 0:
                sgn = 1
            elif offset == ((1 << compSize) - 1):
                sgn = -1
            else:
                continue
                
            if compareOp.opcode == "INT_NOTEQUAL":
                sgn = -sgn
                
            zeroVn = len(data.nodes)
            data.nodes.append(PcodeVarnode(zeroVn, hex(0), sz, "const"))
            
            if sgn == 1:
                data.op_set_input(compareOp, inVn, 1)
                data.op_set_input(compareOp, zeroVn, 0)
                data.op_set_opcode(compareOp, "INT_SLESSEQUAL")
            else:
                data.op_set_input(compareOp, inVn, 0)
                data.op_set_input(compareOp, zeroVn, 1)
                data.op_set_opcode(compareOp, "INT_SLESS")
                
            resultCode = 1
            
        return resultCode


class RuleSubZext(Rule):
    """Simplify INT_ZEXT applied to SUBPIECE expressions"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "subzext")

    def get_op_list(self) -> list[str]:
        return ["INT_ZEXT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 1: return 0
        subvn = op.inputs[0]
        subop = None
        for cand in data.ops:
            if cand.output == subvn:
                subop = cand
                break
        if not subop: return 0
        
        if subop.opcode == "SUBPIECE" and len(subop.inputs) == 2:
            basevn = subop.inputs[0]
            if data.nodes[basevn].size_bits != data.nodes[op.output].size_bits: return 0
            
            cnode = data.nodes[subop.inputs[1]]
            if cnode.kind != "const": return 0
            
            offset = int(cnode.name, 16)
            if offset != 0:
                count = sum(1 for cand in data.ops if subvn in cand.inputs)
                if count != 1: return 0
                
                newvn = len(data.nodes)
                data.nodes.append(PcodeVarnode(newvn, None, data.nodes[basevn].size_bits, "unique"))
                
                data.op_set_input(op, newvn, 0)
                data.op_set_opcode(subop, "INT_RIGHT")
                cnew = len(data.nodes)
                data.nodes.append(PcodeVarnode(cnew, hex(offset * 8), cnode.size_bits, "const"))
                data.op_set_input(subop, cnew, 1)
                subop.output = newvn
            else:
                data.op_set_input(op, basevn, 0)
                
            val = (1 << data.nodes[subvn].size_bits) - 1
            constvn = len(data.nodes)
            data.nodes.append(PcodeVarnode(constvn, hex(val), data.nodes[basevn].size_bits, "const"))
            
            data.op_set_opcode(op, "INT_AND")
            op.inputs = (op.inputs[0], constvn)
            return 1
            
        elif subop.opcode == "INT_RIGHT" and len(subop.inputs) == 2:
            shiftop = subop
            scnode = data.nodes[shiftop.inputs[1]]
            if scnode.kind != "const": return 0
            
            midvn = shiftop.inputs[0]
            subop2 = None
            for cand in data.ops:
                if cand.output == midvn:
                    subop2 = cand
                    break
            if not subop2 or subop2.opcode != "SUBPIECE" or len(subop2.inputs) != 2: return 0
            
            basevn = subop2.inputs[0]
            if data.nodes[basevn].size_bits != data.nodes[op.output].size_bits: return 0
            
            count_mid = sum(1 for cand in data.ops if midvn in cand.inputs)
            if count_mid != 1: return 0
            
            count_sub = sum(1 for cand in data.ops if subvn in cand.inputs)
            if count_sub != 1: return 0
            
            subcnode = data.nodes[subop2.inputs[1]]
            if subcnode.kind != "const": return 0
            
            val = (1 << data.nodes[midvn].size_bits) - 1
            sa = int(scnode.name, 16)
            val >>= sa
            sa += int(subcnode.name, 16) * 8
            
            newvn = len(data.nodes)
            data.nodes.append(PcodeVarnode(newvn, None, data.nodes[basevn].size_bits, "unique"))
            
            data.op_set_input(op, newvn, 0)
            data.op_set_input(shiftop, basevn, 0)
            
            cnew2 = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew2, hex(sa), scnode.size_bits, "const"))
            data.op_set_input(shiftop, cnew2, 1)
            shiftop.output = newvn
            
            constvn = len(data.nodes)
            data.nodes.append(PcodeVarnode(constvn, hex(val), data.nodes[basevn].size_bits, "const"))
            
            data.op_set_opcode(op, "INT_AND")
            op.inputs = (op.inputs[0], constvn)
            return 1
            
        return 0


class RuleDumptyHump(Rule):
    """Simplify join and break apart: sub( concat(V,W), c) => sub(W,c)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "dumptyhump")

    def get_op_list(self) -> list[str]:
        return ["SUBPIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        base = op.inputs[0]
        pieceop = None
        for cand in data.ops:
            if cand.output == base:
                pieceop = cand
                break
        if not pieceop or pieceop.opcode != "PIECE" or len(pieceop.inputs) != 2: return 0
        
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        offset = int(cnode.name, 16)
        outsize = data.nodes[op.output].size_bits if op.output is not None else 0
        
        vn1 = pieceop.inputs[0]
        vn2 = pieceop.inputs[1]
        sz2 = data.nodes[vn2].size_bits
        
        offset_bits = offset * 8
        if offset_bits < sz2:
            if offset_bits + outsize > sz2: return 0
            vn = vn2
            new_offset = offset
        else:
            vn = vn1
            new_offset = offset - (sz2 // 8)
            
        vnode = data.nodes[vn]
        sz = vnode.size_bits
        
        if new_offset == 0 and outsize == sz:
            data.op_set_opcode(op, "COPY")
            data.op_remove_input(op, 1)
            data.op_set_input(op, vn, 0)
        else:
            data.op_set_input(op, vn, 0)
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, hex(new_offset), cnode.size_bits, "const"))
            data.op_set_input(op, cnew, 1)
        return 1

class RuleHumptyOr(Rule):
    """Simplify masked pieces INT_ORed together: (V & ff00) | (V & 00ff) => V"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "humptyor")

    def get_op_list(self) -> list[str]:
        return ["INT_OR"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        vn1, vn2 = op.inputs[0], op.inputs[1]
        
        and1 = next((cand for cand in data.ops if cand.output == vn1), None)
        if not and1 or and1.opcode != "INT_AND" or len(and1.inputs) != 2: return 0
        
        and2 = next((cand for cand in data.ops if cand.output == vn2), None)
        if not and2 or and2.opcode != "INT_AND" or len(and2.inputs) != 2: return 0
        
        a, b = and1.inputs[0], and1.inputs[1]
        c, d = and2.inputs[0], and2.inputs[1]
        
        if a == c:
            c = d
        elif a == d:
            pass
        elif b == c:
            b, a, c = a, c, d
        elif b == d:
            b, a = a, d
        else:
            return 0
            
        bnode, cnode = data.nodes[b], data.nodes[c]
        if bnode.kind == "const" and cnode.kind == "const":
            totalbits = int(bnode.name, 16) | int(cnode.name, 16)
            asz = data.nodes[a].size_bits
            if totalbits == ((1 << asz) - 1):
                data.op_set_opcode(op, "COPY")
                data.op_remove_input(op, 1)
                data.op_set_input(op, a, 0)
            else:
                data.op_set_opcode(op, "INT_AND")
                data.op_set_input(op, a, 0)
                cnew = len(data.nodes)
                data.nodes.append(PcodeVarnode(cnew, hex(totalbits), asz, "const"))
                data.op_set_input(op, cnew, 1)
        else:
            amask = data.get_nzmask(a)
            bmask = data.get_nzmask(b)
            cmask = data.get_nzmask(c)
            if (bmask & amask) == 0 or (cmask & amask) == 0: return 0
            
            newOrOp = PcodeOp(op.addr, "INT_OR", len(data.nodes), (b, c))
            data.nodes.append(PcodeVarnode(newOrOp.output, None, data.nodes[a].size_bits, "unique"))
            data.ops.append(newOrOp)
            
            data.op_set_input(op, a, 0)
            data.op_set_input(op, newOrOp.output, 1)
            data.op_set_opcode(op, "INT_AND")
        return 1


class RuleHumptyDumpty(Rule):
    """Simplify concat of adjacent subpieces: concat(sub(V, c+s), sub(V, c)) => sub(V, c) or V"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "humptydumpty")

    def get_op_list(self) -> list[str]:
        return ["PIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        vn1 = op.inputs[0]
        sub1 = next((cand for cand in data.ops if cand.output == vn1), None)
        if not sub1 or sub1.opcode != "SUBPIECE" or len(sub1.inputs) != 2: return 0
        
        vn2 = op.inputs[1]
        sub2 = next((cand for cand in data.ops if cand.output == vn2), None)
        if not sub2 or sub2.opcode != "SUBPIECE" or len(sub2.inputs) != 2: return 0
        
        root = sub1.inputs[0]
        if root != sub2.inputs[0]: return 0
        
        cnode1 = data.nodes[sub1.inputs[1]]
        cnode2 = data.nodes[sub2.inputs[1]]
        if cnode1.kind != "const" or cnode2.kind != "const": return 0
        
        pos1 = int(cnode1.name, 16)
        pos2 = int(cnode2.name, 16)
        
        size1 = data.nodes[vn1].size_bits
        size2 = data.nodes[vn2].size_bits
        
        # In PIECE, vn1 is the high part, vn2 is the low part.
        # Wait, the C++ code says pos1 == pos2 + size2 (bytes)
        # However, our sizes are in bits. We must divide by 8 for bytes, or adjust if sizes are bits.
        # Ghidra works in bytes for sizes, our size_bits are bits. So pos1 == pos2 + (size2 // 8).
        # Actually in minighidra SUBPIECE offset is in bytes (same as Ghidra).
        if pos1 != pos2 + (size2 // 8): return 0
        
        if pos2 == 0 and (size1 + size2 == data.nodes[root].size_bits):
            data.op_remove_input(op, 1)
            data.op_set_input(op, root, 0)
            data.op_set_opcode(op, "COPY")
        else:
            data.op_set_input(op, root, 0)
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, hex(pos2), cnode2.size_bits, "const"))
            data.op_set_input(op, cnew, 1)
            data.op_set_opcode(op, "SUBPIECE")
        return 1


class RuleBooleanNegate(Rule):
    """Simplify comparison of boolean value with 0 or 1: V != 0 => V"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "booleannegate")

    def get_op_list(self) -> list[str]:
        return ["INT_EQUAL", "INT_NOTEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        val = int(cnode.name, 16)
        if val not in (0, 1): return 0
        
        subbool = op.inputs[0]
        if not data.is_boolean_value(subbool): return 0
        
        negate = (op.opcode == "INT_NOTEQUAL")
        if val == 0:
            negate = not negate
            
        data.op_remove_input(op, 1)
        data.op_set_input(op, subbool, 0)
        
        if negate:
            data.op_set_opcode(op, "BOOL_NEGATE")
        else:
            data.op_set_opcode(op, "COPY")
        return 1


class RuleXorSwap(Rule):
    """Simplify limited chains of XOR operations: V = (a ^ b) ^ a => V = b"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "xorswap")

    def get_op_list(self) -> list[str]:
        return ["INT_XOR"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        for i in range(2):
            vn = op.inputs[i]
            op2 = next((cand for cand in data.ops if cand.output == vn), None)
            if not op2 or op2.opcode != "INT_XOR" or len(op2.inputs) != 2: continue
            
            othervn = op.inputs[1 - i]
            vn0 = op2.inputs[0]
            vn1 = op2.inputs[1]
            
            if othervn == vn0:
                data.op_remove_input(op, 1)
                data.op_set_opcode(op, "COPY")
                data.op_set_input(op, vn1, 0)
                return 1
            elif othervn == vn1:
                data.op_remove_input(op, 1)
                data.op_set_opcode(op, "COPY")
                data.op_set_input(op, vn0, 0)
                return 1
        return 0


class RuleOrMask(Rule):
    """Collapse unnecessary INT_OR: V | 0xffffffff => 0xffffffff"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "ormask")

    def get_op_list(self) -> list[str]:
        return ["INT_OR"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        val = int(cnode.name, 16)
        outsz = data.nodes[op.output].size_bits if op.output is not None else data.nodes[op.inputs[0]].size_bits
        mask = (1 << outsz) - 1
        
        if (val & mask) != mask: return 0
        
        data.op_set_opcode(op, "COPY")
        data.op_set_input(op, op.inputs[1], 0)
        data.op_remove_input(op, 1)
        return 1


class RuleTrivialShift(Rule):
    """Simplify shifts by 0 (to COPY) or by >= size (to 0)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "trivialshift")

    def get_op_list(self) -> list[str]:
        return ["INT_LEFT", "INT_RIGHT", "INT_SRIGHT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        val = int(cnode.name, 16)
        sz = data.nodes[op.inputs[0]].size_bits
        
        if val != 0:
            if val < sz: return 0
            if op.opcode == "INT_SRIGHT": return 0
            
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, "0x0", sz, "const"))
            data.op_set_input(op, cnew, 0)
            
        data.op_remove_input(op, 1)
        data.op_set_opcode(op, "COPY")
        return 1


class RuleLessNotEqual(Rule):
    """Convert (a <= b) & (a != b) to a < b"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "lessnotequal")

    def get_op_list(self) -> list[str]:
        return ["INT_AND"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        vn1, vn2 = op.inputs[0], op.inputs[1]
        op_less = next((cand for cand in data.ops if cand.output == vn1), None)
        op_equal = next((cand for cand in data.ops if cand.output == vn2), None)
        
        if not op_less or not op_equal: return 0
        
        opc = op_less.opcode
        if opc not in ("INT_LESSEQUAL", "INT_SLESSEQUAL"):
            op_less, op_equal = op_equal, op_less
            opc = op_less.opcode
            if opc not in ("INT_LESSEQUAL", "INT_SLESSEQUAL"): return 0
            
        if op_equal.opcode != "INT_NOTEQUAL": return 0
        
        a, b = op_less.inputs[0], op_less.inputs[1]
        c, d = op_equal.inputs[0], op_equal.inputs[1]
        
        if (a == c and b == d) or (a == d and b == c):
            data.op_set_input(op, a, 0)
            data.op_set_input(op, b, 1)
            data.op_set_opcode(op, "INT_SLESS" if opc == "INT_SLESSEQUAL" else "INT_LESS")
            return 1
        return 0


class RuleNegateNegate(Rule):
    """Simplify double negate: ~ (~ V) => V"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "negatenegate")

    def get_op_list(self) -> list[str]:
        return ["INT_NEGATE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 1: return 0
        vn1 = op.inputs[0]
        neg2 = next((cand for cand in data.ops if cand.output == vn1), None)
        if not neg2 or neg2.opcode != "INT_NEGATE" or len(neg2.inputs) != 1: return 0
        
        vn2 = neg2.inputs[0]
        data.op_set_input(op, vn2, 0)
        data.op_set_opcode(op, "COPY")
        return 1


class RuleOrCompare(Rule):
    """Simplify (V | W) == 0 => (V == 0) && (W == 0)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "orcompare")

    def get_op_list(self) -> list[str]:
        return ["INT_OR"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        outvn = op.output
        if outvn is None: return 0
        
        has_compares = False
        descendants = [cand for cand in data.ops if outvn in cand.inputs]
        for comp_op in descendants:
            if comp_op.opcode not in ("INT_EQUAL", "INT_NOTEQUAL"): return 0
            if len(comp_op.inputs) != 2: return 0
            
            cnode = data.nodes[comp_op.inputs[1]]
            if cnode.kind != "const" or int(cnode.name, 16) != 0: return 0
            has_compares = True
            
        if not has_compares: return 0
        
        v, w = op.inputs[0], op.inputs[1]
        
        for comp_op in descendants:
            opc = comp_op.opcode
            
            zero_v = len(data.nodes)
            data.nodes.append(PcodeVarnode(zero_v, "0x0", data.nodes[v].size_bits, "const"))
            eq_v = PcodeOp(comp_op.addr, opc, len(data.nodes), (v, zero_v))
            data.nodes.append(PcodeVarnode(eq_v.output, None, 1, "unique"))
            data.ops.append(eq_v)
            
            zero_w = len(data.nodes)
            data.nodes.append(PcodeVarnode(zero_w, "0x0", data.nodes[w].size_bits, "const"))
            eq_w = PcodeOp(comp_op.addr, opc, len(data.nodes), (w, zero_w))
            data.nodes.append(PcodeVarnode(eq_w.output, None, 1, "unique"))
            data.ops.append(eq_w)
            
            data.op_set_input(comp_op, eq_v.output, 0)
            data.op_set_input(comp_op, eq_w.output, 1)
            data.op_set_opcode(comp_op, "BOOL_AND" if opc == "INT_EQUAL" else "BOOL_OR")
            
        return 1


class RuleShiftSub(Rule):
    """Simplify SUBPIECE of INT_LEFT: sub(V << (n*8), c) => sub(V, c - n)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "shiftsub")

    def get_op_list(self) -> list[str]:
        return ["SUBPIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        vn = op.inputs[0]
        shiftop = next((cand for cand in data.ops if cand.output == vn), None)
        if not shiftop or shiftop.opcode != "INT_LEFT" or len(shiftop.inputs) != 2: return 0
        
        sa = data.nodes[shiftop.inputs[1]]
        if sa.kind != "const": return 0
        
        n = int(sa.name, 16)
        if (n & 7) != 0: return 0
        
        c = int(data.nodes[op.inputs[1]].name, 16)
        shift_in = shiftop.inputs[0]
        
        insize = data.nodes[shift_in].size_bits
        outsize = data.nodes[op.output].size_bits if op.output is not None else 0
        
        c -= (n // 8)
        if c < 0 or c * 8 + outsize > insize: return 0
        
        data.op_set_input(op, shift_in, 0)
        
        cnew = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew, hex(c), data.nodes[op.inputs[1]].size_bits, "const"))
        data.op_set_input(op, cnew, 1)
        return 1


class RuleSubRight(Rule):
    """Convert high SUBPIECE to INT_RIGHT + low SUBPIECE"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "subright")

    def get_op_list(self) -> list[str]:
        return ["SUBPIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        c = int(cnode.name, 16)
        if c == 0: return 0
        
        a = op.inputs[0]
        outvn = op.output
        if outvn is None: return 0
        
        opc = "INT_RIGHT"
        d = c * 8
        
        descendants = [cand for cand in data.ops if outvn in cand.inputs]
        lone = descendants[0] if len(descendants) == 1 else None
        
        if lone:
            opc2 = lone.opcode
            if opc2 in ("INT_RIGHT", "INT_SRIGHT") and len(lone.inputs) == 2:
                scnode = data.nodes[lone.inputs[1]]
                if scnode.kind == "const":
                    if data.nodes[outvn].size_bits + c * 8 == data.nodes[a].size_bits:
                        d += int(scnode.name, 16)
                        if d >= data.nodes[a].size_bits:
                            if opc2 == "INT_RIGHT": return 0
                            d = data.nodes[a].size_bits - 1
                        
                        op = lone
                        data.op_set_opcode(op, "SUBPIECE")
                        opc = opc2
        
        shiftop = PcodeOp(op.addr, opc, len(data.nodes), (a,))
        newout = len(data.nodes)
        data.nodes.append(PcodeVarnode(newout, None, data.nodes[a].size_bits, "unique"))
        shiftop.output = newout
        
        cnew = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew, hex(d), 4 * 8, "const"))
        shiftop.inputs = (a, cnew)
        data.ops.append(shiftop)
        
        data.op_set_input(op, newout, 0)
        cnew2 = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew2, "0x0", 4 * 8, "const"))
        data.op_set_input(op, cnew2, 1)
        return 1


class RuleAddUnsigned(Rule):
    """Convert INT_ADD of large unsigned constants to INT_SUB"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "addunsigned")

    def get_op_list(self) -> list[str]:
        return ["INT_ADD"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        constvn = data.nodes[op.inputs[1]]
        if constvn.kind != "const": return 0
        
        val = int(constvn.name, 16)
        sz = constvn.size_bits
        mask = (1 << sz) - 1
        
        sa = sz // 4 * 3  # Roughly 3/4 bitsize to match '1/4 less than full bitsize'
        # Ghidra uses: int4 sa = constvn->getSize() * 6; since size is in bytes. 1 byte = 8 bits, so 6 is 3/4 of 8.
        # sz is in bits, so we want sa = sz // 4 * 3
        
        quarter = (mask >> sa) << sa
        if (val & quarter) != quarter: return 0
        
        negated_val = (-val) & mask
        
        data.op_set_opcode(op, "INT_SUB")
        cnew = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew, hex(negated_val), sz, "const"))
        data.op_set_input(op, cnew, 1)
        return 1


class RuleSubExtComm(Rule):
    """Commute SUBPIECE and EXTENSION: sub(zext(V), c) => zext(sub(V, c))"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "subextcomm")

    def get_op_list(self) -> list[str]:
        return ["SUBPIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        base = op.inputs[0]
        extop = next((cand for cand in data.ops if cand.output == base), None)
        if not extop or extop.opcode not in ("INT_ZEXT", "INT_SEXT"): return 0
        
        invn = extop.inputs[0]
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        subcut = int(cnode.name, 16)
        outsz = data.nodes[op.output].size_bits if op.output is not None else 0
        insz = data.nodes[invn].size_bits
        
        if outsz + subcut * 8 <= insz:
            data.op_set_input(op, invn, 0)
            if insz == outsz:
                data.op_remove_input(op, 1)
                data.op_set_opcode(op, "COPY")
            return 1
            
        if subcut * 8 >= insz: return 0
        
        if subcut != 0:
            newop = PcodeOp(op.addr, "SUBPIECE", len(data.nodes), (invn, op.inputs[1]))
            newvn = len(data.nodes)
            data.nodes.append(PcodeVarnode(newvn, None, insz - subcut * 8, "unique"))
            newop.output = newvn
            data.ops.append(newop)
        else:
            newvn = invn
            
        data.op_remove_input(op, 1)
        data.op_set_opcode(op, extop.opcode)
        data.op_set_input(op, newvn, 0)
        return 1


class RuleSubNormal(Rule):
    """Pull-back SUBPIECE through INT_RIGHT/INT_SRIGHT"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "subnormal")

    def get_op_list(self) -> list[str]:
        return ["SUBPIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        shiftout = op.inputs[0]
        shiftop = next((cand for cand in data.ops if cand.output == shiftout), None)
        if not shiftop or shiftop.opcode not in ("INT_RIGHT", "INT_SRIGHT") or len(shiftop.inputs) != 2: return 0
        
        sa = data.nodes[shiftop.inputs[1]]
        if sa.kind != "const": return 0
        
        n = int(sa.name, 16)
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        c = int(cnode.name, 16)
        k = n // 8
        
        a = shiftop.inputs[0]
        insize = data.nodes[a].size_bits
        outsize = data.nodes[op.output].size_bits if op.output is not None else 0
        
        if n + 8 * c + outsize < insize and n != k * 8: return 0
        
        if (k + c) * 8 + outsize > insize:
            trunc_size = insize - c * 8 - k * 8
            if n == k * 8 and trunc_size > 0 and (trunc_size & (trunc_size - 1)) == 0:
                c += k
                newop = PcodeOp(op.addr, "SUBPIECE", len(data.nodes), (a, op.inputs[1]))
                newout = len(data.nodes)
                data.nodes.append(PcodeVarnode(newout, None, trunc_size, "unique"))
                newop.output = newout
                data.ops.append(newop)
                
                cnew = len(data.nodes)
                data.nodes.append(PcodeVarnode(cnew, hex(c), cnode.size_bits, "const"))
                data.op_set_input(newop, cnew, 1)
                
                data.op_set_input(op, newout, 0)
                data.op_remove_input(op, 1)
                data.op_set_opcode(op, "INT_SEXT" if shiftop.opcode == "INT_SRIGHT" else "INT_ZEXT")
                return 1
            else:
                k = (insize - outsize) // 8 - c
                
        c += k
        n -= k * 8
        if n == 0:
            data.op_set_input(op, a, 0)
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, hex(c), cnode.size_bits, "const"))
            data.op_set_input(op, cnew, 1)
            return 1
        elif n >= outsize:
            n = outsize
            if shiftop.opcode == "INT_SRIGHT": n -= 1
            
        newop = PcodeOp(op.addr, "SUBPIECE", len(data.nodes), (a, op.inputs[1]))
        newout = len(data.nodes)
        data.nodes.append(PcodeVarnode(newout, None, outsize, "unique"))
        newop.output = newout
        data.ops.append(newop)
        
        cnew = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew, hex(c), cnode.size_bits, "const"))
        data.op_set_input(newop, cnew, 1)
        
        data.op_set_input(op, newout, 0)
        cnew_sa = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew_sa, hex(n), sa.size_bits, "const"))
        data.op_set_input(op, cnew_sa, 1)
        data.op_set_opcode(op, shiftop.opcode)
        return 1


class RuleCarryElim(Rule):
    """Simplify INT_CARRY when adding a constant: V carry c => c <= V (where c is 2s-comp)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "carryelim")

    def get_op_list(self) -> list[str]:
        return ["INT_CARRY"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        off = int(cnode.name, 16)
        if off == 0:
            data.op_remove_input(op, 1)
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, "0x0", 1, "const"))
            data.op_set_input(op, cnew, 0)
            data.op_set_opcode(op, "COPY")
            return 1
            
        mask = (1 << cnode.size_bits) - 1
        off = (-off) & mask
        
        data.op_set_opcode(op, "INT_LESSEQUAL")
        data.op_set_input(op, op.inputs[0], 1)
        
        cnew = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew, hex(off), data.nodes[op.inputs[0]].size_bits, "const"))
        data.op_set_input(op, cnew, 0)
        return 1


class RuleScarry(Rule):
    """Simplify signed comparisons using INT_SCARRY and constants"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "scarry")

    def get_op_list(self) -> list[str]:
        return ["INT_SCARRY"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        avn = data.nodes[op.inputs[0]]
        bvn = data.nodes[op.inputs[1]]
        
        if (avn.kind == "const" and int(avn.name, 16) == 0) or (bvn.kind == "const" and int(bvn.name, 16) == 0):
            data.op_remove_input(op, 1)
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, "0x0", 1, "const"))
            data.op_set_input(op, cnew, 0)
            data.op_set_opcode(op, "COPY")
            return 1
            
        return 0


from minighidra.expression import AddExpression

class RuleSborrow(Rule):
    """Simplify signed comparisons using INT_SBORROW"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "sborrow")

    def get_op_list(self) -> list[str]:
        return ["INT_SBORROW"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        bvn = data.nodes[op.inputs[1]]
        if bvn.kind == "const" and int(bvn.name, 16) == 0:
            data.op_remove_input(op, 1)
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, "0x0", 1, "const"))
            data.op_set_input(op, cnew, 0)
            data.op_set_opcode(op, "COPY")
            return 1
            
        if not op.output: return 0
        
        avn_id = op.inputs[0]
        bvn_id = op.inputs[1]
        
        # Check descendents of sborrow output
        changed = False
        descends = []
        for other in data.ops:
            if op.output in other.inputs:
                descends.append(other)
                
        for compop in descends:
            if compop.opcode not in ("INT_EQUAL", "INT_NOTEQUAL"):
                continue
            
            slot = compop.inputs.index(op.output)
            cvn_id = compop.inputs[1 - slot]
            cvn = data.nodes[cvn_id]
            
            signop = None
            for dop in data.ops:
                if dop.output == cvn_id:
                    signop = dop
                    break
                    
            if not signop or signop.opcode != "INT_SLESS":
                continue
                
            in0_const = (data.nodes[signop.inputs[0]].kind == "const" and int(data.nodes[signop.inputs[0]].name, 16) == 0) if signop.inputs[0] >= 0 else False
            in1_const = (data.nodes[signop.inputs[1]].kind == "const" and int(data.nodes[signop.inputs[1]].name, 16) == 0) if signop.inputs[1] >= 0 else False
            
            if not in0_const:
                if not in1_const: continue
                zside = 1
            else:
                zside = 0
                
            xvn_id = signop.inputs[1 - zside]
            
            expr1 = AddExpression(data)
            expr1.gather_two_terms_subtract(avn_id, bvn_id)
            expr2 = AddExpression(data)
            expr2.gather_two_terms_root(xvn_id)
            
            if not expr1.is_equivalent(expr2):
                continue
                
            if compop.opcode == "INT_NOTEQUAL":
                data.op_set_opcode(compop, "INT_SLESS")
                data.op_set_input(compop, avn_id, 1 - zside)
                data.op_set_input(compop, bvn_id, zside)
            else:
                data.op_set_opcode(compop, "INT_SLESSEQUAL")
                data.op_set_input(compop, avn_id, zside)
                data.op_set_input(compop, bvn_id, 1 - zside)
            changed = True
            
        return 1 if changed else 0


class RuleDivChain(Rule):
    """Simplify chained divisions: (V / c) / d => V / (c * d)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "divchain")

    def get_op_list(self) -> list[str]:
        return ["INT_DIV", "INT_SDIV"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        opc2 = op.opcode
        cnode2 = data.nodes[op.inputs[1]]
        if cnode2.kind != "const": return 0
        
        vn = op.inputs[0]
        divOp = next((cand for cand in data.ops if cand.output == vn), None)
        if not divOp or len(divOp.inputs) != 2: return 0
        
        opc1 = divOp.opcode
        if opc1 != opc2 and (opc2 != "INT_DIV" or opc1 != "INT_RIGHT"): return 0
        
        cnode1 = data.nodes[divOp.inputs[1]]
        if cnode1.kind != "const": return 0
        
        count = sum(1 for cand in data.ops if vn in cand.inputs)
        if count != 1: return 0
        
        val2 = int(cnode2.name, 16)
        if opc1 == opc2:
            val1 = int(cnode1.name, 16)
        else:
            sa = int(cnode1.name, 16)
            val1 = 1 << sa
            
        baseVn = divOp.inputs[0]
        sz = data.nodes[vn].size_bits
        mask = (1 << sz) - 1
        
        resval = (val1 * val2) & mask
        if resval == 0: return 0
        
        def signbit_negative(v, size):
            return (v & (1 << (size - 1))) != 0
            
        def mostsigbit_set(v):
            if v == 0: return -1
            return v.bit_length() - 1
            
        abs1 = val1
        if signbit_negative(val1, sz): abs1 = (~val1 + 1) & mask
        abs2 = val2
        if signbit_negative(val2, sz): abs2 = (~val2 + 1) & mask
        
        bitcount = mostsigbit_set(abs1) + mostsigbit_set(abs2) + 2
        if opc2 == "INT_DIV" and bitcount > sz: return 0
        if opc2 == "INT_SDIV" and bitcount > sz - 2: return 0
        
        data.op_set_input(op, baseVn, 0)
        cnew = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew, hex(resval), sz, "const"))
        data.op_set_input(op, cnew, 1)
        return 1


class RuleSignForm(Rule):
    """Normalize sign extraction: sub(sext(V), c) => V s>> 31"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "signform")

    def get_op_list(self) -> list[str]:
        return ["SUBPIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        sextout = op.inputs[0]
        sextop = next((cand for cand in data.ops if cand.output == sextout), None)
        if not sextop or sextop.opcode != "INT_SEXT" or len(sextop.inputs) != 1: return 0
        
        a = sextop.inputs[0]
        if data.nodes[a].size_bits != data.nodes[op.output].size_bits: return 0
        
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        c = int(cnode.name, 16)
        
        if c * 8 < data.nodes[a].size_bits: return 0
        
        data.op_set_input(op, a, 0)
        n = data.nodes[a].size_bits - 1
        cnew = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew, hex(n), 4 * 8, "const"))
        data.op_set_input(op, cnew, 1)
        data.op_set_opcode(op, "INT_SRIGHT")
        return 1


class RuleSignNearMult(Rule):
    """Simplify optimization of division: (x + (x s>> 31) >> (size-n)) & mask => (x s/ 2^n) * 2^n"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "signnearmult")

    def get_op_list(self) -> list[str]:
        return ["INT_AND"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        
        vn = op.inputs[0]
        addop = next((cand for cand in data.ops if cand.output == vn), None)
        if not addop or addop.opcode != "INT_ADD" or len(addop.inputs) != 2: return 0
        
        shiftvn = None
        unshiftop = None
        idx = -1
        for i in range(2):
            shiftvn = addop.inputs[i]
            unshiftop = next((cand for cand in data.ops if cand.output == shiftvn), None)
            if unshiftop and unshiftop.opcode == "INT_RIGHT" and len(unshiftop.inputs) == 2:
                if data.nodes[unshiftop.inputs[1]].kind == "const":
                    idx = i
                    break
        if idx == -1: return 0
        
        x = addop.inputs[1 - idx]
        
        n = int(data.nodes[unshiftop.inputs[1]].name, 16)
        if n <= 0: return 0
        n = data.nodes[shiftvn].size_bits - n
        if n <= 0: return 0
        
        mask = (1 << data.nodes[shiftvn].size_bits) - 1
        mask = (mask << n) & mask
        if mask != int(cnode.name, 16): return 0
        
        sgnvn = unshiftop.inputs[0]
        sshiftop = next((cand for cand in data.ops if cand.output == sgnvn), None)
        if not sshiftop or sshiftop.opcode != "INT_SRIGHT" or len(sshiftop.inputs) != 2: return 0
        
        scnode = data.nodes[sshiftop.inputs[1]]
        if scnode.kind != "const": return 0
        if sshiftop.inputs[0] != x: return 0
        
        val = int(scnode.name, 16)
        if val != data.nodes[x].size_bits - 1: return 0
        
        pow2 = 1 << n
        newdiv = PcodeOp(op.addr, "INT_SDIV", len(data.nodes), (x, len(data.nodes)))
        c_pow = len(data.nodes)
        data.nodes.append(PcodeVarnode(c_pow, hex(pow2), data.nodes[x].size_bits, "const"))
        newdiv.inputs = (x, c_pow)
        
        divvn = len(data.nodes)
        data.nodes.append(PcodeVarnode(divvn, None, data.nodes[x].size_bits, "unique"))
        newdiv.output = divvn
        data.ops.append(newdiv)
        
        data.op_set_opcode(op, "INT_MULT")
        data.op_set_input(op, divvn, 0)
        c_pow2 = len(data.nodes)
        data.nodes.append(PcodeVarnode(c_pow2, hex(pow2), data.nodes[x].size_bits, "const"))
        data.op_set_input(op, c_pow2, 1)
        return 1


class RuleModOpt(Rule):
    """Simplify expressions that optimize INT_REM and INT_SREM"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "modopt")

    def get_op_list(self) -> list[str]:
        return ["INT_DIV", "INT_SDIV"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        x = op.inputs[0]
        div = op.inputs[1]
        outvn = op.output
        if outvn is None: return 0
        
        descendants1 = [cand for cand in data.ops if outvn in cand.inputs]
        for multop in descendants1:
            if multop.opcode != "INT_MULT" or len(multop.inputs) != 2: continue
            
            div2 = multop.inputs[1]
            if div2 == outvn: div2 = multop.inputs[0]
            
            cdiv2 = data.nodes[div2]
            cdiv = data.nodes[div]
            
            if cdiv2.kind == "const":
                if cdiv.kind != "const": continue
                mask = (1 << cdiv2.size_bits) - 1
                v2 = int(cdiv2.name, 16)
                v1 = int(cdiv.name, 16)
                if (((v2 ^ mask) + 1) & mask) != v1: continue
            else:
                op2comp = next((cand for cand in data.ops if cand.output == div2), None)
                if not op2comp or op2comp.opcode != "INT_2COMP" or len(op2comp.inputs) != 1: continue
                if op2comp.inputs[0] != div: continue
                
            outvn2 = multop.output
            if outvn2 is None: continue
            
            descendants2 = [cand for cand in data.ops if outvn2 in cand.inputs]
            for addop in descendants2:
                if addop.opcode != "INT_ADD" or len(addop.inputs) != 2: continue
                
                lvn = addop.inputs[0]
                if lvn == outvn2: lvn = addop.inputs[1]
                
                if lvn != x: continue
                
                data.op_set_input(addop, x, 0)
                if cdiv.kind == "const":
                    cnew = len(data.nodes)
                    data.nodes.append(PcodeVarnode(cnew, cdiv.name, cdiv.size_bits, "const"))
                    data.op_set_input(addop, cnew, 1)
                else:
                    data.op_set_input(addop, div, 1)
                    
                data.op_set_opcode(addop, "INT_REM" if op.opcode == "INT_DIV" else "INT_SREM")
                return 1
        return 0


class RuleSignMod2nOpt(Rule):
    """Convert INT_SREM forms: (V + (sign >> (64-n)) & (2^n-1)) - (sign >> (64-n)) => V s% 2^n"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "signmod2nopt")

    def get_op_list(self) -> list[str]:
        return ["INT_RIGHT"]
        
    def check_sign_extraction(self, outVn, data):
        signOp = next((cand for cand in data.ops if cand.output == outVn), None)
        if not signOp or signOp.opcode != "INT_SRIGHT" or len(signOp.inputs) != 2: return None
        constVn = data.nodes[signOp.inputs[1]]
        if constVn.kind != "const": return None
        
        val = int(constVn.name, 16)
        resVn = signOp.inputs[0]
        insize = data.nodes[resVn].size_bits
        if val != insize - 1: return None
        return resVn

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const": return 0
        shiftAmt = int(cnode.name, 16)
        
        a = self.check_sign_extraction(op.inputs[0], data)
        if a is None: return 0
        
        correctVn = op.output
        if correctVn is None: return 0
        
        n = data.nodes[a].size_bits - shiftAmt
        mask = (1 << n) - 1
        
        descendants_correct = [cand for cand in data.ops if correctVn in cand.inputs]
        for multop in descendants_correct:
            if multop.opcode != "INT_MULT" or len(multop.inputs) != 2: continue
            negone = data.nodes[multop.inputs[1]]
            if negone.kind != "const": continue
            if int(negone.name, 16) != (1 << data.nodes[correctVn].size_bits) - 1: continue
            
            out_mult = multop.output
            if out_mult is None: continue
            
            descendants_mult = [cand for cand in data.ops if out_mult in cand.inputs]
            if len(descendants_mult) != 1: continue
            baseOp = descendants_mult[0]
            if baseOp.opcode != "INT_ADD" or len(baseOp.inputs) != 2: continue
            
            slot = 1 if baseOp.inputs[0] == out_mult else 0
            andOut = baseOp.inputs[slot]
            andOp = next((cand for cand in data.ops if cand.output == andOut), None)
            
            truncSize = -1
            if andOp and andOp.opcode == "INT_ZEXT" and len(andOp.inputs) == 1:
                andOut = andOp.inputs[0]
                andOp = next((cand for cand in data.ops if cand.output == andOut), None)
                if andOp and andOp.opcode == "INT_AND":
                    truncSize = data.nodes[andOut].size_bits
            elif not andOp or andOp.opcode != "INT_AND" or len(andOp.inputs) != 2:
                continue
                
            if not andOp: continue
            
            constVn2 = data.nodes[andOp.inputs[1]]
            if constVn2.kind != "const" or int(constVn2.name, 16) != mask: continue
            
            addOut = andOp.inputs[0]
            addOp = next((cand for cand in data.ops if cand.output == addOut), None)
            if not addOp or addOp.opcode != "INT_ADD" or len(addOp.inputs) != 2: continue
            
            aSlot = -1
            for idx in range(2):
                vn = addOp.inputs[idx]
                if truncSize >= 0:
                    subOp = next((cand for cand in data.ops if cand.output == vn), None)
                    if subOp and subOp.opcode == "SUBPIECE" and len(subOp.inputs) == 2:
                        subc = data.nodes[subOp.inputs[1]]
                        if subc.kind == "const" and int(subc.name, 16) == 0:
                            vn = subOp.inputs[0]
                if a == vn:
                    aSlot = idx
                    break
            
            if aSlot == -1: continue
            
            extVn = addOp.inputs[1 - aSlot]
            shiftOp = next((cand for cand in data.ops if cand.output == extVn), None)
            if not shiftOp or shiftOp.opcode != "INT_RIGHT" or len(shiftOp.inputs) != 2: continue
            
            sconst = data.nodes[shiftOp.inputs[1]]
            if sconst.kind != "const": continue
            shiftval = int(sconst.name, 16)
            
            if truncSize >= 0:
                shiftval += (data.nodes[a].size_bits - truncSize)
                
            if shiftval != shiftAmt: continue
            
            extVn = shiftOp.inputs[0]
            if truncSize >= 0:
                subOp = next((cand for cand in data.ops if cand.output == extVn), None)
                if subOp and subOp.opcode == "SUBPIECE" and len(subOp.inputs) == 2:
                    subc = data.nodes[subOp.inputs[1]]
                    if subc.kind == "const" and int(subc.name, 16) == (truncSize // 8):
                        extVn = subOp.inputs[0]
            
            actual_a = self.check_sign_extraction(extVn, data)
            if actual_a != a: continue
            
            data.op_set_opcode(baseOp, "INT_SREM")
            data.op_set_input(baseOp, a, 0)
            
            c_pow = len(data.nodes)
            data.nodes.append(PcodeVarnode(c_pow, hex(1 << n), data.nodes[a].size_bits, "const"))
            data.op_set_input(baseOp, c_pow, 1)
            return 1
        return 0


class RuleSignMod2Opt(Rule):
    """Convert INT_SREM form: (V - sign)&1 + sign => V s% 2"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "signmod2opt")

    def get_op_list(self) -> list[str]:
        return ["INT_AND"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        constVn = data.nodes[op.inputs[1]]
        if constVn.kind != "const" or int(constVn.name, 16) != 1: return 0
        
        addOut = op.inputs[0]
        addOp = next((cand for cand in data.ops if cand.output == addOut), None)
        if not addOp or addOp.opcode != "INT_ADD" or len(addOp.inputs) != 2: return 0
        
        multSlot = -1
        multOp = None
        for i in range(2):
            vn = addOp.inputs[i]
            mOp = next((cand for cand in data.ops if cand.output == vn), None)
            if mOp and mOp.opcode == "INT_MULT" and len(mOp.inputs) == 2:
                cVn = data.nodes[mOp.inputs[1]]
                if cVn.kind == "const" and int(cVn.name, 16) == (1 << cVn.size_bits) - 1:
                    multSlot = i
                    multOp = mOp
                    break
        
        if multSlot == -1: return 0
        
        from .rule import RuleSignMod2nOpt
        # We need an instance to access the check_sign_extraction, or we can just redefine it inline since we didn't save it as a classmethod
        
        def check_sign_extraction(outVn):
            signOp = next((cand for cand in data.ops if cand.output == outVn), None)
            if not signOp or signOp.opcode != "INT_SRIGHT" or len(signOp.inputs) != 2: return None
            cVn = data.nodes[signOp.inputs[1]]
            if cVn.kind != "const": return None
            val = int(cVn.name, 16)
            resVn = signOp.inputs[0]
            insize = data.nodes[resVn].size_bits
            if val != insize - 1: return None
            return resVn
            
        base = check_sign_extraction(multOp.inputs[0])
        if base is None: return 0
        
        otherBase = addOp.inputs[1 - multSlot]
        trunc = False
        if base != otherBase:
            sub1 = next((cand for cand in data.ops if cand.output == base), None)
            if not sub1 or sub1.opcode != "SUBPIECE" or len(sub1.inputs) != 2: return 0
            subc = data.nodes[sub1.inputs[1]]
            if subc.kind != "const": return 0
            truncAmt = int(subc.name, 16) * 8 # offset to bits
            if truncAmt + data.nodes[base].size_bits != data.nodes[sub1.inputs[0]].size_bits: return 0
            base = sub1.inputs[0]
            
            sub2 = next((cand for cand in data.ops if cand.output == otherBase), None)
            if not sub2 or sub2.opcode != "SUBPIECE" or len(sub2.inputs) != 2: return 0
            subc2 = data.nodes[sub2.inputs[1]]
            if subc2.kind != "const" or int(subc2.name, 16) != 0: return 0
            otherBase = sub2.inputs[0]
            if otherBase != base: return 0
            trunc = True
            
        andOut = op.output
        if andOut is None: return 0
        
        if trunc:
            descendants = [cand for cand in data.ops if andOut in cand.inputs]
            if len(descendants) != 1 or descendants[0].opcode != "INT_ZEXT": return 0
            extOp = descendants[0]
            andOut = extOp.output
            if andOut is None: return 0
            
        descendants = [cand for cand in data.ops if andOut in cand.inputs]
        for rootOp in descendants:
            if rootOp.opcode != "INT_ADD" or len(rootOp.inputs) != 2: continue
            slot = 1 if rootOp.inputs[0] == andOut else 0
            otherBase = check_sign_extraction(rootOp.inputs[1 - slot])
            if otherBase != base: continue
            
            data.op_set_opcode(rootOp, "INT_SREM")
            data.op_set_input(rootOp, base, 0)
            
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, "0x2", data.nodes[base].size_bits, "const"))
            data.op_set_input(rootOp, cnew, 1)
            return 1
        return 0


class RuleSignMod2nOpt2(Rule):
    """Convert INT_SREM form: V - (Vadj & ~(2^n-1)) => V s% 2^n"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "signmod2nopt2")

    def get_op_list(self) -> list[str]:
        return ["INT_MULT"]

    def check_sign_ext_form(self, op: PcodeOp, data: PcodeLikeGraph):
        for slot in range(2):
            minusVn = op.inputs[slot]
            multOp = next((cand for cand in data.ops if cand.output == minusVn), None)
            if not multOp or multOp.opcode != "INT_MULT" or len(multOp.inputs) != 2: continue
            constVn = data.nodes[multOp.inputs[1]]
            if constVn.kind != "const": continue
            if int(constVn.name, 16) != (1 << constVn.size_bits) - 1: continue
            
            base = op.inputs[1 - slot]
            signExt = multOp.inputs[0]
            shiftOp = next((cand for cand in data.ops if cand.output == signExt), None)
            if not shiftOp or shiftOp.opcode != "INT_SRIGHT" or len(shiftOp.inputs) != 2: continue
            if shiftOp.inputs[0] != base: continue
            
            sconst = data.nodes[shiftOp.inputs[1]]
            if sconst.kind != "const": continue
            if int(sconst.name, 16) != data.nodes[base].size_bits * 8 - 1: continue
            return base
        return None

    def check_multiequal_form(self, op: PcodeOp, npow: int, data: PcodeLikeGraph):
        if len(op.inputs) != 2: return None
        npow_minus_1 = npow - 1
        
        base = None
        for slot in range(2):
            addOut = op.inputs[slot]
            addOp = next((cand for cand in data.ops if cand.output == addOut), None)
            if not addOp or addOp.opcode != "INT_ADD" or len(addOp.inputs) != 2: continue
            constVn = data.nodes[addOp.inputs[1]]
            if constVn.kind != "const" or int(constVn.name, 16) != npow_minus_1: continue
            base = addOp.inputs[0]
            otherBase = op.inputs[1 - slot]
            if otherBase == base:
                return base
        return None

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        constVn = data.nodes[op.inputs[1]]
        if constVn.kind != "const": return 0
        
        mask = (1 << constVn.size_bits) - 1
        if int(constVn.name, 16) != mask: return 0
        
        andOut = op.inputs[0]
        andOp = next((cand for cand in data.ops if cand.output == andOut), None)
        if not andOp or andOp.opcode != "INT_AND" or len(andOp.inputs) != 2: return 0
        
        cAndVn = data.nodes[andOp.inputs[1]]
        if cAndVn.kind != "const": return 0
        
        npow = (~int(cAndVn.name, 16) + 1) & mask
        if bin(npow).count('1') != 1: return 0
        if npow == 1: return 0
        
        adjVn = andOp.inputs[0]
        adjOp = next((cand for cand in data.ops if cand.output == adjVn), None)
        if not adjOp: return 0
        
        base = None
        if adjOp.opcode == "INT_ADD":
            if npow != 2: return 0
            base = self.check_sign_ext_form(adjOp, data)
        elif adjOp.opcode == "MULTIEQUAL":
            base = self.check_multiequal_form(adjOp, npow, data)
        
        if base is None: return 0
        
        multOut = op.output
        if multOut is None: return 0
        
        descendants = [cand for cand in data.ops if multOut in cand.inputs]
        for rootOp in descendants:
            if rootOp.opcode != "INT_ADD" or len(rootOp.inputs) != 2: continue
            slot = 1 if rootOp.inputs[0] == multOut else 0
            if rootOp.inputs[1 - slot] != base: continue
            
            if slot == 0:
                data.op_set_input(rootOp, base, 0)
            
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, hex(npow), data.nodes[base].size_bits, "const"))
            data.op_set_input(rootOp, cnew, 1)
            data.op_set_opcode(rootOp, "INT_SREM")
            return 1
        return 0


class RulePositiveDiv(Rule):
    """Signed division of positive values is unsigned division"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "positivediv")

    def get_op_list(self) -> list[str]:
        return ["INT_SDIV", "INT_SREM"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        sa = data.nodes[op.output].size_bits if op.output is not None else data.nodes[op.inputs[0]].size_bits
        sa = sa - 1
        
        in0 = op.inputs[0]
        in1 = op.inputs[1]
        
        mask0 = data.get_nzmask(in0)
        mask1 = data.get_nzmask(in1)
        
        if ((mask0 >> sa) & 1) != 0: return 0
        if ((mask1 >> sa) & 1) != 0: return 0
        
        opc = "INT_DIV" if op.opcode == "INT_SDIV" else "INT_REM"
        data.op_set_opcode(op, opc)
        return 1


class RuleDivTermAdd(Rule):
    """Simplify expressions associated with optimized division expressions"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "divtermadd")

    def get_op_list(self) -> list[str]:
        return ["SUBPIECE", "INT_RIGHT", "INT_SRIGHT"]
        
    def find_subshift(self, op: PcodeOp, data: PcodeLikeGraph):
        shiftopc = op.opcode
        n = 0
        if shiftopc != "SUBPIECE":
            vn = op.inputs[0]
            subop = next((cand for cand in data.ops if cand.output == vn), None)
            if not subop or subop.opcode != "SUBPIECE" or len(subop.inputs) != 2: return None, "MAX", 0
            sconst = data.nodes[op.inputs[1]]
            if sconst.kind != "const": return None, "MAX", 0
            n = int(sconst.name, 16)
        else:
            shiftopc = "MAX"
            subop = op
            
        cnode = data.nodes[subop.inputs[1]]
        if cnode.kind != "const": return None, "MAX", 0
        c = int(cnode.name, 16)
        
        in_size = data.nodes[subop.inputs[0]].size_bits
        out_size = data.nodes[subop.output].size_bits if subop.output is not None else 0
        if out_size + c * 8 != in_size: return None, "MAX", 0
        
        n += 8 * c
        return subop, shiftopc, n

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        subop, shiftopc, n = self.find_subshift(op, data)
        if subop is None: return 0
        if n > 127: return 0
        
        multvn = subop.inputs[0]
        multop = next((cand for cand in data.ops if cand.output == multvn), None)
        if not multop or multop.opcode != "INT_MULT" or len(multop.inputs) != 2: return 0
        
        multConstNode = data.nodes[multop.inputs[1]]
        if multConstNode.kind != "const": return 0
        multConst = int(multConstNode.name, 16)
        
        extvn = multop.inputs[0]
        extop = next((cand for cand in data.ops if cand.output == extvn), None)
        if not extop or extop.opcode not in ("INT_ZEXT", "INT_SEXT"): return 0
        
        opc = extop.opcode
        if opc == "INT_ZEXT" and op.opcode == "INT_SRIGHT": return 0
        if opc == "INT_SEXT" and op.opcode == "INT_RIGHT": return 0
        
        multConst += (1 << n)
        multConst &= (1 << data.nodes[extvn].size_bits) - 1
        
        x = extop.inputs[0]
        out_op = op.output
        if out_op is None: return 0
        
        descendants = [cand for cand in data.ops if out_op in cand.inputs]
        for addop in descendants:
            if addop.opcode != "INT_ADD" or len(addop.inputs) != 2: continue
            if addop.inputs[0] != x and addop.inputs[1] != x: continue
            
            newmultop = PcodeOp(op.addr, "INT_MULT", len(data.nodes), (extvn, len(data.nodes)))
            c_new = len(data.nodes)
            data.nodes.append(PcodeVarnode(c_new, hex(multConst), data.nodes[extvn].size_bits, "const"))
            newmultop.inputs = (extvn, c_new)
            newmultvn = len(data.nodes)
            data.nodes.append(PcodeVarnode(newmultvn, None, data.nodes[extvn].size_bits, "unique"))
            newmultop.output = newmultvn
            data.ops.append(newmultop)
            
            newshiftop = PcodeOp(op.addr, "INT_RIGHT" if shiftopc == "MAX" else shiftopc, len(data.nodes), (newmultvn, len(data.nodes)))
            c_n = len(data.nodes)
            data.nodes.append(PcodeVarnode(c_n, hex(n), 4 * 8, "const"))
            newshiftop.inputs = (newmultvn, c_n)
            newshiftvn = len(data.nodes)
            data.nodes.append(PcodeVarnode(newshiftvn, None, data.nodes[extvn].size_bits, "unique"))
            newshiftop.output = newshiftvn
            data.ops.append(newshiftop)
            
            data.op_set_opcode(addop, "SUBPIECE")
            data.op_set_input(addop, newshiftvn, 0)
            c_0 = len(data.nodes)
            data.nodes.append(PcodeVarnode(c_0, "0x0", 4 * 8, "const"))
            data.op_set_input(addop, c_0, 1)
            return 1
        return 0


class RuleDivOpt(Rule):
    """Convert INT_MULT and shift forms into INT_DIV or INT_SDIV"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "divopt")

    def get_op_list(self) -> list[str]:
        return ["SUBPIECE", "INT_RIGHT", "INT_SRIGHT"]

    def find_form(self, op: PcodeOp, data: PcodeLikeGraph):
        curOp = op
        shiftopc = curOp.opcode
        if shiftopc in ("INT_RIGHT", "INT_SRIGHT"):
            vn = curOp.inputs[0]
            cvn = data.nodes[curOp.inputs[1]]
            if cvn.kind != "const": return None, 0, 0, 0, ""
            n = int(cvn.name, 16)
            curOp = next((cand for cand in data.ops if cand.output == vn), None)
            if not curOp: return None, 0, 0, 0, ""
        else:
            n = 0
            if shiftopc != "SUBPIECE": return None, 0, 0, 0, ""
            shiftopc = "MAX"
            
        if curOp.opcode == "SUBPIECE":
            c = int(data.nodes[curOp.inputs[1]].name, 16)
            inVn = curOp.inputs[0]
            if curOp.output is None: return None, 0, 0, 0, ""
            outsz = data.nodes[curOp.output].size_bits
            insz = data.nodes[inVn].size_bits
            if outsz + c * 8 != insz: return None, 0, 0, 0, ""
            n += 8 * c
            curOp = next((cand for cand in data.ops if cand.output == inVn), None)
            if not curOp: return None, 0, 0, 0, ""
            
        if curOp.opcode != "INT_MULT": return None, 0, 0, 0, ""
        
        inVn = curOp.inputs[0]
        y_node = data.nodes[curOp.inputs[1]]
        if y_node.kind == "const":
            y = int(y_node.name, 16)
        else:
            y_node = data.nodes[inVn]
            if y_node.kind == "const":
                y = int(y_node.name, 16)
                inVn = curOp.inputs[1]
            else:
                return None, 0, 0, 0, ""
                
        extOp = next((cand for cand in data.ops if cand.output == inVn), None)
        if not extOp: return None, 0, 0, 0, ""
        extopc = extOp.opcode
        
        if extopc != "INT_SEXT":
            if extopc == "INT_ZEXT":
                nzMask = data.get_nzmask(extOp.inputs[0])
            else:
                nzMask = data.get_nzmask(inVn)
            if nzMask == 0: return None, 0, 0, 0, ""
            xsize = nzMask.bit_length()
            if xsize == 0 or xsize > 4 * data.nodes[inVn].size_bits: return None, 0, 0, 0, ""
        else:
            xsize = data.nodes[extOp.inputs[0]].size_bits
            
        if extopc in ("INT_SEXT", "INT_ZEXT"):
            return extOp.inputs[0], n, y, xsize, extopc
        return inVn, n, y, xsize, extopc

    def check_form_overlap(self, op: PcodeOp, data: PcodeLikeGraph):
        if op.opcode != "SUBPIECE" or op.output is None: return False
        vn = op.output
        descendants = [cand for cand in data.ops if vn in cand.inputs]
        for superOp in descendants:
            if superOp.opcode not in ("INT_RIGHT", "INT_SRIGHT"): continue
            cvn = data.nodes[superOp.inputs[1]]
            if cvn.kind != "const": return True
            inVn, _, _, _, _ = self.find_form(superOp, data)
            if inVn is not None: return True
        return False

    def calc_divisor(self, n: int, y: int, xsize: int):
        if n > 127 or xsize > 64: return 0
        power = 1 << n
        if y <= 1: return 0
        y_minus_1 = y - 1
        q = power // y_minus_1
        r = power % y_minus_1
        if q >= (1 << 64): return 0
        if y_minus_1 < q: return 0
        
        diff = 0
        if r >= q:
            q += 1
            r = r - y_minus_1 + q
            if r >= q: return 0
            diff = q
            
        maxx = 0 if xsize == 64 else (1 << xsize)
        maxx -= 1
        diff += q - r
        if diff == 0:
            return q
        tmp = power // diff
        if tmp >= (1 << 64): return q
        if tmp <= maxx: return 0
        return q

    def move_sign_bit_extraction(self, firstVn, replaceVn, data: PcodeLikeGraph):
        testList = [firstVn]
        op_def = next((cand for cand in data.ops if cand.output == firstVn), None)
        if op_def and op_def.opcode == "INT_SRIGHT":
            testList.append(op_def.inputs[0])
            
        visited = set()
        for vn in testList:
            if vn in visited: continue
            visited.add(vn)
            descendants = [cand for cand in data.ops if vn in cand.inputs]
            for op in descendants:
                if op.opcode in ("INT_RIGHT", "INT_SRIGHT"):
                    cvn_idx = op.inputs[1]
                    cvn = data.nodes[cvn_idx]
                    if cvn.kind != "const": continue
                    sa = data.nodes[firstVn].size_bits * 8 - 1
                    if sa == int(cvn.name, 16):
                        data.op_set_input(op, replaceVn, 0)
                elif op.opcode == "COPY":
                    if op.output is not None:
                        testList.append(op.output)

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        inVn, n, y, xsize, extOpc = self.find_form(op, data)
        if inVn is None: return 0
        if self.check_form_overlap(op, data): return 0
        if extOpc == "INT_SEXT": xsize -= 1
        
        divisor = self.calc_divisor(n, y, xsize)
        if divisor == 0: return 0
        
        outSize = data.nodes[op.output].size_bits
        
        if data.nodes[inVn].size_bits < outSize:
            inExt = PcodeOp(op.addr, extOpc, len(data.nodes), (inVn,))
            extOut = len(data.nodes)
            data.nodes.append(PcodeVarnode(extOut, None, outSize, "unique"))
            inExt.output = extOut
            data.ops.append(inExt)
            inVn = extOut
        elif data.nodes[inVn].size_bits > outSize:
            newop = PcodeOp(op.addr, "INT_ADD", len(data.nodes), op.inputs)
            resVn = len(data.nodes)
            data.nodes.append(PcodeVarnode(resVn, None, data.nodes[inVn].size_bits, "unique"))
            newop.output = resVn
            data.ops.append(newop)
            
            data.op_set_opcode(op, "SUBPIECE")
            data.op_set_input(op, resVn, 0)
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, "0x0", 4 * 8, "const"))
            data.op_set_input(op, cnew, 1)
            op = newop
            outSize = data.nodes[inVn].size_bits
            
        if extOpc == "INT_ZEXT":
            data.op_set_input(op, inVn, 0)
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, hex(divisor), outSize, "const"))
            data.op_set_input(op, cnew, 1)
            data.op_set_opcode(op, "INT_DIV")
        else:
            self.move_sign_bit_extraction(op.output, inVn, data)
            divop = PcodeOp(op.addr, "INT_SDIV", len(data.nodes), (inVn, len(data.nodes)))
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, hex(divisor), outSize, "const"))
            divop.inputs = (inVn, cnew)
            newout = len(data.nodes)
            data.nodes.append(PcodeVarnode(newout, None, outSize, "unique"))
            divop.output = newout
            data.ops.append(divop)
            
            sgnop = PcodeOp(op.addr, "INT_SRIGHT", len(data.nodes), (inVn, len(data.nodes)))
            cnew2 = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew2, hex(outSize * 8 - 1), outSize, "const"))
            sgnop.inputs = (inVn, cnew2)
            sgnvn = len(data.nodes)
            data.nodes.append(PcodeVarnode(sgnvn, None, outSize, "unique"))
            sgnop.output = sgnvn
            data.ops.append(sgnop)
            
            data.op_set_input(op, newout, 0)
            data.op_set_input(op, sgnvn, 1)
            data.op_set_opcode(op, "INT_ADD")
            
        return 1


class RuleSignDiv2(Rule):
    """Convert INT_SRIGHT form into INT_SDIV: (V + -1*(V s>> 31)) s>> 1 => V s/ 2"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "signdiv2")

    def get_op_list(self) -> list[str]:
        return ["INT_SRIGHT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const" or int(cnode.name, 16) != 1: return 0
        
        addout = op.inputs[0]
        addop = next((cand for cand in data.ops if cand.output == addout), None)
        if not addop or addop.opcode != "INT_ADD" or len(addop.inputs) != 2: return 0
        
        a = None
        for i in range(2):
            multout = addop.inputs[i]
            multop = next((cand for cand in data.ops if cand.output == multout), None)
            if not multop or multop.opcode != "INT_MULT" or len(multop.inputs) != 2: continue
            
            cnode2 = data.nodes[multop.inputs[1]]
            if cnode2.kind != "const" or int(cnode2.name, 16) != (1 << cnode2.size_bits) - 1: continue
            
            shiftout = multop.inputs[0]
            shiftop = next((cand for cand in data.ops if cand.output == shiftout), None)
            if not shiftop or shiftop.opcode != "INT_SRIGHT" or len(shiftop.inputs) != 2: continue
            
            cnode3 = data.nodes[shiftop.inputs[1]]
            if cnode3.kind != "const": continue
            
            n = int(cnode3.name, 16)
            a_cand = shiftop.inputs[0]
            if a_cand != addop.inputs[1 - i]: continue
            if n != data.nodes[a_cand].size_bits * 8 - 1: continue
            
            a = a_cand
            break
            
        if a is None: return 0
        
        data.op_set_input(op, a, 0)
        cnew = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew, "0x2", data.nodes[a].size_bits, "const"))
        data.op_set_input(op, cnew, 1)
        data.op_set_opcode(op, "INT_SDIV")
        return 1


class RuleDivTermAdd2(Rule):
    """Simplify another expression associated with optimized division (unsigned)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "divtermadd2")

    def get_op_list(self) -> list[str]:
        return ["INT_RIGHT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        cnode = data.nodes[op.inputs[1]]
        if cnode.kind != "const" or int(cnode.name, 16) != 1: return 0
        
        subvn = op.inputs[0]
        subop = next((cand for cand in data.ops if cand.output == subvn), None)
        if not subop or subop.opcode != "INT_ADD" or len(subop.inputs) != 2: return 0
        
        x = None
        compvn = None
        compop = None
        for i in range(2):
            compvn = subop.inputs[i]
            compop = next((cand for cand in data.ops if cand.output == compvn), None)
            if compop and compop.opcode == "INT_MULT" and len(compop.inputs) == 2:
                invn = data.nodes[compop.inputs[1]]
                if invn.kind == "const" and int(invn.name, 16) == (1 << invn.size_bits) - 1:
                    x = subop.inputs[1 - i]
                    break
        
        if x is None: return 0
        
        z = compop.inputs[0]
        subpieceop = next((cand for cand in data.ops if cand.output == z), None)
        if not subpieceop or subpieceop.opcode != "SUBPIECE" or len(subpieceop.inputs) != 2: return 0
        
        csub = data.nodes[subpieceop.inputs[1]]
        if csub.kind != "const": return 0
        n = int(csub.name, 16) * 8
        if n > 127 or n != 8 * (data.nodes[subpieceop.inputs[0]].size_bits - data.nodes[z].size_bits): return 0
        
        multvn = subpieceop.inputs[0]
        multop = next((cand for cand in data.ops if cand.output == multvn), None)
        if not multop or multop.opcode != "INT_MULT" or len(multop.inputs) != 2: return 0
        
        cmult = data.nodes[multop.inputs[1]]
        if cmult.kind != "const": return 0
        multConst = int(cmult.name, 16)
        
        zextvn = multop.inputs[0]
        zextop = next((cand for cand in data.ops if cand.output == zextvn), None)
        if not zextop or zextop.opcode != "INT_ZEXT" or len(zextop.inputs) != 1: return 0
        if zextop.inputs[0] != x: return 0
        
        out_op = op.output
        if out_op is None: return 0
        
        descendants = [cand for cand in data.ops if out_op in cand.inputs]
        for addop in descendants:
            if addop.opcode != "INT_ADD" or len(addop.inputs) != 2: continue
            if addop.inputs[0] != z and addop.inputs[1] != z: continue
            
            pow2 = 1 << n
            multConst += pow2
            multConst &= (1 << data.nodes[zextvn].size_bits) - 1
            
            newmultop = PcodeOp(op.addr, "INT_MULT", len(data.nodes), (zextvn, len(data.nodes)))
            cnew1 = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew1, hex(multConst), data.nodes[zextvn].size_bits, "const"))
            newmultop.inputs = (zextvn, cnew1)
            
            newmultvn = len(data.nodes)
            data.nodes.append(PcodeVarnode(newmultvn, None, data.nodes[zextvn].size_bits, "unique"))
            newmultop.output = newmultvn
            data.ops.append(newmultop)
            
            newshiftop = PcodeOp(op.addr, "INT_RIGHT", len(data.nodes), (newmultvn, len(data.nodes)))
            cnew2 = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew2, hex(n + 1), 4 * 8, "const"))
            newshiftop.inputs = (newmultvn, cnew2)
            
            newshiftvn = len(data.nodes)
            data.nodes.append(PcodeVarnode(newshiftvn, None, data.nodes[zextvn].size_bits, "unique"))
            newshiftop.output = newshiftvn
            data.ops.append(newshiftop)
            
            data.op_set_opcode(addop, "SUBPIECE")
            data.op_set_input(addop, newshiftvn, 0)
            cnew3 = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew3, "0x0", 4 * 8, "const"))
            data.op_set_input(addop, cnew3, 1)
            return 1
        return 0


class RuleConcatLeftShift(Rule):
    """Simplify concat(V, zext(W) << c) => concat(concat(V, W), 0)"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "concatleftshift")

    def get_op_list(self) -> list[str]:
        return ["PIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        vn2 = op.inputs[1]
        shiftop = next((cand for cand in data.ops if cand.output == vn2), None)
        if not shiftop or shiftop.opcode != "INT_LEFT" or len(shiftop.inputs) != 2: return 0
        
        cnode = data.nodes[shiftop.inputs[1]]
        if cnode.kind != "const": return 0
        
        sa = int(cnode.name, 16)
        if (sa & 7) != 0: return 0
        
        tmpvn = shiftop.inputs[0]
        zextop = next((cand for cand in data.ops if cand.output == tmpvn), None)
        if not zextop or zextop.opcode != "INT_ZEXT" or len(zextop.inputs) != 1: return 0
        
        b = zextop.inputs[0]
        vn1 = op.inputs[0]
        
        sa //= 8
        if sa + data.nodes[b].size_bits != data.nodes[tmpvn].size_bits: return 0
        
        newop = PcodeOp(op.addr, "PIECE", len(data.nodes), (vn1, b))
        newout = len(data.nodes)
        data.nodes.append(PcodeVarnode(newout, None, data.nodes[vn1].size_bits + data.nodes[b].size_bits, "unique"))
        newop.output = newout
        data.ops.append(newop)
        
        out_size = data.nodes[op.output].size_bits if op.output is not None else 0
        diff = out_size - data.nodes[newout].size_bits
        if diff < 0: return 0
        
        data.op_set_input(op, newout, 0)
        c_diff = len(data.nodes)
        data.nodes.append(PcodeVarnode(c_diff, "0x0", diff, "const"))
        data.op_set_input(op, c_diff, 1)
        return 1


class RuleExpandLoad(Rule):
    """Push BOOL_AND through INT_AND into INT_EQUAL where one side is a LOAD/INDIRECT"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "expandload")

    def get_op_list(self) -> list[str]:
        return ["LOAD"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        outVn = op.output
        if outVn is None: return 0
        outSize = data.nodes[outVn].size_bits // 8
        rootPtr = op.inputs[1]
        
        # Checking logic here is heavily tied to type recovery which is not fully present.
        # We skip implementing a full equivalent since TypePointer, etc. aren't implemented here.
        # Returning 0 to keep it stubbed safely.
        return 0


class RuleExtensionPush(Rule):
    """Duplicate extension (INT_ZEXT/INT_SEXT) through operations to avoid truncating."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "extensionpush")

    def get_op_list(self) -> list[str]:
        return ["INT_ZEXT", "INT_SEXT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        inVn = op.inputs[0]
        if data.nodes[inVn].kind == "const": return 0
        outVn = op.output
        if outVn is None: return 0
        
        addcount = 0
        ptrcount = 0
        descendants = [cand for cand in data.ops if outVn in cand.inputs]
        for decOp in descendants:
            if decOp.opcode == "PTRADD":
                ptrcount += 1
            elif decOp.opcode == "INT_ADD":
                sub_out = decOp.output
                if sub_out is None: return 0
                sub_desc = [c for c in data.ops if sub_out in c.inputs]
                if len(sub_desc) != 1 or sub_desc[0].opcode != "PTRADD":
                    return 0
                addcount += 1
            else:
                return 0
                
        if addcount + ptrcount <= 1: return 0
        
        if addcount > 0:
            if len([c for c in data.ops if inVn in c.inputs]) != 1: return 0
            
        # Duplicating extension to descendants:
        for decOp in descendants:
            newext = PcodeOp(decOp.addr, op.opcode, len(data.nodes), (inVn,))
            newext_out = len(data.nodes)
            data.nodes.append(PcodeVarnode(newext_out, None, data.nodes[outVn].size_bits, "unique"))
            newext.output = newext_out
            data.ops.append(newext)
            
            for i, slot_vn in enumerate(decOp.inputs):
                if slot_vn == outVn:
                    data.op_set_input(decOp, newext_out, i)
        
        return 1


class RuleStructOffset0(Rule):
    """Convert LOAD/STORE to first element of structure into PTRSUB"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "structoffset0")

    def get_op_list(self) -> list[str]:
        return ["LOAD", "STORE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        # Requires advanced Type propagation which is not in minighidra
        # Safely return 0.
        return 0


class RulePtrsubCharConstant(Rule):
    """Convert PTRSUB into a pointer constant to a string if appropriate"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "ptrsubcharconstant")

    def get_op_list(self) -> list[str]:
        return ["PTRSUB"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        # Relies heavily on full typing + StringManager which we do not have.
        return 0


class RuleRightShiftSub(Rule):
    """This rule is commented out in original Ghidra."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "rightshiftsub")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleConditionalMove(Rule):
    """This rule requires block structure analysis in rules which isn't currently implemented."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "conditionalmove")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return ["MULTIEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RulePullsubIndirect(Rule):
    """This rule relies heavily on alias analysis and indirects which are not fully structured in minighidra yet"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "pullsubindirect")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RulePushMulti(Rule):
    """This rule requires flow block analysis which isn't currently ported to the minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "pushmulti")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RulePushPtr(Rule):
    """This rule requires flow block analysis which isn't currently ported to the minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "pushptr")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RulePtrsubUndo(Rule):
    """This rule requires Type propagation (for pointer alignment info) which isn't currently ported to the minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "ptrsubundo")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RulePtraddUndo(Rule):
    """This rule requires Type propagation which isn't currently ported to the minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "ptraddundo")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RulePiecePathology(Rule):
    """This rule relies heavily on indirects, call specifications, and consumption which are not ported into minighidra's AST rules."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "piecepathology")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RulePieceStructure(Rule):
    """This rule requires Type propagation (for struct layout info) which isn't currently ported to the minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "piecestructure")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RulePopcountBoolXor(Rule):
    """Simplify POPCOUNT of boolean results combined into INT_XOR."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "popcountboolxor")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return ["POPCOUNT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        # minighidra lacks full POPCOUNT semantics and NZMask tracking of bit positions through AST natively
        return 0


class RulePtrArith(Rule):
    """This rule requires Type propagation (for pointer types) which isn't currently ported to the minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "ptrarith")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RulePtrFlow(Rule):
    """This rule requires Type propagation (for pointer types) which isn't currently ported to the minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "ptrflow")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleRangeMeld(Rule):
    """This rule requires CircleRange math/interval analysis which isn't ported into the minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "rangemeld")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleSegment(Rule):
    """This rule requires SegmentOp userops support which isn't ported into the minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "segment")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleSelectCse(Rule):
    """This rule requires CSE Hash calculation which isn't ported into the minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "selectcse")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


from typing import Tuple
from minighidra.expression import TermOrder

class RuleCollectTerms(Rule):
    """Collects and simplifies additive terms in expressions like a*3 + a*2 -> a*5."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "collectterms")

    def get_op_list(self) -> list[str]:
        return ["INT_ADD"]

    def _get_mult_coeff(self, vn_id: int, data: PcodeLikeGraph) -> Tuple[int, int]:
        """Returns the underlying varnode and its coefficient."""
        node = data.nodes[vn_id]
        if node.kind == "const":
            return vn_id, 1
            
        def_op = None
        for op in data.ops:
            if op.output == vn_id:
                def_op = op
                break
                
        if def_op and def_op.opcode == "INT_MULT" and len(def_op.inputs) == 2:
            in1 = data.nodes[def_op.inputs[1]]
            if in1.kind == "const":
                val = int(in1.name, 16) if in1.name else 0
                return def_op.inputs[0], val
        return vn_id, 1

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if not op.output: return 0
        
        # Check if next op is an ADD
        lone_descend = None
        count = 0
        for other_op in data.ops:
            if op.output in other_op.inputs:
                count += 1
                lone_descend = other_op
        if count == 1 and lone_descend and lone_descend.opcode == "INT_ADD":
            return 0
            
        term_order = TermOrder(op, data)
        term_order.collect()
        term_order.sort_terms()
        
        order = term_order.sorter
        if len(order) < 2:
            return 0
            
        first_node = data.nodes[order[0].varnode]
        if not (first_node.kind == "const"):
            for i in range(1, len(order)):
                vn1_id = order[i-1].varnode
                vn2_id = order[i].varnode
                
                n2 = data.nodes[vn2_id]
                if n2.kind == "const":
                    break
                    
                base_vn1, coef1 = self._get_mult_coeff(vn1_id, data)
                base_vn2, coef2 = self._get_mult_coeff(vn2_id, data)
                
                if base_vn1 == base_vn2:
                    if order[i-1].multiplier or order[i].multiplier:
                        # distributeIntMultAdd logic skipped for now
                        pass
                    else:
                        base_node = data.nodes[base_vn1]
                        sz = base_node.size_bits
                        new_coef = (coef1 + coef2) & ((1 << sz) - 1)
                        
                        new_cnode = len(data.nodes)
                        data.nodes.append(PcodeVarnode(new_cnode, hex(new_coef), sz, "const"))
                        zero_cnode = len(data.nodes)
                        data.nodes.append(PcodeVarnode(zero_cnode, hex(0), sz, "const"))
                        
                        data.op_set_input(order[i-1].op, zero_cnode, order[i-1].slot)
                        if new_coef == 0:
                            data.op_set_input(order[i].op, zero_cnode, order[i].slot)
                        else:
                            # Create multiplication
                            nextop = PcodeOp(len(data.ops), "INT_MULT", [base_vn1, new_cnode], None, order[i].op.addr)
                            vn2 = len(data.nodes)
                            data.nodes.append(PcodeVarnode(vn2, f"tmp_{vn2}", sz, "temp"))
                            nextop.output = vn2
                            data.op_insert_before(nextop, order[i].op)
                            data.op_set_input(order[i].op, vn2, order[i].slot)
                        return 1
        return 0


class RuleEarlyRemoval(Rule):
    """Remove purely dead op early in rule application."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "earlyremoval")

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if op.opcode in {"CALL", "CALLIND", "CALLOTHER", "STORE", "BRANCH", "CBRANCH", "BRANCHIND", "RETURN", "IMARK"}:
            return 0
        if op.output is None: return 0
        
        node = data.nodes[op.output]
        if node.kind not in {"temporary", "unique"}: return 0
        
        for cand in data.ops:
            if cand.opcode == "IMARK": continue
            if op.output in cand.inputs:
                return 0
                
        data.op_destroy(op)
        return 1

class RuleFloatCast(Rule):
    """Simplify FLOAT_FLOAT2FLOAT and FLOAT_TRUNC applied to FLOAT_FLOAT2FLOAT or FLOAT_INT2FLOAT"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "floatcast")

    def get_op_list(self) -> list[str]:
        return ["FLOAT_FLOAT2FLOAT", "FLOAT_TRUNC"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 1: return 0
        vn1 = op.inputs[0]
        castop = next((cand for cand in data.ops if cand.output == vn1), None)
        if not castop or len(castop.inputs) != 1: return 0
        
        opc2 = castop.opcode
        if opc2 not in ("FLOAT_FLOAT2FLOAT", "FLOAT_INT2FLOAT"): return 0
        
        opc1 = op.opcode
        vn2 = castop.inputs[0]
        
        insize1 = data.nodes[vn1].size_bits
        insize2 = data.nodes[vn2].size_bits
        outsize = data.nodes[op.output].size_bits if op.output is not None else 0
        
        if opc2 == "FLOAT_FLOAT2FLOAT" and opc1 == "FLOAT_FLOAT2FLOAT":
            if insize1 > outsize:
                data.op_set_input(op, vn2, 0)
                if outsize == insize2:
                    data.op_set_opcode(op, "COPY")
                return 1
            elif insize2 < insize1:
                data.op_set_input(op, vn2, 0)
                return 1
        elif opc2 == "FLOAT_INT2FLOAT" and opc1 == "FLOAT_FLOAT2FLOAT":
            data.op_set_input(op, vn2, 0)
            data.op_set_opcode(op, "FLOAT_INT2FLOAT")
            return 1
        elif opc2 == "FLOAT_FLOAT2FLOAT" and opc1 == "FLOAT_TRUNC":
            data.op_set_input(op, vn2, 0)
            return 1
        return 0


class RuleIgnoreNan(Rule):
    """This rule requires floating-point comparisons tracking and data configuration which isn't ported into the minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "ignorenan")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleUnsigned2Float(Rule):
    """Simplify conversion: T = int2float((X >> 1) | X & 1); T + T => int2float( zext(X) )"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "unsigned2float")

    def get_op_list(self) -> list[str]:
        return ["FLOAT_INT2FLOAT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 1: return 0
        invn = op.inputs[0]
        orop = next((cand for cand in data.ops if cand.output == invn), None)
        if not orop or orop.opcode != "INT_OR" or len(orop.inputs) != 2: return 0
        
        shiftop = None
        andop = None
        for i in range(2):
            cnd = next((cand for cand in data.ops if cand.output == orop.inputs[i]), None)
            if cnd and cnd.opcode == "INT_RIGHT":
                shiftop = cnd
                andop = next((cand for cand in data.ops if cand.output == orop.inputs[1 - i]), None)
                break
                
        if not shiftop: return 0
        
        cshift = data.nodes[shiftop.inputs[1]]
        if cshift.kind != "const" or int(cshift.name, 16) != 1: return 0
        
        basevn = shiftop.inputs[0]
        if not andop: return 0
        
        if andop.opcode == "INT_ZEXT" and len(andop.inputs) == 1:
            andop = next((cand for cand in data.ops if cand.output == andop.inputs[0]), None)
            
        if not andop or andop.opcode != "INT_AND" or len(andop.inputs) != 2: return 0
        
        cand_const = data.nodes[andop.inputs[1]]
        if cand_const.kind != "const" or int(cand_const.name, 16) != 1: return 0
        
        vn = andop.inputs[0]
        if basevn != vn:
            subop = next((cand for cand in data.ops if cand.output == vn), None)
            if not subop or subop.opcode != "SUBPIECE" or len(subop.inputs) != 2: return 0
            csub = data.nodes[subop.inputs[1]]
            if csub.kind != "const" or int(csub.name, 16) != 0: return 0
            vn = subop.inputs[0]
            if basevn != vn: return 0
            
        outvn = op.output
        if outvn is None: return 0
        
        descendants = [cand for cand in data.ops if outvn in cand.inputs]
        for addop in descendants:
            if addop.opcode != "FLOAT_ADD" or len(addop.inputs) != 2: continue
            if addop.inputs[0] != outvn or addop.inputs[1] != outvn: continue
            
            preferred_zext_size = 8 if data.nodes[basevn].size_bits == 4 * 8 else data.nodes[basevn].size_bits * 2
            # Simulating TypeOpFloatInt2Float::preferredZextSize approximation
            
            zextop = PcodeOp(addop.addr, "INT_ZEXT", len(data.nodes), (basevn,))
            zextout = len(data.nodes)
            data.nodes.append(PcodeVarnode(zextout, None, preferred_zext_size, "unique"))
            zextop.output = zextout
            data.ops.append(zextop)
            
            data.op_set_opcode(addop, "FLOAT_INT2FLOAT")
            data.op_remove_input(addop, 1)
            data.op_set_input(addop, zextout, 0)
            return 1
            
        return 0


from minighidra.block import FlowBlock

class RuleInt2FloatCollapse(Rule):
    """Collapse an unsigned FLOAT_INT2FLOAT and a signed FLOAT_INT2FLOAT converging into one float result."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "int2floatcollapse")

    def get_op_list(self) -> list[str]:
        return ["FLOAT_INT2FLOAT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if op.inputs[0] < 0: return 0
        zextop = None
        for dop in data.ops:
            if dop.output == op.inputs[0]:
                zextop = dop
                break
                
        if not zextop or zextop.opcode != "INT_ZEXT":
            return 0
            
        basevn = zextop.inputs[0]
        if basevn < 0: return 0
        
        multiop = None
        count = 0
        for dop in data.ops:
            if op.output in dop.inputs:
                multiop = dop
                count += 1
                
        if count != 1 or not multiop or multiop.opcode != "MULTIEQUAL" or len(multiop.inputs) != 2:
            return 0
            
        slot = multiop.inputs.index(op.output)
        otherout = multiop.inputs[1 - slot]
        
        op2 = None
        for dop in data.ops:
            if dop.output == otherout:
                op2 = dop
                break
                
        if not op2 or op2.opcode != "FLOAT_INT2FLOAT": return 0
        if op2.inputs[0] != basevn: return 0
        
        # Need multiop.getParent() logic, but minighidra PcodeOp doesn't have parent blocks directly stored.
        # We can find the parent block by checking data.blocks
        parent_bl = None
        for b in data.blocks:
            if hasattr(b, 'ops') and multiop in b.ops:
                parent_bl = b
                break
        if not parent_bl: return 0
        
        cond, dir2unsigned = FlowBlock.find_condition(parent_bl, slot, parent_bl, 1 - slot)
        if not cond: return 0
        
        cbranch = None
        if hasattr(cond, 'ops') and cond.ops:
            cbranch = cond.ops[-1]
            
        if not cbranch or cbranch.opcode != "CBRANCH": return 0
        
        # check isBooleanFlip
        if hasattr(data, 'op_is_boolean_flip') and data.op_is_boolean_flip(cbranch):
            return 0
            
        compare = None
        for dop in data.ops:
            if dop.output == cbranch.inputs[1]:
                compare = dop
                break
                
        if not compare or compare.opcode != "INT_SLESS": return 0
        
        in1_node = data.nodes[compare.inputs[1]] if compare.inputs[1] >= 0 else None
        in0_node = data.nodes[compare.inputs[0]] if compare.inputs[0] >= 0 else None
        
        if in1_node and in1_node.kind == "const" and int(in1_node.name, 16) == 0:
            if compare.inputs[0] != basevn: return 0
            if dir2unsigned != 1: return 0
        else:
            base_node = data.nodes[basevn]
            mask = (1 << base_node.size_bits) - 1
            if in0_node and in0_node.kind == "const" and int(in0_node.name, 16) == mask:
                if compare.inputs[1] != basevn: return 0
                if dir2unsigned == 1: return 0
            else:
                return 0
                
        data.op_set_opcode(multiop, "FLOAT_INT2FLOAT")
        # Ensure multiop has only 1 input now: the basevn
        while len(multiop.inputs) > 0:
            data.op_remove_input(multiop, 0)
        data.op_set_input(multiop, basevn, 0)
        
        return 1

class RuleFuncPtrEncoding(Rule):
    """This rule relies on funcptr_align from architecture info, which isn't currently surfaced in minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "funcptrencoding")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleLzcountShiftBool(Rule):
    """Simplify LZCOUNT of boolean results."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "lzcountshiftbool")

    def get_op_list(self) -> list[str]:
        return ["LZCOUNT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        outVn = op.output
        if outVn is None: return 0
        
        max_return = 8 * data.nodes[op.inputs[0]].size_bits
        if bin(max_return).count('1') != 1: return 0
        
        descendants = [cand for cand in data.ops if outVn in cand.inputs]
        for baseOp in descendants:
            if baseOp.opcode not in ("INT_RIGHT", "INT_SRIGHT"): continue
            
            vn1 = data.nodes[baseOp.inputs[1]]
            if vn1.kind != "const": continue
            
            shift = int(vn1.name, 16)
            if (max_return >> shift) == 1:
                newOp = PcodeOp(baseOp.addr, "INT_EQUAL", len(data.nodes), (op.inputs[0], len(data.nodes)))
                b = len(data.nodes)
                data.nodes.append(PcodeVarnode(b, "0x0", data.nodes[op.inputs[0]].size_bits, "const"))
                newOp.inputs = (op.inputs[0], b)
                
                eqResVn = len(data.nodes)
                data.nodes.append(PcodeVarnode(eqResVn, None, 1, "unique"))
                newOp.output = eqResVn
                data.ops.append(newOp)
                
                data.op_remove_input(baseOp, 1)
                
                if data.nodes[baseOp.output].size_bits == 1:
                    data.op_set_opcode(baseOp, "COPY")
                else:
                    data.op_set_opcode(baseOp, "INT_ZEXT")
                data.op_set_input(baseOp, eqResVn, 0)
                return 1
        return 0


class RuleFloatSign(Rule):
    """Converts INT_AND or INT_XOR matching float sign operations into FLOAT_ABS or FLOAT_NEG"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "floatsign")

    def get_op_list(self) -> list[str]:
        return ["FLOAT_INT2FLOAT", "FLOAT_ADD", "FLOAT_DIV", "FLOAT_MULT", "FLOAT_SUB", "FLOAT_SQRT", "FLOAT_EQUAL", "FLOAT_NOTEQUAL", "FLOAT_LESS", "FLOAT_LESSEQUAL", "FLOAT_TRUNC"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        from minighidra.typeop import TypeOp
        
        res = 0
        opc = op.opcode
        
        def process_input(in_slot: int) -> int:
            if op.inputs[in_slot] < 0: return 0
            vn = op.inputs[in_slot]
            signOp = None
            for dop in data.ops:
                if dop.output == vn:
                    signOp = dop
                    break
            if signOp:
                resCode = TypeOp.float_sign_manipulation(signOp, data)
                if resCode:
                    data.op_remove_input(signOp, 1)
                    data.op_set_opcode(signOp, resCode)
                    return 1
            return 0
            
        if opc != "FLOAT_INT2FLOAT":
            res |= process_input(0)
            if len(op.inputs) > 1:
                res |= process_input(1)
                
        # Float comparisons have bool outputs, so we check if op returns bool
        is_bool_output = opc in ("FLOAT_EQUAL", "FLOAT_NOTEQUAL", "FLOAT_LESS", "FLOAT_LESSEQUAL")
        if is_bool_output or opc == "FLOAT_TRUNC":
            return 1 if res else 0
            
        if not op.output: return res
        
        # Check descends
        descends = []
        for dop in data.ops:
            if op.output in dop.inputs:
                descends.append(dop)
                
        for readOp in descends:
            resCode = TypeOp.float_sign_manipulation(readOp, data)
            if resCode:
                data.op_remove_input(readOp, 1)
                data.op_set_opcode(readOp, resCode)
                res = 1
                
        return res

class RuleFloatSignCleanup(Rule):
    """Cleanup INT_AND or INT_XOR matching float sign operations on floats directly"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "floatsigncleanup")

    def get_op_list(self) -> list[str]:
        return ["INT_AND", "INT_XOR"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        from minighidra.typeop import TypeOp
        
        # We need to know if it's float type. In minighidra we check if it is explicitly used as float or derived from float.
        # This is a bit tricky without Type propagation. But floatSignManipulation handles the mask check.
        # We can just apply it and rely on the mask check to be specific enough, or check if the other input is float.
        
        # Let's just use floatSignManipulation and assume if it perfectly matches a float mask, we should convert it.
        # In reality, GHIDRA checks if output type is TYPE_FLOAT.
        
        # We'll do a loose check: if it perfectly matches float sign mask, convert it.
        resCode = TypeOp.float_sign_manipulation(op, data)
        if not resCode: return 0
        
        data.op_remove_input(op, 1)
        data.op_set_opcode(op, resCode)
        return 1


class RuleFloatRange(Rule):
    """Simplify combinations of float comparisons: (x < y) | (x == y) => x <= y"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "floatrange")

    def get_op_list(self) -> list[str]:
        return ["BOOL_AND", "BOOL_OR"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        vn1, vn2 = op.inputs[0], op.inputs[1]
        
        

class RuleTransformCpool(Rule):
    """This rule requires CPool resolution which isn't currently ported to the minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "transformcpool")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleAliasUpdate(Rule):
    """This rule requires alias collapsing operations not implemented in the minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "aliasupdate")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleMultiCollapse(Rule):
    """Collapses MULTIEQUAL operations where all branches produce the same varnode."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "multicollapse")

    def get_op_list(self) -> list[str]:
        return ["MULTIEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if not op.inputs: return 0
        
        # We only match absolute identical inputs or loop-backs to the output itself
        # Functional equality (value equivalence via identical AST expressions) is skipped for now.
        defcopyr = None
        for i, in_vn in enumerate(op.inputs):
            # Skip loop-backs
            if in_vn == op.output:
                continue
            if defcopyr is None:
                defcopyr = in_vn
            elif in_vn != defcopyr:
                return 0
                
        if defcopyr is None:
            return 0
            
        data.op_set_opcode(op, "COPY")
        while len(op.inputs) > 1:
            data.op_remove_input(op, 1)
        data.op_set_input(op, defcopyr, 0)
        return 1


class RuleCollapseConstants(Rule):
    """Evaluates and collapses operations where all inputs are constants."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "collapseconstants")

    def get_op_list(self) -> list[str]:
        return [
            "INT_ADD", "INT_SUB", "INT_MULT", "INT_DIV", "INT_SDIV", 
            "INT_REM", "INT_SREM", "INT_AND", "INT_OR", "INT_XOR", 
            "INT_LEFT", "INT_RIGHT", "INT_SRIGHT", 
            "INT_EQUAL", "INT_NOTEQUAL", "INT_LESS", "INT_LESSEQUAL", 
            "INT_SLESS", "INT_SLESSEQUAL",
            "BOOL_NEGATE", "BOOL_AND", "BOOL_OR", "BOOL_XOR", "INT_ZEXT", "INT_SEXT"
        ]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if not op.inputs: return 0
        
        # Check if all inputs are constant
        c_vals = []
        for inp in op.inputs:
            node = data.nodes[inp]
            if node.kind != "const":
                return 0
            if node.name is None:
                return 0
            c_vals.append(int(node.name, 16))
            
        out_node = data.nodes[op.output] if op.output is not None else None
        if not out_node: return 0
        
        out_sz = out_node.size_bits
        opc = op.opcode
        
        try:
            if opc == "INT_ADD":
                res = (c_vals[0] + c_vals[1])
            elif opc == "INT_SUB":
                res = (c_vals[0] - c_vals[1])
            elif opc == "INT_MULT":
                res = (c_vals[0] * c_vals[1])
            elif opc in {"INT_DIV", "INT_SDIV"}:
                if c_vals[1] == 0: return 0
                res = (c_vals[0] // c_vals[1]) # Ignoring sdiv sign for this simple stub
            elif opc in {"INT_REM", "INT_SREM"}:
                if c_vals[1] == 0: return 0
                res = (c_vals[0] % c_vals[1])
            elif opc in {"INT_AND", "BOOL_AND"}:
                res = (c_vals[0] & c_vals[1])
            elif opc in {"INT_OR", "BOOL_OR"}:
                res = (c_vals[0] | c_vals[1])
            elif opc in {"INT_XOR", "BOOL_XOR"}:
                res = (c_vals[0] ^ c_vals[1])
            elif opc == "INT_LEFT":
                res = (c_vals[0] << c_vals[1])
            elif opc == "INT_RIGHT":
                res = (c_vals[0] >> c_vals[1])
            elif opc == "INT_SRIGHT":
                res = (c_vals[0] >> c_vals[1])
            elif opc == "INT_EQUAL":
                res = 1 if c_vals[0] == c_vals[1] else 0
            elif opc == "INT_NOTEQUAL":
                res = 1 if c_vals[0] != c_vals[1] else 0
            elif opc == "INT_LESS":
                res = 1 if c_vals[0] < c_vals[1] else 0
            elif opc == "INT_LESSEQUAL":
                res = 1 if c_vals[0] <= c_vals[1] else 0
            elif opc == "INT_SLESS":
                res = 1 if c_vals[0] < c_vals[1] else 0
            elif opc == "INT_SLESSEQUAL":
                res = 1 if c_vals[0] <= c_vals[1] else 0
            elif opc == "BOOL_NEGATE":
                res = 1 if c_vals[0] == 0 else 0
            elif opc in {"INT_ZEXT", "INT_SEXT"}:
                res = c_vals[0]
            else:
                return 0
        except Exception:
            return 0
            
        mask = (1 << out_sz) - 1
        res &= mask
        
        new_cnode = len(data.nodes)
        data.nodes.append(PcodeVarnode(new_cnode, hex(res), out_sz, "const"))
        
        while len(op.inputs) > 1:
            data.op_remove_input(op, 1)
        data.op_set_input(op, new_cnode, 0)
        data.op_set_opcode(op, "COPY")
        
        return 1


















        
        









class RuleLoadVarnode(Rule):
    """This rule requires architecture Spacebase resolution which isn't currently ported."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "loadvarnode")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleStoreVarnode(Rule):
    """This rule requires architecture Spacebase resolution which isn't currently ported."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "storevarnode")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleShadowVar(Rule):
    """This rule is commented out in original Ghidra."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "shadowvar")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleSwitchSingle(Rule):
    """This rule requires JumpTable tracking which isn't currently ported to minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "switchsingle")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleThreeWayCompare(Rule):
    """This rule requires detectThreeWay analysis which isn't currently ported to minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "threewaycompare")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleUndistribute(Rule):
    """This rule requires undistribute functionality which is partially covered by BitUndistribute and BooleanUndistribute but the base RuleUndistribute logic isn't ported directly."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "undistribute")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleAndStructure(Rule):
    """This rule requires Type propagation (for pointer types/structs) which isn't currently ported to minighidra Rule engine."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "andstructure")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleShiftLess(Rule):
    """This rule handles shift tracking inside conditions but uses architecture bits missing right now."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "shiftless")
        self.disabled = True

    def get_op_list(self) -> list[str]:
        return []

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

        return 0


class RuleBoolZext(Rule):
    """Simplify boolean expressions of the form zext(V) * -1"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "boolzext")

    def get_op_list(self) -> list[str]:
        return ["INT_ZEXT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 1: return 0
        boolVn1 = op.inputs[0]
        if not data.is_boolean_value(boolVn1): return 0
        
        outvn = op.output
        if outvn is None: return 0
        
        descendants = [cand for cand in data.ops if outvn in cand.inputs]
        if len(descendants) != 1: return 0
        
        multop1 = descendants[0]
        if multop1.opcode != "INT_MULT" or len(multop1.inputs) != 2: return 0
        
        cmult1 = data.nodes[multop1.inputs[1]]
        if cmult1.kind != "const": return 0
        
        coeff = int(cmult1.name, 16)
        sz = data.nodes[multop1.output].size_bits if multop1.output is not None else 0
        mask = (1 << sz) - 1
        if coeff != mask: return 0
        
        if multop1.output is None: return 0
        action_desc = [cand for cand in data.ops if multop1.output in cand.inputs]
        if len(action_desc) != 1: return 0
        actionop = action_desc[0]
        
        opc = actionop.opcode
        if opc == "INT_ADD":
            cadd = data.nodes[actionop.inputs[1]]
            if cadd.kind != "const": return 0
            if int(cadd.name, 16) == 1:
                newop = PcodeOp(op.addr, "BOOL_NEGATE", len(data.nodes), (boolVn1,))
                vn = len(data.nodes)
                data.nodes.append(PcodeVarnode(vn, None, 1, "unique"))
                newop.output = vn
                data.ops.append(newop)
                
                data.op_set_input(op, vn, 0)
                data.op_remove_input(actionop, 1)
                data.op_set_opcode(actionop, "COPY")
                data.op_set_input(actionop, op.output, 0)
                return 1
            return 0
            
        elif opc in ("INT_EQUAL", "INT_NOTEQUAL"):
            cadd = data.nodes[actionop.inputs[1]]
            if cadd.kind != "const": return 0
            val = int(cadd.name, 16)
            if val == coeff: val = 1
            elif val != 0: return 0
            
            data.op_set_input(actionop, boolVn1, 0)
            cnew = len(data.nodes)
            data.nodes.append(PcodeVarnode(cnew, hex(val), 1, "const"))
            data.op_set_input(actionop, cnew, 1)
            return 1
            
        elif opc == "INT_AND":
            opc = "BOOL_AND"
        elif opc == "INT_OR":
            opc = "BOOL_OR"
        elif opc == "INT_XOR":
            opc = "BOOL_XOR"
        else:
            return 0
            
        other_in = actionop.inputs[1] if actionop.inputs[0] == multop1.output else actionop.inputs[0]
        multop2 = next((cand for cand in data.ops if cand.output == other_in), None)
        if not multop2 or multop2.opcode != "INT_MULT" or len(multop2.inputs) != 2: return 0
        
        cmult2 = data.nodes[multop2.inputs[1]]
        if cmult2.kind != "const" or int(cmult2.name, 16) != mask: return 0
        
        zextop2 = next((cand for cand in data.ops if cand.output == multop2.inputs[0]), None)
        if not zextop2 or zextop2.opcode != "INT_ZEXT" or len(zextop2.inputs) != 1: return 0
        
        boolVn2 = zextop2.inputs[0]
        if not data.is_boolean_value(boolVn2): return 0
        
        newop = PcodeOp(actionop.addr, opc, len(data.nodes), (boolVn1, boolVn2))
        newres = len(data.nodes)
        data.nodes.append(PcodeVarnode(newres, None, 1, "unique"))
        newop.output = newres
        data.ops.append(newop)
        
        newzext = PcodeOp(actionop.addr, "INT_ZEXT", len(data.nodes), (newres,))
        newzout = len(data.nodes)
        data.nodes.append(PcodeVarnode(newzout, None, sz, "unique"))
        newzext.output = newzout
        data.ops.append(newzext)
        
        data.op_set_opcode(actionop, "INT_MULT")
        data.op_set_input(actionop, newzout, 0)
        cnew = len(data.nodes)
        data.nodes.append(PcodeVarnode(cnew, hex(coeff), sz, "const"))
        data.op_set_input(actionop, cnew, 1)
        return 1


class RuleSignForm2(Rule):
    """Normalize sign extraction: sub(sext(V) * small, c) s>> 31 => V s>> 31"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "signform2")

    def get_op_list(self) -> list[str]:
        return ["INT_SRIGHT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        constVn = data.nodes[op.inputs[1]]
        if constVn.kind != "const": return 0
        inVn = op.inputs[0]
        sizeout = data.nodes[inVn].size_bits
        if int(constVn.name, 16) != sizeout - 1: return 0
        
        subOp = next((cand for cand in data.ops if cand.output == inVn), None)
        if not subOp or subOp.opcode != "SUBPIECE" or len(subOp.inputs) != 2: return 0
        
        cnode = data.nodes[subOp.inputs[1]]
        if cnode.kind != "const": return 0
        c = int(cnode.name, 16)
        
        multOut = subOp.inputs[0]
        multSize = data.nodes[multOut].size_bits
        if c * 8 + sizeout != multSize: return 0
        
        multOp = next((cand for cand in data.ops if cand.output == multOut), None)
        if not multOp or multOp.opcode != "INT_MULT" or len(multOp.inputs) != 2: return 0
        
        slot = -1
        sextOp = None
        for i in range(2):
            vn = multOp.inputs[i]
            c_op = next((cand for cand in data.ops if cand.output == vn), None)
            if c_op and c_op.opcode == "INT_SEXT" and len(c_op.inputs) == 1:
                slot = i
                sextOp = c_op
                break
        if slot == -1: return 0
        
        a = sextOp.inputs[0]
        if data.nodes[a].size_bits != sizeout: return 0
        
        otherVn = multOp.inputs[1 - slot]
        otherNode = data.nodes[otherVn]
        if otherNode.kind == "const":
            if int(otherNode.name, 16) > ((1 << sizeout) - 1): return 0
            if 2 * sizeout > multSize: return 0
        else:
            zextOp = next((cand for cand in data.ops if cand.output == otherVn), None)
            if not zextOp or zextOp.opcode != "INT_ZEXT" or len(zextOp.inputs) != 1: return 0
            if data.nodes[zextOp.inputs[0]].size_bits + sizeout > multSize: return 0
            
        data.op_set_input(op, a, 0)
        return 1










class RuleDumptyHumpLate(Rule):
    """Rule to collapse SUBPIECE of SUBPIECE into a single SUBPIECE."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "dumptyhumplate")

    def get_op_list(self) -> list[str]:
        return ["SUBPIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        if len(op.inputs) != 2: return 0
        vn = op.inputs[0]
        if vn < 0: return 0
        
        node = data.nodes[vn]
        # In Python minighidra we check if it is written.
        # But data.op_get_def will do that.
        piece_op = None
        for cand_op in data.ops:
            if cand_op.output == vn:
                piece_op = cand_op
                break
                
        if piece_op is None or piece_op.opcode != "SUBPIECE":
            return 0
            
        if len(piece_op.inputs) != 2: return 0
        
        base = piece_op.inputs[0]
        trunc1_vn = op.inputs[1]
        trunc2_vn = piece_op.inputs[1]
        
        if trunc1_vn < 0 or trunc2_vn < 0: return 0
        node1 = data.nodes[trunc1_vn]
        node2 = data.nodes[trunc2_vn]
        
        if node1.kind == "const" and node2.kind == "const":
            val1 = int(node1.name, 16)
            val2 = int(node2.name, 16)
            
            trunc = val1 + val2
            
            data.op_set_input(op, base, 0)
            if trunc != val1:
                from minighidra.paramid import PcodeVarnode
                new_cnode = len(data.nodes)
                data.nodes.append(PcodeVarnode(new_cnode, hex(trunc), node1.size_bits, "const"))
                data.op_set_input(op, new_cnode, 1)
                
            return 1
            
        return 0























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


class RuleSubvarAnd(Rule):
    """Perform SubVariableFlow analysis triggered by INT_AND"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "subvar_and")

    def get_op_list(self) -> list[str]:
        return ["INT_AND"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        from minighidra.subflow import SubvariableFlow
        if len(op.inputs) != 2: return 0
        in1 = data.nodes[op.inputs[1]]
        if in1.kind != "const": return 0
        
        vn = op.inputs[0]
        outvn = op.output
        if vn < 0 or outvn is None: return 0
        
        out_consume = getattr(data, 'get_consume', lambda x: 0)(outvn)
        val = int(in1.name, 16) if in1.name else 0
        if out_consume != val: return 0
        if (out_consume & 1) == 0: return 0
        
        cmask = 0
        if out_consume == 1:
            cmask = 1
        else:
            cmask = (1 << data.nodes[vn].size_bits) - 1
            cmask >>= 8
            while cmask != 0:
                if cmask == out_consume: break
                cmask >>= 8
                
        if cmask == 0: return 0
        
        # if op.getOut()->hasNoDescend() check omitted for simplicity
        
        subflow = SubvariableFlow(data, vn, cmask, False, False, False)
        if not subflow.do_trace(): return 0
        subflow.do_replacement()
        return 1

class RuleSubvarSubpiece(Rule):
    """Perform SubVariableFlow analysis triggered by SUBPIECE"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "subvar_subpiece")

    def get_op_list(self) -> list[str]:
        return ["SUBPIECE"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        from minighidra.subflow import SubvariableFlow
        if len(op.inputs) != 2: return 0
        vn = op.inputs[0]
        outvn = op.output
        if vn < 0 or outvn is None: return 0
        
        in1 = data.nodes[op.inputs[1]]
        if in1.kind != "const": return 0
        
        flowsize = data.nodes[outvn].size_bits // 8
        sa = int(in1.name, 16) if in1.name else 0
        
        if flowsize + sa > 8: return 0
        
        mask = ((1 << (flowsize * 8)) - 1) << (8 * sa)
        
        aggressive = False
        if not aggressive:
            consume = getattr(data, 'get_consume', lambda x: 0)(vn)
            if (consume & mask) != consume: return 0
            
        big = False
        if flowsize >= 8 and data.nodes[vn].kind == "input":
            count = sum(1 for dop in data.ops if vn in dop.inputs)
            if count == 1:
                big = True
                
        subflow = SubvariableFlow(data, vn, mask, aggressive, False, big)
        if not subflow.do_trace(): return 0
        subflow.do_replacement()
        return 1

class RuleSubvarCompZero(Rule):
    """Perform SubVariableFlow analysis triggered by testing of a single bit"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "subvar_compzero")

    def get_op_list(self) -> list[str]:
        return ["INT_EQUAL", "INT_NOTEQUAL"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        from minighidra.subflow import SubvariableFlow
        if len(op.inputs) != 2: return 0
        in1 = data.nodes[op.inputs[1]] if op.inputs[1] >= 0 else None
        if not in1 or in1.kind != "const": return 0
        
        vn = op.inputs[0]
        if vn < 0: return 0
        
        mask = getattr(data, 'get_nzmask', lambda x: 0)(vn)
        if mask == 0: return 0
        
        bitnum = -1
        for i in range(64):
            if (mask >> i) & 1:
                bitnum = i
                break
                
        if bitnum == -1: return 0
        if (mask >> bitnum) != 1: return 0
        
        val = int(in1.name, 16) if in1.name else 0
        if val != mask and val != 0: return 0
        
        subflow = SubvariableFlow(data, vn, mask, False, False, False)
        if not subflow.do_trace(): return 0
        subflow.do_replacement()
        return 1

class RuleSubvarShift(Rule):
    """Perform SubvariableFlow analysis triggered by INT_RIGHT"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "subvar_shift")

    def get_op_list(self) -> list[str]:
        return ["INT_RIGHT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        from minighidra.subflow import SubvariableFlow
        if len(op.inputs) != 2: return 0
        vn = op.inputs[0]
        if vn < 0 or data.nodes[vn].size_bits != 8: return 0
        
        in1 = data.nodes[op.inputs[1]] if op.inputs[1] >= 0 else None
        if not in1 or in1.kind != "const": return 0
        
        sa = int(in1.name, 16) if in1.name else 0
        mask = getattr(data, 'get_nzmask', lambda x: 0)(vn)
        if (mask >> sa) != 1: return 0
        
        mask = (mask >> sa) << sa
        
        subflow = SubvariableFlow(data, vn, mask, False, False, False)
        if not subflow.do_trace(): return 0
        subflow.do_replacement()
        return 1

class RuleSubvarZext(Rule):
    """Perform SubvariableFlow analysis triggered by INT_ZEXT"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "subvar_zext")

    def get_op_list(self) -> list[str]:
        return ["INT_ZEXT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        from minighidra.subflow import SubvariableFlow
        if len(op.inputs) != 1: return 0
        vn = op.output
        invn = op.inputs[0]
        if vn is None or invn < 0: return 0
        
        sz = data.nodes[invn].size_bits
        mask = (1 << sz) - 1
        
        subflow = SubvariableFlow(data, vn, mask, False, False, False)
        if not subflow.do_trace(): return 0
        subflow.do_replacement()
        return 1

class RuleSubvarSext(Rule):
    """Perform SubvariableFlow analysis triggered by INT_SEXT"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "subvar_sext")

    def get_op_list(self) -> list[str]:
        return ["INT_SEXT"]

    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        from minighidra.subflow import SubvariableFlow
        if len(op.inputs) != 1: return 0
        vn = op.output
        invn = op.inputs[0]
        if vn is None or invn < 0: return 0
        
        sz = data.nodes[invn].size_bits
        mask = (1 << sz) - 1
        
        subflow = SubvariableFlow(data, vn, mask, False, True, False)
        if not subflow.do_trace(): return 0
        subflow.do_replacement()
        return 1

class RuleStringCopy(Rule):
    """Stub for RuleStringCopy"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "stringcopy")
    def get_op_list(self) -> list[str]:
        return ["COPY"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleStringStore(Rule):
    """Stub for RuleStringStore"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "stringstore")
    def get_op_list(self) -> list[str]:
        return ["STORE"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleSplitLoad(Rule):
    """Stub for RuleSplitLoad"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "splitload")
    def get_op_list(self) -> list[str]:
        return ["LOAD"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleSplitStore(Rule):
    """Stub for RuleSplitStore"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "splitstore")
    def get_op_list(self) -> list[str]:
        return ["STORE"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleSplitCopy(Rule):
    """Stub for RuleSplitCopy"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "splitcopy")
    def get_op_list(self) -> list[str]:
        return ["COPY"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleSplitFlow(Rule):
    """Stub for RuleSplitFlow"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "splitflow")
    def get_op_list(self) -> list[str]:
        return []
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleGeneric(Rule):
    """Stub for RuleGeneric"""
    def __init__(self, group: str) -> None:
        super().__init__(group, "generic")
    def get_op_list(self) -> list[str]:
        return []
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleSubfloatConvert(Rule):
    """Stub for RuleSubfloatConvert"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "subfloat_convert")
    def get_op_list(self) -> list[str]:
        return ["FLOAT_FLOAT2FLOAT"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleBitFieldIn(Rule):
    """Stub for RuleBitFieldIn"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "bitfield_in")
    def get_op_list(self) -> list[str]:
        return ["INT_AND"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleBitFieldLoad(Rule):
    """Stub for RuleBitFieldLoad"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "bitfield_load")
    def get_op_list(self) -> list[str]:
        return ["LOAD"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleBitFieldOut(Rule):
    """Stub for RuleBitFieldOut"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "bitfield_out")
    def get_op_list(self) -> list[str]:
        return ["INT_LEFT", "INT_RIGHT", "INT_SRIGHT"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleBitFieldStore(Rule):
    """Stub for RuleBitFieldStore"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "bitfield_store")
    def get_op_list(self) -> list[str]:
        return ["STORE"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleInsertAbsorb(Rule):
    """Stub for RuleInsertAbsorb"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "insert_absorb")
    def get_op_list(self) -> list[str]:
        # Typically triggered by INSERT
        return ["CALLOTHER"] # or similar
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RulePullAbsorb(Rule):
    """Stub for RulePullAbsorb"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "pull_absorb")
    def get_op_list(self) -> list[str]:
        return ["CALLOTHER"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0


class RuleDoubleIn(Rule):
    """Stub for RuleDoubleIn"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "doublein")
    def get_op_list(self) -> list[str]:
        return ["SUBPIECE"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleDoubleOut(Rule):
    """Stub for RuleDoubleOut"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "doubleout")
    def get_op_list(self) -> list[str]:
        return ["PIECE"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleDoubleStore(Rule):
    """Stub for RuleDoubleStore"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "doublestore")
    def get_op_list(self) -> list[str]:
        return ["STORE"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0

class RuleOrPredicate(Rule):
    """Stub for RuleOrPredicate"""
    def __init__(self, group: str) -> None:
        super().__init__(group, 0, "orpredicate")
    def get_op_list(self) -> list[str]:
        return ["INT_OR", "INT_XOR"]
    def apply_op(self, op: PcodeOp, data: PcodeLikeGraph) -> int:
        return 0
