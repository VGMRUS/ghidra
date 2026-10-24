from typing import Optional, List, Tuple
from minighidra.model import PcodeOp, PcodeVarnode, PcodeLikeGraph

class AdditiveEdge:
    def __init__(self, op: PcodeOp, slot: int, multop: Optional[PcodeOp]):
        self.op = op
        self.slot = slot
        self.multop = multop

    @property
    def varnode(self) -> int:
        return self.op.inputs[self.slot]

    @property
    def multiplier(self) -> Optional[PcodeOp]:
        return self.multop

class TermOrder:
    def __init__(self, root: PcodeOp, data: PcodeLikeGraph):
        self.root = root
        self.data = data
        self.terms: List[AdditiveEdge] = []
        self.sorter: List[AdditiveEdge] = []

    def _get_def(self, vn_id: int) -> Optional[PcodeOp]:
        for op in self.data.ops:
            if op.output == vn_id:
                return op
        return None

    def _lone_descend(self, vn_id: int) -> Optional[PcodeOp]:
        count = 0
        lone_op = None
        for op in self.data.ops:
            if vn_id in op.inputs:
                count += 1
                lone_op = op
        return lone_op if count == 1 else None

    def collect(self) -> None:
        opstack = [self.root]
        multstack = [None]
        
        while opstack:
            curop = opstack.pop()
            multop = multstack.pop()
            
            for i, vn_id in enumerate(curop.inputs):
                if vn_id < 0:
                    continue
                node = self.data.nodes[vn_id]
                subop = self._get_def(vn_id)
                
                # If not defined by another op, or has more than one use
                if not subop or self._lone_descend(vn_id) is None:
                    self.terms.append(AdditiveEdge(curop, i, multop))
                    continue
                    
                if subop.opcode != "INT_ADD":
                    if subop.opcode == "INT_MULT" and len(subop.inputs) == 2:
                        mult_in1 = self.data.nodes[subop.inputs[1]] if subop.inputs[1] >= 0 else None
                        if mult_in1 and mult_in1.kind == "const":
                            addop = self._get_def(subop.inputs[0])
                            if addop and addop.opcode == "INT_ADD":
                                if self._lone_descend(addop.output) is not None:
                                    opstack.append(addop)
                                    multstack.append(subop)
                                    continue
                    self.terms.append(AdditiveEdge(curop, i, multop))
                else:
                    opstack.append(subop)
                    multstack.append(multop)
                    
    def sort_terms(self) -> None:
        self.sorter = list(self.terms)
        
        def term_key(e: AdditiveEdge) -> Tuple[int, int]:
            node = self.data.nodes[e.varnode]
            if node.kind == "const":
                val = int(node.name, 16) if node.name else 0
                return (0, val)
            return (1, e.varnode)
            
        self.sorter.sort(key=term_key)


class AddTerm:
    def __init__(self, vn: int = -1, coeff: int = 0):
        self.vn = vn
        self.coeff = coeff

    def is_equivalent(self, op2: 'AddTerm') -> bool:
        if self.vn != op2.vn: return False
        return self.coeff == op2.coeff

class AddExpression:
    def __init__(self, data: PcodeLikeGraph):
        self.data = data
        self.constval: int = 0
        self.terms: List[AddTerm] = []

    def _get_def(self, vn_id: int) -> Optional[PcodeOp]:
        for op in self.data.ops:
            if op.output == vn_id:
                return op
        return None

    def add(self, vn: int, coeff: int) -> None:
        if len(self.terms) < 2:
            self.terms.append(AddTerm(vn, coeff))

    def gather(self, vn_id: int, coeff: int, depth: int) -> None:
        if vn_id < 0: return
        node = self.data.nodes[vn_id]
        if node.kind == "const":
            val = int(node.name, 16) if node.name else 0
            sz = node.size_bits
            self.constval = (self.constval + coeff * val) & ((1 << sz) - 1)
            return

        def_op = self._get_def(vn_id)
        if def_op:
            if def_op.opcode == "INT_ADD":
                in1 = def_op.inputs[1]
                in1_node = self.data.nodes[in1] if in1 >= 0 else None
                if not in1_node or in1_node.kind != "const":
                    depth -= 1
                if depth >= 0:
                    self.gather(def_op.inputs[0], coeff, depth)
                    self.gather(def_op.inputs[1], coeff, depth)
                    return
            elif def_op.opcode == "INT_MULT" and len(def_op.inputs) == 2:
                in1 = def_op.inputs[1]
                in1_node = self.data.nodes[in1] if in1 >= 0 else None
                if in1_node and in1_node.kind == "const":
                    val = int(in1_node.name, 16) if in1_node.name else 0
                    sz = node.size_bits
                    new_coeff = (coeff * val) & ((1 << sz) - 1)
                    self.gather(def_op.inputs[0], new_coeff, depth)
                    return

        self.add(vn_id, coeff)

    def gather_two_terms_subtract(self, a: int, b: int) -> None:
        a_node = self.data.nodes[a] if a >= 0 else None
        b_node = self.data.nodes[b] if b >= 0 else None
        a_is_const = a_node and a_node.kind == "const"
        b_is_const = b_node and b_node.kind == "const"
        depth = 1 if (a_is_const or b_is_const) else 0
        
        self.gather(a, 1, depth)
        b_sz = b_node.size_bits if b_node else 32
        mask = (1 << b_sz) - 1
        self.gather(b, mask, depth)

    def gather_two_terms_add(self, a: int, b: int) -> None:
        a_node = self.data.nodes[a] if a >= 0 else None
        b_node = self.data.nodes[b] if b >= 0 else None
        a_is_const = a_node and a_node.kind == "const"
        b_is_const = b_node and b_node.kind == "const"
        depth = 1 if (a_is_const or b_is_const) else 0
        
        self.gather(a, 1, depth)
        self.gather(b, 1, depth)

    def gather_two_terms_root(self, root: int) -> None:
        self.gather(root, 1, 1)

    def is_equivalent(self, op2: 'AddExpression') -> bool:
        if self.constval != op2.constval: return False
        if len(self.terms) != len(op2.terms): return False
        
        if len(self.terms) == 0: return True
        if len(self.terms) == 1:
            return self.terms[0].is_equivalent(op2.terms[0])
            
        if self.terms[0].is_equivalent(op2.terms[0]) and self.terms[1].is_equivalent(op2.terms[1]):
            return True
        if self.terms[0].is_equivalent(op2.terms[1]) and self.terms[1].is_equivalent(op2.terms[0]):
            return True
            
        return False
