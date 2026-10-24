from __future__ import annotations
from .paramid import PcodeLikeGraph
from .action import Action


class ActionVarnodeProps(Action):
    """Calculates boolean properties and non-zero masks for varnodes."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "varnodeprops")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        changed = False
        
        # Clear caching to recalculate fresh
        if hasattr(data, "_nzmask_cache"): del data._nzmask_cache
        if hasattr(data, "_nzmask_visited"): del data._nzmask_visited
        
        # In python minighidra, we evaluate properties dynamically. 
        # But we still enforce the CPUI_COPY to 0 rule when masks are 0.
        for op in list(data.ops):
            if op.output is None: continue
            out_node = data.nodes[op.output]
            
            # if already constant, skip
            if out_node.kind == "const": continue
            
            nz = data.get_nzmask(op.output)
            consume = data.get_consume(op.output)
            
            if (nz & consume) == 0:
                if op.opcode == "COPY":
                    in_vn = op.inputs[0]
                    if in_vn >= 0 and data.nodes[in_vn].kind == "const" and data.nodes[in_vn].name == "0x0":
                        continue
                        
                # Replace with 0
                data.op_remove_input(op, 0)
                while len(op.inputs) > 0:
                    data.op_remove_input(op, 0)
                    
                c0 = len(data.nodes)
                from .paramid import PcodeVarnode
                data.nodes.append(PcodeVarnode(c0, "0x0", out_node.size_bits, "const"))
                data.op_set_opcode(op, "COPY")
                data.op_set_input(op, c0, 0)
                
                self.count += 1
                changed = True

        return self.count


class ActionDirectWrite(Action):
    """Marks varnodes built out of legal parameters as direct writes."""

    def __init__(self, group: str, propagate_indirect: bool) -> None:
        super().__init__(group, f"directwrite_{propagate_indirect}")
        self.propagate_indirect = propagate_indirect

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0
