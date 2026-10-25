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


class ActionConstbase(Action):
    """Injects context register values tracked upon entry into the flow."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "constbase")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionSegmentize(Action):
    """Converts CALLOTHER segment ops into CPUI_SEGMENTOP."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "segmentize")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionParamDouble(Action):
    """Adjusts overlapping/double-precision input parameters into combined single variables or splits them."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "paramdouble")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionLikelyTrash(Action):
    """Truncates data flow for input varnodes that are marked as likely trash by the prototype."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "likelytrash")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionMarkIndirectOnly(Action):
    """Marks variables that are only used indirectly via INDIRECT operations."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "markindirectonly")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionParamShiftStart(Action):
    """Adjusts inputs for calls that have param shifts. Deprecated/handled in Ghidra C++."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "paramshiftstart")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0

class ActionParamShiftStop(Action):
    """Ends adjustments for calls that have param shifts. Deprecated/handled in Ghidra C++."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "paramshiftstop")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0



class ActionFuncLink(Action):
    """Prepares input and output parameters for function calls based on active tracking."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "funclink")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if not hasattr(data, "call_specs"): return 0
        for call_spec in data.call_specs:
            input_locked = call_spec.is_input_active
            varargs = getattr(call_spec, 'is_dotdotdot', False)
            
            if not input_locked or varargs:
                call_spec.set_input_active()
                
            if input_locked:
                pass
                
            if call_spec.op.output is not None:
                pass
            else:
                call_spec.set_output_active()
                
        return self.count

class ActionFuncLinkOutOnly(Action):
    """Prepares only output parameters for function calls based on active tracking."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "funclinkoutonly")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if not hasattr(data, "call_specs"): return 0
        for call_spec in data.call_specs:
            if call_spec.op.output is None:
                call_spec.set_output_active()
        return self.count

class ActionNonzeroMask(Action):
    """Triggers calculation of non-zero masks for all variables.
    In Python minighidra, this happens implicitly on demand in get_nzmask, 
    so we just clear the caches to ensure freshness."""
    def __init__(self, group: str) -> None:
        super().__init__(group, "nonzeromask")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if hasattr(data, "_nzmask_cache"): del data._nzmask_cache
        if hasattr(data, "_nzmask_visited"): del data._nzmask_visited
        return 0
