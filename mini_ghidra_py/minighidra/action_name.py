from __future__ import annotations

from .action import Action
from .merge import Merge
from .paramid import PcodeLikeGraph, PcodeOp


class ActionNameVars(Action):
    """Assign stable C names to merged high variables and all their aliases."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "namevars")

    @staticmethod
    def _preferred_name(data: PcodeLikeGraph, instances: list[int]) -> str | None:
        parameter_names = sorted(
            {
                name
                for node_id in instances
                for name in data._node_parameter_names.get(node_id, ())
            }
        )
        if parameter_names:
            return parameter_names[0]

        candidates: list[tuple[int, str]] = []
        for node_id in instances:
            if node_id < 0 or node_id >= len(data.nodes):
                continue
            node = data.nodes[node_id]
            name = data.node_names.get(node_id, node.name)
            if not name or name.startswith(("const_", "split_", "unique_")):
                continue
            if node.kind in {"constant", "const", "temporary"}:
                continue
            if name.startswith(("param_", "arg_", "stack_arg_", "home_slot_")):
                priority = 0
            elif name.startswith(("stack_", "local_", "var_")):
                priority = 1
            elif node.kind == "input":
                priority = 2
            else:
                priority = 3
            candidates.append((priority, name))
        if not candidates:
            return None
        return min(candidates)[1]

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        merge = getattr(data, "merge", None)
        if merge is None:
            merge = Merge(data)
            data.merge = merge  # type: ignore[attr-defined]
        if len(merge.node_to_high) < len(data.nodes):
            merge.assign_highs()
        for high in merge.highs:
            if high.name is None:
                high.name = self._preferred_name(data, high.instances)
                if high.name is None:
                    high.name = f"var_{high.high_id}"
                self.count += 1
            for node_id in high.instances:
                data.node_names[node_id] = high.name
        for alias, root in data.shadow_aliases.items():
            if alias < 0 or alias >= len(data.nodes):
                continue
            root_name = data.get_node_name(root)
            if root_name is not None and data.node_names.get(alias) != root_name:
                data.node_names[alias] = root_name
                self.count += 1
        return self.count


class ActionDynamicSymbols(Action):
    """Links explicitly named dynamic symbols to variables in the graph."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "dynamicsymbols")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionRestructureVarnode(Action):
    """Re-synchronizes varnodes with symbol scope."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "restructurevarnode")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if hasattr(data, "sync_varnodes_with_symbols"):
            if data.sync_varnodes_with_symbols(False, True):
                self.count += 1
        return 0


class ActionMappedLocalSync(Action):
    """Synchronizes mapped locals with their underlying varnodes."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "mappedlocalsync")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if hasattr(data, "sync_varnodes_with_symbols"):
            if data.sync_varnodes_with_symbols(True, True):
                self.count += 1
        return 0


class ActionMapGlobals(Action):
    """Maps global variables based on references discovered during analysis."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "mapglobals")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        if hasattr(data, "map_globals"):
            if data.map_globals():
                self.count += 1
        return 0


class ActionSpacebase(Action):
    """Establishes spacebase references (like stack or register tracking)."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "spacebase")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        SPACEBASE_NAMES = {"RSP", "ESP", "SP", "RBP", "EBP", "BP"}
        
        for vn, node in enumerate(data.nodes):
            if not node: continue
            name_upper = (node.name or "").upper()
            
            # If the node represents a known spacebase or spacebase_in
            if any(name_upper == base or name_upper.startswith(base + "_IN") for base in SPACEBASE_NAMES):
                object.__setattr__(node, "is_spacebase", True)
                
                def_op = data.get_def(vn)
                if def_op is not None and def_op.opcode == "INT_ADD":
                    uses = data.get_uses(vn)
                    if len(uses) > 1:
                        # Split uses: clone the ADD operation for each extra use
                        for use_op in uses[1:]:
                            new_vn = data._new_temporary_node(f"{node.name}_split", node.size_bits)
                            object.__setattr__(data.nodes[new_vn], "is_spacebase", True)
                            
                            new_add = PcodeOp()
                            new_add.opcode = "INT_ADD"
                            new_add.inputs = tuple(def_op.inputs)
                            new_add.address = def_op.address
                            new_add.output = new_vn
                            data._append_operation(new_add)
                            
                            # Update the use
                            idx = list(use_op.inputs).index(vn)
                            data.op_set_input(use_op, new_vn, idx)
                        
                        self.count += 1
                        
        return 0


class ActionStackPtrFlow(Action):
    """Analyzes stack pointer flow and aligns stack usage across the function."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "stackptrflow")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionUnjustifiedParams(Action):
    """Adjusts unjustified input parameters to their proper ABI containers."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "unjustifiedparams")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0


class ActionHideShadow(Action):
    """Hides redundant variables overshadowed by others during merging."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "hideshadow")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        for alias, root in data.shadow_aliases.items():
            name = data.get_node_name(root)
            if name is not None:
                data.node_names[alias] = name
                self.count += 1
        return self.count


class ActionDynamicMapping(Action):
    """Resolves dynamically mapped symbols from hash tracking."""

    def __init__(self, group: str) -> None:
        super().__init__(group, "dynamicmapping")

    def apply(self, data: PcodeLikeGraph) -> int:
        self.count = 0
        return 0
