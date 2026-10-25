from __future__ import annotations
from typing import Any
import struct

from .paramid import PcodeLikeGraph, PcodeOp
from .analysis import Analyzer

class JumpTable:
    """Manager for full recovery of a single switch statement / jump table."""
    def __init__(self, data: PcodeLikeGraph, indop: PcodeOp) -> None:
        self.data = data
        self.indop = indop
        self.addresses: list[int] = []
        self.model: JumpBasic | None = None
        self.label = []
        self.loadpoints = []

    def recover_model(self, analyzer: Analyzer) -> bool:
        model = JumpBasic(self.data, self.indop)
        if model.recover_model(analyzer):
            self.model = model
            return True
        return False

    def build_addresses(self, analyzer: Analyzer) -> None:
        if not self.model: return
        self.addresses = self.model.build_addresses(analyzer)

    def is_labelled(self) -> bool:
        return len(self.addresses) > 0


class JumpBasic:
    """Basic jump model: index -> bounds check -> calculation -> BRANCHIND."""
    def __init__(self, data: PcodeLikeGraph, indop: PcodeOp) -> None:
        self.data = data
        self.indop = indop
        self.path_ops: list[PcodeOp] = []
        self.norm_vn: int | None = None
        self.max_cases: int = 0
        self.guard_op: PcodeOp | None = None

    def _trace_path(self) -> bool:
        """Trace backward from BRANCHIND to find the switch variable."""
        # Current node is the input to BRANCHIND
        if not self.indop.inputs: return False
        curr_vn = self.indop.inputs[0]
        
        while True:
            def_op = self.data.get_def(curr_vn)
            if not def_op:
                self.norm_vn = curr_vn
                break
            
            self.path_ops.append(def_op)
            
            if def_op.opcode in ("INT_ADD", "INT_MULT", "PTRADD", "INT_ZEXT", "INT_SEXT", "INT_AND"):
                # Pick the non-constant side
                non_consts = [vn for vn in def_op.inputs if not self.data.is_constant(vn)]
                if len(non_consts) == 1:
                    curr_vn = non_consts[0]
                    continue
                elif len(non_consts) == 2:
                    # Prefer the one with a guard or smaller size? We just pick 0 for now.
                    # In a full pathmeld we trace all, but for JumpBasic linear trace is often enough.
                    curr_vn = non_consts[0]
                    continue
                else:
                    self.norm_vn = curr_vn
                    break
            elif def_op.opcode == "LOAD":
                # We stop at the load output, since the index is used to load from the table.
                # Actually, the index is used to compute the address for the load!
                # So we continue tracing into the pointer calculation.
                curr_vn = def_op.inputs[1]
                continue
            elif def_op.opcode in ("COPY", "CAST"):
                curr_vn = def_op.inputs[0]
                continue
            else:
                self.norm_vn = curr_vn
                break
                
        return self.norm_vn is not None

    def _find_guard(self) -> bool:
        """Find CBRANCH guarding the switch variable."""
        if self.norm_vn is None: return False
        
        # Look at uses of the switch variable
        descend = self.data.nodes[self.norm_vn]
        # In minighidra, we don't store descend on nodes. We have to iterate over all ops or use def-use chains.
        # Let's iterate all ops that use norm_vn
        uses = self.data.get_uses(self.norm_vn)
                
        for use_op in uses:
            if use_op.opcode in ("INT_LESS", "INT_SLESS", "INT_LESSEQUAL", "INT_SLESSEQUAL"):
                if self.data.is_constant(use_op.inputs[1]):
                    limit = self.data.get_constant_value(use_op.inputs[1])
                    if use_op.opcode.endswith("EQUAL"): limit += 1
                    
                    # Ensure limit is reasonable
                    if 0 < limit <= 1024:
                        self.max_cases = limit
                        self.guard_op = use_op
                        return True
                        
        return False

    def recover_model(self, analyzer: Analyzer) -> bool:
        if not self._trace_path(): return False
        if not self._find_guard(): return False
        return True

    def build_addresses(self, analyzer: Analyzer) -> list[int]:
        """Emulate the path for each case index to recover addresses."""
        addresses = []
        # Reverse path ops to execute from root to indop
        exec_path = list(reversed(self.path_ops))
        
        for case_idx in range(self.max_cases):
            env = {self.norm_vn: case_idx}
            addr = self._emulate(exec_path, env, analyzer)
            if addr is not None:
                addresses.append(addr)
                
        return addresses
        
    def _emulate(self, path: list[PcodeOp], env: dict[int, int], analyzer: Analyzer) -> int | None:
        def get_val(vn: int) -> int:
            if self.data.is_constant(vn): return self.data.get_constant_value(vn)
            return env.get(vn, 0)
            
        for op in path:
            out_vn = op.output
            if out_vn is None: continue
            
            if op.opcode == "INT_ADD":
                env[out_vn] = get_val(op.inputs[0]) + get_val(op.inputs[1])
            elif op.opcode == "INT_MULT":
                env[out_vn] = get_val(op.inputs[0]) * get_val(op.inputs[1])
            elif op.opcode == "PTRADD":
                env[out_vn] = get_val(op.inputs[0]) + get_val(op.inputs[1]) * get_val(op.inputs[2])
            elif op.opcode == "INT_ZEXT" or op.opcode == "INT_SEXT":
                env[out_vn] = get_val(op.inputs[0])
            elif op.opcode == "INT_AND":
                env[out_vn] = get_val(op.inputs[0]) & get_val(op.inputs[1])
            elif op.opcode == "COPY" or op.opcode == "CAST":
                env[out_vn] = get_val(op.inputs[0])
            elif op.opcode == "LOAD":
                addr = get_val(op.inputs[1])
                # Read from analyzer.image
                sz = self.data.nodes[out_vn].size_bits // 8
                try:
                    data = analyzer.image.read_memory(addr, sz)
                    env[out_vn] = int.from_bytes(data, "little")
                except Exception:
                    return None
            else:
                return None
                
            # Truncate to size
            sz_mask = (1 << self.data.nodes[out_vn].size_bits) - 1
            env[out_vn] &= sz_mask
            
        return env.get(self.indop.inputs[0])

