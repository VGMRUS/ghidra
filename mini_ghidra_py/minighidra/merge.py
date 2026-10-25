from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

from .high import HighVariable
from .paramid import PcodeLikeGraph, PcodeOp


@dataclass(frozen=True, slots=True)
class VariablePiece:
    """Byte range of a Varnode within a joined logical variable."""

    group: Hashable
    offset: int
    size: int

    def __post_init__(self) -> None:
        if self.offset < 0 or self.size <= 0:
            raise ValueError("VariablePiece necesita offset >= 0 y tamaño positivo")


class Merge:
    """Manage conservative logical-variable groups over the p-code graph.

    This is deliberately smaller than Ghidra's cover/interference analysis,
    but it preserves the key safety property for copy aliases: a COPY is merged
    only when the source's last use is the copy itself, so the source and the
    copied value do not need two simultaneously-live names.
    """

    _NON_ARITHMETIC = frozenset(
        {"COPY", "MULTIEQUAL", "INDIRECT", "PIECE", "SUBPIECE"}
    )

    def __init__(self, data: PcodeLikeGraph) -> None:
        self.data = data
        self.highs: list[HighVariable] = []
        self.node_to_high: dict[int, HighVariable] = {}
        self._next_high_id = 1

    def new_high(self) -> HighVariable:
        high = HighVariable(self._next_high_id)
        self._next_high_id += 1
        self.highs.append(high)
        return high

    def assign_highs(self) -> None:
        """Give every value an initial HighVariable if it does not have one."""
        for node_id in range(len(self.data.nodes)):
            if node_id not in self.node_to_high:
                high = self.new_high()
                high.add_instance(node_id)
                high.data_type = self.data.get_node_type(node_id)
                self.node_to_high[node_id] = high

    def _merge_highs(self, target: HighVariable, source: HighVariable) -> None:
        if target is source:
            return
        for node_id in source.instances:
            target.add_instance(node_id)
            self.node_to_high[node_id] = target
        if target.data_type is None:
            target.data_type = source.data_type
        if target.name is None:
            target.name = source.name
        if source in self.highs:
            self.highs.remove(source)

    def _can_merge_copy(self, operation: PcodeOp, index: int) -> bool:
        if operation.output is None or len(operation.inputs) != 1:
            return False
        source_id = operation.inputs[0]
        if source_id < 0 or source_id >= len(self.data.nodes):
            return False
        if self.data.is_constant(source_id):
            return False
        source = self.data.nodes[source_id]
        output = self.data.nodes[operation.output]
        if source.size_bits != output.size_bits:
            return False
        source_type = self.data.get_node_type(source_id)
        output_type = self.data.get_node_type(operation.output)
        if (
            source_type is not None
            and output_type is not None
            and source_type.size != output_type.size
        ):
            return False
        use_indices = [
            use_index
            for use_index, use in enumerate(self.data.ops)
            if source_id in use.inputs
        ]
        return not use_indices or max(use_indices) <= index

    def merge_op(self, opcode: str) -> None:
        """Merge operation values when the available evidence is safe."""
        for index, operation in enumerate(tuple(self.data.ops)):
            if operation.opcode != opcode or operation.output is None:
                continue
            if opcode == "COPY" and not self._can_merge_copy(operation, index):
                continue
            output_high = self.node_to_high.get(operation.output)
            if output_high is None:
                continue
            output_node = self.data.nodes[operation.output]
            for input_id in operation.inputs:
                if input_id < 0 or input_id >= len(self.data.nodes):
                    continue
                if self.data.is_constant(input_id):
                    continue
                if self.data.nodes[input_id].size_bits != output_node.size_bits:
                    continue
                input_type = self.data.get_node_type(input_id)
                output_type = self.data.get_node_type(operation.output)
                if (
                    input_type is not None
                    and output_type is not None
                    and input_type.size != output_type.size
                ):
                    continue
                input_high = self.node_to_high.get(input_id)
                if input_high is not None:
                    self._merge_highs(output_high, input_high)

    def _mark_non_printing(self, operation: PcodeOp, reason: str) -> int:
        if operation.non_printing:
            return 0
        self.data.op_mark_non_printing(operation, reason)
        return 1

    def _node_piece_values(self, node_id: int) -> tuple[VariablePiece, ...]:
        stored = self.data.node_pieces.get(node_id)
        if isinstance(stored, VariablePiece):
            return (stored,)
        if isinstance(stored, dict):
            return tuple(
                piece
                for piece in stored.values()
                if isinstance(piece, VariablePiece)
            )
        return ()

    def _piece_is_internal(self, operation: PcodeOp) -> bool:
        output_id = operation.output
        if output_id is None or not operation.inputs:
            return False
        output_pieces = self._node_piece_values(output_id)
        if not output_pieces:
            return False

        if operation.opcode == "SUBPIECE":
            source_id = operation.inputs[0]
            source_pieces = self._node_piece_values(source_id)
            if not source_pieces:
                return False
            raw_offset = (
                self.data.get_constant_value(operation.inputs[1])
                if len(operation.inputs) > 1 and self.data.is_constant(operation.inputs[1])
                else 0
            )
            for output_piece in output_pieces:
                for source_piece in source_pieces:
                    if source_piece.group != output_piece.group:
                        continue
                    byte_offset = raw_offset
                    if self.data.big_endian:
                        byte_offset = source_piece.size - output_piece.size - byte_offset
                    if (
                        output_piece.offset == source_piece.offset + byte_offset
                        and output_piece.size
                        == self.data.nodes[output_id].size_bits // 8
                    ):
                        return True
            return False

        if operation.opcode == "PIECE" and len(operation.inputs) == 2:
            high_pieces = self._node_piece_values(operation.inputs[0])
            low_pieces = self._node_piece_values(operation.inputs[1])
            for output_piece in output_pieces:
                for high_piece in high_pieces:
                    for low_piece in low_pieces:
                        if (
                            high_piece.group != output_piece.group
                            or low_piece.group != output_piece.group
                        ):
                            continue
                        if self.data.big_endian:
                            matches = (
                                high_piece.offset == output_piece.offset
                                and low_piece.offset
                                == output_piece.offset + high_piece.size
                            )
                        else:
                            matches = (
                                low_piece.offset == output_piece.offset
                                and high_piece.offset
                                == output_piece.offset + low_piece.size
                            )
                        if (
                            matches
                            and output_piece.size
                            == high_piece.size + low_piece.size
                        ):
                            return True
        return False

    def _copy_is_internal_piece(self, operation: PcodeOp) -> bool:
        if operation.output is None or len(operation.inputs) != 1:
            return False
        source_pieces = self._node_piece_values(operation.inputs[0])
        output_pieces = self._node_piece_values(operation.output)
        for source_piece in source_pieces:
            for output_piece in output_pieces:
                if (
                    source_piece.group == output_piece.group
                    and source_piece.offset == output_piece.offset
                    and source_piece.size == output_piece.size
                    and source_piece.size
                    == self.data.nodes[operation.output].size_bits // 8
                ):
                    return True
        return False

    def mark_internal_copies(self) -> int:
        """Mark internal COPY/PIECE/SUBPIECE ops as non-printing."""
        changed = 0
        for operation in tuple(self.data.ops):
            if operation.opcode == "COPY" and operation.output is not None:
                output_high = self.node_to_high.get(operation.output)
                source_high = (
                    self.node_to_high.get(operation.inputs[0])
                    if len(operation.inputs) == 1
                    else None
                )
                if output_high is not None and output_high is source_high:
                    changed += self._mark_non_printing(operation, "internal-copy")
                elif self._copy_is_internal_piece(operation):
                    changed += self._mark_non_printing(
                        operation, "joined-variable-piece"
                    )
                elif (
                    len(operation.inputs) == 1
                    and not self.data.get_uses(operation.output)
                    and self.data.get_node_type(operation.output)
                    is not None
                    and self.data.get_node_type(operation.output)
                    == self.data.get_node_type(operation.inputs[0])
                ):
                    # An unused exact type-preserving shadow has no independent
                    # source-level value even when no HighVariable merge ran.
                    changed += self._mark_non_printing(operation, "unused-shadow-copy")
            elif operation.opcode in {"PIECE", "SUBPIECE"}:
                if self._piece_is_internal(operation):
                    changed += self._mark_non_printing(operation, "joined-variable-piece")
        return changed

    def mark_pure_padding_copies(self) -> int:
        """Hide zero-initializations of nodes declared as pure join padding."""
        changed = 0
        for node_id in sorted(self.data.padding_nodes):
            operation = self.data.get_def(node_id)
            if (
                operation is not None
                and operation.opcode == "COPY"
                and len(operation.inputs) == 1
                and self.data.is_constant(operation.inputs[0])
                and self.data.get_constant_value(operation.inputs[0]) == 0
            ):
                changed += self._mark_non_printing(operation, "pure-join-padding")
        return changed

    def merge_adjacent(self) -> None:
        """Placeholder for adjacent-storage high-variable merging."""
        return None

    def mark_explicit_types(self) -> None:
        return None

    def mark_implied_types(self) -> None:
        return None
