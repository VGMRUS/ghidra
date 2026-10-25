"""Type-directed splitting of aggregate p-code operations.

This module is a deliberately conservative Python port of the ``Datatype ->
SplitDatatype`` path in Ghidra's ``subflow.cc``.  It translates the parts that
can be represented by MiniGhidraPy's CType and PcodeLikeGraph models: composite
layout discovery, component compatibility, field-sized COPY/LOAD/STORE
rewrites, constant slicing, and pointer-offset construction.  It refuses
ambiguous or incomplete types instead of guessing at a memory layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .merge import VariablePiece
from .paramid import PcodeLikeGraph, PcodeOp
from .typesys import CType, TypeDatabase, TypeKind


@dataclass(frozen=True, slots=True)
class SplitPiece:
    """A pair of matching logical component types within an aggregate value."""

    input_type: CType
    output_type: CType
    offset: int
    is_padding: bool = False

    @property
    def size(self) -> int:
        return self.output_type.size or self.input_type.size or 0


class SplitDatatype:
    """Split COPY, LOAD and STORE operations according to registered C types."""

    _ARITHMETIC_OPS = frozenset(
        {
            "INT_ADD",
            "INT_SUB",
            "INT_MULT",
            "INT_DIV",
            "INT_SDIV",
            "INT_REM",
            "INT_SREM",
            "INT_AND",
            "INT_OR",
            "INT_XOR",
            "INT_LEFT",
            "INT_RIGHT",
            "INT_SRIGHT",
            "FLOAT_ADD",
            "FLOAT_SUB",
            "FLOAT_MULT",
            "FLOAT_DIV",
        }
    )

    def __init__(
        self,
        data: PcodeLikeGraph,
        type_database: TypeDatabase | None = None,
        *,
        split_structures: bool = True,
        split_arrays: bool = True,
    ) -> None:
        self.data = data
        self.types = type_database or getattr(data, "type_database", None)
        self.split_structures = split_structures
        self.split_arrays = split_arrays
        self.is_load_store = False
        self.data_type_pieces: list[SplitPiece] = []
        self.last_reason = ""

    @staticmethod
    def _size_bytes(data: PcodeLikeGraph, vn_id: int) -> int:
        if vn_id < 0 or vn_id >= len(data.nodes):
            return 0
        return (data.nodes[vn_id].size_bits + 7) // 8

    def _type(self, spec: Any) -> CType | None:
        if spec is None:
            return None
        if isinstance(spec, CType):
            return spec
        if self.types is None:
            return None
        try:
            return self.types.resolve(spec)
        except (KeyError, TypeError, ValueError):
            return None

    def _node_type(self, vn_id: int) -> CType | None:
        if vn_id < 0:
            return None
        return self._type(self.data.get_node_type(vn_id))

    def _sized_type(self, ctype: CType | None, size: int) -> CType | None:
        if ctype is None or size <= 0:
            return None
        if ctype.size == size:
            return ctype
        if self.types is None:
            return None
        if ctype.kind in {TypeKind.STRUCT, TypeKind.ARRAY, TypeKind.PARTIAL_STRUCT}:
            if ctype.size is None or size > ctype.size:
                return None
            try:
                return self.types.partial(ctype, size)
            except ValueError:
                return None
        if (
            ctype.kind in {
                TypeKind.BOOL,
                TypeKind.INTEGER,
                TypeKind.FLOAT,
                TypeKind.ENUM,
            }
            and ctype.size is not None
            and size > ctype.size
            and size % ctype.size == 0
        ):
            return self.types.array(ctype, size // ctype.size)
        return None

    def _undefined(self, size: int) -> CType:
        if self.types is not None:
            return self.types.undefined(size)
        return CType(f"undefined{size}", TypeKind.UNKNOWN, size, 1)

    def _category(self, ctype: CType | None) -> int:
        """Return -1 (unsplittable), 0 (struct), 1 (array), 2 (primitive)."""
        if ctype is None:
            return -1
        base = ctype
        if base.kind == TypeKind.PARTIAL_STRUCT:
            parent = base.parent or base.target
            if parent is None:
                return -1
            if parent.kind == TypeKind.ARRAY:
                base = parent
            elif parent.kind == TypeKind.STRUCT:
                return 0 if self.split_structures else -1
            else:
                return -1
        if base.kind == TypeKind.STRUCT:
            if not self.split_structures:
                return -1
            if self.types is None:
                return 0 if len(base.fields) > 1 else -1
            components = self.types.components(base)
            return 0 if len(components) > 1 else -1
        if base.kind == TypeKind.ARRAY:
            if not self.split_arrays or base.target is None:
                return -1
            target = base.target.unwrapped()
            if target.kind != TypeKind.UNKNOWN or target.size != 1:
                return 1
            return 2
        if base.kind in {TypeKind.INTEGER, TypeKind.UNKNOWN}:
            return 2
        return -1

    def test_datatype_compatibility(
        self,
        in_type: CType | None,
        out_type: CType | None,
        *,
        in_constant: bool = False,
        is_load_store: bool | None = None,
    ) -> tuple[SplitPiece, ...] | None:
        """Find matching subcomponents, rejecting type/layout mismatches."""
        self.data_type_pieces.clear()
        self.last_reason = ""
        if self.types is None or in_type is None or out_type is None:
            self.last_reason = "missing type database or operand datatype"
            return None
        byte_size = in_type.size
        if byte_size is None or byte_size <= 0 or out_type.size != byte_size:
            self.last_reason = "input/output datatype sizes do not match"
            return None
        in_category = self._category(in_type)
        out_category = self._category(out_type)
        if in_category < 0 or out_category < 0:
            self.last_reason = "one datatype is not splittable"
            return None
        if in_category == 2 and out_category == 2:
            self.last_reason = "both datatypes are primitive"
            return None
        load_store = self.is_load_store if is_load_store is None else is_load_store
        in_base = in_type.parent if in_type.kind == TypeKind.PARTIAL_STRUCT else in_type
        if (
            not load_store
            and not in_constant
            and in_type == out_type
            and in_base is not None
            and in_base.kind == TypeKind.STRUCT
        ):
            self.last_reason = "whole-structure copies are left intact"
            return None
        if load_store and out_category == 2 and in_category == 1:
            self.last_reason = "array-to-primitive memory access is ambiguous"
            return None
        if load_store and in_category == 2 and out_category == 1 and not in_constant:
            self.last_reason = "primitive-to-array memory access is ambiguous"
            return None
        if load_store and in_category == 1 and out_category == 1 and not in_constant:
            self.last_reason = "array-to-array memory copies are left intact"
            return None

        in_components = self.types.components(in_type, size=byte_size)
        out_components = self.types.components(out_type, size=byte_size)
        if not in_components or not out_components:
            self.last_reason = "composite layout is incomplete"
            return None
        in_by_offset = {piece.offset: piece for piece in in_components}
        out_by_offset = {piece.offset: piece for piece in out_components}
        pieces: list[SplitPiece] = []
        cursor = 0
        while cursor < byte_size:
            in_component = in_by_offset.get(cursor)
            out_component = out_by_offset.get(cursor)
            if in_category == 2:
                if out_component is None:
                    self.last_reason = "output has no component at byte offset"
                    return None
                span = out_component.size
                input_component_type = (
                    out_component.type if in_constant else self._undefined(span)
                )
                output_component_type = out_component.type
                hole = out_component.is_hole
            elif out_category == 2:
                if in_component is None:
                    self.last_reason = "input has no component at byte offset"
                    return None
                span = in_component.size
                input_component_type = in_component.type
                output_component_type = self._undefined(span)
                hole = in_component.is_hole
            else:
                if in_component is None or out_component is None:
                    self.last_reason = "datatype components do not align"
                    return None
                if in_component.size != out_component.size:
                    self.last_reason = "component widths do not match"
                    return None
                span = in_component.size
                input_component_type = in_component.type
                output_component_type = out_component.type
                hole = in_component.is_hole or out_component.is_hole
            if span <= 0 or cursor + span > byte_size:
                self.last_reason = "component extends beyond the value boundary"
                return None
            pieces.append(
                SplitPiece(
                    input_component_type,
                    output_component_type,
                    cursor,
                    is_padding=hole,
                )
            )
            if hole and (
                len(pieces) == 1
                or (cursor + span == byte_size and len(pieces) == 2)
            ):
                self.last_reason = "layout is likely only padding"
                return None
            cursor += span
        if len(pieces) <= 1:
            self.last_reason = "datatype has only one logical component"
            return None
        self.data_type_pieces = pieces
        return tuple(pieces)

    def _piece_group(self, aggregate_id: int) -> tuple[str, int, int]:
        return ("splitdatatype", id(self.data), aggregate_id)

    def _set_node_piece(
        self,
        node_id: int,
        group: tuple[str, int, int],
        offset: int,
        size: int,
        *,
        is_padding: bool = False,
    ) -> None:
        if node_id < 0 or node_id >= len(self.data.nodes) or size <= 0:
            return
        piece = VariablePiece(group, offset, size)
        existing = self.data.node_pieces.get(node_id)
        if isinstance(existing, VariablePiece):
            if existing.group == group:
                self.data.node_pieces[node_id] = piece
            else:
                self.data.node_pieces[node_id] = {
                    existing.group: existing,
                    group: piece,
                }
        elif isinstance(existing, dict):
            existing[group] = piece
        else:
            self.data.node_pieces[node_id] = piece
        if is_padding:
            self.data.padding_nodes.add(node_id)

    def _node_piece_values(self, node_id: int) -> tuple[VariablePiece, ...]:
        existing = self.data.node_pieces.get(node_id)
        if isinstance(existing, VariablePiece):
            return (existing,)
        if isinstance(existing, dict):
            return tuple(
                piece
                for piece in existing.values()
                if isinstance(piece, VariablePiece)
            )
        return ()

    def _set_piece_list(
        self,
        node_ids: list[int],
        pieces: tuple[SplitPiece, ...],
        group: tuple[str, int, int],
    ) -> None:
        for node_id, piece in zip(node_ids, pieces, strict=True):
            self._set_node_piece(
                node_id,
                group,
                piece.offset,
                piece.size,
                is_padding=piece.is_padding,
            )

    def _constant_pieces(
        self, source_id: int, pieces: tuple[SplitPiece, ...], total_size: int
    ) -> list[int]:
        base_value = self.data.get_constant_value(source_id)
        result: list[int] = []
        for piece in pieces:
            shift_offset = piece.offset
            if self.data.big_endian:
                shift_offset = total_size - piece.offset - piece.size
            mask = (1 << (piece.size * 8)) - 1
            value = (base_value >> (shift_offset * 8)) & mask
            result.append(self.data.new_constant(piece.size * 8, value))
        return result

    def _new_subpiece(
        self,
        source_id: int,
        piece: SplitPiece,
        total_size: int,
        address: int | None,
    ) -> int:
        if self._size_bytes(self.data, source_id) == piece.size and piece.offset == 0:
            return source_id
        byte_offset = piece.offset
        if self.data.big_endian:
            byte_offset = total_size - piece.offset - piece.size
        const_id = self.data.new_constant(32, byte_offset)
        output_id = self.data.new_unique(piece.size * 8, "split_in")
        self.data.set_node_type(output_id, piece.input_type)
        self.data.append_operation(
            PcodeOp(
                "SUBPIECE",
                (source_id, const_id),
                output_id,
                address=address,
                evidence="splitdatatype-input-piece",
            )
        )
        return output_id

    def _split_input(
        self,
        source_id: int,
        pieces: tuple[SplitPiece, ...],
        total_size: int,
        address: int | None,
    ) -> list[int]:
        if self.data.is_constant(source_id):
            values = self._constant_pieces(source_id, pieces, total_size)
            for value_id, piece in zip(values, pieces, strict=True):
                self.data.set_node_type(value_id, piece.input_type)
            return values
        return [
            self._new_subpiece(source_id, piece, total_size, address)
            for piece in pieces
        ]

    def _annotate_concat_output(
        self, output_id: int, input_ids: tuple[int, int]
    ) -> None:
        for first in self._node_piece_values(input_ids[0]):
            for second in self._node_piece_values(input_ids[1]):
                if first.group != second.group:
                    continue
                if not (
                    first.offset + first.size == second.offset
                    or second.offset + second.size == first.offset
                ):
                    continue
                self._set_node_piece(
                    output_id,
                    first.group,
                    min(first.offset, second.offset),
                    first.size + second.size,
                )
                return

    def _append_concat(
        self,
        piece_ids: list[int],
        final_output: int,
        address: int | None,
        result_type: CType,
    ) -> None:
        """Concatenate low-to-high byte pieces to preserve the original value."""
        if not piece_ids:
            return
        if len(piece_ids) == 1:
            source_id = piece_ids[0]
            if source_id != final_output:
                self.data.append_operation(
                    PcodeOp(
                        "COPY",
                        (source_id,),
                        final_output,
                        address=address,
                        evidence="splitdatatype-reassemble",
                    )
                )
            self.data.set_node_type(final_output, result_type)
            return
        if self.data.big_endian:
            current_id = piece_ids[0]
            for index, next_id in enumerate(piece_ids[1:], start=1):
                is_final = index == len(piece_ids) - 1
                output_id = final_output if is_final else self.data.new_unique(
                    sum(self._size_bytes(self.data, item) for item in piece_ids[: index + 1]) * 8,
                    "split_concat",
                )
                high_id, low_id = current_id, next_id
                self.data.append_operation(
                    PcodeOp(
                        "PIECE",
                        (high_id, low_id),
                        output_id,
                        address=address,
                        evidence="splitdatatype-reassemble",
                    )
                )
                self._annotate_concat_output(output_id, (high_id, low_id))
                if not is_final:
                    current_id = output_id
        else:
            current_id = piece_ids[-1]
            for index in range(len(piece_ids) - 2, -1, -1):
                is_final = index == 0
                output_id = final_output if is_final else self.data.new_unique(
                    sum(self._size_bytes(self.data, item) for item in piece_ids[index:]) * 8,
                    "split_concat",
                )
                self.data.append_operation(
                    PcodeOp(
                        "PIECE",
                        (current_id, piece_ids[index]),
                        output_id,
                        address=address,
                        evidence="splitdatatype-reassemble",
                    )
                )
                self._annotate_concat_output(
                    output_id, (current_id, piece_ids[index])
                )
                if not is_final:
                    current_id = output_id
        self.data.set_node_type(final_output, result_type)

    def _arithmetic_input(self, vn_id: int) -> bool:
        return any(
            use.opcode in self._ARITHMETIC_OPS
            for use in self.data.get_uses(vn_id)
        )

    def _arithmetic_output(self, vn_id: int) -> bool:
        definition = self.data.get_def(vn_id)
        return definition is not None and definition.opcode in self._ARITHMETIC_OPS

    def _input_indices(self, operation: PcodeOp, slot: int) -> tuple[int, ...]:
        if operation.input_slots:
            return tuple(
                value_id
                for value_id, input_slot in zip(
                    operation.inputs, operation.input_slots, strict=False
                )
                if input_slot == slot
            )
        if operation.opcode == "STORE":
            if len(operation.inputs) >= 3:
                index = 1 if slot == 1 else 2
            elif len(operation.inputs) == 2:
                index = 0 if slot == 1 else 1
            else:
                return ()
            return (operation.inputs[index],) if index < len(operation.inputs) else ()
        if operation.opcode == "LOAD":
            return (operation.inputs[0],) if slot == 1 and operation.inputs else ()
        return ()

    def _assemble_operand(
        self, operation: PcodeOp, slot: int, *, ctype: CType | None = None
    ) -> int | None:
        value_ids = self._input_indices(operation, slot)
        if not value_ids:
            return None
        if ctype is None:
            known_types = [self._node_type(value_id) for value_id in value_ids]
            typed = [candidate for candidate in known_types if candidate is not None]
            if typed and all(candidate == typed[0] for candidate in typed):
                ctype = typed[0]
            if ctype is None and self.types is not None:
                parameter_names = {
                    name
                    for value_id in value_ids
                    for name in self.data._node_parameter_names.get(value_id, ())
                }
                declared = {
                    self._type(self.data.parameter_types.get(name))
                    for name in parameter_names
                    if self.data.parameter_types.get(name) is not None
                }
                declared.discard(None)
                if len(declared) == 1:
                    ctype = next(iter(declared))
        if len(value_ids) == 1:
            if ctype is not None and self.data.get_node_type(value_ids[0]) is None:
                self.data.set_node_type(value_ids[0], ctype)
            return value_ids[0]
        # SSA heritage models register values as byte cells.  Reassemble those
        # cells into a value node before applying a field-sized transformation.
        ordered = list(value_ids)
        if self.data.big_endian:
            current = ordered[0]
            rest = ordered[1:]
            for index, item in enumerate(rest):
                size_bits = sum(self.data.nodes[vn].size_bits for vn in ordered[: index + 2])
                output_id = self.data.new_unique(size_bits, "split_operand")
                self.data.append_operation(
                    PcodeOp("PIECE", (current, item), output_id, address=operation.address)
                )
                current = output_id
        else:
            current = ordered[-1]
            rest = list(reversed(ordered[:-1]))
            for index, item in enumerate(rest):
                size_bits = sum(self.data.nodes[vn].size_bits for vn in ordered[-(index + 2) :])
                output_id = self.data.new_unique(size_bits, "split_operand")
                self.data.append_operation(
                    PcodeOp("PIECE", (current, item), output_id, address=operation.address)
                )
                current = output_id
        if ctype is not None:
            self.data.set_node_type(current, ctype)
        return current

    def _pointer_source_type(self, pointer_id: int) -> CType | None:
        current_id = pointer_id
        seen: set[int] = set()
        for _ in range(4):
            if current_id in seen:
                break
            seen.add(current_id)
            pointer_type = self._node_type(current_id)
            if pointer_type is not None and pointer_type.kind == TypeKind.POINTER:
                return pointer_type
            definition = self.data.get_def(current_id)
            if (
                definition is None
                or definition.opcode not in {"COPY", "PTRSUB", "PTRADD", "INT_ADD"}
                or not definition.inputs
            ):
                break
            current_id = definition.inputs[0]
        return None

    def _access_type_from_pointer(self, pointer_id: int, size: int) -> CType | None:
        if self.types is None:
            return None
        pointer_type = self._pointer_source_type(pointer_id)
        if pointer_type is None:
            return None
        return self.types.get_value_datatype(pointer_type, size)

    def _find_root_pointer(
        self, pointer_id: int, value_type: CType
    ) -> tuple[int, CType, int] | None:
        pointer_type = self._pointer_source_type(pointer_id)
        if pointer_type is None or pointer_type.target is None:
            return None
        implied_base: CType | None = None
        target = value_type
        partial_offset = 0
        if target.kind == TypeKind.PARTIAL_STRUCT:
            implied_base = target.parent or target.target
            if implied_base is None:
                return None
            partial_offset = target.offset
            target = implied_base
        elif target.kind == TypeKind.ARRAY and target.target is not None:
            implied_base = target.target
        base_offset = (
            partial_offset
            if pointer_type.target is not value_type
            and pointer_type.target != value_type
            else 0
        )
        current_id = pointer_id
        current_type = pointer_type
        for _ in range(4):
            pointed = current_type.target
            pointed_base = (
                pointed.parent or pointed.target
                if pointed is not None and pointed.kind == TypeKind.PARTIAL_STRUCT
                else pointed
            )
            if pointed_base is target:
                return current_id, current_type, base_offset
            if pointed_base == target:
                return current_id, current_type, base_offset
            definition = self.data.get_def(current_id)
            if definition is None or not definition.inputs:
                break
            if definition.opcode == "COPY":
                offset = 0
            elif definition.opcode in {"PTRSUB", "INT_ADD"} and len(definition.inputs) >= 2:
                offset_id = definition.inputs[1]
                if not self.data.is_constant(offset_id):
                    break
                offset = self.data.get_constant_value(offset_id)
                if offset & (1 << (self.data.nodes[offset_id].size_bits - 1)):
                    offset -= 1 << self.data.nodes[offset_id].size_bits
            elif definition.opcode == "PTRADD" and len(definition.inputs) >= 3:
                index_id, stride_id = definition.inputs[1:3]
                if not self.data.is_constant(index_id) or not self.data.is_constant(stride_id):
                    break
                offset = self.data.get_constant_value(index_id) * self.data.get_constant_value(stride_id)
            else:
                break
            base_id = definition.inputs[0]
            base_type = self._pointer_source_type(base_id)
            if base_type is None or base_type.target is None:
                break
            parent = base_type.target
            if (
                parent.kind not in {TypeKind.STRUCT, TypeKind.ARRAY, TypeKind.PARTIAL_STRUCT}
                and (definition.opcode not in {"COPY", "PTRADD"} or parent != implied_base)
            ):
                break
            base_offset += offset
            current_id = base_id
            current_type = base_type
        return None

    def _pointer_for_piece(
        self,
        root_id: int,
        root_type: CType,
        base_offset: int,
        piece: SplitPiece,
        address: int | None,
    ) -> int:
        if self.types is None:
            return root_id
        pointee = piece.output_type
        pointer_type = self.types.pointer(pointee)
        byte_offset = base_offset + piece.offset
        constant_id = self.data.new_constant(
            self.data.nodes[root_id].size_bits, byte_offset
        )
        pointer_id = self.data.new_unique(
            self.data.nodes[root_id].size_bits, "split_ptr"
        )
        self.data.set_node_type(pointer_id, pointer_type)
        self.data.append_operation(
            PcodeOp(
                "PTRSUB",
                (root_id, constant_id),
                pointer_id,
                address=address,
                evidence="splitdatatype-field-pointer",
            )
        )
        return pointer_id

    def _split_copy_types(
        self, copy_op: PcodeOp, in_type: CType | None, out_type: CType | None
    ) -> tuple[int, int, CType, CType, tuple[SplitPiece, ...]] | None:
        if copy_op.output is None or len(copy_op.inputs) != 1:
            return None
        source_id = copy_op.inputs[0]
        output_id = copy_op.output
        if source_id < 0 or source_id >= len(self.data.nodes):
            return None
        if self.data.nodes[source_id].kind in {"input", "memory_input"}:
            self.last_reason = "function inputs are not split by COPY"
            return None
        total_size = self._size_bytes(self.data, output_id)
        input_type = in_type or self._node_type(source_id)
        if input_type is None and self.data.is_constant(source_id):
            input_type = self._undefined(total_size)
        in_type = self._sized_type(input_type, total_size)
        out_type = self._sized_type(out_type or self._node_type(output_id), total_size)
        if in_type is None or out_type is None:
            self.last_reason = "COPY input/output types are not available"
            return None
        pieces = self.test_datatype_compatibility(
            in_type,
            out_type,
            in_constant=self.data.is_constant(source_id),
            is_load_store=False,
        )
        if pieces is None:
            return None
        if self._arithmetic_output(source_id) or self._arithmetic_input(output_id):
            self.last_reason = "arithmetic value cannot be split safely"
            return None
        return source_id, output_id, in_type, out_type, pieces

    def split_copy(
        self,
        copy_op: PcodeOp,
        in_type: CType | None = None,
        out_type: CType | None = None,
    ) -> bool:
        """Replace a full-width COPY with field-sized copies and a PIECE tree."""
        if self.types is None:
            self.last_reason = "no TypeDatabase is attached to the p-code graph"
            return False
        resolved = self._split_copy_types(copy_op, self._type(in_type), self._type(out_type))
        if resolved is None:
            return False
        source_id, output_id, _in_type, result_type, pieces = resolved
        total_size = self._size_bytes(self.data, source_id)
        group = self._piece_group(output_id)
        self._set_node_piece(source_id, group, 0, total_size)
        input_ids = self._split_input(source_id, pieces, total_size, copy_op.address)
        self._set_piece_list(input_ids, pieces, group)
        output_ids: list[int] = []
        for index, piece in enumerate(pieces):
            output_id_piece = self.data.new_unique(piece.size * 8, f"split_copy_{index}")
            self.data.set_node_type(output_id_piece, piece.output_type)
            self._set_node_piece(
                output_id_piece,
                group,
                piece.offset,
                piece.size,
                is_padding=piece.is_padding,
            )
            self.data.append_operation(
                PcodeOp(
                    "COPY",
                    (input_ids[index],),
                    output_id_piece,
                    address=copy_op.address,
                    evidence="splitdatatype-copy-piece",
                )
            )
            output_ids.append(output_id_piece)
        self._set_node_piece(output_id, group, 0, total_size)
        self.data.op_destroy(copy_op)
        self._append_concat(output_ids, output_id, copy_op.address, result_type)
        return True

    def _load_pointer_and_types(
        self, load_op: PcodeOp, in_type: CType | None
    ) -> tuple[int, CType, CType, int, int, PcodeOp | None] | None:
        if load_op.output is None:
            return None
        pointer_id = self._assemble_operand(load_op, 1)
        if pointer_id is None:
            self.last_reason = "LOAD pointer operand is unavailable"
            return None
        total_size = self._size_bytes(self.data, load_op.output)
        access_type = self._type(in_type) or self._access_type_from_pointer(pointer_id, total_size)
        output_type = self._node_type(load_op.output) or access_type
        copy_op: PcodeOp | None = None
        uses = self.data.get_uses(load_op.output)
        if len(uses) == 1 and uses[0].opcode == "COPY" and uses[0].output is not None:
            copy_op = uses[0]
            output_type = self._node_type(copy_op.output) or output_type
        if access_type is None or output_type is None:
            self.last_reason = "LOAD types cannot be inferred from the pointer"
            return None
        access_type = self._sized_type(access_type, total_size) or access_type
        output_type = self._sized_type(output_type, total_size) or output_type
        return (
            pointer_id,
            access_type,
            output_type,
            load_op.output,
            copy_op.output if copy_op is not None and copy_op.output is not None else load_op.output,
            copy_op,
        )

    def split_load(self, load_op: PcodeOp, in_type: CType | None = None) -> bool:
        """Replace a full aggregate LOAD with field-addressed smaller LOADs."""
        self.is_load_store = True
        self.data_type_pieces.clear()
        if self.types is None:
            self.last_reason = "no TypeDatabase is attached to the p-code graph"
            return False
        resolved = self._load_pointer_and_types(load_op, self._type(in_type))
        if resolved is None:
            return False
        pointer_id, access_type, output_type, _load_output, final_output, copy_op = resolved
        pieces = self.test_datatype_compatibility(
            access_type, output_type, is_load_store=True
        )
        if pieces is None:
            return False
        if self._arithmetic_input(final_output):
            self.last_reason = "arithmetic result cannot be split safely"
            return False
        root = self._find_root_pointer(pointer_id, access_type)
        if root is None:
            self.last_reason = "LOAD pointer does not resolve to a typed aggregate root"
            return False
        root_id, root_type, base_offset = root
        group = self._piece_group(final_output)
        self._set_node_piece(
            load_op.output, group, 0, self._size_bytes(self.data, load_op.output)
        )
        field_values: list[int] = []
        for index, piece in enumerate(pieces):
            field_pointer = self._pointer_for_piece(
                root_id, root_type, base_offset, piece, load_op.address
            )
            value_id = self.data.new_unique(piece.size * 8, f"split_load_{index}")
            self.data.set_node_type(value_id, piece.output_type)
            self._set_node_piece(
                value_id,
                group,
                piece.offset,
                piece.size,
                is_padding=piece.is_padding,
            )
            self.data.append_operation(
                PcodeOp(
                    "LOAD",
                    (field_pointer,),
                    value_id,
                    address=load_op.address,
                    evidence="splitdatatype-load-piece",
                    input_slots=(1,),
                )
            )
            field_values.append(value_id)
        self._set_node_piece(
            final_output, group, 0, self._size_bytes(self.data, final_output)
        )
        if copy_op is not None:
            self.data.op_destroy(copy_op)
        self.data.op_destroy(load_op)
        self._append_concat(field_values, final_output, load_op.address, output_type)
        return True

    def _store_pointer_value(
        self, store_op: PcodeOp, out_type: CType | None
    ) -> tuple[int, int, CType, CType, int] | None:
        pointer_id = self._assemble_operand(store_op, 1)
        value_id = self._assemble_operand(store_op, 2)
        if pointer_id is None or value_id is None:
            self.last_reason = "STORE pointer/value operands are unavailable"
            return None
        total_size = self._size_bytes(self.data, value_id)
        output_type = self._type(out_type) or self._access_type_from_pointer(pointer_id, total_size)
        input_type = self._node_type(value_id)
        if output_type is None:
            self.last_reason = "STORE destination type cannot be inferred from its pointer"
            return None
        if input_type is None:
            input_type = self._undefined(total_size)
        input_type = self._sized_type(input_type, total_size) or input_type
        output_type = self._sized_type(output_type, total_size) or output_type
        return pointer_id, value_id, input_type, output_type, total_size

    def split_store(self, store_op: PcodeOp, out_type: CType | None = None) -> bool:
        """Replace a full aggregate STORE with component-sized stores."""
        self.is_load_store = True
        self.data_type_pieces.clear()
        if self.types is None or len(store_op.inputs) < 2:
            self.last_reason = "no TypeDatabase or STORE operands are available"
            return False
        resolved = self._store_pointer_value(store_op, self._type(out_type))
        if resolved is None:
            return False
        pointer_id, value_id, input_type, output_type, total_size = resolved
        pieces = self.test_datatype_compatibility(
            input_type,
            output_type,
            in_constant=self.data.is_constant(value_id),
            is_load_store=True,
        )
        if pieces is None:
            return False
        if self._arithmetic_output(value_id):
            self.last_reason = "arithmetic value cannot be split safely"
            return False
        root = self._find_root_pointer(pointer_id, output_type)
        if root is None:
            self.last_reason = "STORE pointer does not resolve to a typed aggregate root"
            return False
        root_id, root_type, base_offset = root
        group = self._piece_group(value_id)
        self._set_node_piece(value_id, group, 0, total_size)
        input_ids = self._split_input(value_id, pieces, total_size, store_op.address)
        self._set_piece_list(input_ids, pieces, group)
        field_pointers = [
            self._pointer_for_piece(
                root_id, root_type, base_offset, piece, store_op.address
            )
            for piece in pieces
        ]
        # Preserve the original STORE object as the first component, mirroring
        # Ghidra's treatment of INDIRECT/store references.
        self.data.op_rewrite(
            store_op,
            inputs=(field_pointers[0], input_ids[0]),
            input_slots=(1, 2),
            evidence="splitdatatype-store-piece",
        )
        for pointer_piece, input_piece in zip(
            field_pointers[1:], input_ids[1:], strict=True
        ):
            self.data.append_operation(
                PcodeOp(
                    "STORE",
                    (pointer_piece, input_piece),
                    address=store_op.address,
                    evidence="splitdatatype-store-piece",
                    input_slots=(1, 2),
                )
            )
        return True

    def split_operation(self, operation: PcodeOp) -> bool:
        """Dispatch one supported p-code operation to the appropriate splitter."""
        if operation.opcode == "COPY":
            return self.split_copy(operation)
        if operation.opcode == "LOAD":
            return self.split_load(operation)
        if operation.opcode == "STORE":
            return self.split_store(operation)
        return False

    @classmethod
    def apply_to_graph(cls, data: PcodeLikeGraph, *, max_passes: int = 4) -> int:
        """Run aggregate split rewrites to a fixed point on a p-code graph."""
        splitter = cls(data)
        changed = 0
        for _ in range(max_passes):
            pass_changed = 0
            for operation in tuple(data.ops):
                if splitter.split_operation(operation):
                    pass_changed += 1
            changed += pass_changed
            if not pass_changed:
                break
        return changed
