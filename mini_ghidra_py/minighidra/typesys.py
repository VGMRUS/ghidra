"""Small C type system used by the x86 prototype resolver.

The model intentionally separates source-level types from ABI storage.  It is
sufficient for scalar, pointer, array, enum, vector, structure, and union
layouts used by the x86 compiler specifications; it is not a replacement for
Ghidra's complete data-type manager or DWARF/PDB type importers.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum


class TypeKind(str, Enum):
    UNKNOWN = "unknown"
    VOID = "void"
    BOOL = "bool"
    INTEGER = "integer"
    FLOAT = "float"
    POINTER = "pointer"
    ARRAY = "array"
    STRUCT = "struct"
    UNION = "union"
    ENUM = "enum"
    VECTOR = "vector"
    TYPEDEF = "typedef"
    FUNCTION = "function"
    PARTIAL_STRUCT = "partial_struct"


@dataclass(frozen=True, slots=True)
class CField:
    name: str
    type: CType
    offset: int


@dataclass(frozen=True, slots=True)
class CTypeComponent:
    """One addressable leaf (or padding hole) in a composite C datatype."""

    type: CType
    offset: int
    size: int
    path: tuple[str, ...] = ()
    is_hole: bool = False


@dataclass(frozen=True, slots=True)
class CType:
    name: str
    kind: TypeKind
    size: int | None
    alignment: int | None = None
    signed: bool | None = None
    target: CType | None = None
    count: int | None = None
    fields: tuple[CField, ...] = ()
    packed: bool = False
    parameters: tuple[CType, ...] = ()
    variadic: bool = False
    parent: CType | None = None
    offset: int = 0

    @property
    def align_size(self) -> int | None:
        """Size rounded up to the type's alignment, like Ghidra's getAlignSize."""
        base = self.unwrapped()
        if base.size is None:
            return None
        if base.kind in {TypeKind.STRUCT, TypeKind.UNION}:
            return _align(base.size, base.alignment or 1)
        return base.size

    @property
    def is_aggregate(self) -> bool:
        return self.kind in {
            TypeKind.STRUCT,
            TypeKind.UNION,
            TypeKind.ARRAY,
            TypeKind.PARTIAL_STRUCT,
        }

    @property
    def is_scalar(self) -> bool:
        return self.kind in {
            TypeKind.BOOL,
            TypeKind.INTEGER,
            TypeKind.FLOAT,
            TypeKind.POINTER,
            TypeKind.ENUM,
            TypeKind.VECTOR,
        }

    def declaration(self, identifier: str = "") -> str:
        if self.kind == TypeKind.POINTER and self.target is not None:
            decorated = f"*{identifier}" if identifier else "*"
            if self.target.kind in {TypeKind.ARRAY, TypeKind.FUNCTION}:
                decorated = f"({decorated})"
            return self.target.declaration(decorated)
        if self.kind == TypeKind.ARRAY and self.target is not None:
            suffix = "" if self.count is None else str(self.count)
            return self.target.declaration(f"{identifier}[{suffix}]")
        if self.kind == TypeKind.FUNCTION and self.target is not None:
            parameters = [parameter.declaration() for parameter in self.parameters]
            if self.variadic:
                parameters.append("...")
            if not parameters:
                parameters.append("void")
            return self.target.declaration(f"{identifier}({', '.join(parameters)})")
        if identifier:
            return f"{self.name} {identifier}"
        return self.name

    def unwrapped(self) -> CType:
        current = self
        seen: set[int] = set()
        while current.kind in {TypeKind.TYPEDEF, TypeKind.ENUM} and current.target:
            if id(current) in seen:
                break
            seen.add(id(current))
            current = current.target
        return current

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class StorageLocation:
    """One logical piece of a parameter/return value in a calling convention.

    ``register_offset`` identifies a subrange within a physical register (for
    example, the upper SSEUP lane in ``xmm0``); ``value_offset`` remains the
    corresponding byte offset in the logical value. ``join_record`` describes
    the physical piece(s) when a logical location is backed by Ghidra join
    space, including x87 precision-extension records.
    """

    kind: str
    size: int
    register: str | None = None
    stack_offset: int | None = None
    value_offset: int = 0
    indirect: bool = False
    hidden: bool = False
    register_offset: int = 0
    join_record: JoinRecord | None = None

    @property
    def is_register(self) -> bool:
        return self.kind == "register"

    @property
    def is_stack(self) -> bool:
        return self.kind == "stack"

    @property
    def physical_pieces(self) -> tuple[StorageLocation, ...]:
        """Return underlying register/stack pieces, flattening any join record."""
        if self.join_record is not None:
            return self.join_record.pieces
        return (self,)

    @property
    def is_joined(self) -> bool:
        return self.join_record is not None


@dataclass(frozen=True, slots=True)
class JoinRecord:
    """Ghidra-style mapping from one logical join-space value to physical pieces.

    Piece order is most-significant to least-significant, as in
    ``Architecture::joinPieces``. A float-extension record is the special
    single-piece case where the logical precision is smaller than the physical
    register width (for example, a 4-byte ``float`` in a 10-byte ``st0``).
    """

    pieces: tuple[StorageLocation, ...]
    logical_size: int
    float_extension: bool = False
    big_endian: bool = False

    def __post_init__(self) -> None:
        if not self.pieces or self.logical_size <= 0:
            raise ValueError("JoinRecord necesita piezas físicas y tamaño lógico positivo")
        if any(piece.size <= 0 for piece in self.pieces):
            raise ValueError("Las piezas físicas de JoinRecord deben tener tamaño positivo")
        physical_size = sum(piece.size for piece in self.pieces)
        if self.float_extension:
            if len(self.pieces) != 1 or self.logical_size >= physical_size:
                raise ValueError(
                    "Un JoinRecord float requiere una pieza más ancha que el valor lógico"
                )
        elif self.logical_size != physical_size:
            raise ValueError("El tamaño lógico de JoinRecord debe sumar sus piezas")

    def get_equivalent_piece(
        self, logical_offset: int
    ) -> tuple[StorageLocation, int] | None:
        """Map a relative join-space offset to ``(piece, byte offset)``."""
        if logical_offset < 0 or logical_offset >= self.logical_size:
            return None
        remaining = logical_offset
        indices = (
            range(len(self.pieces))
            if self.big_endian
            else range(len(self.pieces) - 1, -1, -1)
        )
        for index in indices:
            piece = self.pieces[index]
            if remaining < piece.size:
                return piece, remaining
            remaining -= piece.size
        return None


def construct_float_extension_location(
    real_location: StorageLocation,
    real_size: int,
    logical_size: int,
    *,
    big_endian: bool = False,
) -> StorageLocation:
    """Model ``constructFloatExtensionAddress`` for one real storage location."""
    if real_size <= 0 or logical_size <= 0 or logical_size > real_size:
        raise ValueError("Float extension requiere 0 < logical_size <= real_size")
    if real_location.size < real_size:
        raise ValueError("La ubicación real no cubre el ancho físico del registro")
    physical_piece = replace(
        real_location,
        size=real_size,
        value_offset=0,
        join_record=None,
    )
    if logical_size == real_size:
        return physical_piece
    record = JoinRecord(
        pieces=(physical_piece,),
        logical_size=logical_size,
        float_extension=True,
        big_endian=big_endian,
    )
    return replace(
        physical_piece,
        size=logical_size,
        join_record=record,
    )


class TypeDatabase:
    """Architecture-aware registry for explicit and inferred C data types."""

    def __init__(self, bits: int, format_name: str = "") -> None:
        if bits not in {32, 64}:
            raise ValueError("TypeDatabase solo admite x86 de 32 y 64 bits")
        self.bits = bits
        self.format = format_name.upper()
        self.pointer_size = bits // 8
        self._types: dict[str, CType] = {}
        self._install_builtins()

    def _install_builtins(self) -> None:
        def add(
            name: str,
            kind: TypeKind,
            size: int | None,
            *,
            signed: bool | None = None,
            alignment: int | None = None,
        ) -> CType:
            ctype = CType(name, kind, size, alignment or max(1, size or 1), signed)
            self._types[name] = ctype
            return ctype

        add("void", TypeKind.VOID, 0)
        add("_Bool", TypeKind.BOOL, 1, signed=False)
        add("bool", TypeKind.BOOL, 1, signed=False)
        add("char", TypeKind.INTEGER, 1, signed=True)
        add("signed char", TypeKind.INTEGER, 1, signed=True)
        add("unsigned char", TypeKind.INTEGER, 1, signed=False)
        add("short", TypeKind.INTEGER, 2, signed=True)
        add("short int", TypeKind.INTEGER, 2, signed=True)
        add("signed short", TypeKind.INTEGER, 2, signed=True)
        add("signed short int", TypeKind.INTEGER, 2, signed=True)
        add("short signed int", TypeKind.INTEGER, 2, signed=True)
        add("unsigned short", TypeKind.INTEGER, 2, signed=False)
        add("unsigned short int", TypeKind.INTEGER, 2, signed=False)
        add("int", TypeKind.INTEGER, 4, signed=True)
        add("signed", TypeKind.INTEGER, 4, signed=True)
        add("signed int", TypeKind.INTEGER, 4, signed=True)
        add("unsigned", TypeKind.INTEGER, 4, signed=False)
        add("unsigned int", TypeKind.INTEGER, 4, signed=False)
        long_size = 4 if self.format == "PE" or self.bits == 32 else 8
        add("long", TypeKind.INTEGER, long_size, signed=True)
        add("long int", TypeKind.INTEGER, long_size, signed=True)
        add("signed long", TypeKind.INTEGER, long_size, signed=True)
        add("unsigned long", TypeKind.INTEGER, long_size, signed=False)
        add("unsigned long int", TypeKind.INTEGER, long_size, signed=False)
        add("long unsigned int", TypeKind.INTEGER, long_size, signed=False)
        add("long signed int", TypeKind.INTEGER, long_size, signed=True)
        add("long long", TypeKind.INTEGER, 8, signed=True)
        add("long long int", TypeKind.INTEGER, 8, signed=True)
        add("signed long long", TypeKind.INTEGER, 8, signed=True)
        add("signed long long int", TypeKind.INTEGER, 8, signed=True)
        add("long long signed int", TypeKind.INTEGER, 8, signed=True)
        add("unsigned long long", TypeKind.INTEGER, 8, signed=False)
        add("unsigned long long int", TypeKind.INTEGER, 8, signed=False)
        add("long long unsigned int", TypeKind.INTEGER, 8, signed=False)
        add("__int64", TypeKind.INTEGER, 8, signed=True)
        add("signed __int64", TypeKind.INTEGER, 8, signed=True)
        add("unsigned __int64", TypeKind.INTEGER, 8, signed=False)
        add("float", TypeKind.FLOAT, 4)
        add("double", TypeKind.FLOAT, 8)
        if self.format == "PE":
            long_double_size = 8
            long_double_alignment = 8
        elif self.bits == 64:
            long_double_size = 16
            long_double_alignment = 16
        else:
            long_double_size = 12
            long_double_alignment = 4
        add(
            "long double",
            TypeKind.FLOAT,
            long_double_size,
            alignment=long_double_alignment,
        )
        add("size_t", TypeKind.INTEGER, self.pointer_size, signed=False)
        add("ssize_t", TypeKind.INTEGER, self.pointer_size, signed=True)
        add("intptr_t", TypeKind.INTEGER, self.pointer_size, signed=True)
        add("uintptr_t", TypeKind.INTEGER, self.pointer_size, signed=False)
        add("int8_t", TypeKind.INTEGER, 1, signed=True)
        add("uint8_t", TypeKind.INTEGER, 1, signed=False)
        add("int16_t", TypeKind.INTEGER, 2, signed=True)
        add("uint16_t", TypeKind.INTEGER, 2, signed=False)
        add("int32_t", TypeKind.INTEGER, 4, signed=True)
        add("uint32_t", TypeKind.INTEGER, 4, signed=False)
        add("int64_t", TypeKind.INTEGER, 8, signed=True)
        add("uint64_t", TypeKind.INTEGER, 8, signed=False)
        if self.bits == 64:
            add("__int128", TypeKind.INTEGER, 16, signed=True, alignment=16)
            add("__int128_t", TypeKind.INTEGER, 16, signed=True, alignment=16)
            add("unsigned __int128", TypeKind.INTEGER, 16, signed=False, alignment=16)
            add("__uint128_t", TypeKind.INTEGER, 16, signed=False, alignment=16)
        wchar_size = 2 if self.format == "PE" else 4
        add("wchar_t", TypeKind.INTEGER, wchar_size, signed=False)
        for name in ("__m128", "__m128d", "__m128i"):
            self._types[name] = CType(name, TypeKind.VECTOR, 16, 16)

    def get(self, name: str) -> CType:
        try:
            return self._types[name.strip()]
        except KeyError as exc:
            raise KeyError(f"Tipo no definido: {name}") from exc

    def resolve(self, spec: str | CType) -> CType:
        if isinstance(spec, CType):
            return spec
        declaration = " ".join(spec.strip().split())
        declaration = re.sub(
            r"\b(const|volatile|restrict|__restrict)\b", "", declaration
        )
        declaration = " ".join(declaration.split())
        if declaration in self._types:
            return self._types[declaration]
        array_match = re.fullmatch(r"(.+?)\s*\[\s*(\d*)\s*\]", declaration)
        if array_match:
            element = self.resolve(array_match.group(1))
            return self.array(
                element, int(array_match.group(2)) if array_match.group(2) else None
            )
        pointer_match = re.search(r"(?:\s*\*+\s*)+$", declaration)
        pointer_depth = pointer_match.group(0).count("*") if pointer_match else 0
        base_name = (
            declaration[: pointer_match.start()].strip()
            if pointer_match
            else declaration
        )
        key = base_name
        if key not in self._types and key.startswith(("struct ", "union ")):
            bare = key.split(" ", 1)[1]
            key = bare if bare in self._types else key
        if key not in self._types:
            raise KeyError(f"Tipo no definido: {spec}")
        ctype = self._types[key]
        for _ in range(pointer_depth):
            ctype = self.pointer(ctype)
        return ctype

    def add(self, ctype: CType, *aliases: str) -> CType:
        for key in (ctype.name, *aliases):
            self._types[key.strip()] = ctype
        return ctype

    def typedef(self, name: str, target: str | CType) -> CType:
        base = self.resolve(target)
        return self.add(
            CType(
                name,
                TypeKind.TYPEDEF,
                base.size,
                base.alignment,
                base.signed,
                target=base,
            )
        )

    def forward_structure(self, name: str) -> CType:
        """Register an incomplete struct tag for self-referential pointers."""
        return self._forward_aggregate(name, TypeKind.STRUCT)

    def forward_union(self, name: str) -> CType:
        """Register an incomplete union tag for self-referential pointers."""
        return self._forward_aggregate(name, TypeKind.UNION)

    def _forward_aggregate(self, name: str, kind: TypeKind) -> CType:
        prefix = "struct" if kind == TypeKind.STRUCT else "union"
        bare_name = name.removeprefix(f"{prefix} ").strip()
        if not bare_name:
            raise ValueError("Un tipo agregado requiere una etiqueta")
        tag_name = f"{prefix} {bare_name}"
        existing = self._types.get(tag_name) or self._types.get(bare_name)
        if existing is not None:
            if existing.kind != kind:
                raise ValueError(f"{bare_name} ya está registrado con otro tipo")
            return existing
        return self.add(CType(tag_name, kind, None, 1), bare_name)

    def pointer(self, target: str | CType) -> CType:
        base = self.resolve(target)
        return CType(
            f"{base.name} *",
            TypeKind.POINTER,
            self.pointer_size,
            self.pointer_size,
            signed=False,
            target=base,
        )

    def array(self, target: str | CType, count: int | None) -> CType:
        base = self.resolve(target)
        if (
            base.unwrapped().kind in {TypeKind.VOID, TypeKind.FUNCTION}
            or base.size is None
        ):
            raise ValueError("Un array requiere un tipo de elemento completo")
        if count is not None and count < 0:
            raise ValueError("El tamaño de un array no puede ser negativo")
        size = (
            None
            if count is None or base.align_size is None
            else count * base.align_size
        )
        suffix = "" if count is None else str(count)
        return CType(
            f"{base.name}[{suffix}]",
            TypeKind.ARRAY,
            size,
            base.alignment,
            target=base,
            count=count,
        )

    def function(
        self,
        return_type: str | CType,
        parameters: Iterable[str | CType] = (),
        *,
        variadic: bool = False,
    ) -> CType:
        result = self.resolve(return_type)
        result_base = result.unwrapped()
        if result_base.kind in {TypeKind.UNKNOWN, TypeKind.ARRAY, TypeKind.FUNCTION}:
            raise ValueError("El tipo de retorno de función C no es válido")
        if result_base.kind != TypeKind.VOID and result_base.size is None:
            raise ValueError("Una función C no puede retornar un tipo incompleto")
        parameter_specs = tuple(parameters)
        resolved_parameters = []
        for parameter in parameter_specs:
            ctype = self.resolve(parameter)
            base = ctype.unwrapped()
            if base.kind == TypeKind.UNKNOWN:
                raise ValueError("Un parámetro de función requiere un tipo definido")
            if base.kind == TypeKind.VOID:
                if resolved_parameters or len(parameter_specs) != 1:
                    raise ValueError("void solo puede aparecer como parámetro único")
                continue
            if base.kind == TypeKind.ARRAY and base.target is not None:
                ctype = self.pointer(base.target)
            elif base.kind == TypeKind.FUNCTION:
                ctype = self.pointer(base)
            elif base.size is None:
                raise ValueError("Un parámetro de función no puede ser incompleto")
            resolved_parameters.append(ctype)
        parameter_names = [parameter.declaration() for parameter in resolved_parameters]
        if variadic:
            parameter_names.append("...")
        if not parameter_names:
            parameter_names.append("void")
        name = f"{result.name} ({', '.join(parameter_names)})"
        return self.add(
            CType(
                name,
                TypeKind.FUNCTION,
                None,
                target=result,
                parameters=tuple(resolved_parameters),
                variadic=variadic,
            )
        )

    def enum(self, name: str, underlying: str | CType = "int") -> CType:
        base = self.resolve(underlying)
        if base.unwrapped().kind not in {TypeKind.BOOL, TypeKind.INTEGER}:
            raise ValueError("El tipo subyacente de un enum debe ser entero")
        return self.add(
            CType(
                name,
                TypeKind.ENUM,
                base.size,
                base.alignment,
                base.signed,
                target=base,
            )
        )

    def structure(
        self,
        name: str,
        fields: Iterable[
            tuple[str, str | CType] | tuple[str, str | CType, int] | CField
        ],
        *,
        packed: bool = False,
        size: int | None = None,
        alignment: int | None = None,
    ) -> CType:
        bare_name = name.removeprefix("struct ").strip()
        if not bare_name:
            raise ValueError("Un struct requiere una etiqueta")
        if alignment is not None and alignment <= 0:
            raise ValueError("La alineación debe ser positiva")
        built: list[CField] = []
        cursor = 0
        maximum_alignment = 1
        for item in fields:
            if isinstance(item, CField):
                field_name, field_type, offset = (
                    item.name,
                    self.resolve(item.type),
                    item.offset,
                )
            elif len(item) == 2:
                field_name, field_spec = item
                field_type = self.resolve(field_spec)
                field_alignment = 1 if packed else (field_type.alignment or 1)
                offset = _align(cursor, field_alignment)
            else:
                field_name, field_spec, offset = item
                field_type = self.resolve(field_spec)
            if field_type.size is None:
                raise ValueError("No se puede incluir un tipo incompleto por valor")
            if offset < 0:
                raise ValueError("El offset de un campo no puede ser negativo")
            field_alignment = 1 if packed else (field_type.alignment or 1)
            maximum_alignment = max(maximum_alignment, field_alignment)
            built.append(CField(field_name, field_type, int(offset)))
            if field_type.align_size is not None:
                cursor = max(cursor, int(offset) + field_type.align_size)
        alignment = alignment or maximum_alignment
        computed_size = _align(cursor, alignment)
        if size is not None:
            if size < cursor:
                raise ValueError("El tamaño declarado del struct no cubre sus campos")
            computed_size = size
        ctype = CType(
            f"struct {bare_name}",
            TypeKind.STRUCT,
            computed_size,
            alignment,
            fields=tuple(built),
            packed=packed,
        )
        return self.add(ctype, bare_name)

    def union(
        self,
        name: str,
        fields: Iterable[tuple[str, str | CType] | CField],
        *,
        packed: bool = False,
        size: int | None = None,
        alignment: int | None = None,
    ) -> CType:
        bare_name = name.removeprefix("union ").strip()
        if not bare_name:
            raise ValueError("Un union requiere una etiqueta")
        if alignment is not None and alignment <= 0:
            raise ValueError("La alineación debe ser positiva")
        built: list[CField] = []
        maximum_size = 0
        maximum_alignment = 1
        for item in fields:
            if isinstance(item, CField):
                field_name, field_type = item.name, self.resolve(item.type)
            else:
                field_name, field_spec = item
                field_type = self.resolve(field_spec)
            if field_type.size is None:
                raise ValueError("No se puede incluir un tipo incompleto por valor")
            maximum_size = max(maximum_size, field_type.size)
            maximum_alignment = max(
                maximum_alignment, 1 if packed else (field_type.alignment or 1)
            )
            built.append(CField(field_name, field_type, 0))
        alignment = alignment or maximum_alignment
        computed_size = _align(maximum_size, alignment)
        if size is not None:
            if size < maximum_size:
                raise ValueError("El tamaño declarado del union no cubre sus campos")
            computed_size = size
        ctype = CType(
            f"union {bare_name}",
            TypeKind.UNION,
            computed_size,
            alignment,
            fields=tuple(built),
            packed=packed,
        )
        return self.add(ctype, bare_name)

    def partial(self, parent: str | CType, size: int, offset: int = 0) -> CType:
        """Create a Ghidra-like partial composite view over a struct or array.

        ``Datatype`` values in Ghidra can describe a bounded slice of a larger
        structure.  Keeping the parent and byte offset explicit lets the
        splitter reason about a LOAD/STORE through a pointer-to-field without
        inventing a second, disconnected layout model.
        """
        base = self.resolve(parent)
        if base.kind == TypeKind.PARTIAL_STRUCT:
            offset += base.offset
            base = base.parent or base.target  # type: ignore[assignment]
            if base is None:
                raise ValueError("El tipo parcial no conserva su tipo padre")
        base = base.unwrapped()
        if base.kind not in {TypeKind.STRUCT, TypeKind.ARRAY} or base.size is None:
            raise ValueError("Una vista parcial requiere un struct/array completo")
        if size <= 0 or offset < 0 or offset + size > base.size:
            raise ValueError("El intervalo parcial queda fuera del tipo compuesto")
        return CType(
            f"{base.name}/*+0x{offset:x}:{size}*/",
            TypeKind.PARTIAL_STRUCT,
            size,
            base.alignment,
            target=base,
            parent=base,
            offset=offset,
        )

    def undefined(self, size: int) -> CType:
        """Return an anonymous byte-sized type for a composite padding hole."""
        if size <= 0:
            raise ValueError("Un tipo undefined debe tener tamaño positivo")
        return CType(f"undefined{size}", TypeKind.UNKNOWN, size, 1)

    def get_value_datatype(
        self, pointer_type: str | CType, size: int, *, offset: int = 0
    ) -> CType | None:
        """Map a typed pointer and access width to the accessed ``Datatype``.

        This is the Python counterpart of ``SplitDatatype::getValueDatatype``:
        composite accesses become bounded partial views, and repeated scalar
        accesses become arrays when the requested width is an exact multiple
        of the scalar's aligned size.
        """
        if size <= 0:
            return None
        try:
            pointer = self.resolve(pointer_type)
        except (KeyError, TypeError):
            return None
        if pointer.kind != TypeKind.POINTER or pointer.target is None:
            return None
        target = pointer.target
        if target.kind == TypeKind.PARTIAL_STRUCT:
            offset += target.offset
            target = target.parent or target.target  # type: ignore[assignment]
            if target is None:
                return None
        target = target.unwrapped()
        if target.size is None:
            return None
        aligned_size = target.align_size or target.size
        if target.kind in {
            TypeKind.BOOL,
            TypeKind.INTEGER,
            TypeKind.FLOAT,
            TypeKind.POINTER,
            TypeKind.ENUM,
        } and size > aligned_size and aligned_size > 0 and size % aligned_size == 0:
            return self.array(target, size // aligned_size)
        if target.kind in {TypeKind.STRUCT, TypeKind.ARRAY}:
            if offset < 0 or offset + size > target.size:
                return None
            if offset == 0 and size == target.size:
                return target
            return self.partial(target, size, offset)
        return None

    def get_component(
        self, spec: str | CType, offset: int, *, hole_limit: int = 8
    ) -> CTypeComponent | None:
        """Find the deepest leaf type or padding hole beginning at ``offset``."""
        if hole_limit <= 0:
            raise ValueError("hole_limit debe ser positivo")
        ctype = self.resolve(spec)
        view_offset = 0
        view_size = ctype.size
        if ctype.kind == TypeKind.PARTIAL_STRUCT:
            view_offset = ctype.offset
            view_size = ctype.size
            ctype = ctype.parent or ctype.target  # type: ignore[assignment]
            if ctype is None:
                return None
        ctype = ctype.unwrapped()
        if (
            ctype.size is None
            or offset < 0
            or view_size is None
            or offset >= view_size
        ):
            return None

        absolute_offset = view_offset + offset
        current_start = 0
        current = ctype
        path: list[str] = []
        while True:
            base = current.unwrapped()
            local_offset = absolute_offset - current_start
            if base.size is None or local_offset < 0 or local_offset >= base.size:
                return None
            if base.kind == TypeKind.STRUCT:
                matching = [
                    field
                    for field in base.fields
                    if field.offset <= local_offset
                    < field.offset + (field.type.size or 0)
                ]
                if matching:
                    field = min(
                        matching,
                        key=lambda item: (item.type.size or 0, item.offset, item.name),
                    )
                    current_start += field.offset
                    current = field.type
                    path.append(field.name)
                    continue
                next_offsets = [
                    field.offset
                    for field in base.fields
                    if field.offset > local_offset
                ]
                boundary = min(next_offsets, default=base.size)
                hole_size = min(hole_limit, max(1, boundary - local_offset))
                return CTypeComponent(
                    self.undefined(hole_size),
                    absolute_offset - view_offset,
                    hole_size,
                    tuple(path),
                    True,
                )
            if base.kind == TypeKind.UNION:
                matching = [
                    field
                    for field in base.fields
                    if local_offset < (field.type.size or 0)
                ]
                if not matching:
                    hole_size = min(hole_limit, base.size - local_offset)
                    return CTypeComponent(
                        self.undefined(hole_size),
                        absolute_offset - view_offset,
                        hole_size,
                        tuple(path),
                        True,
                    )
                field = min(
                    matching,
                    key=lambda item: (item.type.size or 0, item.name),
                )
                current = field.type
                path.append(field.name)
                continue
            if base.kind == TypeKind.ARRAY and base.target is not None:
                element = base.target
                stride = element.align_size or element.size
                if stride is None or stride <= 0:
                    return None
                index, element_offset = divmod(local_offset, stride)
                if base.count is not None and index >= base.count:
                    return None
                if element.size is None or element_offset >= element.size:
                    hole_size = min(hole_limit, stride - element_offset)
                    return CTypeComponent(
                        self.undefined(hole_size),
                        absolute_offset - view_offset,
                        hole_size,
                        tuple(path + [str(index)]),
                        True,
                    )
                current_start += index * stride
                current = element
                path.append(str(index))
                continue
            component_size = base.size
            component_offset = current_start - view_offset
            if component_size <= 0:
                return None
            return CTypeComponent(
                current,
                component_offset,
                component_size,
                tuple(path),
                False,
            )

    def components(
        self,
        spec: str | CType,
        *,
        size: int | None = None,
        offset: int = 0,
        hole_limit: int = 8,
    ) -> tuple[CTypeComponent, ...]:
        """Flatten a datatype interval into ordered leaf/padding components."""
        ctype = self.resolve(spec)
        available = ctype.size
        if ctype.kind == TypeKind.PARTIAL_STRUCT:
            available = ctype.size
        requested = available - offset if size is None and available is not None else size
        if requested is None or requested <= 0 or offset < 0:
            return ()
        if available is not None and offset + requested > available:
            raise ValueError("El intervalo excede el tamaño del tipo")
        result: list[CTypeComponent] = []
        cursor = 0
        while cursor < requested:
            component = self.get_component(
                ctype, offset + cursor, hole_limit=hole_limit
            )
            if component is None:
                return ()
            span = min(component.size, requested - cursor)
            component_type = component.type
            if span != component.size:
                component_type = self.undefined(span)
            result.append(
                CTypeComponent(
                    component_type,
                    cursor,
                    span,
                    component.path,
                    component.is_hole,
                )
            )
            cursor += span
        return tuple(result)

    def promote_vararg(self, spec: str | CType) -> CType:
        """Apply C's default argument promotions to a variadic value."""
        ctype = self.resolve(spec)
        base = ctype.unwrapped()
        if base.kind == TypeKind.ARRAY and base.target is not None:
            return self.pointer(base.target)
        if base.kind == TypeKind.FUNCTION:
            return self.pointer(base)
        if base.kind == TypeKind.FLOAT and base.size == 4:
            return self.get("double")
        if ctype.kind == TypeKind.ENUM:
            if base.size is not None and base.size < 4:
                return self.get("int")
            if base.size == 4:
                return self.get("unsigned int" if base.signed is False else "int")
            return base
        if (
            base.kind in {TypeKind.BOOL, TypeKind.INTEGER}
            and base.size is not None
            and base.size < 4
        ):
            return self.get("int")
        return ctype

    @property
    def types(self) -> tuple[CType, ...]:
        unique = {id(value): value for value in self._types.values()}
        return tuple(sorted(unique.values(), key=lambda item: item.name))


def _align(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) & -alignment
