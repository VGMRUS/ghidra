"""C type/prototype layout translated from the x86 Ghidra compiler specs.

This module implements the type-directed half of prototype resolution: C type
classification, register/stack assignment, aggregate splitting, hidden structure
returns, and variadic argument layout. Binary-only type inference remains a
separate heuristic because machine code does not uniquely encode source types.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .abi import CallingConvention
from .model import FunctionParameter, FunctionSignature
from .typesys import CType, StorageLocation, TypeDatabase, TypeKind


@dataclass(frozen=True, slots=True)
class PrototypeLayout:
    signature: FunctionSignature
    convention: CallingConvention
    variadic_parameters: tuple[FunctionParameter, ...] = ()
    varargs_sse_count: int | None = None


class PrototypeResolver:
    """Resolve a typed prototype to ABI locations for x86 and x86-64.

    The C++ decompiler's ``ParamList``/``ParamEntry`` engine also performs
    data-flow-driven parameter discovery, type recovery, parameter joining,
    symbol updates, and model selection. This class ports the type-directed
    layout rules from the local Cspecs; it deliberately does not pretend that
    those broader heritage/type-propagation analyses can be reconstructed from
    a single instruction scan.
    """

    _SYSV_INTEGER_ARGUMENTS = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
    _SYSV_FLOAT_ARGUMENTS = tuple(f"xmm{index}" for index in range(8))
    _SYSV_INTEGER_RETURNS = ("rax", "rdx")
    _SYSV_FLOAT_RETURNS = ("xmm0", "xmm1")

    def __init__(self, types: TypeDatabase, bits: int) -> None:
        if bits not in {32, 64} or types.bits != bits:
            raise ValueError(
                "El resolvedor y la base de tipos deben compartir x86/x86-64"
            )
        self.types = types
        self.bits = bits
        self.pointer_size = bits // 8

    def resolve(
        self,
        signature: FunctionSignature,
        convention: CallingConvention,
        *,
        variadic_types: tuple[str | CType, ...] = (),
    ) -> PrototypeLayout:
        return_type = self._resolve_type(
            signature.return_data_type, signature.return_type
        )
        parameters = tuple(
            self._resolve_parameter(parameter) for parameter in signature.parameters
        )
        hidden_return = self._needs_hidden_return(return_type, convention)
        hidden_location: StorageLocation | None = None
        return_locations: tuple[StorageLocation, ...]
        if hidden_return:
            hidden_location = self._hidden_return_input_location(convention, parameters)
            return_locations = self._hidden_pointer_result(convention)
        else:
            return_locations = self._return_locations(return_type, convention)

        if self.bits == 64 and convention.name in {"sysv_amd64", "__stdcall"}:
            located, _next_stack, _gp_used, _sse_used = self._layout_sysv_parameters(
                parameters, hidden_return
            )
            varargs_count_register = "al" if signature.variadic else None
        elif self.bits == 64:
            located, _next_stack, _gp_used, _sse_used = (
                self._layout_windows64_parameters(parameters, hidden_return, convention)
            )
            varargs_count_register = None
        else:
            located, _next_stack, _gp_used, _sse_used = self._layout_x86_parameters(
                parameters, hidden_return, convention
            )
            varargs_count_register = None

        resolved = replace(
            signature,
            return_type=return_type.name,
            return_data_type=return_type,
            parameters=located,
            calling_convention=convention.name,
            return_locations=return_locations,
            hidden_return=hidden_return,
            hidden_return_location=hidden_location,
            varargs_sse_count_register=varargs_count_register,
        )
        variadic_parameters: tuple[FunctionParameter, ...] = ()
        varargs_sse_count: int | None = None
        if variadic_types:
            variadic_parameters = self.layout_variadic_arguments(
                resolved, convention, variadic_types
            )
            if varargs_count_register == "al":
                used_xmm = {
                    location.register
                    for parameter in (*resolved.parameters, *variadic_parameters)
                    for location in parameter.locations
                    if location.register and location.register.startswith("xmm")
                }
                varargs_sse_count = len(used_xmm)
        return PrototypeLayout(
            resolved, convention, variadic_parameters, varargs_sse_count
        )

    def layout_variadic_arguments(
        self,
        signature: FunctionSignature,
        convention: CallingConvention,
        argument_types: tuple[str | CType, ...],
    ) -> tuple[FunctionParameter, ...]:
        """Place actual ``...`` arguments, applying C default promotions."""
        promoted = tuple(self.types.promote_vararg(item) for item in argument_types)
        if not signature.variadic:
            raise ValueError("El prototipo no está marcado como variádico")
        if self.bits == 64 and convention.name in {"sysv_amd64", "__stdcall"}:
            fixed = signature.parameters
            all_parameters = tuple(
                FunctionParameter(
                    f"vararg_{index + 1}",
                    item.name,
                    data_type=item,
                    fixed=False,
                )
                for index, item in enumerate(promoted)
            )
            layout, _stack, _gp, _sse = self._layout_sysv_parameters(
                (*fixed, *all_parameters), signature.hidden_return
            )
            return tuple(
                replace(parameter, fixed=False) for parameter in layout[len(fixed) :]
            )
        if self.bits == 64:
            fixed = signature.parameters
            all_parameters = tuple(
                FunctionParameter(
                    f"vararg_{index + 1}",
                    item.name,
                    data_type=item,
                    fixed=False,
                )
                for index, item in enumerate(promoted)
            )
            layout, _stack, _gp, _sse = self._layout_windows64_parameters(
                (*fixed, *all_parameters), signature.hidden_return, convention
            )
            extras = layout[len(fixed) :]
            # Microsoft x64 varargs duplicate a floating-point argument in the
            # corresponding integer argument register as well as its XMM slot.
            duplicated: list[FunctionParameter] = []
            for parameter in extras:
                dtype = parameter.data_type
                locations = list(parameter.locations)
                if dtype and dtype.unwrapped().kind == TypeKind.FLOAT:
                    group_index = self._windows_group_index(parameter.locations)
                    if group_index is not None and group_index < len(
                        convention.argument_registers
                    ):
                        locations.append(
                            StorageLocation(
                                "register",
                                self.pointer_size,
                                register=convention.argument_registers[group_index],
                            )
                        )
                duplicated.append(
                    replace(parameter, locations=tuple(locations), fixed=False)
                )
            return tuple(duplicated)
        # The i386 C variadic conventions pass every argument on the stack;
        # use the fixed prototype's actual stack footprint (including a hidden
        # result pointer) before laying out the unnamed arguments.
        _fixed, cursor, _registers, _vectors = self._layout_x86_parameters(
            signature.parameters, signature.hidden_return, convention
        )
        cursor = max(cursor, convention.stack_argument_offset)
        result = []
        for index, dtype in enumerate(promoted):
            alignment = min(max(self.pointer_size, dtype.alignment or 1), 4)
            aligned = _align(cursor, alignment)
            size = _align(dtype.size or self.pointer_size, self.pointer_size)
            result.append(
                FunctionParameter(
                    f"vararg_{index + 1}",
                    dtype.name,
                    data_type=dtype,
                    locations=(StorageLocation("stack", size, stack_offset=aligned),),
                    fixed=False,
                )
            )
            cursor = aligned + size
        return tuple(result)

    def _resolve_parameter(self, parameter: FunctionParameter) -> FunctionParameter:
        dtype = self._resolve_type(parameter.data_type, parameter.type_name)
        base = dtype.unwrapped()
        if base.kind == TypeKind.ARRAY and base.target is not None:
            dtype = self.types.pointer(base.target)
        elif base.kind == TypeKind.FUNCTION:
            dtype = self.types.pointer(base)
        return replace(parameter, type_name=dtype.name, data_type=dtype)

    def _resolve_type(self, ctype: CType | None, name: str) -> CType:
        if ctype is not None:
            return ctype
        try:
            return self.types.resolve(name)
        except KeyError:
            return CType(name, TypeKind.UNKNOWN, None)

    def _needs_hidden_return(self, ctype: CType, convention: CallingConvention) -> bool:
        base = ctype.unwrapped()
        if not base.is_aggregate:
            return False
        if self.bits == 64 and convention.name in {"sysv_amd64", "__stdcall"}:
            return self._sysv_classes(base)[0] == "MEMORY"
        if self.bits == 64:
            return base.size not in {1, 2, 4, 8}
        if self.types.format != "PE" and convention.name in {
            "__cdecl",
            "__cdeclf",
            "__thiscall",
            "__regparm1",
            "__regparm2",
            "__regparm3",
        }:
            # The x86 GCC Cspecs mark struct results hidden for these models,
            # regardless of their byte size.
            return True
        return base.size is None or base.size > 8

    def _hidden_return_input_location(
        self,
        convention: CallingConvention,
        parameters: tuple[FunctionParameter, ...],
    ) -> StorageLocation:
        if self.bits == 64 and convention.name in {"sysv_amd64", "__stdcall"}:
            return StorageLocation(
                "register",
                self.pointer_size,
                register="rdi",
                indirect=True,
                hidden=True,
            )
        if self.bits == 64:
            index = sum(parameter.implicit for parameter in parameters)
            if index < len(convention.argument_registers):
                register = convention.argument_registers[index]
                return StorageLocation(
                    "register",
                    self.pointer_size,
                    register=register,
                    indirect=True,
                    hidden=True,
                )
            stack_offset = (
                convention.stack_argument_offset
                + max(0, index - len(convention.argument_registers)) * self.pointer_size
            )
            return StorageLocation(
                "stack",
                self.pointer_size,
                stack_offset=stack_offset,
                indirect=True,
                hidden=True,
            )
        has_implicit_this = (
            any(parameter.implicit for parameter in parameters)
            and convention.name == "__thiscall"
        )
        if convention.argument_registers and not has_implicit_this:
            return StorageLocation(
                "register",
                self.pointer_size,
                register=convention.argument_registers[0],
                indirect=True,
                hidden=True,
            )
        return StorageLocation(
            "stack",
            self.pointer_size,
            stack_offset=convention.stack_argument_offset,
            indirect=True,
            hidden=True,
        )

    def _hidden_pointer_result(
        self, convention: CallingConvention
    ) -> tuple[StorageLocation, ...]:
        register = "rax" if self.bits == 64 else "eax"
        return (
            StorageLocation(
                "register", self.pointer_size, register=register, hidden=True
            ),
        )

    def _return_locations(
        self, ctype: CType, convention: CallingConvention
    ) -> tuple[StorageLocation, ...]:
        base = ctype.unwrapped()
        if base.kind == TypeKind.VOID:
            return ()
        if self.bits == 64 and convention.name in {"sysv_amd64", "__stdcall"}:
            if base.kind == TypeKind.FLOAT:
                if base.size and base.size > 8:
                    return (StorageLocation("register", base.size, register="st0"),)
                return (StorageLocation("register", base.size or 8, register="xmm0"),)
            if base.kind == TypeKind.VECTOR:
                return self._sysv_return_locations(
                    self._sysv_classes(base), base.size or 0
                )
            if base.is_aggregate:
                classes = self._sysv_classes(base)
                return self._sysv_return_locations(classes, base.size or 0)
            size = base.size or self.pointer_size
            locations = [
                StorageLocation(
                    "register",
                    min(self.pointer_size, max(1, size - index * self.pointer_size)),
                    register=register,
                    value_offset=index * self.pointer_size,
                )
                for index, register in enumerate(self._SYSV_INTEGER_RETURNS)
                if index * self.pointer_size < size
            ]
            return tuple(locations)
        if self.bits == 64:
            if base.kind == TypeKind.FLOAT:
                return (StorageLocation("register", base.size or 8, register="xmm0"),)
            if base.kind == TypeKind.VECTOR:
                return (StorageLocation("register", base.size or 16, register="xmm0"),)
            return (
                StorageLocation("register", min(base.size or 8, 8), register="rax"),
            )
        if base.kind == TypeKind.FLOAT:
            return (StorageLocation("register", base.size or 10, register="st0"),)
        if base.kind == TypeKind.VECTOR:
            return (StorageLocation("register", base.size or 16, register="xmm0"),)
        size = base.size or self.pointer_size
        return tuple(
            StorageLocation(
                "register",
                min(self.pointer_size, max(1, size - index * self.pointer_size)),
                register=register,
                value_offset=index * self.pointer_size,
            )
            for index, register in enumerate(convention.integer_return_registers)
            if index * self.pointer_size < size
        )

    def _layout_sysv_parameters(
        self,
        parameters: tuple[FunctionParameter, ...],
        hidden_return: bool,
    ) -> tuple[tuple[FunctionParameter, ...], int, int, int]:
        gp_index = 1 if hidden_return else 0
        sse_index = 0
        stack_offset = 8
        located: list[FunctionParameter] = []
        for parameter in parameters:
            dtype = parameter.data_type or self._resolve_type(None, parameter.type_name)
            if parameter.implicit:
                dtype = parameter.data_type or self.types.pointer("void")
                classes = ["INTEGER"]
            else:
                classes = self._sysv_classes(dtype.unwrapped())
            if classes == ["MEMORY"]:
                stack_offset = _align(stack_offset, max(8, dtype.alignment or 1))
                stack_size = _align(dtype.size or 8, 8)
                locations = (
                    StorageLocation("stack", stack_size, stack_offset=stack_offset),
                )
                stack_offset += stack_size
            else:
                gp_needed = sum(item == "INTEGER" for item in classes)
                sse_needed = sum(item == "SSE" for item in classes)
                if gp_index + gp_needed > len(
                    self._SYSV_INTEGER_ARGUMENTS
                ) or sse_index + sse_needed > len(self._SYSV_FLOAT_ARGUMENTS):
                    stack_offset = _align(stack_offset, max(8, dtype.alignment or 1))
                    stack_size = _align(dtype.size or 8, 8)
                    locations = (
                        StorageLocation("stack", stack_size, stack_offset=stack_offset),
                    )
                    stack_offset += stack_size
                else:
                    locations_list: list[StorageLocation] = []
                    pending_sse: str | None = None
                    pending_sse_value_offset: int | None = None
                    for index, abi_class in enumerate(classes):
                        offset = index * 8
                        size = min(8, max(1, (dtype.size or 8) - offset))
                        if abi_class == "INTEGER":
                            register = self._SYSV_INTEGER_ARGUMENTS[gp_index]
                            gp_index += 1
                            locations_list.append(
                                StorageLocation(
                                    "register",
                                    size,
                                    register=register,
                                    value_offset=offset,
                                )
                            )
                            pending_sse = None
                            pending_sse_value_offset = None
                        elif abi_class == "SSE":
                            register = self._SYSV_FLOAT_ARGUMENTS[sse_index]
                            sse_index += 1
                            pending_sse = register
                            pending_sse_value_offset = offset
                            locations_list.append(
                                StorageLocation(
                                    "register",
                                    size,
                                    register=register,
                                    value_offset=offset,
                                )
                            )
                        elif abi_class == "SSEUP":
                            register = (
                                pending_sse or self._SYSV_FLOAT_ARGUMENTS[sse_index]
                            )
                            if pending_sse is None:
                                sse_index += 1
                                pending_sse_value_offset = offset
                            pending_sse = register
                            locations_list.append(
                                StorageLocation(
                                    "register",
                                    size,
                                    register=register,
                                    value_offset=offset,
                                    register_offset=(
                                        offset
                                        - (
                                            pending_sse_value_offset
                                            if pending_sse_value_offset is not None
                                            else offset
                                        )
                                    ),
                                )
                            )
                    locations = tuple(locations_list)
            located.append(replace(parameter, locations=locations))
        return tuple(located), stack_offset, gp_index, sse_index

    def _layout_windows64_parameters(
        self,
        parameters: tuple[FunctionParameter, ...],
        hidden_return: bool,
        convention: CallingConvention,
    ) -> tuple[tuple[FunctionParameter, ...], int, int, int]:
        implicit_count = sum(parameter.implicit for parameter in parameters)
        implicit_index = 0
        user_index = implicit_count + (1 if hidden_return else 0)
        stack_offset = convention.stack_argument_offset
        located: list[FunctionParameter] = []
        for parameter in parameters:
            dtype = parameter.data_type or self._resolve_type(None, parameter.type_name)
            base = dtype.unwrapped()
            indirect = False
            if parameter.implicit:
                index = implicit_index
                implicit_index += 1
                is_float = False
            else:
                index = user_index
                user_index += 1
                if base.kind == TypeKind.FLOAT:
                    is_float = True
                elif (base.is_aggregate and base.size not in {1, 2, 4, 8}) or (
                    base.kind == TypeKind.VECTOR and (base.size or 0) > 8
                ):
                    is_float = False
                    indirect = True
                else:
                    is_float = False
            size = (
                self.pointer_size
                if indirect
                else min(dtype.size or self.pointer_size, self.pointer_size)
            )
            if index < len(convention.register_argument_groups):
                group = convention.register_argument_groups[index]
                register = group[1] if is_float and len(group) > 1 else group[0]
                locations = (
                    StorageLocation(
                        "register", size, register=register, indirect=indirect
                    ),
                )
            else:
                locations = (
                    StorageLocation(
                        "stack", size, stack_offset=stack_offset, indirect=indirect
                    ),
                )
                stack_offset += self.pointer_size
            located.append(replace(parameter, locations=locations))
        return tuple(located), stack_offset, min(user_index + implicit_index, 4), 0

    def _layout_x86_parameters(
        self,
        parameters: tuple[FunctionParameter, ...],
        hidden_return: bool,
        convention: CallingConvention,
    ) -> tuple[tuple[FunctionParameter, ...], int, int, int]:
        registers = convention.argument_registers
        implicit_count = sum(parameter.implicit for parameter in parameters)
        implicit_index = 0
        # A fastcall/regparm hidden result pointer consumes the first argument
        # register. A thiscall `this` pointer precedes the hidden result pointer
        # and the latter is therefore placed in the first stack slot.
        hidden_in_register = bool(registers) and not (
            convention.name == "__thiscall" and implicit_count
        )
        user_register_index = implicit_count + int(hidden_return and hidden_in_register)
        stack_offset = (
            convention.stack_argument_offset
            + int(hidden_return and not hidden_in_register) * self.pointer_size
        )
        located: list[FunctionParameter] = []
        for parameter in parameters:
            dtype = parameter.data_type or self._resolve_type(None, parameter.type_name)
            base = dtype.unwrapped()
            # The i386 Cspecs' general pentries accept integer-class values
            # and aggregates up to one register wide; FP/vector values use the
            # stack in these non-SSE models.
            can_use_register = (
                base.kind not in {TypeKind.FLOAT, TypeKind.VECTOR}
                and (base.size or 0) <= self.pointer_size
            )
            if parameter.implicit and registers:
                register_index = implicit_index
                implicit_index += 1
                can_use_register = True
            else:
                register_index = user_register_index
                user_register_index += 1
            if can_use_register and register_index < len(registers):
                locations = (
                    StorageLocation(
                        "register",
                        min(dtype.size or self.pointer_size, self.pointer_size),
                        register=registers[register_index],
                    ),
                )
            else:
                alignment = min(max(dtype.alignment or 1, self.pointer_size), 4)
                stack_offset = _align(stack_offset, alignment)
                stack_size = _align(dtype.size or self.pointer_size, self.pointer_size)
                locations = (
                    StorageLocation("stack", stack_size, stack_offset=stack_offset),
                )
                stack_offset += stack_size
            located.append(replace(parameter, locations=locations))
        return tuple(located), stack_offset, user_register_index, 0

    def _sysv_classes(self, ctype: CType) -> list[str]:
        base = ctype.unwrapped()
        if base.size is None:
            return ["MEMORY"]
        if base.kind == TypeKind.FLOAT:
            if base.size in {4, 8}:
                return ["SSE"]
            return ["MEMORY"]
        if base.kind == TypeKind.VECTOR:
            if base.size == 16:
                return ["SSE", "SSEUP"]
            if base.size and base.size <= 8:
                return ["SSE"]
            return ["MEMORY"]
        if base.kind in {
            TypeKind.INTEGER,
            TypeKind.BOOL,
            TypeKind.POINTER,
            TypeKind.ENUM,
        }:
            slots = max(1, (base.size + 7) // 8)
            return ["INTEGER"] * slots if slots <= 2 else ["MEMORY"]
        if base.kind not in {TypeKind.STRUCT, TypeKind.UNION, TypeKind.ARRAY}:
            return ["INTEGER"]
        if base.size > 16:
            return ["MEMORY"]
        slot_count = max(1, (base.size + 7) // 8)
        classes = ["NO_CLASS"] * slot_count
        if not self._merge_type_into_classes(base, 0, classes):
            return ["MEMORY"]
        if any(item == "MEMORY" for item in classes):
            return ["MEMORY"]
        for index, item in enumerate(classes):
            if item == "NO_CLASS":
                classes[index] = "INTEGER"
            elif item == "SSEUP" and (
                index == 0 or classes[index - 1] not in {"SSE", "SSEUP"}
            ):
                classes[index] = "SSE"
        return classes

    def _merge_type_into_classes(
        self, ctype: CType, base_offset: int, classes: list[str]
    ) -> bool:
        base = ctype.unwrapped()
        size = base.size or 0
        if base.kind in {
            TypeKind.INTEGER,
            TypeKind.BOOL,
            TypeKind.POINTER,
            TypeKind.ENUM,
        }:
            return self._merge_range(classes, base_offset, size, "INTEGER")
        if base.kind == TypeKind.FLOAT:
            if size not in {4, 8}:
                return False
            return self._merge_range(classes, base_offset, size, "SSE")
        if base.kind == TypeKind.VECTOR:
            if size == 16:
                return self._merge_class(
                    classes, base_offset // 8, "SSE"
                ) and self._merge_class(classes, base_offset // 8 + 1, "SSEUP")
            if size <= 8:
                return self._merge_range(classes, base_offset, size, "SSE")
            return False
        if (
            base.kind == TypeKind.ARRAY
            and base.target is not None
            and base.count is not None
        ):
            if base.target.size is None:
                return False
            return all(
                self._merge_type_into_classes(
                    base.target, base_offset + index * base.target.size, classes
                )
                for index in range(base.count)
            )
        if base.kind in {TypeKind.STRUCT, TypeKind.UNION}:
            if not base.fields:
                return self._merge_range(classes, base_offset, size, "INTEGER")
            for field in base.fields:
                field_type = field.type.unwrapped()
                natural_alignment = field_type.alignment or 1
                if (
                    natural_alignment > 1
                    and (base_offset + field.offset) % natural_alignment
                ):
                    return False
                if not self._merge_type_into_classes(
                    field_type, base_offset + field.offset, classes
                ):
                    return False
            return True
        return self._merge_range(classes, base_offset, size, "INTEGER")

    def _merge_range(
        self, classes: list[str], byte_offset: int, size: int, incoming: str
    ) -> bool:
        if size <= 0:
            return True
        first = byte_offset // 8
        last = (byte_offset + size - 1) // 8
        if first < 0 or last >= len(classes):
            return False
        for index in range(first, last + 1):
            if not self._merge_class(classes, index, incoming):
                return False
        return True

    @staticmethod
    def _merge_class(classes: list[str], index: int, incoming: str) -> bool:
        if index < 0 or index >= len(classes):
            return False
        current = classes[index]
        if current in {"NO_CLASS", incoming}:
            classes[index] = incoming
        elif current == "MEMORY" or incoming == "MEMORY":
            classes[index] = "MEMORY"
        elif current == "INTEGER" or incoming == "INTEGER":
            classes[index] = "INTEGER"
        elif {current, incoming} <= {"SSE", "SSEUP"}:
            classes[index] = "SSE"
        else:
            classes[index] = "MEMORY"
        return True

    def _sysv_return_locations(
        self, classes: list[str], size: int
    ) -> tuple[StorageLocation, ...]:
        locations: list[StorageLocation] = []
        gp_index = 0
        sse_index = 0
        previous_sse: str | None = None
        previous_sse_value_offset: int | None = None
        for index, abi_class in enumerate(classes):
            part_size = min(8, max(1, size - index * 8))
            if abi_class == "INTEGER":
                if gp_index >= len(self._SYSV_INTEGER_RETURNS):
                    return ()
                locations.append(
                    StorageLocation(
                        "register",
                        part_size,
                        register=self._SYSV_INTEGER_RETURNS[gp_index],
                        value_offset=index * 8,
                    )
                )
                gp_index += 1
                previous_sse = None
                previous_sse_value_offset = None
            elif abi_class == "SSE":
                if sse_index >= len(self._SYSV_FLOAT_RETURNS):
                    return ()
                previous_sse = self._SYSV_FLOAT_RETURNS[sse_index]
                previous_sse_value_offset = index * 8
                sse_index += 1
                locations.append(
                    StorageLocation(
                        "register",
                        part_size,
                        register=previous_sse,
                        value_offset=index * 8,
                    )
                )
            elif abi_class == "SSEUP":
                if previous_sse is None:
                    if sse_index >= len(self._SYSV_FLOAT_RETURNS):
                        return ()
                    previous_sse = self._SYSV_FLOAT_RETURNS[sse_index]
                    previous_sse_value_offset = index * 8
                    sse_index += 1
                locations.append(
                    StorageLocation(
                        "register",
                        part_size,
                        register=previous_sse,
                        value_offset=index * 8,
                        register_offset=(
                            index * 8
                            - (
                                previous_sse_value_offset
                                if previous_sse_value_offset is not None
                                else index * 8
                            )
                        ),
                    )
                )
        return tuple(locations)

    @staticmethod
    def _windows_group_index(locations: tuple[StorageLocation, ...]) -> int | None:
        for location in locations:
            register = location.register or ""
            if register in {"rcx", "xmm0"}:
                return 0
            if register in {"rdx", "xmm1"}:
                return 1
            if register in {"r8", "xmm2"}:
                return 2
            if register in {"r9", "xmm3"}:
                return 3
        return None


def _align(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) & -alignment
