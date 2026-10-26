"""Executable loading and address/file-offset mapping."""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from pathlib import Path
from typing import Any, Iterable, Optional

from .model import PEUnwindInfo, PEUnwindOperation


class LoadError(ValueError):
    """Raised when a file cannot be loaded as a supported x86 image."""


class BinaryImage:
    """A lightweight, read-only view of a PE/ELF/Mach-O or raw x86 image."""

    def __init__(
        self,
        *,
        path: Path,
        raw: bytes,
        format_name: str,
        architecture: str,
        bits: int,
        image_base: int,
        entrypoint: Optional[int],
        sections: list,
        symbols: list,
        imports: list,
        exports: list,
        libraries: list[str],
        lief_binary: Any = None,
        is_fat_macho: bool = False,
        plt_stubs: Optional[dict[int, str]] = None,
        import_slots: Optional[dict[int, str]] = None,
        macho_function_starts: Optional[list[int]] = None,
        pe_exception_function_ranges: Optional[list[tuple[int, int]]] = None,
        pe_tls_callbacks: Optional[list[int]] = None,
        pe_unwind_infos: Optional[dict[int, PEUnwindInfo]] = None,
    ) -> None:
        self.path = path
        self.raw = raw
        self._section_index_cache: dict[bool, tuple] = {}
        self.format = format_name
        self.architecture = architecture
        self.bits = bits
        self.image_base = image_base
        self.entrypoint = entrypoint
        self.sections = sections
        self.symbols = symbols
        self.imports = imports
        self.exports = exports
        self.libraries = libraries
        self.lief_binary = lief_binary
        self.is_fat_macho = is_fat_macho
        self.plt_stubs = dict(plt_stubs or {})
        self.import_slots = dict(import_slots or {})
        self.macho_function_starts = sorted(set(macho_function_starts or []))
        self.pe_exception_function_ranges = sorted(set(pe_exception_function_ranges or []))
        self.pe_tls_callbacks = sorted(set(pe_tls_callbacks or []))
        self.pe_unwind_infos = dict(pe_unwind_infos or {})
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.pe = None
        if self.format == "PE":
            try:
                import pefile
                self.pe = pefile.PE(str(path))
            except ImportError:
                pass
            except Exception:
                pass

    @classmethod
    def load(
        cls,
        file_path: str | Path,
        *,
        raw_arch: Optional[str] = None,
        base: int = 0,
    ) -> "BinaryImage":
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise LoadError(f"No existe el archivo: {path}")

        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise LoadError(f"No se pudo leer {path}: {exc}") from exc
        if not raw:
            raise LoadError("El archivo está vacío.")

        if raw_arch:
            architecture, bits = _normalize_architecture(raw_arch)
            from .model import Section

            section = Section(
                name=".raw",
                start=base,
                virtual_size=len(raw),
                file_offset=0,
                data=raw,
                executable=True,
                readable=True,
                writable=False,
            )
            return cls(
                path=path,
                raw=raw,
                format_name="RAW",
                architecture=architecture,
                bits=bits,
                image_base=base,
                entrypoint=base,
                sections=[section],
                symbols=[],
                imports=[],
                exports=[],
                libraries=[],
            )

        try:
            import lief
        except ImportError as exc:  # pragma: no cover - dependency declared by the package
            raise LoadError("Falta LIEF. Instálalo con: python -m pip install lief") from exc

        try:
            parsed = lief.parse(str(path))
        except Exception as exc:
            raise LoadError(
                f"LIEF no pudo reconocer el formato. Para un volcado crudo, usa --raw-arch x86 o x86_64. ({exc})"
            ) from exc
        if parsed is None:
            raise LoadError(
                "Formato no reconocido. Para un volcado crudo, usa --raw-arch x86 o x86_64."
            )

        parsed, is_fat_macho = _choose_macho_slice(parsed)
        format_name = _format_name(parsed)
        architecture, bits = _detect_architecture(parsed, format_name)
        image_base = _image_base(parsed, format_name)

        from .model import Section, Symbol

        sections: list[Section] = []
        for raw_section in getattr(parsed, "sections", ()):
            try:
                data = bytes(raw_section.content)
            except (AttributeError, TypeError, ValueError):
                data = b""
            try:
                virtual_address = int(raw_section.virtual_address)
            except (AttributeError, TypeError, ValueError):
                virtual_address = 0
            try:
                virtual_size = int(
                    getattr(raw_section, "virtual_size", getattr(raw_section, "size", len(data)))
                )
            except (AttributeError, TypeError, ValueError):
                virtual_size = len(data)

            if format_name == "PE":
                start = image_base + virtual_address
                file_offset = _first_int_attr(
                    raw_section, ("pointerto_raw_data", "offset"), default=None
                )
                flags = _first_int_attr(raw_section, ("characteristics",), default=0) or 0
                executable = bool(flags & 0x20000000)
                readable = bool(flags & 0x40000000)
                writable = bool(flags & 0x80000000)
            elif format_name == "ELF":
                start = virtual_address
                file_offset = _first_int_attr(raw_section, ("offset",), default=None)
                flags = _first_int_attr(raw_section, ("flags",), default=0) or 0
                executable = bool(flags & 0x4)  # SHF_EXECINSTR
                readable = bool(flags & 0x2)  # SHF_ALLOC (best available section-level proxy)
                writable = bool(flags & 0x1)  # SHF_WRITE
            else:  # Mach-O
                start = virtual_address
                file_offset = _first_int_attr(raw_section, ("offset",), default=None)
                executable, readable, writable = _macho_permissions(raw_section)

            if format_name == "PE" and file_offset == 0 and not data:
                file_offset = None
            if file_offset is not None and file_offset < 0:
                file_offset = None
            if not executable and _looks_like_code_section(str(getattr(raw_section, "name", ""))):
                executable = True
            if virtual_size <= 0:
                virtual_size = len(data)
            section_name = str(getattr(raw_section, "name", "")) or "<sin nombre>"
            sections.append(
                Section(
                    name=section_name,
                    start=start,
                    virtual_size=virtual_size,
                    file_offset=file_offset,
                    data=data,
                    executable=executable,
                    readable=readable,
                    writable=writable,
                )
            )

        if not sections:
            raise LoadError("El archivo se reconoció, pero no contiene secciones analizables.")

        entrypoint = _entrypoint(parsed, format_name, image_base)
        symbols, imports, exports = _collect_symbols(parsed, format_name, image_base, sections)
        plt_stubs, import_slots = _collect_elf_plt_metadata(parsed, format_name)
        for address, name in plt_stubs.items():
            if not _inside_any_section(address, sections):
                continue
            if not any(symbol.address == address and symbol.name == name for symbol in symbols):
                symbols.append(Symbol(name, address, kind="thunk", imported=True))
        symbols.sort(key=lambda item: (item.address, item.name.lower()))
        libraries = _collect_libraries(parsed)
        macho_function_starts = _collect_macho_function_starts(parsed, format_name)
        pe_exception_function_ranges = _collect_pe_exception_function_ranges(
            parsed, format_name, bits, image_base, sections
        )
        pe_tls_callbacks = _collect_pe_tls_callbacks(parsed, format_name, image_base)
        pe_unwind_infos = _collect_pe_unwind_infos(
            parsed,
            format_name,
            bits,
            image_base,
            sections,
            pe_exception_function_ranges,
        )

        return cls(
            path=path,
            raw=raw,
            format_name=format_name,
            architecture=architecture,
            bits=bits,
            image_base=image_base,
            entrypoint=entrypoint,
            sections=sections,
            symbols=symbols,
            imports=imports,
            exports=exports,
            libraries=libraries,
            lief_binary=parsed,
            is_fat_macho=is_fat_macho,
            plt_stubs=plt_stubs,
            import_slots=import_slots,
            macho_function_starts=macho_function_starts,
            pe_exception_function_ranges=pe_exception_function_ranges,
            pe_tls_callbacks=pe_tls_callbacks,
            pe_unwind_infos=pe_unwind_infos,
        )

    @property
    def resources(self):
        """Devuelve el gestor de recursos de LIEF para binarios PE, o None."""
        if self.format == "PE" and self.lief_binary is not None:
            try:
                return self.lief_binary.resources_manager
            except Exception:
                pass
        return None

    @property
    def executable_sections(self):
        return [section for section in self.sections if section.executable and section.data]

    def read_memory(self, address: int, size: int) -> bytes:
        section = self.section_at(address, file_backed=True)
        if section is None:
            raise ValueError(f"Address {address:#x} is not mapped.")
        offset = address - section.start
        if offset + size > len(section.data):
            raise ValueError(f"Read out of bounds at {address:#x}")
        return bytes(section.data[offset:offset+size])

    def _section_index(self, file_backed: bool):
        """Índice ordenado de secciones para ``section_at`` (P1 del informe).

        Devuelve ``(orden, arranques, máximos_prefijo, finales)`` donde
        ``orden[k]`` es el índice original de la k-ésima sección por dirección
        de inicio y ``máximos_prefijo[k]`` el mayor ``end`` de las ``k``
        primeras, que es lo que permite podar la búsqueda hacia atrás.
        """
        cached = self._section_index_cache.get(file_backed)
        if cached is not None:
            return cached
        sections = self.sections
        ends = [
            section.data_end if file_backed else section.end
            for section in sections
        ]
        order = sorted(range(len(sections)), key=lambda i: sections[i].start)
        starts = [sections[i].start for i in order]
        prefix_max = [0] * (len(order) + 1)
        for position, original in enumerate(order):
            previous = prefix_max[position]
            end = ends[original]
            prefix_max[position + 1] = end if end > previous else previous
        cached = (order, starts, prefix_max, ends)
        self._section_index_cache[file_backed] = cached
        return cached

    def section_at(self, address: int, *, file_backed: bool = False):
        """Sección que contiene ``address``, o ``None``.

        Equivalente al recorrido lineal anterior (devuelve la primera sección
        en el orden de ``self.sections`` que la contiene), pero con búsqueda
        binaria sobre los arranques: era el punto caliente del análisis
        (385.927 llamadas y 1,49 M de ``Section.contains`` en el informe).
        """
        order, starts, prefix_max, ends = self._section_index(file_backed)
        position = bisect_right(starts, address)
        best: int | None = None
        while position > 0 and prefix_max[position] > address:
            original = order[position - 1]
            if ends[original] > address and (best is None or original < best):
                best = original
            position -= 1
        return None if best is None else self.sections[best]

    def patch_to_file(self, address: int, patch: bytes, output_path: str | Path) -> Path:
        """Write a patched copy; never modifies the input image."""
        if not patch:
            raise ValueError("No hay bytes que parchear.")
        if self.is_fat_macho:
            raise ValueError("El parcheo de archivos Mach-O universales todavía no está soportado.")
        output = Path(output_path).expanduser().resolve()
        if output == self.path:
            raise ValueError("La salida no puede ser el archivo original; elige otra ruta.")

        section = self.section_at(address, file_backed=True)
        if section is None:
            raise ValueError(f"La dirección {address:#x} no pertenece a bytes cargados de una sección.")
        if section.file_offset is None:
            raise ValueError("LIEF no proporcionó el desplazamiento de archivo para esta sección.")
        relative = address - section.start
        if relative + len(patch) > len(section.data):
            raise ValueError("El parche cruza el final de los bytes de esta sección.")
        file_offset = section.file_offset + relative
        if file_offset < 0 or file_offset + len(patch) > len(self.raw):
            raise ValueError("El rango del parche no corresponde a bytes existentes en el archivo.")

        updated = bytearray(self.raw)
        updated[file_offset : file_offset + len(patch)] = patch
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(updated)
        except OSError as exc:
            raise ValueError(f"No se pudo escribir {output}: {exc}") from exc
        return output


def _normalize_architecture(value: str) -> tuple[str, int]:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"x86", "i386", "i486", "i586", "i686", "32", "x86_32"}:
        return "x86", 32
    if normalized in {"x86_64", "amd64", "x64", "64", "x8664"}:
        return "x86_64", 64
    raise LoadError("Arquitectura no válida. Usa x86 (32 bits) o x86_64 (64 bits).")


def _enum_name(value: Any) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if name:
        return str(name).upper()
    return str(value).upper().split(".")[-1]


def _format_name(binary: Any) -> str:
    value = getattr(binary, "format", "")
    name = _enum_name(value)
    if "ELF" in name:
        return "ELF"
    if "PE" in name:
        return "PE"
    if "MACHO" in name or "MACH_O" in name:
        return "Mach-O"
    return name or type(binary).__name__


def _choose_macho_slice(binary: Any) -> tuple[Any, bool]:
    """Pick the first x86 slice of a universal Mach-O, if LIEF returned one."""
    type_name = type(binary).__name__.lower()
    if "fatbinary" not in type_name:
        return binary, False
    try:
        slices = list(binary.it_binaries)
    except (AttributeError, TypeError):
        try:
            slices = list(binary)
        except TypeError:
            slices = []
    supported = []
    for item in slices:
        try:
            _detect_architecture(item, "Mach-O")
            supported.append(item)
        except LoadError:
            continue
    if not supported:
        raise LoadError("El Mach-O universal no contiene una variante x86/x86-64.")
    # Prefer x86-64, which is the common host slice, while keeping both modes supported.
    chosen = next((item for item in supported if _detect_architecture(item, "Mach-O")[1] == 64), supported[0])
    return chosen, True


def _detect_architecture(binary: Any, format_name: str) -> tuple[str, int]:
    header = getattr(binary, "header", None)
    if format_name == "ELF":
        raw_arch = getattr(header, "machine_type", None)
    elif format_name == "PE":
        raw_arch = getattr(header, "machine", None)
    else:
        raw_arch = getattr(header, "cpu_type", None)

    machine = _enum_name(raw_arch)
    # Test 64-bit names first: X86_64 also contains the substring X86.
    if any(token in machine for token in ("X86_64", "AMD64", "EM64T")):
        return "x86_64", 64
    if any(token in machine for token in ("I386", "I486", "I586", "I686", "386", "X86")):
        return "x86", 32
    raise LoadError(
        f"Arquitectura no compatible ({machine or 'desconocida'}). Solo se admite x86 de 32 bits o x86-64."
    )


def _image_base(binary: Any, format_name: str) -> int:
    if format_name == "PE":
        optional_header = getattr(binary, "optional_header", None)
        value = getattr(optional_header, "imagebase", None)
        if value is not None:
            return int(value)
    value = getattr(binary, "imagebase", 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _entrypoint(binary: Any, format_name: str, image_base: int) -> Optional[int]:
    if format_name == "PE":
        optional_header = getattr(binary, "optional_header", None)
        rva = getattr(optional_header, "addressof_entrypoint", None)
        if rva is not None:
            rva = int(rva)
            return image_base + rva if rva else None
    try:
        value = int(getattr(binary, "entrypoint", 0) or 0)
    except (TypeError, ValueError):
        return None
    return value or None


def _first_int_attr(obj: Any, names: Iterable[str], *, default: Optional[int]) -> Optional[int]:
    for name in names:
        try:
            value = getattr(obj, name)
            if value is not None:
                return int(value)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return default


def _macho_permissions(section: Any) -> tuple[bool, bool, bool]:
    segment = getattr(section, "segment", None)
    protection = _first_int_attr(segment, ("init_protection",), default=0) or 0
    name = str(getattr(section, "name", "")).lower()
    segment_name = str(getattr(section, "segment_name", "")).lower()
    executable = bool(protection & 0x4) or name in {"__text", "__stubs", "__stub_helper"}
    readable = bool(protection & 0x1) or segment_name == "__text"
    writable = bool(protection & 0x2)
    return executable, readable, writable


def _looks_like_code_section(name: str) -> bool:
    lowered = name.lower()
    return lowered in {".text", "__text", "text", "code"} or lowered.endswith(".text")


def _normalize_address(value: Any, format_name: str, image_base: int) -> int:
    try:
        address = int(value)
    except (TypeError, ValueError):
        return 0
    if format_name == "PE" and address and address < image_base:
        return image_base + address
    return address


def _inside_any_section(address: int, sections: list) -> bool:
    return any(section.contains(address) for section in sections)


def _collect_macho_function_starts(binary: Any, format_name: str) -> list[int]:
    """Translate LIEF's LC_FUNCTION_STARTS offsets to virtual addresses."""
    if format_name != "Mach-O":
        return []

    text_base = None
    for segment in _iter_attr(binary, "segments"):
        if str(getattr(segment, "name", "")).strip() == "__TEXT":
            text_base = _first_int_attr(segment, ("virtual_address",), default=None)
            break
    if text_base is None:
        return []

    try:
        command = getattr(binary, "function_starts", None)
    except (AttributeError, RuntimeError, TypeError):
        command = None
    if command is None:
        commands = [
            item
            for item in _iter_attr(binary, "commands")
            if "functionstarts" in type(item).__name__.casefold()
        ]
    elif isinstance(command, (list, tuple)):
        commands = list(command)
    else:
        commands = [command]

    starts = set()
    for item in commands:
        for offset in _iter_attr(item, "functions"):
            try:
                relative_address = int(offset)
            except (TypeError, ValueError):
                continue
            if relative_address >= 0:
                starts.add(text_base + relative_address)
    return sorted(starts)


def _collect_pe_exception_function_ranges(
    binary: Any,
    format_name: str,
    bits: int,
    image_base: int,
    sections: Iterable = (),
) -> list[tuple[int, int]]:
    """Read x64 PE RUNTIME_FUNCTION entries, following Ghidra's PE loader.

    LIEF exposes parsed exception functions on some binaries/versions. When it
    returns none, fall back to the PE exception data directory: each x64
    RUNTIME_FUNCTION is three little-endian DWORD RVAs in ``.pdata``.
    """
    if format_name != "PE" or bits != 64:
        return []

    ranges = set()
    for entry in _iter_attr(binary, "exception_functions"):
        start_rva = _first_int_attr(entry, ("rva_start", "begin_address"), default=None)
        end_rva = _first_int_attr(entry, ("rva_end", "end_address"), default=None)
        if start_rva is None or end_rva is None or start_rva <= 0 or end_rva <= start_rva:
            continue
        if _has_chained_pe_unwind_info(entry):
            continue
        ranges.add((image_base + start_rva, image_base + end_rva))

    directory_rva, directory_size = _pe_exception_directory(binary)
    if directory_rva is not None and directory_size:
        table = _read_pe_rva(sections, image_base, directory_rva, directory_size)
        if table is not None:
            entry_count = min(len(table) // 12, 200_000)
            for index in range(entry_count):
                offset = index * 12
                start_rva = int.from_bytes(table[offset : offset + 4], "little")
                end_rva = int.from_bytes(table[offset + 4 : offset + 8], "little")
                unwind_rva = int.from_bytes(table[offset + 8 : offset + 12], "little")
                if start_rva == 0 or end_rva <= start_rva:
                    continue
                if unwind_rva:
                    unwind_header = _read_pe_rva(sections, image_base, unwind_rva, 1)
                    if unwind_header is not None:
                        unwind_flags = (unwind_header[0] >> 3) & 0x1F
                        if unwind_flags & 0x4:  # UNW_FLAG_CHAININFO
                            continue
                ranges.add((image_base + start_rva, image_base + end_rva))
    return sorted(ranges)


def _pe_exception_directory(binary: Any) -> tuple[Optional[int], int]:
    for directory in _iter_attr(binary, "data_directories"):
        if "EXCEPTION" not in _enum_name(getattr(directory, "type", None)):
            continue
        rva = _first_int_attr(directory, ("rva", "virtual_address"), default=None)
        size = _first_int_attr(directory, ("size",), default=0) or 0
        return rva, size

    try:
        directory = getattr(binary, "exceptions_dir", None)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        directory = None
    if directory is not None:
        rva = _first_int_attr(directory, ("rva", "virtual_address"), default=None)
        size = _first_int_attr(directory, ("size",), default=0) or 0
        return rva, size
    return None, 0


def _read_pe_rva(sections: Iterable, image_base: int, rva: int, size: int) -> Optional[bytes]:
    if rva < 0 or size < 0:
        return None
    for section in sections:
        section_rva = int(section.start) - image_base
        relative = rva - section_rva
        if relative >= 0 and relative + size <= len(section.data):
            return section.data[relative : relative + size]
    return None


def _collect_pe_unwind_infos(
    binary: Any,
    format_name: str,
    bits: int,
    image_base: int,
    sections: Iterable,
    function_ranges: Iterable[tuple[int, int]],
) -> dict[int, PEUnwindInfo]:
    """Parse UNWIND_INFO for function starts retained from the PE exception table."""
    if format_name != "PE" or bits != 64:
        return {}

    valid_starts = {start for start, _end in function_ranges}
    unwind_by_start: dict[int, int] = {}
    for entry in _iter_attr(binary, "exception_functions"):
        start_rva = _first_int_attr(entry, ("rva_start", "begin_address"), default=None)
        unwind_rva = _first_int_attr(entry, ("unwind_rva",), default=None)
        if start_rva is not None and unwind_rva:
            start = image_base + start_rva
            if start in valid_starts:
                unwind_by_start[start] = unwind_rva

    directory_rva, directory_size = _pe_exception_directory(binary)
    if directory_rva is not None and directory_size:
        table = _read_pe_rva(sections, image_base, directory_rva, directory_size)
        if table is not None:
            entry_count = min(len(table) // 12, 200_000)
            for index in range(entry_count):
                offset = index * 12
                start_rva = int.from_bytes(table[offset : offset + 4], "little")
                unwind_rva = int.from_bytes(table[offset + 8 : offset + 12], "little")
                start = image_base + start_rva
                if start_rva and unwind_rva and start in valid_starts:
                    unwind_by_start[start] = unwind_rva

    result = {}
    for start, unwind_rva in unwind_by_start.items():
        info = _parse_pe_unwind_info(unwind_rva, sections, image_base)
        if info is not None and not (info.flags & 0x4):
            result[start] = info
    return result


def _parse_pe_unwind_info(
    unwind_rva: int, sections: Iterable, image_base: int
) -> Optional[PEUnwindInfo]:
    """Decode the x64 UNWIND_INFO header/codes and summarize stack effects."""
    header = _read_pe_rva(sections, image_base, unwind_rva, 4)
    if header is None:
        return None

    version = header[0] & 0x07
    flags = (header[0] >> 3) & 0x1F
    prolog_size = header[1]
    code_slot_count = header[2]
    frame_register_id = header[3] & 0x0F
    frame_offset = (header[3] >> 4) * 16
    code_data = _read_pe_rva(sections, image_base, unwind_rva + 4, code_slot_count * 2)
    if code_data is None:
        return None

    registers = (
        "RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI",
        "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15",
    )
    frame_register = registers[frame_register_id] if frame_register_id else None
    operations: list[PEUnwindOperation] = []
    saved_registers: list[str] = []
    stack_allocation = 0
    stack_allocation_known = True

    def slot_value(index: int, slot_count: int) -> Optional[int]:
        start = index * 2
        end = start + slot_count * 2
        if start < 0 or end > len(code_data):
            return None
        return int.from_bytes(code_data[start:end], "little")

    index = 0
    while index < code_slot_count:
        code_offset = code_data[index * 2]
        encoded_op = code_data[index * 2 + 1]
        opcode = encoded_op & 0x0F
        op_info = encoded_op >> 4
        name = f"UWOP_{opcode}"
        operand = None
        register = None
        extra_slots = 0
        stack_effect = 0
        known = True

        if opcode == 0:  # UWOP_PUSH_NONVOL
            name = "UWOP_PUSH_NONVOL"
            register = registers[op_info]
            saved_registers.append(register)
            stack_effect = 8
        elif opcode == 1:  # UWOP_ALLOC_LARGE
            name = "UWOP_ALLOC_LARGE"
            if op_info == 0:
                operand = slot_value(index + 1, 1)
                extra_slots = 1
                stack_effect = operand * 8 if operand is not None else 0
            elif op_info == 1:
                operand = slot_value(index + 1, 2)
                extra_slots = 2
                stack_effect = operand if operand is not None else 0
            else:
                known = False
            if operand is None:
                known = False
        elif opcode == 2:  # UWOP_ALLOC_SMALL
            name = "UWOP_ALLOC_SMALL"
            stack_effect = op_info * 8 + 8
            operand = stack_effect
        elif opcode == 3:  # UWOP_SET_FPREG
            name = "UWOP_SET_FPREG"
        elif opcode == 4:  # UWOP_SAVE_NONVOL
            name = "UWOP_SAVE_NONVOL"
            register = registers[op_info]
            saved_registers.append(register)
            raw_offset = slot_value(index + 1, 1)
            extra_slots = 1
            operand = raw_offset * 8 if raw_offset is not None else None
            known = operand is not None
        elif opcode == 5:  # UWOP_SAVE_NONVOL_FAR
            name = "UWOP_SAVE_NONVOL_FAR"
            register = registers[op_info]
            saved_registers.append(register)
            operand = slot_value(index + 1, 2)
            extra_slots = 2
            known = operand is not None
        elif opcode in (6, 7) and version == 1:  # Ghidra's legacy SAVE_XMM codes
            name = "UWOP_SAVE_XMM" if opcode == 6 else "UWOP_SAVE_XMM_FAR"
            register = f"XMM{op_info}"
            saved_registers.append(register)
            if opcode == 6:
                raw_offset = slot_value(index + 1, 1)
                extra_slots = 1
                operand = raw_offset * 8 if raw_offset is not None else None
            else:
                operand = slot_value(index + 1, 2)
                extra_slots = 2
            known = operand is not None
        elif opcode == 6 and version == 2:
            name = "UWOP_EPILOG"
        elif opcode == 7 and version == 2:
            name = "UWOP_SPARE_CODE"
            known = False
        elif opcode == 8:  # UWOP_SAVE_XMM128
            name = "UWOP_SAVE_XMM128"
            register = f"XMM{op_info}"
            saved_registers.append(register)
            raw_offset = slot_value(index + 1, 1)
            extra_slots = 1
            operand = raw_offset * 16 if raw_offset is not None else None
            known = operand is not None
        elif opcode == 9:  # UWOP_SAVE_XMM128_FAR
            name = "UWOP_SAVE_XMM128_FAR"
            register = f"XMM{op_info}"
            saved_registers.append(register)
            operand = slot_value(index + 1, 2)
            extra_slots = 2
            known = operand is not None
        elif opcode == 10:  # UWOP_PUSH_MACHFRAME
            name = "UWOP_PUSH_MACHFRAME"
            if op_info in (0, 1):
                stack_effect = 40 if op_info == 0 else 48
                operand = stack_effect
            else:
                known = False
        else:
            known = False

        operations.append(
            PEUnwindOperation(
                opcode=opcode,
                name=name,
                code_offset=code_offset,
                op_info=op_info,
                operand=operand,
                register=register,
            )
        )
        if not known:
            stack_allocation_known = False
        stack_allocation += stack_effect
        index += 1 + extra_slots

    handler_rva = None
    if flags & 0x03:  # UNW_FLAG_EHANDLER or UNW_FLAG_UHANDLER
        handler_offset = (4 + code_slot_count * 2 + 3) & ~3
        handler = _read_pe_rva(sections, image_base, unwind_rva + handler_offset, 4)
        if handler is not None:
            handler_rva = int.from_bytes(handler, "little")

    return PEUnwindInfo(
        unwind_rva=unwind_rva,
        version=version,
        flags=flags,
        prolog_size=prolog_size,
        frame_register=frame_register,
        frame_offset=frame_offset,
        code_slot_count=code_slot_count,
        operations=tuple(operations),
        stack_allocation_bytes=stack_allocation if stack_allocation_known else None,
        saved_registers=tuple(dict.fromkeys(saved_registers)),
        handler_rva=handler_rva,
    )


def _has_chained_pe_unwind_info(entry: Any) -> bool:
    """Check UNW_FLAG_CHAININFO (0x4) without depending on LIEF's enum type."""
    try:
        unwind_info = getattr(entry, "unwind_info", None)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
    if unwind_info is None:
        return False

    try:
        chained = getattr(unwind_info, "has_chained_unwind_info", None)
        if chained is not None and bool(chained() if callable(chained) else chained):
            return True
    except (RuntimeError, TypeError, ValueError):
        pass

    flags = _first_int_attr(unwind_info, ("flags", "unwind_flags"), default=None)
    if flags is not None and flags & 0x4:
        return True
    try:
        flag_text = str(getattr(unwind_info, "flags", "")).upper().replace("_", "")
    except (RuntimeError, TypeError, ValueError):
        flag_text = ""
    return "CHAININFO" in flag_text or "CHAINED" in flag_text


def _collect_pe_tls_callbacks(binary: Any, format_name: str, image_base: int) -> list[int]:
    """Return PE TLS callback VAs, bounded like Ghidra's TLS markup loop."""
    if format_name != "PE":
        return []
    try:
        tls = getattr(binary, "tls", None)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return []
    if tls is None:
        return []

    callbacks = []
    for raw_address in _iter_attr(tls, "callbacks")[:20]:
        try:
            address = _normalize_address(int(raw_address), "PE", image_base)
        except (TypeError, ValueError):
            continue
        if address == 0:
            break
        if address > 0 and address not in callbacks:
            callbacks.append(address)
    return callbacks


def _collect_symbols(binary: Any, format_name: str, image_base: int, sections: list):
    from .model import Symbol

    symbols: list[Symbol] = []
    imports: list[Symbol] = []
    exports: list[Symbol] = []
    seen_symbols: set[tuple[int, str]] = set()
    seen_imports: set[tuple[int, str, Optional[str]]] = set()
    seen_exports: set[tuple[int, str]] = set()

    def add_symbol(name: Any, value: Any, kind: str, size: Any = 0) -> None:
        text = str(name or "").strip()
        address = _normalize_address(value, format_name, image_base)
        if not text or address <= 0 or not _inside_any_section(address, sections):
            return
        key = (address, text)
        if key in seen_symbols:
            return
        seen_symbols.add(key)
        try:
            size_value = int(size or 0)
        except (TypeError, ValueError):
            size_value = 0
        symbols.append(Symbol(text, address, kind=kind, size=size_value))

    def add_import(name: Any, value: Any = 0, library: Any = None) -> None:
        text = str(name or "").strip()
        if not text:
            return
        address = _normalize_address(value, format_name, image_base)
        if address < 0:
            address = 0
        lib = str(library) if library else None
        key = (address, text, lib)
        if key in seen_imports:
            return
        seen_imports.add(key)
        imports.append(Symbol(text, address, kind="import", imported=True, library=lib))

    def add_export(name: Any, value: Any, size: Any = 0) -> None:
        text = str(name or "").strip()
        address = _normalize_address(value, format_name, image_base)
        if not text or not address:
            return
        key = (address, text)
        if key in seen_exports:
            return
        seen_exports.add(key)
        try:
            size_value = int(size or 0)
        except (TypeError, ValueError):
            size_value = 0
        exports.append(Symbol(text, address, kind="export", size=size_value))
        add_symbol(text, address, "function", size_value)

    # ELF, Mach-O and PE expose different symbol collections; duck-typing keeps
    # this adapter compatible with LIEF's changing enum/type wrappers.
    try:
        raw_symbols = list(getattr(binary, "symbols", ()))
    except (TypeError, RuntimeError):
        raw_symbols = []
    for raw_symbol in raw_symbols:
        name = getattr(raw_symbol, "name", "")
        value = getattr(raw_symbol, "value", getattr(raw_symbol, "address", 0))
        kind_name = _enum_name(getattr(raw_symbol, "type", None)) or "symbol"
        add_symbol(name, value, kind_name.lower(), getattr(raw_symbol, "size", 0))

    for function in _iter_attr(binary, "functions"):
        add_symbol(
            getattr(function, "name", ""),
            getattr(function, "address", getattr(function, "value", 0)),
            "function",
            getattr(function, "size", 0),
        )

    for function in _iter_attr(binary, "exported_functions"):
        add_export(
            getattr(function, "name", ""),
            getattr(function, "address", getattr(function, "value", 0)),
            getattr(function, "size", 0),
        )

    for exported in _iter_attr(binary, "exported_symbols"):
        add_export(
            getattr(exported, "name", ""),
            getattr(exported, "value", getattr(exported, "address", 0)),
            getattr(exported, "size", 0),
        )

    if format_name == "PE":
        for library in _iter_attr(binary, "imports"):
            library_name = getattr(library, "name", None)
            for entry in _iter_attr(library, "entries"):
                entry_name = getattr(entry, "name", None)
                if not entry_name:
                    ordinal = getattr(entry, "ordinal", None)
                    entry_name = f"ordinal_{ordinal}" if ordinal is not None else ""
                add_import(entry_name, getattr(entry, "iat_address", 0), library_name)
    else:
        for function in _iter_attr(binary, "imported_functions"):
            add_import(
                getattr(function, "name", ""),
                getattr(function, "address", getattr(function, "value", 0)),
            )
        for symbol in _iter_attr(binary, "imported_symbols"):
            add_import(
                getattr(symbol, "name", ""),
                getattr(symbol, "value", getattr(symbol, "address", 0)),
            )

    symbols.sort(key=lambda item: (item.address, item.name.lower()))
    imports.sort(key=lambda item: (item.address, item.name.lower()))
    exports.sort(key=lambda item: (item.address, item.name.lower()))
    return symbols, imports, exports


def _collect_elf_plt_metadata(binary: Any, format_name: str) -> tuple[dict[int, str], dict[int, str]]:
    """Map ELF PLT entries/GOT slots back to dynamic symbols when possible.

    This is deliberately limited to the common x86 layout: one resolver entry
    followed by fixed-width PLT stubs, or a `.plt.sec` section with one stub per
    PLT relocation. Unknown layouts are skipped instead of guessed.
    """
    if format_name != "ELF":
        return {}, {}
    relocations = _iter_attr(binary, "pltgot_relocations")
    if not relocations:
        return {}, {}
    sections = _iter_attr(binary, "sections")
    section_by_name = {str(getattr(section, "name", "")): section for section in sections}
    plt_section = section_by_name.get(".plt.sec")
    has_reserved_entry = plt_section is None
    if plt_section is None:
        plt_section = section_by_name.get(".plt")
    if plt_section is None:
        return {}, {}

    size = _first_int_attr(plt_section, ("size",), default=0) or 0
    virtual_address = _first_int_attr(plt_section, ("virtual_address",), default=0) or 0
    if size <= 0:
        return {}, {}
    denominator = len(relocations) + (1 if has_reserved_entry else 0)
    if denominator <= 0 or size % denominator:
        return {}, {}
    stride = size // denominator
    if stride < 4 or stride > 64:
        return {}, {}

    plt_stubs: dict[int, str] = {}
    import_slots: dict[int, str] = {}
    first_stub_index = 1 if has_reserved_entry else 0
    for index, relocation in enumerate(relocations):
        slot = _first_int_attr(relocation, ("address",), default=0) or 0
        raw_symbol = getattr(relocation, "symbol", None)
        name = str(getattr(raw_symbol, "name", "") or "").strip()
        if name and slot >= 0:
            import_slots[slot] = name
        if not name:
            continue
        stub_offset = (index + first_stub_index) * stride
        if stub_offset + stride > size:
            continue
        plt_stubs[virtual_address + stub_offset] = name
    return plt_stubs, import_slots


def _collect_libraries(binary: Any) -> list[str]:
    libraries = []
    for library in _iter_attr(binary, "libraries"):
        text = str(getattr(library, "name", library) or "").strip()
        if text and text not in libraries:
            libraries.append(text)
    return sorted(libraries, key=str.lower)


def _iter_attr(obj: Any, name: str):
    try:
        value = getattr(obj, name, ())
        return list(value)
    except (AttributeError, TypeError, RuntimeError):
        return []
