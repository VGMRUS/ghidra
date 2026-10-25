"""Command-line and interactive interface for MiniGhidraPy."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .analysis import Analyzer
from .image import BinaryImage, LoadError
from .project import ProjectState


HELP_TEXT = """Comandos disponibles:
  info                              resumen del binario y del análisis
  sections                          secciones y permisos
  functions [filtro]                funciones recuperadas y origen
  patterns [filtro]                 inicios recuperados por firmas x86
  thunks                            saltos simples a funciones/imports
  tables [filtro]                   tablas de punteros a instrucciones
  unwind <funcion>                  metadata UNWIND_INFO de PE x64
  noreturn                          funciones/imports no-return detectados
  function <direccion|nombre>       desensamblar una función
  disasm [direccion] [cantidad]     desensamblar desde una dirección
  pseudo <direccion|nombre>         pseudocódigo semántico x86/x64 (experimental)
  emit <direccion|nombre> [opciones]  emisión completa con el motor Emit/PrintC de Ghidra
                                    --markup (XML <clang_document>), --flat, --width N,
                                    --indent N, --no-casts, --hex|--dec|--best,
                                    --comments c|cplusplus, --brace same_line|next_line|skip_line,
                                    --labels goto|join|none, --no-convention, --output archivo
  graph <funcion> [archivo.dot]     grafo CFG en formato Graphviz DOT
  strings [filtro] [cantidad]       cadenas ASCII y UTF-16LE
  imports | exports | symbols [filtro]
  xrefs <direccion|nombre>          referencias que llegan a una dirección
  calls <direccion|nombre>          llamadas que salen de una función
  search <texto>                    buscar bytes UTF-8/ASCII en secciones
  findhex <hex>                     buscar bytes, por ejemplo: findhex 48 8b 05
  rename <direccion> <nombre>       asignar una etiqueta
  comment <direccion> <texto>       guardar un comentario
  patch <direccion> <hex...> <salida>  parchear una copia, nunca el original
  save [proyecto.json]              guardar etiquetas y comentarios
  load [proyecto.json]              cargar anotaciones del mismo binario
  export-json <archivo.json>        exportar un informe de análisis
  export-asm <archivo.asm>          exportar el desensamblado lineal
  help | quit                       ayuda o salir

Las direcciones aceptan notación decimal o hexadecimal (0x...)."""


class Console:
    def __init__(
        self, image: BinaryImage, state: ProjectState, project_path: Path
    ) -> None:
        self.image = image
        self.state = state
        self.project_path = project_path
        self.analyzer = Analyzer(image, state)

    def execute(self, line: str) -> bool:
        """Run one command. Return False when the console should exit."""
        try:
            args = shlex.split(line)
        except ValueError as exc:
            raise ValueError(f"Sintaxis de comillas no válida: {exc}") from exc
        if not args:
            return True
        command = args[0].lower()
        rest = args[1:]

        if command in {"quit", "exit", "q"}:
            return False
        if command in {"help", "h", "?"}:
            print(HELP_TEXT)
        elif command == "info":
            self._info()
        elif command == "sections":
            self._sections()
        elif command in {"functions", "funcs"}:
            self._functions(rest)
        elif command in {"patterns", "function-patterns"}:
            self._patterns(rest)
        elif command == "thunks":
            self._thunks(rest)
        elif command in {"tables", "address-tables"}:
            self._address_tables(rest)
        elif command in {"unwind", "unwind-info"}:
            self._unwind(rest)
        elif command in {"noreturn", "no-return"}:
            self._noreturn(rest)
        elif command in {"function", "func"}:
            self._function(rest)
        elif command in {"disasm", "dis"}:
            self._disasm(rest)
        elif command in {"pseudo", "decompile"}:
            self._pseudo(rest)
        elif command in {"emit", "print", "printc"}:
            self._emit(rest)
        elif command in {"graph", "cfg"}:
            self._graph(rest)
        elif command == "strings":
            self._strings(rest)
        elif command == "imports":
            self._symbols_table(self.image.imports, "Importaciones", rest)
        elif command == "exports":
            self._symbols_table(self.image.exports, "Exportaciones", rest)
        elif command == "symbols":
            self._symbols_table(self.image.symbols, "Símbolos", rest)
        elif command == "xrefs":
            self._xrefs(rest)
        elif command == "calls":
            self._calls(rest)
        elif command == "search":
            self._search(rest)
        elif command in {"findhex", "searchhex"}:
            self._find_hex(rest)
        elif command == "rename":
            self._rename(rest)
        elif command in {"comment", "comentario"}:
            self._comment(rest)
        elif command == "patch":
            self._patch(rest)
        elif command == "save":
            self._save(rest)
        elif command == "load":
            self._load(rest)
        elif command in {"export-json", "report-json"}:
            self._export_json(rest)
        elif command in {"export-asm", "report-asm"}:
            self._export_asm(rest)
        elif command in {"export-pseudo", "report-pseudo"}:
            self._export_pseudo(rest)
        else:
            raise ValueError(f"Comando desconocido: {command!r}. Escribe help.")
        return True

    def _info(self) -> None:
        info = self.analyzer.summary()
        print(f"Archivo       : {info['file']}")
        print(f"Formato       : {info['format']}")
        print(f"Arquitectura  : {info['architecture']} ({info['bits']} bits)")
        print(f"Base          : {self.image.image_base:#x}")
        print(
            f"Punto entrada : {hex(info['entrypoint']) if info['entrypoint'] is not None else 'no disponible'}"
        )
        print(
            f"Secciones     : {info['sections']} ({info['executable_sections']} ejecutables)"
        )
        print(f"Instrucciones : {info['instructions']}")
        print(f"Funciones     : {info['functions']}")
        print(f"Thunks        : {info['thunks']}")
        print(f"Tablas addr.  : {info['address_tables']}")
        print(f"Saltos resueltos: {info['computed_jumps']}")
        print(f"Por patrones  : {info['pattern_functions']}")
        print(f"Mach-O starts : {info['macho_function_starts']}")
        print(f"PE .pdata     : {info['pe_exception_functions']} rangos x64")
        print(f"PE unwind    : {info['pe_unwind_functions']} funciones con metadata")
        print(f"PE TLS        : {info['pe_tls_callbacks']} callbacks")
        print(
            f"No-return     : {info['no_return_functions']} funciones, {info['no_return_imports']} imports conocidos"
        )
        print(
            f"Purga pila    : {info['stack_purge_functions']} funciones con valor inferido"
        )
        print(f"Cadenas       : {info['strings']}")
        print(f"Importaciones : {info['imports']}")
        print(f"Exportaciones : {info['exports']}")
        print(f"Referencias   : {info['xrefs']}")
        print(f"SHA-256       : {info['sha256']}")
        if self.image.libraries:
            print(f"Bibliotecas   : {', '.join(self.image.libraries)}")

    def _sections(self) -> None:
        print(
            f"{'Nombre':<22} {'Inicio':>18} {'Tamaño':>10} {'Permisos':<5} {'Archivo':>12}"
        )
        for section in self.image.sections:
            permissions = "".join(
                letter if enabled else "-"
                for letter, enabled in (
                    ("r", section.readable),
                    ("w", section.writable),
                    ("x", section.executable),
                )
            )
            file_size = len(section.data)
            print(
                f"{section.name[:21]:<22} {section.start:#018x} "
                f"{section.virtual_size:>10} {permissions:<5} {file_size:>12}"
            )

    def _functions(self, args: list[str]) -> None:
        query = " ".join(args).casefold().removeprefix("0x")
        functions = [
            function
            for function in sorted(
                self.analyzer.functions.values(), key=lambda item: item.address
            )
            if query in function.name.casefold() or query in f"{function.address:x}"
        ]
        if not functions:
            print("No se encontraron funciones recuperadas.")
            return
        print(f"Funciones: {len(functions)}")
        print(
            f"{'Dirección':>18} {'Bytes':>8} {'Instrucciones':>14} {'Origen':<18} Nombre"
        )
        for function in functions:
            origin = (
                "patrón Ghidra"
                if function.source.startswith("Ghidra pattern:")
                else function.source
            )
            markers = []
            if function.is_thunk:
                target = function.thunk_target_name or (
                    f"{function.thunk_target:#x}"
                    if function.thunk_target is not None
                    else "externa"
                )
                markers.append(f"THUNK -> {target}")
            if function.no_return:
                markers.append("NORETURN")
            if function.stack_purge_size is not None:
                markers.append(f"STACK_PURGE={function.stack_purge_size}")
            if function.pe_unwind_info is not None:
                allocation = function.pe_unwind_info.stack_allocation_bytes
                markers.append(
                    f"UNWIND_STACK={allocation}" if allocation is not None else "UNWIND"
                )
            suffix = f" [{' | '.join(markers)}]" if markers else ""
            print(
                f"{function.address:#018x} {function.size:>8} "
                f"{len(function.instructions):>14} {origin:<18} {function.name}{suffix}"
            )

    def _patterns(self, args: list[str]) -> None:
        query = " ".join(args).casefold().removeprefix("0x")
        candidates = [
            item
            for item in self.analyzer.function_start_candidates
            if query in f"{item.address:x}" or query in item.pattern.casefold()
        ]
        print(f"Inicios recuperados por patrones Ghidra/x86: {len(candidates)}")
        for item in candidates[:500]:
            label = f" -> {item.label}" if item.label else ""
            print(
                f"{item.address:#018x}  x86-{item.bits}  confianza {item.confidence:<5} "
                f"{item.pattern}{label}"
            )
        if len(candidates) > 500:
            print(f"... {len(candidates) - 500} candidatos omitidos")

    def _thunks(self, args: list[str]) -> None:
        query = " ".join(args).casefold()
        thunks = [
            function
            for function in sorted(
                self.analyzer.functions.values(), key=lambda item: item.address
            )
            if function.is_thunk
            and (
                not query
                or query in function.name.casefold()
                or query in (function.thunk_target_name or "").casefold()
            )
        ]
        print(f"Thunks recuperados: {len(thunks)}")
        for function in thunks[:500]:
            target = function.thunk_target_name or (
                f"{function.thunk_target:#x}"
                if function.thunk_target is not None
                else "indirecto"
            )
            noreturn = " [NORETURN]" if function.no_return else ""
            print(f"{function.address:#018x}  {function.name} -> {target}{noreturn}")
        if len(thunks) > 500:
            print(f"... {len(thunks) - 500} thunks omitidos")

    def _address_tables(self, args: list[str]) -> None:
        if len(args) > 1:
            raise ValueError("Uso: tables [filtro]")
        query = args[0].casefold() if args else ""
        if query.startswith("0x"):
            query = query[2:]
        tables = [
            table
            for table in self.analyzer.address_tables
            if not query
            or query in f"{table.address:x}"
            or query in table.section.casefold()
        ]
        print(f"Tablas de punteros a código: {len(tables)}")
        for table in tables[:500]:
            targets = ", ".join(f"{target:#x}" for target in table.targets[:12])
            if len(table.targets) > 12:
                targets += f", ... (+{len(table.targets) - 12})"
            print(
                f"{table.address:#018x}  {table.section}  "
                f"{len(table.targets)} x {table.entry_size} B -> {targets}"
            )
            if table.jump_sources:
                sources = ", ".join(f"{source:#x}" for source in table.jump_sources[:8])
                if len(table.jump_sources) > 8:
                    sources += f", ... (+{len(table.jump_sources) - 8})"
                print(f"    saltos indirectos: {sources}")
        if len(tables) > 500:
            print(f"... {len(tables) - 500} tablas omitidas")

    def _noreturn(self, args: list[str]) -> None:
        if args:
            raise ValueError("Uso: noreturn")
        functions = [
            function
            for function in self.analyzer.functions.values()
            if function.no_return
        ]
        print(f"Funciones no-return detectadas: {len(functions)}")
        for function in sorted(functions, key=lambda item: item.address):
            evidence = self.analyzer.no_return_evidence.get(function.address, ())
            source = "inferencia" if evidence else "nombre conocido"
            print(f"{function.address:#018x}  {function.name}  [{source}]")
            for reason in evidence[:10]:
                print(f"    - {reason}")
            if len(evidence) > 10:
                print(f"    ... {len(evidence) - 10} indicios omitidos")
        print(f"Imports no-return conocidos: {len(self.analyzer.no_return_imports)}")
        for symbol, matched_name in self.analyzer.no_return_imports:
            location = (
                f"{symbol.address:#018x}" if symbol.address else "(sin dirección)"
            )
            library = f" [{symbol.library}]" if symbol.library else ""
            print(f"{location}  {symbol.name}{library}  [regla: {matched_name}]")

    def _unwind(self, args: list[str]) -> None:
        if len(args) != 1:
            raise ValueError("Uso: unwind <direccion|nombre>")
        function = self.analyzer.function_at(args[0])
        info = function.pe_unwind_info
        if info is None:
            print(f"{function.name} no tiene UNWIND_INFO PE x64 recuperada.")
            return

        flag_names = [
            name
            for bit, name in ((0x1, "EHANDLER"), (0x2, "UHANDLER"), (0x4, "CHAININFO"))
            if info.flags & bit
        ]
        flags = "|".join(flag_names) if flag_names else "ninguno"
        allocation = (
            f"{info.stack_allocation_bytes} bytes"
            if info.stack_allocation_bytes is not None
            else "desconocida"
        )
        print(
            f"{function.name} @ {function.address:#x}: UNWIND_INFO RVA {info.unwind_rva:#x}, "
            f"versión {info.version}, flags {flags}"
        )
        print(
            f"Prólogo: {info.prolog_size} bytes; frame: {info.frame_register or 'ninguno'} "
            f"+ {info.frame_offset} bytes; stack estimado: {allocation}"
        )
        print(f"Registros guardados: {', '.join(info.saved_registers) or 'ninguno'}")
        for operation in info.operations:
            details = []
            if operation.register:
                details.append(operation.register)
            if operation.operand is not None:
                details.append(f"{operation.operand:#x}")
            suffix = f" ({', '.join(details)})" if details else ""
            print(f"  +{operation.code_offset:02x}: {operation.name}{suffix}")
        if info.handler_rva is not None:
            handler = self.image.image_base + info.handler_rva
            print(f"Handler RVA: {info.handler_rva:#x} (VA {handler:#x})")

    def _function(self, args: list[str]) -> None:
        if len(args) != 1:
            raise ValueError("Uso: function <direccion|nombre>")
        function = self.analyzer.function_at(args[0])
        print(self.analyzer.format_function(function))

    def _disasm(self, args: list[str]) -> None:
        if len(args) > 2:
            raise ValueError("Uso: disasm [direccion] [cantidad]")
        if args:
            start = self.analyzer.resolve_address(args[0])
        else:
            start = self.image.entrypoint
            if start is None and self.analyzer.instruction_addresses:
                start = self.analyzer.instruction_addresses[0]
        if start is None:
            raise ValueError("No hay punto de entrada; indica una dirección explícita.")
        count = int(args[1], 0) if len(args) == 2 else 64
        if count < 1 or count > 100_000:
            raise ValueError("La cantidad debe estar entre 1 y 100000.")
        instructions = self.analyzer.iter_instructions(start, count)
        if not instructions:
            print(f"No hay instrucciones decodificadas desde {start:#x}.")
            return
        for instruction in instructions:
            print(self.analyzer.format_instruction(instruction))

    def _pseudo(self, args: list[str]) -> None:
        if len(args) != 1:
            raise ValueError("Uso: pseudo <direccion|nombre>")
        function = self.analyzer.function_at(args[0])
        print(self.analyzer.pseudocode(function))

    def _emit(self, args: list[str]) -> None:
        """Emit a whole function through the ported Ghidra Emit/PrintC engine."""
        if not args:
            raise ValueError(
                "Uso: emit <direccion|nombre> [--markup] [--flat] [--width N] "
                "[--indent N] [--no-casts] [--hex|--dec|--best] "
                "[--comments c|cplusplus] [--brace same_line|next_line|skip_line] "
                "[--labels goto|join|none] [--no-convention] [--output archivo]"
            )
        options = {
            "markup": False,
            "flat": False,
            "max_line_size": 100,
            "indent_increment": 2,
            "casts": True,
            "integer_format": "best",
            "comment_style": "c",
            "brace": None,
            "labels": "goto",
            "convention": None,
        }
        output: Optional[Path] = None
        positional: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            index += 1
            if token == "--markup":
                options["markup"] = True
            elif token == "--flat":
                options["flat"] = True
            elif token == "--no-casts":
                options["casts"] = False
            elif token == "--no-convention":
                options["convention"] = False
            elif token in ("--hex", "--dec", "--best"):
                options["integer_format"] = token[2:]
            elif token == "--width":
                options["max_line_size"] = _positive_int(args, index, token)
                index += 1
            elif token == "--indent":
                options["indent_increment"] = _positive_int(args, index, token)
                index += 1
            elif token == "--comments":
                if index >= len(args):
                    raise ValueError("--comments necesita c o cplusplus")
                options["comment_style"] = args[index]
                index += 1
            elif token == "--brace":
                if index >= len(args):
                    raise ValueError(
                        "--brace necesita same_line, next_line o skip_line"
                    )
                options["brace"] = args[index]
                index += 1
            elif token == "--labels":
                if index >= len(args):
                    raise ValueError("--labels necesita goto, join o none")
                options["labels"] = args[index]
                index += 1
            elif token == "--output":
                if index >= len(args):
                    raise ValueError("--output necesita una ruta")
                output = Path(args[index]).expanduser().resolve()
                index += 1
            elif token.startswith("--"):
                raise ValueError(f"Opción desconocida: {token}")
            else:
                positional.append(token)
        if len(positional) != 1:
            raise ValueError("Uso: emit <direccion|nombre> [opciones]")
        function = self.analyzer.function_at(positional[0])
        text = self.analyzer.print_function(
            function,
            markup=options["markup"],
            max_line_size=options["max_line_size"],
            indent_increment=options["indent_increment"],
            flat=options["flat"],
            comment_style=options["comment_style"],
            integer_format=options["integer_format"],
            casts=options["casts"],
            convention=options["convention"],
            brace=options["brace"],
            labels=options["labels"],
        )
        if output is not None:
            output.write_text(text, encoding="utf-8")
            print(f"Emisión guardada en {output}")
        else:
            print(text)

    def _graph(self, args: list[str]) -> None:
        if not 1 <= len(args) <= 2:
            raise ValueError("Uso: graph <direccion|nombre> [archivo.dot]")
        function = self.analyzer.function_at(args[0])
        dot = self.analyzer.cfg_dot(function)
        if len(args) == 2:
            destination = Path(args[1]).expanduser().resolve()
            destination.write_text(dot + "\n", encoding="utf-8")
            print(f"Grafo CFG guardado en {destination}")
        else:
            print(dot)

    def _strings(self, args: list[str]) -> None:
        query_parts = args[:]
        limit = 200
        if query_parts:
            try:
                possible_limit = int(query_parts[-1], 0)
                if possible_limit >= 0:
                    limit = possible_limit
                    query_parts = query_parts[:-1]
            except ValueError:
                pass
        query = " ".join(query_parts)
        matches = self.analyzer.strings_matching(query)
        print(f"Cadenas: {len(matches)} (mostrando hasta {limit})")
        for item in matches[:limit]:
            safe_text = item.text.encode("unicode_escape").decode("ascii")
            print(
                f"{item.address:#018x}  {item.encoding:<8} {item.section:<16} {safe_text}"
            )

    def _symbols_table(self, symbols, title: str, args: list[str]) -> None:
        query = " ".join(args).casefold()
        filtered = [symbol for symbol in symbols if query in symbol.name.casefold()]
        print(f"{title}: {len(filtered)}")
        for symbol in filtered:
            location = (
                f"{symbol.address:#018x}" if symbol.address else "(sin dirección)"
            )
            extra = f"  [{symbol.library}]" if symbol.library else ""
            print(f"{location}  {symbol.name}{extra}")

    def _xrefs(self, args: list[str]) -> None:
        if len(args) != 1:
            raise ValueError("Uso: xrefs <direccion|nombre>")
        target = self.analyzer.resolve_address(args[0])
        references = self.analyzer.xrefs_to(target)
        target_name = self.analyzer.name_at(target)
        print(
            f"Referencias a {target:#x}"
            + (f" ({target_name})" if target_name else "")
            + f": {len(references)}"
        )
        for xref in references:
            source_name = self.analyzer.name_at(xref.source)
            label = f" ({source_name})" if source_name else ""
            print(f"{xref.source:#018x}{label:<24} {xref.kind}")

    def _calls(self, args: list[str]) -> None:
        if len(args) != 1:
            raise ValueError("Uso: calls <direccion|nombre>")
        function = self.analyzer.function_at(args[0])
        calls = [
            xref
            for instruction in function.instructions
            for xref in self.analyzer.xrefs_from(instruction.address)
            if xref.kind == "call"
        ]
        print(f"Llamadas desde {function.name}: {len(calls)}")
        for xref in calls:
            target_name = self.analyzer.name_at(xref.target)
            suffix = f" ({target_name})" if target_name else ""
            print(f"{xref.source:#018x} -> {xref.target:#018x}{suffix}")

    def _search(self, args: list[str]) -> None:
        query = " ".join(args)
        if not query:
            raise ValueError("Uso: search <texto>")
        hits = self.analyzer.search_bytes(query.encode("utf-8"))
        print(f"Coincidencias de {query!r}: {len(hits)}")
        for address in hits[:500]:
            print(f"{address:#018x}")
        if len(hits) > 500:
            print(f"... {len(hits) - 500} coincidencias omitidas")

    def _find_hex(self, args: list[str]) -> None:
        if not args:
            raise ValueError("Uso: findhex <bytes hexadecimales>")
        pattern = _parse_hex_bytes(args)
        hits = self.analyzer.search_bytes(pattern)
        print(f"Coincidencias ({pattern.hex(' ')}): {len(hits)}")
        for address in hits[:500]:
            print(f"{address:#018x}")
        if len(hits) > 500:
            print(f"... {len(hits) - 500} coincidencias omitidas")

    def _rename(self, args: list[str]) -> None:
        if len(args) != 2:
            raise ValueError("Uso: rename <direccion> <nombre_sin_espacios>")
        address = self.analyzer.resolve_address(args[0])
        name = args[1]
        if not name or any(character in name for character in "\r\n\x00"):
            raise ValueError(
                "La etiqueta no puede estar vacía ni contener saltos de línea."
            )
        self.state.labels[address] = name
        self.analyzer._name_by_address[address] = name
        if address in self.analyzer.functions:
            self.analyzer.functions[address].name = name
        print(f"Etiqueta {address:#x} = {name}")

    def _comment(self, args: list[str]) -> None:
        if len(args) < 2:
            raise ValueError('Uso: comment <direccion> "texto del comentario"')
        address = self.analyzer.resolve_address(args[0])
        text = " ".join(args[1:])
        if any(character in text for character in "\r\n\x00"):
            raise ValueError("El comentario debe ocupar una sola línea.")
        self.state.comments[address] = text
        print(f"Comentario guardado en {address:#x}.")

    def _patch(self, args: list[str]) -> None:
        if len(args) < 3:
            raise ValueError("Uso: patch <direccion> <bytes hex> <archivo_salida>")
        address = self.analyzer.resolve_address(args[0])
        output = args[-1]
        patch = _parse_hex_bytes(args[1:-1])
        result = self.image.patch_to_file(address, patch, output)
        print(
            f"Copia parcheada escrita en {result} ({len(patch)} bytes en {address:#x})."
        )

    def _save(self, args: list[str]) -> None:
        if len(args) > 1:
            raise ValueError("Uso: save [proyecto.json]")
        if args:
            self.project_path = Path(args[0]).expanduser().resolve()
        destination = self.state.save(self.project_path, self.image)
        print(f"Proyecto guardado en {destination}")

    def _load(self, args: list[str]) -> None:
        if len(args) > 1:
            raise ValueError("Uso: load [proyecto.json]")
        source = Path(args[0]).expanduser().resolve() if args else self.project_path
        self.state = ProjectState.load(source, self.image)
        self.project_path = source
        self.analyzer.project_state = self.state
        for address, name in self.state.labels.items():
            self.analyzer._name_by_address[address] = name
            if address in self.analyzer.functions:
                self.analyzer.functions[address].name = name
        print(f"Proyecto cargado desde {source}")

    def _export_json(self, args: list[str]) -> None:
        if len(args) != 1:
            raise ValueError("Uso: export-json <archivo.json>")
        destination = Path(args[0]).expanduser().resolve()
        functions = []
        for function in sorted(
            self.analyzer.functions.values(), key=lambda item: item.address
        ):
            functions.append(
                {
                    "address": function.address,
                    "name": function.name,
                    "size": function.size,
                    "source": function.source,
                    "no_return": function.no_return,
                    "stack_purge_size": function.stack_purge_size,
                    "calling_convention": function.calling_convention,
                    "signature": (
                        asdict(function.signature)
                        if function.signature is not None
                        else None
                    ),
                    "pe_unwind_info": (
                        asdict(function.pe_unwind_info)
                        if function.pe_unwind_info is not None
                        else None
                    ),
                    "no_return_evidence": self.analyzer.no_return_evidence.get(
                        function.address, []
                    ),
                    "is_thunk": function.is_thunk,
                    "thunk_target": function.thunk_target,
                    "thunk_target_name": function.thunk_target_name,
                    "instructions": [
                        {
                            "address": instruction.address,
                            "bytes": instruction.raw.hex(),
                            "mnemonic": instruction.mnemonic,
                            "operands": instruction.op_str,
                            "memory_references": list(instruction.memory_references),
                            "indexed_memory_references": list(
                                instruction.indexed_memory_references
                            ),
                            "scalar_references": list(instruction.scalar_references),
                            "computed_targets": list(instruction.computed_targets),
                        }
                        for instruction in function.instructions
                    ],
                }
            )
        report = {
            "summary": self.analyzer.summary(),
            "sections": [
                {
                    "name": section.name,
                    "start": section.start,
                    "size": section.virtual_size,
                    "executable": section.executable,
                    "file_size": len(section.data),
                }
                for section in self.image.sections
            ],
            "symbols": [symbol.__dict__ for symbol in self.image.symbols],
            "imports": [symbol.__dict__ for symbol in self.image.imports],
            "plt_stubs": {
                f"0x{address:x}": name for address, name in self.image.plt_stubs.items()
            },
            "import_slots": {
                f"0x{address:x}": name
                for address, name in self.image.import_slots.items()
            },
            "address_tables": [
                {
                    "address": table.address,
                    "section": table.section,
                    "entry_size": table.entry_size,
                    "targets": list(table.targets),
                    "jump_sources": list(table.jump_sources),
                }
                for table in self.analyzer.address_tables
            ],
            "no_return_imports": [
                {**symbol.__dict__, "matched_rule": matched_name}
                for symbol, matched_name in self.analyzer.no_return_imports
            ],
            "exports": [symbol.__dict__ for symbol in self.image.exports],
            "functions": functions,
            "function_start_candidates": [
                {
                    "address": item.address,
                    "pattern": item.pattern,
                    "bits": item.bits,
                    "confidence": item.confidence,
                    "label": item.label,
                }
                for item in self.analyzer.function_start_candidates
            ],
            "macho_function_start_candidates": list(
                self.analyzer.macho_function_start_candidates
            ),
            "pe_exception_function_ranges": [
                {"start": start, "end": end}
                for start, end in getattr(
                    self.image, "pe_exception_function_ranges", ()
                )
            ],
            "pe_tls_callbacks": list(getattr(self.image, "pe_tls_callbacks", ())),
            "strings": [entry.__dict__ for entry in self.analyzer.strings],
            "xrefs": [xref.__dict__ for xref in self.analyzer.xrefs],
            "annotations": {
                "labels": {
                    f"0x{address:x}": name
                    for address, name in self.state.labels.items()
                },
                "comments": {
                    f"0x{address:x}": text
                    for address, text in self.state.comments.items()
                },
            },
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Informe JSON escrito en {destination}")

    def _export_asm(self, args: list[str]) -> None:
        if len(args) != 1:
            raise ValueError("Uso: export-asm <archivo.asm>")
        destination = Path(args[0]).expanduser().resolve()
        lines = [
            f"; MiniGhidraPy | {self.image.path}",
            f"; {self.image.format} {self.image.architecture} ({self.image.bits} bits)",
            f"; SHA-256: {self.image.sha256}",
            "",
        ]
        for instruction in self.analyzer.iter_instructions():
            lines.append(self.analyzer.format_instruction(instruction))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Desensamblado escrito en {destination}")

    def _export_pseudo(self, args: list[str]) -> None:
        if not args:
            raise ValueError("Uso: export-pseudo --all <archivo.c> | export-pseudo <0xaddr> <archivo.c>")
        
        all_funcs = False
        target_file = None
        func_addr_str = None

        if args[0] == "--all":
            all_funcs = True
            if len(args) != 2:
                raise ValueError("Uso: export-pseudo --all <archivo.c>")
            target_file = args[1]
        else:
            if len(args) != 2:
                raise ValueError("Uso: export-pseudo <0xaddr> <archivo.c>")
            func_addr_str = args[0]
            target_file = args[1]

        destination = Path(target_file).expanduser().resolve()
        lines = [
            f"/* MiniGhidraPy | {self.image.path} */",
            f"/* {self.image.format} {self.image.architecture} ({self.image.bits} bits) */",
            f"/* SHA-256: {self.image.sha256} */",
            "",
        ]

        if all_funcs:
            print("Decompilando todas las funciones (puede tardar un rato)...")
            functions = sorted(self.analyzer.functions.values(), key=lambda f: f.address)
        else:
            address = self.analyzer.resolve_address(func_addr_str)
            function = self.analyzer.function_at(address)
            if function is None:
                raise ValueError(f"No hay ninguna función detectada en la dirección {address:#x}.")
            functions = [function]

        from .decompiler import SemanticDecompiler
        for function in functions:
            decompiler = SemanticDecompiler(self.analyzer)
            try:
                code = decompiler.decompile(function)
            except Exception as e:
                code = f"/* Error al decompilar {function.name}: {e} */"
            lines.append(code)
            lines.append("")

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Pseudocódigo escrito en {destination}")


def _parse_hex_bytes(tokens: list[str]) -> bytes:
    cleaned = []
    for token in tokens:
        cleaned.append(token.strip().replace("0x", "").replace("0X", ""))
    try:
        result = bytes.fromhex(" ".join(cleaned))
    except ValueError as exc:
        raise ValueError(f"Bytes hexadecimales no válidos: {exc}") from exc
    if not result:
        raise ValueError("No se indicaron bytes.")
    return result


def _parse_int(value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"dirección no válida: {value}") from exc


def _positive_int(args: list[str], index: int, option: str) -> int:
    """Read the integer argument of a console option such as ``--width``."""
    if index >= len(args):
        raise ValueError(f"{option} necesita un número entero positivo")
    try:
        value = int(args[index], 0)
    except ValueError as exc:
        raise ValueError(f"{option} necesita un número entero positivo") from exc
    if value < 1:
        raise ValueError(f"{option} necesita un número entero positivo")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minighidra",
        description="Analizador estático pequeño para binarios x86 y x86-64.",
    )
    parser.add_argument("binary", help="archivo ELF/PE o volcado crudo")
    parser.add_argument(
        "--raw-arch",
        choices=("x86", "x86_64"),
        help="trata el archivo como bytes crudos con esta arquitectura",
    )
    parser.add_argument(
        "--base",
        type=_parse_int,
        default=0,
        help="base virtual para un volcado crudo (default: 0)",
    )
    parser.add_argument(
        "--project", help="archivo JSON de anotaciones (default: junto al binario)"
    )
    parser.add_argument(
        "-c",
        "--command",
        help="ejecuta un comando y termina; sin esta opción abre la consola",
    )
    parser.add_argument(
        "--gui", action="store_true", help="abre la interfaz gráfica Tkinter"
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gui and args.command is not None:
        parser.error("--gui y --command no se pueden usar al mismo tiempo")
    try:
        image = BinaryImage.load(args.binary, raw_arch=args.raw_arch, base=args.base)
        if args.project:
            project_path = Path(args.project).expanduser().resolve()
        else:
            project_path = ProjectState.default_path(image.path)
        if project_path.exists():
            state = ProjectState.load(project_path, image)
        else:
            state = ProjectState()
        console = Console(image, state, project_path)
    except (LoadError, ValueError, RuntimeError) as exc:
        print(f"minighidra: error: {exc}", file=sys.stderr)
        return 2

    if args.gui:
        try:
            from .gui import launch_gui

            launch_gui(console)
        except RuntimeError as exc:
            print(f"minighidra: error: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command is not None:
        try:
            console.execute(args.command)
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    print(
        f"MiniGhidraPy — {image.path.name} | {image.architecture} ({image.bits} bits) | "
        f"{len(console.analyzer.functions)} funciones recuperadas"
    )
    print(
        "Escribe help para ver comandos. La herramienta solo hace análisis estático; no ejecuta el binario."
    )
    while True:
        try:
            line = input("minighidra> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nUsa quit para salir.")
            continue
        try:
            if not console.execute(line):
                break
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"error: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
