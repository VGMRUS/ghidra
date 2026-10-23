"""x86/x86-64 disassembly, lightweight function recovery and xrefs."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace

try:
    from capstone import (
        CS_ARCH_X86,
        CS_GRP_CALL,
        CS_GRP_JUMP,
        CS_GRP_RET,
        CS_MODE_32,
        CS_MODE_64,
        Cs,
    )
    from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_REG_RIP
except ImportError as exc:  # pragma: no cover - dependency declared by the package
    raise RuntimeError(
        "Falta Capstone. Instálalo con: python -m pip install capstone"
    ) from exc

from .abi import (
    CallingConvention,
    abi_argument_registers,
    calling_convention_by_name,
    calling_convention_for,
)
from .function_starts import FunctionStartCandidate, find_function_start_candidates
from .model import (
    AddressTable,
    BasicBlock,
    Function,
    FunctionSignature,
    Instruction,
    StringEntry,
    Xref,
)
from .noreturn import NoReturnCatalog
from .ssa import SSAFunction, SSAHeritage
from .typesys import TypeDatabase, TypeKind

_SCALAR_SENTINELS = frozenset(
    {
        0xFFFF,
        0xFF00,
        0xFFFFFF,
        0xFF0000,
        0xFF00FF,
        0xFFFFFFFF,
        0xFFFFFF00,
        0xFFFF0000,
        0xFF000000,
    }
)


class Analyzer:
    """Build an in-memory static-analysis index for one :class:`BinaryImage`."""

    @classmethod
    def from_path(cls, path: str, project_state=None, **kwargs):
        from .image import BinaryImage
        img = BinaryImage.load(path)
        return cls(img, project_state=project_state, **kwargs)

    def __init__(
        self, image, project_state=None, *, noreturn_evidence_threshold: int = 3
    ) -> None:
        if (
            not isinstance(noreturn_evidence_threshold, int)
            or isinstance(noreturn_evidence_threshold, bool)
            or noreturn_evidence_threshold < 1
        ):
            raise ValueError(
                "El umbral de evidencia no-return debe ser un entero positivo."
            )
        self.image = image
        self.project_state = project_state
        self.type_database = TypeDatabase(image.bits, image.format)
        self._calling_convention_overrides = {}
        self.noreturn_evidence_threshold = noreturn_evidence_threshold
        minimum_address = min((section.start for section in image.sections), default=0)
        self._scalar_reference_analysis_enabled = (
            image.format != "ELF" and minimum_address != 0
        )
        self._md = Cs(CS_ARCH_X86, CS_MODE_32 if image.bits == 32 else CS_MODE_64)
        self._md.detail = True
        self._md.skipdata = True

        self.instructions: dict[int, Instruction] = {}
        self.instruction_addresses: list[int] = []
        self.xrefs: list[Xref] = []
        self._xref_keys: set[tuple[int, int, str]] = set()
        self.xrefs_by_target: dict[int, list[Xref]] = defaultdict(list)
        self.xrefs_by_source: dict[int, list[Xref]] = defaultdict(list)
        self.function_seeds: dict[int, tuple[str, str]] = {}
        self.functions: dict[int, Function] = {}
        self.function_start_candidates: list[FunctionStartCandidate] = []
        self.macho_function_start_candidates: list[int] = []
        self.address_tables: list[AddressTable] = []
        self.strings: list[StringEntry] = []
        self.noreturn_catalog = NoReturnCatalog.for_format(image.format)
        self.no_return_addresses: set[int] = set()
        self.no_return_matches: dict[int, str] = {}
        self.no_return_evidence: dict[int, list[str]] = {}
        self.no_return_imports: list[tuple[object, str]] = []
        self._name_by_address: dict[int, str] = {}
        self._index_names()
        self._disassemble_sections()
        self._find_strings()
        self._find_function_seeds()
        self._mark_known_noreturns()
        self._recover_functions()
        self._discover_macho_function_starts()
        self._discover_pattern_function_starts()
        self._record_scalar_reference_xrefs()
        self._discover_address_tables()
        self._resolve_computed_jumps()
        self._infer_no_return_functions()
        self._infer_stack_purge_sizes()
        self.instruction_addresses = sorted(self.instructions)

    def _index_names(self) -> None:
        # Prefer user names, then exports/functions, then other symbols.
        for symbol in self.image.symbols:
            if symbol.address > 0 and symbol.name:
                self._name_by_address.setdefault(symbol.address, symbol.name)
        for symbol in self.image.exports:
            if symbol.address > 0 and symbol.name:
                self._name_by_address[symbol.address] = symbol.name
        if self.project_state is not None:
            for address, name in self.project_state.labels.items():
                self._name_by_address[int(address)] = name
        self._imports_by_address: dict[int, list] = defaultdict(list)
        for symbol in self.image.imports:
            if symbol.address > 0:
                self._imports_by_address[symbol.address].append(symbol)
                self._name_by_address.setdefault(symbol.address, symbol.name)
        self._import_name_by_address: dict[int, str] = {}
        for address, name in self.image.import_slots.items():
            self._import_name_by_address[address] = name
            self._name_by_address.setdefault(address, name)

    def _disassemble_sections(self) -> None:
        for section in self.image.executable_sections:
            try:
                decoded = self._md.disasm(section.data, section.start)
                for raw_instruction in decoded:
                    # Capstone's skip-data pseudo-instructions have id == 0.
                    if not raw_instruction.id:
                        continue
                    instruction = self._make_instruction(raw_instruction)
                    if instruction.address not in self.instructions:
                        self.instructions[instruction.address] = instruction
                        self._record_instruction_xrefs(instruction)
            except Exception as exc:
                raise RuntimeError(
                    f"Error desensamblando la sección {section.name} en {section.start:#x}: {exc}"
                ) from exc

    def _make_instruction(self, raw_instruction) -> Instruction:
        groups = []
        if raw_instruction.group(CS_GRP_CALL):
            groups.append("call")
        if raw_instruction.group(CS_GRP_JUMP):
            groups.append("jump")
        if raw_instruction.group(CS_GRP_RET):
            groups.append("ret")

        branch_target = None
        memory_references = []
        indexed_memory_references = []
        scalar_references = []
        mask = (1 << self.image.bits) - 1
        try:
            for operand in raw_instruction.operands:
                if operand.type == X86_OP_IMM:
                    value = int(operand.imm) & mask
                    if "call" in groups or "jump" in groups:
                        if branch_target is None:
                            branch_target = value
                    elif self._is_scalar_reference_candidate(value):
                        if value not in scalar_references:
                            scalar_references.append(value)
                elif operand.type == X86_OP_MEM:
                    memory = operand.mem
                    references = memory_references
                    if memory.base == X86_REG_RIP:
                        target = (
                            raw_instruction.address
                            + raw_instruction.size
                            + int(memory.disp)
                        )
                    elif memory.base == 0:
                        target = int(memory.disp)
                        if memory.index:
                            references = indexed_memory_references
                    else:
                        continue
                    target &= mask
                    if target and target not in references:
                        references.append(target)
        except (AttributeError, TypeError, ValueError):
            # Some Capstone pseudo-instructions do not expose full x86 operands.
            pass

        return Instruction(
            address=int(raw_instruction.address),
            size=int(raw_instruction.size),
            raw=bytes(raw_instruction.bytes),
            mnemonic=str(raw_instruction.mnemonic),
            op_str=str(raw_instruction.op_str),
            groups=tuple(groups),
            branch_target=branch_target,
            memory_references=tuple(memory_references),
            indexed_memory_references=tuple(indexed_memory_references),
            scalar_references=tuple(scalar_references),
        )

    def _is_scalar_reference_candidate(self, value: int) -> bool:
        if not self._scalar_reference_analysis_enabled:
            return False
        if value < 4096 or value in _SCALAR_SENTINELS:
            return False
        return self.image.section_at(value) is not None

    def _record_xref(self, source: int, target: int, kind: str) -> None:
        key = (source, target, kind)
        if key in self._xref_keys:
            return
        self._xref_keys.add(key)
        xref = Xref(source=source, target=target, kind=kind)
        self.xrefs.append(xref)
        self.xrefs_by_target[target].append(xref)
        self.xrefs_by_source[source].append(xref)

    def _record_instruction_xrefs(self, instruction: Instruction) -> None:
        if instruction.branch_target is not None:
            if instruction.is_call:
                kind = "call"
            elif instruction.is_conditional_jump:
                kind = "conditional jump"
            else:
                kind = "jump"
            self._record_xref(instruction.address, instruction.branch_target, kind)
        for target in (
            *instruction.memory_references,
            *instruction.indexed_memory_references,
        ):
            self._record_xref(instruction.address, target, "memory")

    def _find_strings(self, min_length: int = 4) -> None:
        ascii_pattern = re.compile(rb"[\x20-\x7e]{" + str(min_length).encode() + rb",}")
        wide_pattern = re.compile(
            rb"(?=((?:[\x20-\x7e]\x00){" + str(min_length).encode() + rb",}))"
        )
        seen: set[tuple[int, str, str]] = set()
        for section in self.image.sections:
            data = section.data
            ascii_matches = list(ascii_pattern.finditer(data))
            ascii_spans = [(match.start(), match.end()) for match in ascii_matches]
            for match in ascii_matches:
                text = match.group().decode("ascii", errors="replace")
                key = (section.start + match.start(), text, "ascii")
                if key not in seen:
                    seen.add(key)
                    self.strings.append(
                        StringEntry(key[0], text, "ascii", section.name)
                    )

            wide_candidates = []
            for match in wide_pattern.finditer(data):
                start = match.start()
                raw_wide = match.group(1)
                end = start + len(raw_wide)
                # A final ASCII character + NUL can otherwise be mistaken for
                # the first UTF-16 code unit. Lookahead preserves the real wide
                # string immediately after that terminator.
                if any(
                    start < ascii_end and ascii_start < end
                    for ascii_start, ascii_end in ascii_spans
                ):
                    continue
                try:
                    text = raw_wide.decode("utf-16le")
                except UnicodeDecodeError:
                    continue
                wide_candidates.append((start, end, text))
            for start, end, text in wide_candidates:
                # Keep the longest run, rather than also reporting every suffix
                # of the same UTF-16 string as a separate string.
                if any(
                    other_start < start and other_end >= end
                    for other_start, other_end, _ in wide_candidates
                ):
                    continue
                key = (section.start + start, text, "utf-16le")
                if key not in seen:
                    seen.add(key)
                    self.strings.append(
                        StringEntry(key[0], text, "utf-16le", section.name)
                    )
        self.strings.sort(key=lambda item: item.address)

    def _add_seed(
        self, address: int | None, name: str = "", source: str = "heuristic"
    ) -> None:
        if address is None or address < 0:
            return
        section = self.image.section_at(address, file_backed=True)
        if section is None or not section.executable:
            return
        if name:
            self._name_by_address.setdefault(address, name)
        existing = self.function_seeds.get(address)
        if existing is None or (not existing[0] and name):
            self.function_seeds[address] = (name, source)

    def _find_function_seeds(self) -> None:
        if self.image.entrypoint is not None:
            self._add_seed(self.image.entrypoint, "entry", "entrypoint")

        for symbol in self.image.symbols:
            kind = symbol.kind.lower()
            if "thunk" in kind:
                self._add_seed(symbol.address, symbol.name, "PLT thunk")
            elif "func" in kind or "function" in kind:
                self._add_seed(symbol.address, symbol.name, "symbol")
        for symbol in self.image.exports:
            self._add_seed(symbol.address, symbol.name, "export")

        # Ghidra's PE ExceptionDataDirectory marks non-chained x64 runtime
        # function entries at image-base + BeginAddress (an RVA).
        for start, _end in getattr(self.image, "pe_exception_function_ranges", ()):
            self._add_seed(
                start, self._name_by_address.get(start, ""), "PE exception directory"
            )
        for index, address in enumerate(getattr(self.image, "pe_tls_callbacks", ())):
            name = self._name_by_address.get(address, f"tls_callback_{index}")
            self._add_seed(address, name, "PE TLS callback")

        # Direct calls are a relatively reliable source of additional function
        # boundaries. Data references into executable ranges are weaker clues,
        # so they are kept only when the target is an instruction boundary.
        for instruction in self.instructions.values():
            target = instruction.branch_target
            if instruction.is_call and target is not None:
                # Ghidra's FunctionAnalyzer ignores call-next PC thunks: they
                # capture the return address rather than call a new function.
                if target == instruction.next_address:
                    continue
                self._add_seed(
                    target, self._name_by_address.get(target, ""), "call target"
                )
        for instruction in self.instructions.values():
            for target in instruction.memory_references:
                if target in self.instructions:
                    self._add_seed(
                        target, self._name_by_address.get(target, ""), "code reference"
                    )

        if not self.function_seeds and self.instructions:
            self._add_seed(min(self.instructions), "entry", "fallback")

    def _record_scalar_reference_xrefs(self) -> None:
        """Record valid scalar address operands without treating data as code."""
        for instruction in self.instructions.values():
            for target in instruction.scalar_references:
                section = self.image.section_at(target)
                if section is None:
                    continue
                if (
                    section.executable
                    and target not in self.function_seeds
                    and target not in self._name_by_address
                ):
                    continue
                self._record_xref(instruction.address, target, "scalar")

    def _discover_address_tables(
        self, minimum_entries: int = 3, max_entries: int = 512
    ) -> None:
        """Find referenced runs of absolute pointers into decoded code.

        This is a small, conservative subset of Ghidra's ``AddressTable`` and
        ``OperandReferenceAnalyzer``: it only scans a data-section address that
        an instruction already references, and it accepts a run when at least
        three consecutive pointer-sized values land on decoded instructions.
        """
        pointer_size = self.image.bits // 8
        candidates: set[int] = set()
        for instruction in self.instructions.values():
            for target in (
                *instruction.memory_references,
                *instruction.indexed_memory_references,
                *instruction.scalar_references,
            ):
                section = self.image.section_at(target, file_backed=True)
                if section is not None and not section.executable:
                    candidates.add(target)

        accepted_ranges: list[tuple[int, int]] = []
        for address in sorted(candidates):
            section = self.image.section_at(address, file_backed=True)
            if section is None or section.executable:
                continue
            if any(start <= address < end for start, end in accepted_ranges):
                continue

            offset = address - section.start
            targets = []
            for index in range(max_entries):
                entry_offset = offset + index * pointer_size
                entry_end = entry_offset + pointer_size
                if entry_end > len(section.data):
                    break
                target = int.from_bytes(section.data[entry_offset:entry_end], "little")
                target_section = self.image.section_at(target, file_backed=True)
                if (
                    target not in self.instructions
                    or target_section is None
                    or not target_section.executable
                ):
                    break
                targets.append(target)

            if len(targets) < minimum_entries:
                continue
            table = AddressTable(address, section.name, pointer_size, tuple(targets))
            self.address_tables.append(table)
            accepted_ranges.append((table.address, table.end))
            for index, target in enumerate(table.targets):
                self._record_xref(table.address + index * pointer_size, target, "table")

    def _resolve_computed_jumps(self) -> bool:
        """Link indexed indirect jumps to any recognized absolute address table."""
        tables_by_address = {table.address: table for table in self.address_tables}
        jump_sources: dict[int, list[int]] = defaultdict(list)
        updated = False

        for address, instruction in list(self.instructions.items()):
            if not instruction.is_jump or instruction.branch_target is not None:
                continue
            linked_tables = [
                tables_by_address[target]
                for target in instruction.indexed_memory_references
                if target in tables_by_address
            ]
            if not linked_tables:
                continue

            computed_targets = tuple(
                dict.fromkeys(
                    target for table in linked_tables for target in table.targets
                )
            )
            self.instructions[address] = replace(
                instruction, computed_targets=computed_targets
            )
            for table in linked_tables:
                jump_sources[table.address].append(address)
            for target in computed_targets:
                self._record_xref(address, target, "switch")
            updated = True

        if not updated:
            return False
        updated_tables = []
        for table in self.address_tables:
            sources = tuple(sorted(set(jump_sources.get(table.address, ()))))
            updated_tables.append(replace(table, jump_sources=sources))
        self.address_tables = updated_tables
        # Rebuild bodies and CFGs now that computed jump destinations are known.
        self._recover_functions()
        return True

    def _mark_known_noreturns(self) -> None:
        """Apply Ghidra's known-name no-return analysis to this image."""
        self.no_return_addresses.clear()
        self.no_return_matches.clear()
        self.no_return_imports.clear()

        for address, name in self._name_by_address.items():
            match = self.noreturn_catalog.match(name)
            if match is not None:
                self.no_return_addresses.add(address)
                self.no_return_matches[address] = name

        for symbol in (*self.image.symbols, *self.image.exports):
            match = self.noreturn_catalog.match(symbol.name)
            if match is None or symbol.address < 0:
                continue
            self.no_return_addresses.add(symbol.address)
            self.no_return_matches.setdefault(symbol.address, symbol.name)

        seen_imports = set()
        for symbol in self.image.imports:
            match = self.noreturn_catalog.match(symbol.name)
            if match is None:
                continue
            key = (symbol.address, symbol.name, symbol.library)
            if key in seen_imports:
                continue
            seen_imports.add(key)
            self.no_return_imports.append((symbol, match))
            # PE IAT addresses are useful for recognizing indirect calls. ELF
            # imports often have no address, in which case the PLT symbol/name
            # or a direct function symbol provides the usable address instead.
            if symbol.address > 0:
                self.no_return_addresses.add(symbol.address)
                self.no_return_matches.setdefault(symbol.address, symbol.name)

        for address, name in self.image.import_slots.items():
            if self.noreturn_catalog.match(name) is not None:
                self.no_return_addresses.add(address)
                self.no_return_matches.setdefault(address, name)

    def _call_is_noreturn(self, instruction: Instruction) -> bool:
        if not instruction.is_call:
            return False
        if instruction.branch_target in self.no_return_addresses:
            return True
        return any(
            target in self.no_return_addresses
            for target in instruction.memory_references
        )

    def _discover_macho_function_starts(self) -> None:
        """Use Mach-O LC_FUNCTION_STARTS as a late, validated seed source.

        Ghidra deliberately runs this analyzer late because Mach-O metadata can
        contain false starts (for example, entries that overlap switch data).
        This subset requires a decoded instruction boundary in executable bytes,
        rejects traps, and avoids instructions already owned by recovered bodies.
        """
        if self.image.format != "Mach-O":
            return

        owned_instructions = {
            instruction.address
            for function in self.functions.values()
            for instruction in function.instructions
        }
        added = False
        for address in getattr(self.image, "macho_function_starts", ()):
            if address in self.function_seeds or address in owned_instructions:
                continue
            instruction = self.instructions.get(address)
            section = self.image.section_at(address, file_backed=True)
            if (
                instruction is None
                or instruction.is_trap
                or section is None
                or not section.executable
            ):
                continue
            self.macho_function_start_candidates.append(address)
            self._add_seed(
                address,
                self._name_by_address.get(address, ""),
                "Mach-O LC_FUNCTION_STARTS",
            )
            added = True
        if added:
            self._recover_functions()

    def _discover_pattern_function_starts(self) -> None:
        """Apply the conservative x86 signature subset after flow recovery.

        Ghidra's FunctionStartAnalyzer avoids marking code already owned by an
        existing function as a new start. We mirror that safeguard by ignoring
        instruction addresses already covered by recovered function bodies.
        """
        candidates = find_function_start_candidates(
            self.image,
            self.instructions,
            known_starts=self.function_seeds,
        )
        owned_instructions = {
            instruction.address
            for function in self.functions.values()
            for instruction in function.instructions
        }
        added = False
        for candidate in candidates:
            if candidate.address in owned_instructions:
                continue
            self.function_start_candidates.append(candidate)
            name = candidate.label or self._name_by_address.get(candidate.address, "")
            self._add_seed(
                candidate.address, name, f"Ghidra pattern: {candidate.pattern}"
            )
            added = True
        if added:
            self._recover_functions()

    def _infer_stack_purge_sizes(self) -> None:
        """Port the simple return-instruction subset of FunctionPurgeAnalysisCmd.

        A return without an immediate has purge size zero; ``ret imm16``
        supplies the number of stack bytes popped by the callee. If reachable
        return sites disagree, the value is left unknown rather than guessed.
        This does not infer stack frames, parameters, or a full calling convention.
        """
        for function in self.functions.values():
            purge_values: set[int] = set()
            uncertain_return = False
            for instruction in function.instructions:
                mnemonic = instruction.mnemonic.lower()
                if not instruction.is_return or not mnemonic.startswith("ret"):
                    continue
                operand = instruction.op_str.strip()
                if not operand:
                    purge_values.add(0)
                    continue
                try:
                    purge = int(operand.split(",", 1)[0].strip(), 0)
                except ValueError:
                    uncertain_return = True
                    break
                if purge < 0 or purge > 0xFFFF:
                    uncertain_return = True
                    break
                purge_values.add(purge)

            if not uncertain_return and len(purge_values) == 1:
                function.stack_purge_size = next(iter(purge_values))

    def _infer_no_return_functions(self) -> None:
        """Apply a bounded subset of Ghidra's evidence-based no-return pass.

        The Ghidra analyzer looks for suspicious call fall-throughs, aggregates
        evidence from call sites (default threshold: three), then propagates the
        result to functions whose reachable terminal paths only call known
        no-return functions. This port uses the strongest recoverable signals:
        a trap, a known function boundary, or the end of decoded code after a
        call. It deliberately omits Ghidra's data/listing repair machinery.
        """
        added_targets = self._infer_no_return_from_call_sites(
            self.noreturn_evidence_threshold
        )
        if added_targets:
            self._recover_functions()

        # Propagate wrapper chains using the already recovered instruction
        # bodies, then rebuild all callers once with the final no-return set.
        wrappers_added = False
        while self._infer_no_return_wrappers():
            wrappers_added = True
        if wrappers_added:
            self._recover_functions()

    def _infer_no_return_from_call_sites(self, threshold: int) -> bool:
        evidence: dict[int, dict[int, str]] = defaultdict(dict)
        for instruction in self.instructions.values():
            if not instruction.is_call or self._call_is_noreturn(instruction):
                continue
            reason = self._call_site_nonreturn_indicator(instruction)
            if reason is None:
                continue

            for target in self._call_target_addresses(instruction):
                if target in self.no_return_addresses:
                    continue
                evidence[target][instruction.address] = reason

        added = False
        for target, call_sites in evidence.items():
            accumulated = self.no_return_evidence.setdefault(target, [])
            for source, reason in sorted(call_sites.items()):
                detail = f"{source:#x}: {reason}"
                if detail not in accumulated:
                    accumulated.append(detail)
            if len(call_sites) < threshold or target in self.no_return_addresses:
                continue
            self.no_return_addresses.add(target)
            self.no_return_matches[target] = self.name_at(target) or f"FUN_{target:X}"
            added = True
        return added

    def _call_site_nonreturn_indicator(self, call: Instruction) -> str | None:
        """Check a bounded fall-through chain for Ghidra-style warning signs."""
        address = call.next_address
        visited: set[int] = set()
        for _ in range(256):
            if address in self.function_seeds:
                return "function entry at call fall-through"
            if address in visited:
                return None
            visited.add(address)
            instruction = self.instructions.get(address)
            if instruction is None:
                return "no decoded instruction at call fall-through"
            if instruction.is_trap:
                return f"{instruction.mnemonic.upper()} on call fall-through"
            if instruction.is_call and self._call_is_noreturn(instruction):
                return None
            if instruction.is_return or (
                instruction.is_jump and not instruction.is_conditional_jump
            ):
                return None
            address = instruction.next_address
        return None

    def _call_target_addresses(self, instruction: Instruction) -> list[int]:
        if instruction.branch_target is not None:
            return [instruction.branch_target]
        targets = []
        for address in instruction.memory_references:
            if (
                address in self.image.import_slots
                or address in self._imports_by_address
                or address in self.function_seeds
                or address in self.no_return_addresses
            ):
                targets.append(address)
        return list(dict.fromkeys(targets))

    def _infer_no_return_wrappers(self) -> bool:
        added = False
        for function in list(self.functions.values()):
            if (
                function.address in self.no_return_addresses
                or not function.instructions
            ):
                continue
            if not self._function_paths_only_call_no_return(function):
                continue
            self.no_return_addresses.add(function.address)
            self.no_return_matches[function.address] = function.name
            self.no_return_evidence[function.address] = [
                "all recovered terminal paths call or jump to a no-return function"
            ]
            added = True
        return added

    def _function_paths_only_call_no_return(self, function: Function) -> bool:
        """Return whether every recovered terminating path is no-return."""
        instructions = {item.address: item for item in function.instructions}
        pending = deque([function.address])
        visited: set[int] = set()
        saw_no_return = False

        while pending:
            address = pending.popleft()
            if address in visited:
                continue
            instruction = instructions.get(address)
            if instruction is None:
                return False
            visited.add(address)

            if instruction.is_call and self._call_is_noreturn(instruction):
                saw_no_return = True
                continue
            if instruction.is_return or instruction.is_trap:
                return False

            if instruction.is_jump:
                target = instruction.branch_target
                if instruction.is_conditional_jump:
                    if target in instructions:
                        pending.append(target)
                    elif self._jump_is_noreturn(instruction):
                        saw_no_return = True
                    else:
                        return False
                    if instruction.next_address not in instructions:
                        return False
                    pending.append(instruction.next_address)
                elif target in instructions:
                    pending.append(target)
                elif self._jump_is_noreturn(instruction):
                    saw_no_return = True
                else:
                    return False
                continue

            if instruction.next_address not in instructions:
                return False
            pending.append(instruction.next_address)

        return saw_no_return

    def _jump_is_noreturn(self, instruction: Instruction) -> bool:
        return instruction.is_jump and (
            instruction.branch_target in self.no_return_addresses
            or any(
                target in self.no_return_addresses
                for target in instruction.memory_references
            )
        )

    def _recover_functions(self) -> None:
        self.functions = {}
        starts = set(self.function_seeds)
        owners: dict[int, int] = {}
        for start in sorted(starts):
            if start not in self.instructions:
                continue
            seed_name, source = self.function_seeds[start]
            name = self._user_or_symbol_name(start, seed_name or f"FUN_{start:X}")
            thunk = self._detect_thunk(start)
            if thunk is None:
                body = self._walk_function(start, starts, owners)
                thunk_target = None
                thunk_target_name = None
            else:
                body, thunk_target, thunk_target_name = thunk
            if not body:
                continue
            function = Function(
                address=start,
                name=name,
                instructions=body,
                source=source,
                no_return=start in self.no_return_addresses,
                is_thunk=thunk is not None,
                thunk_target=thunk_target,
                thunk_target_name=thunk_target_name,
                pe_unwind_info=getattr(self.image, "pe_unwind_infos", {}).get(start),
            )
            function.blocks = self._build_blocks(function.instructions)
            self.functions[start] = function
            for instruction in body:
                owners.setdefault(instruction.address, start)

    def _detect_thunk(self, start: int):
        """Recognize simple x86 thunks, following Ghidra's bounded flow check.

        The full Ghidra implementation checks p-code register side effects for
        up to eight instructions. This subset accepts only harmless NOP/ENDBR
        prefixes followed by one unconditional direct jump or a jump through a
        loader-known import slot.
        """
        body = []
        address = start
        for _ in range(8):
            instruction = self.instructions.get(address)
            if instruction is None:
                return None
            mnemonic = instruction.mnemonic.lower()
            if mnemonic in {"nop", "endbr32", "endbr64"}:
                body.append(instruction)
                address = instruction.next_address
                continue
            if not instruction.is_jump or instruction.is_conditional_jump:
                return None
            body.append(instruction)
            if (
                instruction.branch_target is not None
                and instruction.branch_target != start
            ):
                return (
                    body,
                    instruction.branch_target,
                    self.name_at(instruction.branch_target),
                )
            for target in instruction.memory_references:
                target_name = self.image.import_slots.get(target)
                if target_name is None:
                    imported = self._imports_by_address.get(target, ())
                    target_name = imported[0].name if imported else None
                if target_name:
                    return body, None, target_name
            return None
        return None

    def _walk_function(
        self, start: int, known_starts: set[int], owners: dict[int, int]
    ) -> list[Instruction]:
        visited: dict[int, Instruction] = {}
        pending = deque([start])
        max_instructions = 100_000

        while pending and len(visited) < max_instructions:
            pc = pending.popleft()
            steps = 0
            while len(visited) < max_instructions and steps < max_instructions:
                if pc in visited:
                    break
                if pc in known_starts and pc != start:
                    break
                if pc in owners and owners[pc] != start:
                    break
                instruction = self.instructions.get(pc)
                if instruction is None:
                    break
                visited[pc] = instruction
                steps += 1
                following = instruction.next_address

                if (
                    instruction.is_return
                    or instruction.is_trap
                    or self._call_is_noreturn(instruction)
                ):
                    # Returns, traps, and calls to no-return targets are terminal;
                    # do not absorb unreachable instructions after them.
                    break
                if instruction.is_jump:
                    targets = (
                        (instruction.branch_target,)
                        if instruction.branch_target is not None
                        else instruction.computed_targets
                    )
                    for target in targets:
                        if target not in self.instructions:
                            continue
                        if target == start or target not in known_starts:
                            pending.append(target)
                    if instruction.is_conditional_jump:
                        pc = following
                        continue
                    break
                pc = following

        return [visited[address] for address in sorted(visited)]

    def _build_blocks(self, instructions: Iterable[Instruction]) -> list[BasicBlock]:
        ordered = sorted(instructions, key=lambda instruction: instruction.address)
        if not ordered:
            return []
        addresses = {instruction.address for instruction in ordered}
        boundaries = {ordered[0].address}
        for instruction in ordered:
            following = instruction.next_address
            if instruction.is_jump:
                if instruction.branch_target in addresses:
                    boundaries.add(instruction.branch_target)
                boundaries.update(
                    target
                    for target in instruction.computed_targets
                    if target in addresses
                )
                if following in addresses:
                    boundaries.add(following)
            elif (
                instruction.is_return or instruction.is_trap
            ) and following in addresses:
                boundaries.add(following)

        chunks: list[list[Instruction]] = []
        current: list[Instruction] = []
        for instruction in ordered:
            if current and instruction.address in boundaries:
                chunks.append(current)
                current = []
            current.append(instruction)
            if (
                instruction.is_return
                or instruction.is_trap
                or (instruction.is_jump and not instruction.is_conditional_jump)
            ):
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        block_starts = {chunk[0].address for chunk in chunks}
        blocks: list[BasicBlock] = []
        for chunk in chunks:
            last = chunk[-1]
            successors: list[int] = []
            if last.is_jump:
                targets = (
                    (last.branch_target,)
                    if last.branch_target is not None
                    else last.computed_targets
                )
                successors.extend(
                    target for target in targets if target in block_starts
                )
                if last.is_conditional_jump and last.next_address in block_starts:
                    successors.append(last.next_address)
            elif (
                not last.is_return
                and not last.is_trap
                and last.next_address in block_starts
            ):
                successors.append(last.next_address)
            blocks.append(
                BasicBlock(
                    start=chunk[0].address,
                    instructions=tuple(chunk),
                    successors=tuple(dict.fromkeys(successors)),
                )
            )
        return blocks

    def _user_or_symbol_name(self, address: int, fallback: str) -> str:
        if self.project_state is not None:
            name = self.project_state.labels.get(address)
            if name:
                return name
        return self._name_by_address.get(address, fallback)

    def resolve_address(self, value: str | int) -> int:
        """
        Devuelve la dirección de memoria asociada a un nombre de símbolo,
        función o cadena hexadecimal. Si se pasa un entero, se devuelve tal cual.
        Para obtener el nombre asociado a una dirección, usa name_for_address().
        """
        if isinstance(value, int):
            return value
        token = value.strip()
        try:
            return int(token, 0)
        except ValueError:
            pass
        folded = token.casefold()
        for address, name in self._name_by_address.items():
            if name.casefold() == folded:
                return address
        for function in self.functions.values():
            if function.name.casefold() == folded:
                return function.address
        raise ValueError(f"No conozco la dirección o símbolo {token!r}.")

    def name_for_address(self, address: int) -> str | None:
        """
        Devuelve el nombre del símbolo, importación o función en la dirección dada.
        """
        if address in self._name_by_address:
            return self._name_by_address[address]
        if address in self.functions:
            return self.functions[address].name
        return None

    def function_at(self, value: str | int) -> Function:
        address = self.resolve_address(value)
        function = self.functions.get(address)
        if function is not None:
            return function
        # Permit a user to select an instruction inside a recovered function.
        for candidate in self.functions.values():
            if candidate.address <= address < candidate.end:
                return candidate
        raise ValueError(f"No hay una función recuperada en {address:#x}.")

    def name_at(self, address: int) -> str | None:
        if self.project_state is not None and address in self.project_state.labels:
            return self.project_state.labels[address]
        if address in self._name_by_address:
            return self._name_by_address[address]
        function = self.functions.get(address)
        return function.name if function else None

    def iter_instructions(
        self, start: int | "Function" | None = None, count: int | None = None
    ) -> list[Instruction]:
        """
        Devuelve las instrucciones a partir de una dirección o dentro de una función.
        
        Si 'start' es una Function, devuelve las instrucciones de esa función.
        Si 'start' es una dirección, devuelve las instrucciones secuenciales
        (ignorando los límites de las funciones).
        """
        from .model import Function
        if isinstance(start, Function):
            if count is not None:
                return start.instructions[: max(0, count)]
            return start.instructions

        addresses = self.instruction_addresses
        if start is not None:
            index = _lower_bound(addresses, start)
            addresses = addresses[index:]
        if count is not None:
            addresses = addresses[: max(0, count)]
        return [self.instructions[address] for address in addresses]

    def xrefs_to(self, address: int) -> list[Xref]:
        """
        Devuelve todas las referencias cruzadas (Xrefs) que apuntan a la dirección dada.
        """
        return sorted(
            self.xrefs_by_target.get(address, ()), key=lambda item: item.source
        )

    def xrefs_from(self, address: int) -> list[Xref]:
        return sorted(
            self.xrefs_by_source.get(address, ()), key=lambda item: item.target
        )

    def strings_matching(self, query: str = "") -> list[StringEntry]:
        folded = query.casefold()
        return [item for item in self.strings if folded in item.text.casefold()]

    def search_bytes(self, pattern: bytes) -> list[int]:
        if not pattern:
            raise ValueError("El patrón de búsqueda está vacío.")
        found: list[int] = []
        for section in self.image.sections:
            offset = 0
            while True:
                offset = section.data.find(pattern, offset)
                if offset < 0:
                    break
                found.append(section.start + offset)
                offset += 1
        return sorted(set(found))

    def format_instruction(self, instruction: Instruction) -> str:
        raw = " ".join(f"{byte:02x}" for byte in instruction.raw)
        text = f"{instruction.address:0{self.image.bits // 4}x}  {raw:<24} {instruction.mnemonic:<8} {instruction.op_str}".rstrip()
        annotation = None
        if instruction.branch_target is not None:
            target_name = self.name_at(instruction.branch_target)
            if target_name:
                annotation = f"->{target_name}"
        if annotation is None:
            for target in (
                *instruction.memory_references,
                *instruction.indexed_memory_references,
            ):
                target_name = self.name_at(target)
                if target_name:
                    annotation = f"[{target_name}]"
                    break
        if annotation is None and instruction.computed_targets:
            annotation = f"switch->{len(instruction.computed_targets)} destinos"
        if annotation is None and instruction.scalar_references:
            target = instruction.scalar_references[0]
            target_name = self.name_at(target) or format(target, "#x")
            annotation = f"scalar->{target_name}"
        comment = (
            self.project_state.comments.get(instruction.address)
            if self.project_state
            else None
        )
        suffix = " ; " + annotation if annotation else ""
        if comment:
            suffix += (" " if suffix else " ; ") + comment
        return text + suffix

    def format_function(self, function: Function) -> str:
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
        marker_text = f" [{' | '.join(markers)}]" if markers else ""
        lines = [
            f"{function.name} @ {function.address:#x}{marker_text}  "
            f"({len(function.instructions)} instrucciones, aprox. {function.size} bytes; origen: {function.source})"
        ]
        for instruction in function.instructions:
            if (
                instruction.address in self.function_seeds
                and instruction.address != function.address
            ):
                lines.append(
                    f"\n{self.name_at(instruction.address) or f'loc_{instruction.address:X}'}:"
                )
            lines.append("  " + self.format_instruction(instruction))
        return "\n".join(lines)

    def set_prototype(
        self,
        function: int | Function,
        signature: FunctionSignature,
        *,
        calling_convention: CallingConvention | str | None = None,
    ) -> FunctionSignature:
        """Attach a type-checked, ABI-resolved prototype to a function.

        ``FunctionSignature`` contains the source-level return/parameter types;
        the stored copy additionally records register, stack, aggregate-piece,
        hidden-sret, and variadic metadata. Registered aggregate types can be
        declared through ``analyzer.type_database`` before this call.
        """
        target = self.functions.get(function) if isinstance(function, int) else function
        if target is None:
            raise KeyError(f"No existe función en {function:#x}")

        if isinstance(calling_convention, CallingConvention):
            convention = calling_convention
        elif isinstance(calling_convention, str):
            convention = calling_convention_by_name(
                calling_convention, self.image.bits, self.image.format
            )
        elif signature.calling_convention not in {"", "inferred"}:
            convention = calling_convention_by_name(
                signature.calling_convention, self.image.bits, self.image.format
            )
        else:
            convention = calling_convention_for(self, target)

        return_type = self.type_database.resolve(
            signature.return_data_type or signature.return_type
        )
        return_base = return_type.unwrapped()
        if return_base.kind in {
            TypeKind.UNKNOWN,
            TypeKind.ARRAY,
            TypeKind.FUNCTION,
        }:
            raise ValueError(
                "Una función C no puede retornar un tipo desconocido, array o función"
            )
        if return_base.kind != TypeKind.VOID and return_base.size is None:
            raise ValueError("Una función C no puede retornar un agregado incompleto")
        for parameter in signature.parameters:
            parameter_type = self.type_database.resolve(
                parameter.data_type or parameter.type_name
            )
            parameter_base = parameter_type.unwrapped()
            parameter_kind = parameter_base.kind
            if parameter_kind in {TypeKind.VOID, TypeKind.UNKNOWN}:
                raise ValueError("Un parámetro requiere un tipo C definido y no void")
            if (
                parameter_kind not in {TypeKind.ARRAY, TypeKind.FUNCTION}
                and parameter_base.size is None
            ):
                raise ValueError(
                    "Un parámetro por valor no puede usar un agregado incompleto"
                )

        from .prototype import PrototypeResolver

        requested = replace(
            signature,
            parameters=tuple(
                replace(parameter, confidence=None, evidence=())
                for parameter in signature.parameters
            ),
            calling_convention=convention.name,
            source="user",
            locked=True,
            inference_confidence=None,
            inference_evidence=(),
        )
        resolved = (
            PrototypeResolver(self.type_database, self.image.bits)
            .resolve(requested, convention)
            .signature
        )
        target.calling_convention = convention.name
        target.signature = resolved
        self._calling_convention_overrides[target.address] = convention
        return resolved

    def infer_prototype(self, function: int | Function) -> FunctionSignature:
        """Return a locked signature or an unlocked candidate with heuristic evidence."""
        target = self.functions.get(function) if isinstance(function, int) else function
        if target is None:
            raise KeyError(f"No existe función en {function:#x}")
        if target.signature is not None and target.signature.locked:
            return target.signature
        self.pseudocode(target)
        if target.signature is None:
            raise RuntimeError("No se pudo generar un candidato de prototipo")
        return target.signature

    def pseudocode(self, function: Function) -> str:
        """Return experimental semantic pseudocode for a recovered function."""
        from .decompiler import SemanticDecompiler

        return SemanticDecompiler(self).decompile(function)

    def ssa(self, function: int | Function) -> SSAFunction:
        """Return the p-code SSA/heritage graph for a recovered function."""
        from .sleigh import SleighTranslator

        target = self.functions.get(function) if isinstance(function, int) else function
        if target is None:
            raise KeyError(f"No existe función en {function:#x}")
        convention = calling_convention_for(self, target)
        translator = SleighTranslator(self.image.bits)
        translations = translator.translate_instructions(target.instructions)
        return SSAHeritage(
            target,
            translations,
            caller_saved_registers=convention.caller_saved_registers,
            return_registers=(
                *convention.integer_return_registers,
                *convention.floating_return_registers,
            ),
            call_argument_registers=abi_argument_registers(convention),
            no_return_calls=(
                instruction.address
                for instruction in target.instructions
                if self._call_is_noreturn(instruction)
                and instruction.branch_target != instruction.next_address
            ),
            registers=translator.registers,
        ).build()

    def cfg_dot(self, function: Function) -> str:
        safe_name = function.name.replace('"', r"\"")
        lines = [
            "digraph cfg {",
            "  rankdir=TB;",
            f'  label="{safe_name} @ {function.address:#x}";',
        ]
        for block in function.blocks:
            label_lines = [
                f"{instruction.address:#x}: {instruction.mnemonic} {instruction.op_str}"
                for instruction in block.instructions
            ]
            label = "\\l".join(line.replace('"', r"\"") for line in label_lines) + "\\l"
            lines.append(
                f'  n{block.start:x} [shape=box,fontname="monospace",label="{label}"];'
            )
        for block in function.blocks:
            for successor in block.successors:
                lines.append(f"  n{block.start:x} -> n{successor:x};")
        lines.append("}")
        return "\n".join(lines)

    def summary(self) -> dict[str, int | str | None]:
        return {
            "file": str(self.image.path),
            "format": self.image.format,
            "architecture": self.image.architecture,
            "bits": self.image.bits,
            "entrypoint": self.image.entrypoint,
            "sections": len(self.image.sections),
            "executable_sections": len(self.image.executable_sections),
            "instructions": len(self.instructions),
            "functions": len(self.functions),
            "thunks": sum(function.is_thunk for function in self.functions.values()),
            "address_tables": len(self.address_tables),
            "computed_jumps": sum(
                bool(item.computed_targets) for item in self.instructions.values()
            ),
            "pattern_functions": len(self.function_start_candidates),
            "macho_function_starts": len(self.macho_function_start_candidates),
            "pe_exception_functions": len(
                getattr(self.image, "pe_exception_function_ranges", ())
            ),
            "pe_unwind_functions": sum(
                function.pe_unwind_info is not None
                for function in self.functions.values()
            ),
            "pe_tls_callbacks": len(getattr(self.image, "pe_tls_callbacks", ())),
            "no_return_functions": sum(
                function.no_return for function in self.functions.values()
            ),
            "stack_purge_functions": sum(
                function.stack_purge_size is not None
                for function in self.functions.values()
            ),
            "no_return_imports": len(self.no_return_imports),
            "strings": len(self.strings),
            "imports": len(self.image.imports),
            "exports": len(self.image.exports),
            "xrefs": len(self.xrefs),
            "sha256": self.image.sha256,
        }


def _lower_bound(values: list[int], value: int) -> int:
    left, right = 0, len(values)
    while left < right:
        middle = (left + right) // 2
        if values[middle] < value:
            left = middle + 1
        else:
            right = middle
    return left
