"""x86 calling-convention models ported from Ghidra compiler specifications.

The locations and register volatility below are small Python translations of
``x86win.cspec``, ``x86gcc.cspec``, ``x86-64-win.cspec`` and
``x86-64-gcc.cspec``.  Binary-only convention selection is necessarily
heuristic: a compiler specification describes a convention, but does not by
itself identify which convention an arbitrary function uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

try:
    from capstone import CS_ARCH_X86, CS_MODE_32, CS_MODE_64, Cs, CsError
    from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_OP_REG
except ImportError as exc:  # pragma: no cover - dependency declared by package
    raise RuntimeError(
        "Falta Capstone. Instálalo con: python -m pip install capstone"
    ) from exc


@dataclass(frozen=True, slots=True)
class CallingConvention:
    """Subset of a Ghidra compiler-spec prototype model."""

    name: str
    argument_registers: tuple[str, ...]
    stack_argument_offset: int
    integer_return_registers: tuple[str, ...]
    floating_return_registers: tuple[str, ...]
    caller_saved_registers: tuple[str, ...]
    callee_saved_registers: tuple[str, ...]
    vector_argument_registers: tuple[str, ...] = ()
    # Each group is one argument position with ABI-dependent register classes.
    register_argument_groups: tuple[tuple[str, ...], ...] = ()
    home_space_start: int = 0
    home_space_size: int = 0
    cleanup: str = "caller"
    print_name: bool = False
    hidden_argument_registers: tuple[str, ...] = ()

    @property
    def parameter_register_count(self) -> int:
        if self.register_argument_groups:
            return len(self.register_argument_groups)
        return len(self.argument_registers)

    def vector_parameter_name(self, register: str) -> str | None:
        """Return the ABI parameter name for an incoming vector register."""
        register = register.lower()
        if register not in self.vector_argument_registers:
            return None
        index = self.vector_argument_registers.index(register) + 1
        if self.register_argument_groups:
            return f"param_{index}"
        return f"float_param_{index}"


_X86_PRESERVED = ("ebx", "ebp", "esi", "edi", "esp")
_X86_VOLATILE = ("eax", "ecx", "edx", *(f"xmm{index}" for index in range(8)))
_X64_SYSV_PRESERVED = ("rbx", "rbp", "rsp", "r12", "r13", "r14", "r15")
_X64_SYSV_VOLATILE = (
    "rax",
    "rcx",
    "rdx",
    "rsi",
    "rdi",
    "r8",
    "r9",
    "r10",
    "r11",
    *(f"xmm{index}" for index in range(16)),
)
_X64_WIN_PRESERVED = (
    "rbx",
    "rbp",
    "rdi",
    "rsi",
    "rsp",
    "r12",
    "r13",
    "r14",
    "r15",
    *(f"xmm{index}" for index in range(6, 16)),
)
_X64_WIN_VOLATILE = (
    "rax",
    "rcx",
    "rdx",
    "r8",
    "r9",
    "r10",
    "r11",
    *(f"xmm{index}" for index in range(6)),
)

X86_CDECL = CallingConvention(
    "__cdecl",
    (),
    4,
    ("eax", "edx"),
    ("st0",),
    _X86_VOLATILE,
    _X86_PRESERVED,
)
X86_STDCALL = CallingConvention(
    "__stdcall",
    (),
    4,
    ("eax", "edx"),
    ("st0",),
    _X86_VOLATILE,
    _X86_PRESERVED,
    cleanup="callee",
)
X86_FASTCALL = CallingConvention(
    "__fastcall",
    ("ecx", "edx"),
    4,
    ("eax", "edx"),
    ("st0",),
    _X86_VOLATILE,
    _X86_PRESERVED,
    cleanup="callee",
    print_name=True,
)
X86_THISCALL = CallingConvention(
    "__thiscall",
    ("ecx",),
    4,
    ("eax", "edx"),
    ("st0",),
    _X86_VOLATILE,
    _X86_PRESERVED,
    cleanup="callee",
    print_name=True,
    hidden_argument_registers=("ecx",),
)
X86_GCC_THISCALL = CallingConvention(
    "__thiscall",
    (),
    4,
    ("eax", "edx"),
    ("st0",),
    _X86_VOLATILE,
    _X86_PRESERVED,
    print_name=True,
)
X86_REGPARM1 = CallingConvention(
    "__regparm1",
    ("eax",),
    4,
    ("eax", "edx"),
    ("st0",),
    _X86_VOLATILE,
    _X86_PRESERVED,
    print_name=True,
)
X86_REGPARM2 = CallingConvention(
    "__regparm2",
    ("eax", "edx"),
    4,
    ("eax", "edx"),
    ("st0",),
    _X86_VOLATILE,
    _X86_PRESERVED,
    print_name=True,
)
X86_REGPARM3 = CallingConvention(
    "__regparm3",
    ("eax", "edx", "ecx"),
    4,
    ("eax", "edx"),
    ("st0",),
    _X86_VOLATILE,
    _X86_PRESERVED,
    print_name=True,
)

SYSV_AMD64 = CallingConvention(
    "sysv_amd64",
    ("rdi", "rsi", "rdx", "rcx", "r8", "r9"),
    8,
    ("rax", "rdx"),
    ("xmm0", "xmm1"),
    _X64_SYSV_VOLATILE,
    _X64_SYSV_PRESERVED,
    vector_argument_registers=tuple(f"xmm{index}" for index in range(8)),
)
WINDOWS_X64 = CallingConvention(
    "__fastcall",
    ("rcx", "rdx", "r8", "r9"),
    40,
    ("rax",),
    ("xmm0",),
    _X64_WIN_VOLATILE,
    _X64_WIN_PRESERVED,
    vector_argument_registers=tuple(f"xmm{index}" for index in range(4)),
    register_argument_groups=tuple(
        (general, vector)
        for general, vector in zip(
            ("rcx", "rdx", "r8", "r9"),
            ("xmm0", "xmm1", "xmm2", "xmm3"),
            strict=True,
        )
    ),
    home_space_start=8,
    home_space_size=32,
)
WINDOWS_X64_THISCALL = replace(
    WINDOWS_X64,
    name="__thiscall",
    print_name=True,
    hidden_argument_registers=("rcx",),
)


def abi_argument_registers(
    convention: CallingConvention,
) -> tuple[tuple[str, str], ...]:
    """Return ABI register/name pairs used to seed callsite argument snapshots."""
    result: list[tuple[str, str]] = []
    if convention.register_argument_groups:
        for index, group in enumerate(convention.register_argument_groups, start=1):
            result.extend((f"param_{index}", register) for register in group)
    else:
        result.extend(
            (f"param_{index}", register)
            for index, register in enumerate(convention.argument_registers, start=1)
        )
        for index, register in enumerate(
            convention.vector_argument_registers, start=1
        ):
            result.append(
                (
                    convention.vector_parameter_name(register)
                    or f"float_param_{index}",
                    register,
                )
            )
    return tuple(dict.fromkeys(result))


def default_calling_convention(analyzer: Any) -> CallingConvention:
    """Return the ABI selected by Ghidra's architecture/compiler-spec defaults."""
    bits = int(analyzer.image.bits)
    format_name = str(getattr(analyzer.image, "format", ""))
    if bits == 64:
        return WINDOWS_X64 if format_name == "PE" else SYSV_AMD64
    return X86_STDCALL if format_name == "PE" else X86_CDECL


def calling_convention_by_name(
    name: str, bits: int, format_name: str = ""
) -> CallingConvention:
    """Return a named profile matching the local x86 Cspecs."""
    normalized = name.strip().lower()
    fmt = format_name.upper()
    if bits == 64:
        profiles = {
            "sysv_amd64": SYSV_AMD64,
            "__stdcall": SYSV_AMD64 if fmt != "PE" else WINDOWS_X64,
            "__fastcall": WINDOWS_X64,
            "__thiscall": WINDOWS_X64_THISCALL,
        }
    elif bits == 32:
        profiles = {
            "__cdecl": X86_CDECL,
            "__cdeclf": X86_CDECL,
            "__stdcall": X86_STDCALL,
            "__fastcall": X86_FASTCALL,
            "__thiscall": X86_THISCALL if fmt == "PE" else X86_GCC_THISCALL,
            "__regparm1": X86_REGPARM1,
            "__regparm2": X86_REGPARM2,
            "__regparm3": X86_REGPARM3,
        }
    else:
        raise ValueError("Solo se admiten convenciones x86 y x86-64")
    try:
        return profiles[normalized]
    except KeyError as exc:
        raise ValueError(f"Convención no compatible con x86: {name}") from exc


def calling_convention_for(analyzer: Any, function: Any) -> CallingConvention:
    """Infer a function's likely ABI from xrefs, entry reads and stack cleanup.

    Ghidra's ``.cspec`` files define the register/stack locations.  Its
    ``resolveprototype`` analysis is richer than this static heuristic, so this
    routine prefers the file-format default unless call-site evidence is clear.
    """
    overrides = getattr(analyzer, "_calling_convention_overrides", {})
    overridden = overrides.get(function.address)
    if overridden is not None:
        return overridden

    cache = getattr(analyzer, "_calling_convention_cache", None)
    if cache is None:
        cache = {}
        analyzer._calling_convention_cache = cache
    cached = cache.get(function.address)
    if cached is not None:
        return cached

    if function.is_thunk and function.thunk_target is not None:
        target = analyzer.functions.get(function.thunk_target)
        if target is not None and target.address != function.address:
            selected = calling_convention_for(analyzer, target)
            cache[function.address] = selected
            return selected

    default = default_calling_convention(analyzer)
    if int(analyzer.image.bits) == 64:
        selected = (
            WINDOWS_X64_THISCALL
            if str(getattr(analyzer.image, "format", "")) == "PE"
            and function.name.startswith("?")
            else default
        )
        cache[function.address] = selected
        return selected

    selected = _decorated_convention(function.name, default)
    if selected is None:
        selected = _infer_x86_convention(analyzer, function, default)
    cache[function.address] = selected
    return selected


def _decorated_convention(
    name: str, default: CallingConvention
) -> CallingConvention | None:
    """Recognize common Microsoft x86 symbol decorations."""
    if name.startswith("?"):
        return replace(X86_THISCALL, print_name=True)
    if re.match(r"^@[^@]+@\d+$", name):
        return replace(X86_FASTCALL, print_name=True)
    if re.match(r"^_[^@]+@\d+$", name):
        return replace(X86_STDCALL, print_name=True)
    if re.fullmatch(r"_[^@]+", name):
        return replace(X86_CDECL, print_name=True)
    return None


def _infer_x86_convention(
    analyzer: Any, function: Any, default: CallingConvention
) -> CallingConvention:
    is_windows = str(getattr(analyzer.image, "format", "")) == "PE"
    call_sites = [
        xref for xref in analyzer.xrefs_to(function.address) if xref.kind == "call"
    ]
    entry_reads = _entry_reads_before_write(analyzer, function, {"ecx", "edx", "eax"})
    call_writes = callsite_register_writes(analyzer, call_sites, {"ecx", "edx", "eax"})
    threshold = max(1, (len(call_sites) + 1) // 2)
    supplied = {
        register for register, sites in call_writes.items() if len(sites) >= threshold
    }
    evidence = entry_reads & supplied

    if is_windows:
        if {"ecx", "edx"} <= evidence:
            return X86_FASTCALL
        if "ecx" in evidence:
            return X86_THISCALL
    else:
        if {"eax", "edx", "ecx"} <= evidence:
            return X86_REGPARM3
        if {"eax", "edx"} <= evidence:
            return X86_REGPARM2
        if "eax" in evidence:
            return X86_REGPARM1

    purge = function.stack_purge_size
    if purge is not None and purge > 0:
        if is_windows or default.name == "__stdcall":
            return X86_STDCALL
        # ret imm16 is direct evidence of a callee-cleanup prototype even in
        # binaries whose compiler-spec default is cdecl.
        return X86_STDCALL

    caller_cleanup = _caller_cleanup_evidence(analyzer, call_sites)
    if caller_cleanup:
        return X86_CDECL
    return default


def _capstone(analyzer: Any) -> Cs:
    md = Cs(CS_ARCH_X86, CS_MODE_32 if analyzer.image.bits == 32 else CS_MODE_64)
    md.detail = True
    return md


def _register_families(raw: Any, register_ids: list[int], bits: int) -> set[str]:
    families: set[str] = set()
    for register_id in register_ids:
        name = raw.reg_name(register_id).lower()
        family = _register_family(name, bits)
        if family is not None:
            families.add(family)
    return families


def _register_family(name: str, bits: int) -> str | None:
    if re.fullmatch(r"xmm\d+", name):
        return name
    if bits == 64:
        x64_aliases = {
            "rax": "rax",
            "eax": "rax",
            "ax": "rax",
            "al": "rax",
            "ah": "rax",
            "rbx": "rbx",
            "ebx": "rbx",
            "bx": "rbx",
            "bl": "rbx",
            "bh": "rbx",
            "rcx": "rcx",
            "ecx": "rcx",
            "cx": "rcx",
            "cl": "rcx",
            "ch": "rcx",
            "rdx": "rdx",
            "edx": "rdx",
            "dx": "rdx",
            "dl": "rdx",
            "dh": "rdx",
            "rsi": "rsi",
            "esi": "rsi",
            "si": "rsi",
            "sil": "rsi",
            "rdi": "rdi",
            "edi": "rdi",
            "di": "rdi",
            "dil": "rdi",
            "rbp": "rbp",
            "ebp": "rbp",
            "bp": "rbp",
            "bpl": "rbp",
            "rsp": "rsp",
            "esp": "rsp",
            "sp": "rsp",
            "spl": "rsp",
        }
        family = x64_aliases.get(name)
        if family is not None:
            return family
    aliases = {
        "eax": "eax",
        "ax": "eax",
        "al": "eax",
        "ah": "eax",
        "ebx": "ebx",
        "bx": "ebx",
        "bl": "ebx",
        "bh": "ebx",
        "ecx": "ecx",
        "cx": "ecx",
        "cl": "ecx",
        "ch": "ecx",
        "edx": "edx",
        "dx": "edx",
        "dl": "edx",
        "dh": "edx",
        "esi": "esi",
        "si": "esi",
        "edi": "edi",
        "di": "edi",
        "ebp": "ebp",
        "bp": "ebp",
        "esp": "esp",
        "sp": "esp",
        "rax": "rax",
        "rbx": "rbx",
        "rcx": "rcx",
        "rdx": "rdx",
        "rsi": "rsi",
        "rdi": "rdi",
        "rbp": "rbp",
        "rsp": "rsp",
    }
    family = aliases.get(name)
    if family is not None:
        return family
    if bits == 64:
        match = re.fullmatch(r"r(8|9|1[0-5])(?:d|w|b)?", name)
        if match:
            return f"r{match.group(1)}"
    return None


def _decoded_accesses(analyzer: Any, instruction: Any):
    md = _capstone(analyzer)
    try:
        raw = next(md.disasm(instruction.raw, instruction.address, count=1), None)
        if raw is None:
            return None, set(), set()
        reads, writes = raw.regs_access()
        bits = int(analyzer.image.bits)
        return (
            raw,
            _register_families(raw, reads, bits),
            _register_families(raw, writes, bits),
        )
    except (CsError, ValueError):
        return None, set(), set()


def _entry_reads_before_write(
    analyzer: Any, function: Any, candidates: set[str]
) -> set[str]:
    written: set[str] = set()
    read_before_write: set[str] = set()
    for instruction in sorted(function.instructions, key=lambda item: item.address)[
        :64
    ]:
        raw, reads, writes = _decoded_accesses(analyzer, instruction)
        if raw is None:
            continue
        read_before_write.update((reads & candidates) - written)
        written.update(writes)
        if instruction.is_return or instruction.is_trap:
            break
        if instruction.is_jump and raw.mnemonic.lower() in {"jmp", "ljmp"}:
            break
    return read_before_write


def callsite_register_writes(
    analyzer: Any, call_sites: list[Any], candidates: set[str]
) -> dict[str, set[int]]:
    """Collect candidate-register writes in a bounded window before each call."""
    writes_by_register = {register: set() for register in candidates}
    for xref in call_sites:
        try:
            caller = analyzer.function_at(xref.source)
        except ValueError:
            continue
        index = next(
            (
                index
                for index, instruction in enumerate(caller.instructions)
                if instruction.address == xref.source
            ),
            None,
        )
        if index is None:
            continue
        found: set[str] = set()
        for instruction in reversed(caller.instructions[max(0, index - 16) : index]):
            raw, _reads, writes = _decoded_accesses(analyzer, instruction)
            if raw is None:
                continue
            found.update(writes & candidates)
            if found == candidates:
                break
            if (
                instruction.is_call
                or instruction.is_return
                or (instruction.is_jump and raw.mnemonic.lower() in {"jmp", "ljmp"})
            ):
                break
        for register in found:
            writes_by_register[register].add(xref.source)
    return writes_by_register


def callsite_stack_writes(
    analyzer: Any, call_sites: list[Any], candidate_offsets: set[int]
) -> dict[int, set[int]]:
    """Find bounded callsite writes to outgoing stack slots, relative to call SP."""
    writes_by_offset = {offset: set() for offset in candidate_offsets}
    bits = int(analyzer.image.bits)
    pointer_size = bits // 8
    stack_register = "rsp" if bits == 64 else "esp"
    store_prefixes = ("mov", "vmov", "fst", "set")

    for xref in call_sites:
        try:
            caller = analyzer.function_at(xref.source)
        except ValueError:
            continue
        call_index = next(
            (
                index
                for index, instruction in enumerate(caller.instructions)
                if instruction.address == xref.source
            ),
            None,
        )
        if call_index is None:
            continue

        # Walk backward from the call and express stack locations relative to
        # the caller's SP at the call instruction.  Stop at control-flow edges
        # and frame setup to avoid counting a saved-register prologue as args.
        stack_adjustment = 0
        prior = caller.instructions[max(0, call_index - 24) : call_index]
        for instruction in reversed(prior):
            raw, _reads, register_writes = _decoded_accesses(analyzer, instruction)
            if raw is None:
                continue
            mnemonic = raw.mnemonic.lower()
            if instruction.is_call or instruction.is_return or instruction.is_trap:
                break
            if instruction.is_jump:
                break
            operands = raw.operands
            if (
                mnemonic.startswith("mov")
                and len(operands) >= 2
                and operands[0].type == X86_OP_REG
                and operands[1].type == X86_OP_REG
                and raw.reg_name(operands[0].reg).lower()
                in {"ebp", "rbp"}
                and raw.reg_name(operands[1].reg).lower() == stack_register
            ):
                break

            if mnemonic == "push":
                if stack_adjustment in writes_by_offset:
                    writes_by_offset[stack_adjustment].add(xref.source)
                stack_adjustment += pointer_size
                continue
            if mnemonic == "pop":
                stack_adjustment -= pointer_size
                continue

            if (
                mnemonic in {"sub", "add"}
                and len(operands) >= 2
                and operands[0].type == X86_OP_REG
                and raw.reg_name(operands[0].reg).lower() == stack_register
                and operands[1].type == X86_OP_IMM
            ):
                amount = int(operands[1].imm)
                stack_adjustment += amount if mnemonic == "sub" else -amount
                continue
            if (
                mnemonic == "lea"
                and len(operands) >= 2
                and operands[0].type == X86_OP_REG
                and raw.reg_name(operands[0].reg).lower() == stack_register
                and operands[1].type == X86_OP_MEM
                and raw.reg_name(operands[1].mem.base).lower() == stack_register
                and not operands[1].mem.index
            ):
                stack_adjustment -= int(operands[1].mem.disp)
                continue
            if stack_register in register_writes:
                break

            if (
                mnemonic.startswith(store_prefixes)
                and operands
                and operands[0].type == X86_OP_MEM
            ):
                memory = operands[0].mem
                base = raw.reg_name(memory.base).lower() if memory.base else ""
                index = raw.reg_name(memory.index).lower() if memory.index else ""
                if base == stack_register and not index:
                    offset = stack_adjustment + int(memory.disp)
                    if offset in writes_by_offset:
                        writes_by_offset[offset].add(xref.source)

    return writes_by_offset


def _caller_cleanup_evidence(analyzer: Any, call_sites: list[Any]) -> bool:
    if not call_sites:
        return False
    cleanup_sizes: list[int] = []
    for xref in call_sites:
        try:
            caller = analyzer.function_at(xref.source)
        except ValueError:
            continue
        index = next(
            (
                index
                for index, instruction in enumerate(caller.instructions)
                if instruction.address == xref.source
            ),
            None,
        )
        if index is None:
            continue
        cleanup = 0
        for instruction in caller.instructions[index + 1 : index + 5]:
            raw, _reads, _writes = _decoded_accesses(analyzer, instruction)
            if raw is None:
                break
            if raw.mnemonic.lower() == "add" and len(raw.operands) >= 2:
                destination, amount = raw.operands[:2]
                if (
                    destination.type == X86_OP_REG
                    and raw.reg_name(destination.reg).lower() == "esp"
                    and amount.type == X86_OP_IMM
                    and int(amount.imm) > 0
                ):
                    cleanup += int(amount.imm)
                    break
            if raw.mnemonic.lower() == "lea" and len(raw.operands) >= 2:
                destination, address = raw.operands[:2]
                if (
                    destination.type == X86_OP_REG
                    and raw.reg_name(destination.reg).lower() == "esp"
                    and address.type == X86_OP_MEM
                    and raw.reg_name(address.mem.base).lower() == "esp"
                    and address.mem.index == 0
                    and int(address.mem.disp) > 0
                ):
                    cleanup += int(address.mem.disp)
                    break
            if raw.mnemonic.lower().startswith("pop"):
                cleanup += int(analyzer.image.bits) // 8
                continue
            if instruction.is_call or instruction.is_jump or instruction.is_return:
                break
        if cleanup:
            cleanup_sizes.append(cleanup)
    if not cleanup_sizes:
        return False
    return len(cleanup_sizes) >= max(1, (len(call_sites) + 1) // 2)
