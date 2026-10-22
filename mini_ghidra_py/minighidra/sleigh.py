"""Real SLEIGH translation through the pypcode bindings.

The surrounding analyzer and pseudocode renderer are Python. This module
uses pypcode's native SLEIGH runtime with compiled x86 language definitions
from this repository's Ghidra 12.3 checkout. It obtains real,
instruction-addressed p-code instead of pretending the reduced
symbolic-expression graph is SLEIGH output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import ClassVar
from xml.etree.ElementTree import Element

import pypcode

from .model import Instruction


@dataclass(frozen=True, slots=True)
class SleighVarnode:
    """A pypcode/Ghidra varnode with its address-space coordinates."""

    space: str
    offset: int
    size: int
    display: str

    @property
    def size_bits(self) -> int:
        return self.size * 8

    @property
    def is_register(self) -> bool:
        return self.space == "register"

    @property
    def is_constant(self) -> bool:
        return self.space == "const"

    @property
    def is_unique(self) -> bool:
        return self.space == "unique"


@dataclass(frozen=True, slots=True)
class SleighPcodeOp:
    """One exact SLEIGH-generated p-code operation."""

    address: int
    sequence: int
    opcode: str
    inputs: tuple[SleighVarnode, ...]
    output: SleighVarnode | None


@dataclass(frozen=True, slots=True)
class SleighInstruction:
    """SLEIGH translation and status for one recovered machine instruction."""

    address: int
    length: int
    mnemonic: str
    assembly: str
    ops: tuple[SleighPcodeOp, ...]
    error: str | None = None

    @property
    def translated(self) -> bool:
        return self.error is None and bool(self.ops)

    @property
    def operation_names(self) -> tuple[str, ...]:
        return tuple(operation.opcode for operation in self.ops)


class SleighTranslator:
    """Translate x86/x86-64 instructions using Ghidra's SLEIGH specifications."""

    _LANGUAGES: ClassVar[dict[int, tuple[str, str, str]]] = {
        32: ("x86:LE:32:default", "x86.sla", "x86.pspec"),
        64: ("x86:LE:64:default", "x86-64.sla", "x86-64.pspec"),
    }

    def __init__(self, bits: int) -> None:
        try:
            self.language_id, slafile, processorspec = self._LANGUAGES[bits]
        except KeyError as exc:
            raise ValueError(f"SLEIGH x86 no admite {bits} bits") from exc
        self.bits = bits
        specs_dir = Path(__file__).resolve().parent / "data" / "sleigh"
        slafile_path = specs_dir / slafile
        pspec_path = specs_dir / processorspec
        if not slafile_path.is_file() or not pspec_path.is_file():
            raise RuntimeError(
                f"Faltan definiciones SLEIGH empaquetadas para x86-{bits}: "
                f"{slafile_path}"
            )
        language_definition = Element(
            "language",
            {
                "id": self.language_id,
                "slafile": slafile,
                "processorspec": processorspec,
            },
        )
        language = pypcode.ArchLanguage(str(specs_dir), language_definition)
        self._context = pypcode.Context(language)
        # pypcode contexts carry translation state; serialize use in case an
        # Analyzer is shared by concurrent Python callers.
        self._lock = RLock()

    def translate_instruction(
        self,
        raw: bytes,
        address: int,
        *,
        mnemonic: str = "",
        op_str: str = "",
        expected_length: int | None = None,
    ) -> SleighInstruction:
        """Translate one instruction, preserving unsupported cases as evidence."""
        assembly = " ".join(part for part in (mnemonic, op_str) if part).strip()
        if not raw:
            return SleighInstruction(
                address,
                expected_length or 0,
                mnemonic,
                assembly,
                (),
                "empty instruction bytes",
            )
        try:
            with self._lock:
                translation = self._context.translate(
                    raw,
                    base_address=address,
                    max_bytes=len(raw),
                    max_instructions=1,
                )
                native_operations = translation.ops
                operations = tuple(
                    SleighPcodeOp(
                        address=address,
                        sequence=index,
                        opcode=operation.opcode.name,
                        inputs=tuple(
                            _convert_varnode(varnode) for varnode in operation.inputs
                        ),
                        output=(
                            _convert_varnode(operation.output)
                            if operation.output is not None
                            else None
                        ),
                    )
                    for index, operation in enumerate(native_operations)
                )
                imarks = [
                    varnode
                    for operation in native_operations
                    if operation.opcode == pypcode.OpCode.IMARK
                    for varnode in operation.inputs
                    if varnode.space.name == "ram"
                ]
                actual_length = (
                    imarks[0].size if imarks else (expected_length or len(raw))
                )
            if not operations:
                return SleighInstruction(
                    address,
                    actual_length,
                    mnemonic,
                    assembly,
                    (),
                    "SLEIGH returned no p-code operations",
                )
            return SleighInstruction(
                address, actual_length, mnemonic, assembly, operations
            )
        except Exception as exc:  # noqa: BLE001
            # Errors raised by the native SLEIGH bindings vary by decoder path.
            return SleighInstruction(
                address,
                expected_length or len(raw),
                mnemonic,
                assembly,
                (),
                f"{type(exc).__name__}: {exc}",
            )

    def translate_instructions(
        self, instructions: Iterable[Instruction]
    ) -> tuple[SleighInstruction, ...]:
        """Translate recovered instructions independently in address order."""
        return tuple(
            self.translate_instruction(
                instruction.raw,
                instruction.address,
                mnemonic=instruction.mnemonic,
                op_str=instruction.op_str,
                expected_length=instruction.size,
            )
            for instruction in sorted(instructions, key=lambda item: item.address)
        )


def _convert_varnode(varnode: pypcode.Varnode) -> SleighVarnode:
    """Copy a native pypcode varnode without calling unsafe space-specific APIs."""
    return SleighVarnode(
        space=str(varnode.space.name),
        offset=int(varnode.offset),
        size=int(varnode.size),
        display=str(varnode),
    )
