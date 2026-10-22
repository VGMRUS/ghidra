"""Known non-returning function names, ported from Ghidra's name analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_DATA_DIR = Path(__file__).with_name("data") / "noreturn"
_FILES_BY_FORMAT = {
    "ELF": ("elf.txt", "golang.txt", "rust.txt"),
    "PE": ("pe.txt", "golang.txt", "rust.txt"),
    "Mach-O": ("macho.txt", "golang.txt", "rust.txt"),
    # Raw images have no format/compiler metadata; using all catalogs is useful
    # only when a user supplies a label or symbol for a raw address.
    "RAW": ("elf.txt", "pe.txt", "macho.txt", "golang.txt", "rust.txt"),
}


@dataclass(frozen=True)
class NoReturnCatalog:
    exact_names: frozenset[str]
    prefix_names: tuple[str, ...]

    @classmethod
    def for_format(cls, format_name: str) -> "NoReturnCatalog":
        names: set[str] = set()
        prefixes: set[str] = set()
        for filename in _FILES_BY_FORMAT.get(format_name, ()):
            path = _DATA_DIR / filename
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                # The package can still run if an optional data file was
                # omitted by a downstream packager.
                continue
            for line in lines:
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                # The Ghidra analyzer stores wildcard names with one trailing
                # '*' and treats them as prefix matches.
                if entry.endswith("*"):
                    prefixes.add(_normalize_name(entry[:-1]))
                else:
                    names.add(_normalize_name(entry))
        return cls(frozenset(name for name in names if name), tuple(sorted(name for name in prefixes if name)))

    def match(self, symbol_name: str) -> str | None:
        """Return the matching catalog name, or ``None``."""
        normalized = _normalize_name(symbol_name)
        if not normalized:
            return None
        if normalized in self.exact_names:
            return normalized
        for prefix in self.prefix_names:
            if normalized.startswith(prefix):
                return prefix + "*"
        return None


def _normalize_name(name: str) -> str:
    value = name.strip()
    # LIEF can expose ELF symbol versions or PLT helper suffixes as part of a
    # name. Ghidra's loader normally presents the undecorated symbol name.
    if value.endswith("@plt"):
        value = value[:-4]
    elif not value.startswith("?") and "@" in value:
        value = value.split("@", 1)[0]
    if value.startswith("__imp_"):
        value = value[len("__imp_") :]
    return value.lstrip("_")
