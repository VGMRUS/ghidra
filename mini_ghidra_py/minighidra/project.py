"""Persistent user annotations, keyed to the input file's SHA-256."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class ProjectState:
    VERSION = 1

    def __init__(self) -> None:
        self.labels: dict[int, str] = {}
        self.comments: dict[int, str] = {}

    @staticmethod
    def default_path(binary_path: Path) -> Path:
        return binary_path.with_name(binary_path.name + ".minighidra.json")

    def save(self, path: str | Path, image) -> Path:
        destination = Path(path).expanduser().resolve()
        payload = {
            "format_version": self.VERSION,
            "binary_path": str(image.path),
            "binary_sha256": image.sha256,
            "labels": {f"0x{address:x}": name for address, name in sorted(self.labels.items())},
            "comments": {f"0x{address:x}": text for address, text in sorted(self.comments.items())},
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=destination.name + ".",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary_path, destination)
        except OSError:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        return destination

    @classmethod
    def load(cls, path: str | Path, image) -> "ProjectState":
        source = Path(path).expanduser().resolve()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"No se pudo leer el proyecto {source}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"El proyecto no es JSON válido: {exc}") from exc

        if not isinstance(payload, dict) or payload.get("format_version") != cls.VERSION:
            raise ValueError("Versión de proyecto no compatible.")
        if payload.get("binary_sha256") != image.sha256:
            raise ValueError("El proyecto pertenece a otro binario (SHA-256 diferente).")

        state = cls()
        state.labels = _parse_annotations(payload.get("labels", {}), "etiquetas")
        state.comments = _parse_annotations(payload.get("comments", {}), "comentarios")
        return state


def _parse_annotations(value, description: str) -> dict[int, str]:
    if not isinstance(value, dict):
        raise ValueError(f"El campo de {description} debe ser un objeto JSON.")
    result: dict[int, str] = {}
    for raw_address, text in value.items():
        try:
            address = int(str(raw_address), 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Dirección inválida en {description}: {raw_address!r}.") from exc
        if not isinstance(text, str):
            raise ValueError(f"El contenido de {description} debe ser texto.")
        result[address] = text
    return result
