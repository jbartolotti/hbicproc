from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd


class EPrimeReadError(RuntimeError):
    """Raised when an E-Prime file cannot be read."""


class EPrimeReader:
    """Simple abstraction over reading E-Prime .edat3 files."""

    def __init__(self, backend: str = "auto") -> None:
        self.backend = backend

    def read(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"E-Prime file not found: {path}")
        if path.suffix.lower() != ".edat3":
            raise ValueError(f"Expected an .edat3 file, received: {path}")

        for candidate in self._backend_candidates():
            try:
                data_frame = self._read_with_backend(candidate, path)
            except Exception:
                continue
            if data_frame is not None:
                return self._normalize(data_frame)

        raise EPrimeReadError(
            "Unable to read E-Prime file. Install a supported E-Prime reader such as 'edat' or provide a tabular export."
        )

    def _backend_candidates(self) -> list[str]:
        if self.backend != "auto":
            return [self.backend]
        return ["edat", "edat2", "pyedat", "pandas"]

    def _read_with_backend(self, backend: str, path: Path) -> pd.DataFrame | None:
        if backend == "pandas":
            return self._read_text_export(path)

        module_name = backend
        module = importlib.import_module(module_name)
        for attribute_name in ("read", "read_edat", "read_file", "load"):
            attribute = getattr(module, attribute_name, None)
            if callable(attribute):
                result = attribute(path)
                if isinstance(result, pd.DataFrame):
                    return result
                if isinstance(result, (list, tuple)) and result:
                    candidate = result[0]
                    if isinstance(candidate, pd.DataFrame):
                        return candidate
        for attribute_name in ("EDAT", "EPrimeReader"):
            attribute = getattr(module, attribute_name, None)
            if callable(attribute):
                instance = attribute(path)
                if hasattr(instance, "read"):
                    result = instance.read()
                    if isinstance(result, pd.DataFrame):
                        return result
        return None

    def _read_text_export(self, path: Path) -> pd.DataFrame:
        for delimiter in ("\t", ","):
            try:
                return pd.read_csv(path, sep=delimiter, engine="python")
            except Exception:
                continue
        raise EPrimeReadError(f"Unable to parse text export at {path}")

    def _normalize(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        normalized = data_frame.copy()
        for column in normalized.columns:
            if pd.api.types.is_object_dtype(normalized[column]):
                normalized[column] = normalized[column].astype(str).str.strip()
        for column in normalized.columns:
            if column.lower() in {"accuracy", "correct", "iscorrect"}:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("Int64")
            elif column.lower() in {"rt", "response_time", "responsetime", "stimulusonsettime", "stimulusduration", "scannertriggertime"}:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        return normalized
