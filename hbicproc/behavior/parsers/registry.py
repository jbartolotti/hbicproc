from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Type

from .base import TaskParser


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: dict[str, Type[TaskParser]] = {}

    def register(self, name: str, parser_cls: Type[TaskParser]) -> None:
        self._parsers[name.lower()] = parser_cls

    def get(self, name: str) -> Type[TaskParser]:
        parser_cls = self._parsers.get(name.lower())
        if parser_cls is None:
            raise KeyError(f"No parser registered for task '{name}'.")
        return parser_cls

    def load_plugins(self, plugin_dir: str | Path | None = None) -> None:
        if not plugin_dir:
            return
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists() or not plugin_path.is_dir():
            return
        for module_path in sorted(plugin_path.glob("*.py")):
            if module_path.name.startswith("_") or module_path.name == "__init__.py":
                continue
            self._load_module(module_path)

    def _load_module(self, module_path: Path) -> None:
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for _, candidate in vars(module).items():
            if not isinstance(candidate, type):
                continue
            if not issubclass(candidate, TaskParser) or candidate is TaskParser:
                continue
            task_name = getattr(candidate, "task_name", None)
            if not task_name:
                task_name = module_path.stem
            self.register(task_name, candidate)
