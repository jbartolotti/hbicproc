from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .dataset import AnalysisRun
from .derivatives import DerivativePathBuilder


@dataclass(frozen=True)
class AnalysisOutputInventory:
    """Expected core outputs for one subject/session/task/run analysis."""

    output_dir: Path
    analysis_run: AnalysisRun
    contrast_names: tuple[str, ...]

    @classmethod
    def for_run(
        cls,
        output_dir: str | Path,
        analysis_run: AnalysisRun,
        contrasts: Iterable[str],
    ) -> "AnalysisOutputInventory":
        contrast_names = tuple(
            dict.fromkeys(str(name).strip() for name in contrasts if str(name).strip())
        )
        return cls(Path(output_dir), analysis_run, contrast_names)

    def required_outputs(self) -> tuple[Path, ...]:
        run_label = self.analysis_run.run or "1"
        common_entities = {
            "subject": self.analysis_run.subject,
            "session": self.analysis_run.session,
            "task": self.analysis_run.task,
            "run": run_label,
        }
        outputs = [
            DerivativePathBuilder.build(
                self.output_dir,
                **common_entities,
                desc="design_matrix",
                suffix=".csv",
            )
        ]
        for contrast_name in self.contrast_names:
            for stat in ("effect", "z", "variance"):
                outputs.append(
                    DerivativePathBuilder.build(
                        self.output_dir,
                        **common_entities,
                        desc=contrast_name,
                        stat=stat,
                        suffix=".nii.gz",
                    )
                )
        return tuple(outputs)

    def missing_outputs(self) -> tuple[Path, ...]:
        return tuple(path for path in self.required_outputs() if not path.is_file())

    def is_complete(self) -> bool:
        return not self.missing_outputs()