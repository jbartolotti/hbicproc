from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import resample_to_img


@dataclass(frozen=True)
class Atlas:
    """An atlas image and optional labels resolved from the external cache."""

    name: str
    maps: Path
    labels: tuple[str, ...] = ()


class AtlasCache:
    """Fetch and cache atlas source files outside the BIDS tree."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else (
            Path.home() / ".cache" / "hbicproc" / "atlases"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._atlases: dict[str, Atlas] = {}

    def get(self, name: str) -> Atlas:
        atlas_name = str(name).strip()
        if atlas_name in self._atlases:
            return self._atlases[atlas_name]
        configured_path = Path(atlas_name).expanduser()
        if configured_path.exists():
            atlas = Atlas(name=atlas_name, maps=configured_path)
            self._atlases[atlas_name] = atlas
            return atlas
        if not atlas_name or atlas_name in {".", ".."} or "/" in atlas_name or "\\" in atlas_name:
            raise ValueError(f"Atlas names must be single path components: {name!r}.")

        if not atlas_name.lower().startswith("schaefer"):
            raise ValueError(
                f"Unsupported atlas '{atlas_name}'. Use a Schaefer atlas name or an existing image path."
            )
        try:
            n_rois = int(atlas_name[len("schaefer"):])
        except ValueError as exc:
            raise ValueError(f"Schaefer atlas '{atlas_name}' must include its ROI count.") from exc
        if n_rois <= 0:
            raise ValueError(f"Atlas '{atlas_name}' must have a positive ROI count.")

        fetched = fetch_atlas_schaefer_2018(
            n_rois=n_rois,
            yeo_networks=7,
            resolution_mm=2,
            data_dir=str(self.cache_dir),
            verbose=0,
        )
        maps = Path(str(fetched.maps))
        labels = tuple(str(label) for label in getattr(fetched, "labels", ()))
        atlas = Atlas(name=atlas_name, maps=maps, labels=labels)
        self._atlases[atlas_name] = atlas
        return atlas

    def resample(self, name: str, target_img: Any, output_path: str | Path) -> Atlas:
        atlas = self.get(name)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        resampled = resample_to_img(str(atlas.maps), target_img, interpolation="nearest")
        resampled.to_filename(output)
        return atlas

    def roi_activation_summary(
        self,
        name: str,
        images: dict[str, Any],
        target_img: Any,
        output_path: str | Path,
    ) -> Path:
        atlas = self.get(name)
        atlas_image = resample_to_img(str(atlas.maps), target_img, interpolation="nearest")
        labels = np.asarray(atlas_image.get_fdata())
        rows = []
        for roi in sorted(int(value) for value in np.unique(labels) if value > 0):
            row: dict[str, Any] = {"roi": roi}
            mask = labels == roi
            for condition, image in images.items():
                values = np.asarray(image.get_fdata())
                row[condition] = float(np.nanmean(values[mask]))
            rows.append(row)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
        return output

    def roi_residual_timeseries(
        self,
        name: str,
        residuals: list[Any],
        target_img: Any,
        output_path: str | Path,
    ) -> Path:
        atlas = self.get(name)
        atlas_image = resample_to_img(str(atlas.maps), target_img, interpolation="nearest")
        labels = np.asarray(atlas_image.get_fdata())
        rois = sorted(int(value) for value in np.unique(labels) if value > 0)
        rows = []
        for run_index, residual in enumerate(residuals, start=1):
            values = np.asarray(residual.get_fdata())
            if values.ndim == 3:
                values = values[..., np.newaxis]
            for timepoint in range(values.shape[-1]):
                for roi in rois:
                    rows.append({
                        "run": run_index,
                        "timepoint": timepoint,
                        "roi": roi,
                        "value": float(np.nanmean(values[..., timepoint][labels == roi])),
                    })
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
        return output


__all__ = ["Atlas", "AtlasCache"]
