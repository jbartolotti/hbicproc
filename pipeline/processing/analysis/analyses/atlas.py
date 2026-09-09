from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import (
    fetch_atlas_aal,
    fetch_atlas_harvard_oxford,
    fetch_atlas_schaefer_2018,
)
from nilearn.image import new_img_like, resample_to_img


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AtlasParcel:
    """Standardized metadata for one atlas parcel."""

    parcel_id: int
    parcel_label: str
    atlas_name: str = ""
    network: str = ""
    hemisphere: str = ""


@dataclass(frozen=True)
class Atlas:
    """An atlas image and optional labels resolved from the external cache."""

    name: str
    maps: Path
    labels: tuple[str, ...] = ()
    metadata: tuple[AtlasParcel, ...] = ()


class AtlasCache:
    """Fetch and cache atlas source files outside the BIDS tree."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else (
            Path.home() / ".cache" / "hbicproc" / "atlases"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._atlases: dict[str, Atlas] = {}
        self._specifications: dict[str, dict[str, Any]] = {}

    def register(self, name: str, specification: Mapping[str, Any] | None = None) -> None:
        """Register a configured atlas name before extraction begins."""

        if specification is not None:
            self._specifications[str(name)] = dict(specification)

    def get(self, name: str) -> Atlas:
        atlas_name = str(name).strip()
        if atlas_name in self._atlases:
            return self._atlases[atlas_name]
        specification = self._specifications.get(atlas_name, {})
        atlas_type = str(specification.get("type", "")).strip().lower()
        if atlas_type == "custom_label_atlas":
            atlas = self._load_custom_label_atlas(atlas_name, specification)
            self._atlases[atlas_name] = atlas
            return atlas
        if atlas_type == "coordinate_spheres":
            atlas = self._load_coordinate_sphere_metadata(atlas_name, specification)
            self._atlases[atlas_name] = atlas
            return atlas
        configured_path = Path(atlas_name).expanduser()
        if configured_path.exists():
            atlas = Atlas(name=atlas_name, maps=configured_path)
            self._atlases[atlas_name] = atlas
            return atlas
        if not atlas_name or atlas_name in {".", ".."} or "/" in atlas_name or "\\" in atlas_name:
            raise ValueError(f"Atlas names must be single path components: {name!r}.")

        normalized = atlas_name.lower().replace("-", "_")
        if normalized.startswith("schaefer"):
            try:
                n_rois = int(normalized[len("schaefer"):])
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
        elif normalized in {"harvard_oxford_cortical", "harvardoxfordcortical"}:
            fetched = fetch_atlas_harvard_oxford(
                "cort-maxprob-thr25-2mm", data_dir=str(self.cache_dir), verbose=0
            )
        elif normalized in {"harvard_oxford_subcortical", "harvardoxfordsubcortical"}:
            fetched = fetch_atlas_harvard_oxford(
                "sub-maxprob-thr25-2mm", data_dir=str(self.cache_dir), verbose=0
            )
        elif normalized == "aal":
            fetched = fetch_atlas_aal(data_dir=str(self.cache_dir), verbose=0)
        else:
            raise ValueError(
                f"Unsupported atlas '{atlas_name}'. Use a built-in atlas, a configured custom atlas, "
                "or an existing image path."
            )
        maps = Path(str(fetched.maps))
        labels = tuple(str(label) for label in getattr(fetched, "labels", ()))
        atlas = Atlas(name=atlas_name, maps=maps, labels=labels)
        label_metadata = self._parcel_metadata(atlas)
        atlas = Atlas(
            name=atlas_name,
            maps=maps,
            labels=labels,
            metadata=tuple(
                AtlasParcel(
                    parcel_id=parcel_id,
                    parcel_label=parcel["parcel_label"],
                    atlas_name=atlas_name,
                    network=parcel["network"],
                    hemisphere=parcel["hemisphere"],
                )
                for parcel_id, parcel in label_metadata.items()
            ),
        )
        self._atlases[atlas_name] = atlas
        return atlas

    def _load_custom_label_atlas(
        self, name: str, specification: Mapping[str, Any]
    ) -> Atlas:
        atlas_path = self._required_path(specification, "atlas_file", name)
        labels_path = self._required_path(specification, "labels_file", name)
        image = nib.load(str(atlas_path))
        values = image.get_fdata()
        if not np.all(np.isfinite(values)) or not np.allclose(values, np.round(values)):
            raise ValueError(f"Custom atlas '{name}' must contain integer ROI labels.")
        table = pd.read_csv(labels_path, sep="\t")
        required = {"parcel_id", "parcel_label"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"Custom atlas labels file is missing columns: {sorted(missing)}.")
        metadata = self._metadata_from_table(name, table)
        self._validate_metadata_against_image(name, image, metadata)
        return Atlas(name=name, maps=atlas_path, metadata=metadata)

    def _load_coordinate_sphere_metadata(
        self, name: str, specification: Mapping[str, Any]
    ) -> Atlas:
        roi_path = self._required_path(specification, "roi_file", name)
        table = pd.read_csv(roi_path, sep="\t")
        required = {"roi_label", "x", "y", "z", "radius_mm"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"Coordinate sphere file is missing columns: {sorted(missing)}.")
        if table.empty:
            raise ValueError(f"Coordinate sphere file for '{name}' is empty.")
        if (pd.to_numeric(table["radius_mm"], errors="coerce") <= 0).any():
            raise ValueError(f"Coordinate sphere atlas '{name}' requires positive radius_mm values.")
        metadata = tuple(
            AtlasParcel(
                parcel_id=index,
                parcel_label=str(row["roi_label"]),
                atlas_name=name,
                network=str(row.get("network", "")),
                hemisphere=str(row.get("hemisphere", "")),
            )
            for index, (_, row) in enumerate(table.iterrows(), start=1)
        )
        return Atlas(name=name, maps=Path(), metadata=metadata)

    @staticmethod
    def _required_path(specification: Mapping[str, Any], key: str, name: str) -> Path:
        value = str(specification.get(key, "")).strip()
        if not value:
            raise ValueError(f"Configured atlas '{name}' requires '{key}'.")
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configured atlas '{name}' file does not exist: {path}")
        return path

    @staticmethod
    def _metadata_from_table(name: str, table: pd.DataFrame) -> tuple[AtlasParcel, ...]:
        metadata = []
        for _, row in table.iterrows():
            parcel_id = int(row["parcel_id"])
            if parcel_id <= 0:
                raise ValueError(f"Atlas '{name}' parcel_id values must be positive integers.")
            metadata.append(
                AtlasParcel(
                    parcel_id=parcel_id,
                    parcel_label=str(row["parcel_label"]),
                    atlas_name=name,
                    network=str(row.get("network", "")),
                    hemisphere=str(row.get("hemisphere", "")),
                )
            )
        if len({parcel.parcel_id for parcel in metadata}) != len(metadata):
            raise ValueError(f"Atlas '{name}' labels contain duplicate parcel_id values.")
        return tuple(metadata)

    @staticmethod
    def _validate_metadata_against_image(
        name: str, image: Any, metadata: tuple[AtlasParcel, ...]
    ) -> None:
        values = np.round(image.get_fdata()).astype(np.int32)
        image_ids = {int(value) for value in np.unique(values) if value > 0}
        metadata_ids = {parcel.parcel_id for parcel in metadata}
        missing = sorted(image_ids - metadata_ids)
        unused = sorted(metadata_ids - image_ids)
        if missing:
            raise ValueError(f"Atlas '{name}' labels are missing parcel IDs: {missing}.")
        if unused:
            LOGGER.warning("Atlas '%s' labels have no atlas voxels for parcel IDs: %s", name, unused)

    def _atlas_for(self, name: str, target_img: Any) -> Atlas:
        atlas = self.get(name)
        specification = self._specifications.get(str(name), {})
        if str(specification.get("type", "")).strip().lower() != "coordinate_spheres":
            return atlas
        roi_path = self._required_path(specification, "roi_file", str(name))
        table = pd.read_csv(roi_path, sep="\t")
        data = np.zeros(tuple(target_img.shape[:3]), dtype=np.int16)
        coordinates = np.indices(data.shape).reshape(3, -1).T
        world = nib.affines.apply_affine(target_img.affine, coordinates)
        for parcel_id, (_, row) in enumerate(table.iterrows(), start=1):
            center = np.asarray([row["x"], row["y"], row["z"]], dtype=float)
            radius = float(row["radius_mm"])
            inside = np.sum((world - center) ** 2, axis=1) <= radius**2
            data.reshape(-1)[inside] = parcel_id
        sphere_image = new_img_like(target_img, data)
        maps = self.cache_dir / "coordinate_spheres" / f"{name}.nii.gz"
        maps.parent.mkdir(parents=True, exist_ok=True)
        sphere_image.to_filename(maps)
        return Atlas(name=atlas.name, maps=maps, metadata=atlas.metadata)

    def resample(self, name: str, target_img: Any, output_path: str | Path) -> Atlas:
        atlas = self._atlas_for(name, target_img)
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
        atlas = self._atlas_for(name, target_img)
        atlas_image = resample_to_img(str(atlas.maps), target_img, interpolation="nearest")
        labels = np.round(atlas_image.get_fdata()).astype(np.int32)
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

    def roi_contrast_summary(
        self,
        name: str,
        contrast_images: dict[str, Any],
        target_img: Any,
        output_path: str | Path,
    ) -> Path:
        """Extract long-format parcel values from saved contrast effect images."""

        atlas = self._atlas_for(name, target_img)
        atlas_image = resample_to_img(str(atlas.maps), target_img, interpolation="nearest")
        labels = np.round(atlas_image.get_fdata()).astype(np.int32)
        metadata = self._parcel_metadata(atlas)
        parcel_ids = sorted(int(value) for value in np.unique(labels) if value > 0)
        max_parcel_id = max(parcel_ids, default=0)
        missing_ids = [parcel_id for parcel_id in parcel_ids if parcel_id not in metadata]
        network_count = sum(1 for parcel_id in parcel_ids if metadata.get(parcel_id, {}).get("network"))
        LOGGER.info(
            "Atlas contrast extraction metadata: atlas=%s labels=%d parcels=%d "
            "network_labels=%d max_roi=%d",
            name,
            len(atlas.labels),
            len(metadata),
            network_count,
            max_parcel_id,
        )
        if missing_ids:
            raise ValueError(
                f"Atlas '{name}' has no metadata for parcel IDs: {missing_ids}."
            )
        LOGGER.info("Atlas '%s' has metadata for all parcel IDs present in the image.", name)
        contrast_data = {
            contrast: np.asarray(image.get_fdata())
            for contrast, image in contrast_images.items()
        }
        rows = []
        for roi in parcel_ids:
            mask = labels == roi
            parcel = metadata.get(roi, {"parcel_label": "", "network": "", "hemisphere": ""})
            for contrast, values in contrast_data.items():
                rows.append({
                    "parcel_id": roi,
                    "parcel_label": parcel["parcel_label"],
                    "network": parcel["network"],
                    "hemisphere": parcel["hemisphere"],
                    "contrast": contrast,
                    "effect": float(np.nanmean(values[mask])),
                })
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            rows,
            columns=["parcel_id", "parcel_label", "network", "hemisphere", "contrast", "effect"],
        ).to_csv(output, sep="\t", index=False)
        return output

    @staticmethod
    def _parcel_metadata(atlas: Atlas) -> dict[int, dict[str, str]]:
        if atlas.metadata:
            return {
                parcel.parcel_id: {
                    "parcel_label": parcel.parcel_label,
                    "network": parcel.network,
                    "hemisphere": parcel.hemisphere,
                }
                for parcel in atlas.metadata
            }
        labels = list(atlas.labels)
        if labels and AtlasCache._normalize_label(labels[0]).lower() == "background":
            labels = labels[1:]
        metadata = {}
        network_names = {
            "default": "Default",
            "sommot": "Somatomotor",
            "somatomotor": "Somatomotor",
            "vis": "Visual",
            "visual": "Visual",
            "limbic": "Limbic",
            "salventattn": "Salience",
            "salience": "Salience",
            "dorsattn": "Dorsal Attention",
            "dorsalattention": "Dorsal Attention",
            "cont": "Frontoparietal",
            "frontoparietal": "Frontoparietal",
        }
        for parcel_id, raw_label in enumerate(labels, start=1):
            label = AtlasCache._normalize_label(raw_label)
            parts = label.split("_")
            hemisphere = parts[1] if len(parts) > 1 and parts[1] in {"LH", "RH"} else ""
            token = next(
                (part for part in parts if part.lower().replace("_", "") in network_names),
                "",
            )
            metadata[parcel_id] = {
                "parcel_label": label,
                "network": network_names.get(token.lower().replace("_", ""), token),
                "hemisphere": hemisphere,
            }
        return metadata

    @staticmethod
    def _normalize_label(raw_label: str | bytes) -> str:
        """Normalize atlas labels without stripping valid leading/trailing characters."""

        if isinstance(raw_label, bytes):
            return raw_label.decode("utf-8", errors="replace").strip()
        label = str(raw_label).strip()
        if len(label) >= 3 and label[:2] in {"b'", 'b"'} and label[-1] == label[1]:
            try:
                decoded = ast.literal_eval(label)
            except (SyntaxError, ValueError):
                return label
            if isinstance(decoded, bytes):
                return decoded.decode("utf-8", errors="replace").strip()
        return label

    def roi_residual_timeseries(
        self,
        name: str,
        residuals: list[Any],
        target_img: Any,
        output_path: str | Path,
    ) -> Path:
        atlas = self._atlas_for(name, target_img)
        atlas_image = resample_to_img(str(atlas.maps), target_img, interpolation="nearest")
        labels = np.round(atlas_image.get_fdata()).astype(np.int32)
        rois = sorted(int(value) for value in np.unique(labels) if value > 0)
        rows = []
        timepoint = 0
        for residual in residuals:
            values = np.asarray(residual.get_fdata())
            if values.ndim == 3:
                values = values[..., np.newaxis]
            for residual_timepoint in range(values.shape[-1]):
                row: dict[str, Any] = {"timepoint": timepoint}
                for roi in rois:
                    row[f"roi-{roi}"] = float(
                        np.nanmean(values[..., residual_timepoint][labels == roi])
                    )
                rows.append(row)
                timepoint += 1
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
        return output


__all__ = ["Atlas", "AtlasCache", "AtlasParcel"]
