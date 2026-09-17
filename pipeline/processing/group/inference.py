from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm import threshold_stats_img
from scipy import ndimage


@dataclass(frozen=True)
class InferenceResult:
    stat_map: Path
    thresholded_map: Path
    significance_mask: Path
    cluster_table: Path
    metadata: dict[str, Any]


class InferenceMethod(ABC):
    name: str

    @abstractmethod
    def run(self, stat_map: Path, output_dir: Path, configuration: dict[str, Any]) -> InferenceResult:
        raise NotImplementedError


def _cluster_table(image: nib.Nifti1Image, mask: np.ndarray) -> pd.DataFrame:
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    data = np.asarray(image.get_fdata(), dtype=float)
    rows = []
    for cluster_index in range(1, count + 1):
        component = labels == cluster_index
        peak_index = np.unravel_index(int(np.argmax(np.where(component, np.abs(data), -np.inf))), data.shape)
        coordinate = nib.affines.apply_affine(image.affine, peak_index)
        rows.append({
            "cluster_index": cluster_index,
            "peak_x": float(coordinate[0]),
            "peak_y": float(coordinate[1]),
            "peak_z": float(coordinate[2]),
            "peak_statistic": float(data[peak_index]),
            "cluster_size_voxels": int(component.sum()),
        })
    return pd.DataFrame(rows, columns=[
        "cluster_index", "peak_x", "peak_y", "peak_z", "peak_statistic", "cluster_size_voxels"
    ])


class FDRInference(InferenceMethod):
    name = "fdr"

    def run(self, stat_map: Path, output_dir: Path, configuration: dict[str, Any]) -> InferenceResult:
        alpha = float(configuration.get("alpha", 0.05))
        image = nib.load(str(stat_map))
        thresholded_image, computed_threshold = threshold_stats_img(
            image,
            alpha=alpha,
            height_control="fdr",
            cluster_threshold=0,
            two_sided=True,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        thresholded_path = output_dir / "thresholded_stat_map.nii.gz"
        mask_path = output_dir / "significance_mask.nii.gz"
        cluster_table_path = output_dir / "cluster_table.tsv"
        thresholded_image.to_filename(thresholded_path)
        mask = np.isfinite(thresholded_image.get_fdata()) & (thresholded_image.get_fdata() != 0)
        nib.Nifti1Image(mask.astype(np.uint8), image.affine, image.header).to_filename(mask_path)
        table = _cluster_table(image, mask)
        table.to_csv(cluster_table_path, sep="\t", index=False)
        metadata = {
            "method": self.name,
            "method_label": "FDR",
            "alpha": alpha,
            "computed_threshold": float(computed_threshold),
            "threshold_scale": "z",
            "two_sided": True,
            "cluster_definition": "connected components in significant voxel mask",
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return InferenceResult(stat_map, thresholded_path, mask_path, cluster_table_path, metadata)


INFERENCE_METHODS = {FDRInference.name: FDRInference}


def create_inference_method(configuration: dict[str, Any]) -> InferenceMethod:
    method = str(configuration.get("method", "")).strip().lower()
    inference_class = INFERENCE_METHODS.get(method)
    if inference_class is None:
        supported = ", ".join(sorted(INFERENCE_METHODS))
        raise ValueError(f"Unsupported group inference method '{method}'. Supported methods: {supported}.")
    return inference_class()