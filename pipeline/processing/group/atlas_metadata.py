from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..analysis.analyses.atlas import AtlasCache


_NETWORK_NAMES = {
    "default": "Default",
    "sommot": "Somatomotor",
    "somatomotor": "Somatomotor",
    "visual": "Visual",
    "limbic": "Limbic",
    "salventattn": "Salience",
    "salience": "Salience",
    "dorsattn": "Dorsal Attention",
    "dorsalattention": "Dorsal Attention",
    "cont": "Frontoparietal",
    "frontoparietal": "Frontoparietal",
}


@dataclass(frozen=True)
class AtlasParcel:
    parcel_id: int
    parcel_label: str
    network: str
    hemisphere: str


class AtlasMetadata:
    """Network-aware metadata parsed from atlas parcel labels."""

    def __init__(self, parcels: Iterable[AtlasParcel]) -> None:
        self.parcels = tuple(parcels)

    @classmethod
    def from_cache(cls, atlas_cache: AtlasCache, atlas_name: str) -> "AtlasMetadata":
        atlas = atlas_cache.get(atlas_name)
        labels = tuple(atlas.labels)
        if not labels:
            labels = cls._cached_schaefer_labels(atlas_cache, atlas_name)
        parcels = []
        for index, raw_label in enumerate(labels, start=1):
            label = str(raw_label).strip().strip("b'").strip('"')
            parts = label.split("_")
            hemisphere = parts[1] if len(parts) > 1 and parts[1] in {"LH", "RH"} else ""
            network_token = next(
                (part for part in parts if part.lower().replace("_", "") in _NETWORK_NAMES),
                parts[2] if len(parts) > 2 else "Unknown",
            )
            network = _NETWORK_NAMES.get(
                network_token.lower().replace("_", ""), network_token
            )
            parcels.append(AtlasParcel(index, label, network, hemisphere))
        return cls(parcels)

    @staticmethod
    def _cached_schaefer_labels(atlas_cache: AtlasCache, atlas_name: str) -> tuple[str, ...]:
        """Read the label sidecar used by cached Schaefer atlases, if present."""

        candidates = []
        atlas_token = str(atlas_name).lower().replace("_", "")
        for path in atlas_cache.cache_dir.rglob("*.txt"):
            name = path.name.lower().replace("_", "")
            if "schaefer" in name and atlas_token.replace("schaefer", "") in name:
                candidates.append(path)
        if not candidates:
            return ()
        labels = []
        for line in candidates[0].read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.replace("\t", " ").split()
            if len(fields) >= 2 and fields[0].isdigit():
                labels.append(" ".join(fields[1:]))
        return tuple(labels)

    def get_parcels(self, *, network: str | None = None) -> tuple[AtlasParcel, ...]:
        if network is None:
            return self.parcels
        requested = _NETWORK_NAMES.get(network.lower().replace(" ", ""), network)
        return tuple(parcel for parcel in self.parcels if parcel.network == requested)
