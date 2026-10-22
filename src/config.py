"""Typed config loaders for cameras.yaml, regions.yaml, and pipeline.yaml.

Every tunable in the pipeline lives in PipelineConfig — code takes thresholds
from here, never from literals.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CameraConfig:
    """Per-mount calibration (spec §6). Keyed by the manifest's camera_config."""

    name: str
    fov_horizontal_deg: float
    yaw_offset_deg: float
    image_width: int
    image_height: int


@dataclass(frozen=True)
class Region:
    """Named bounding box for Overpass queries (south, west, north, east)."""

    name: str
    south: float
    west: float
    north: float
    east: float


@dataclass(frozen=True)
class PipelineConfig:
    """All pipeline tunables. Defaults mirror config/pipeline.yaml."""

    fps: float = 2.0
    min_speed_mps: float = 3.0
    min_rays: int = 3
    min_angular_spread_deg: float = 15.0
    max_condition_number: float = 1.0e4
    max_residual_m: float = 8.0
    max_road_distance_m: float = 60.0
    cluster_eps_m: float = 15.0
    cluster_min_samples: int = 1


def load_cameras(path: Path) -> dict[str, CameraConfig]:
    raw = yaml.safe_load(Path(path).read_text())
    return {name: CameraConfig(name=name, **fields) for name, fields in raw.items()}


def load_regions(path: Path) -> dict[str, Region]:
    raw = yaml.safe_load(Path(path).read_text())
    return {name: Region(name=name, **fields) for name, fields in raw.items()}


def load_pipeline(path: Path | None = None) -> PipelineConfig:
    """Load pipeline tunables; defaults when no file is given."""
    if path is None:
        return PipelineConfig()
    raw = yaml.safe_load(Path(path).read_text())
    # Coerce by declared field type: YAML 1.1 reads e.g. "1.0e4" as a string,
    # and a threshold that is silently a string breaks every comparison.
    types = {f.name: f.type for f in fields(PipelineConfig)}
    unknown = set(raw) - set(types)
    if unknown:
        raise ValueError(f"{path}: unknown pipeline settings {sorted(unknown)}")
    coerced = {k: (int(v) if types[k] in (int, "int") else float(v)) for k, v in raw.items()}
    return PipelineConfig(**coerced)
