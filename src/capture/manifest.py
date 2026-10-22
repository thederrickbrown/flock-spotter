"""Drive manifest: the JSON sidecar each recorded drive produces (spec §3)."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DriveManifest:
    drive_id: str
    video: Path
    gpx: Path
    time_offset_sec: float
    camera_config: str
    notes: str = ""


def load_manifest(path: Path) -> DriveManifest:
    """Load a manifest; video/gpx paths resolve relative to the manifest file."""
    path = Path(path)
    raw = json.loads(path.read_text())
    base = path.parent
    return DriveManifest(
        drive_id=raw["drive_id"],
        video=(base / raw["video"]).resolve(),
        gpx=(base / raw["gpx"]).resolve(),
        time_offset_sec=float(raw["time_offset_sec"]),
        camera_config=raw["camera_config"],
        notes=raw.get("notes", ""),
    )
