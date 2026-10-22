"""Hand-labeled detection input for M1 (stand-in for the M2 detector+tracker).

One CSV per drive at data/labels/<drive_id>.csv:

    frame_id,track_id,class,x_min,y_min,x_max,y_max
    000042,cam_A,alpr_flock,1412,388,1470,502

Pixel coordinates, origin top-left. track_id is hand-assigned per physical
camera. Bearing math uses the horizontal box center (x_min + x_max) / 2.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ["frame_id", "track_id", "class", "x_min", "y_min", "x_max", "y_max"]


@dataclass(frozen=True)
class Detection:
    frame_id: str
    track_id: str
    class_name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def center_x_px(self) -> float:
        return (self.x_min + self.x_max) / 2.0


def load_labels(path: Path) -> list[Detection]:
    """Read a hand-label CSV, sorted by (track_id, frame_id)."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        detections = [
            Detection(
                frame_id=row["frame_id"],
                track_id=row["track_id"],
                class_name=row["class"],
                x_min=float(row["x_min"]),
                y_min=float(row["y_min"]),
                x_max=float(row["x_max"]),
                y_max=float(row["y_max"]),
            )
            for row in reader
        ]
    detections.sort(key=lambda d: (d.track_id, d.frame_id))
    return detections


def group_by_track(detections: list[Detection]) -> dict[str, list[Detection]]:
    """Group detections by track_id, preserving frame order within each track."""
    tracks: dict[str, list[Detection]] = {}
    for det in detections:
        tracks.setdefault(det.track_id, []).append(det)
    return tracks
