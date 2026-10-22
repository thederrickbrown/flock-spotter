"""Geotag extracted frames from a GPX track (spec §4).

Frame k (1-based, matching ffmpeg's %06d) gets timestamp
    t_k = gpx_track_start + (k - 1) / fps + time_offset_sec

Position is linearly interpolated between GPX fixes. Heading is the bearing
between the bracketing GPX points — never the device compass (magnetometers
are unreliable inside a car). Frames slower than min_speed_mps are dropped:
no triangulation baseline, and GPX heading is garbage at low speed.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import gpxpy
import numpy as np

from src.geo.frames import LocalFrame


@dataclass(frozen=True)
class GeotaggedFrame:
    frame_id: str  # zero-padded, matches the JPEG basename (e.g. "000042")
    timestamp: float  # epoch seconds
    lat: float
    lon: float
    heading_deg: float  # degrees clockwise from true north
    speed_mps: float


def load_gpx_track(gpx_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a GPX file -> (epoch_times, lats, lons), sorted by time."""
    with open(gpx_path) as fh:
        gpx = gpxpy.parse(fh)
    rows = [
        (p.time.timestamp(), p.latitude, p.longitude)
        for p in gpx.walk(only_points=True)
        if p.time is not None
    ]
    if len(rows) < 2:
        raise ValueError(f"{gpx_path}: need at least 2 timestamped GPX points")
    rows.sort(key=lambda r: r[0])
    times, lats, lons = (np.array(col, dtype=float) for col in zip(*rows, strict=True))
    return times, lats, lons


def geotag_frames(
    gpx_path: Path,
    n_frames: int,
    fps: float,
    time_offset_sec: float,
    min_speed_mps: float,
) -> list[GeotaggedFrame]:
    """Interpolate lat/lon/heading/speed for frames 1..n_frames.

    Frames outside the GPX time range or below min_speed_mps are dropped.
    """
    times, lats, lons = load_gpx_track(gpx_path)
    frame = LocalFrame.centered_on(lats, lons)
    east, north = frame.to_enu(lats, lons)

    # Per-GPX-segment heading and speed (segment i spans fix i -> i+1).
    d_east, d_north, d_t = np.diff(east), np.diff(north), np.diff(times)
    seg_heading = np.degrees(np.arctan2(d_east, d_north)) % 360.0
    seg_speed = np.hypot(d_east, d_north) / d_t

    result: list[GeotaggedFrame] = []
    for k in range(1, n_frames + 1):
        t = times[0] + (k - 1) / fps + time_offset_sec
        if t < times[0] or t > times[-1]:
            continue
        seg = min(int(np.searchsorted(times, t, side="right")) - 1, len(times) - 2)
        seg = max(seg, 0)
        if seg_speed[seg] < min_speed_mps:
            continue
        result.append(
            GeotaggedFrame(
                frame_id=f"{k:06d}",
                timestamp=t,
                lat=float(np.interp(t, times, lats)),
                lon=float(np.interp(t, times, lons)),
                heading_deg=float(seg_heading[seg]),
                speed_mps=float(seg_speed[seg]),
            )
        )
    return result


def write_frames_csv(frames: list[GeotaggedFrame], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_id", "timestamp", "lat", "lon", "heading_deg", "speed_mps"])
        for f in frames:
            writer.writerow(
                [
                    f.frame_id,
                    f"{f.timestamp:.3f}",
                    f"{f.lat:.7f}",
                    f"{f.lon:.7f}",
                    f"{f.heading_deg:.2f}",
                    f"{f.speed_mps:.2f}",
                ]
            )
