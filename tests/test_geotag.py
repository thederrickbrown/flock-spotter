"""GPX interpolation tests: position, heading-from-track, speed gating, offset."""

from pathlib import Path

import numpy as np
import pytest

from src.geo.frames import LocalFrame
from src.prep.geotag import geotag_frames, load_gpx_track
from tests.synthetic import GPX_START_EPOCH_S, ORIGIN_LAT_DEG, ORIGIN_LON_DEG, write_gpx

FRAME = LocalFrame(ORIGIN_LAT_DEG, ORIGIN_LON_DEG)


def _eastbound_gpx(path: Path, speed_mps: float = 10.0, duration_s: float = 20.0) -> None:
    """Constant-velocity drive due east, 1 Hz fixes."""
    times = GPX_START_EPOCH_S + np.arange(0.0, duration_s + 1.0)
    east = speed_mps * (times - GPX_START_EPOCH_S)
    north = np.zeros_like(east)
    lats, lons = FRAME.to_latlon(east, north)
    write_gpx(path, times, lats, lons)


def test_interpolates_position_heading_speed(tmp_path: Path) -> None:
    gpx_path = tmp_path / "east.gpx"
    _eastbound_gpx(gpx_path, speed_mps=10.0, duration_s=20.0)

    frames = geotag_frames(gpx_path, n_frames=40, fps=2.0, time_offset_sec=0.0, min_speed_mps=3.0)
    assert len(frames) == 40  # all within GPX range, all fast enough

    # Frame 6 (1-based) -> t = 2.5s -> 25m east of start.
    f = frames[5]
    assert f.frame_id == "000006"
    east, north = FRAME.to_enu(f.lat, f.lon)
    assert float(east) == pytest.approx(25.0, abs=0.01)
    assert float(north) == pytest.approx(0.0, abs=0.01)
    assert f.heading_deg == pytest.approx(90.0, abs=0.01)  # due east, from GPX not compass
    assert f.speed_mps == pytest.approx(10.0, abs=0.01)


def test_time_offset_shifts_positions(tmp_path: Path) -> None:
    gpx_path = tmp_path / "east.gpx"
    _eastbound_gpx(gpx_path, speed_mps=10.0, duration_s=20.0)

    frames = geotag_frames(gpx_path, n_frames=10, fps=2.0, time_offset_sec=2.0, min_speed_mps=3.0)
    # Frame 1 -> t = 0 + 2.0s offset -> 20m east, not 0.
    east, _ = FRAME.to_enu(frames[0].lat, frames[0].lon)
    assert float(east) == pytest.approx(20.0, abs=0.01)


def test_drops_frames_outside_gpx_range(tmp_path: Path) -> None:
    gpx_path = tmp_path / "east.gpx"
    _eastbound_gpx(gpx_path, duration_s=10.0)
    # 40 frames at 2fps spans 19.5s but the GPX ends at 10s.
    frames = geotag_frames(gpx_path, n_frames=40, fps=2.0, time_offset_sec=0.0, min_speed_mps=3.0)
    assert len(frames) == 21  # t = 0.0 .. 10.0 inclusive
    assert frames[-1].frame_id == "000021"


def test_drops_slow_frames(tmp_path: Path) -> None:
    """A stop in the middle of the drive must drop those frames (spec §4)."""
    times = GPX_START_EPOCH_S + np.arange(0.0, 16.0)
    east = np.concatenate(
        [
            10.0 * np.arange(0.0, 6.0),  # 0-5s: 10 m/s
            50.0 + 0.5 * np.arange(1.0, 6.0),  # 5-10s: 0.5 m/s (stopped at a light)
            52.5 + 10.0 * np.arange(1.0, 6.0),  # 10-15s: 10 m/s again
        ]
    )
    lats, lons = FRAME.to_latlon(east, np.zeros_like(east))
    gpx_path = tmp_path / "stop.gpx"
    write_gpx(gpx_path, times, lats, lons)

    frames = geotag_frames(gpx_path, n_frames=30, fps=2.0, time_offset_sec=0.0, min_speed_mps=3.0)
    dropped_window = [f for f in frames if 5.0 <= f.timestamp - GPX_START_EPOCH_S < 10.0]
    assert dropped_window == []
    assert any(f.timestamp - GPX_START_EPOCH_S < 5.0 for f in frames)
    assert any(f.timestamp - GPX_START_EPOCH_S >= 10.0 for f in frames)


def test_load_gpx_track_requires_timestamps(tmp_path: Path) -> None:
    gpx_path = tmp_path / "min.gpx"
    gpx_path.write_text('<gpx version="1.1"><trk><trkseg></trkseg></trk></gpx>')
    with pytest.raises(ValueError, match="at least 2"):
        load_gpx_track(gpx_path)
