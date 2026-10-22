"""End-to-end CLI test on a fully synthetic drive.

Builds a real drive on disk — testsrc video (ffmpeg), GPX file, hand-label CSV,
manifest — runs the CLI, and checks candidates.geojson recovers both virtual
cameras within 2m and that the undersized track lands in the rejection log.
"""

import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.cli import main
from src.prep.extract_frames import extract_frames
from tests.synthetic import SYNTH_CAMERA, make_drive, synth_detections, write_gpx

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def _make_test_video(path: Path, duration_s: float) -> None:
    """Tiny synthetic video; only frame count/timing matter, never the pixels."""
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_s}:size=160x120:rate=6",
            str(path),
        ],
        check=True,
    )


def test_extract_frames_counts(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_test_video(video, duration_s=3.0)
    jpegs = extract_frames(video, tmp_path / "frames", fps=2.0)
    assert len(jpegs) == 6
    assert jpegs[0].name == "000001.jpg"


def test_cli_end_to_end(tmp_path: Path) -> None:
    drive = make_drive(duration_s=40.0, speed_mps=13.0)
    truth = {"cam_A": (12.0, 260.0), "cam_B": (-15.0, 380.0)}

    # Drive artifacts on disk: video, GPX, manifest, camera config, labels.
    _make_test_video(tmp_path / "drive.mp4", duration_s=drive.duration_s)
    lats, lons = drive.gpx_latlon()
    write_gpx(tmp_path / "drive.gpx", drive.gpx_times, lats, lons)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "drive_id": "synthetic-01",
                "video": "drive.mp4",
                "gpx": "drive.gpx",
                "time_offset_sec": 0.0,
                "camera_config": SYNTH_CAMERA.name,
                "notes": "synthetic end-to-end fixture",
            }
        )
    )
    (tmp_path / "cameras.yaml").write_text(
        f"{SYNTH_CAMERA.name}:\n"
        f"  fov_horizontal_deg: {SYNTH_CAMERA.fov_horizontal_deg}\n"
        f"  yaw_offset_deg: {SYNTH_CAMERA.yaw_offset_deg}\n"
        f"  image_width: {SYNTH_CAMERA.image_width}\n"
        f"  image_height: {SYNTH_CAMERA.image_height}\n"
    )

    labels_path = tmp_path / "labels.csv"
    with open(labels_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_id", "track_id", "class", "x_min", "y_min", "x_max", "y_max"])
        for track_id, camera in truth.items():
            for k, _, u in synth_detections(drive, camera):
                writer.writerow(
                    [f"{k:06d}", track_id, "alpr_flock", round(u) - 25, 380, round(u) + 25, 500]
                )
        # One deliberately undersized track: must land in the rejection log.
        for k, _, u in synth_detections(drive, (20.0, 100.0))[:2]:
            writer.writerow(
                [f"{k:06d}", "cam_short", "alpr_flock", round(u) - 25, 380, round(u) + 25, 500]
            )

    output = tmp_path / "out" / "candidates.geojson"
    rc = main(
        [
            "run",
            str(tmp_path / "manifest.json"),
            "--labels",
            str(labels_path),
            "--cameras",
            str(tmp_path / "cameras.yaml"),
            "--output",
            str(output),
            "--frames-dir",
            str(tmp_path / "frames"),
        ]
    )
    assert rc == 0

    geojson = json.loads(output.read_text())
    features = {f["properties"]["track_ids"][0]: f for f in geojson["features"]}
    assert set(features) == set(truth)
    for track_id, camera in truth.items():
        lon, lat = features[track_id]["geometry"]["coordinates"]
        east, north = drive.frame.to_enu(lat, lon)
        error_m = float(np.hypot(float(east) - camera[0], float(north) - camera[1]))
        assert error_m < 2.0, f"{track_id} recovered {error_m:.2f}m from truth"
        assert features[track_id]["properties"]["source_drives"] == ["synthetic-01"]

    rejection_log = output.parent / "synthetic-01-rejections.log"
    assert "too_few_rays" in rejection_log.read_text()
    assert "cam_short" in rejection_log.read_text()

    frames_csv = tmp_path / "frames" / "frames.csv"
    assert frames_csv.exists()
    header = frames_csv.read_text().splitlines()[0]
    assert header == "frame_id,timestamp,lat,lon,heading_deg,speed_mps"
