# flock-spotter

Offline pipeline: dashcam video + GPS track → geolocated ALPR camera positions.
Full design in [FLOCK_SPOTTER_SPEC.md](FLOCK_SPOTTER_SPEC.md).

**Current milestone: M1 — geometry harness, no ML.** Hand-labeled bounding boxes
stand in for the detector; the pipeline proves the bearing + triangulation math
against a synthetic fixture with known camera positions (recovery within 2m).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy gpxpy pyyaml pytest ruff
```

Requires `ffmpeg` on PATH (frame extraction).

## Run a drive

```bash
.venv/bin/python -m src.cli run path/to/manifest.json \
    --labels data/labels/<drive_id>.csv
```

- **Manifest** — JSON sidecar per drive (spec §3): `drive_id`, `video`, `gpx`
  (paths relative to the manifest file), `time_offset_sec`, `camera_config`.
- **Labels** — one CSV per drive, pixel coords, origin top-left, `track_id`
  hand-assigned per physical camera:

  ```csv
  frame_id,track_id,class,x_min,y_min,x_max,y_max
  000042,cam_A,alpr_flock,1412,388,1470,502
  ```

- **Output** — `data/output/candidates.geojson` plus
  `<drive_id>-rejections.log` alongside it (every rejected track with its
  structured reason: `too_few_rays`, `low_angular_spread`, `ill_conditioned`,
  `behind_path`, `high_residual`, `far_from_road`).

Frames land in `data/frames/<drive_id>/` with a `frames.csv`
(`frame_id,timestamp,lat,lon,heading_deg,speed_mps`).

## Conventions

- Bearings are **degrees clockwise from true north**, everywhere.
- Geometry runs in a local ENU frame in meters (`src/geo/frames.py`),
  equirectangular around the drive's centroid.
- Frame `k` (1-based, ffmpeg `%06d`) has timestamp
  `gpx_start + (k-1)/fps + time_offset_sec`. Measure `time_offset_sec` per
  drive — a 1s clock error at 40 mph silently shifts every camera ~18m
  (demonstrated in `tests/test_rejections.py::test_time_sync_error_is_silent_and_large`).
- Every tunable lives in `config/pipeline.yaml`; per-mount calibration
  (FOV, yaw offset) in `config/cameras.yaml`. Measure the yaw offset (spec §6),
  don't assume zero.

## Tests and lint

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
```

`tests/synthetic.py` is the primary correctness gate: virtual cameras at known
coordinates, back-computed pixel positions, forward pipeline must recover them
within 2m — plus rejection cases for every spec §7 criterion.
