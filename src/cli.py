"""One command: manifest -> frames -> geotag -> hand labels -> bearings ->
triangulate -> cluster -> candidates.geojson.

    python -m src.cli run drive-manifest.json --labels data/labels/<drive_id>.csv

Rejected tracks are written to a rejection log next to the GeoJSON output —
per spec §7 the rejection log is a debugging tool, not noise.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.capture.manifest import DriveManifest, load_manifest
from src.config import CameraConfig, PipelineConfig, load_cameras, load_pipeline
from src.geo.bearing import bearing_to_target
from src.geo.cluster import Cluster, cluster_points
from src.geo.frames import LocalFrame
from src.geo.triangulate import Ray, TriangulationResult, triangulate_track
from src.labels import group_by_track, load_labels
from src.prep.extract_frames import extract_frames
from src.prep.geotag import GeotaggedFrame, geotag_frames, load_gpx_track, write_frames_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flock-spotter")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="process one drive end to end")
    run_p.add_argument("manifest", type=Path, help="drive manifest JSON (spec §3)")
    run_p.add_argument("--labels", type=Path, required=True, help="hand-label CSV for this drive")
    run_p.add_argument("--cameras", type=Path, default=Path("config/cameras.yaml"))
    run_p.add_argument(
        "--pipeline",
        type=Path,
        default=None,
        help="pipeline tunables YAML (default: config/pipeline.yaml if present)",
    )
    run_p.add_argument("--output", type=Path, default=Path("data/output/candidates.geojson"))
    run_p.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="frame output dir (default: data/frames/<drive_id>)",
    )
    run_p.add_argument(
        "--skip-extract",
        action="store_true",
        help="reuse already-extracted JPEGs in the frames dir",
    )
    return parser


def _resolve_pipeline_config(path: Path | None) -> PipelineConfig:
    if path is not None:
        return load_pipeline(path)
    default = Path("config/pipeline.yaml")
    return load_pipeline(default) if default.exists() else load_pipeline(None)


def _build_tracks(
    labels_path: Path,
    frame_lookup: dict[str, GeotaggedFrame],
    local: LocalFrame,
    cam: CameraConfig,
) -> tuple[dict[str, list[Ray]], int]:
    """Hand labels -> per-track bearing rays. Returns (tracks, n_skipped)."""
    skipped = 0
    tracks: dict[str, list[Ray]] = {}
    for track_id, dets in group_by_track(load_labels(labels_path)).items():
        rays = []
        for det in dets:
            frame = frame_lookup.get(det.frame_id)
            if frame is None:  # frame dropped by geotag (low speed / outside GPX range)
                skipped += 1
                continue
            east, north = local.to_enu(frame.lat, frame.lon)
            bearing = bearing_to_target(
                det.center_x_px,
                cam.image_width,
                cam.fov_horizontal_deg,
                frame.heading_deg,
                cam.yaw_offset_deg,
            )
            rays.append(Ray(float(east), float(north), bearing))
        tracks[track_id] = rays
    return tracks, skipped


def _clusters_to_geojson(
    clusters: list[Cluster],
    by_track: dict[str, TriangulationResult],
    local: LocalFrame,
    manifest: DriveManifest,
) -> dict:
    features = []
    for cluster in clusters:
        lat, lon = local.to_latlon(cluster.centroid_enu[0], cluster.centroid_enu[1])
        residuals = [by_track[tid].mean_residual_m for tid in cluster.track_ids]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": {
                    "status": "candidate",  # confirmed/new/stale comes with M3 reconciliation
                    "cluster_size": cluster.size,
                    "n_tracks": cluster.size,
                    "track_ids": list(cluster.track_ids),
                    "mean_residual_m": round(float(np.mean(residuals)), 2),
                    "source_drives": [manifest.drive_id],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    cameras = load_cameras(args.cameras)
    if manifest.camera_config not in cameras:
        print(
            f"error: camera_config '{manifest.camera_config}' not in {args.cameras} "
            f"(have: {', '.join(cameras)})",
            file=sys.stderr,
        )
        return 1
    cam = cameras[manifest.camera_config]
    config = _resolve_pipeline_config(args.pipeline)

    # 1. Extract frames.
    frames_dir = args.frames_dir or Path("data/frames") / manifest.drive_id
    if args.skip_extract:
        jpegs = sorted(Path(frames_dir).glob("*.jpg"))
    else:
        jpegs = extract_frames(manifest.video, frames_dir, config.fps)
    print(f"[{manifest.drive_id}] {len(jpegs)} frames in {frames_dir}")

    # 2. Geotag from GPX.
    geotagged = geotag_frames(
        manifest.gpx, len(jpegs), config.fps, manifest.time_offset_sec, config.min_speed_mps
    )
    frames_csv = Path(frames_dir) / "frames.csv"
    write_frames_csv(geotagged, frames_csv)
    print(
        f"[{manifest.drive_id}] geotagged {len(geotagged)}/{len(jpegs)} frames -> {frames_csv} "
        f"(dropped: outside GPX range or slower than {config.min_speed_mps} m/s)"
    )

    # Shared local ENU frame for the whole drive.
    _, lats, lons = load_gpx_track(manifest.gpx)
    local = LocalFrame.centered_on(lats, lons)
    path_east, path_north = local.to_enu(lats, lons)
    path_enu = np.column_stack([path_east, path_north])

    # 3. Hand labels -> bearings -> per-track triangulation.
    frame_lookup = {g.frame_id: g for g in geotagged}
    tracks, skipped = _build_tracks(args.labels, frame_lookup, local, cam)
    if skipped:
        print(f"[{manifest.drive_id}] {skipped} labeled detections on dropped frames, ignored")
    results = [
        triangulate_track(track_id, rays, config, path_enu=path_enu)
        for track_id, rays in tracks.items()
    ]
    accepted = [r for r in results if r.accepted]
    rejected = [r for r in results if not r.accepted]

    # 4. Rejection log — a debugging tool, not noise (spec §7).
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejection_log = output_path.parent / f"{manifest.drive_id}-rejections.log"
    rejection_log.write_text("".join(f"[{r.rejection_reason}] {r.detail}\n" for r in rejected))
    for r in rejected:
        print(f"[{manifest.drive_id}] REJECTED [{r.rejection_reason}] {r.detail}", file=sys.stderr)
    print(
        f"[{manifest.drive_id}] triangulated {len(accepted)}/{len(results)} tracks "
        f"({len(rejected)} rejected -> {rejection_log})"
    )

    # 5. Cluster accepted tracks and write GeoJSON.
    clusters: list[Cluster] = []
    if accepted:
        points = np.array([r.point_enu for r in accepted])
        track_ids = [r.track_id for r in accepted]
        clusters = cluster_points(
            points, track_ids, config.cluster_eps_m, config.cluster_min_samples
        )
    by_track = {r.track_id: r for r in accepted}
    geojson = _clusters_to_geojson(clusters, by_track, local, manifest)
    output_path.write_text(json.dumps(geojson, indent=2) + "\n")
    print(f"[{manifest.drive_id}] {len(clusters)} candidate(s) -> {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run(args)
    return 2  # unreachable: argparse enforces the subcommand


if __name__ == "__main__":
    raise SystemExit(main())
