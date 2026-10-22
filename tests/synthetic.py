"""Synthetic test fixture: a virtual world with known camera positions.

Places virtual ALPR cameras at known ENU coordinates, drives a constant-velocity
vehicle past them, and back-computes the bounding-box pixel column each camera
would occupy in each frame (the exact inverse of src.geo.bearing's pinhole
model). Running the pipeline forward over these detections must recover the
camera positions to within 2m — that's the M1 correctness gate.

Everything is deterministic: fixed origin, fixed epoch, exact linear motion
(so the pipeline's linear GPX interpolation is exact, isolating geometry error
from interpolation error).
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import gpxpy.gpx
import numpy as np

from src.config import CameraConfig
from src.geo.bearing import bearing_to_target, focal_length_px
from src.geo.frames import LocalFrame
from src.geo.triangulate import Ray

ORIGIN_LAT_DEG = 32.80
ORIGIN_LON_DEG = -117.20
GPX_START_EPOCH_S = 1_700_000_000.0

# Deliberately non-zero yaw offset so the tests exercise the mount-rotation term.
SYNTH_CAMERA = CameraConfig(
    name="synthetic-cam",
    fov_horizontal_deg=68.0,
    yaw_offset_deg=2.0,
    image_width=1920,
    image_height=1080,
)


def wrap180(deg: float) -> float:
    """Wrap an angle to (-180, 180]."""
    return -((-deg + 180.0) % 360.0 - 180.0)


def true_bearing_deg(from_e: float, from_n: float, to_e: float, to_n: float) -> float:
    """Ground-truth bearing between ENU points, degrees clockwise from north."""
    return math.degrees(math.atan2(to_e - from_e, to_n - from_n)) % 360.0


@dataclass(frozen=True)
class StraightDrive:
    """Constant-velocity drive due north along east=0, GPX fixes at 1 Hz."""

    frame: LocalFrame
    speed_mps: float
    duration_s: float
    heading_deg: float = 0.0

    @property
    def gpx_times(self) -> np.ndarray:
        return GPX_START_EPOCH_S + np.arange(0.0, self.duration_s + 1.0, 1.0)

    def position_at(self, t_epoch: float) -> tuple[float, float]:
        """Exact vehicle ENU position at an epoch time (east, north)."""
        return 0.0, self.speed_mps * (t_epoch - GPX_START_EPOCH_S)

    def path_enu(self) -> np.ndarray:
        norths = self.speed_mps * (self.gpx_times - GPX_START_EPOCH_S)
        return np.column_stack([np.zeros_like(norths), norths])

    def gpx_latlon(self) -> tuple[np.ndarray, np.ndarray]:
        path = self.path_enu()
        return self.frame.to_latlon(path[:, 0], path[:, 1])


def make_drive(duration_s: float = 40.0, speed_mps: float = 13.0) -> StraightDrive:
    return StraightDrive(
        frame=LocalFrame(ORIGIN_LAT_DEG, ORIGIN_LON_DEG),
        speed_mps=speed_mps,
        duration_s=duration_s,
    )


def project_to_pixel(
    camera_enu: tuple[float, float],
    vehicle_e: float,
    vehicle_n: float,
    heading_deg: float,
    cam: CameraConfig,
) -> float | None:
    """Inverse of bearing_to_target: pixel column where the camera would appear.

    Returns None when the camera is outside the horizontal FOV.
    """
    bearing = true_bearing_deg(vehicle_e, vehicle_n, camera_enu[0], camera_enu[1])
    rel_deg = wrap180(bearing - heading_deg - cam.yaw_offset_deg)
    if abs(rel_deg) >= cam.fov_horizontal_deg / 2.0:
        return None
    f_px = focal_length_px(cam.image_width, cam.fov_horizontal_deg)
    return cam.image_width / 2.0 + f_px * math.tan(math.radians(rel_deg))


def synth_detections(
    drive: StraightDrive,
    camera_enu: tuple[float, float],
    cam: CameraConfig = SYNTH_CAMERA,
    fps: float = 2.0,
) -> list[tuple[int, float, float]]:
    """Back-compute detections: (frame_number, epoch_time, center_x_px).

    frame_number is 1-based to match ffmpeg's %06d output; frame k's timestamp
    is (k - 1) / fps after GPX start.
    """
    detections = []
    n_frames = int(drive.duration_s * fps)
    for k in range(1, n_frames + 1):
        t = GPX_START_EPOCH_S + (k - 1) / fps
        east, north = drive.position_at(t)
        u = project_to_pixel(camera_enu, east, north, drive.heading_deg, cam)
        if u is not None:
            detections.append((k, t, u))
    return detections


def write_gpx(path: Path, times_epoch: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> None:
    """Write a minimal single-track GPX file with timestamped points."""
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    segment = gpxpy.gpx.GPXTrackSegment()
    for t, lat, lon in zip(times_epoch, lats, lons, strict=True):
        segment.points.append(
            gpxpy.gpx.GPXTrackPoint(
                latitude=float(lat),
                longitude=float(lon),
                time=datetime.fromtimestamp(float(t), tz=UTC),
            )
        )
    track.segments.append(segment)
    gpx.tracks.append(track)
    Path(path).write_text(gpx.to_xml())


def rays_from_detections(
    drive: StraightDrive,
    detections: list[tuple[int, float, float]],
    cam: CameraConfig = SYNTH_CAMERA,
    time_sync_error_s: float = 0.0,
    round_pixels: bool = False,
) -> list[Ray]:
    """Run the forward bearing math over synthetic detections.

    time_sync_error_s simulates a video-vs-GPX clock offset (spec §3): ray
    origins come from the vehicle position at t + error while the pixel came
    from the true position at t.
    """
    rays = []
    for _, t, u in detections:
        east, north = drive.position_at(t + time_sync_error_s)
        u_obs = float(round(u)) if round_pixels else u
        bearing = bearing_to_target(
            u_obs, cam.image_width, cam.fov_horizontal_deg, drive.heading_deg, cam.yaw_offset_deg
        )
        rays.append(Ray(origin_east_m=east, origin_north_m=north, bearing_deg=bearing))
    return rays
