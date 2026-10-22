"""Every spec §7 rejection criterion must fire, with the right structured reason.

Also documents the one failure the geometry CANNOT catch: a time-sync error on
a straight path shifts the answer silently (spec §3) — which is exactly why the
capture procedure demands measuring the clock offset.
"""

import numpy as np

from src.config import PipelineConfig
from src.geo.triangulate import (
    REJECT_BEHIND_PATH,
    REJECT_FAR_FROM_ROAD,
    REJECT_HIGH_RESIDUAL,
    REJECT_ILL_CONDITIONED,
    REJECT_LOW_ANGULAR_SPREAD,
    REJECT_TOO_FEW_RAYS,
    Ray,
    triangulate_track,
)
from tests.synthetic import make_drive, rays_from_detections, synth_detections

CONFIG = PipelineConfig()


def test_rejects_two_detections() -> None:
    drive = make_drive()
    rays = rays_from_detections(drive, synth_detections(drive, (12.0, 260.0))[:2])
    result = triangulate_track("short", rays, CONFIG)
    assert not result.accepted
    assert result.rejection_reason == REJECT_TOO_FEW_RAYS


def test_rejects_camera_dead_ahead() -> None:
    """Driving straight at a camera: rays never diverge -> low angular spread.

    Spec §7 calls this the main failure mode.
    """
    drive = make_drive()
    camera = (0.0, 600.0)  # exactly on the driving line, never reached
    rays = rays_from_detections(drive, synth_detections(drive, camera))
    assert len(rays) >= CONFIG.min_rays
    result = triangulate_track("dead_ahead", rays, CONFIG, path_enu=drive.path_enu())
    assert not result.accepted
    assert result.rejection_reason == REJECT_LOW_ANGULAR_SPREAD


def test_rejects_ill_conditioned_geometry() -> None:
    """Antiparallel rays on one line pass the first/last spread check (180°)
    but make A singular — the condition-number guard must catch them."""
    rays = [Ray(0.0, 0.0, 0.0), Ray(0.0, 50.0, 0.0), Ray(0.0, 100.0, 180.0)]
    result = triangulate_track("colinear", rays, CONFIG)
    assert not result.accepted
    assert result.rejection_reason == REJECT_ILL_CONDITIONED


def test_rejects_solution_behind_path() -> None:
    """Flipping every bearing 180° leaves the least-squares point unchanged
    but puts it behind every ray — must reject."""
    drive = make_drive()
    rays = rays_from_detections(drive, synth_detections(drive, (12.0, 260.0)))
    flipped = [
        Ray(r.origin_east_m, r.origin_north_m, (r.bearing_deg + 180.0) % 360.0) for r in rays
    ]
    result = triangulate_track("flipped", flipped, CONFIG, path_enu=drive.path_enu())
    assert not result.accepted
    assert result.rejection_reason == REJECT_BEHIND_PATH


def test_rejects_far_from_road() -> None:
    drive = make_drive()
    # 80m off the road (> 60m threshold), far enough ahead that the bearings
    # still sweep >15° before the camera leaves the FOV — so the ONLY reason
    # to reject is road distance.
    camera = (80.0, 300.0)
    rays = rays_from_detections(drive, synth_detections(drive, camera))
    assert len(rays) >= CONFIG.min_rays
    result = triangulate_track("far", rays, CONFIG, path_enu=drive.path_enu())
    assert not result.accepted
    assert result.rejection_reason == REJECT_FAR_FROM_ROAD


def test_rejects_inconsistent_bearings() -> None:
    """Alternating ±5° bearing noise -> rays disagree -> high residual."""
    drive = make_drive()
    rays = rays_from_detections(drive, synth_detections(drive, (12.0, 260.0)))
    noisy = [
        Ray(r.origin_east_m, r.origin_north_m, r.bearing_deg + (5.0 if i % 2 else -5.0))
        for i, r in enumerate(rays)
    ]
    result = triangulate_track("noisy", noisy, CONFIG, path_enu=drive.path_enu())
    assert not result.accepted
    assert result.rejection_reason == REJECT_HIGH_RESIDUAL


def test_time_sync_error_is_silent_and_large() -> None:
    """A 1.5s video-vs-GPX clock error on a straight pass translates every ray
    origin down-track, so the rays still agree perfectly (tiny residual, track
    ACCEPTED) — but the answer is ~20m wrong at 13 m/s, blowing the 15m budget.

    This is spec §3's 'single most common source of silent error': the geometry
    cannot detect it, which is why the capture procedure measures the offset.
    """
    drive = make_drive()
    camera = (12.0, 260.0)
    detections = synth_detections(drive, camera)
    rays = rays_from_detections(drive, detections, time_sync_error_s=1.5)
    result = triangulate_track("desynced", rays, CONFIG, path_enu=drive.path_enu())

    assert result.accepted, "the error is silent — nothing in the geometry flags it"
    assert result.mean_residual_m < 0.01
    error_m = float(np.linalg.norm(result.point_enu - np.array(camera)))
    expected_shift = 1.5 * drive.speed_mps
    assert error_m > 15.0
    assert abs(error_m - expected_shift) < 1.0
