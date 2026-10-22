"""Triangulation correctness against the synthetic fixture (spec §7).

The M1 gate: cameras placed at known coordinates must be recovered to
within 2m by the forward pipeline math.
"""

import math

import numpy as np
import pytest

from src.config import PipelineConfig
from src.geo.triangulate import Ray, point_to_polyline_distance_m, triangulate_track
from tests.synthetic import make_drive, rays_from_detections, synth_detections

CONFIG = PipelineConfig()


def test_exact_three_ray_intersection() -> None:
    """Rays constructed through a known point must recover it near-exactly."""
    target = np.array([5.0, 5.0])
    origins = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    rays = [
        Ray(e, n, math.degrees(math.atan2(target[0] - e, target[1] - n)) % 360.0)
        for e, n in origins
    ]
    result = triangulate_track("exact", rays, CONFIG)
    assert result.accepted, result.detail
    assert np.linalg.norm(result.point_enu - target) < 1e-6
    assert result.mean_residual_m == pytest.approx(0.0, abs=1e-9)


def test_recovers_synthetic_camera_within_2m() -> None:
    """Primary M1 gate: drive past a camera 12m off the road, recover its position."""
    drive = make_drive()
    camera = (12.0, 260.0)
    detections = synth_detections(drive, camera)
    assert len(detections) >= CONFIG.min_rays, "fixture must yield a triangulable track"

    rays = rays_from_detections(drive, detections)
    result = triangulate_track("cam_A", rays, CONFIG, path_enu=drive.path_enu())

    assert result.accepted, result.detail
    error_m = float(np.linalg.norm(result.point_enu - np.array(camera)))
    assert error_m < 2.0, f"recovered {error_m:.2f}m from truth"
    # Diagnostics must be coherent for a clean track.
    assert result.n_rays == len(detections)
    assert result.angular_spread_deg > CONFIG.min_angular_spread_deg
    assert result.mean_residual_m < 0.01
    assert result.condition_number < CONFIG.max_condition_number


def test_recovery_survives_integer_pixel_labels() -> None:
    """Hand labels carry integer pixels; rounding must stay well inside 2m."""
    drive = make_drive()
    camera = (12.0, 260.0)
    rays = rays_from_detections(drive, synth_detections(drive, camera), round_pixels=True)
    result = triangulate_track("cam_A", rays, CONFIG, path_enu=drive.path_enu())
    assert result.accepted, result.detail
    assert float(np.linalg.norm(result.point_enu - np.array(camera))) < 2.0


def test_left_side_camera_also_recovers() -> None:
    """Negative relative bearings (camera left of travel) work symmetrically."""
    drive = make_drive()
    camera = (-15.0, 300.0)
    rays = rays_from_detections(drive, synth_detections(drive, camera))
    result = triangulate_track("cam_L", rays, CONFIG, path_enu=drive.path_enu())
    assert result.accepted, result.detail
    assert float(np.linalg.norm(result.point_enu - np.array(camera))) < 2.0


def test_point_to_polyline_distance() -> None:
    polyline = np.array([[0.0, 0.0], [0.0, 100.0]])
    assert point_to_polyline_distance_m(np.array([30.0, 50.0]), polyline) == pytest.approx(30.0)
    # Beyond the end of the polyline: distance to the endpoint, not the infinite line.
    assert point_to_polyline_distance_m(np.array([0.0, 140.0]), polyline) == pytest.approx(40.0)
