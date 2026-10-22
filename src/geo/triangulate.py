"""Least-squares closest point to N bearing rays (spec §7).

Works entirely in a local ENU frame in meters. Each ray is a vehicle position
(origin) plus a unit direction from the absolute bearing:

    d = [sin β, cos β]        (β degrees clockwise from true north)
    Aᵢ = I - dᵢ dᵢᵀ
    A  = Σ Aᵢ,  b = Σ Aᵢ pᵢ,  x = A⁻¹ b

x minimizes the summed squared perpendicular distance to all rays.

Every rejection criterion from spec §7 is applied, each with a structured
reason plus a human-readable detail line — the rejection log is a debugging
tool, not noise.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.config import PipelineConfig

# Structured rejection reasons (stable strings — grep the log by these).
REJECT_TOO_FEW_RAYS = "too_few_rays"
REJECT_LOW_ANGULAR_SPREAD = "low_angular_spread"
REJECT_ILL_CONDITIONED = "ill_conditioned"
REJECT_BEHIND_PATH = "behind_path"
REJECT_HIGH_RESIDUAL = "high_residual"
REJECT_FAR_FROM_ROAD = "far_from_road"


@dataclass(frozen=True)
class Ray:
    """One detection: vehicle ENU position + absolute bearing to the target."""

    origin_east_m: float
    origin_north_m: float
    bearing_deg: float

    @property
    def origin(self) -> np.ndarray:
        return np.array([self.origin_east_m, self.origin_north_m])

    @property
    def direction(self) -> np.ndarray:
        b = math.radians(self.bearing_deg)
        return np.array([math.sin(b), math.cos(b)])


@dataclass(frozen=True)
class TriangulationResult:
    track_id: str
    status: str  # "ok" | "rejected"
    rejection_reason: str | None
    point_enu: np.ndarray | None  # [east_m, north_m], None when rejected before solve
    n_rays: int
    angular_spread_deg: float
    condition_number: float | None
    mean_residual_m: float | None
    detail: str  # human-readable one-liner for the rejection log

    @property
    def accepted(self) -> bool:
        return self.status == "ok"


def angular_spread_deg(first_bearing_deg: float, last_bearing_deg: float) -> float:
    """Absolute angle between two bearings, wrapped to [0, 180]."""
    d = abs(last_bearing_deg - first_bearing_deg) % 360.0
    return min(d, 360.0 - d)


def point_to_polyline_distance_m(point: np.ndarray, polyline: np.ndarray) -> float:
    """Min distance from a point to a polyline (N,2), all in ENU meters."""
    if len(polyline) == 1:
        return float(np.linalg.norm(point - polyline[0]))
    seg_a = polyline[:-1]
    seg_ab = polyline[1:] - seg_a
    len_sq = np.sum(seg_ab**2, axis=1)
    ap = point - seg_a
    t = np.divide(np.sum(ap * seg_ab, axis=1), len_sq, out=np.zeros_like(len_sq), where=len_sq > 0)
    t = np.clip(t, 0.0, 1.0)
    closest = seg_a + t[:, None] * seg_ab
    return float(np.min(np.linalg.norm(point - closest, axis=1)))


def triangulate_track(
    track_id: str,
    rays: Sequence[Ray],
    config: PipelineConfig,
    path_enu: np.ndarray | None = None,
) -> TriangulationResult:
    """Triangulate one track's rays; apply every rejection criterion in spec §7.

    `rays` must be in frame order (angular spread is first-to-last).
    `path_enu` is the drive's GPX track as an (N,2) ENU polyline; when given,
    the far-from-road criterion is applied.
    """
    n = len(rays)
    spread = angular_spread_deg(rays[0].bearing_deg, rays[-1].bearing_deg) if n >= 2 else 0.0

    def rejected(
        reason: str,
        detail: str,
        point: np.ndarray | None = None,
        cond: float | None = None,
        residual: float | None = None,
    ) -> TriangulationResult:
        return TriangulationResult(
            track_id=track_id,
            status="rejected",
            rejection_reason=reason,
            point_enu=point,
            n_rays=n,
            angular_spread_deg=spread,
            condition_number=cond,
            mean_residual_m=residual,
            detail=f"track {track_id}: {detail}",
        )

    if n < config.min_rays:
        return rejected(REJECT_TOO_FEW_RAYS, f"{n} rays < {config.min_rays} minimum")
    if spread < config.min_angular_spread_deg:
        return rejected(
            REJECT_LOW_ANGULAR_SPREAD,
            f"angular spread {spread:.1f}° < {config.min_angular_spread_deg:.1f}° minimum "
            "(near-parallel rays — camera likely dead ahead)",
        )

    origins = np.array([r.origin for r in rays])  # (n, 2)
    dirs = np.array([r.direction for r in rays])  # (n, 2), unit
    identity = np.eye(2)
    projections = identity[None, :, :] - dirs[:, :, None] * dirs[:, None, :]  # (n, 2, 2)
    a_mat = projections.sum(axis=0)
    b_vec = np.einsum("nij,nj->i", projections, origins)

    cond = float(np.linalg.cond(a_mat))
    if not math.isfinite(cond) or cond > config.max_condition_number:
        return rejected(
            REJECT_ILL_CONDITIONED,
            f"condition number {cond:.3g} > {config.max_condition_number:.3g} "
            "(degenerate ray geometry)",
            cond=cond,
        )

    x = np.linalg.solve(a_mat, b_vec)
    to_x = x - origins
    perp = np.einsum("nij,nj->ni", projections, to_x)
    residuals = np.linalg.norm(perp, axis=1)
    mean_residual = float(residuals.mean())
    ranges_along_ray = np.einsum("ni,ni->n", dirs, to_x)

    if float(np.median(ranges_along_ray)) <= 0.0:
        return rejected(
            REJECT_BEHIND_PATH,
            f"solution is behind the vehicle path (median range along rays "
            f"{float(np.median(ranges_along_ray)):.1f}m)",
            point=x,
            cond=cond,
            residual=mean_residual,
        )
    if mean_residual > config.max_residual_m:
        return rejected(
            REJECT_HIGH_RESIDUAL,
            f"mean residual {mean_residual:.1f}m > {config.max_residual_m:.1f}m "
            "(rays don't agree — suspect time sync or yaw calibration)",
            point=x,
            cond=cond,
            residual=mean_residual,
        )
    if path_enu is not None:
        road_dist = point_to_polyline_distance_m(x, path_enu)
        if road_dist > config.max_road_distance_m:
            return rejected(
                REJECT_FAR_FROM_ROAD,
                f"solution {road_dist:.0f}m from the driven track > "
                f"{config.max_road_distance_m:.0f}m maximum",
                point=x,
                cond=cond,
                residual=mean_residual,
            )

    return TriangulationResult(
        track_id=track_id,
        status="ok",
        rejection_reason=None,
        point_enu=x,
        n_rays=n,
        angular_spread_deg=spread,
        condition_number=cond,
        mean_residual_m=mean_residual,
        detail=(
            f"track {track_id}: ok — {n} rays, spread {spread:.1f}°, "
            f"residual {mean_residual:.2f}m, cond {cond:.1f}"
        ),
    )
