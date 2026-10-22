"""Cluster triangulated points so multiple passes collapse into one candidate.

Spec §8 calls for DBSCAN with eps ≈ 15m and min_samples = 1. With
min_samples = 1, DBSCAN reduces exactly to connected components of the
eps-neighborhood graph (every point is a core point; clusters are chains of
points within eps of each other). Implemented with a KD-tree + union-find —
identical output, no ML library needed in this milestone.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Cluster:
    centroid_enu: np.ndarray  # [east_m, north_m]
    member_indices: tuple[int, ...]
    track_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.member_indices)


def cluster_points(
    points_enu: np.ndarray,
    track_ids: Sequence[str],
    eps_m: float,
    min_samples: int = 1,
) -> list[Cluster]:
    """Group (N,2) ENU points into clusters of chained eps_m-neighbors.

    Only min_samples=1 is supported — the value at which this is exactly
    DBSCAN. Anything else needs true density-based clustering.
    """
    if min_samples != 1:
        raise ValueError(
            "cluster_points implements DBSCAN's min_samples=1 case only "
            f"(got min_samples={min_samples})"
        )
    if len(points_enu) != len(track_ids):
        raise ValueError("points_enu and track_ids must be the same length")
    if len(points_enu) == 0:
        return []

    parent = list(range(len(points_enu)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = cKDTree(points_enu)
    for i, j in tree.query_pairs(eps_m):
        parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(points_enu)):
        groups.setdefault(find(i), []).append(i)

    clusters = [
        Cluster(
            centroid_enu=points_enu[members].mean(axis=0),
            member_indices=tuple(members),
            track_ids=tuple(track_ids[i] for i in members),
        )
        for members in groups.values()
    ]
    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters
