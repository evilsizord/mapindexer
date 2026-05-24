from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import atan2, degrees
from typing import Iterable, Optional

import numpy as np
from bsp_tool import load_bsp

from mapindexer.flood_fill_volume import (
    can_fly_between,
    has_fly_clearance,
    make_blocked_fn,
    parse_origin,
    point_to_leaf_index,
)


GAMEPLAY_CLASS_WEIGHTS = {
    "team_ctf_redflag": 10.0,
    "team_ctf_blueflag": 10.0,
    "team_ctf_redspawn": 5.0,
    "team_ctf_bluespawn": 5.0,
    "info_player_deathmatch": 4.0,
    "info_player_start": 4.0,
    "info_ut_spawn": 4.0,
    "ut_item": 3.0,
    "weapon_": 2.5,
    "ammo_": 1.5,
    "item_": 1.5,
}


@dataclass
class Anchor:
    classname: str
    origin: np.ndarray
    weight: float


@dataclass
class AnchorCluster:
    center: np.ndarray
    anchors: list[Anchor]
    weight: float


@dataclass
class CameraCandidate:
    kind: str
    position: np.ndarray
    target: np.ndarray
    yaw: float
    pitch: float
    score: float
    visible_anchors: int
    cluster_weight: float

    def to_dict(self) -> dict:
        return {
            "type": self.kind,
            "position": self.position.astype(float).round(3).tolist(),
            "target": self.target.astype(float).round(3).tolist(),
            "yaw": round(float(self.yaw), 3),
            "pitch": round(float(self.pitch), 3),
            "score": round(float(self.score), 3),
            "visible_anchors": self.visible_anchors,
            "cluster_weight": round(float(self.cluster_weight), 3),
        }


def find_camera_candidates(
    bsp,
    *,
    max_cameras: int = 8,
    cluster_radius: float = 768.0,
    min_separation: float = 512.0,
    validate: bool = True,
    verbose: bool = False,
) -> list[CameraCandidate]:
    anchors = extract_gameplay_anchors(bsp)
    if not anchors:
        anchors = fallback_leaf_anchors(bsp)
    if not anchors:
        return []

    clusters = cluster_anchors(anchors, radius=cluster_radius)
    world_bounds = _world_bounds(bsp)
    blocked_fn = make_blocked_fn(bsp, voxel=128.0) if validate else None

    candidates = []
    for cluster in clusters:
        for position in generate_candidate_positions(cluster.center, world_bounds):
            if validate and not is_valid_camera_position(bsp, position, blocked_fn):
                continue
            yaw, pitch = yaw_pitch(position, cluster.center)
            visible = count_visible_anchors(bsp, position, anchors, blocked_fn) if validate else len(anchors)
            score = score_candidate(position, cluster, visible, anchors, world_bounds)
            candidates.append(
                CameraCandidate(
                    kind="anchor_cluster",
                    position=position,
                    target=cluster.center,
                    yaw=yaw,
                    pitch=pitch,
                    score=score,
                    visible_anchors=visible,
                    cluster_weight=cluster.weight,
                )
            )

    selected = select_diverse_candidates(candidates, max_cameras=max_cameras, min_separation=min_separation)

    if verbose:
        print(f"Anchors: {len(anchors)}")
        print(f"Clusters: {len(clusters)}")
        print(f"Candidates: {len(candidates)}")
        print(f"Selected: {len(selected)}")

    return selected


def extract_gameplay_anchors(bsp) -> list[Anchor]:
    anchors = []
    for entity in getattr(bsp, "ENTITIES", []):
        origin = parse_origin(entity.get("origin", ""))
        if origin is None:
            continue
        classname = entity.get("classname", "").lower()
        weight = entity_weight(classname)
        if weight <= 0:
            continue
        anchors.append(Anchor(classname=classname, origin=origin.astype(np.float32), weight=weight))
    return anchors


def entity_weight(classname: str) -> float:
    classname = classname.lower()
    if classname in GAMEPLAY_CLASS_WEIGHTS:
        return GAMEPLAY_CLASS_WEIGHTS[classname]
    for prefix, weight in GAMEPLAY_CLASS_WEIGHTS.items():
        if prefix.endswith("_") and classname.startswith(prefix):
            return weight
    return 0.0


def fallback_leaf_anchors(bsp, min_leaf_volume: float = 128.0 * 128.0 * 128.0) -> list[Anchor]:
    anchors = []
    for leaf in bsp.LEAVES:
        if leaf.cluster < 0:
            continue
        mins = np.array(leaf.bounds.mins, dtype=np.float32)
        maxs = np.array(leaf.bounds.maxs, dtype=np.float32)
        size = maxs - mins
        volume = float(np.prod(size))
        if volume < min_leaf_volume:
            continue
        anchors.append(
            Anchor(
                classname="leaf_centroid",
                origin=(mins + maxs) * 0.5,
                weight=min(volume / min_leaf_volume, 5.0),
            )
        )
    return anchors


def cluster_anchors(anchors: Iterable[Anchor], radius: float = 768.0) -> list[AnchorCluster]:
    clusters: list[list[Anchor]] = []
    radius_sq = radius * radius

    for anchor in sorted(anchors, key=lambda a: -a.weight):
        best_index = None
        best_dist = None
        for i, cluster in enumerate(clusters):
            center = weighted_center(cluster)
            dist_sq = float(np.sum((anchor.origin - center) ** 2))
            if dist_sq <= radius_sq and (best_dist is None or dist_sq < best_dist):
                best_index = i
                best_dist = dist_sq
        if best_index is None:
            clusters.append([anchor])
        else:
            clusters[best_index].append(anchor)

    result = []
    for cluster in clusters:
        weight = sum(a.weight for a in cluster)
        result.append(AnchorCluster(center=weighted_center(cluster), anchors=cluster, weight=weight))
    return sorted(result, key=lambda c: -c.weight)


def weighted_center(anchors: list[Anchor]) -> np.ndarray:
    total = sum(a.weight for a in anchors)
    if total <= 0:
        return np.mean([a.origin for a in anchors], axis=0).astype(np.float32)
    return (sum((a.origin * a.weight for a in anchors), np.zeros(3, dtype=np.float32)) / total).astype(np.float32)


def generate_candidate_positions(center: np.ndarray, world_bounds: tuple[np.ndarray, np.ndarray]) -> list[np.ndarray]:
    mins, maxs = world_bounds
    positions = []
    offsets = [
        np.array([0.0, 0.0, 512.0], dtype=np.float32),
        np.array([768.0, 0.0, 256.0], dtype=np.float32),
        np.array([-768.0, 0.0, 256.0], dtype=np.float32),
        np.array([0.0, 768.0, 256.0], dtype=np.float32),
        np.array([0.0, -768.0, 256.0], dtype=np.float32),
        np.array([512.0, 512.0, 384.0], dtype=np.float32),
        np.array([-512.0, -512.0, 384.0], dtype=np.float32),
    ]
    for offset in offsets:
        positions.append(np.clip(center + offset, mins + 64.0, maxs - 64.0).astype(np.float32))
    return positions


def is_valid_camera_position(bsp, position: np.ndarray, blocked_fn) -> bool:
    leaf_index = point_to_leaf_index(bsp, position)
    if leaf_index is None or bsp.LEAVES[leaf_index].cluster < 0:
        return False
    return has_fly_clearance(bsp, position, blocked_fn=blocked_fn, player_height=32.0)


def count_visible_anchors(bsp, position: np.ndarray, anchors: list[Anchor], blocked_fn) -> int:
    visible = 0
    for anchor in anchors:
        target = anchor.origin + np.array([0.0, 0.0, 48.0], dtype=np.float32)
        if can_fly_between(
            bsp,
            position,
            target,
            blocked_fn=blocked_fn,
            player_height=16.0,
            sample_step=128.0,
        ):
            visible += 1
    return visible


def score_candidate(
    position: np.ndarray,
    cluster: AnchorCluster,
    visible_anchors: int,
    all_anchors: list[Anchor],
    world_bounds: tuple[np.ndarray, np.ndarray],
) -> float:
    mins, maxs = world_bounds
    map_size = np.maximum(maxs - mins, 1.0)
    map_diag = float(np.linalg.norm(map_size))
    dist = float(np.linalg.norm(position - cluster.center))
    distance_score = max(0.0, 1.0 - abs(dist - 900.0) / max(map_diag, 1.0))
    visibility_score = visible_anchors / max(len(all_anchors), 1)
    height = float(position[2] - cluster.center[2])
    height_score = max(0.0, 1.0 - abs(height - 384.0) / 1024.0)
    return cluster.weight * 3.0 + visibility_score * 20.0 + distance_score * 8.0 + height_score * 5.0


def select_diverse_candidates(
    candidates: list[CameraCandidate],
    *,
    max_cameras: int,
    min_separation: float,
) -> list[CameraCandidate]:
    selected = []
    for candidate in sorted(candidates, key=lambda c: -c.score):
        if len(selected) >= max_cameras:
            break
        if all(np.linalg.norm(candidate.position - other.position) >= min_separation for other in selected):
            selected.append(candidate)
    return selected


def yaw_pitch(from_pos: np.ndarray, to_pos: np.ndarray) -> tuple[float, float]:
    dx, dy, dz = to_pos - from_pos
    yaw = degrees(atan2(float(dy), float(dx)))
    pitch = degrees(atan2(float(dz), float(np.linalg.norm([dx, dy]))))
    return yaw, pitch


def _world_bounds(bsp) -> tuple[np.ndarray, np.ndarray]:
    world_model = bsp.MODELS[0]
    return (
        np.array(world_model.bounds.mins, dtype=np.float32),
        np.array(world_model.bounds.maxs, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate screenshot camera candidates from BSP gameplay anchors")
    parser.add_argument("path", help="Path to a BSP file")
    parser.add_argument("--max-cameras", type=int, default=8)
    parser.add_argument("--cluster-radius", type=float, default=768.0)
    parser.add_argument("--min-separation", type=float, default=512.0)
    parser.add_argument("--no-validate", action="store_true", help="Skip BSP collision/visibility validation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    bsp = load_bsp(args.path)
    cameras = find_camera_candidates(
        bsp,
        max_cameras=args.max_cameras,
        cluster_radius=args.cluster_radius,
        min_separation=args.min_separation,
        validate=not args.no_validate,
        verbose=args.verbose,
    )
    print(json.dumps([camera.to_dict() for camera in cameras], indent=2))


if __name__ == "__main__":
    main()
