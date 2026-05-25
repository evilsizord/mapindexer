from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from math import atan2, degrees
from typing import Iterable

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
    progress_interval: float = 5.0,
) -> list[CameraCandidate]:
    debug = CameraCandidateDebug(verbose=verbose, progress_interval=progress_interval)

    debug.stage_start("extract anchors")
    anchors = extract_gameplay_anchors(bsp)
    debug.stage_end("extract anchors", anchors=len(anchors), source="entities")
    if not anchors:
        debug.stage_start("fallback leaf anchors")
        anchors = fallback_leaf_anchors(bsp)
        debug.stage_end("fallback leaf anchors", anchors=len(anchors), source="leaves")
    if not anchors:
        debug.finish(empty=True)
        return []

    debug.stage_start("cluster anchors")
    clusters = cluster_anchors(anchors, radius=cluster_radius)
    debug.stage_end("cluster anchors", clusters=len(clusters))
    world_bounds = _world_bounds(bsp)
    debug.print_map_summary(bsp, anchors, clusters, world_bounds, validate)

    debug.stage_start("build blocked function")
    blocked_fn = (
        make_blocked_fn(bsp, voxel=128.0, verbose=verbose, progress_interval=progress_interval)
        if validate
        else None
    )
    debug.stage_end(
        "build blocked function",
        enabled=validate,
        callable=callable(blocked_fn) if validate else False,
    )

    candidates = []
    debug.stage_start("evaluate candidates")
    for cluster_index, cluster in enumerate(clusters, start=1):
        cluster_start = time.perf_counter()
        cluster_generated = 0
        cluster_valid = 0
        cluster_visible_checks = 0
        for position in generate_candidate_positions(cluster.center, world_bounds):
            cluster_generated += 1
            debug.generated_candidates += 1

            valid_start = time.perf_counter()
            valid, invalid_reason = validate_camera_position(bsp, position, blocked_fn) if validate else (True, None)
            debug.validity_time += time.perf_counter() - valid_start
            if not valid:
                debug.invalid_reasons[invalid_reason] += 1
                continue

            cluster_valid += 1
            debug.valid_candidates += 1
            yaw, pitch = yaw_pitch(position, cluster.center)
            visibility_start = time.perf_counter()
            visible = count_visible_anchors(bsp, position, anchors, blocked_fn) if validate else len(anchors)
            visibility_elapsed = time.perf_counter() - visibility_start
            debug.visibility_time += visibility_elapsed
            debug.visibility_checks += len(anchors) if validate else 0
            cluster_visible_checks += len(anchors) if validate else 0
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
        debug.record_cluster(
            cluster_index,
            cluster,
            generated=cluster_generated,
            valid=cluster_valid,
            visibility_checks=cluster_visible_checks,
            elapsed=time.perf_counter() - cluster_start,
        )
        debug.maybe_print_progress(cluster_index, len(clusters), len(candidates), blocked_fn)
    debug.stage_end("evaluate candidates", candidates=len(candidates))

    debug.stage_start("select diverse candidates")
    selected = select_diverse_candidates(candidates, max_cameras=max_cameras, min_separation=min_separation)
    debug.stage_end("select diverse candidates", selected=len(selected))

    debug.finish(
        generated=len(candidates),
        selected=len(selected),
        blocked_fn=blocked_fn,
    )

    return selected


@dataclass
class CameraCandidateDebug:
    verbose: bool = False
    progress_interval: float = 5.0
    generated_candidates: int = 0
    valid_candidates: int = 0
    visibility_checks: int = 0
    validity_time: float = 0.0
    visibility_time: float = 0.0

    def __post_init__(self):
        self.started_at = time.perf_counter()
        self.last_progress_at = self.started_at
        self.stage_started_at = None
        self.stage_times = {}
        self.invalid_reasons = Counter()
        self.slowest_clusters = []

    def stage_start(self, name: str):
        if not self.verbose:
            return
        self.stage_started_at = time.perf_counter()
        print(f"[camera-candidates] starting {name}")

    def stage_end(self, name: str, **details):
        if not self.verbose:
            return
        elapsed = time.perf_counter() - self.stage_started_at if self.stage_started_at is not None else 0.0
        self.stage_times[name] = self.stage_times.get(name, 0.0) + elapsed
        suffix = _format_debug_details(details)
        print(f"[camera-candidates] finished {name} in {elapsed:.2f}s{suffix}")

    def print_map_summary(self, bsp, anchors: list[Anchor], clusters: list[AnchorCluster], world_bounds, validate: bool):
        if not self.verbose:
            return
        mins, maxs = world_bounds
        anchor_classes = Counter(anchor.classname for anchor in anchors)
        top_anchor_classes = ", ".join(f"{name}={count}" for name, count in anchor_classes.most_common(8))
        cluster_sizes = [len(cluster.anchors) for cluster in clusters]
        cluster_weights = [cluster.weight for cluster in clusters]
        print(
            "[camera-candidates] map summary: "
            f"entities={len(getattr(bsp, 'ENTITIES', []))}, brushes={len(getattr(bsp, 'BRUSHES', []))}, "
            f"leaves={len(getattr(bsp, 'LEAVES', []))}, anchors={len(anchors)}, clusters={len(clusters)}, "
            f"validate={validate}"
        )
        print(
            "[camera-candidates] world bounds: "
            f"mins={mins.astype(float).round(1).tolist()}, maxs={maxs.astype(float).round(1).tolist()}, "
            f"size={(maxs - mins).astype(float).round(1).tolist()}"
        )
        if top_anchor_classes:
            print(f"[camera-candidates] top anchor classes: {top_anchor_classes}")
        if cluster_sizes:
            print(
                "[camera-candidates] cluster stats: "
                f"max_size={max(cluster_sizes)}, avg_size={sum(cluster_sizes) / len(cluster_sizes):.1f}, "
                f"max_weight={max(cluster_weights):.1f}, avg_weight={sum(cluster_weights) / len(cluster_weights):.1f}"
            )

    def record_cluster(
        self,
        cluster_index: int,
        cluster: AnchorCluster,
        *,
        generated: int,
        valid: int,
        visibility_checks: int,
        elapsed: float,
    ):
        if not self.verbose:
            return
        self.slowest_clusters.append(
            {
                "index": cluster_index,
                "elapsed": elapsed,
                "anchors": len(cluster.anchors),
                "weight": cluster.weight,
                "generated": generated,
                "valid": valid,
                "visibility_checks": visibility_checks,
            }
        )
        self.slowest_clusters = sorted(self.slowest_clusters, key=lambda item: -item["elapsed"])[:5]

    def maybe_print_progress(self, cluster_index: int, cluster_count: int, candidates: int, blocked_fn):
        if not self.verbose:
            return
        now = time.perf_counter()
        if now - self.last_progress_at < self.progress_interval and cluster_index < cluster_count:
            return
        elapsed = now - self.started_at
        blocked_suffix = _blocked_stats_suffix(blocked_fn)
        print(
            "[camera-candidates] progress: "
            f"clusters={cluster_index}/{cluster_count}, generated_positions={self.generated_candidates}, "
            f"valid_positions={self.valid_candidates}, candidates={candidates}, "
            f"visibility_checks={self.visibility_checks}, elapsed={elapsed:.2f}s"
            f"{blocked_suffix}"
        )
        self.last_progress_at = now

    def finish(self, *, generated: int = 0, selected: int = 0, blocked_fn=None, empty: bool = False):
        if not self.verbose:
            return
        elapsed = time.perf_counter() - self.started_at
        print(f"[camera-candidates] total time: {elapsed:.2f}s")
        if empty:
            print("[camera-candidates] no anchors found; returning no candidates")
            return
        print(
            "[camera-candidates] totals: "
            f"generated_positions={self.generated_candidates}, valid_positions={self.valid_candidates}, "
            f"candidates={generated}, selected={selected}, visibility_checks={self.visibility_checks}, "
            f"validity_time={self.validity_time:.2f}s, visibility_time={self.visibility_time:.2f}s"
            f"{_blocked_stats_suffix(blocked_fn)}"
        )
        if self.invalid_reasons:
            reasons = ", ".join(f"{reason}={count}" for reason, count in self.invalid_reasons.most_common())
            print(f"[camera-candidates] invalid camera positions: {reasons}")
        if self.slowest_clusters:
            print("[camera-candidates] slowest clusters:")
            for item in self.slowest_clusters:
                print(
                    "  "
                    f"#{item['index']}: {item['elapsed']:.2f}s, anchors={item['anchors']}, "
                    f"weight={item['weight']:.1f}, generated={item['generated']}, "
                    f"valid={item['valid']}, visibility_checks={item['visibility_checks']}"
                )


def _format_debug_details(details: dict) -> str:
    visible_details = {key: value for key, value in details.items() if value is not None}
    if not visible_details:
        return ""
    return ": " + ", ".join(f"{key}={value}" for key, value in visible_details.items())


def _blocked_stats_suffix(blocked_fn) -> str:
    stats = getattr(blocked_fn, "_stats", None)
    if not stats:
        return ""
    return (
        f", blocked_calls={stats.get('calls', 0)}, blocked_time={stats.get('time', 0.0):.2f}s, "
        f"blocked_build_time={stats.get('build_time', 0.0):.2f}s"
    )


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
    valid, _ = validate_camera_position(bsp, position, blocked_fn)
    return valid


def validate_camera_position(bsp, position: np.ndarray, blocked_fn) -> tuple[bool, str | None]:
    leaf_index = point_to_leaf_index(bsp, position)
    if leaf_index is None or bsp.LEAVES[leaf_index].cluster < 0:
        return False, "non_playable_leaf"
    if not has_fly_clearance(bsp, position, blocked_fn=blocked_fn, player_height=32.0):
        return False, "clearance_failed"
    return True, None


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
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Seconds between verbose progress messages while evaluating candidates",
    )
    args = parser.parse_args()

    bsp = load_bsp(args.path)
    cameras = find_camera_candidates(
        bsp,
        max_cameras=args.max_cameras,
        cluster_radius=args.cluster_radius,
        min_separation=args.min_separation,
        validate=not args.no_validate,
        verbose=args.verbose,
        progress_interval=args.progress_interval,
    )
    print(json.dumps([camera.to_dict() for camera in cameras], indent=2))


if __name__ == "__main__":
    main()
