from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from bsp_tool import load_bsp

from mapindexer.flood_fill_volume import (
    PLAYER_HEIGHT,
    VOXEL,
    cell_center,
    get_spawn_origins,
    make_blocked_fn,
    point_to_leaf_index,
    has_fly_clearance,
)

"""
What it does:

Finds spawn seed leaves via BSP tree traversal.
Builds an approximate reachable leaf graph using face-touching leaf AABBs.
Optionally expands through same-cluster leaves.
Computes reachable leaf/cluster sets and a reachable leaf AABB.
Optionally voxelizes only the reachable leaf bounds, instead of the whole world grid.
Includes a small CLI:

One caveat: BSP files don’t expose compiler portal adjacency directly here, so leaf adjacency is approximated 
from touching leaf bounds. It should be much faster than full-world voxel BFS, but we’ll want to compare its 
output against a few known maps.
"""


@dataclass
class HybridFloodResult:
    seed_leaves: set[int]
    seed_clusters: set[int]
    reachable_leaves: set[int]
    reachable_clusters: set[int]
    leaf_aabb: Optional[tuple[np.ndarray, np.ndarray]]
    volume: Optional[set[tuple[int, int, int]]] = None
    volume_aabb: Optional[tuple[np.ndarray, np.ndarray]] = None


def flood_reachable_leaves(
    bsp,
    *,
    include_cluster_peers: bool = True,
    adjacency_cell_size: float = 512.0,
    touch_eps: float = 1.0,
    verbose: bool = False,
) -> HybridFloodResult:
    start_time = time.perf_counter()
    playable_leafs = _playable_leaf_indices(bsp)
    cluster_to_leafs = _cluster_to_leafs(bsp, playable_leafs)
    seed_leaves = _seed_leaf_indices(bsp, playable_leafs)

    adjacency = build_leaf_adjacency(
        bsp,
        playable_leafs,
        cell_size=adjacency_cell_size,
        touch_eps=touch_eps,
    )

    reachable_leaves = set()
    q = deque(seed_leaves)

    while q:
        leaf_index = q.popleft()
        if leaf_index in reachable_leaves:
            continue
        reachable_leaves.add(leaf_index)

        leaf = bsp.LEAVES[leaf_index]
        neighbors = set(adjacency.get(leaf_index, ()))
        if include_cluster_peers and leaf.cluster >= 0:
            neighbors.update(cluster_to_leafs.get(leaf.cluster, ()))

        for neighbor in neighbors:
            if neighbor not in reachable_leaves:
                q.append(neighbor)

    reachable_clusters = {
        bsp.LEAVES[i].cluster for i in reachable_leaves if bsp.LEAVES[i].cluster >= 0
    }
    seed_clusters = {
        bsp.LEAVES[i].cluster for i in seed_leaves if bsp.LEAVES[i].cluster >= 0
    }
    leaf_aabb = leaf_bounds_aabb(bsp, reachable_leaves)

    if verbose:
        elapsed = time.perf_counter() - start_time
        print(f"Seed leaves: {len(seed_leaves)}")
        print(f"Seed clusters: {len(seed_clusters)}")
        print(f"Playable leaves: {len(playable_leafs)}")
        print(f"Reachable leaves: {len(reachable_leaves)}")
        print(f"Reachable clusters: {len(reachable_clusters)}")
        print(f"Leaf flood completed in {elapsed:.2f} seconds")

    return HybridFloodResult(
        seed_leaves=seed_leaves,
        seed_clusters=seed_clusters,
        reachable_leaves=reachable_leaves,
        reachable_clusters=reachable_clusters,
        leaf_aabb=leaf_aabb,
    )


def hybrid_flood_fill(
    bsp,
    *,
    voxelize: bool = False,
    voxel: float = VOXEL,
    validate_clearance: bool = True,
    include_cluster_peers: bool = True,
    verbose: bool = False,
) -> HybridFloodResult:
    result = flood_reachable_leaves(
        bsp,
        include_cluster_peers=include_cluster_peers,
        verbose=verbose,
    )

    if voxelize:
        volume, volume_aabb = voxelize_reachable_leaf_bounds(
            bsp,
            result.reachable_leaves,
            voxel=voxel,
            validate_clearance=validate_clearance,
            verbose=verbose,
        )
        result.volume = volume
        result.volume_aabb = volume_aabb

    return result


def build_leaf_adjacency(
    bsp,
    leaf_indices: set[int],
    *,
    cell_size: float = 512.0,
    touch_eps: float = 1.0,
) -> dict[int, set[int]]:
    bounds = {i: _leaf_bounds(bsp.LEAVES[i]) for i in leaf_indices}
    bins = defaultdict(list)
    inv_cell = 1.0 / cell_size

    for leaf_index, (mins, maxs) in bounds.items():
        c0 = np.floor(mins * inv_cell).astype(int)
        c1 = np.floor(maxs * inv_cell).astype(int)
        for x in range(int(c0[0]), int(c1[0]) + 1):
            for y in range(int(c0[1]), int(c1[1]) + 1):
                for z in range(int(c0[2]), int(c1[2]) + 1):
                    bins[(x, y, z)].append(leaf_index)

    adjacency: dict[int, set[int]] = defaultdict(set)
    checked = set()

    for candidates in bins.values():
        for pos, a in enumerate(candidates):
            for b in candidates[pos + 1:]:
                pair = (a, b) if a < b else (b, a)
                if pair in checked:
                    continue
                checked.add(pair)
                if _bounds_touch(bounds[a], bounds[b], touch_eps):
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    return adjacency


def voxelize_reachable_leaf_bounds(
    bsp,
    reachable_leaves: set[int],
    *,
    voxel: float = VOXEL,
    validate_clearance: bool = True,
    verbose: bool = False,
) -> tuple[set[tuple[int, int, int]], Optional[tuple[np.ndarray, np.ndarray]]]:
    start_time = time.perf_counter()
    if not reachable_leaves:
        return set(), None

    leaf_aabb = leaf_bounds_aabb(bsp, reachable_leaves)
    if leaf_aabb is None:
        return set(), None

    mins, maxs = leaf_aabb
    dims = np.ceil((maxs - mins) / voxel).astype(int)
    nx, ny, nz = map(int, dims.tolist())
    reachable_leaf_set = set(reachable_leaves)
    blocked_fn = make_blocked_fn(bsp, voxel=voxel) if validate_clearance else None
    volume = set()

    for reachable_leaf in reachable_leaves:
        leaf_mins, leaf_maxs = _leaf_bounds(bsp.LEAVES[reachable_leaf])
        c0 = np.floor((leaf_mins - mins) / voxel).astype(int)
        c1 = np.ceil((leaf_maxs - mins) / voxel).astype(int)
        c0 = np.maximum(c0, 0)
        c1 = np.minimum(c1, dims)

        for ix in range(int(c0[0]), int(c1[0])):
            for iy in range(int(c0[1]), int(c1[1])):
                for iz in range(int(c0[2]), int(c1[2])):
                    state = (ix, iy, iz)
                    if state in volume:
                        continue
                    p = cell_center(ix, iy, iz, mins, voxel)
                    leaf_index = point_to_leaf_index(bsp, p)
                    if leaf_index not in reachable_leaf_set:
                        continue
                    if validate_clearance and not has_fly_clearance(
                        bsp,
                        p,
                        blocked_fn=blocked_fn,
                        player_height=PLAYER_HEIGHT,
                    ):
                        continue
                    volume.add(state)

    volume_aabb = _volume_aabb(volume, mins, maxs, voxel)

    if verbose:
        elapsed = time.perf_counter() - start_time
        print("Voxelized reachable leaf bounds:", nx, ny, nz)
        print(f"Reachable volume cells: {len(volume)}")
        print(f"Voxelization completed in {elapsed:.2f} seconds")

    return volume, volume_aabb


def leaf_bounds_aabb(bsp, leaf_indices: set[int]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not leaf_indices:
        return None
    mins = []
    maxs = []
    for leaf_index in leaf_indices:
        leaf_mins, leaf_maxs = _leaf_bounds(bsp.LEAVES[leaf_index])
        mins.append(leaf_mins)
        maxs.append(leaf_maxs)
    return np.min(np.stack(mins), axis=0), np.max(np.stack(maxs), axis=0)


def _seed_leaf_indices(bsp, playable_leafs: set[int]) -> set[int]:
    seeds = set()
    for origin in get_spawn_origins(bsp):
        leaf_index = point_to_leaf_index(bsp, origin)
        if leaf_index in playable_leafs:
            seeds.add(leaf_index)
    if not seeds:
        raise RuntimeError("No spawn origins landed in playable BSP leaves.")
    return seeds


def _playable_leaf_indices(bsp) -> set[int]:
    playable = set()
    for i, leaf in enumerate(bsp.LEAVES):
        if leaf.cluster < 0:
            continue
        mins, maxs = _leaf_bounds(leaf)
        if np.any(maxs <= mins):
            continue
        playable.add(i)
    return playable


def _cluster_to_leafs(bsp, leaf_indices: set[int]) -> dict[int, set[int]]:
    clusters = defaultdict(set)
    for i in leaf_indices:
        cluster = bsp.LEAVES[i].cluster
        if cluster >= 0:
            clusters[cluster].add(i)
    return clusters


def _leaf_bounds(leaf) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(leaf.bounds.mins, dtype=np.float32),
        np.array(leaf.bounds.maxs, dtype=np.float32),
    )


def _bounds_touch(
    a: tuple[np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray],
    eps: float,
) -> bool:
    a_min, a_max = a
    b_min, b_max = b
    overlaps = 0
    touches = 0

    for axis in range(3):
        overlap = min(a_max[axis], b_max[axis]) - max(a_min[axis], b_min[axis])
        if overlap > eps:
            overlaps += 1
            continue

        gap = max(a_min[axis], b_min[axis]) - min(a_max[axis], b_max[axis])
        if -eps <= overlap <= eps or 0 <= gap <= eps:
            touches += 1

    return overlaps >= 2 and touches >= 1


def _volume_aabb(
    volume: set[tuple[int, int, int]],
    mins: np.ndarray,
    maxs: np.ndarray,
    voxel: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not volume:
        return None

    half = 0.5 * voxel
    cells = np.array(list(volume), dtype=np.float32)
    cell_mins = mins + (np.min(cells, axis=0) + 0.5) * voxel - half
    cell_maxs = mins + (np.max(cells, axis=0) + 0.5) * voxel + half
    return np.maximum(cell_mins, mins), np.minimum(cell_maxs, maxs)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid leaf/cluster flood fill for Quake 3 BSP maps")
    parser.add_argument("path", help="Path to a BSP file")
    parser.add_argument("--voxelize", action="store_true", help="Voxelize only reachable leaf bounds")
    parser.add_argument("--voxel", type=float, default=128.0, help="Voxel size for optional voxelization")
    parser.add_argument(
        "--no-clearance",
        action="store_true",
        help="Skip player clearance checks during optional voxelization",
    )
    parser.add_argument(
        "--leaf-only",
        action="store_true",
        help="Do not include all leaves from each reached cluster during leaf flood",
    )
    args = parser.parse_args()

    bsp = load_bsp(args.path)
    result = hybrid_flood_fill(
        bsp,
        voxelize=args.voxelize,
        voxel=args.voxel,
        validate_clearance=not args.no_clearance,
        include_cluster_peers=not args.leaf_only,
        verbose=True,
    )

    if result.leaf_aabb:
        mins, maxs = result.leaf_aabb
        print("Reachable leaf AABB:", mins, maxs)
    if result.volume_aabb:
        mins, maxs = result.volume_aabb
        print("Reachable volume AABB:", mins, maxs)


if __name__ == "__main__":
    main()
