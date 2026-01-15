from pathlib import Path
import json
import numpy as np
from bsp_tool import load_bsp

# -----------------------------
# Tunable heuristics
# -----------------------------

MIN_LEAF_VOLUME = 128 * 128 * 128   # ignore tiny spaces (vents, trims)
MIN_ROOM_LEAFS = 3                 # cluster must contain >= N leafs
VERTICALITY_NORMALIZER = 4096.0    # typical Q3 vertical span

# -----------------------------
# Core analysis
# -----------------------------

def analyze_bsp(bsp_path: str) -> dict:
    bsp_path = Path(bsp_path)
    bsp = load_bsp(bsp_path)

    stats = {}

    # ------------------------------------------------
    # 1. Map bounding box (true world size)
    # ------------------------------------------------
    world_model = bsp.models[0]

    mins = np.array(world_model.mins)
    maxs = np.array(world_model.maxs)

    size = maxs - mins

    stats["bounds"] = {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "size": size.tolist()
    }

    stats["bsp_size"] = bsp_path.stat().st_size

    # ------------------------------------------------
    # 2. Verticality score (0.0 - 1.0)
    # ------------------------------------------------
    vertical_span = size[2]

    verticality_score = min(
        vertical_span / VERTICALITY_NORMALIZER,
        1.0
    )

    stats["vertical_span"] = float(vertical_span)
    stats["verticality_score"] = round(verticality_score, 3)

    # ------------------------------------------------
    # 3. Leaf volume calculation
    # ------------------------------------------------
    leaf_volumes = []
    leaf_clusters = []

    for leaf in bsp.leafs:
        mins = np.array(leaf.mins)
        maxs = np.array(leaf.maxs)
        vol = np.prod(maxs - mins)

        if vol >= MIN_LEAF_VOLUME and leaf.cluster >= 0:
            leaf_volumes.append(vol)
            leaf_clusters.append(leaf.cluster)

    stats["significant_leaf_count"] = len(leaf_volumes)

    # ------------------------------------------------
    # 4. Room count estimation
    # ------------------------------------------------
    room_count = estimate_rooms(
        leaf_clusters=leaf_clusters,
        visdata=bsp.visdata
    )

    stats["room_count"] = room_count

    # ------------------------------------------------
    # 5. Complexity score (heuristic)
    # ------------------------------------------------
    complexity_score = compute_complexity(
        room_count=room_count,
        leaf_count=len(leaf_volumes),
        verticality_score=verticality_score
    )

    stats["complexity_score"] = round(complexity_score, 3)

    return stats


# -----------------------------
# Room estimation logic
# -----------------------------

def estimate_rooms(leaf_clusters, visdata):
    """
    Heuristic:
    - Group leafs by visibility cluster
    - Remove clusters with minimal representation
    - Resulting clusters approximate rooms / areas
    """

    if not leaf_clusters or not visdata:
        return 0

    cluster_counts = {}

    for c in leaf_clusters:
        cluster_counts[c] = cluster_counts.get(c, 0) + 1

    # Filter weak clusters (noise, corridors)
    valid_clusters = [
        c for c, count in cluster_counts.items()
        if count >= MIN_ROOM_LEAFS
    ]

    return len(valid_clusters)


# -----------------------------
# Complexity heuristic
# -----------------------------

def compute_complexity(room_count, leaf_count, verticality_score):
    """
    Produces normalized 0.0 - 1.0 score

    Factors:
    - number of rooms
    - spatial partitioning density
    - verticality
    """

    room_factor = min(room_count / 30.0, 1.0)
    leaf_factor = min(leaf_count / 400.0, 1.0)
    vertical_factor = verticality_score

    complexity = (
        0.45 * room_factor +
        0.35 * leaf_factor +
        0.20 * vertical_factor
    )

    return complexity


# -----------------------------
# CLI usage
# -----------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python bsp_stats.py <map.bsp>")
        sys.exit(1)

    stats = analyze_bsp(sys.argv[1])
    print(json.dumps(stats, indent=2))


""" Example output:
{
  "bounds": {
    "min": [-2048, -2048, -512],
    "max": [2048, 2048, 1536],
    "size": [4096, 4096, 2048]
  },
  "bsp_size": 18745321,
  "vertical_span": 2048.0,
  "verticality_score": 0.5,
  "significant_leaf_count": 312,
  "room_count": 18,
  "complexity_score": 0.63
}

"""