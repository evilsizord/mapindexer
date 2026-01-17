import os
from pathlib import Path
import json
import numpy as np
from bsp_tool import load_bsp
from dotenv import load_dotenv
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

load_dotenv()

RAW = Path(os.getenv("MAPS_RAW_DIR"))
OUT = Path(os.getenv("MAPS_OUT_DIR"))
db_path = os.getenv("MAPINDEXER_DB")

if not db_path:
    raise SystemExit("MAPINDEXER_DB environment variable not set")

try:
    dbconn = sqlite3.connect(db_path)
except sqlite3.Error as e:
    raise SystemExit(f"Failed to connect to database {db_path}: {e}")

# -----------------------------
# Tunable heuristics
# -----------------------------

MIN_LEAF_VOLUME = 128 * 128 * 128   # ignore tiny spaces (vents, trims)
MIN_ROOM_LEAFS = 3                 # cluster must contain >= N leafs
VERTICALITY_NORMALIZER = 4096.0    # typical Q3 vertical span

# -----------------------------
# Core analysis
# -----------------------------

def analyze_bsp(bsp_path: Path) -> dict:
    #bsp_path = Path(bsp_path)
    bsp = load_bsp(str(bsp_path))

    stats = {}

    #debug
    #print([L.name for L in bsp.branch.LUMP if bsp.headers[L.name].offset != 0])
    #['ENTITIES', 'TEXTURES', 'PLANES', 'NODES', 'LEAVES', 'LEAF_FACES', 'LEAF_BRUSHES', 
    # 'MODELS', 'BRUSHES', 'BRUSH_SIDES', 'VERTICES', 'INDICES', 'EFFECTS', 'FACES', 
    # 'LIGHTMAPS', 'LIGHT_GRID', 'VISIBILITY']

    #print(bsp.headers["MODELS"])
    #<LumpHeader (offset: 513476, length: 2440)>

    #print(bsp.branch)
    #<module 'bsp_tool.branches.id_software.quake3' from 'C:\\Python310\\lib\\site-packages\\bsp_tool\\branches\\id_software\\quake3.py'>

    #print(bsp.MODELS[0])

    #return False

    # ------------------------------------------------
    # 1. Map bounding box (true world size)
    # ------------------------------------------------
    world_model = bsp.MODELS[0]

    mins = np.array(world_model.bounds.mins)
    maxs = np.array(world_model.bounds.maxs)

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

    for leaf in bsp.LEAVES:
        mins = np.array(leaf.bounds.mins)
        maxs = np.array(leaf.bounds.maxs)
        vol = np.prod(maxs - mins)

        if vol >= MIN_LEAF_VOLUME and leaf.cluster >= 0:
            leaf_volumes.append(vol)
            leaf_clusters.append(leaf.cluster)

    stats["significant_leaf_count"] = len(leaf_volumes)

    # ------------------------------------------------
    # 4. Room count estimation
    # ------------------------------------------------
    room_count = estimate_rooms(
        leaf_clusters=leaf_clusters
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

    (wmins, wmaxes) = compute_world_vertex_bounds(
        models=bsp.MODELS,
        faces=bsp.FACES,
        vertices=bsp.VERTICES,
        textures=bsp.TEXTURES
    )

    stats["world_vertex_bounds"] = {
        "min": wmins,
        "max": wmaxes,
        "size": (np.array(wmaxes) - np.array(wmins)).tolist()
    }

    (occ_mins, occ_maxs, n) = compute_leaf_occupancy_bounds(bsp.LEAVES)

    #print("Leaf occupancy bounds:", occ_mins, occ_maxs, "leafs used:", n)

    size = (
        occ_maxs[0] - occ_mins[0],
        occ_maxs[1] - occ_mins[1],
        occ_maxs[2] - occ_mins[2],
    )
    #print("Occupancy size:", size)

    # todo - initial testing, this was actually the same as world_vertex_bounds? Do we need both?
    stats["leaf_occupancy_bounds"] = {
        "min": occ_mins,
        "max": occ_maxs,
        "size": list(size)
    }

    return stats


# -----------------------------
# Room estimation logic
# -----------------------------
# todo lol this doesnt use visdata at all right
# todo this not working rn
def estimate_rooms(leaf_clusters):
    """
    Heuristic:
    - Group leafs by visibility cluster
    - Remove clusters with minimal representation
    - Resulting clusters approximate rooms / areas
    """

    if not leaf_clusters:
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


def compute_world_vertex_bounds(models, faces, vertices, textures):
    """
    Computes an AABB from all vertices referenced by faces
    belonging to Models[0] (the world model).
    """

    ## NOTE - this slightly improves world model size calculation
    ## But is a little computationally expensive. (idk like 1 sec per map)

    world = models[0]

    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]

    # Iterate faces that belong to the world model
    start = world.first_face
    end   = start + world.num_faces

    #debug
    print(f"World model faces: {start} to {end - 1}")

    for face_index in range(start, end):
        face = faces[face_index]

        # Skip faces without vertices
        if face.num_vertices <= 0:
            continue

        # Skip sky and nodraw faces
        shader = str(textures[face.texture].name)
        if shader.startswith("common/sky") or "nodraw" in shader:
            continue

        v_start = face.first_vertex
        v_end   = v_start + face.num_vertices
        for v_index in range(v_start, v_end):
            v = vertices[v_index]
            x, y, z = v.position

            if x < mins[0]: mins[0] = x
            if y < mins[1]: mins[1] = y
            if z < mins[2]: mins[2] = z

            if x > maxs[0]: maxs[0] = x
            if y > maxs[1]: maxs[1] = y
            if z > maxs[2]: maxs[2] = z

    return tuple(mins), tuple(maxs)


Vec3 = Tuple[float, float, float]

def compute_leaf_occupancy_bounds(leafs) -> Tuple[Vec3, Vec3, int]:
    """
    Compute a leaf-based AABB intended to approximate "spatial occupancy".

    Heuristics (configurable):
      - Exclude leafs with cluster == -1 (typically solid leafs) unless include_cluster_minus_one=True
      - Optionally require the leaf to reference faces/brushes (require_refs=True)
      - Optionally skip degenerate leaf bounds (mins > maxs) if clamp_degenerate=True

    Returns: (mins, maxs, used_leaf_count)
    Raises: RuntimeError if no leafs pass the filters.
    """
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    used = 0

    for lf in leafs:
        # 1) Filter out solid/unreachable-ish leafs (common Quake3 convention)
        if lf.cluster == -1:
            continue

        # 2) Filter out leafs that don't reference anything (often empty partitions)
        if (lf.num_leaf_faces <= 0 and lf.num_leaf_brushes <= 0):
            continue

        lx0, ly0, lz0 = lf.bounds.mins
        lx1, ly1, lz1 = lf.bounds.maxs

        # 3) Handle degenerate bounds (rare, but can exist in broken BSPs)
        
        x0, x1 = (lx0, lx1) if lx0 <= lx1 else (lx1, lx0)
        y0, y1 = (ly0, ly1) if ly0 <= ly1 else (ly1, ly0)
        z0, z1 = (lz0, lz1) if lz0 <= lz1 else (lz1, lz0)


        # Expand global AABB
        if x0 < mins[0]: mins[0] = x0
        if y0 < mins[1]: mins[1] = y0
        if z0 < mins[2]: mins[2] = z0

        if x1 > maxs[0]: maxs[0] = x1
        if y1 > maxs[1]: maxs[1] = y1
        if z1 > maxs[2]: maxs[2] = z1

        used += 1

    if used == 0 or mins[0] == float("inf"):
        raise RuntimeError("No leafs passed filters; cannot compute occupancy bounds.")

    return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2]), used



for mapdir in OUT.iterdir():
    if mapdir.is_dir():
        try:
            stats = analyze_bsp(mapdir / f"maps/{mapdir.stem}.bsp")
            #print(f"Extracted {mapdir.name}")

            #debug
            print(json.dumps(stats, indent=2))
            break
        except Exception as e:
            print(f"Failed {mapdir.name}: {e}")
            break



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