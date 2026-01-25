from __future__ import annotations
import os
from pathlib import Path
import json
import numpy as np
from bsp_tool import load_bsp
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
from itertools import combinations
from typing import List, Tuple, Optional


Vec3 = Tuple[float, float, float]

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
    occ_size = (
        occ_maxs[0] - occ_mins[0],
        occ_maxs[1] - occ_mins[1],
        occ_maxs[2] - occ_mins[2],
    )
    stats["leaf_occupancy_bounds"] = {
        "min": occ_mins,
        "max": occ_maxs,
        "size": list(occ_size)
    }

    col_mins, col_maxs, n = compute_collision_bounds_from_brushes(
        bsp.BRUSHES, bsp.BRUSH_SIDES, bsp.PLANES, bsp.TEXTURES,
        include_solid=True,
        include_playerclip=True,   # try True for UrT
    )

    stats["collision_bounds"] = {
        "min": col_mins,
        "max": col_maxs,
        "size": (
            col_maxs[0] - col_mins[0],
            col_maxs[1] - col_mins[1],
            col_maxs[2] - col_mins[2],
        )
    }

    print("Collision bounds:", col_mins, col_maxs, "brushes used:", n)
    print("Size:", (col_maxs[0]-col_mins[0], col_maxs[1]-col_mins[1], col_maxs[2]-col_mins[2]))

    # Final best-guess of "playable size" (min of all three methods)
    stats["size_best_guess"] = [
        min( stats["leaf_occupancy_bounds"]["size"][0], stats["bounds"]["size"][0], stats["world_vertex_bounds"]["size"][0] ),
        min( stats["leaf_occupancy_bounds"]["size"][1], stats["bounds"]["size"][1], stats["world_vertex_bounds"]["size"][1] ),
        min( stats["leaf_occupancy_bounds"]["size"][2], stats["bounds"]["size"][2], stats["world_vertex_bounds"]["size"][2] )
    ]

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

def compute_leaf_occupancy_bounds(leafs):
    """
    Compute a leaf-based AABB intended to approximate "spatial occupancy".

    Heuristics (configurable):
      - Exclude leafs with cluster < 0 (all solid leafs, not just -1)
      - Optionally require the leaf to reference faces/brushes (require_refs=True)
      - Optionally skip degenerate leaf bounds (mins > maxs) if clamp_degenerate=True

    Returns: (mins, maxs, used_leaf_count)
    Raises: RuntimeError if no leafs pass the filters.
    """
    # NOTE - in practice this doens't seem like the best technique. Some maps like layout_del_1 have 
    # areas outside the playable map but inside the leaf bounds. So they throw off the size calc.
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    used = 0

    for lf in leafs:
        # 1) Filter out ALL solid leafs (cluster < 0, not just == -1)
        # This ensures we only include leafs in the playable/visible PVS clusters
        if lf.cluster < 0:
            continue

        # 2) Filter out leafs that don't reference anything (often empty partitions)
        # Use 'or' instead of 'and' to ensure the leaf has at least faces or brushes
        if lf.num_leaf_faces <= 0 and lf.num_leaf_brushes <= 0:
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




CONTENTS_SOLID     = 0x00000001
CONTENTS_PLAYERCLIP= 0x00010000  # commonly used in Q3-derived games (value used in many toolchains)
CONTENTS_BODY      = 0x02000000  # sometimes present




def dot(a,b) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def sub(a,b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def cross(a,b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

def add(a,b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def mul(a,s):
    return (a[0]*s, a[1]*s, a[2]*s)

def plane_eval(normal, dist, p):
    # plane equation: dot(n, p) - dist
    return dot(normal, p) - dist


def intersect_3_planes(n1, d1: float, n2, d2: float, n3, d3: float, eps: float = 1e-6):
    # Using formula: p = (d1*(n2×n3) + d2*(n3×n1) + d3*(n1×n2)) / (n1·(n2×n3))
    n2xn3 = cross(n2, n3)
    denom = dot(n1, n2xn3)
    if abs(denom) < eps:
        return None

    term1 = mul(n2xn3, d1)
    term2 = mul(cross(n3, n1), d2)
    term3 = mul(cross(n1, n2), d3)
    p = mul(add(add(term1, term2), term3), 1.0 / denom)
    return p



def point_inside_planes(p, planes, eps: float = 0.02):
    # Inside means dot(n,p) - dist <= eps for all planes
    for n, d in planes:
        if plane_eval(n, d, p) > eps:
            return False
    return True

def is_junk_shader(name: str) -> bool:
    n = str(name).lower()
    # These frequently create huge bounds and aren't "playable envelope"
    return (
        "trigger" in n or
        "hint" in n or
        "skip" in n or
        "areaportal" in n or
        "portal" in n or
        "fog" in n
    )

def compute_collision_bounds_from_brushes(brushes, brushSides, planes, textures, *,
                                         include_solid: bool = True,
                                         include_playerclip: bool = True,
                                         use_shader_name_for_clip: bool = True,
                                         eps_inside: float = 0.02):
    """
    Computes an AABB by converting selected brushes (solid / playerclip) into vertices
    via 3-plane intersections, then bounding those vertices.

    Returns: (mins, maxs, brush_count_used)
    """
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    used = 0

    def is_clip_shader(name: str) -> bool:
        n = str(name).lower()
        return n in ("common/clip", "common/playerclip") # todo  is this complete list?
        #return ("playerclip" in n) or (n.endswith("/clip")) or ("/clip" in n)

    for bi, b in enumerate(brushes):
        if b.num_sides is None or b.num_sides <= 3:
            continue

        # Decide whether to a include this brush based on contents OR shader name
        tex_idx = b.texture
        shader_name = textures[tex_idx].name if 0 <= tex_idx < len(textures) else ""
        contents = textures[tex_idx].flags[1] if 0 <= tex_idx < len(textures) else 0

        include = False
        if include_solid and (contents & CONTENTS_SOLID):
            include = True

        # Playerclip in UrT is often expressed via shader name even if contents flags are odd
        if not include and include_playerclip:
            if use_shader_name_for_clip and shader_name and is_clip_shader(shader_name):
                include = True
            else:
                # If your toolchain provides CONTENTS_PLAYERCLIP reliably, you can use it:
                if contents & CONTENTS_PLAYERCLIP:
                    include = True

        if not include:
            continue

        # Exclude common non-playable/junk volumes, but DO NOT exclude clip/playerclip
        if shader_name and is_junk_shader(shader_name) and not is_clip_shader(shader_name):
            continue

        # Collect this brush's planes
        brush_plane_list = []
        for si in range(b.first_side, b.first_side + b.num_sides):
            ps = brushSides[si].plane
            pl = planes[ps]
            brush_plane_list.append((pl.normal, pl.distance))

        # Enumerate candidate vertices from all 3-plane intersections
        verts= []
        for (n1, d1), (n2, d2), (n3, d3) in combinations(brush_plane_list, 3):
            p = intersect_3_planes(n1, d1, n2, d2, n3, d3)
            if p is None:
                continue
            if point_inside_planes(p, brush_plane_list, eps=eps_inside):
                verts.append(p)

        if not verts:
            continue

        # Expand AABB with brush vertices
        for x, y, z in verts:
            if x < mins[0]: mins[0] = x
            if y < mins[1]: mins[1] = y
            if z < mins[2]: mins[2] = z
            if x > maxs[0]: maxs[0] = x
            if y > maxs[1]: maxs[1] = y
            if z > maxs[2]: maxs[2] = z

        used += 1

    if used == 0 or mins[0] == float("inf"):
        raise RuntimeError("No collision brushes found with the chosen filters.")

    return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2]), used


