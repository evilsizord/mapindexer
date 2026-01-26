from __future__ import annotations
import os
from pathlib import Path
import json
import numpy as np
from bsp_tool import load_bsp
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional, Tuple, Set
from itertools import combinations
from bsp_helpers import Vec3, intersect_3_planes, point_inside_planes, is_junk_shader, is_clip_shader, is_solid_shader
from bsp_tool.branches.id_software.quake3 import Leaf, Plane, Node, Brush, BrushSide, Texture


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



def compute_collision_bounds_from_brushes(brushes, brushSides, planes, textures, *,
                                         include_solid: bool = True,
                                         include_playerclip: bool = True,
                                         eps_inside: float = 0.02):
    """
    Computes an AABB by converting selected brushes (solid / playerclip) into vertices
    via 3-plane intersections, then bounding those vertices.

    Returns: (mins, maxs, brush_count_used)
    """
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    used = 0

    for bi, b in enumerate(brushes):
        if b.num_sides is None or b.num_sides <= 3:
            continue

        # Decide whether to a include this brush based on contents OR shader name
        # todo: couldn't a brush have multiple textures..?
        tex_idx = b.texture
        tex = textures[tex_idx] if 0 <= tex_idx < len(textures) else None

        include = False
        if include_solid and is_solid_shader(tex): #todo errr handling. Would shader be None?
            include = True

        if not include and include_playerclip:
            if tex and is_clip_shader(tex):
                include = True

        if not include:
            continue

        # Exclude common non-playable/junk volumes, but DO NOT exclude clip/playerclip
        if tex and is_junk_shader(tex) and not is_clip_shader(tex):
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



def parse_origin(origin_str: str) -> Optional[Vec3]:
    try:
        parts = origin_str.replace(",", " ").split()
        if len(parts) != 3:
            return None
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except Exception:
        return None
    

def pick_spawn_origin(ents) -> Optional[Vec3]:
    """
    Tries common UrT/Q3 spawn entity classnames in priority order.
    Returns the first valid origin found.
    """
    spawn_classnames = [
        "info_ut_spawn",              # UrT commonly
        "info_player_deathmatch",     # Q3 / many mods
        "info_player_start",
        "info_player_team1",
        "info_player_team2",
    ]

    # first pass: preferred classnames
    for cn in spawn_classnames:
        for e in ents:
            if e.get("classname", "").lower() == cn.lower():
                o = parse_origin(e.get("origin", ""))
                if o is not None:
                    return o

    # fallback: any entity with an origin
    for e in ents:
        o = parse_origin(e.get("origin", ""))
        if o is not None:
            return o

    return None



# ----------------------------
# BSP traversal: point -> leaf
# ----------------------------

def point_to_leaf_index(
    point: Vec3,
    nodes: List[Node],
    leafs: List[Leaf],
    planes: List[Plane],
    root_node_index: int = 0,
) -> int:
    """
    Traverse BSP nodes to find the leaf containing 'point'.
    In Quake3, root is node 0.
    Node children: >=0 = node index, <0 = leaf index encoded as -child - 1.
    """
    idx = root_node_index
    while True:
        if idx < 0:
            leaf_index = -idx - 1
            if 0 <= leaf_index < len(leafs):
                return leaf_index
            raise RuntimeError(f"Invalid leaf index computed: {leaf_index}")

        node = nodes[idx]
        plane = planes[node.planeIndex]
        d = (plane.normal[0] * point[0] +
             plane.normal[1] * point[1] +
             plane.normal[2] * point[2]) - plane.dist

        # front if d >= 0, back otherwise (standard Q3 convention)
        idx = node.children[0] if d >= 0 else node.children[1]



# ----------------------------
# VIS/PVS: seed cluster -> visible clusters
# ----------------------------

# def pvs_clusters_for_seed(vis: Optional[VisData], seed_cluster: int) -> Set[int]:
#     """
#     Returns the set of clusters visible from seed_cluster, using the PVS bitset.
#     If VIS is missing/unusable, fall back to "all clusters" (less strict).
#     """
#     if vis is None or vis.numClusters <= 0 or vis.bytesPerCluster <= 0:
#         return set()  # signal "unknown"

#     if seed_cluster < 0 or seed_cluster >= vis.numClusters:
#         return set()

#     bpc = vis.bytesPerCluster
#     start = seed_cluster * bpc
#     end = start + bpc
#     if end > len(vis.bitsets):
#         return set()

#     bits = vis.bitsets[start:end]
#     visible: Set[int] = set()
#     for cluster in range(vis.numClusters):
#         byte_i = cluster // 8
#         bit_i = cluster % 8
#         if byte_i >= len(bits):
#             break
#         if (bits[byte_i] >> bit_i) & 1:
#             visible.add(cluster)
#     # Typically a cluster can "see itself"; but even if not set, include it.
#     visible.add(seed_cluster)
#     return visible



def brush_vertices(
    brush: Brush,
    brushSides: List[BrushSide],
    planes: List[Plane],
    eps_inside: float = 0.02,
) -> List[Vec3]:
    plane_list: List[Tuple[Vec3, float]] = []
    s0 = int(brush.firstSide)
    s1 = s0 + int(brush.num_sides)
    for si in range(s0, s1):
        pl = planes[int(brushSides[si].plane)]
        plane_list.append((tuple(pl.normal), float(pl.distance)))

    verts: List[Vec3] = []
    for (n1, d1), (n2, d2), (n3, d3) in combinations(plane_list, 3):
        p = intersect_3_planes(n1, d1, n2, d2, n3, d3)
        if p is None:
            continue
        if point_inside_planes(p, plane_list, eps=eps_inside):
            verts.append(p)
    return verts



def compute_playable_bounds_spawn_seeded(
    *,
    entities_text: str,
    nodes: List[Node],
    leafs: List[Leaf],
    planes: List[Plane],
    #vis: Optional[VisData],
    leafBrushes: List[int],
    brushes: List[Brush],
    brushSides: List[BrushSide],
    textures: List[Texture],
    include_common_playerclip: bool = True,
    eps_inside: float = 0.02,
) -> Tuple[Vec3, Vec3, Dict[str, int]]:
    """
    Returns collision-based playable bounds seeded from a spawn point, restricted by PVS clusters.

    Includes brushes that are:
      - SOLID (contentsFlags & CONTENTS_SOLID)
      - OR shader name == common/clip (and optionally common/playerclip)

    Only considers brushes referenced by leafs whose cluster is visible in the seed cluster PVS.
    """

    # 1) Choose seed point
    seed = pick_spawn_origin(entities_text)
    if seed is None:
        raise RuntimeError("Could not find a spawn origin in Entities lump to seed bounds.")

    # 2) Find leaf containing seed
    seed_leaf = point_to_leaf_index(seed, nodes, leafs, planes)
    seed_cluster = leafs[seed_leaf].cluster
    if seed_cluster < 0:
        # Seed ended up in solid/outside; bounds will be unreliable.
        raise RuntimeError(f"Seed point is in a leaf with cluster={seed_cluster} (likely solid/outside). Try a different seed.")

    # 3) Determine visible clusters from seed cluster
    #visible = pvs_clusters_for_seed(vis, seed_cluster)
    have_pvs = False

    # 4) Collect brushes from leafs in visible clusters
    CLIP_NAMES = {"common/clip"}
    if include_common_playerclip:
        CLIP_NAMES.add("common/playerclip")

    def tex_name(idx: int) -> str:
        if 0 <= idx < len(textures):
            return (textures[idx].name or "").lower()
        return ""

    def tex_contents(idx: int) -> int:
        if 0 <= idx < len(textures):
            return int(textures[idx].contentsFlags)
        return 0

    brush_indices: Set[int] = set()
    leafs_considered = 0

    for lf in leafs:
        c = lf.cluster
        if c < 0:
            continue
        #if have_pvs and c not in visible:
        #    continue

        leafs_considered += 1
        lb0 = int(lf.first_leaf_brush)
        lb1 = lb0 + int(lf.num_leaf_brushes)
        for i in range(lb0, lb1):
            if 0 <= i < len(leafBrushes):
                bi = int(leafBrushes[i])
                if 0 <= bi < len(brushes):
                    brush_indices.add(bi)

    # 5) Build bounds from included brush vertices
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    included_brushes = 0

    for bi in brush_indices:
        b = brushes[bi]
        if int(b.num_sides) <= 3:
            continue

        t = int(b.texture)
        tex = textures[t] if 0 <= t < len(textures) else None
        #name = tex_name(t)
        #cflags = tex_contents(t)

        include = is_solid_shader(tex) or is_clip_shader(tex)
        if not include:
            continue

        verts = brush_vertices(b, brushSides, planes, eps_inside=eps_inside)
        if not verts:
            continue

        for x, y, z in verts:
            if x < mins[0]: mins[0] = x
            if y < mins[1]: mins[1] = y
            if z < mins[2]: mins[2] = z
            if x > maxs[0]: maxs[0] = x
            if y > maxs[1]: maxs[1] = y
            if z > maxs[2]: maxs[2] = z

        included_brushes += 1

    if included_brushes == 0 or mins[0] == float("inf"):
        raise RuntimeError("No SOLID/common-clip brushes included after PVS filtering. Check VIS, entities seed, or parsing.")

    stats = {
        "seed_leaf": seed_leaf,
        "seed_cluster": seed_cluster,
        "have_pvs": 1 if have_pvs else 0,
        #"visible_clusters": len(visible) if have_pvs else 0,
        "leafs_considered": leafs_considered,
        "candidate_brushes": len(brush_indices),
        "included_brushes": included_brushes,
    }

    return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2]), stats

