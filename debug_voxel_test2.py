from pathlib import Path
import sys
import numpy as np
from math import atan2, degrees
from bsp_tool import load_bsp
from collections import deque, defaultdict
from itertools import combinations

## In this version, trying a 3d grid sampling approach instead of 2.5d

CONTENTS_SOLID     = 0x00000001
VOXEL = 64.0            # grid resolution
STAND_HEIGHT = 48.0     # player origin above floor
PLAYER_HEIGHT = 72.0    # standing clearance
MAX_STEP_UP = 18.0      # normal step height
MAX_STEP_DOWN = 64.0    # allow dropping off small ledges




class SnapCache:
    def __init__(self, bucket_voxels=2):
        # bucket_voxels=2 means bucket size = 2 * voxel
        self.bucket_voxels = int(bucket_voxels)
        self.cache = {}  # (ix, iy, iz_bucket) -> (z_origin or None)

    def bucket(self, iz):
        return iz // self.bucket_voxels

    def get(self, ix, iy, iz):
        return self.cache.get((ix, iy, self.bucket(iz)), "MISS")

    def set(self, ix, iy, iz, value):
        self.cache[(ix, iy, self.bucket(iz))] = value


# -------------------------
# Brush plane extraction
# -------------------------

def brush_planes(bsp, brush):
    sides = bsp.BRUSH_SIDES[brush.first_side : brush.first_side + brush.num_sides]
    normals = []
    distances = []
    for s in sides:
        p = bsp.PLANES[s.plane]
        normals.append([p.normal.x, p.normal.y, p.normal.z])
        distances.append(p.distance)
    return np.asarray(normals, dtype=np.float64), np.asarray(distances, dtype=np.float64)

def brush_aabb_from_planes(normals, distances, inside_eps=0.25, det_eps=1e-8):
    """
    Compute convex brush vertices from plane triplets, then AABB.
    Robust to plane orientation by accepting either <= or >= convention.
    """
    m = normals.shape[0]
    if m < 4:
        return None

    pts_le = []
    pts_ge = []

    for i, j, k in combinations(range(m), 3):
        x = plane_triplet_intersection(normals[i], distances[i], normals[j], distances[j], normals[k], distances[k], det_eps)
        if x is None:
            continue
        v = normals.astype(np.float64) @ x
        if np.all(v <= distances.astype(np.float64) + inside_eps):
            pts_le.append(x)
        if np.all(v >= distances.astype(np.float64) - inside_eps):
            pts_ge.append(x)

    pts = pts_le if len(pts_le) >= len(pts_ge) else pts_ge
    if not pts:
        return None

    P = np.stack(pts, axis=0)
    return P.min(axis=0).astype(np.float32), P.max(axis=0).astype(np.float32)

def point_in_convex_brush(point, normals, distances, eps=0.25):
    """
    Robust to plane orientation: treat inside as whichever inequality fits better.
    For most Q3 BSPs, one of these will consistently work.
    """
    v = normals @ point
    inside_le = np.all(v <= distances + eps)
    inside_ge = np.all(v >= distances - eps)
    return inside_le or inside_ge

def plane_triplet_intersection(n1, d1, n2, d2, n3, d3, det_eps=1e-8):
    A = np.stack([n1, n2, n3], axis=0).astype(np.float64)
    b = np.array([d1, d2, d3], dtype=np.float64)
    if abs(np.linalg.det(A)) < det_eps:
        return None
    return np.linalg.solve(A, b)


def parse_origin(s):
    """
    "x y z" -> np.array([x,y,z], float32)
    """
    parts = s.split()
    if len(parts) != 3:
        return None
    try:
        return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
    except ValueError:
        return None

def get_spawn_origins(bsp):
    #ents = parse_entities(bsp.ENTITIES)
    ents = bsp.ENTITIES 
    spawn_classnames = {
        "info_player_deathmatch",
        "info_player_start",
        "info_player_intermission",
        "team_CTF_redspawn",
        "team_CTF_bluespawn",
        "team_ctf_redspawn",
        "team_ctf_bluespawn",
        # UrT-specific sometimes includes variations; add more if you find them
        "info_ut_spawn",
        "team_CTF_redflag",
        "team_CTF_blueflag"
    }

    seeds = []
    for e in ents:
        cn = e.get("classname", "")
        if cn in spawn_classnames:
            o = e.get("origin")
            if o:
                v = parse_origin(o)
                if v is not None:
                    seeds.append(v)

    return seeds



# -------------------------
# Point classification
# -------------------------

def point_inside_any(point, brush_set):
    for n, d in brush_set:
        if point_in_convex_brush(point, n, d):
            return True
    return False

def is_playable_point(point, solid_brushes, playerclip_brushes=None):
    # Must not be inside solid
    if point_inside_any(point, solid_brushes):
        return False
    # If you have playerclip, require being inside it (optional)
    if playerclip_brushes is not None and len(playerclip_brushes) > 0:
        if not point_inside_any(point, playerclip_brushes):
            return False
    return True

def is_air_point(point, blocking_brushes):
    return not point_inside_any(point, blocking_brushes)

def is_walkable_point(point, blocking_brushes, support_step=32.0):
    if not is_air_point(point, blocking_brushes):
        return False
    below = point - np.array([0.0, 0.0, support_step], dtype=np.float64)
    return point_inside_any(below, blocking_brushes)

def query_candidates(grid, point, cell=1024.0):
    c = tuple(np.floor(point / cell).astype(int))
    return grid.get(c, [])


def is_blocked_point(point, grid, cell_size, aabb_mins, aabb_maxs, normals_list, dists_list, eps=0.25, blocked_cache=None, cache_eps=None):
    """
    Test whether a point is inside any blocking brush. Optionally use `blocked_cache` to
    cache recent results. `cache_eps` controls the quantization radius used for cache keys
    (defaults to `eps` if not provided). When checking the cache we search neighbor buckets
    within +/-1 to allow an epsilon variance when matching.
    """
    px, py, pz = float(point[0]), float(point[1]), float(point[2])

    if blocked_cache is not None:
        if cache_eps is None:
            cache_eps = eps
        # quantize to bins of size cache_eps
        kx = int(round(px / cache_eps))
        ky = int(round(py / cache_eps))
        kz = int(round(pz / cache_eps))
        # check neighbors within one bin to allow variance
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (kx + dx, ky + dy, kz + dz)
                    if key in blocked_cache:
                        #print("blocked_cache hit:", key, "->", blocked_cache[key])
                        return blocked_cache[key]
    else:
        # ensure local variables exist for potential store below
        kx = ky = kz = None

    # no cache hit — compute normally
    for i in query_candidates(grid, point, cell=cell_size):
        mn = aabb_mins[i]; mx = aabb_maxs[i]
        # AABB early-out
        if (px < mn[0]-eps or py < mn[1]-eps or pz < mn[2]-eps or
            px > mx[0]+eps or py > mx[1]+eps or pz > mx[2]+eps):
            continue
        if point_in_convex_brush(point.astype(np.float32), normals_list[i], dists_list[i], eps=eps):
            # store result in cache if provided (store at central quantized bin)
            if blocked_cache is not None:
                blocked_cache[(kx, ky, kz)] = True
            return True

    # store negative result in cache if provided
    if blocked_cache is not None:
        blocked_cache[(kx, ky, kz)] = False
    return False

# -------------------------
# Filtering brushes
# -------------------------

def tex_name(bsp, tex_index):
    t = bsp.TEXTURES[tex_index]
    return t.name.decode("utf-8", errors="ignore").lower() if isinstance(t.name, (bytes, bytearray)) else str(t.name).lower()

def is_playerclip(bsp, brush):
    name = tex_name(bsp, brush.texture)
    return ("playerclip" in name) or (name.endswith("/clip")) or ("common/clip" in name)

def is_solid_world(bsp, brush):
    # This is the part you may need to adapt based on your Contents flags.
    # As a fallback, treat "nodraw" etc. as not solid; but best is contents flags.
    name = tex_name(bsp, brush.texture)
    if "playerclip" in name or "weapclip" in name:
        return False
    # Many real solids are regular textures; avoid excluding too much.
    return bool(bsp.TEXTURES[brush.texture].flags[1] & CONTENTS_SOLID)


# -------------------------
# Improve performance
# -------------------------

def build_spatial_hash(aabb_mins, aabb_maxs, cell=1024.0):
    grid = defaultdict(list)
    inv = 1.0 / cell
    for i, (mn, mx) in enumerate(zip(aabb_mins, aabb_maxs)):
        c0 = np.floor(mn * inv).astype(int)
        c1 = np.floor(mx * inv).astype(int)
        for x in range(c0[0], c1[0] + 1):
            for y in range(c0[1], c1[1] + 1):
                for z in range(c0[2], c1[2] + 1):
                    grid[(x, y, z)].append(i)
    return grid


def find_floor_z(x, y, z_start, z_min, blocked_fn, *, coarse=128.0, refine=8.0):
    """
    Finds the highest 'solid' point below z_start (approx floor).
    We want a transition from air -> blocked.
    Returns z_floor (solid-ish), or None.
    """
    z = z_start
    last_air = None
    #print("find_floor_z: x,y,z_start:", x, y, z_start, "z_min:", z_min)

    # march downward until we hit solid
    while z >= z_min:
        p = np.array([x, y, z], dtype=np.float32)
        if blocked_fn(p):
            # refine upward to find boundary
            z2 = z
            best = None
            # stop refining when we reach the last known air level (or the original start)
            upper = (last_air if last_air is not None else z_start)
            while z2 <= upper:
                p2 = np.array([x, y, z2], dtype=np.float32)
                if blocked_fn(p2):
                    best = z2
                z2 += refine
            return best
        last_air = z
        z -= coarse

    return None

def find_ceiling_z(x, y, z_start, z_max, blocked_fn, *, coarse=128.0, refine=8.0):
    """
    Finds the lowest 'solid' point above z_start (approx ceiling).
    We want a transition from air -> blocked above.
    Returns z_ceil (solid-ish), or None.
    """
    z = z_start
    last_air = None
    #print("Find ceiling from z_start:", z_start, "to z_max:", z_max)

    # march upward until we hit solid
    while z <= z_max:
        p = np.array([x, y, z], dtype=np.float32)
        if blocked_fn(p):
            # refine downward to find boundary
            z2 = z
            best = None
            # stop refining when we reach the last known air level (or the original start)
            lower = (last_air if last_air is not None else z_start)
            while z2 >= lower:
                p2 = np.array([x, y, z2], dtype=np.float32)
                if blocked_fn(p2):
                    best = z2
                z2 -= refine
            return best
        last_air = z
        z += coarse

    return None


def snap_to_standing_z(x, y, z_guess, *,
                       blocked_fn,
                       z_min, z_max,
                       stand_height=48.0,
                       player_height=72.0,
                       max_step_up=18.0,
                       max_step_down=64.0,
                       eps=1.0):
    """
    Returns z_origin for player (standing) if (x,y) has a reachable floor near z_guess.
    Otherwise returns None.
    """

    # Search for floor near the guess: allow stepping up/down a bit.
    # We probe from (z_guess + some headroom) downward.
    probe_top = min(z_guess + player_height, z_max)
    floor = find_floor_z(x, y, probe_top, z_min, blocked_fn, coarse=128.0, refine=8.0)
    if floor is None:
        return None

    z_origin = floor + stand_height

    # Step constraint relative to guess:
    dz = z_origin - z_guess
    if dz > max_step_up + eps:
        return None
    if dz < -max_step_down - eps:
        return None

    # Must be air at origin and at head
    p_origin = np.array([x, y, z_origin], dtype=np.float32)
    p_head   = np.array([x, y, z_origin + player_height], dtype=np.float32)
    if blocked_fn(p_origin) or blocked_fn(p_head):
        return None

    # Ceiling constraint: ceiling must be above head
    ceil = find_ceiling_z(x, y, z_origin, z_max, blocked_fn, coarse=128.0, refine=8.0)
    if ceil is not None and ceil <= z_origin + player_height + eps:
        return None

    return z_origin


def snap_to_standing_z_cached(ix, iy, iz_guess, *,
                              world_mins, voxel, nz,
                              blocked_fn, z_min, z_max,
                              cache: SnapCache,
                              stand_height=48.0, player_height=72.0,
                              max_step_up=18.0, max_step_down=64.0):
    hit = cache.get(ix, iy, iz_guess)
    if hit != "MISS":
        return hit  # may be None or z_origin

    # Convert cell indices to world coords
    x = float(world_mins[0] + (ix + 0.5) * voxel)
    y = float(world_mins[1] + (iy + 0.5) * voxel)
    z_guess = float(world_mins[2] + (iz_guess + 0.5) * voxel)

    z_origin = snap_to_standing_z(
        x, y, z_guess,
        blocked_fn=blocked_fn,
        z_min=z_min,
        z_max=z_max,
        stand_height=stand_height,
        player_height=player_height,
        max_step_up=max_step_up,
        max_step_down=max_step_down,
    )

    cache.set(ix, iy, iz_guess, z_origin)
    return z_origin


def world_to_cell(p, world_mins, voxel, nx, ny, nz):
    rel = (p - world_mins) / voxel
    ix, iy, iz = int(rel[0]), int(rel[1]), int(rel[2])
    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
        return ix, iy, iz
    return None

def cell_center(ix, iy, iz, world_mins, voxel):
    return world_mins + np.array([(ix + 0.5) * voxel, (iy + 0.5) * voxel, (iz + 0.5) * voxel], dtype=np.float32)

def z_to_iz(z, world_mins_z, voxel, nz):
    iz = int((z - world_mins_z) / voxel)
    return iz if 0 <= iz < nz else None


# main
def flood_fill_3d_from_spawns(bsp, blocked_fn, *,
                              voxel=64.0,
                              stand_height=48.0,
                              player_height=72.0,
                              max_step_up=18.0,
                              max_step_down=64.0):
    """
    Returns:
      visited: set of (ix,iy,iz) reachable standing cells
      aabb: (mins, maxs) of visited points (in world space, expanded by half voxel)
    """

    model0 = bsp.MODELS[0]
    world_mins = np.array(model0.bounds.mins, dtype=np.float32)
    world_maxs = np.array(model0.bounds.maxs, dtype=np.float32)

    dims = np.ceil((world_maxs - world_mins) / voxel).astype(int)
    nx, ny, nz = map(int, dims.tolist())
    print("World bounds:", world_mins, world_maxs)
    print("Voxel grid dims:", nx, ny, nz)

    seeds = get_spawn_origins(bsp)
    q = deque()
    visited = set()
    print("Seed spawns:", len(seeds))

    snap_cache = SnapCache(bucket_voxels=2)

    # Seed initialization: snap each spawn to a standing z at its (x,y)
    for s in seeds:
        print("Processing seed:", s)
        cell = world_to_cell(s, world_mins, voxel, nx, ny, nz)
        if cell is None:
            continue
        ix, iy, iz = cell
        c = cell_center(ix, iy, iz, world_mins, voxel)
        x, y, z_guess = float(c[0]), float(c[1]), float(c[2])

        # todo: snap_to_standing_z() taking way too long. even with cache still slow.
        # this is a little confusing - this actually checks that the player can stand at that location. And then gets the z.
        # also it gets a world z, not a cell z. which is confusing because snap makes it sound lik eit would return iz not z.
        z_cur = snap_to_standing_z_cached(
            ix, iy, iz,
            world_mins=world_mins, voxel=voxel, nz=nz,
            blocked_fn=blocked_fn, z_min=float(world_mins[2]), z_max=float(world_maxs[2]),
            cache=snap_cache,
            stand_height=stand_height, player_height=player_height,
            max_step_up=max_step_up, max_step_down=max_step_down
        )
        if z_cur is None:
            continue

        # z0 = snap_to_standing_z(
        #     x, y, z_guess,
        #     blocked_fn=blocked_fn,
        #     z_min=float(world_mins[2]),
        #     z_max=float(world_maxs[2]),
        #     stand_height=stand_height,
        #     player_height=player_height,
        #     max_step_up=max_step_up,
        #     max_step_down=max_step_down,
        # )
        # if z0 is None:
        #     continue

        iz0 = z_to_iz(z_cur, float(world_mins[2]), voxel, nz)
        if iz0 is None:
            continue

        state = (ix, iy, iz0)
        if state not in visited:
            visited.add(state)
            q.append(state)

    # note - the initial q/visited arrays are smaller than expected. But I think it might be because, depending on grid size, 
    # you might have multiple spawns in the same cell, so they get de-duped.
    print("Initial reachable cells:", len(visited))
    #sys.exit(0)

    # 6-neighbor expansion in grid
    nbrs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    # todo: it seems like this loops over each spawn, and finds just the immediate neighbors? It should continue expanding.
    # oh no wait, it is continually adding to q, so it does continue expanding.
    while q:
        ix, iy, iz = q.popleft()
        c = cell_center(ix, iy, iz, world_mins, voxel)
        x0, y0, z_guess0 = float(c[0]), float(c[1]), float(c[2])
        print("Evaluating cell:", (ix, iy, iz), "visited:", len(visited))

        # Snap current to get a stable z_guess for neighbors (optional but helps)
        z_current = snap_to_standing_z(
            x0, y0, z_guess0,
            blocked_fn=blocked_fn,
            z_min=float(world_mins[2]),
            z_max=float(world_maxs[2]),
            stand_height=stand_height,
            player_height=player_height,
            max_step_up=max_step_up,
            max_step_down=max_step_down,
        )
        if z_current is None:
            continue

        for dx, dy, dz in nbrs:
            ix2, iy2, iz2 = ix + dx, iy + dy, iz + dz
            if not (0 <= ix2 < nx and 0 <= iy2 < ny and 0 <= iz2 < nz):
                continue

            c2 = cell_center(ix2, iy2, iz2, world_mins, voxel)
            x2, y2 = float(c2[0]), float(c2[1])

            # Neighbor z guess: keep close to current snapped z
            z_guess2 = z_current + dz * voxel

            # todo optimize>? - we are calling this here when we add to q, but then again above when we pop and process q
            z2 = snap_to_standing_z(
                x2, y2, z_guess2,
                blocked_fn=blocked_fn,
                z_min=float(world_mins[2]),
                z_max=float(world_maxs[2]),
                stand_height=stand_height,
                player_height=player_height,
                max_step_up=max_step_up,
                max_step_down=max_step_down,
            )
            if z2 is None:
                continue

            iz_snapped = z_to_iz(z2, float(world_mins[2]), voxel, nz)
            if iz_snapped is None:
                continue

            st2 = (ix2, iy2, iz_snapped)
            if st2 in visited:
                continue
            visited.add(st2)
            q.append(st2)

    # Build AABB in world space
    if not visited:
        return visited, None

    half = 0.5 * voxel
    xs, ys, zs = [], [], []
    for (ix, iy, iz) in visited:
        p = cell_center(ix, iy, iz, world_mins, voxel)
        xs.append(float(p[0])); ys.append(float(p[1])); zs.append(float(p[2]))

    mn = np.array([min(xs) - half, min(ys) - half, min(zs) - half], dtype=np.float32)
    mx = np.array([max(xs) + half, max(ys) + half, max(zs) + half], dtype=np.float32)
    return visited, (mn, mx)


def make_blocked_fn(bsp):
    # Bounds from model 0
    model0 = bsp.MODELS[0]
    world_mins = np.array(model0.bounds.mins, dtype=np.float32)
    world_maxs = np.array(model0.bounds.maxs, dtype=np.float32)

    # 1) Compile blocking brushes + precompute planes + AABBs
    normals_list = []
    dists_list = []
    aabb_mins = []
    aabb_maxs = []

    for br in bsp.BRUSHES:
        if (not is_playerclip(bsp, br)) and (not is_solid_world(bsp, br)):
            continue
        n, d = brush_planes(bsp, br)
        aabb = brush_aabb_from_planes(n, d)
        if aabb is None:
            continue
        mn, mx = aabb
        normals_list.append(n.astype(np.float32))
        dists_list.append(d.astype(np.float32))
        aabb_mins.append(mn)
        aabb_maxs.append(mx)

    if not aabb_mins:
        return []

    aabb_mins = np.stack(aabb_mins, axis=0)
    aabb_maxs = np.stack(aabb_maxs, axis=0)

    # 2) Spatial hash broadphase
    grid = build_spatial_hash(aabb_mins, aabb_maxs, cell=VOXEL)

    # Cache for is_blocked_point results. Keys are quantized bins (kx,ky,kz) -> bool
    blocked_cache = {}
    cache_eps = 0.25

    def blocked_fn(p):
        return is_blocked_point(p, grid, VOXEL, aabb_mins, aabb_maxs, normals_list, dists_list, eps=cache_eps, blocked_cache=blocked_cache, cache_eps=cache_eps)

    return blocked_fn


test_bsp = Path("C:\\repos\\mapindexer\\maps\\processed\\layout_del_1\\maps\\layout_del_1.bsp")
bsp = load_bsp(str(test_bsp))

# 1) Build blocking query (your existing optimized brush precompute)
blocked_fn = make_blocked_fn(bsp)  # you implement using spatial hash + point-in-brush

# 2) Flood fill from spawns (3D, on-demand)
visited, aabb = flood_fill_3d_from_spawns(
    bsp, blocked_fn,
    voxel=64.0,
    stand_height=48.0,
    player_height=72.0,
    max_step_up=18.0,
    max_step_down=64.0,
)

print("reachable cells:", len(visited))
print("reachable AABB:", aabb)



###
# this is doing something but takes a long time.
# snap_to_standing_z() taking way too long. even with cache still slow. investigate.
##

## somehow z is exceeding z_max
## exmaples:
#blocked_cache hit: (2696, 1288, 141304992) -> False
#blocked_cache hit: (2696, 1288, 141305024) -> False

## ok the snap_to_standing() issue is resolved now, it had an infinite loop now fixed.
# now need to continue a full test run see what hapen
