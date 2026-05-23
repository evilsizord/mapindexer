from pathlib import Path
import time
import os
import random
import numpy as np
# Allow forcing Numba off via environment variable `MAPINDEXER_NO_NUMBA=1`
_NUMBA_AVAILABLE = False
if os.environ.get("MAPINDEXER_NO_NUMBA", "0") != "1":
    try:
        from numba import njit
        _NUMBA_AVAILABLE = True
    except Exception:
        _NUMBA_AVAILABLE = False
from bsp_tool import load_bsp
from collections import deque, defaultdict
from itertools import combinations

## In this version, trying a 3d grid sampling approach instead of 2.5d

CONTENTS_SOLID     = 0x00000001
VOXEL = 32.0            # grid resolution
PLAYER_HEIGHT = 72.0    # player hull height for clearance checks


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
    # If Numba is available, use the compiled implementation which is faster
    if _NUMBA_AVAILABLE:
        return _point_in_convex_brush_numba(point.astype(np.float32), normals.astype(np.float32), distances.astype(np.float32), float(eps))

    v = normals @ point
    inside_le = np.all(v <= distances + eps)
    inside_ge = np.all(v >= distances - eps)
    return inside_le or inside_ge

# Numba gives small savings in the blocked function time but overall runtime difference is small (~0.16s).. :-/
if _NUMBA_AVAILABLE:
    @njit
    def _point_in_convex_brush_numba(point, normals, distances, eps):
        m = normals.shape[0]
        # compute v = normals @ point for each plane
        inside_le = True
        for i in range(m):
            v = normals[i, 0] * point[0] + normals[i, 1] * point[1] + normals[i, 2] * point[2]
            if not (v <= distances[i] + eps):
                inside_le = False
                break

        inside_ge = True
        for i in range(m):
            v = normals[i, 0] * point[0] + normals[i, 1] * point[1] + normals[i, 2] * point[2]
            if not (v >= distances[i] - eps):
                inside_ge = False
                break

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



def query_candidates(grid, point, cell=1024.0):
    c = tuple(np.floor(point / cell).astype(int))
    return grid.get(c, [])


def is_blocked_point(point, grid, cell_size, aabb_mins, aabb_maxs, normals_list, dists_list, eps=0.25, blocked_cache=None, cache_eps=None):
    """
    Test whether a point is inside any blocking brush. Optionally use `blocked_cache` to
    cache recent results. `cache_eps` controls the quantization radius used for cache keys
    (defaults to `eps` if not provided).
    """
    px, py, pz = float(point[0]), float(point[1]), float(point[2])

    if blocked_cache is not None:
        if cache_eps is None:
            cache_eps = eps
        # quantize to bins of size cache_eps
        kx = int(round(px / cache_eps))
        ky = int(round(py / cache_eps))
        kz = int(round(pz / cache_eps))
        key = (kx, ky, kz)
        if key in blocked_cache:
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
    """
    Check if a brush is playerclip by examining all its sides.
    Returns True if ANY side has a playerclip texture.
    """
    for side_idx in range(brush.first_side, brush.first_side + brush.num_sides):
        side = bsp.BRUSH_SIDES[side_idx]
        name = tex_name(bsp, side.texture)
        if ("playerclip" in name) or (name.endswith("/clip")) or ("common/clip" in name):
            return True
    return False

def is_solid_world(bsp, brush):
    """
    Check if brush is world solid by examining all its sides.
    Excludes playerclip and weapclip brushes.
    Returns True if ANY side has CONTENTS_SOLID and no side is clip.
    """
    # First check: exclude clip brushes
    for side_idx in range(brush.first_side, brush.first_side + brush.num_sides):
        side = bsp.BRUSH_SIDES[side_idx]
        name = tex_name(bsp, side.texture)
        if "playerclip" in name or "weapclip" in name or "clip" in name:
            return False
    # Check if any side is solid
    for side_idx in range(brush.first_side, brush.first_side + brush.num_sides):
        side = bsp.BRUSH_SIDES[side_idx]
        if bsp.TEXTURES[side.texture].flags[1] & CONTENTS_SOLID:
            return True
    return False


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


def point_to_leaf_index(bsp, point):
    """
    Traverse the BSP tree and return the leaf containing point.
    Q3 children use negative indexes encoded as -leaf_index - 1.
    """
    idx = 0
    while idx >= 0:
        node = bsp.NODES[idx]
        plane = bsp.PLANES[node.plane]
        d = (
            plane.normal.x * float(point[0]) +
            plane.normal.y * float(point[1]) +
            plane.normal.z * float(point[2]) -
            plane.distance
        )
        idx = node.children.front if d >= 0 else node.children.back

    leaf_index = -idx - 1
    if 0 <= leaf_index < len(bsp.LEAVES):
        return leaf_index
    return None


def point_in_playable_leaf(bsp, point):
    leaf_index = point_to_leaf_index(bsp, point)
    return leaf_index is not None and bsp.LEAVES[leaf_index].cluster >= 0


def has_fly_clearance(bsp, point, *, blocked_fn, player_height=72.0):
    """
    Validate a player-origin point for fly/noclip-style reachability.
    No floor is required; the origin, mid-body, and head must stay in air leafs.
    """
    p_origin = np.asarray(point, dtype=np.float32)
    sample_offsets = (0.0, player_height * 0.5, player_height)

    for dz in sample_offsets:
        p = p_origin + np.array([0.0, 0.0, dz], dtype=np.float32)
        if blocked_fn(p):
            return False
        if not point_in_playable_leaf(bsp, p):
            return False

    return True


def can_fly_between(bsp, p0, p1, *,
                    blocked_fn,
                    player_height=72.0,
                    sample_step=8.0):
    """
    Approximate a swept fly movement by sampling the player hull along a voxel edge.
    """
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    dist = float(np.linalg.norm(p1 - p0))
    steps = max(2, int(np.ceil(dist / sample_step)) + 1)

    for i in range(steps + 1):
        t = i / steps
        p = p0 + (p1 - p0) * t
        if not has_fly_clearance(
            bsp, p,
            blocked_fn=blocked_fn,
            player_height=player_height,
        ):
            return False

    return True


def world_to_cell(p, world_mins, voxel, nx, ny, nz):
    rel = (p - world_mins) / voxel
    ix, iy, iz = int(rel[0]), int(rel[1]), int(rel[2])
    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
        return ix, iy, iz
    return None

def cell_center(ix, iy, iz, world_mins, voxel):
    return world_mins + np.array([(ix + 0.5) * voxel, (iy + 0.5) * voxel, (iz + 0.5) * voxel], dtype=np.float32)


def flood_fill_flyable_volume_from_spawns(bsp, blocked_fn, *,
                                           voxel=64.0,
                                           player_height=72.0):
    """
    Returns:
      visited: set of (ix,iy,iz) reachable fly-space cells
      aabb: (mins, maxs) of visited points (in world space, expanded by half voxel)
    """
    start_time = time.perf_counter()

    model0 = bsp.MODELS[0]
    world_mins = np.array(model0.bounds.mins, dtype=np.float32)
    world_maxs = np.array(model0.bounds.maxs, dtype=np.float32)

    dims = np.ceil((world_maxs - world_mins) / voxel).astype(int)
    nx, ny, nz = map(int, dims.tolist())
    print("World bounds:", world_mins, world_maxs)
    print("Voxel grid dims:", nx, ny, nz)

    seeds = get_spawn_origins(bsp)
    q = deque()
    
    enqueued = set()
    print("Seed spawns:", len(seeds))

    clearance_cache = {}

    def cell_has_fly_clearance(ix, iy, iz):
        state = (ix, iy, iz)
        if state in clearance_cache:
            return clearance_cache[state]
        p = cell_center(ix, iy, iz, world_mins, voxel)
        ok = has_fly_clearance(
            bsp, p,
            blocked_fn=blocked_fn,
            player_height=player_height,
        )
        clearance_cache[state] = ok
        return ok

    def enqueue_seed_cell(ix, iy, iz):
        state = (ix, iy, iz)
        if state in enqueued:
            return False
        if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
            return False
        if not cell_has_fly_clearance(ix, iy, iz):
            return False
        enqueued.add(state)
        q.append(state)
        return True

    # Seed initialization: use spawn origins as air-volume seeds. If the voxel center
    # nearest a spawn is clipped, search a tiny radius for a valid air cell.
    for s in seeds:
        cell = world_to_cell(s, world_mins, voxel, nx, ny, nz)
        if cell is None:
            continue
        ix, iy, iz = cell
        if enqueue_seed_cell(ix, iy, iz):
            continue

        seeded = False
        for r in range(1, 3):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if enqueue_seed_cell(ix + dx, iy + dy, iz + dz):
                            seeded = True
                            break
                    if seeded:
                        break
                if seeded:
                    break
            if seeded:
                break

    print("Initial reachable cells:", len(enqueued))

    # 6-neighbor expansion through connected flyable air volume.
    nbrs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    visited = set()

    # explore nodes in queue. Neighbor nodes are added and explored in turn, until queu is exhausted.
    while q:
        ix, iy, iz = q.popleft()

        if not cell_has_fly_clearance(ix, iy, iz):
            continue

        st = (ix, iy, iz)
        if st in visited:
            continue
        #print("Adding reachable cell:", st)
        visited.add(st)

        # add neighbors to explore queue
        for dx, dy, dz in nbrs:
            ix2, iy2, iz2 = ix + dx, iy + dy, iz + dz
            if not (0 <= ix2 < nx and 0 <= iy2 < ny and 0 <= iz2 < nz):
                continue

            st2 = (ix2, iy2, iz2)
            if st2 in visited or st2 in enqueued:
                continue
            if not cell_has_fly_clearance(ix2, iy2, iz2):
                continue

            p0 = cell_center(ix, iy, iz, world_mins, voxel)
            p1 = cell_center(ix2, iy2, iz2, world_mins, voxel)
            if not can_fly_between(
                bsp, p0, p1,
                blocked_fn=blocked_fn,
                player_height=player_height,
                sample_step=max(4.0, voxel / 4.0),
            ):
                continue
            #print("Adding reachable cell:", st2)
            enqueued.add(st2)
            q.append(st2)

    # Build AABB in world space
    if not visited:
        return visited, None

    half = 0.5 * voxel
    xs, ys, zs = [], [], []
    for (ix, iy, iz) in visited:
        p = cell_center(ix, iy, iz, world_mins, voxel)
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))

    mn = np.array([min(xs) - half, min(ys) - half, min(zs) - half], dtype=np.float32)
    mx = np.array([max(xs) + half, max(ys) + half, max(zs) + half], dtype=np.float32)
    mn = np.maximum(mn, world_mins)
    mx = np.minimum(mx, world_maxs)
    
    elapsed = time.perf_counter() - start_time
    print(f"flood_fill_flyable_volume_from_spawns completed in {elapsed:.2f} seconds")
    
    return visited, (mn, mx)


def make_blocked_fn(bsp, *, voxel=64.0):
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
    grid_cell = voxel * 4
    grid = build_spatial_hash(aabb_mins, aabb_maxs, cell=grid_cell)

    # Cache for is_blocked_point results. Keys are quantized bins (kx,ky,kz) -> bool
    blocked_cache = {}
    # Use a coarser cache epsilon (in world units) to increase reuse across nearby probes
    cache_eps = max(0.25, voxel / 8.0)  # e.g. 8.0 for voxel=64

    # lightweight stats for profiling call counts and cumulative time
    stats = {"calls": 0, "time": 0.0}

    def blocked_fn(p):
        stats["calls"] += 1
        t0 = time.perf_counter()
        res = is_blocked_point(p, grid, grid_cell, aabb_mins, aabb_maxs, normals_list, dists_list, eps=cache_eps, blocked_cache=blocked_cache, cache_eps=cache_eps)
        stats["time"] += (time.perf_counter() - t0)
        return res

    # attach stats object to the function for external inspection
    blocked_fn._stats = stats

    return blocked_fn


def print_debug_cells(visited, world_mins, voxel):
    print("reachable cells:", len(visited))
    if not visited:
        print("No reachable cells found.")
        return

    vlist = list(visited)
    x, y, z = vlist[0]
    print("Example cell 1:", cell_center(x, y, z, world_mins, voxel))
    for i, (rx, ry, rz) in enumerate(random.sample(vlist, min(4, len(vlist))), start=1):
        print(f"Random cell {i}:", (rx, ry, rz), "world coords:", cell_center(rx, ry, rz, world_mins, voxel))

    max_z_cell = max(visited, key=lambda c: c[2])
    print("Cell with max z:", max_z_cell, "world coords:", cell_center(max_z_cell[0], max_z_cell[1], max_z_cell[2], world_mins, voxel))

    min_z_cell = min(visited, key=lambda c: c[2])
    print("Cell with min z:", min_z_cell, "world coords:", cell_center(min_z_cell[0], min_z_cell[1], min_z_cell[2], world_mins, voxel))


def plot_visited_cells(visited, world_mins, voxel):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    xs, ys, zs = [], [], []
    for (ix, iy, iz) in visited:
        p = cell_center(ix, iy, iz, world_mins, voxel)
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c="blue", marker="o")
    plt.show()


def main():
    test_bsp = Path("C:\\repos\\mapindexer\\maps\\processed\\layout_del_1\\maps\\layout_del_1.bsp")
    bsp = load_bsp(str(test_bsp))

    blocked_fn = make_blocked_fn(bsp, voxel=VOXEL)
    visited, aabb = flood_fill_flyable_volume_from_spawns(
        bsp, blocked_fn,
        voxel=VOXEL,
        player_height=PLAYER_HEIGHT,
    )

    model0 = bsp.MODELS[0]
    world_mins = np.array(model0.bounds.mins, dtype=np.float32)

    print("reachable AABB:", aabb)
    print_debug_cells(visited, world_mins, VOXEL)
    # plot_visited_cells(visited, world_mins, VOXEL)


if __name__ == "__main__":
    main()


