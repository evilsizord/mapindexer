from pathlib import Path
import numpy as np
from math import atan2, degrees
from bsp_tool import load_bsp
from collections import deque, defaultdict
from itertools import combinations

CONTENTS_SOLID     = 0x00000001

test_bsp = Path("C:\\repos\\mapindexer\\maps\\processed\\layout_del_1\\maps\\layout_del_1.bsp")

bsp = load_bsp(str(test_bsp))



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

def compile_brush_sets(bsp):
    solid = []
    clip = []
    for br in bsp.BRUSHES:
        n, d = brush_planes(bsp, br)
        if n.shape[0] < 4:  # too few planes to form a volume
            continue
        if is_playerclip(bsp, br):
            clip.append((n, d))
        elif is_solid_world(bsp, br):
            solid.append((n, d))
    return solid, clip

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

def query_cell_candidates(grid, point, cell=512.0, neighbor_radius=0):
    c = tuple(np.floor(point / cell).astype(int))
    if neighbor_radius == 0:
        return grid.get(c, [])
    out = []
    rx = range(c[0]-neighbor_radius, c[0]+neighbor_radius+1)
    ry = range(c[1]-neighbor_radius, c[1]+neighbor_radius+1)
    rz = range(c[2]-neighbor_radius, c[2]+neighbor_radius+1)
    for x in rx:
        for y in ry:
            for z in rz:
                out.extend(grid.get((x,y,z), []))
    return out

def query_candidates(grid, point, cell=1024.0):
    c = tuple(np.floor(point / cell).astype(int))
    return grid.get(c, [])


def is_blocked_point(point, grid, cell_size, aabb_mins, aabb_maxs, normals_list, dists_list, eps=0.25):
    px, py, pz = float(point[0]), float(point[1]), float(point[2])
    for i in query_candidates(grid, point, cell=cell_size):
        mn = aabb_mins[i]; mx = aabb_maxs[i]
        # AABB early-out
        if (px < mn[0]-eps or py < mn[1]-eps or pz < mn[2]-eps or
            px > mx[0]+eps or py > mx[1]+eps or pz > mx[2]+eps):
            continue
        if point_in_convex_brush(point.astype(np.float32), normals_list[i], dists_list[i], eps=eps):
            return True
    return False

def has_clearance(p_stand, blocked_fn, player_height=64.0):
    p_head = p_stand + np.array([0.0, 0.0, player_height], dtype=np.float32)
    return (not blocked_fn(p_stand)) and (not blocked_fn(p_head))


def origin_to_cell(origin, world_mins, xy_step, nx, ny):
    ix = int((origin[0] - world_mins[0]) / xy_step)
    iy = int((origin[1] - world_mins[1]) / xy_step)
    if 0 <= ix < nx and 0 <= iy < ny:
        return ix, iy
    return None

def nearest_valid_cell(ix, iy, valid, max_r=6):
    """
    If the seed cell isn't valid, find the nearest valid cell within a small radius.
    """
    nx, ny = valid.shape
    if 0 <= ix < nx and 0 <= iy < ny and valid[ix, iy]:
        return ix, iy

    for r in range(1, max_r + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x = ix + dx
                y = iy + dy
                if 0 <= x < nx and 0 <= y < ny and valid[x, y]:
                    return x, y
    return None


def flood_fill_from_spawns(valid, nodes_z, seed_cells, connect_step_height=80.0):
    """
    valid: (nx,ny) uint8/bool
    nodes_z: (nx,ny) float (nan where invalid)
    seed_cells: list of (ix,iy) already snapped to valid cells
    returns: reachable mask (nx,ny) uint8
    """
    nx, ny = valid.shape
    reachable = np.zeros((nx, ny), dtype=np.uint8)
    q = deque()

    for (sx, sy) in seed_cells:
        if valid[sx, sy] and not reachable[sx, sy]:
            reachable[sx, sy] = 1
            q.append((sx, sy))

    nbrs = [(1,0),(-1,0),(0,1),(0,-1)]

    while q:
        x, y = q.popleft()
        z0 = float(nodes_z[x, y])

        for dx, dy in nbrs:
            x2, y2 = x + dx, y + dy
            if 0 <= x2 < nx and 0 <= y2 < ny and valid[x2, y2] and not reachable[x2, y2]:
                z1 = float(nodes_z[x2, y2])
                if abs(z1 - z0) <= connect_step_height:
                    reachable[x2, y2] = 1
                    q.append((x2, y2))

    return reachable


# ---------------------------
# Downward "floor probe" per (x,y)
# ---------------------------

def find_floor_z(x, y, z_top, z_bottom, *, blocked_fn, coarse_step=256.0, refine_step=16.0):
    """
    Returns an estimated floor height z where space switches from air->blocked.
    We search downward: find first blocked point, then refine around boundary.
    """
    z = z_top
    last_air = None
    first_block = None

    # Coarse march
    while z >= z_bottom:
        p = np.array([x, y, z], dtype=np.float32)
        if blocked_fn(p):
            first_block = z
            break
        last_air = z
        z -= coarse_step

    if first_block is None:
        return None  # never hit ground
    if last_air is None:
        # started in solid; move up a bit or treat as invalid
        return None

    # Refine boundary between last_air (air) and first_block (blocked)
    lo = first_block
    hi = last_air
    # Ensure lo <= hi in z-space? Here lo is lower, hi is higher (air)
    # We'll scan upward from blocked to air
    z = lo
    best = None
    while z <= hi:
        p = np.array([x, y, z], dtype=np.float32)
        if blocked_fn(p):
            best = z
        z += refine_step

    return best  # approximate "solid" height



# ---------------------------
# 2D grid sampling -> nodes -> flood fill -> AABBs
# ---------------------------

def compute_playable_sections_2p5d(bsp, *, xy_step=96.0, stand_height=48.0,
                                  cell_size=1024.0, floor_coarse_step=256.0,
                                  floor_refine_step=16.0, connect_step_height=80.0,
                                  eps=0.25):
    """
    Returns list of (mins, maxs, num_nodes) for each playable "section".
    """

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
    grid = build_spatial_hash(aabb_mins, aabb_maxs, cell=cell_size)

    def blocked_fn(p):
        return is_blocked_point(p, grid, cell_size, aabb_mins, aabb_maxs, normals_list, dists_list, eps=eps)

    # 3) Sample XY grid, find floor, create a node at z=floor+stand_height
    nx = int(np.ceil((world_maxs[0] - world_mins[0]) / xy_step))
    ny = int(np.ceil((world_maxs[1] - world_mins[1]) / xy_step))

    nodes_z = np.full((nx, ny), np.nan, dtype=np.float32)
    valid = np.zeros((nx, ny), dtype=np.uint8)

    z_top = float(world_maxs[2])
    z_bottom = float(world_mins[2])

    for ix in range(nx):
        x = float(world_mins[0] + (ix + 0.5) * xy_step)
        for iy in range(ny):
            y = float(world_mins[1] + (iy + 0.5) * xy_step)
            floor_z = find_floor_z(
                x, y, z_top, z_bottom,
                blocked_fn=blocked_fn,
                coarse_step=floor_coarse_step,
                refine_step=floor_refine_step,
            )
            if floor_z is None:
                continue
            z_stand = floor_z + stand_height

            # sanity: stand point should be in air
            p_stand = np.array([x, y, z_stand], dtype=np.float32)
            if not has_clearance(p_stand, blocked_fn, player_height=64.0): 
                continue

            nodes_z[ix, iy] = z_stand
            valid[ix, iy] = 1


    #NEW
    nx, ny = valid.shape
    spawn_origins = get_spawn_origins(bsp)
    print("Found", len(spawn_origins), "spawn origins for flood fill seeds")

    seed_cells = []
    for o in spawn_origins:
        cell = origin_to_cell(o, world_mins, xy_step, nx, ny)
        if cell is None:
            print("Spawn origin", o, "out of grid bounds")
            continue
        snapped = nearest_valid_cell(cell[0], cell[1], valid, max_r=8)
        print("Spawn origin", o, "mapped to cell", cell, "snapped to", snapped)
        if snapped is not None:
            seed_cells.append(snapped)

    #debug
    print("Using", len(seed_cells), "seed cells for flood fill")

    if seed_cells:
        reachable = flood_fill_from_spawns(valid, nodes_z, seed_cells, connect_step_height=connect_step_height)
        # Keep only reachable nodes
        valid = (valid & reachable).astype(np.uint8)
    else:
        # No seeds found — fallback: keep everything or choose largest component later
        pass
    #/NEW

    # 4) Flood fill components in 2D with height tolerance
    visited = np.zeros_like(valid)
    comps = []
    nbrs = [(1,0),(-1,0),(0,1),(0,-1)]

    for ix in range(nx):
        for iy in range(ny):
            if not valid[ix, iy] or visited[ix, iy]:
                continue
            q = deque([(ix, iy)])
            visited[ix, iy] = 1
            comp = []

            while q:
                x, y = q.popleft()
                comp.append((x, y))
                z0 = float(nodes_z[x, y])
                for dx, dy in nbrs:
                    x2, y2 = x + dx, y + dy
                    if 0 <= x2 < nx and 0 <= y2 < ny and valid[x2, y2] and not visited[x2, y2]:
                        z1 = float(nodes_z[x2, y2])
                        if abs(z1 - z0) <= connect_step_height:
                            visited[x2, y2] = 1
                            q.append((x2, y2))

            comps.append(comp)

    # 5) Component AABBs in world space
    out = []
    half = 0.5 * xy_step
    for comp in comps:
        xs = []
        ys = []
        zs = []
        for ix, iy in comp:
            x = float(world_mins[0] + (ix + 0.5) * xy_step)
            y = float(world_mins[1] + (iy + 0.5) * xy_step)
            z = float(nodes_z[ix, iy])
            xs.append(x); ys.append(y); zs.append(z)

        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        zs = np.array(zs, dtype=np.float32)

        mn = np.array([xs.min() - half, ys.min() - half, zs.min() - stand_height], dtype=np.float32)
        mx = np.array([xs.max() + half, ys.max() + half, zs.max() + stand_height], dtype=np.float32)
        out.append((mn, mx, len(comp)))

    # sort largest first
    out.sort(key=lambda t: float(np.prod(t[1] - t[0])), reverse=True)
    return out





# -------------------------
# Voxelization + flood fill
# -------------------------

def voxelize_playable(bsp, world_mins, world_maxs, voxel=64.0, use_playerclip=True):
    solid, clip = compile_brush_sets(bsp)
    clip_set = clip if use_playerclip else None

    blocking = solid + clip_set  # treat playerclip as blocking

    world_mins = np.asarray(world_mins, dtype=np.float64)
    world_maxs = np.asarray(world_maxs, dtype=np.float64)

    dims = np.ceil((world_maxs - world_mins) / voxel).astype(int)
    nx, ny, nz = dims.tolist()

    playable = np.zeros((nx, ny, nz), dtype=np.uint8)

    # Sample at voxel centers
    for ix in range(nx):
        x = world_mins[0] + (ix + 0.5) * voxel
        for iy in range(ny):
            y = world_mins[1] + (iy + 0.5) * voxel
            for iz in range(nz):
                z = world_mins[2] + (iz + 0.5) * voxel
                p = np.array([x, y, z], dtype=np.float64)
                if is_walkable_point(p, blocking, support_step=32.0):
                     playable[ix, iy, iz] = 1
                #if is_playable_point(p, solid, clip_set):
                #    playable[ix, iy, iz] = 1

    return playable, world_mins, voxel

def flood_fill_components(playable):
    nx, ny, nz = playable.shape
    visited = np.zeros_like(playable, dtype=np.uint8)
    comps = []

    nbrs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if playable[ix,iy,iz] == 0 or visited[ix,iy,iz]:
                    continue
                q = deque()
                q.append((ix,iy,iz))
                visited[ix,iy,iz] = 1
                voxels = []

                while q:
                    x,y,z = q.popleft()
                    voxels.append((x,y,z))
                    for dx,dy,dz in nbrs:
                        x2,y2,z2 = x+dx, y+dy, z+dz
                        if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz:
                            if playable[x2,y2,z2] and not visited[x2,y2,z2]:
                                visited[x2,y2,z2] = 1
                                q.append((x2,y2,z2))

                comps.append(voxels)

    return comps

def component_aabbs_from_voxels(components, origin, voxel):
    out = []
    origin = np.asarray(origin, dtype=np.float64)

    for voxels in components:
        arr = np.asarray(voxels, dtype=np.int32)
        vmin = arr.min(axis=0)
        vmax = arr.max(axis=0)

        # Convert voxel index range to world AABB (full voxel extents)
        mins = origin + vmin * voxel
        maxs = origin + (vmax + 1) * voxel
        out.append((mins, maxs, len(voxels)))

    return out


def centroid(mins, maxs):
    return (mins + maxs) / 2.0


sections = compute_playable_sections_2p5d(
    bsp,
    xy_step=64.0,              # bigger = faster / coarser
    stand_height=48.0,         # approx player origin above floor
    cell_size=512.0,          # spatial hash cell size
    floor_coarse_step=128.0,   # coarse downward marching step
    floor_refine_step=16.0,    # refine near floor
    connect_step_height=80.0,  # allow stairs/ramps
    eps=0.25
)

print(f"Playable sections found: {len(sections)}")
for i, (mn, mx, n) in enumerate(sections[:10]):
    size = mx - mn
    vol = float(size[0] * size[1] * size[2])
    cent = centroid(mn, mx)
    print(f"[{i}] nodes={n:5d}  mins={mn}  maxs={mx}  cent={cent}  volume={vol:.0f}")


# inspect some brushes
# for bi, br in enumerate(bsp.BRUSHES[0:10]):
#     n, d = brush_planes(bsp, br)
#     aabb = brush_aabb_from_planes(n, d)
#     if aabb is None:
#         print(f"Brush {br} could not compute AABB from planes!")
#     else:
#         mn, mx = aabb
#         print(f"Brush {br} AABB: mins={mn}, maxs={mx}")
#         print(f"Tex name: {tex_name(bsp, br.texture)}")
#         print(f"Is playerclip: {is_playerclip(bsp, br)}, is solid world: {is_solid_world(bsp, br)}")


# world = bsp.MODELS[0]
# world_mins = np.array(world.bounds.mins, dtype=np.float64)  # adjust based on how bsp_tool exposes these
# world_maxs = np.array(world.bounds.maxs, dtype=np.float64)

# playable_grid, origin, vox = voxelize_playable(bsp, world_mins, world_maxs, voxel=64.0, use_playerclip=True)

# print(f"Voxel grid shape: {playable_grid.shape}, origin: {origin}, voxel size: {vox}")


#components = flood_fill_components(playable_grid)
#section_aabbs = component_aabbs_from_voxels(components, origin, vox)

# sort by volume
#section_aabbs.sort(key=lambda t: float(np.prod(t[1]-t[0])), reverse=True)

# print(f"Found {len(section_aabbs)} playable sections:")
# print("Example: largest section AABB (mins, maxs, num_voxels):")
# if section_aabbs:
#     mins, maxs, num_voxels = section_aabbs[0]
#     print(f"  {mins}, {maxs}, {num_voxels}")

