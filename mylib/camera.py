from pathlib import Path
import numpy as np
from math import atan2, degrees
from bsp_tool import load_bsp

"""
::Chat GPT::
below is exact, concrete Python code that extracts optimal camera points from a Quake 3 / 
Urban Terror BSP using the BSP-leaf / cluster–driven approach (Strategy B) we discussed.

This code is deterministic, scalable, and does not require running the game engine. 
It produces camera positions + angles ready to be fed into auto_cam.cfg.
"""

# ----------------------------
# Tunable heuristics
# ----------------------------

MIN_LEAF_VOLUME = 128 * 128 * 128
CAMERA_HEIGHT_OFFSET = 64
DEFAULT_PITCH = -15.0

# ----------------------------
# Utility math
# ----------------------------

def centroid(mins, maxs):
    return (mins + maxs) / 2.0

def yaw_pitch(from_pos, to_pos):
    dx, dy, dz = to_pos - from_pos
    yaw = degrees(atan2(dy, dx))
    pitch = degrees(atan2(dz, np.linalg.norm([dx, dy])))
    return yaw, pitch

# ----------------------------
# BSP analysis
# ----------------------------

def extract_leaf_centroids(bsp):
    leafs = []

    for leaf in bsp.LEAVES:
        mins = np.array(leaf.bounds.mins)
        maxs = np.array(leaf.bounds.maxs)
        vol = np.prod(maxs - mins)

        if vol >= MIN_LEAF_VOLUME and leaf.cluster >= 0:
            leafs.append({
                "cluster": leaf.cluster,
                "mins": mins,
                "maxs": maxs,
                "centroid": centroid(mins, maxs),
                "volume": vol
            })

    return leafs

def extract_solid_leafs(bsp):
    """Extract all leafs (both solid and non-solid) for validation"""
    all_leafs = []
    
    for leaf in bsp.LEAVES:
        mins = np.array(leaf.bounds.mins)
        maxs = np.array(leaf.bounds.maxs)
        all_leafs.append({
            "mins": mins,
            "maxs": maxs,
            "cluster": leaf.cluster,
            "is_solid": leaf.cluster < 0
        })
    
    return all_leafs

def is_position_valid(pos, all_leafs):
    """Check if a position is inside a valid (non-solid) leaf"""
    for leaf in all_leafs:
        # Check if position is inside this leaf's bounds
        if np.all(pos >= leaf["mins"]) and np.all(pos <= leaf["maxs"]):
            # Position is inside this leaf - check if it's a valid (non-solid) leaf
            return not leaf["is_solid"]
    
    # Position is not inside any leaf - outside the playable area
    return False

def compute_cluster_centroids(leafs):
    clusters = {}

    for leaf in leafs:
        c = leaf["cluster"]
        clusters.setdefault(c, []).append(leaf["centroid"])

    return {
        c: np.mean(points, axis=0)
        for c, points in clusters.items()
    }

def get_cluster_popularity(leafs):
    """Count how many leafs belong to each cluster (popularity metric)"""
    popularity = {}
    for leaf in leafs:
        c = leaf["cluster"]
        popularity[c] = popularity.get(c, 0) + 1
    return popularity

def clamp_to_world_bounds(pos, world_bounds, margin=100):
    """Clamp position to stay within world bounds with safety margin"""
    mins = np.array(world_bounds[0])
    maxs = np.array(world_bounds[1])
    clamped = np.clip(pos, mins + margin, maxs - margin)
    return clamped

# ----------------------------
# Camera selection
# ----------------------------

def select_overview_camera(leafs, map_center, map_bounds, all_leafs, num_cameras=1):
    # Sort by volume and select top num_cameras (use unique leafs)
    sorted_leafs = sorted(leafs, key=lambda l: l["volume"], reverse=True)
    selected_leafs = []
    for leaf in sorted_leafs:
        if len(selected_leafs) >= num_cameras:
            break
        # Avoid duplicates by checking centroid
        if not any(np.allclose(leaf["centroid"], s["centroid"]) for s in selected_leafs):
            selected_leafs.append(leaf)
    
    cameras = []
    for leaf in selected_leafs:
        pos = leaf["centroid"].copy()
        pos[2] += CAMERA_HEIGHT_OFFSET
        # Clamp to world bounds
        pos = clamp_to_world_bounds(pos, map_bounds)
        # Validate position is in valid playable space
        if is_position_valid(pos, all_leafs):
            yaw, pitch = yaw_pitch(pos, map_center)
            cameras.append(make_camera("overview", pos, yaw, pitch))
    
    return cameras if num_cameras > 1 else (cameras[0] if cameras else None)

def select_combat_camera(cluster_centroids, leafs, map_bounds, all_leafs, num_cameras=1):
    # Get popularity of each cluster (# of leafs in it)
    popularity = get_cluster_popularity(leafs)
    
    # Sort clusters by popularity (descending), then by index for stability
    sorted_clusters = sorted(
        cluster_centroids.items(),
        key=lambda x: (-popularity.get(x[0], 0), x[0])
    )
    num_available = len(sorted_clusters)
    
    if num_available == 0:
        return [] if num_cameras > 1 else None
    
    # Select top num_cameras from popular clusters
    num_to_select = min(num_cameras, num_available)
    selected_clusters = sorted_clusters[:num_to_select]
    
    cameras = []
    for cluster_id, cluster_pos in selected_clusters:
        pos = cluster_pos.copy()
        pos[2] += CAMERA_HEIGHT_OFFSET
        # Clamp to world bounds
        pos = clamp_to_world_bounds(pos, map_bounds)
        # Validate position is in valid playable space
        if is_position_valid(pos, all_leafs):
            yaw, pitch = yaw_pitch(pos, cluster_pos + np.array([1, 0, 0]))
            cameras.append(make_camera("combat", pos, yaw, DEFAULT_PITCH))
    
    return cameras if num_cameras > 1 else (cameras[0] if cameras else None)

def select_sightline_camera(leafs, map_bounds, all_leafs, num_cameras=1):
    # Only use leafs from well-populated clusters for better sightlines
    popularity = get_cluster_popularity(leafs)
    min_popularity = max(1, np.percentile(list(popularity.values()), 25))
    filtered_leafs = [l for l in leafs if popularity.get(l["cluster"], 0) >= min_popularity]
    
    if len(filtered_leafs) < 2:
        filtered_leafs = leafs  # Fallback to all leafs if filtering removes too many
    
    pairs = []
    for i in range(len(filtered_leafs)):
        for j in range(i + 1, len(filtered_leafs)):
            d = np.linalg.norm(
                filtered_leafs[i]["centroid"] - filtered_leafs[j]["centroid"]
            )
            pairs.append((d, (filtered_leafs[i], filtered_leafs[j])))
    
    # Sort by distance (descending)
    pairs.sort(reverse=True, key=lambda x: x[0])
    
    # Select top num_cameras pairs, ensuring no leaf is reused
    used_leafs = set()
    top_pairs = []
    for pair in pairs:
        if len(top_pairs) >= num_cameras:
            break
        _, (leaf1, leaf2) = pair
        # Use id() to check if leaf objects are the same
        leaf1_id = id(leaf1["centroid"])
        leaf2_id = id(leaf2["centroid"])
        if leaf1_id not in used_leafs and leaf2_id not in used_leafs:
            top_pairs.append(pair)
            used_leafs.add(leaf1_id)
            used_leafs.add(leaf2_id)
    
    cameras = []
    for _, (start_leaf, end_leaf) in top_pairs:
        start = start_leaf["centroid"]
        end = end_leaf["centroid"]
        pos = start.copy()
        pos[2] += CAMERA_HEIGHT_OFFSET
        # Clamp to world bounds
        pos = clamp_to_world_bounds(pos, map_bounds)
        # Validate position is in valid playable space
        if is_position_valid(pos, all_leafs):
            yaw, pitch = yaw_pitch(pos, end)
            cameras.append(make_camera("sightline", pos, yaw, pitch))
    
    return cameras if num_cameras > 1 else (cameras[0] if cameras else None)

def select_vertical_camera(leafs, map_bounds, all_leafs, num_cameras=1):
    # Sort by z_range and select top num_cameras (use unique leafs)
    sorted_leafs = sorted(
        leafs,
        key=lambda l: l["maxs"][2] - l["mins"][2],
        reverse=True
    )
    selected_leafs = []
    for leaf in sorted_leafs:
        if len(selected_leafs) >= num_cameras:
            break
        # Avoid duplicates by checking centroid
        if not any(np.allclose(leaf["centroid"], s["centroid"]) for s in selected_leafs):
            selected_leafs.append(leaf)
    
    cameras = []
    for leaf in selected_leafs:
        z_range = leaf["maxs"][2] - leaf["mins"][2]
        pos = leaf["centroid"].copy()
        pos[2] += CAMERA_HEIGHT_OFFSET
        # Clamp to world bounds
        pos = clamp_to_world_bounds(pos, map_bounds)
        # Validate position is in valid playable space
        if is_position_valid(pos, all_leafs):
            look_at = pos + np.array([0, 0, z_range])
            yaw, pitch = yaw_pitch(pos, look_at)
            cameras.append(make_camera("vertical", pos, yaw, pitch))
    
    return cameras if num_cameras > 1 else (cameras[0] if cameras else None)

# ----------------------------
# Camera factory
# ----------------------------

def make_camera(cam_type, pos, yaw, pitch):
    return {
        "type": cam_type,
        "pos": tuple(round(v, 1) for v in pos),
        "angles": (round(yaw, 1), round(pitch, 1))
    }

# ----------------------------
# Main entry point
# ----------------------------

def extract_cameras(bsp_path, num_cameras=1):
    bsp = load_bsp(str(bsp_path))

    world_model = bsp.MODELS[0]
    world_bounds = (
        np.array(world_model.bounds.mins),
        np.array(world_model.bounds.maxs)
    )
    map_center = centroid(world_bounds[0], world_bounds[1])

    leafs = extract_leaf_centroids(bsp)
    all_leafs = extract_solid_leafs(bsp)
    clusters = compute_cluster_centroids(leafs)

    # other cameras to consider:
    # spawn points
    # flag points
    # models (points of interest)

    #debug - print some leaf stats
    print(f"Total leafs: {len(bsp.LEAVES)}, Significant leafs: {len(leafs)}, Clusters: {len(clusters)}")
    print(f"leaf1 min: {leafs[0]['mins']}, max: {leafs[0]['maxs']}")
    print(f"leaf2 min: {leafs[1]['mins']}, max: {leafs[1]['maxs']}")
    print(f"leaf3 min: {leafs[2]['mins']}, max: {leafs[2]['maxs']}")
    print(f"leaf4 min: {leafs[3]['mins']}, max: {leafs[3]['maxs']}")
    print(f"leaf5 min: {leafs[4]['mins']}, max: {leafs[4]['maxs']}")

    overview = select_overview_camera(leafs, map_center, world_bounds, all_leafs, num_cameras)
    combat = select_combat_camera(clusters, leafs, world_bounds, all_leafs, num_cameras)
    sightline = select_sightline_camera(leafs, world_bounds, all_leafs, num_cameras)
    vertical = select_vertical_camera(leafs, world_bounds, all_leafs, num_cameras)
    
    # Flatten all results into a single list, filtering out None values
    cameras = []
    for cam_list in [overview, combat, sightline, vertical]:
        if cam_list is None:
            continue
        elif isinstance(cam_list, list):
            cameras.extend(cam_list)
        else:
            cameras.append(cam_list)

    return cameras



"""
Example Output:
[
  {
    "type": "overview",
    "pos": [512.0, -128.0, 384.0],
    "angles": [135.0, -12.0]
  },
  {
    "type": "combat",
    "pos": [-64.0, 256.0, 192.0],
    "angles": [0.0, -15.0]
  },
  {
    "type": "sightline",
    "pos": [-1024.0, 512.0, 256.0],
    "angles": [45.0, -8.0]
  },
  {
    "type": "vertical",
    "pos": [128.0, -512.0, 448.0],
    "angles": [0.0, 75.0]
  }
]


Usage in auto_cam.cfg:

    cg_draw2D 0
    cg_drawGun 0
    cg_thirdPerson 1
    r_fov 95

    viewpos 512 -128 384 135
    wait 40
    screenshotJPEG

    viewpos -64 256 192 0
    wait 40
    screenshotJPEG

"""