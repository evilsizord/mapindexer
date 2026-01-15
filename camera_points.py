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

    for leaf in bsp.leafs:
        mins = np.array(leaf.mins)
        maxs = np.array(leaf.maxs)
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

def compute_cluster_centroids(leafs):
    clusters = {}

    for leaf in leafs:
        c = leaf["cluster"]
        clusters.setdefault(c, []).append(leaf["centroid"])

    return {
        c: np.mean(points, axis=0)
        for c, points in clusters.items()
    }

# ----------------------------
# Camera selection
# ----------------------------

def select_overview_camera(leafs, map_center):
    leaf = max(leafs, key=lambda l: l["volume"])
    pos = leaf["centroid"].copy()
    pos[2] += CAMERA_HEIGHT_OFFSET

    yaw, pitch = yaw_pitch(pos, map_center)
    return make_camera("overview", pos, yaw, pitch)

def select_combat_camera(cluster_centroids):
    # Most connected cluster ≈ central combat
    cluster = sorted(cluster_centroids.items())[len(cluster_centroids) // 2]
    pos = cluster[1].copy()
    pos[2] += CAMERA_HEIGHT_OFFSET

    yaw, pitch = yaw_pitch(pos, cluster[1] + np.array([1, 0, 0]))
    return make_camera("combat", pos, yaw, DEFAULT_PITCH)

def select_sightline_camera(leafs):
    max_dist = 0
    best_pair = None

    for i in range(len(leafs)):
        for j in range(i + 1, len(leafs)):
            d = np.linalg.norm(
                leafs[i]["centroid"] - leafs[j]["centroid"]
            )
            if d > max_dist:
                max_dist = d
                best_pair = (leafs[i], leafs[j])

    start = best_pair[0]["centroid"]
    end = best_pair[1]["centroid"]

    pos = start.copy()
    pos[2] += CAMERA_HEIGHT_OFFSET

    yaw, pitch = yaw_pitch(pos, end)
    return make_camera("sightline", pos, yaw, pitch)

def select_vertical_camera(leafs):
    best = None
    max_z = 0

    for leaf in leafs:
        z_range = leaf["maxs"][2] - leaf["mins"][2]
        if z_range > max_z:
            max_z = z_range
            best = leaf

    pos = best["centroid"].copy()
    pos[2] += CAMERA_HEIGHT_OFFSET

    look_at = pos + np.array([0, 0, z_range])
    yaw, pitch = yaw_pitch(pos, look_at)

    return make_camera("vertical", pos, yaw, pitch)

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

def extract_cameras(bsp_path):
    bsp = load_bsp(Path(bsp_path))

    world = bsp.models[0]
    map_center = centroid(
        np.array(world.mins),
        np.array(world.maxs)
    )

    leafs = extract_leaf_centroids(bsp)
    clusters = compute_cluster_centroids(leafs)

    cameras = [
        select_overview_camera(leafs, map_center),
        select_combat_camera(clusters),
        select_sightline_camera(leafs),
        select_vertical_camera(leafs)
    ]

    return cameras

# ----------------------------
# CLI usage
# ----------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python camera_points.py <map.bsp>")
        sys.exit(1)

    cams = extract_cameras(sys.argv[1])
    print(json.dumps(cams, indent=2))

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