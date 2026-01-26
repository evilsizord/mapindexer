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

def is_junk_shader(shader) -> bool:
    n = str(shader.name).lower()
    # These frequently create huge bounds and aren't "playable envelope"
    return (
        "trigger" in n or
        "hint" in n or
        "skip" in n or
        "areaportal" in n or
        "portal" in n or
        "fog" in n
    )

def is_clip_shader(shader) -> bool:
    n = str(shader.name).lower()
    # Playerclip in UrT is often expressed via shader name instead of contents flag
    return n in ("common/clip", "common/playerclip") # todo  is this complete list?
    #return ("playerclip" in n) or (n.endswith("/clip")) or ("/clip" in n)


def is_solid_shader(shader) -> bool:
    return shader.flags[1] & CONTENTS_SOLID

