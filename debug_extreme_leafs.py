from pathlib import Path
import numpy as np
from math import atan2, degrees
from bsp_tool import load_bsp

test_bsp = Path("C:\\repos\\mapindexer\\maps\\processed\\layout_del_1\\maps\\layout_del_1.bsp")

def find_extreme_contributors(models, faces, vertices, textures):
    world = models[0]
    start = world.first_face
    end = start + world.num_faces

    # Track min/max contributors for x/y/z
    extremes = {
        "min_x": (float("inf"), None, None),
        "max_x": (float("-inf"), None, None),
        "min_y": (float("inf"), None, None),
        "max_y": (float("-inf"), None, None),
        "min_z": (float("inf"), None, None),
        "max_z": (float("-inf"), None, None),
    }

    for fi in range(start, end):
        f = faces[fi]
        if f.num_vertices <= 0:
            continue

        v_start = f.first_vertex
        v_end = v_start + f.num_vertices
        for vi in range(v_start, v_end):
            x, y, z = vertices[vi].position

            if x < extremes["min_x"][0]: extremes["min_x"] = (x, fi, vi)
            if x > extremes["max_x"][0]: extremes["max_x"] = (x, fi, vi)
            if y < extremes["min_y"][0]: extremes["min_y"] = (y, fi, vi)
            if y > extremes["max_y"][0]: extremes["max_y"] = (y, fi, vi)
            if z < extremes["min_z"][0]: extremes["min_z"] = (z, fi, vi)
            if z > extremes["max_z"][0]: extremes["max_z"] = (z, fi, vi)

    def describe(label, value, fi, vi):
        f = faces[fi]
        shader_name = textures[f.texture].name if 0 <= f.texture < len(textures) else "<bad textureIndex>"
        pos = vertices[vi].position
        return {
            "extreme": label,
            "value": value,
            "faceIndex": fi,
            "vertexIndex": vi,
            "vertexPos": pos,
            "faceType": getattr(f, "faceType", None),
            "shader": shader_name,
        }

    return [describe(k, *v) for k, v in extremes.items()]


if __name__ == "__main__":
    bsp = load_bsp(str(test_bsp))
    extremes = find_extreme_contributors(bsp.MODELS, bsp.FACES, bsp.VERTICES, bsp.TEXTURES)
    for ext in extremes:
        print(f"{ext['extreme']}: value={ext['value']}, face={ext['faceIndex']}, vertex={ext['vertexIndex']}, pos={ext['vertexPos']}, shader={ext['shader']}")
