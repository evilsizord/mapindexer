import zipfile
import json
import os
from pathlib import Path

def extract_pk3(pk3_path, output_root):
    pk3_path = Path(pk3_path)
    map_name = pk3_path.stem.lower()

    out_dir = Path(output_root) / map_name
    out_dir.mkdir(parents=True, exist_ok=True)

    arena_data = {}
    bsp_files = []
    levelshots = []

    with zipfile.ZipFile(pk3_path, 'r') as z:
        for name in z.namelist():
            lname = name.lower()

            if lname.endswith('.arena'):
                arena_text = z.read(name).decode('utf-8', errors='ignore')
                arena_data = parse_arena(arena_text)

            elif lname.endswith('.bsp'):
                bsp_path = out_dir / 'bsp' / Path(name).name
                bsp_path.parent.mkdir(exist_ok=True)
                bsp_path.write_bytes(z.read(name))
                bsp_files.append(str(bsp_path))

            elif 'levelshots/' in lname and lname.endswith(('.jpg', '.png')):
                ls_path = out_dir / 'levelshots' / Path(name).name
                ls_path.parent.mkdir(exist_ok=True)
                ls_path.write_bytes(z.read(name))
                levelshots.append(str(ls_path))

    meta = {
        "map_name": arena_data.get("map"),
        "author": arena_data.get("author"),
        "gametypes": arena_data.get("type"),
        "bsp_files": bsp_files,
        "existing_levelshots": levelshots
    }

    with open(out_dir / "arena.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def parse_arena(text):
    """
    Very simple Quake3 arena parser
    """
    result = {}
    for line in text.splitlines():
        if '"' in line:
            parts = line.replace('"', '').split()
            if len(parts) >= 2:
                result[parts[0]] = " ".join(parts[1:])
    return result
