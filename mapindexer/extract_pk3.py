import zipfile
import json
import os
import sqlite3
from pathlib import Path

def extract_pk3(pk3_path, output_root, db_conn):
    pk3_path = Path(pk3_path)
    map_name = pk3_path.stem.lower()

    out_dir = Path(output_root) / map_name #todo: wtf is this
    out_dir.mkdir(parents=True, exist_ok=True)

    arena_text = ""
    arena_data = {}

    # Check if already extracted
    if out_dir.exists():
        # Directory has content, try to parse existing arena file
        arena_files = list(out_dir.glob("**/*.arena"))
        if arena_files:
            arena_text = arena_files[0].read_text(encoding='utf-8', errors='ignore')
        else:
            print(f"No arena file found in existing extraction for {map_name}")
    else:
        # Extract all contents
        with zipfile.ZipFile(pk3_path, 'r') as z:
            z.extractall(out_dir)

            # Parse arena file if present
            for name in z.namelist():
                if name.lower().endswith('.arena'):
                    arena_text = z.read(name).decode('utf-8', errors='ignore')
                    break

    # NOTE - some maps do not have an arena file
    
    arena_data = parse_arena(arena_text)

    meta = {
        "map_name": arena_data.get("map"),
        "arena_longname": arena_data.get("longname"),
        "gametypes": arena_data.get("type"),
    }

    #debug
    #print("arena data:", arena_data)

    # Validate and clean data before database insertion
    clean_meta = validate_and_clean_arena_data(meta, map_name)

    file_name = pk3_path.name.lower()
    file_size = pk3_path.stat().st_size

    try:
        # Check if entry already exists
        cursor = db_conn.execute(
            "SELECT id FROM Maps WHERE file_name = ?",
            (file_name,),
        )
        existing = cursor.fetchone()

        if existing:
            print("Skipping database insert for existing map:", file_name)
        else:
            # Insert new entry
            db_conn.execute(
                "INSERT INTO Maps (file_name, file_size, name, arena_longname, supported_modes) VALUES (?, ?, ?, ?, ?)",
                (
                    file_name,
                    file_size,
                    clean_meta["map_name"],
                    clean_meta["arena_longname"],
                    clean_meta["gametypes"],
                ),
            )
        db_conn.commit()
    except sqlite3.Error as e:
        print(f"Database error for {file_name}: {e}")

    return clean_meta

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

def validate_and_clean_arena_data(meta, default_map_name):
    """
    Validate and clean arena metadata before database insertion.
    - Sets missing author/gametypes to "Unknown"
    - Strips/trims all whitespace
    - Ensures map_name has a value
    """
    map_name = (meta.get("map_name") or "").strip()
    arena_longname = (meta.get("arena_longname") or "").strip() or None
    gametypes = (meta.get("gametypes") or "").strip() or None
    
    # todo validate and split gametypes

    return {
        "map_name": map_name,
        "arena_longname": arena_longname,
        "gametypes": gametypes,
    }
