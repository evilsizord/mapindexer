import os
import sys
import argparse
from pathlib import Path
import json
import pickle
from datetime import datetime
from bsp_tool import load_bsp
from dotenv import load_dotenv
from colorama import Fore, Style
import sqlite3
from mapindexer.bsp_stats import analyze_bsp
from mapindexer.flood_fill_volume import flood_fill_flyable_volume_from_spawns
import traceback
from mapindexer.util import print_error, print_time

load_dotenv()

db_path = os.getenv("MAPINDEXER_DB")

if not db_path:
    raise SystemExit("MAPINDEXER_DB environment variable not set")

try:
    dbconn = sqlite3.connect(db_path)
except sqlite3.Error as e:
    raise SystemExit(f"Failed to connect to database {db_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze BSP file(s) for map statistics")
    parser.add_argument("path", help="Path to a BSP file or directory containing map subfolders")
    parser.add_argument("--verbose", action="store_true", help="Print flood-fill diagnostics while processing")
    parser.add_argument("--max-volume-cells", type=int, help="Stop flood fill after this many visited cells")
    parser.add_argument("--volume-voxel", type=float, default=64.0, help="Flood-fill voxel size in map units")
    parser.add_argument(
        "--skip-volume-edge-checks",
        action="store_true",
        help="Skip expensive line checks between adjacent reachable volume cells",
    )
    args = parser.parse_args()
    start_time = datetime.now()
    maps_processed = 0

    print_time(f"Start time: {start_time.isoformat(timespec='seconds')}")
    
    path = Path(args.path)
    
    if not path.exists():
        print_error(f"Error: Path does not exist: {path}")
        sys.exit(1)
    
    if path.is_file() and path.suffix.lower() == ".bsp":
        gen_bsp_stats(
            path,
            verbose=args.verbose,
            max_volume_cells=args.max_volume_cells,
            volume_voxel=args.volume_voxel,
            validate_volume_edges=not args.skip_volume_edge_checks,
        )
        maps_processed += 1
    
    elif path.is_dir():
        for mapdir in path.iterdir():
            if mapdir.is_dir():
                bsp_path = mapdir / f"maps/{mapdir.stem}.bsp"
                gen_bsp_stats(
                    bsp_path,
                    update_database=True,
                    file_name=f"{mapdir.stem}.pk3",
                    verbose=args.verbose,
                    max_volume_cells=args.max_volume_cells,
                    volume_voxel=args.volume_voxel,
                    validate_volume_edges=not args.skip_volume_edge_checks,
                )
                maps_processed += 1

    else:
        print_error(f"Error: Path must be a BSP file or directory: {path}")
        sys.exit(1)

    end_time = datetime.now()
    print_time(f"End time: {end_time.isoformat(timespec='seconds')}")
    print(f"Maps processed: {maps_processed}")


def gen_bsp_stats(
    bsp_path,
    update_database=False,
    file_name=None,
    verbose=False,
    max_volume_cells=None,
    volume_voxel=64.0,
    validate_volume_edges=True,
):
    print(f"Analyzing BSP: {bsp_path}")
    try:
        # todo - should use the volume for analyze_bsp actually, as well as the bsp.
        bsp = load_bsp(str(bsp_path))
        volume, aabb = flood_fill_flyable_volume_from_spawns(
            bsp,
            verbose=verbose,
            max_cells=max_volume_cells,
            voxel=volume_voxel,
            validate_edges=validate_volume_edges,
        )
        volume_path = bsp_path.with_suffix(".volume.pkl")
        # todo - will this overwrite an existing volume file?
        with volume_path.open("wb") as f:
            pickle.dump({"volume": volume, "aabb": aabb}, f, protocol=pickle.HIGHEST_PROTOCOL)
        stats = analyze_bsp(bsp_path)
        #print(json.dumps(stats, indent=2))
        if update_database:
            add_to_database(file_name, stats)
    except Exception as e:
        print(f"{Fore.RED}Failed to analyze {bsp_path}: {e}")
        traceback.print_exc()
        sys.exit(1)


def add_to_database(file_name, stats):
    try:
        # Check if entry already exists (should have been created during PK3 extraction)
        # todo idk if this necessary. Maybe just let it fail if UPDATE fails?
        cursor = dbconn.execute(
            "SELECT id FROM Maps WHERE file_name = ?",
            (file_name,),
        )
        existing = cursor.fetchone()

        if not existing:
            print("Map not found in database! Skipping:", file_name)
        else:
            # update entry
            dbconn.execute(
                "UPDATE Maps SET bsp_metrics = ? WHERE file_name = ?",
                (
                    json.dumps(stats),
                    file_name
                ),
            )
            dbconn.commit()
    except sqlite3.Error as e:
        print(f"{Fore.RED}Database error for {file_name}: {e}")



if __name__ == "__main__":
    main()



""" Example output:
{
  "bounds": {
    "min": [-2048, -2048, -512],
    "max": [2048, 2048, 1536],
    "size": [4096, 4096, 2048]
  },
  "bsp_size": 18745321,
  "vertical_span": 2048.0,
  "verticality_score": 0.5,
  "significant_leaf_count": 312,
  "room_count": 18,
  "complexity_score": 0.63
}

"""
