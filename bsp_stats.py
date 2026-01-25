import os
import sys
import argparse
from pathlib import Path
import json
from dotenv import load_dotenv
import sqlite3
from mylib.bsp import analyze_bsp

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
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)
    
    if path.is_file() and path.suffix.lower() == ".bsp":
        # Single BSP file
        try:
            stats = analyze_bsp(path)
            print(json.dumps(stats, indent=2))
        except Exception as e:
            print(f"Failed to analyze {path.name}: {e}")
            sys.exit(1)
    
    elif path.is_dir():
        # Directory - iterate over subfolders
        for mapdir in path.iterdir():
            if mapdir.is_dir():
                try:
                    bsp_path = mapdir / f"maps/{mapdir.stem}.bsp"
                    stats = analyze_bsp(bsp_path)
                    add_to_database(f"{mapdir.stem}.pk3", stats)
                    #print(json.dumps(stats, indent=2))
                    #break
                except Exception as e:
                    print(f"Failed {mapdir.name}: {e}")
                    break
    else:
        print(f"Error: Path must be a BSP file or directory: {path}")
        sys.exit(1)


def add_to_database(file_name, stats):
    try:
        # Check if entry already exists
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
        print(f"Database error for {file_name}: {e}")



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