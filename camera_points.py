import os
from pathlib import Path
from dotenv import load_dotenv
from mylib.camera import extract_cameras
import sqlite3

load_dotenv()

db_path = os.getenv("MAPINDEXER_DB")
game_dir = os.getenv("GAME_DIR")

try:
    dbconn = sqlite3.connect(db_path)
except sqlite3.Error as e:
    raise SystemExit(f"Failed to connect to database {db_path}: {e}")

# ----------------------------
# CLI usage
# ----------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python camera_points.py <map.bsp>")
        sys.exit(1)

    bsp_path = Path(sys.argv[1])
    if not bsp_path.exists() or not bsp_path.is_file():
        print(f"Error: BSP file does not exist: {bsp_path}")
        sys.exit(1)

    cams = extract_cameras(bsp_path, num_cameras=5)
    #print(json.dumps(cams, indent=2))
    
    # Write cameras to auto_cam.cfg
    if game_dir:
        cfg_path = Path(game_dir) / "q3ut4" / f"_{bsp_path.stem}.cam.cfg"
        with open(cfg_path, 'w') as f:
            f.write("exec base_cam.cfg\n\n")
            for i, cam in enumerate(cams):
                # Extract camera position and angle from the camera data
                f.write(f"// Camera {i} - Type: {cam['type']}\n")
                f.write(f"setviewpos {int(cam['pos'][0])} {int(cam['pos'][1])} {int(cam['pos'][2])} {int(cam['angles'][0])}\n")
                f.write("wait 100\n")
                f.write(f"screenshotJPEG \"{bsp_path.stem}_{i}_{cam['type']}\"\n\n")
            f.write("wait 100\nquit\n")
        print(f"Wrote {len(cams)} cameras to {cfg_path}")
    else:
        print("GAME_DIR environment variable not set")
