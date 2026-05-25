import argparse
import os
import pickle
import sys
from pathlib import Path

from bsp_tool import load_bsp
from dotenv import load_dotenv

from mapindexer.camera_candidates import find_camera_candidates


load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Generate screenshot camera points for BSP file(s)")
    parser.add_argument("path", help="Path to a BSP file or directory containing map subfolders")
    parser.add_argument("--max-cameras", type=int, default=5)
    parser.add_argument("--cluster-radius", type=float, default=768.0)
    parser.add_argument("--min-separation", type=float, default=512.0)
    parser.add_argument("--no-validate", action="store_true", help="Skip BSP collision/visibility validation")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Seconds between verbose progress messages while evaluating candidates",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    maps_processed = 0
    if path.is_file() and path.suffix.lower() == ".bsp":
        process_bsp(path, args)
        maps_processed += 1
    elif path.is_dir():
        for mapdir in path.iterdir():
            if not mapdir.is_dir():
                continue
            bsp_path = mapdir / f"maps/{mapdir.stem}.bsp"
            if not bsp_path.exists():
                print(f"Skipping missing BSP: {bsp_path}")
                continue
            process_bsp(bsp_path, args)
            maps_processed += 1
    else:
        print(f"Error: Path must be a BSP file or directory: {path}")
        sys.exit(1)

    print(f"Maps processed: {maps_processed}")


def process_bsp(bsp_path: Path, args):
    print(f"Generating cameras for BSP: {bsp_path}")
    bsp = load_bsp(str(bsp_path))
    cameras = find_camera_candidates(
        bsp,
        max_cameras=args.max_cameras,
        cluster_radius=args.cluster_radius,
        min_separation=args.min_separation,
        validate=not args.no_validate,
        verbose=args.verbose,
        progress_interval=args.progress_interval,
    )
    camera_dicts = [camera.to_dict() for camera in cameras]

    cameras_path = bsp_path.with_suffix(".cameras.pkl")
    with cameras_path.open("wb") as f:
        pickle.dump(camera_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {len(camera_dicts)} cameras to {cameras_path}")


if __name__ == "__main__":
    main()
