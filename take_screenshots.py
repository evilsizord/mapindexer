import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Take map screenshots from generated camera points")
    parser.add_argument("path", help="Path to a BSP file or directory containing map subfolders")
    parser.add_argument("--game-dir", default=os.getenv("GAME_DIR"), help="Urban Terror install folder")
    parser.add_argument("--exe", default="Quake3-UrT.exe", help="Game executable name or path")
    parser.add_argument("--fs-game", default="q3ut4", help="fs_game value")
    parser.add_argument("--no-run", action="store_true", help="Only write cfg files; do not launch the game")
    parser.add_argument("--wait", type=int, default=100, help="Wait frames between camera move and screenshot")
    parser.add_argument("--base-cfg", default="base_cam.cfg", help="Base cfg to exec before camera commands")
    parser.add_argument("--cfg-prefix", default="_", help="Prefix for generated camera cfg files")
    args = parser.parse_args()

    if not args.game_dir:
        raise SystemExit("GAME_DIR environment variable is not set. Pass --game-dir or set GAME_DIR.")

    game_dir = Path(args.game_dir)
    if not game_dir.exists() or not game_dir.is_dir():
        raise SystemExit(f"Game directory does not exist: {game_dir}")

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    maps_processed = 0
    if path.is_file() and path.suffix.lower() == ".bsp":
        process_bsp(path, game_dir, args)
        maps_processed += 1
    elif path.is_dir():
        for mapdir in path.iterdir():
            if not mapdir.is_dir():
                continue
            bsp_path = mapdir / f"maps/{mapdir.stem}.bsp"
            if not bsp_path.exists():
                print(f"Skipping missing BSP: {bsp_path}")
                continue
            process_bsp(bsp_path, game_dir, args)
            maps_processed += 1
    else:
        print(f"Error: Path must be a BSP file or directory: {path}")
        sys.exit(1)

    print(f"Maps processed: {maps_processed}")


def process_bsp(bsp_path: Path, game_dir: Path, args):
    cameras_path = bsp_path.with_suffix(".cameras.pkl")
    if not cameras_path.exists():
        print(f"Skipping {bsp_path.stem}; camera file not found: {cameras_path}")
        return

    cameras = load_cameras(cameras_path)
    if not cameras:
        print(f"Skipping {bsp_path.stem}; no cameras in {cameras_path}")
        return

    cfg_name = f"{args.cfg_prefix}{bsp_path.stem}.cam.cfg"
    cfg_path = game_dir / args.fs_game / cfg_name
    write_camera_cfg(cfg_path, bsp_path.stem, cameras, wait=args.wait, base_cfg=args.base_cfg)
    print(f"Wrote {len(cameras)} cameras to {cfg_path}")

    if args.no_run:
        return

    run_game_for_map(game_dir, args.exe, args.fs_game, bsp_path.stem, cfg_name)


def load_cameras(cameras_path: Path) -> list[dict]:
    with cameras_path.open("rb") as f:
        cameras = pickle.load(f)
    return [camera_to_dict(camera) for camera in cameras]


def camera_to_dict(camera) -> dict:
    if isinstance(camera, dict):
        return camera
    if hasattr(camera, "to_dict"):
        return camera.to_dict()
    raise TypeError(f"Unsupported camera record: {type(camera)!r}")


def write_camera_cfg(cfg_path: Path, map_name: str, cameras: list[dict], *, wait: int, base_cfg: str):
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w") as f:
        f.write(f"exec {base_cfg}\n\n")
        for i, camera in enumerate(cameras):
            position = camera.get("position") or camera.get("pos")
            if position is None:
                raise ValueError(f"Camera {i} is missing position/pos: {camera}")

            yaw = camera.get("yaw")
            if yaw is None:
                angles = camera.get("angles")
                yaw = angles[0] if angles else 0

            camera_type = camera.get("type", "camera")
            screenshot_name = f"{map_name}_{i}_{camera_type}"
            f.write(f"// Camera {i} - Type: {camera_type} - Score: {camera.get('score', '')}\n")
            f.write(f"setviewpos {int(position[0])} {int(position[1])} {int(position[2])} {int(yaw)}\n")
            f.write(f"wait {wait}\n")
            f.write(f"screenshotJPEG \"{screenshot_name}\"\n\n")
        f.write("wait 100\nquit\n")


def run_game_for_map(game_dir: Path, exe: str, fs_game: str, map_name: str, cfg_name: str):
    exe_path = Path(exe)
    command_exe = str(exe_path if exe_path.is_absolute() else game_dir / exe)
    command = [
        command_exe,
        "+set",
        "fs_game",
        fs_game,
        "+devmap",
        map_name,
        "+exec",
        cfg_name,
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=game_dir, check=True)


if __name__ == "__main__":
    main()
