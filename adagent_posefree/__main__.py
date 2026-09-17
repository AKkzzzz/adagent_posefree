import argparse
import json
from pathlib import Path

from .config import load_config
from .data import index_folders, index_waymo, load_manifest


def main():
    p = argparse.ArgumentParser(description="Standalone RGB-only camera preprocessing; 4090/H200 use the same CUDA backend.")
    commands = p.add_subparsers(dest="command", required=True)
    generic = commands.add_parser("index", help="index scene/camera/numeric-frame.jpg folders")
    generic.add_argument("--input", type=Path, required=True)
    generic.add_argument("--output", type=Path, required=True)
    generic.add_argument("--fps", type=float, required=True)
    generic.add_argument("--reference-camera")
    waymo = commands.add_parser("index-waymo", help="adapt any UFO/Waymo scene list without reading GT cameras")
    waymo.add_argument("--data-root", type=Path, required=True)
    waymo.add_argument("--annotation", type=Path, required=True)
    waymo.add_argument("--output", type=Path, required=True)
    waymo.add_argument("--cameras", default="1,0,2")
    for name in ("prepare", "doctor"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--config", type=Path)
        cmd.add_argument("--num-gpus", type=int, default=1)
        if name == "prepare":
            cmd.add_argument("--manifest", type=Path, required=True)
            cmd.add_argument("--output", type=Path, required=True)
            cmd.add_argument("--check-only", action="store_true")
    for name in ("status", "validate"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.command in ("index", "index-waymo"):
        if a.output.exists():
            raise FileExistsError(f"manifest already exists: {a.output}; choose a new path")
        if a.command == "index":
            if a.fps <= 0:
                raise ValueError("fps must be positive")
            value = index_folders(a.input, a.fps, a.reference_camera)
        else:
            value = index_waymo(a.data_root, a.annotation, tuple(a.cameras.split(",")))
        if not value["scenes"]:
            raise ValueError("no scenes found")
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        try:
            scenes = load_manifest(a.output)
        except Exception:
            a.output.unlink()
            raise
        print(f"MANIFEST=PASS scenes={len(scenes)} output={a.output}")
    elif a.command == "doctor":
        from .runtime import doctor
        doctor(load_config(a.config), a.num_gpus)
    elif a.command == "prepare":
        from .runtime import prepare
        prepare(load_manifest(a.manifest), load_config(a.config), a.output, a.num_gpus, a.check_only)
    else:
        from .output import complete
        scenes = load_manifest(a.output / "input_manifest.json", check_images=False)
        signature = json.loads((a.output / ".posefree_contract.json").read_text())["run_signature"]
        done = [s["scene_name"] for s in scenes if complete(a.output, s, signature)]
        pending = [s["scene_name"] for s in scenes if s["scene_name"] not in set(done)]
        print(json.dumps(dict(validated_final=len(done), total=len(scenes), pending_or_invalid=len(pending),
                              examples=pending[:10]), indent=2))
        if a.command == "validate" and pending:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
