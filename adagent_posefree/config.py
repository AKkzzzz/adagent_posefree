import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(omega_checkpoint="checkpoints/vggt_omega_1b_512.pt",
                moge_checkpoint="checkpoints/moge-2-vitl/model.pt",
                output_image_size=[160, 240], window_frames=20, window_stride=1,
                window_batch_size=179, min_overlap_poses=9, cpu_threads=4,
                raw_compression="stored", min_free_gib_per_gpu=37.5,
                keep_intermediates=False, gca_stats_device="cuda")


def load_config(path=None):
    values = dict(DEFAULTS)
    if path:
        supplied = json.loads(Path(path).read_text())
        unknown = set(supplied) - set(values)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        values.update(supplied)
    for key in ("omega_checkpoint", "moge_checkpoint"):
        p = Path(os.path.expandvars(os.path.expanduser(values[key])))
        values[key] = str((ROOT / p if not p.is_absolute() else p).resolve())
    for key in ("window_frames", "window_stride", "window_batch_size", "min_overlap_poses", "cpu_threads"):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if values["min_overlap_poses"] < 3:
        raise ValueError("min_overlap_poses must be at least 3")
    if values["window_frames"] < 2 or values["window_stride"] >= values["window_frames"]:
        raise ValueError("windows must overlap and contain at least two timesteps")
    size = values["output_image_size"]
    if len(size) != 2 or any(type(i) is not int or i <= 0 for i in size):
        raise ValueError("output_image_size must be positive [height,width]")
    if values["raw_compression"] not in ("stored", "deflate") or values["gca_stats_device"] not in ("cpu", "cuda"):
        raise ValueError("invalid compression or GCA device")
    if type(values["keep_intermediates"]) is not bool or values["min_free_gib_per_gpu"] <= 0:
        raise ValueError("invalid cleanup policy or disk reserve")
    return values


def model_paths():
    return ROOT / "vendor/omega", ROOT / "vendor/moge"


def check_assets(config):
    omega, moge = model_paths()
    for p in (omega / "vggt_omega/models/vggt_omega.py", omega / "vggt_omega/utils/load_fn.py",
              moge / "moge/model/v2.py", Path(config["omega_checkpoint"]), Path(config["moge_checkpoint"])):
        if not p.is_file():
            raise FileNotFoundError(f"missing asset: {p}; run tools/import_server_assets.py or copy the documented checkpoints")


def gpu_selectors(count, visible, detected_count):
    if count < 1:
        raise ValueError("--num-gpus must be positive")
    candidates = [s.strip() for s in visible.split(",")] if visible is not None else [str(i) for i in range(detected_count)]
    if not all(candidates) or len(set(candidates)) != len(candidates) or "-1" in candidates:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain distinct CUDA device selectors")
    if count > detected_count or count > len(candidates):
        raise ValueError(f"requested {count} GPUs, only {detected_count} visible")
    # Preserve the scheduler's physical IDs/UUIDs instead of remapping 2,4 to 0,1.
    return candidates[:count]
