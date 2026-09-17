"""One GPU process per numerical stage; models stay resident across windows."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np

from ..config import model_paths
from .stages import homo, prepare_rgb_cache, stack_cached_rgb, metric_output


def raw_ok(path):
    try:
        with np.load(path, allow_pickle=False) as x:
            return all(k in x for k in ("scene_name", "frame_ids", "camera_ids", "roles", "omega_w2c_raw",
                                       "omega_c2w_raw", "predicted_intrinsics_ufo", "omega_depth_raw", "omega_depth_conf_raw"))
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("omega", "gca", "metric"), required=True)
    p.add_argument("--job", type=Path, required=True)
    a = p.parse_args()
    job = json.loads(a.job.read_text())
    omega, moge = model_paths()
    args = SimpleNamespace(omega_repo=omega, moge_repo=moge,
                           omega_checkpoint=job["config"]["omega_checkpoint"],
                           moge_model=job["config"]["moge_checkpoint"],
                           manifest_list=Path(job["list"]), raw_root=Path(job["raw"]),
                           scale_root=Path(job["scale"]), output_root=Path(job["windows"]))
    paths = [Path(x) for x in args.manifest_list.read_text().splitlines() if x]
    if a.stage == "omega":
        from .posefree_omega_runtime import run_omega
        run_omega(args, paths, raw_ok, prepare_rgb_cache, stack_cached_rgb, homo)
    elif a.stage == "gca":
        from .posefree_gca_runtime import run_gca
        run_gca(args, paths)
    else:
        for path in paths:
            manifest = json.loads(path.read_text())
            output = args.output_root / path.parent.name / path.stem / "omega_pose_override.npz"
            metric_output(manifest, args.raw_root/path.parent.name/(path.stem+".npz"),
                          args.scale_root/path.parent.name/(path.stem+".json"), output)
            print(f"[metric] {path.parent.name}", flush=True)


if __name__ == "__main__":
    main()
