#!/usr/bin/env python3
"""Schedule independent scenes using the installed pose-free preparation code.

Each GPU gets private metadata, logs and alignment output. Intermediate scene
files use the existing shared cache, so previously completed windows are reused.
Only validated global camera files are atomically published for H200 training.
This entry point never launches training or changes the numerical pipeline.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

SHARED_DIRS = ("manifests_rgb_only", "omega_raw_all", "gca_omega_scale_all",
               "omega_gca_metric_all", "batch_lists")
CAMERAS = ("1", "0", "2")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_camera(path, scene):
    with np.load(path, allow_pickle=False) as x:
        if str(x["scene_name"].item()) != scene["scene_name"]:
            raise ValueError(f"wrong scene in {path}")
        if str(x["coordinate_frame"].item()) != "global_metric":
            raise ValueError(f"not a global metric camera cache: {path}")
        frames = x["frame_ids"]
        cameras = x["camera_ids"].astype(str)
        expected = {(i, c) for i in range(int(scene["num_timesteps"])) for c in CAMERAS}
        if frames.ndim != 1 or cameras.shape != frames.shape:
            raise ValueError(f"invalid frame/camera arrays in {path}")
        if len(frames) != len(expected) or set(zip(frames.tolist(), cameras.tolist())) != expected:
            raise ValueError(f"incomplete or duplicate frame/camera coverage: {path}")
        poses = x["omega_camera_to_world_global_metric"]
        intrinsics = x["predicted_intrinsics_ufo"]
        if poses.shape != (len(frames), 4, 4) or intrinsics.shape != (len(frames), 3, 3):
            raise ValueError(f"invalid camera dimensions in {path}")
        if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
            raise ValueError(f"non-finite cameras in {path}")
        if (intrinsics[:, (0, 1), (0, 1)] <= 0).any():
            raise ValueError(f"non-positive focal lengths in {path}")


def worker_cache(cache, rank):
    root = cache / "parallel_workers" / f"worker_{rank:02d}"
    root.mkdir(parents=True, exist_ok=True)
    for name in SHARED_DIRS:
        target = cache / name
        target.mkdir(parents=True, exist_ok=True)
        link = root / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"unexpected cache link: {link}")
        elif link.exists():
            raise RuntimeError(f"expected a cache symlink, found directory: {link}")
        else:
            link.symlink_to(target.resolve(), target_is_directory=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "global_aligned").mkdir(exist_ok=True)
    return root


def publish_camera(worker, cache, scene, copy_report=True):
    name = scene["scene_name"]
    source = worker / "global_aligned" / name / "omega_pose_override.npz"
    validate_camera(source, scene)
    destination = cache / "global_aligned" / name / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validate_camera(destination, scene)
        raise RuntimeError(f"another writer published {destination}; stop overlapping jobs")
    report = worker / "global_aligned" / "alignment_report.json"
    if copy_report and report.is_file():
        # Keep each scene's report, instead of overwriting one global report.
        atomic_json(destination.parent / "alignment_report.json", json.loads(report.read_text()))
    os.replace(source, destination)


def recover_staged(cache, scenes, contract):
    """Recover a finished scene even if it was produced by a previous GPU rank."""
    by_name = {scene["scene_name"]: scene for _, scene in scenes}
    recovered = 0
    for path in sorted((cache / "parallel_workers").glob(
            "worker_*/global_aligned/*/omega_pose_override.npz")):
        name = path.parent.name
        if name not in by_name:
            continue
        destination = cache / "global_aligned" / name / path.name
        if destination.exists():
            validate_camera(destination, by_name[name])
            continue
        worker = path.parents[2]
        saved_path = worker / ".posefree_contract.json"
        if not saved_path.is_file():
            raise RuntimeError(f"staged camera lacks its contract: {path}")
        saved = json.loads(saved_path.read_text())
        if any(saved.get(key) != value for key, value in contract.items()):
            raise RuntimeError(f"staged camera contract differs: {saved_path}")
        try:
            validate_camera(path, by_name[name])
        except Exception:
            # A killed aligner can leave a partial NPZ. Preserve it under a
            # different name so the old driver's existence check cannot skip it.
            quarantine = path.with_name(path.name + f".incomplete.{time.time_ns()}")
            os.replace(path, quarantine)
            print(f"[posefree/retry] incomplete staged camera retained at {quarantine}", flush=True)
            continue
        # The old worker-level report has no scene field and may be stale.
        publish_camera(worker, cache, by_name[name], copy_report=False)
        recovered += 1
        print(f"[posefree/recovered] scene={name}", flush=True)
    return recovered


def load_scenes(data, annotation, first, last):
    lines = [s.strip() for s in annotation.read_text().splitlines() if s.strip()]
    if len(lines) != 798 or not 0 <= first <= last < len(lines):
        raise ValueError("expected 798 scenes and a range within 0..797")
    scenes = []
    names = set()
    for index, line in enumerate(lines):
        p = Path(line)
        scene = json.loads((p if p.is_absolute() else data / p).read_text())
        name = scene["scene_name"]
        if not name or Path(name).name != name or name in (".", "..") or name in names:
            raise ValueError(f"invalid/duplicate scene name at index {index}: {name!r}")
        names.add(name)
        if first <= index <= last:
            scenes.append((index, scene))
    return scenes


class Processes:
    def __init__(self):
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.children = {}

    def cancel(self):
        self.stop.set()
        with self.lock:
            for child in self.children.values():
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    @staticmethod
    def terminate_group(child):
        """Reap our process tree even when its leader exits before its children."""
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            child.wait()
            return
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        # killpg still reaches descendants if their original leader has exited.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()

    def run(self, command, env, log):
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError("preparation cancelled")
            child = subprocess.Popen(command, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            self.children[child.pid] = child
        try:
            while True:
                try:
                    code = child.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if self.stop.is_set():
                        raise RuntimeError("preparation cancelled; intermediate cache retained")
            if code != 0:
                self.cancel()
                raise RuntimeError(f"preparation exited with {code}; inspect worker log")
        finally:
            if child.poll() is None or self.stop.is_set() or child.returncode != 0:
                self.terminate_group(child)
            with self.lock:
                self.children.pop(child.pid, None)


def existing_preparation(repo, cache):
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if any(Path(v).name in ("prepare_posefree_waymo_full.py", "posefree_batch_stage.py")
                   for v in args) and any(str(repo) in v or str(cache) in v for v in args):
                found.append(proc.name)
        except (OSError, ValueError):
            pass
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    p.add_argument("--expected-gpus", type=int,
                   help="Fail if the supplied device list has a different size")
    p.add_argument("--scene-first", type=int, default=0)
    p.add_argument("--scene-last", type=int, default=797)
    p.add_argument("--window-batch-size", type=int, default=179)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--min-free-gib", type=float)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    repo, source, data, cache = (x.resolve() for x in
                               (a.repo, a.source_root, a.data_root, a.cache_root))
    gpus = [s.strip() for s in a.gpus.split(",")]
    n_gpus = len(gpus)
    if len(set(gpus)) != n_gpus or not all(gpus):
        raise ValueError("--gpus must contain distinct device indices/UUIDs")
    if a.expected_gpus is not None and n_gpus != a.expected_gpus:
        raise ValueError(f"expected {a.expected_gpus} GPU selectors, got {gpus}; "
                         "check UFO_POSEFREE_GPUS and CUDA_VISIBLE_DEVICES")
    if a.min_free_gib is None:
        a.min_free_gib = max(50, 37.5 * n_gpus)
    if min(a.window_batch_size, a.cpu_threads, a.min_free_gib) <= 0:
        raise ValueError("batch size, CPU threads and free-space floor must be positive")
    launcher = repo / "scripts/r9/run_r9_waymo_full_posefree.sh"
    prep_python = Path(sys.executable)  # Keep the venv path, do not resolve its symlink.
    ufo = Path(os.environ.get("UFO_POSEFREE_UFO_ROOT", source / "pose-freeufo")).resolve()
    omega = Path(os.environ.get("UFO_POSEFREE_OMEGA_REPO", source / "vggt-omega")).resolve()
    omega_weight = Path(os.environ.get("UFO_POSEFREE_OMEGA_CHECKPOINT",
                                      omega / "checkpoints/vggt_omega_1b_512.pt")).resolve()
    moge = Path(os.environ.get("UFO_POSEFREE_MOGE_REPO", source / "moge")).resolve()
    moge_weight = Path(os.environ.get("UFO_POSEFREE_MOGE_MODEL",
                                     source / "checkpoints/moge-2-vitl/model.pt")).resolve()
    annotation = data / "scene_list/waymo_train.txt"
    config_path = repo / "configs/h200/r9_waymo_full_posefree_100k.json"
    for path in (launcher, ufo, omega, omega_weight, moge, moge_weight, annotation, config_path):
        if not path.exists():
            raise FileNotFoundError(path)
    scenes = load_scenes(data, annotation, a.scene_first, a.scene_last)
    config = json.loads(config_path.read_text())
    contract = {
        "schema_version": 1, "scene_count": 798, "camera_input_protocol": "all_rgb",
        "metric_method": "Omega + MoGe-2 + GCA + overlap SE(3)",
        "ufo_image_size": list(config["input_size"]), "source_ufo_root": str(ufo),
        "omega_checkpoint": str(omega_weight), "moge_model": str(moge_weight),
        "gt_camera_pose_used": False, "gt_intrinsics_used": False,
        "gt_depth_used": False, "gt_ego_pose_used": False,
    }
    if not cache.is_dir():
        raise FileNotFoundError(f"existing shared cache required: {cache}")
    contract_path = cache / ".posefree_contract.json"
    if contract_path.exists():
        saved = json.loads(contract_path.read_text())
        if any(saved.get(key) != value for key, value in contract.items()):
            raise ValueError("existing cache contract differs; refusing to mix camera runs")
    elif any((cache / "global_aligned").glob("*/omega_pose_override.npz")):
        raise ValueError("existing cameras lack their cache contract")

    remaining = []
    for index, scene in scenes:
        final = cache / "global_aligned" / scene["scene_name"] / "omega_pose_override.npz"
        if final.exists():
            validate_camera(final, scene)
        else:
            remaining.append((index, scene))
    free_gib = shutil.disk_usage(cache).free / 1024**3
    print(f"[posefree/{n_gpus}gpu/check] complete={len(scenes)-len(remaining)}/{len(scenes)} "
          f"pending={len(remaining)} filesystem_free_gib={free_gib:.1f}", flush=True)
    print("[posefree/check] Project quota is UNKNOWN; filesystem free space is not quota.", flush=True)
    if not remaining:
        print("POSEFREE_MULTIGPU_DONE: selected scene caches are valid", flush=True)
        return
    if free_gib < a.min_free_gib:
        raise RuntimeError(f"filesystem free space is below {a.min_free_gib} GiB")
    probe_env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
    probe = subprocess.run([str(prep_python), "-c",
        "import json,torch; print(json.dumps([torch.cuda.get_device_name(i) "
        "for i in range(torch.cuda.device_count())]))"], env=probe_env,
        text=True, stdout=subprocess.PIPE, check=True)
    names = json.loads(probe.stdout.strip().splitlines()[-1])
    if len(names) != n_gpus or any("4090" not in name for name in names):
        raise RuntimeError(f"expected {n_gpus} visible 4090 GPUs, got {names}")
    print(f"[posefree/{n_gpus}gpu/check] devices={gpus} names={names}", flush=True)
    if a.check_only:
        print("POSEFREE_MULTIGPU_CHECK=PASS (quota remains unverified)", flush=True)
        return

    with (cache / ".posefree_multigpu.lock").open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = existing_preparation(repo, cache)
        if old:
            raise RuntimeError(f"existing preparation PIDs {old}; stop the single-GPU prep first")
        if not contract_path.exists():
            atomic_json(contract_path, contract)
        recover_staged(cache, scenes, contract)
        # Recheck under the lock: another completed run or recovered staging
        # file may have published cameras since the read-only preflight.
        remaining = []
        for index, scene in scenes:
            final = cache / "global_aligned" / scene["scene_name"] / "omega_pose_override.npz"
            if final.exists():
                validate_camera(final, scene)
            else:
                remaining.append((index, scene))
        print(f"[posefree/{n_gpus}gpu/queue] pending={len(remaining)}", flush=True)
        processes = Processes()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: processes.cancel())
        jobs = iter(remaining)
        job_lock = threading.Lock()

        def worker(rank):
            root = worker_cache(cache, rank)
            env = dict(os.environ)
            env.pop("UFO_POSEFREE_FORCE", None)
            env.update(CUDA_VISIBLE_DEVICES=gpus[rank], PYTHONUNBUFFERED="1",
                       UFO_DATA_ROOT=str(data), UFO_POSEFREE_SOURCE_ROOT=str(source),
                       UFO_POSEFREE_PREP_PYTHON=str(prep_python),
                       UFO_POSEFREE_CACHE_ROOT=str(root),
                       UFO_POSEFREE_WINDOW_BATCH_SIZE=str(a.window_batch_size))
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "UFO_OMEGA_CPU_THREADS", "UFO_GCA_CPU_THREADS"):
                env[key] = str(a.cpu_threads)
            with (root / "worker.log").open("a", buffering=1) as log:
                try:
                    while not processes.stop.is_set():
                        with job_lock:
                            item = next(jobs, None)
                        if item is None:
                            return
                        index, scene = item
                        env["UFO_POSEFREE_SCENE_FIRST"] = str(index)
                        env["UFO_POSEFREE_SCENE_LAST"] = str(index)
                        if shutil.disk_usage(cache).free / 1024**3 < a.min_free_gib:
                            raise RuntimeError("filesystem free space fell below configured floor")
                        print(f"[posefree/{n_gpus}gpu/start] worker={rank} gpu={gpus[rank]} scene={index} "
                              f"log={root / 'worker.log'}", flush=True)
                        started = time.monotonic()
                        processes.run(["bash", str(launcher), "prepare"], env, log)
                        publish_camera(root, cache, scene)
                        print(f"[posefree/{n_gpus}gpu/done] worker={rank} scene={index} "
                              f"wall_s={time.monotonic()-started:.2f}", flush=True)
                except BaseException:
                    processes.cancel()
                    raise

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as pool:
                futures = [pool.submit(worker, rank) for rank in range(n_gpus)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except BaseException:
                        processes.cancel()
                        raise
            if processes.stop.is_set():
                raise RuntimeError("preparation interrupted; completed cameras and partial cache retained")
        finally:
            processes.cancel()
    print(f"POSEFREE_MULTIGPU_DONE global_cache={cache / 'global_aligned'}", flush=True)


if __name__ == "__main__":
    main()
