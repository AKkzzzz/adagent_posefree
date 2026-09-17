"""Independent scene scheduling, bounded disk cache and crash-safe exports."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np
from .config import ROOT, check_assets, gpu_selectors, model_paths
from .data import digest, window_manifest, window_starts
from .output import complete, export_scene, validate_arrays
from .backend.alignment import align_windows
from .backend.posefree_gca_runtime import atomic_json
from .processes import Processes


def inspect_gpu(count, config):
    check_assets(config)
    # Probe in a fresh process. The scheduler itself never creates a CUDA context.
    script = '''import json, torch
names=[]
for i in range(torch.cuda.device_count()):
 with torch.cuda.device(i):
  names.append(dict(name=torch.cuda.get_device_name(i), bf16=torch.cuda.is_bf16_supported(),
                    memory_gib=torch.cuda.get_device_properties(i).total_memory/1024**3))
print(json.dumps(names))
'''
    r = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    names = json.loads(r.stdout.strip().splitlines()[-1])
    selected = gpu_selectors(count, os.environ.get("CUDA_VISIBLE_DEVICES"), len(names))
    if any(not d["bf16"] for d in names[:count]):
        raise ValueError("the preserved Omega backend requires native BF16 support")
    print(json.dumps(dict(selected_gpus=selected, devices=names[:count]), indent=2), flush=True)
    return selected


def doctor(config, num_gpus):
    devices = inspect_gpu(num_gpus, config)
    omega, moge = model_paths()
    script = '''import sys, torch, torchvision
sys.path[:0]=sys.argv[1:3]
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import _balanced_target_shape, _crop_to_supported_aspect_ratio, _load_rgb_image, _pad_images_to_common_size
from moge.model.v2 import MoGeModel
print("MODEL_IMPORTS=PASS", "torch="+torch.__version__, "torchvision="+torchvision.__version__)
'''
    subprocess.run([sys.executable, "-c", script, str(omega), str(moge)], check=True)
    print("DOCTOR=PASS (imports/assets/GPU capabilities; run a scene to verify actual inference)", flush=True)
    return devices


def contract_for(config, scenes):
    code = hashlib.sha256()
    for directory in (ROOT / "adagent_posefree", ROOT / "vendor"):
        for path in sorted(directory.rglob("*.py")):
            code.update(str(path.relative_to(ROOT)).encode())
            code.update(path.read_bytes())
    inputs = hashlib.sha256()
    for scene in scenes:
        for frame in scene["frames"]:
            for camera in scene["camera_ids"]:
                path = Path(frame["images"][camera])
                stat = path.stat()
                inputs.update(json.dumps([str(path), stat.st_size, stat.st_mtime_ns]).encode())
    checkpoints = {}
    for key in ("omega_checkpoint", "moge_checkpoint"):
        path = Path(config[key])
        stat = path.stat()
        checkpoints[key] = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    # Card count and CPU thread count may change on resume; numerical config may not.
    numerical = {k: v for k, v in config.items() if k not in
                 ("cpu_threads", "min_free_gib_per_gpu", "keep_intermediates", "window_batch_size", "raw_compression")}
    values = dict(schema_version=2, scene_count=len(scenes), camera_input_protocol="all_rgb",
                  metric_method="Omega + MoGe-2 + GCA + overlap SE(3)", ufo_image_size=config["output_image_size"],
                  numerical_config=numerical, manifest_hash=digest(scenes), image_stat_hash=inputs.hexdigest(),
                  source_hash=code.hexdigest(), checkpoints=checkpoints,
                  gt_camera_pose_used=False, gt_intrinsics_used=False, gt_depth_used=False, gt_ego_pose_used=False)
    return dict(values, run_signature=digest(values))


def valid_window(path, manifest):
    try:
        with np.load(path, allow_pickle=False) as x:
            if str(x["scene_name"].item()) != manifest["scene_name"] or str(x["coordinate_frame"].item()) != "rig_local_metric":
                return False
            keys = [(e["frame_id"], e["camera_id"]) for e in manifest["images"]]
            if list(zip(x["frame_ids"].tolist(), x["camera_ids"].astype(str).tolist())) != keys:
                return False
            p, k = x["omega_c2w_rig_local"], x["predicted_intrinsics_ufo"]
            return (p.shape == (len(keys), 4, 4) and k.shape == (len(keys), 3, 3)
                    and np.isfinite(p).all() and np.isfinite(k).all())
    except Exception:
        return False


def process_scene(scene, root, config, signature, env, processes):
    name = scene["scene_name"]
    if complete(root, scene, signature):
        return
    work = root / ".work" / name
    logs = root / "logs" / name
    work.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    starts = window_starts(len(scene["frames"]), config["window_frames"], config["window_stride"])
    pending = []
    for start in starts:
        manifest = window_manifest(scene, start, config)
        path = work / "manifests" / f"start_{start:03d}" / f"{name}.json"
        window = work / "windows" / path.parent.name / name / "omega_pose_override.npz"
        if not valid_window(window, manifest):
            atomic_json(path, manifest)
            pending.append(path)
    for batch in range(0, len(pending), config["window_batch_size"]):
        if processes.stop.is_set():
            raise RuntimeError("cancelled")
        paths = pending[batch:batch + config["window_batch_size"]]
        listing = work / f"batch_{batch:04d}.txt"
        listing.write_text("".join(str(p)+"\n" for p in paths))
        job = work / f"batch_{batch:04d}.json"
        atomic_json(job, dict(config=config, list=str(listing), raw=str(work / "raw"),
                             scale=str(work / "scale"), windows=str(work / "windows")))
        for stage in ("omega", "gca", "metric"):
            log_path = logs / f"batch_{batch:04d}_{stage}.log"
            print(f"[stage] scene={name} stage={stage} log={log_path}", flush=True)
            started = time.monotonic()
            with log_path.open("a", buffering=1) as log:
                processes.run([sys.executable, "-u", "-m", "adagent_posefree.backend.worker_stage",
                               "--job", str(job), "--stage", stage], env, log)
            print(f"[stage/done] scene={name} stage={stage} seconds={time.monotonic()-started:.2f}", flush=True)
        for path in paths:
            manifest = json.loads(path.read_text())
            if not valid_window(work / "windows" / path.parent.name / name / "omega_pose_override.npz", manifest):
                raise RuntimeError("metric camera validation failed; raw files retained")
        if not config["keep_intermediates"]:
            for path in paths:
                for directory, suffix in (("raw", ".npz"), ("scale", ".json")):
                    (work / directory / path.parent.name / (name + suffix)).unlink(missing_ok=True)
        listing.unlink(missing_ok=True)
        job.unlink(missing_ok=True)
    cameras, report = align_windows(work / "windows", name, starts, config["min_overlap_poses"])
    export_scene(root, scene, cameras, report, config, signature)
    if not config["keep_intermediates"]:
        shutil.rmtree(work)
    print(f"[scene/done] {name}", flush=True)


def prepare(scenes, config, output, num_gpus, check_only=False):
    devices = doctor(config, num_gpus)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free / 1024**3
    minimum = num_gpus * config["min_free_gib_per_gpu"]
    print(f"filesystem_free_gib={free:.1f} reserve_gib={minimum:.1f}; project quota is not measured", flush=True)
    if free < minimum:
        raise RuntimeError("insufficient filesystem free space for configured concurrent workers")
    contract = contract_for(config, scenes)
    contract_path = root / ".posefree_contract.json"
    if contract_path.exists():
        saved = json.loads(contract_path.read_text())
        if saved.get("run_signature") != contract["run_signature"]:
            raise RuntimeError("output belongs to a different input/config/code/assets run; use a new output directory")
    elif any(p.name != ".run.lock" for p in root.iterdir()):
        raise RuntimeError("output must be empty or an existing standalone run; do not use the legacy running cache")
    for scene in scenes:
        starts = window_starts(len(scene["frames"]), config["window_frames"], config["window_stride"])
        for a, b in zip(starts, starts[1:]):
            if (min(a + config["window_frames"], len(scene["frames"]))-b)*len(scene["camera_ids"]) < config["min_overlap_poses"]:
                raise ValueError(f"insufficient overlap for {scene['scene_name']}; reduce window_stride")
    pending = [s for s in scenes if not complete(root, s, contract["run_signature"])]
    print(f"validated_final={len(scenes)-len(pending)}/{len(scenes)} pending={len(pending)}", flush=True)
    if check_only:
        print("PREPARE_CHECK=PASS (no inference launched)", flush=True)
        return
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recheck after obtaining the shared filesystem lock.
        if contract_path.exists() and json.loads(contract_path.read_text()).get("run_signature") != contract["run_signature"]:
            raise RuntimeError("another run changed the output contract")
        atomic_json(contract_path, contract)
        atomic_json(root / "input_manifest.json", dict(schema_version=1, scenes=scenes))
        pending = [s for s in scenes if not complete(root, s, contract["run_signature"])]
        processes = Processes()
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        for sig in old_handlers:
            signal.signal(sig, lambda *_: processes.cancel())
        queue, queue_lock = iter(pending), threading.Lock()

        def worker(device):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED="1")
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                        "UFO_OMEGA_CPU_THREADS", "UFO_GCA_CPU_THREADS"):
                env[key] = str(config["cpu_threads"])
            env["UFO_OMEGA_RAW_COMPRESSION"] = config["raw_compression"]
            env["UFO_GCA_STATS_DEVICE"] = config["gca_stats_device"]
            try:
                while not processes.stop.is_set():
                    with queue_lock:
                        scene = next(queue, None)
                    if scene is None:
                        return
                    if shutil.disk_usage(root).free / 1024**3 < minimum:
                        raise RuntimeError("filesystem free space fell below reserve")
                    print(f"[worker] gpu={device} scene={scene['scene_name']}", flush=True)
                    process_scene(scene, root, config, contract["run_signature"], env, processes)
            except BaseException:
                processes.cancel()
                raise
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
                futures = [pool.submit(worker, d) for d in devices]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            if processes.stop.is_set():
                raise RuntimeError("interrupted; resume with the same manifest/config/output")
        finally:
            processes.cancel()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    print("PREPARE_DONE", flush=True)
