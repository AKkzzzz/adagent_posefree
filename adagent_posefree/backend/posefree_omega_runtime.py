"""Single-window Omega inference with overlapping lossless writes and timing.

The window list, image order, resolution, precision and model forward remain
unchanged. This is not multi-window inference or reduced-stride inference.
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from .posefree_omega_io import RawWriter, IntrinsicsGeometryCache, require_disk_budget


def run_omega(a, paths, raw_ok, prepare_rgb_cache, stack_cached_rgb, homo):
    stage_started = time.perf_counter()
    import torch
    threads = int(os.environ.get("UFO_OMEGA_CPU_THREADS", "4"))
    compression = os.environ.get("UFO_OMEGA_RAW_COMPRESSION", "stored")
    if threads < 1 or compression not in ("stored", "deflate"):
        raise ValueError("Omega threads must be positive; raw compression must be stored or deflate")
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else threads
    torch.set_num_threads(min(threads, affinity))
    torch.set_num_interop_threads(1)
    pending, skipped = [], 0
    for path in paths:
        out = a.raw_root / path.parent.name / (path.stem + ".npz")
        if raw_ok(out):
            skipped += 1
        else:
            pending.append((path, out, json.loads(path.read_text())))
    print(f"[omega/start] pending={len(pending)} skipped={skipped} inference_batch=1 "
          f"cpu_threads={torch.get_num_threads()} raw_compression={compression}", flush=True)
    if not pending:
        return

    sys.path.insert(0, str(a.omega_repo))
    sys.path.insert(0, str(a.omega_repo / "tools"))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import (
        _balanced_target_shape, _crop_to_supported_aspect_ratio,
        _load_rgb_image, _pad_images_to_common_size,
    )
    from vggt_omega.utils.pose_enc import encoding_to_camera
    from .geometry import transform_intrinsics_to_ufo
    from torchvision import transforms as TF
    from PIL import Image

    unique_paths = list(dict.fromkeys(e["path"] for _, _, m in pending for e in m["images"]))
    print(f"[omega/cache] preparing {len(unique_paths)} unique images", flush=True)
    image_cache = prepare_rgb_cache(unique_paths, _load_rgb_image, _crop_to_supported_aspect_ratio,
                                    _balanced_target_shape, TF.ToTensor(), Image)
    geometry = IntrinsicsGeometryCache(Image, _balanced_target_shape)
    if compression == "stored":
        estimated = 0
        for _, _, manifest in pending:
            images = [image_cache[e["path"]] for e in manifest["images"]]
            h, w = max(x.shape[-2] for x in images), max(x.shape[-1] for x in images)
            estimated += len(images) * h * w * 4 * 2 + 65536
        a.raw_root.mkdir(parents=True, exist_ok=True)
        needed = require_disk_budget(a.raw_root, estimated, 1)
        print(f"[omega/disk] batch raw budget including reserve={needed / 1024**3:.1f} GiB", flush=True)

    model = VGGTOmega().eval()
    state = torch.load(a.omega_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    del state
    model = model.to("cuda")
    torch.cuda.synchronize()
    startup_s = time.perf_counter() - stage_started
    print(f"[omega/ready] startup_s={startup_s:.2f}", flush=True)
    log_root = a.raw_root.parent / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    profile = log_root / (a.manifest_list.stem + "_omega_profile.jsonl")
    completed = 0
    loop_started = time.perf_counter()

    def record(result):
        nonlocal completed
        if result is None:
            return
        completed += 1
        result["stage_elapsed_s"] = time.perf_counter() - stage_started
        result["completed"] = completed
        result["average_loop_s"] = (time.perf_counter() - loop_started) / completed
        with profile.open("a") as f:
            f.write(json.dumps(result, allow_nan=False) + "\n")
        fields = " ".join(f"{key}={result[key]:.2f}" for key in
                          ("input_s", "forward_s", "download_s", "intrinsics_s",
                           "write_s", "wait_writer_s", "average_loop_s"))
        print(f"[batch/omega] {result['scene']} {result['start']} "
              f"{completed}/{len(pending)} {fields} raw_mb={result['raw_bytes'] / 1024**2:.1f}", flush=True)

    writer = RawWriter(compressed=compression == "deflate")
    try:
        for number, (path, out, manifest) in enumerate(pending, 1):
            mark = time.perf_counter()
            times = {}

            def lap(name):
                nonlocal mark
                torch.cuda.synchronize()
                now = time.perf_counter()
                times[name] = now - mark
                mark = now

            entries = manifest["images"]
            rgb_paths = [e["path"] for e in entries]
            torch.cuda.reset_peak_memory_stats()
            images = stack_cached_rgb(rgb_paths, image_cache, _pad_images_to_common_size, torch).to("cuda")
            lap("input_s")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(images)
                extri, intri = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
            lap("forward_s")
            depth = pred["depth"][0].float().cpu().numpy()
            conf = pred["depth_conf"][0].float().cpu().numpy()
            extri_cpu = extri[0].float().cpu().numpy()
            k = intri[0].float().cpu().numpy()
            lap("download_s")
            w2c = homo(extri_cpu)
            c2w = np.linalg.inv(w2c)
            ku = geometry.transform(k, rgb_paths, manifest["ufo_image_size"])
            if number == 1:
                reference, _ = transform_intrinsics_to_ufo(k, rgb_paths, manifest["ufo_image_size"], image_resolution=512)
                if not np.array_equal(ku, reference):
                    raise RuntimeError("Cached K geometry differs from source implementation; no raw file written")
                print(f"[omega/check] {path.parent.name} cached intrinsics PASS", flush=True)
            lap("intrinsics_s")
            times.update({"scene": manifest["scene_name"], "start": path.parent.name,
                          "pid": os.getpid(), "raw_compression": compression,
                          "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                          "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2})
            payload = dict(scene_name=np.asarray(manifest["scene_name"]), scope=np.asarray("all"),
                           frame_ids=np.asarray([e["frame_id"] for e in entries], dtype=np.int32),
                           camera_ids=np.asarray([e["camera_id"] for e in entries]),
                           roles=np.asarray([e["role"] for e in entries]),
                           omega_w2c_raw=w2c.astype(np.float32), omega_c2w_raw=c2w.astype(np.float32),
                           predicted_intrinsics_ufo=ku.astype(np.float32),
                           omega_depth_raw=depth.astype(np.float32), omega_depth_conf_raw=conf.astype(np.float32))
            # Release activations, but retain CUDA's allocator cache.
            del images, pred, extri, intri
            record(writer.collect())
            writer.submit(out, payload, times)
        record(writer.collect())
    finally:
        writer.close()
    wall = time.perf_counter() - stage_started
    print(f"[omega/done] new_windows={completed} reused={skipped} wall_s={wall:.2f} "
          f"startup_s={startup_s:.2f} effective_s_per_new_window={wall / completed:.2f}", flush=True)
