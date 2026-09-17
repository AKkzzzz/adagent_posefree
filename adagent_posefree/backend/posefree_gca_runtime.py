"""GCA execution with bounded MoGe reuse, timings and CPU/CUDA checks.

Model resolution, Top-N selection, confidence percentile, mask rules and
median scale estimator are inherited from the existing source repository.
Only execution placement and reuse of per-image MoGe results change.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


class ByteLRU:
    def __init__(self, budget):
        self.budget = max(0, int(budget))
        self.used = 0
        self.data = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key):
        if key not in self.data:
            self.misses += 1
            return None
        self.hits += 1
        self.data.move_to_end(key)
        return self.data[key][0]

    def put(self, key, value, size):
        if key in self.data:
            _, old_size = self.data.pop(key)
            self.used -= old_size
        if size > self.budget or self.budget == 0:
            return
        while self.used + size > self.budget:
            _, (_, removed) = self.data.popitem(last=False)
            self.used -= removed
        self.data[key] = (value, size)
        self.used += size


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            f.write(encoded)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def valid_scale(path):
    try:
        report = json.loads(path.read_text())
        scale = float(report["global_scale"])
        return (math.isfinite(scale) and scale > 0
                and report["method"] == "gca_metric_scale_adapted_to_vggt_omega"
                and report["num_ratio_pixels"] >= 100)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def rank_images(conf, top_n, percentile, torch):
    # Keep the old masked reduction and reverse tuple-sort tie rule.
    thresholds, scores = [], []
    for i in range(len(conf)):
        threshold = torch.quantile(conf[i].float(), percentile)
        thresholds.append(threshold)
        scores.append((conf[i][conf[i] > threshold].sum().item(), i))
    scores.sort(reverse=True)
    selected = [i for _, i in scores[:min(top_n, len(scores))]]
    return selected, thresholds, scores


def close_ranking(scores, top_n):
    # CUDA and CPU sums need not be bitwise identical. Near ties are resolved
    # with the CPU reference, including ties at the selection boundary.
    head = scores[:min(top_n + 1, len(scores))]
    return any(abs(a[0] - b[0]) <= 1e-4 * max(1.0, abs(a[0]), abs(b[0]))
               for a, b in zip(head, head[1:]))


def summarize(depth, conf, selected, thresholds, metric, entries, torch):
    all_ratios, rows = [], []
    for idx in selected:
        metric_depth, metric_mask = metric[idx]
        metric_depth = metric_depth.to(depth.device)
        metric_mask = metric_mask.to(depth.device)
        valid = ((conf[idx] > thresholds[idx]) & metric_mask
                 & torch.isfinite(metric_depth) & torch.isfinite(depth[idx])
                 & (depth[idx] > 1e-4) & (metric_depth > 1e-4))
        count = int(valid.sum().item())
        if count < 100:
            continue
        ratios = (metric_depth[valid] / depth[idx][valid]).float()
        rows.append({"index": idx, "frame": int(entries[idx]["frame_id"]),
                     "camera": str(entries[idx]["camera_id"]), "valid_pixels": count,
                     "median_scale": float(torch.median(ratios).item())})
        all_ratios.append(ratios)
    if not all_ratios:
        raise RuntimeError("GCA scale estimation failed: no image has 100 valid pixels")
    ratios = torch.cat(all_ratios)
    scale = float(torch.median(ratios).item())
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"invalid GCA scale: {scale}")
    logs = torch.log(ratios)
    log_mad = float(torch.median(torch.abs(logs - torch.median(logs))).item())
    return {"global_scale": scale, "global_log_mad": log_mad,
            "selected": rows, "num_ratio_pixels": int(ratios.numel())}


def compare_stats(actual, reference):
    for name in ("num_ratio_pixels",):
        if actual[name] != reference[name]:
            raise RuntimeError(f"GCA CPU/CUDA check failed: {name}")
    for name in ("global_scale", "global_log_mad"):
        if not math.isclose(actual[name], reference[name], rel_tol=1e-5, abs_tol=1e-6):
            raise RuntimeError(f"GCA CPU/CUDA check failed: {name}: {actual[name]} vs {reference[name]}")
    if len(actual["selected"]) != len(reference["selected"]):
        raise RuntimeError("GCA CPU/CUDA check failed: selected row count")
    for a, b in zip(actual["selected"], reference["selected"]):
        for name in ("index", "frame", "camera", "valid_pixels"):
            if a[name] != b[name]:
                raise RuntimeError(f"GCA CPU/CUDA check failed: selected {name}")
        if not math.isclose(a["median_scale"], b["median_scale"], rel_tol=1e-5, abs_tol=1e-6):
            raise RuntimeError("GCA CPU/CUDA check failed: image median_scale")


class MoGeCache:
    def __init__(self, model, torch, Image, budget):
        self.model, self.torch, self.Image = model, torch, Image
        self.cache = ByteLRU(budget)
        self.verified = False

    def infer(self, path):
        with self.Image.open(path) as image:
            rgb_np = np.asarray(image.convert("RGB")).copy()
        torch = self.torch
        rgb = torch.from_numpy(rgb_np).float().permute(2, 0, 1).to("cuda") / 255.0
        with torch.inference_mode():
            result = self.model.infer(rgb, resolution_level=9, use_fp16=True, apply_mask=False)
        # Native image coordinates: resize for each window separately later.
        return result["depth"].float().cpu(), result["mask"].cpu().bool()

    def get(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self.infer(path)
        if self.cache.budget and not self.verified:
            repeated = self.infer(path)
            if (not self.torch.allclose(result[0], repeated[0], rtol=1e-5, atol=1e-6, equal_nan=True)
                    or not self.torch.equal(result[1], repeated[1])):
                raise RuntimeError("MoGe repeat check failed; use UFO_GCA_CACHE_MB=0 to disable reuse")
            self.verified = True
            print("[gca/check] MoGe repeat PASS", flush=True)
        size = sum(x.numel() * x.element_size() for x in result)
        self.cache.put(key, result, size)
        return result


def run_gca(a, paths):
    import torch
    from PIL import Image
    from .geometry import TOP_N, CONF_PERCENTILE, transform_to_omega

    device = os.environ.get("UFO_GCA_STATS_DEVICE", "cuda")
    if device not in ("cpu", "cuda"):
        raise ValueError("UFO_GCA_STATS_DEVICE must be cpu or cuda")
    threads = int(os.environ.get("UFO_GCA_CPU_THREADS", "4"))
    verify_every = int(os.environ.get("UFO_GCA_VERIFY_EVERY", "50"))
    budget_mb = int(os.environ.get("UFO_GCA_CACHE_MB", "512"))
    if threads < 1 or verify_every < 1 or budget_mb < 0:
        raise ValueError("GCA threads/verify interval must be positive; cache budget must be nonnegative")
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else threads
    torch.set_num_threads(min(threads, affinity))
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(a.moge_repo))
    from moge.model.v2 import MoGeModel

    pending, skipped = [], 0
    for path in paths:
        out = a.scale_root / path.parent.name / (path.stem + ".json")
        if valid_scale(out):
            skipped += 1
        else:
            pending.append((path, out))
    print(f"[gca/start] pending={len(pending)} skipped={skipped} stats_device={device} "
          f"cpu_threads={torch.get_num_threads()} cache_mb={budget_mb}", flush=True)
    if not pending:
        return
    model = MoGeModel.from_pretrained(a.moge_model).eval().to("cuda")
    provider = MoGeCache(model, torch, Image, budget_mb * 1024**2)
    profile_path = a.raw_root.parent / "logs" / (a.manifest_list.stem + "_gca_profile.jsonl")
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    for number, (path, out) in enumerate(pending, 1):
        started = time.perf_counter()
        timings = {}
        mark = started

        def lap(name):
            nonlocal mark
            # All GPU work must finish before attributing time to a phase.
            torch.cuda.synchronize()
            now = time.perf_counter()
            timings[name] = now - mark
            mark = now

        manifest = json.loads(path.read_text())
        entries = manifest["images"]
        raw = a.raw_root / path.parent.name / (path.stem + ".npz")
        with np.load(raw, allow_pickle=False) as payload:
            if str(payload["scene_name"].item()) != manifest["scene_name"]:
                raise RuntimeError(f"raw scene mismatch: {raw}")
            depth_cpu = torch.from_numpy(payload["omega_depth_raw"]).float()
            conf_cpu = torch.from_numpy(payload["omega_depth_conf_raw"]).float()
        if depth_cpu.ndim == 4 and depth_cpu.shape[-1] == 1:
            depth_cpu = depth_cpu[..., 0]
        if conf_cpu.ndim == 4 and conf_cpu.shape[-1] == 1:
            conf_cpu = conf_cpu[..., 0]
        if (depth_cpu.ndim != 3 or depth_cpu.shape != conf_cpu.shape
                or len(depth_cpu) != len(entries)):
            raise RuntimeError(f"bad raw shape: {raw}")
        if not torch.isfinite(conf_cpu).all():
            raise RuntimeError(f"non-finite confidence: {raw}")
        lap("read_s")
        depth, conf = depth_cpu.to(device), conf_cpu.to(device)
        selected, thresholds, scores = rank_images(conf, TOP_N, CONF_PERCENTILE, torch)
        cpu_rank = None
        tie_fallback = device == "cuda" and close_ranking(scores, TOP_N)
        if tie_fallback:
            cpu_rank = rank_images(conf_cpu, TOP_N, CONF_PERCENTILE, torch)
            selected = cpu_rank[0]
            thresholds = [x.to(device) for x in cpu_rank[1]]
        lap("rank_s")
        check = device == "cuda" and (number == 1 or number % verify_every == 0)
        if check:
            if cpu_rank is None:
                cpu_rank = rank_images(conf_cpu, TOP_N, CONF_PERCENTILE, torch)
            if selected != cpu_rank[0]:
                raise RuntimeError("GCA Top-3 differs from CPU reference; retry with UFO_GCA_STATS_DEVICE=cpu")
        lap("verify_rank_s")
        before_hits, before_misses = provider.cache.hits, provider.cache.misses
        native = {idx: provider.get(entries[idx]["path"]) for idx in selected}
        lap("moge_s")
        # Preserve the original CPU interpolation implementation; cache native
        # depth/mask, never a window's confidence, ratios or metric scale.
        metric = {idx: (transform_to_omega(native[idx][0], depth_cpu.shape[-2:], is_mask=False),
                        transform_to_omega(native[idx][1], depth_cpu.shape[-2:], is_mask=True))
                  for idx in selected}
        lap("resize_s")
        report = summarize(depth, conf, selected, thresholds, metric, entries, torch)
        lap("estimate_s")
        if check:
            reference = summarize(depth_cpu, conf_cpu, cpu_rank[0], cpu_rank[1], metric, entries, torch)
            try:
                compare_stats(report, reference)
            except RuntimeError as exc:
                raise RuntimeError(f"{exc}; retry with UFO_GCA_STATS_DEVICE=cpu") from exc
            print(f"[gca/check] {path.parent.name} CPU/CUDA PASS", flush=True)
        lap("verify_scale_s")
        report.update({"method": "gca_metric_scale_adapted_to_vggt_omega", "source_method": "GCA estimate_scale",
                       "top_n": TOP_N, "relative_conf_percentile": CONF_PERCENTILE,
                       "uses_gt_geometry": False, "uses_gt_pose": False, "uses_gt_intrinsics": False,
                       "uses_gt_depth": False, "uses_camera_to_ego": False,
                       "omega_source": "cached_rgb_only_forward",
                       "execution": {"stats_device": device, "cpu_cuda_checked": check,
                                     "cpu_tie_fallback": tie_fallback, "moge_cache_mb": budget_mb}})
        atomic_json(out, report)
        lap("write_s")
        timings.update({"total_s": time.perf_counter() - started, "pid": os.getpid(),
                        "scene": manifest["scene_name"], "start": path.parent.name,
                        "stats_device": device, "cache_hits": provider.cache.hits - before_hits,
                        "cache_misses": provider.cache.misses - before_misses,
                        "cache_used_mb": provider.cache.used / 1024**2,
                        "cpu_cuda_checked": check, "cpu_tie_fallback": tie_fallback})
        with profile_path.open("a") as log:
            log.write(json.dumps(timings, allow_nan=False) + "\n")
        phases = " ".join(f"{key}={value:.2f}" for key, value in timings.items() if key.endswith("_s"))
        print(f"[batch/gca] {manifest['scene_name']} {path.parent.name} {number}/{len(pending)} "
              f"{phases} cache_hit={timings['cache_hits']}/{len(selected)}", flush=True)
