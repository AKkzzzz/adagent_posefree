#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diagnose_gca_omega_scale import (
    TOP_N,
    CONF_PERCENTILE,
    transform_to_omega,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--omega-npz", type=Path, required=True)
    p.add_argument("--moge-repo", type=Path, required=True)
    p.add_argument("--moge-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    entries = manifest["images"]

    with np.load(args.omega_npz, allow_pickle=False) as x:
        depth = torch.from_numpy(x["omega_depth_raw"]).float()
        conf = torch.from_numpy(x["omega_depth_conf_raw"]).float()

    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]

    if depth.ndim != 3 or conf.ndim != 3:
        raise RuntimeError(
            f"unexpected cached shapes: depth={depth.shape}, conf={conf.shape}"
        )

    n_images, OH, OW = depth.shape
    if n_images != len(entries):
        raise RuntimeError(
            f"NPZ images={n_images}, manifest={len(entries)}"
        )

    # Exact GCA image confidence ranking.
    scores = []
    for i in range(n_images):
        c = conf[i]
        threshold = torch.quantile(c.float(), CONF_PERCENTILE)
        mask = c > threshold
        scores.append((c[mask].sum().item(), i))

    scores.sort(reverse=True)
    selected = [idx for _, idx in scores[:min(TOP_N, n_images)]]

    print("[GCA] selected Top-3:", selected, flush=True)

    sys.path.insert(0, str(args.moge_repo))
    from moge.model.v2 import MoGeModel

    device = "cuda"

    moge = MoGeModel.from_pretrained(
        args.moge_model
    ).eval().to(device)

    all_ratios = []
    rows = []

    for idx in selected:
        e = entries[idx]

        rgb_np = np.asarray(
            Image.open(e["path"]).convert("RGB")
        ).copy()

        rgb = (
            torch.from_numpy(rgb_np)
            .float()
            .permute(2, 0, 1)
            .to(device)
            / 255.0
        )

        with torch.inference_mode():
            out = moge.infer(
                rgb,
                resolution_level=9,
                use_fp16=True,
                apply_mask=False,
            )

        metric_depth = out["depth"].float().cpu()
        metric_mask = out["mask"].cpu().bool()

        metric_depth = transform_to_omega(
            metric_depth,
            (OH, OW),
            is_mask=False,
        )
        metric_mask = transform_to_omega(
            metric_mask,
            (OH, OW),
            is_mask=True,
        )

        relative_depth = depth[idx]
        relative_conf = conf[idx]

        threshold = torch.quantile(
            relative_conf.float(),
            CONF_PERCENTILE,
        )

        valid = (
            (relative_conf > threshold)
            & metric_mask
            & torch.isfinite(metric_depth)
            & torch.isfinite(relative_depth)
            & (relative_depth > 1e-4)
            & (metric_depth > 1e-4)
        )

        n = int(valid.sum())
        if n < 100:
            print(f"[skip] idx={idx} valid={n}")
            continue

        ratios = (
            metric_depth[valid] / relative_depth[valid]
        ).float()

        image_scale = torch.median(ratios).item()

        rows.append({
            "index": idx,
            "frame": int(e["frame_id"]),
            "camera": str(e["camera_id"]),
            "valid_pixels": n,
            "median_scale": float(image_scale),
        })
        all_ratios.append(ratios)

        print(
            f"idx={idx:02d} "
            f"frame={int(e['frame_id']):03d} "
            f"cam={e['camera_id']} "
            f"valid={n} scale={image_scale:.6f}",
            flush=True,
        )

    if not all_ratios:
        raise RuntimeError("GCA scale estimation failed")

    ratios = torch.cat(all_ratios)

    # Exact official GCA estimator.
    scale = torch.median(ratios).item()

    logs = torch.log(ratios)
    med_log = torch.median(logs)
    log_mad = torch.median(
        torch.abs(logs - med_log)
    ).item()

    report = {
        "method": "gca_metric_scale_adapted_to_vggt_omega",
        "source_method": "GCA estimate_scale",
        "top_n": TOP_N,
        "relative_conf_percentile": CONF_PERCENTILE,
        "global_scale": float(scale),
        "global_log_mad": float(log_mad),
        "selected": rows,
        "num_ratio_pixels": int(ratios.numel()),
        "uses_gt_geometry": False,
        "uses_gt_pose": False,
        "uses_gt_intrinsics": False,
        "uses_gt_depth": False,
        "uses_camera_to_ego": False,
        "omega_source": "cached_rgb_only_forward",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"GCA scale={scale:.6f} "
        f"logMAD={log_mad:.6f} "
        f"pixels={ratios.numel()}"
    )


if __name__ == "__main__":
    main()
