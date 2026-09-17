#!/usr/bin/env python3
"""Prepare RGB-only pose-free cameras for all Full-Waymo training scenes.

This driver deliberately invokes the already validated main-branch tools:
RGB manifest -> VGGT-Omega -> MoGe/GCA scale -> metric window pose -> overlap
SE(3) global alignment.  It never passes GT camera fields to those tools.
Run it in the 4090 environment; write --cache-root to a path visible to the
later H200 container.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-ufo-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--annotation-file", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--omega-repo", type=Path, required=True)
    p.add_argument("--omega-checkpoint", type=Path, required=True)
    p.add_argument("--moge-repo", type=Path, required=True)
    p.add_argument("--moge-model", type=Path, required=True)
    p.add_argument("--scene-first", type=int, default=0)
    p.add_argument("--scene-last", type=int, default=797)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--window-batch-size",
        type=int,
        default=0,
        help="Keep one Omega/MoGe process per this many windows (0 keeps legacy mode).",
    )
    return p.parse_args()


def path_from(data_root, value):
    p = Path(value)
    return p if p.is_absolute() else data_root / p


def run(cmd, cwd, log_path):
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(str(x) for x in cmd)
    print(f"[posefree] {rendered}", flush=True)
    with log_path.open("ab") as log:
        log.write((f"\n$ {rendered}\n").encode())
        log.flush()
        try:
            subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            print(f"[posefree/error] subprocess log: {log_path}", file=sys.stderr, flush=True)
            with log_path.open("rb") as saved:
                saved.seek(0, os.SEEK_END)
                saved.seek(max(0, saved.tell() - 16000))
                print(saved.read().decode(errors="replace"), file=sys.stderr, flush=True)
            raise
    elapsed = time.perf_counter() - started
    with (log_path.parent / "prepare_stage_profile.jsonl").open("a") as profile:
        profile.write(json.dumps({"log": log_path.name, "wall_s": elapsed,
                                  "completed_unix": time.time(), "pid": os.getpid()}) + "\n")
    if "--stage" in cmd or "alignment" in log_path.name:
        print(f"[posefree/done] {log_path.name} wall_s={elapsed:.2f}", flush=True)


def valid_npz(path, expected_scene):
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as x:
            return (
                str(x["scene_name"].item()) == expected_scene
                and str(x["coordinate_frame"].item()) == "rig_local_metric"
                and x["frame_ids"].ndim == 1
                and x["predicted_intrinsics_ufo"].shape[0] == x["frame_ids"].shape[0]
            )
    except Exception:
        return False


def main():
    a = parse_args()
    cfg = json.loads(a.config.read_text())
    ufo = a.source_ufo_root.resolve()
    tools = ufo / "tools"
    lines = [x.strip() for x in a.annotation_file.read_text().splitlines() if x.strip()]
    if len(lines) != 798:
        raise RuntimeError(f"expected 798 scenes, got {len(lines)}")
    if not (0 <= a.scene_first <= a.scene_last < len(lines)):
        raise ValueError("scene range must be within 0..797")
    if a.plan_only:
        total_windows = 0
        for scene_index in range(a.scene_first, a.scene_last + 1):
            scene = json.loads(path_from(a.data_root, lines[scene_index]).read_text())
            frames_per_chunk = int(cfg["timespan"] * scene["fps"])
            total_frames = frames_per_chunk * cfg["num_target_chunks"]
            starts = int(scene["num_timesteps"]) - total_frames + 1
            total_windows += max(starts, 0)
        print(json.dumps({
            "scenes": a.scene_last - a.scene_first + 1,
            "total_windows": total_windows,
            "scene_first": a.scene_first,
            "scene_last": a.scene_last,
            "rgb_only": True,
            "gt_camera_pose_used": False,
        }, indent=2))
        return
    for p in (a.python, a.omega_repo, a.omega_checkpoint, a.moge_repo, a.moge_model, a.data_root, a.annotation_file):
        if not p.exists():
            raise FileNotFoundError(p)
    a.cache_root.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "scene_count": 798,
        "camera_input_protocol": "all_rgb",
        "metric_method": "Omega + MoGe-2 + GCA + overlap SE(3)",
        "ufo_image_size": list(cfg["input_size"]),
        "source_ufo_root": str(ufo),
        "omega_checkpoint": str(a.omega_checkpoint.resolve()),
        "moge_model": str(a.moge_model.resolve()),
        "gt_camera_pose_used": False,
        "gt_intrinsics_used": False,
        "gt_depth_used": False,
        "gt_ego_pose_used": False,
    }
    (a.cache_root / ".posefree_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    manifest_root = a.cache_root / "manifests_rgb_only"
    raw_root = a.cache_root / "omega_raw_all"
    scale_root = a.cache_root / "gca_omega_scale_all"
    window_root = a.cache_root / "omega_gca_metric_all"
    batch_tool = Path(__file__).with_name("posefree_batch_stage.py")
    if a.window_batch_size < 0:
        raise ValueError("--window-batch-size must be >= 0")
    for scene_index in range(a.scene_first, a.scene_last + 1):
        scene_path = path_from(a.data_root, lines[scene_index])
        scene = json.loads(scene_path.read_text())
        scene_name = scene["scene_name"]
        frames_per_chunk = int(cfg["timespan"] * scene["fps"])
        total_frames = frames_per_chunk * cfg["num_target_chunks"]
        last_start = int(scene["num_timesteps"]) - total_frames
        if last_start < 0:
            raise RuntimeError(f"scene {scene_name} is shorter than the training window")
        print(f"\n========== SCENE {scene_index}/797 {scene_name}: starts 0..{last_start} ==========", flush=True)
        aligned = a.cache_root / "global_aligned" / scene_name / "omega_pose_override.npz"
        if not a.force and aligned.is_file():
            print(f"[skip] global scene cache exists: {aligned}", flush=True)
            continue
        pending = []
        manifest_jobs = []
        for start in range(last_start + 1):
            tag = f"start_{start:03d}"
            manifest = manifest_root / tag / f"{scene_name}.json"
            raw = raw_root / tag / f"{scene_name}.npz"
            scale = scale_root / tag / f"{scene_name}.json"
            final = window_root / tag / scene_name / "omega_pose_override.npz"
            log = a.cache_root / "logs" / f"scene_{scene_index:04d}_start_{start:03d}.log"
            if not a.force and valid_npz(final, scene_name):
                continue
            if not manifest.is_file():
                manifest_jobs.append({"start_index": start, "output": str(manifest)})
            pending.append((manifest, raw, scale, final, log))

        if manifest_jobs:
            jobs_path = a.cache_root / "batch_lists" / f"scene_{scene_index:04d}_manifest_jobs.json"
            jobs_path.parent.mkdir(parents=True, exist_ok=True)
            jobs_path.write_text(json.dumps(manifest_jobs))
            run([str(a.python), str(Path(__file__).with_name("export_posefree_manifest_batch.py")),
                 "--source-ufo-root", str(ufo), "--config", str(a.config),
                 "--data-root", str(a.data_root), "--annotation-file", str(a.annotation_file),
                 "--scene-index", str(scene_index), "--jobs", str(jobs_path)],
                ufo, a.cache_root / "logs" / f"scene_{scene_index:04d}_manifests.log")
            jobs_path.unlink(missing_ok=True)

        if pending and a.window_batch_size:
            # A bounded batch avoids accumulating all raw depth tensors for a
            # scene while reducing model construction from once/window to
            # once/batch.  The numerical operations remain per-window.
            for batch_start in range(0, len(pending), a.window_batch_size):
                batch = pending[batch_start:batch_start + a.window_batch_size]
                print(
                    f"[posefree] scene={scene_index} windows "
                    f"{batch_start}..{batch_start + len(batch) - 1} "
                    f"batch_size={len(batch)}",
                    flush=True,
                )
                list_path = a.cache_root / "batch_lists" / f"scene_{scene_index:04d}_{batch_start:04d}.txt"
                list_path.parent.mkdir(parents=True, exist_ok=True)
                list_path.write_text("".join(str(row[0]) + "\n" for row in batch))
                common = [str(a.python), str(batch_tool), "--source-ufo-root", str(ufo),
                          "--manifest-list", str(list_path), "--raw-root", str(raw_root),
                          "--scale-root", str(scale_root), "--output-root", str(window_root)]
                run(common + ["--stage", "omega", "--omega-repo", str(a.omega_repo),
                              "--omega-checkpoint", str(a.omega_checkpoint)], ufo,
                    a.cache_root / "logs" / f"scene_{scene_index:04d}_batch_{batch_start:04d}_omega.log")
                run(common + ["--stage", "gca", "--moge-repo", str(a.moge_repo),
                              "--moge-model", str(a.moge_model)], ufo,
                    a.cache_root / "logs" / f"scene_{scene_index:04d}_batch_{batch_start:04d}_gca.log")
                run(common + ["--stage", "metric"], ufo,
                    a.cache_root / "logs" / f"scene_{scene_index:04d}_batch_{batch_start:04d}_metric.log")
                for _, raw, scale, final, _ in batch:
                    for temporary in (raw, scale):
                        temporary.unlink(missing_ok=True)
                list_path.unlink(missing_ok=True)
        elif pending:
            for manifest, raw, scale, final, log in pending:
                run([str(a.python), str(tools / "export_rgb_only_omega.py"),
                     "--manifest", str(manifest), "--omega-repo", str(a.omega_repo),
                     "--checkpoint", str(a.omega_checkpoint), "--output", str(raw)], ufo, log)
                run([str(a.python), str(tools / "diagnose_gca_omega_scale_cached.py"),
                     "--manifest", str(manifest), "--omega-npz", str(raw),
                     "--moge-repo", str(a.moge_repo), "--moge-model", str(a.moge_model),
                     "--output", str(scale)], ufo, log)
                run([str(a.python), str(tools / "export_rgb_metric_pose.py"),
                     "--manifest", str(manifest), "--omega-npz", str(raw),
                     "--scale-json", str(scale), "--output", str(final)], ufo, log)
            # The raw depth/confidence NPZ can be tens of MB per window.  It is
            # not needed after metric pose export and must not accumulate over
            # all 798 scenes.
                for temporary in (manifest, raw, scale):
                    temporary.unlink(missing_ok=True)
        if a.force or not aligned.is_file():
            run([str(a.python), str(tools / "build_global_pose_from_overlaps.py"),
                 "--input-root", str(window_root), "--scene", scene_name,
                 "--first", "0", "--last", str(last_start),
                 "--output-root", str(a.cache_root / "global_aligned")], ufo, a.cache_root / "logs" / f"scene_{scene_index:04d}_alignment.log")
        # Keep only the final global scene cache.  If alignment fails, these
        # files remain and the scene can resume without rerunning Omega.
        for start in range(last_start + 1):
            tag = f"start_{start:03d}"
            for temporary in (
                manifest_root / tag / f"{scene_name}.json",
                raw_root / tag / f"{scene_name}.npz",
                scale_root / tag / f"{scene_name}.json",
            ):
                temporary.unlink(missing_ok=True)
            shutil.rmtree(window_root / tag / scene_name, ignore_errors=True)
    print(f"POSEFREE_PREPARE_DONE cache_root={a.cache_root}", flush=True)


if __name__ == "__main__":
    main()
