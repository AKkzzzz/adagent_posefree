#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def project_rotation(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def average_rotation(Rs):
    return project_rotation(np.sum(Rs, axis=0))


def rotation_error_deg(A, B):
    rel = A @ B.T
    c = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def consensus_pose(poses):
    poses = np.asarray(poses)
    out = np.eye(4)
    out[:3, :3] = average_rotation(poses[:, :3, :3])
    out[:3, 3] = np.median(poses[:, :3, 3], axis=0)
    return out


def fit_similarity(src, dst):
    # Rotation primarily from camera orientations.
    M = np.zeros((3, 3))
    for A, B in zip(src[:, :3, :3], dst[:, :3, :3]):
        M += B @ A.T
    R = project_rotation(M)

    xs = src[:, :3, 3]
    ys = dst[:, :3, 3]

    xc = np.median(xs, axis=0)
    yc = np.median(ys, axis=0)

    xr = (R @ (xs - xc).T).T
    yr = ys - yc

    denom = np.sum(xr * xr)
    if denom < 1e-10:
        scale = 1.0
    else:
        scale = float(np.sum(xr * yr) / denom)

    t = np.median(
        ys - scale * (R @ xs.T).T,
        axis=0,
    )

    return scale, R, t


def fit_se3(src, dst):
    """Align src camera poses to dst while preserving GCA metric scale."""
    M = np.zeros((3, 3))
    for A, B in zip(src[:, :3, :3], dst[:, :3, :3]):
        M += B @ A.T

    R = project_rotation(M)

    src_t = src[:, :3, 3]
    dst_t = dst[:, :3, 3]

    offsets = (
        dst_t
        - np.einsum("ij,nj->ni", R, src_t)
    )
    t = np.median(offsets, axis=0)

    return R, t


def apply_se3(poses, R, t):
    out = poses.copy()
    out[:, :3, :3] = np.einsum(
        "ij,njk->nik", R, poses[:, :3, :3]
    )
    out[:, :3, 3] = (
        np.einsum("ij,nj->ni", R, poses[:, :3, 3])
        + t
    )
    return out


def diagnostic_sim3_scale(src, dst, R):
    """Scale diagnostic only; never applied."""
    xs = src[:, :3, 3]
    ys = dst[:, :3, 3]

    xs = xs - xs.mean(axis=0)
    ys = ys - ys.mean(axis=0)

    xr = np.einsum("ij,nj->ni", R, xs)
    den = np.sum(xr * xr)

    if den < 1e-10:
        return float("nan")

    return float(np.sum(xr * ys) / den)


def apply_similarity(poses, scale, R, t):
    out = poses.copy()
    out[:, :3, :3] = np.einsum(
        "ij,njk->nik", R, poses[:, :3, :3]
    )
    out[:, :3, 3] = (
        scale * np.einsum(
            "ij,nj->ni", R, poses[:, :3, 3]
        )
        + t
    )
    return out


def load_window(path):
    with np.load(path, allow_pickle=False) as x:
        return {
            "frame_ids": x["frame_ids"].astype(int),
            "camera_ids": x["camera_ids"].astype(str),
            "opencv": x["omega_c2w_rig_local"].astype(np.float64),
            "native": x[
                "omega_camera_to_world_rig_local"
            ].astype(np.float64),
            "K": x["predicted_intrinsics_ufo"].astype(np.float64),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int, default=177)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()

    pose_obs = {}
    native_obs = {}
    K_obs = {}
    report = []

    def add_window(x):
        for i, (f, c) in enumerate(
            zip(x["frame_ids"], x["camera_ids"])
        ):
            key = (int(f), str(c))
            pose_obs.setdefault(key, []).append(x["opencv"][i])
            native_obs.setdefault(key, []).append(x["native"][i])
            K_obs.setdefault(key, []).append(x["K"][i])

    def global_pose(key):
        return consensus_pose(pose_obs[key])

    # start_000 defines the global gauge.
    first_path = (
        args.input_root
        / f"start_{args.first:03d}"
        / args.scene
        / "omega_pose_override.npz"
    )
    first = load_window(first_path)
    add_window(first)

    report.append({
        "start": args.first,
        "scale": 1.0,
        "translation_median_m": 0.0,
        "rotation_median_deg": 0.0,
        "overlap_poses": 0,
    })

    for start in range(args.first + 1, args.last + 1):
        path = (
            args.input_root
            / f"start_{start:03d}"
            / args.scene
            / "omega_pose_override.npz"
        )
        x = load_window(path)

        local_map = {
            (int(f), str(c)): i
            for i, (f, c) in enumerate(
                zip(x["frame_ids"], x["camera_ids"])
            )
        }

        common = sorted(set(local_map) & set(pose_obs))
        if len(common) < 9:
            raise RuntimeError(
                f"start {start}: only {len(common)} overlap poses"
            )

        src = np.stack([
            x["opencv"][local_map[k]]
            for k in common
        ])
        dst = np.stack([
            global_pose(k)
            for k in common
        ])

        # GCA has already recovered metric scale.
        # Cross-window alignment must preserve it: SE3 only.
        R, t = fit_se3(src, dst)

        diagnostic_scale = diagnostic_sim3_scale(src, dst, R)

        # First-pass robust residual filter.
        aligned_common = apply_se3(src, R, t)

        terr = np.linalg.norm(
            aligned_common[:, :3, 3] - dst[:, :3, 3],
            axis=1,
        )
        rerr = np.asarray([
            rotation_error_deg(a[:3, :3], b[:3, :3])
            for a, b in zip(aligned_common, dst)
        ])

        tmed = np.median(terr)
        tmad = np.median(np.abs(terr - tmed)) + 1e-6
        rmed = np.median(rerr)
        rmad = np.median(np.abs(rerr - rmed)) + 1e-6

        inlier = (
            (terr <= tmed + 3.0 * tmad)
            & (rerr <= rmed + 3.0 * rmad)
        )

        if inlier.sum() >= 9:
            R, t = fit_se3(
                src[inlier],
                dst[inlier],
            )
            diagnostic_scale = diagnostic_sim3_scale(
                src[inlier],
                dst[inlier],
                R,
            )

        aligned_opencv = apply_se3(
            x["opencv"], R, t
        )
        aligned_native = apply_se3(
            x["native"], R, t
        )

        x["opencv"] = aligned_opencv
        x["native"] = aligned_native

        # Final overlap diagnostics.
        final_common = aligned_opencv[
            [local_map[k] for k in common]
        ]

        terr = np.linalg.norm(
            final_common[:, :3, 3] - dst[:, :3, 3],
            axis=1,
        )
        rerr = np.asarray([
            rotation_error_deg(a[:3, :3], b[:3, :3])
            for a, b in zip(final_common, dst)
        ])

        report.append({
            "start": start,
            "scale": 1.0,
            "diagnostic_sim3_scale": float(diagnostic_scale),
            "translation_median_m": float(np.median(terr)),
            "translation_max_m": float(np.max(terr)),
            "rotation_median_deg": float(np.median(rerr)),
            "rotation_max_deg": float(np.max(rerr)),
            "overlap_poses": len(common),
        })

        add_window(x)

    keys = sorted(
        pose_obs,
        key=lambda k: (k[0], int(k[1])),
    )

    frame_ids = np.asarray([k[0] for k in keys], np.int32)
    camera_ids = np.asarray([k[1] for k in keys])

    c2w = np.stack([
        consensus_pose(pose_obs[k])
        for k in keys
    ])

    native = np.stack([
        consensus_pose(native_obs[k])
        for k in keys
    ])

    K = np.stack([
        np.median(
            np.stack(K_obs[k]),
            axis=0,
        )
        for k in keys
    ])

    out_dir = args.output_root / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / "omega_pose_override.npz"

    np.savez_compressed(
        out,
        scene_name=np.asarray(args.scene),
        scope=np.asarray("all"),
        coordinate_frame=np.asarray("global_metric"),
        metric_scale_source=np.asarray(
            "moge2_gca_plus_overlap_camera_se3"
        ),
        world_gauge=np.asarray(
            "start_000_first_timestamp_front_camera"
        ),
        frame_ids=frame_ids,
        camera_ids=camera_ids,
        omega_c2w_global_metric=c2w.astype(np.float32),
        omega_camera_to_world_global_metric=
            native.astype(np.float32),
        predicted_intrinsics_ufo=K.astype(np.float32),
    )

    report_path = (
        args.output_root / "alignment_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    scales = np.asarray([
        r["diagnostic_sim3_scale"]
        for r in report[1:]
        if np.isfinite(r["diagnostic_sim3_scale"])
    ])
    tm = np.asarray([
        r["translation_median_m"] for r in report[1:]
    ])
    rm = np.asarray([
        r["rotation_median_deg"] for r in report[1:]
    ])

    print("========================================")
    print("GLOBAL POSE ALIGNMENT PASS")
    print("frames :", len(set(frame_ids.tolist())))
    print("poses  :", len(keys))
    print(
        "diagnostic Sim3 min/median/max:",
        scales.min(), np.median(scales), scales.max()
    )
    print(
        "Tmed median/max:",
        np.median(tm), np.max(tm)
    )
    print(
        "Rmed median/max:",
        np.median(rm), np.max(rm)
    )
    print("output :", out)
    print("========================================")


if __name__ == "__main__":
    main()
