"""SE(3) overlap alignment from the running pipeline (scale is preserved)."""
import numpy as np
from pathlib import Path

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


def align_windows(root, scene, starts, min_overlap=9):
    """Preserve the R9 SE(3)/median consensus, including robust inlier rules."""
    observations, native_obs, k_obs, reports = {}, {}, {}, []

    def add(window):
        for i, (frame, camera) in enumerate(zip(window["frame_ids"], window["camera_ids"])):
            key = (int(frame), str(camera))
            observations.setdefault(key, []).append(window["opencv"][i])
            native_obs.setdefault(key, []).append(window["native"][i])
            k_obs.setdefault(key, []).append(window["K"][i])

    for position, start in enumerate(starts):
        window = load_window(Path(root) / f"start_{start:03d}" / scene / "omega_pose_override.npz")
        if position == 0:
            reports.append(dict(start=start, scale=1.0, overlap_poses=0,
                                translation_median_m=0.0, rotation_median_deg=0.0))
            add(window)
            continue
        local = {(int(f), str(c)): i for i, (f, c) in enumerate(zip(window["frame_ids"], window["camera_ids"]))}
        common = sorted(set(local) & set(observations))
        if len(common) < min_overlap:
            raise ValueError(f"window {start}: only {len(common)} overlapping cameras; need {min_overlap}")
        src = np.stack([window["opencv"][local[k]] for k in common])
        dst = np.stack([consensus_pose(observations[k]) for k in common])
        rotation, translation = fit_se3(src, dst)
        aligned = apply_se3(src, rotation, translation)
        terr = np.linalg.norm(aligned[:, :3, 3] - dst[:, :3, 3], axis=1)
        rerr = np.asarray([rotation_error_deg(a[:3, :3], b[:3, :3]) for a, b in zip(aligned, dst)])
        tmed, rmed = np.median(terr), np.median(rerr)
        inliers = ((terr <= tmed + 3 * (np.median(np.abs(terr-tmed)) + 1e-6)) &
                   (rerr <= rmed + 3 * (np.median(np.abs(rerr-rmed)) + 1e-6)))
        if inliers.sum() >= min_overlap:
            rotation, translation = fit_se3(src[inliers], dst[inliers])
            diagnostic_scale = diagnostic_sim3_scale(src[inliers], dst[inliers], rotation)
        else:
            diagnostic_scale = diagnostic_sim3_scale(src, dst, rotation)
        window["opencv"] = apply_se3(window["opencv"], rotation, translation)
        window["native"] = apply_se3(window["native"], rotation, translation)
        aligned = window["opencv"][[local[k] for k in common]]
        terr = np.linalg.norm(aligned[:, :3, 3] - dst[:, :3, 3], axis=1)
        rerr = np.asarray([rotation_error_deg(a[:3, :3], b[:3, :3]) for a, b in zip(aligned, dst)])
        reports.append(dict(start=start, scale=1.0, diagnostic_sim3_scale=float(diagnostic_scale) if np.isfinite(diagnostic_scale) else None,
                            translation_median_m=float(np.median(terr)), translation_max_m=float(np.max(terr)),
                            rotation_median_deg=float(np.median(rerr)), rotation_max_deg=float(np.max(rerr)), overlap_poses=len(common)))
        add(window)
    keys = sorted(observations, key=lambda k: (k[0], (0, int(k[1])) if k[1].isdigit() else (1, k[1])))
    return dict(frame_ids=np.asarray([k[0] for k in keys], np.int64),
                camera_ids=np.asarray([k[1] for k in keys]),
                c2w=np.stack([consensus_pose(observations[k]) for k in keys]).astype(np.float32),
                camera_to_world_dataset=np.stack([consensus_pose(native_obs[k]) for k in keys]).astype(np.float32),
                K=np.stack([np.median(np.stack(k_obs[k]), axis=0) for k in keys]).astype(np.float32)), reports
