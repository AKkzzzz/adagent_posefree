#!/usr/bin/env python3
"""Read-only progress inspection; optionally create a separate Git source snapshot.

No inference, training, cache deletion or remote Git operations are performed.
Only --snapshot-out creates files, in a new destination outside the running repos.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

DEFAULT_REPO = Path('/inspire/hdd/global_user/guoluosong-253108120129/yx-ufo/R9UFO')
DEFAULT_SOURCE = Path('/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/ufoposefree')
CAMERAS = ('0', '1', '2')
H200_FILES = (
    'scripts/r9/prepare_posefree_4x4090.sh',
    'scripts/r9/prepare_posefree_8x4090.sh',
    'scripts/r9/run_r9_waymo_full_posefree.sh',
    'scripts/r9/train_r9_waymo_full_h200_posefree_direct.sh',
    'configs/h200/r9_waymo_full_posefree_100k.json',
    'tools/r9/prepare_posefree_multigpu.py',
    'tools/r9/prepare_posefree_waymo_full.py',
    'tools/r9/posefree_batch_stage.py',
    'tools/r9/posefree_gca_runtime.py',
    'tools/r9/posefree_omega_runtime.py',
    'tools/r9/posefree_omega_io.py',
    'tools/r9/export_posefree_manifest_batch.py',
    'tools/r9/check_posefree_full_camera_cache.py',
    'tools/r9/test_posefree_multigpu.py',
    'tools/r9/test_posefree_gca_runtime.py',
    'tools/r9/test_posefree_omega_runtime.py',
    'ufo/dataset/pose_override.py',
    'ufo/dataset/constants.py',
    'ufo/paper_contract.py',
    'LICENSE',
)
UFO_FILES = (
    'tools/export_rgb_only_manifest.py',
    'tools/export_rgb_only_omega.py',
    'tools/diagnose_gca_omega_scale.py',
    'tools/diagnose_gca_omega_scale_cached.py',
    'tools/export_rgb_metric_pose.py',
    'tools/build_global_pose_from_overlaps.py',
    'ufo/dataset/constants.py',
    'ufo/paper_contract.py',
    'LICENSE',
)

SNAPSHOT_README = '''# Pose-free camera preprocessing: runtime snapshot

This is a source snapshot of the installed UFO/Waymo preprocessing pipeline,
not yet a standalone pip package or a new camera estimation model. Captured
working files include uncommitted changes. See runtime_inventory.json for
SHA-256 hashes, dependency versions, source roots, revisions and missing files.

## Current pipeline

RGB manifest -> VGGT-Omega pose/K/depth/confidence -> MoGe-2 + GCA scale
-> metric window cameras -> overlap SE(3) alignment -> full-scene camera NPZ.

The four-GPU scheduler shares a scene queue. Each GPU processes one independent
scene at a time. Models are reused over a window group; this is not DDP training
or a 179-window neural-network forward batch. The installed backend remains
the authority for inference numerics and cleanup.

## What was copied

- h200_adapter/: scheduling, accelerated stages, configuration and UFO reader.
- source_ufo/: RGB manifest and reference camera/scale/alignment exporters.
- omega_adapter/: the existing intrinsic-coordinate conversion helper only.
- dependency_notices/: local model-repository notices; model implementations
  and weights are external dependencies.
- camera_npz_schema.json: key names, shapes/dtypes, and selected convention
  labels from one validated final camera file, when available; no camera arrays.
- runtime_status.json: a time-stamped inspection, not a live dashboard.

Datasets, RGB images, SAM tracks, weights, camera NPZ contents, raw depth arrays,
virtual environments, credentials and original .git directories are not copied.
Only explicitly listed source files are read; this is not a recursive repo copy.
Inspect hardcoded server paths before making this snapshot public. No remote
repository is created and nothing is pushed by the collection tool.

## Current inputs

- UFO-format Waymo RGB images and a scene annotation list.
- Scene JSON metadata: dataset, scene_id, scene_name, fps, num_timesteps,
  relative_image_path indexed by camera ID and frame.
- Camera selection and target image size from config. The current protocol
  uses cameras 1/0/2 and target K at height 160, width 240.
- Omega and MoGe-2 source checkouts, local checkpoints and an installed GPU
  Python environment. No GT camera calibration, poses, depth or LiDAR are
  inputs to the RGB-only camera prediction chain; SAM is used later by UFO.

The legacy annotation JSON may contain GT fields; the RGB-only manifest exporter
selects metadata and RGB paths. It does not use those GT fields for prediction.
The current camera protocol is all_rgb: both context and target RGB enter the
offline camera estimator. It must not be described as context-only or online.

## Current outputs

cache/global_aligned/<scene_name>/omega_pose_override.npz contains frame_ids,
camera_ids, omega_c2w_global_metric (OpenCV camera basis),
omega_camera_to_world_global_metric (dataset camera basis),
predicted_intrinsics_ufo and convention labels. Resolve records by the pair
(frame_id, camera_id), not by assumed input ordering. Each scene has its own
reference frame; global_metric does not mean a GPS/world-map coordinate frame.
Metric scale is estimated by MoGe-2/GCA, not measured or guaranteed exact.

The intrinsics are in pixels for the target image size recorded in the cache
contract. Final NPZs do not contain RGB, masks, depth maps, a mesh or Gaussians.
The file ../.posefree_contract.json records the image size and camera protocol.

Intermediate raw and scale files are removed after successful metric export;
window poses and manifests after successful alignment. Interrupted work may
leave residual files. Final scene cameras and logs remain available.

## Proposed reusable plugin boundary (not implemented by this snapshot)

1. Separate a generic RGB manifest API from Waymo/UFO dataset adapters.
2. Keep Omega, MoGe and scale/alignment backends external and versioned.
3. Make GPU count, reference camera, window length/stride, image geometry,
   output path and cleanup policy configurable.
4. Expose prepare, status, validate and export commands plus a Python API.
5. Export an OpenCV camera-to-world array and per-image intrinsics with explicit
   coordinate conventions, image transforms, frame/camera keys and provenance.
6. Keep a UFO output adapter for the existing omega_pose_override.npz schema.
7. Publish CPU schema tests and GPU equivalence/throughput results separately.
8. Choose the wrapper's license deliberately and retain upstream notices.
   Do not relicense or bundle third-party model weights as wrapper-owned code.

Suggested public description: "Offline RGB-only camera preprocessing with
VGGT-Omega, MoGe-2/GCA scale estimation and overlap alignment, with a UFO/Waymo
adapter." Do not present the dependencies as newly proposed camera models.

Before release, compare this snapshot with the source used for the completed
798-scene run. Audit missing imported modules and replace user-specific paths
with configuration. The manifest API and general data adapters are still work
to implement; this snapshot itself is not a drop-in universal reconstruction
plugin.
'''


def validate_camera(path, scene):
    with np.load(path, allow_pickle=False) as x:
        if str(x['scene_name'].item()) != scene['scene_name']:
            raise ValueError('scene_name mismatch')
        if str(x['coordinate_frame'].item()) != 'global_metric':
            raise ValueError('not global_metric')
        frames, cameras = x['frame_ids'], x['camera_ids'].astype(str)
        count = int(scene['num_timesteps'])
        expected = {(f, c) for f in range(count) for c in CAMERAS}
        if frames.ndim != 1 or frames.dtype.kind not in 'iu' or cameras.shape != frames.shape:
            raise ValueError('invalid frame/camera arrays')
        if len(frames) != len(expected) or set(zip(frames.tolist(), cameras.tolist())) != expected:
            raise ValueError('missing/duplicate frame-camera records')
        poses, intrinsics = x['omega_camera_to_world_global_metric'], x['predicted_intrinsics_ufo']
        if poses.shape != (len(frames), 4, 4) or intrinsics.shape != (len(frames), 3, 3):
            raise ValueError('invalid pose/K dimensions')
        if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
            raise ValueError('non-finite pose/K')
        if (intrinsics[:, (0, 1), (0, 1)] <= 0).any():
            raise ValueError('non-positive focal length')


def tail(path, limit=4096):
    try:
        with path.open('rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - limit))
            return f.read().decode(errors='replace').splitlines()[-4:]
    except OSError:
        return []


def processes(repo, cache):
    names = {'prepare_posefree_multigpu.py', 'prepare_posefree_waymo_full.py', 'posefree_batch_stage.py'}
    result = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            script = next((Path(s).name for s in argv if Path(s).name in names), None)
            if script is None or not any(str(repo) in s or str(cache) in s for s in argv):
                continue
            def arg(flag):
                return argv[argv.index(flag) + 1] if flag in argv else None
            entry = {'pid': int(proc.name), 'script': script, 'stage': arg('--stage')}
            manifest = arg('--manifest-list')
            if manifest:
                entry['batch'] = Path(manifest).stem
            if arg('--scene-first'):
                entry['scene_first'] = arg('--scene-first')
            result.append(entry)
        except (OSError, ValueError, IndexError):
            continue
    return sorted(result, key=lambda row: row['pid'])


def inspect(data, cache, repo):
    annotation = data / 'scene_list/waymo_train.txt'
    lines = [s.strip() for s in annotation.read_text().splitlines() if s.strip()]
    valid, invalid, missing, scenes = [], [], [], {}
    sample = None
    for index, line in enumerate(lines):
        path = Path(line)
        scene = json.loads((path if path.is_absolute() else data / path).read_text())
        name = scene['scene_name']
        if not name or Path(name).name != name or name in ('.', '..') or name in scenes:
            raise ValueError(f'unsafe/duplicate scene name: {name!r}')
        scenes[name] = scene
        output = cache / 'global_aligned' / name / 'omega_pose_override.npz'
        if not output.is_file():
            missing.append(index)
            continue
        try:
            validate_camera(output, scene)
            valid.append(index)
            if sample is None:
                sample = output
        except Exception as exc:
            invalid.append({'index': index, 'scene': name, 'error': str(exc)})
    private_ready, private_invalid = [], []
    for path in sorted((cache / 'parallel_workers').glob('worker_*/global_aligned/*/omega_pose_override.npz')):
        name = path.parent.name
        if name not in scenes or (cache / 'global_aligned' / name / path.name).exists():
            continue
        try:
            validate_camera(path, scenes[name])
            private_ready.append(str(path))
        except Exception as exc:
            private_invalid.append({'path': str(path), 'error': str(exc)})
    logs = []
    for worker in sorted((cache / 'parallel_workers').glob('worker_*')):
        candidates = list((worker / 'logs').glob('*.log'))
        if candidates:
            try:
                latest = max(candidates, key=lambda p: p.stat().st_mtime_ns)
                logs.append({'worker': worker.name, 'latest_stage_log': str(latest),
                             'modified_utc': datetime.fromtimestamp(latest.stat().st_mtime, timezone.utc).isoformat(),
                             'tail': tail(latest)})
            except OSError:
                pass
    raw_count, raw_bytes = 0, 0
    for path in (cache / 'omega_raw_all').glob('start_*/*.npz'):
        try:
            raw_bytes += path.stat().st_size
            raw_count += 1
        except OSError:
            pass
    report = {
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'total_scenes': len(lines), 'validated_final_scenes': len(valid),
        'missing_final_scenes': len(missing), 'invalid_final_scenes': len(invalid),
        'percent_validated': round(100 * len(valid) / max(len(lines), 1), 2),
        'missing_indices_first_20': missing[:20], 'invalid_examples': invalid[:20],
        'private_ready_not_published': private_ready, 'private_invalid': private_invalid,
        'visible_processes_this_container': processes(repo, cache), 'worker_logs': logs,
        'raw_files': raw_count, 'raw_logical_gib': round(raw_bytes / 1024**3, 3),
        'filesystem_free_gib': round(shutil.disk_usage(cache).free / 1024**3, 1),
        'project_quota': 'unknown',
        'note': 'Read-only non-atomic observation while processing may continue; no pose-accuracy assessment.',
    }
    return report, sample


def git_info(path):
    def query(*args):
        try:
            r = subprocess.run(['git', '-C', str(path), *args], text=True,
                               capture_output=True, timeout=15)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    dirty = query('diff', 'HEAD', '--name-only', '--', '.')
    return {'root': str(path), 'commit': query('rev-parse', 'HEAD'),
            'tracked_dirty': None if dirty is None else bool(dirty)}


def capture(repo, source, cache, output, report, sample, init_git=False):
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'destination must be new: {output}')
    for root in (repo, source, cache):
        if output.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'snapshot destination must be outside {root}')
    if init_git and not shutil.which('git'):
        raise RuntimeError('git is required for --init-git')
    groups = [
        ('h200_adapter', repo, H200_FILES),
        ('source_ufo', source / 'pose-freeufo', UFO_FILES),
        ('omega_adapter', source / 'vggt-omega', ('tools/export_ufo_pose_override.py',)),
        ('dependency_notices/omega', source / 'vggt-omega', ('LICENSE', 'README.md', 'pyproject.toml')),
        ('dependency_notices/moge', source / 'moge', ('LICENSE', 'README.md', 'pyproject.toml')),
    ]
    required = [repo / 'tools/r9/prepare_posefree_waymo_full.py',
                repo / 'tools/r9/posefree_batch_stage.py',
                source / 'pose-freeufo/tools/export_rgb_only_manifest.py',
                source / 'pose-freeufo/tools/build_global_pose_from_overlaps.py']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True)
    inventory, missing, captured = [], [], []
    for prefix, root, relatives in groups:
        for relative in relatives:
            src = root / relative
            if not src.is_file():
                missing.append(str(src))
                continue
            if src.stat().st_size > 8 * 1024**2:
                raise ValueError(f'unexpectedly large source file, inspect it manually: {src}')
            contents = src.read_bytes()
            digest = hashlib.sha256(contents).hexdigest()
            dst = output / prefix / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(contents)
            captured.append((src, digest))
            inventory.append({'source': str(src), 'snapshot': str(dst.relative_to(output)),
                              'bytes': len(contents), 'sha256': digest})
    collector = Path(__file__).resolve()
    collector_data = collector.read_bytes()
    collector_target = output / 'tools/inspect_posefree_runtime.py'
    collector_target.parent.mkdir(parents=True, exist_ok=True)
    collector_target.write_bytes(collector_data)
    inventory.append({'source': str(collector), 'snapshot': 'tools/inspect_posefree_runtime.py',
                      'bytes': len(collector_data), 'sha256': hashlib.sha256(collector_data).hexdigest()})
    for path, digest in captured:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'source changed during capture: {path}; snapshot was not committed')
    versions = {}
    for dist in importlib.metadata.distributions():
        if dist.metadata.get('Name'):
            versions[dist.metadata['Name']] = dist.version
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'python': sys.version, 'python_executable': sys.executable,
                'repositories': [git_info(repo), git_info(source / 'pose-freeufo'),
                                 git_info(source / 'vggt-omega'), git_info(source / 'moge')],
                'files': inventory, 'optional_files_not_found': missing,
                'installed_distribution_versions': dict(sorted(versions.items())),
                'status': 'runtime source snapshot; not a standalone release'}
    metadata['checkpoint_files'] = []
    for checkpoint in (source / 'vggt-omega/checkpoints/vggt_omega_1b_512.pt',
                       source / 'checkpoints/moge-2-vitl/model.pt'):
        row = {'path': str(checkpoint), 'exists': checkpoint.is_file(), 'sha256': None}
        if row['exists']:
            stat = checkpoint.stat()
            row.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        metadata['checkpoint_files'].append(row)
    metadata['checkpoint_note'] = 'Weights are neither copied nor hashed by this source collector.'
    (output / 'runtime_inventory.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'runtime_status.json').write_text(json.dumps(report, indent=2) + '\n')
    contract = cache / '.posefree_contract.json'
    if contract.is_file():
        (output / 'camera_contract.json').write_text(json.dumps(json.loads(contract.read_text()), indent=2) + '\n')
    if sample:
        with np.load(sample, allow_pickle=False) as x:
            schema = {'fields': {key: {'shape': list(x[key].shape), 'dtype': str(x[key].dtype)} for key in x.files},
                      'conventions': {key: str(x[key].item()) for key in
                                      ('coordinate_frame', 'metric_scale_source', 'world_gauge') if key in x}}
        (output / 'camera_npz_schema.json').write_text(json.dumps(schema, indent=2) + '\n')
    (output / 'README.md').write_text(SNAPSHOT_README)
    (output / '.gitignore').write_text('.venv*/\n__pycache__/\n*.pyc\n.env*\n*.pt\n*.pth\n*.npz\n*.npy\ndata/\noutputs/\ncheckpoints/\n')
    if init_git:
        subprocess.run(['git', 'init', str(output)], check=True)
        subprocess.run(['git', '-C', str(output), 'symbolic-ref', 'HEAD', 'refs/heads/main'], check=True)
        subprocess.run(['git', '-C', str(output), 'add', '--all'], check=True)
        subprocess.run(['git', '-C', str(output), '-c', 'user.name=PoseFree Snapshot Tool',
                        '-c', 'user.email=snapshot@localhost', 'commit', '-m',
                        'Capture installed pose-free preprocessing sources and interface notes'], check=True)
    return len(inventory), missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    p.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--cache-root', type=Path)
    p.add_argument('--json', action='store_true', help='Print the full status report as JSON')
    p.add_argument('--snapshot-out', type=Path)
    p.add_argument('--init-git', action='store_true', help='Initialize and commit a new local snapshot repository; no push')
    a = p.parse_args()
    if a.init_git and not a.snapshot_out:
        p.error('--init-git requires --snapshot-out')
    if a.json and a.snapshot_out:
        p.error('--json is for status-only use')
    repo, source = a.repo.resolve(), a.source_root.resolve()
    data = (a.data_root or repo / 'data/UFO_paper').resolve()
    cache = (a.cache_root or repo.parent / 'outputs/r9ufo/posefree_camera_full').resolve()
    report, sample = inspect(data, cache, repo)
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"OBSERVED_UTC={report['observed_utc']}")
        print(f"VALIDATED_FINAL={report['validated_final_scenes']}/{report['total_scenes']} "
              f"({report['percent_validated']}%)")
        print(f"MISSING={report['missing_final_scenes']} INVALID={report['invalid_final_scenes']} "
              f"PRIVATE_READY={len(report['private_ready_not_published'])}")
        print(f"RAW={report['raw_files']} files / {report['raw_logical_gib']:.3f} GiB (logical size)")
        print(f"FILESYSTEM_FREE={report['filesystem_free_gib']:.1f} GiB; PROJECT_QUOTA=UNKNOWN")
        print('PROCESSES_VISIBLE_IN_THIS_CONTAINER:')
        for item in report['visible_processes_this_container']:
            print(json.dumps(item, ensure_ascii=False))
        if not report['visible_processes_this_container']:
            print('  none (processes in other containers are not visible here)')
        for item in report['worker_logs']:
            print(f"\n{item['worker']} latest={item['modified_utc']} {item['latest_stage_log']}")
            print('\n'.join(item['tail']))
        if report['invalid_examples']:
            print('INVALID_EXAMPLES=' + json.dumps(report['invalid_examples'], ensure_ascii=False))
        print('Inspection is read-only. Completed count is structural validation, not pose accuracy.')
    if a.snapshot_out:
        count, missing = capture(repo, source, cache, a.snapshot_out, report, sample, a.init_git)
        print(f'POSEFREE_SOURCE_SNAPSHOT={a.snapshot_out.absolute()} FILES={count}')
        if missing:
            print('OPTIONAL_FILES_NOT_FOUND=' + json.dumps(missing, ensure_ascii=False))
        print('No remote Git repository was created or pushed.')


if __name__ == '__main__':
    main()
