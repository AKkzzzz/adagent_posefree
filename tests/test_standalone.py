import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from adagent_posefree.config import ROOT, load_config, gpu_selectors
from adagent_posefree.data import index_folders, index_waymo, load_manifest, window_starts, window_manifest
from adagent_posefree.backend.alignment import align_windows
from adagent_posefree.backend.stages import metric_output
from adagent_posefree.backend.posefree_omega_io import write_npz
from adagent_posefree.output import complete, export_scene
from adagent_posefree.runtime import process_scene, prepare


def make_scene(root, frames=23, cameras=("1", "0", "2"), name="scene"):
    root = Path(root)
    images = {}
    for c in cameras:
        p = root / name / c / "000000.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 16), (120, 30, 40)).save(p)
        images[c] = str(p)
    return dict(scene_name=name, camera_ids=list(cameras), reference_camera=cameras[0], fps=10.,
                opencv_to_dataset=np.eye(4).tolist(),
                frames=[dict(frame_id=i, timestamp=i/10., images=images) for i in range(frames)])


def make_raw(manifest):
    entries = manifest["images"]
    poses = np.repeat(np.eye(4)[None], len(entries), axis=0)
    cameras = manifest["camera_ids"]
    for i, e in enumerate(entries):
        poses[i, :3, 3] = [e["frame_id"] * .1, cameras.index(e["camera_id"]) * .2, 0]
    k = np.repeat(np.asarray([[120, 0, 120], [0, 120, 80], [0, 0, 1]])[None], len(entries), axis=0)
    return dict(scene_name=np.asarray(manifest["scene_name"]), frame_ids=np.asarray([e["frame_id"] for e in entries], dtype=np.int32),
                camera_ids=np.asarray([e["camera_id"] for e in entries]), roles=np.asarray([e["role"] for e in entries]),
                omega_c2w_raw=poses, omega_w2c_raw=np.linalg.inv(poses), predicted_intrinsics_ufo=k,
                omega_depth_raw=np.ones((len(entries), 4, 4)), omega_depth_conf_raw=np.ones((len(entries), 4, 4)))


class FakeStages:
    """Only model outputs are synthetic; execute real metric/alignment/export IO."""
    def __init__(self, fail=None):
        self.stop = threading.Event()
        self.fail = fail
        self.calls = []

    def cancel(self):
        self.stop.set()

    def run(self, command, env, log):
        stage = command[command.index("--stage")+1]
        self.calls.append((stage, env.get("CUDA_VISIBLE_DEVICES")))
        if stage == self.fail:
            raise RuntimeError("simulated model failure")
        job = json.loads(Path(command[command.index("--job")+1]).read_text())
        for value in Path(job["list"]).read_text().splitlines():
            p = Path(value)
            m = json.loads(p.read_text())
            raw = Path(job["raw"])/p.parent.name/(p.stem+".npz")
            scale = Path(job["scale"])/p.parent.name/(p.stem+".json")
            out = Path(job["windows"])/p.parent.name/p.stem/"omega_pose_override.npz"
            if stage == "omega":
                write_npz(raw, make_raw(m))
            elif stage == "gca":
                scale.parent.mkdir(parents=True, exist_ok=True)
                scale.write_text(json.dumps(dict(global_scale=2.)))
            else:
                metric_output(m, raw, scale, out)


class StandaloneTests(unittest.TestCase):
    def test_gpu_count_respects_scheduler_visibility(self):
        self.assertEqual(gpu_selectors(2, "3,5,7", 3), ["3", "5"])
        self.assertEqual(gpu_selectors(1, "GPU-test", 1), ["GPU-test"])
        self.assertEqual(gpu_selectors(4, None, 8), ["0", "1", "2", "3"])
        for count, visible, available in [(2, "3", 1), (0, None, 8), (1, "", 0), (1, "2,2", 2)]:
            with self.assertRaises(ValueError):
                gpu_selectors(count, visible, available)

    def test_generic_index_synchronization_and_arbitrary_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for cam in ("front", "right"):
                for fid in (0, 5, 10):
                    p = root/"images/drive"/cam/f"{fid:06d}.jpg"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8)).save(p)
            m = root/"input.json"
            m.write_text(json.dumps(index_folders(root/"images", 10, "front")))
            scene = load_manifest(m)[0]
            self.assertEqual([f["frame_id"] for f in scene["frames"]], [0, 5, 10])
            self.assertEqual(scene["reference_camera"], "front")
            (root/"images/drive/right/000010.jpg").unlink()
            with self.assertRaisesRegex(ValueError, "synchronized"):
                index_folders(root/"images", 10)

    def test_waymo_rgb_only_accepts_one_scene(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            s = dict(scene_name="one", dataset="waymo", fps=10, num_timesteps=21,
                     camera_to_world="DO_NOT_READ", intrinsics="DO_NOT_READ",
                     relative_image_path={c:[f"one/images/{c}/{i}.jpg" for i in range(21)] for c in ("1","0","2")})
            (root/"one.json").write_text(json.dumps(s))
            (root/"list.txt").write_text("one.json\n")
            manifest = index_waymo(root, root/"list.txt")
            self.assertEqual(len(manifest["scenes"]), 1)
            self.assertNotIn("DO_NOT_READ", json.dumps(manifest))
            self.assertIn("images_4", manifest["scenes"][0]["frames"][0]["images"]["0"])

    def test_window_tail_and_small_scene(self):
        self.assertEqual(window_starts(23, 20, 2), [0, 2, 3])
        self.assertEqual(window_starts(3, 20, 1), [0])
        with self.assertRaises(ValueError):
            window_starts(50, 20, 20)

    def test_metric_conversion_matches_original_front_camera(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene = make_scene(root, frames=20)
            scene["reference_camera"] = "0"
            manifest = window_manifest(scene, 0, load_config())
            raw, scale = root/"raw.npz", root/"scale.json"
            write_npz(raw, make_raw(manifest))
            scale.write_text('{"global_scale": 2.0}')
            original = (ROOT/"h200_adapter/tools/r9/posefree_batch_stage.py").read_text()
            node = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name=="metric_output")
            namespace = dict(np=np, json=json, Path=Path)
            exec(compile(ast.Module(body=[node], type_ignores=[]), "original", "exec"), namespace)
            namespace["metric_output"](manifest, raw, scale, root/"original.npz")
            metric_output(manifest, raw, scale, root/"new.npz")
            with np.load(root/"original.npz") as a, np.load(root/"new.npz") as b:
                for key in a.files:
                    if key != "world_gauge":
                        np.testing.assert_array_equal(a[key], b[key])

    def test_pipeline_outputs_match_old_alignment_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene = make_scene(root)
            config = load_config()
            config["keep_intermediates"] = True
            stages = FakeStages()
            process_scene(scene, root/"out", config, "test", {}, stages)
            self.assertTrue(complete(root/"out", scene, "test"))
            old = root/"old"
            subprocess.run([sys.executable, str(ROOT/"source_ufo/tools/build_global_pose_from_overlaps.py"),
                            "--input-root", str(root/"out/.work/scene/windows"), "--scene", "scene",
                            "--first", "0", "--last", "3", "--output-root", str(old)], check=True, stdout=subprocess.DEVNULL)
            with np.load(old/"scene/omega_pose_override.npz") as a, np.load(root/"out/scenes/scene/cameras.npz") as b:
                for oldkey,newkey in [("omega_c2w_global_metric","c2w"),("omega_camera_to_world_global_metric","camera_to_world_dataset"),("predicted_intrinsics_ufo","K")]:
                    np.testing.assert_allclose(a[oldkey], b[newkey], atol=1e-7)
            before = len(stages.calls)
            process_scene(scene, root/"out", config, "test", {}, stages)
            self.assertEqual(len(stages.calls), before)

    def test_short_sequence_arbitrary_reference_and_corrupt_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene = make_scene(root, frames=3, cameras=("front",))
            process_scene(scene, root/"out", load_config(), "test", {}, FakeStages())
            self.assertTrue(complete(root/"out", scene, "test"))
            self.assertFalse((root/"out/.work/scene").exists())
            camera = root/"out/scenes/scene/cameras.npz"
            camera.write_bytes(b"broken zip")
            self.assertFalse(complete(root/"out", scene, "test"))
            process_scene(scene, root/"out", load_config(), "test", {}, FakeStages())
            self.assertTrue(complete(root/"out", scene, "test"))

    def test_failure_retains_raw_for_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene = make_scene(root, frames=3)
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                process_scene(scene, root/"out", load_config(), "test", {}, FakeStages(fail="gca"))
            self.assertTrue(list((root/"out/.work/scene/raw").rglob("*.npz")))
            self.assertFalse(complete(root/"out", scene, "test"))

    def test_scheduler_resume_changes_card_count_and_rejects_config_mix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = [make_scene(root, frames=3, name=f"s{i}") for i in range(4)]
            config = load_config()
            for key in ("omega_checkpoint", "moge_checkpoint"):
                asset = root/key
                asset.write_bytes(b"test checkpoint")
                config[key] = str(asset)
            config["min_free_gib_per_gpu"] = .001
            with mock.patch("adagent_posefree.runtime.doctor", side_effect=lambda cfg,n: [str(i) for i in range(n)]), mock.patch("adagent_posefree.runtime.Processes", FakeStages):
                prepare(scenes, config, root/"out", 2)
                self.assertTrue(all(complete(root/"out", s) for s in scenes))
                prepare(scenes, config, root/"out", 1)
                config["output_image_size"] = [320,480]
                with self.assertRaisesRegex(RuntimeError, "different"):
                    prepare(scenes, config, root/"out", 1)


if __name__ == "__main__":
    unittest.main()
