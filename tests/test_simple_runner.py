"""Exercise the simple runner with real scheduling/export and synthetic models."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

import run_posefree as runner
from adagent_posefree.config import load_config
from test_standalone import FakeStages


class SimpleRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "RGB images"
        for scene in ("drive_a", "drive_b"):
            for i in range(3):
                path = self.data / scene / "front" / f"{i:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (24, 16), (i, 30, 40)).save(path)
        self.out = self.root / "results"
        self.config = load_config()
        for key in ("omega_checkpoint", "moge_checkpoint"):
            path = self.root / key
            path.write_bytes(b"synthetic test asset")
            self.config[key] = str(path)
        self.config["min_free_gib_per_gpu"] = .001
        self.argv = ["--data", str(self.data), "--output", str(self.out)]

    def run_job(self, extra=(), fail=None):
        text = io.StringIO()
        stages = FakeStages(fail=fail)
        with contextlib.redirect_stdout(text), contextlib.redirect_stderr(text), \
             mock.patch("adagent_posefree.config.load_config", return_value=self.config), \
             mock.patch("adagent_posefree.runtime.doctor", side_effect=lambda cfg, n: [str(i) for i in range(n)]), \
             mock.patch("adagent_posefree.runtime.Processes", return_value=stages):
            code = runner.main([*self.argv, *extra])
        return code, text.getvalue(), stages

    def test_first_run_and_repeat_with_more_gpus(self):
        code, text, stages = self.run_job()
        self.assertEqual(code, 0, text)
        self.assertIn("POSEFREE_DONE 2/2", text)
        self.assertTrue(stages.calls)
        final = self.out / "scenes/drive_a/cameras.npz"
        before = final.read_bytes(), final.stat().st_mtime_ns
        self.assertFalse((self.out / ".work/drive_a").exists())
        code, text, stages = self.run_job(["--gpus", "2"])
        self.assertEqual(code, 0, text)
        self.assertEqual(stages.calls, [])
        self.assertEqual((final.read_bytes(), final.stat().st_mtime_ns), before)

    def test_interrupted_stage_resumes_and_retains_existing_input(self):
        code, text, _ = self.run_job(fail="gca")
        self.assertEqual(code, 1)
        self.assertNotIn("POSEFREE_DONE", text)
        self.assertTrue(list((self.out / ".work").rglob("raw/**/*.npz")))
        saved = (self.out / "input_manifest.json").read_bytes()
        code, text, _ = self.run_job()
        self.assertEqual(code, 0, text)
        self.assertEqual((self.out / "input_manifest.json").read_bytes(), saved)

    def test_changed_fps_and_added_scene_are_rejected_without_overwrite(self):
        self.assertEqual(self.run_job()[0], 0)
        saved = (self.out / "input_manifest.json").read_bytes()
        code, text, stages = self.run_job(["--fps", "5"])
        self.assertEqual(code, 1)
        self.assertEqual(stages.calls, [])
        self.assertIn("与已有结果不一致", text)
        for i in range(2):
            path = self.data / "new_drive/front" / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 16)).save(path)
        code, text, _ = self.run_job()
        self.assertEqual(code, 1)
        self.assertEqual((self.out / "input_manifest.json").read_bytes(), saved)

    def test_bad_input_stops_before_gpu_or_output_creation(self):
        path = self.data / "drive_a/right/000000.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (24, 16)).save(path)
        with mock.patch("adagent_posefree.runtime.doctor") as doctor:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(self.argv), 1)
            doctor.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_output_inside_images_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code = runner.main(["--data", str(self.data), "--output", str(self.data / "outputs")])
        self.assertEqual(code, 1)
        self.assertFalse((self.data / "outputs").exists())

    def test_check_only_does_not_claim_inference(self):
        code, text, stages = self.run_job(["--check-only"])
        self.assertEqual(code, 0, text)
        self.assertNotIn("POSEFREE_DONE", text)
        self.assertEqual(stages.calls, [])
        self.assertFalse((self.out / "input_manifest.json").exists())

    def test_single_scene_does_not_read_siblings_and_has_stable_paths(self):
        self.argv = ["--data", str(self.data / "drive_a"), "--layout", "scene", "--output", str(self.out)]
        code, text, _ = self.run_job()
        self.assertEqual(code, 0, text)
        self.assertIn("POSEFREE_DONE 1/1", text)
        saved = json.loads((self.out / "input_manifest.json").read_text())
        self.assertEqual([s["scene_name"] for s in saved["scenes"]], ["drive_a"])
        image = saved["scenes"][0]["frames"][0]["images"]["front"]
        self.assertEqual(image, str(self.data / "drive_a/front/000000.png"))
        self.assertTrue(Path(image).is_file())
        code, text, stages = self.run_job()
        self.assertEqual(code, 0, text)
        self.assertEqual(stages.calls, [])

    def test_batch_729_scenes_is_not_truncated(self):
        data = self.root / "729_scenes"
        for i in range(729):
            directory = data / f"scene_{i:04d}" / "front"
            directory.mkdir(parents=True)
            for fid in (0, 1):
                # Indexing checks paths; decoding/inference is tested elsewhere.
                (directory / f"{fid:06d}.jpg").write_bytes(b"index fixture")
        scenes = runner.input_scenes(data, "images", 10, None, "scenes")
        self.assertEqual(len(scenes), 729)
        self.assertEqual(scenes[0]["scene_name"], "scene_0000")
        self.assertEqual(scenes[-1]["scene_name"], "scene_0728")

    def test_waymo_adapter_selects_zero_based_scene_without_gt_cameras(self):
        data = self.root / "waymo"
        (data / "scene_list").mkdir(parents=True)
        entries = []
        for n in range(2):
            relative = {c: [f"s{n}/images/{c}/{i}.png" for i in range(2)] for c in ("1", "0", "2")}
            for values in relative.values():
                for value in values:
                    path = data / "datasets/waymo" / value.replace("images", "images_4")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (24, 16)).save(path)
            item = dict(scene_name=f"s{n}", dataset="waymo", fps=10, num_timesteps=2,
                        relative_image_path=relative, camera_to_world="DO_NOT_USE", intrinsics="DO_NOT_USE")
            (data / f"s{n}.json").write_text(json.dumps(item))
            entries.append(f"s{n}.json")
        (data / "scene_list/waymo_train.txt").write_text("\n".join(entries))
        scenes = runner.input_scenes(data, "waymo", 10, 1)
        self.assertEqual([s["scene_name"] for s in scenes], ["s1"])
        self.assertEqual(scenes[0]["reference_camera"], "0")
        self.assertNotIn("DO_NOT_USE", json.dumps(scenes))


if __name__ == "__main__":
    unittest.main()
