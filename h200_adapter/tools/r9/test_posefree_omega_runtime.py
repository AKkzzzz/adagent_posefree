"""CPU regression tests for lossless IO, manifest reuse and K geometry."""
import argparse
import ast
import contextlib
import dataclasses
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from posefree_omega_io import write_npz, RawWriter, IntrinsicsGeometryCache, require_disk_budget
from export_posefree_manifest_batch import export_jobs

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--omega-repo", type=Path, required=True)
parser.add_argument("--source-ufo-root", type=Path, required=True)
LOCATIONS, REST = parser.parse_known_args()


def load_functions(path, names, namespace):
    # Read the supplied reference functions without importing GPU models.
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise AssertionError(f"reference functions missing: {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class StorageTests(unittest.TestCase):
    def test_stored_and_compressed_arrays_are_identical(self):
        payload = {"scene_name": np.asarray("scene"), "frame_ids": np.arange(4, dtype=np.int32),
                   "camera_ids": np.asarray(["0", "1", "2", "0"]),
                   "depth": np.arange(480, dtype=np.float32).reshape(4, 10, 12)[:, :, ::2]}
        payload["depth"][0, 0, 0] = np.nan
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root)/"a.npz", Path(root)/"b.npz"
            write_npz(a, payload, False)
            write_npz(b, payload, True)
            with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
                self.assertEqual(set(x.files), set(y.files))
                for key in payload:
                    np.testing.assert_array_equal(x[key], payload[key])
                    np.testing.assert_array_equal(x[key], y[key])

    def test_partial_write_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root)/"out.npz"
            write_npz(target, {"a": np.arange(3)})
            before = target.read_bytes()
            def fail(f, **kwargs):
                f.write(b"incomplete")
                raise OSError("simulated disk failure")
            with mock.patch("posefree_omega_io.np.savez", side_effect=fail):
                with self.assertRaises(OSError):
                    write_npz(target, {"a": np.arange(7)})
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(len(list(Path(root).iterdir())), 1)

    def test_background_failure_is_propagated(self):
        writer = RawWriter()
        try:
            with mock.patch("posefree_omega_io.write_npz", side_effect=OSError("disk")):
                writer.submit(Path("unused"), {}, {})
                with self.assertRaises(OSError):
                    writer.collect()
        finally:
            writer.close()

    def test_writer_rejects_unbounded_queue(self):
        with tempfile.TemporaryDirectory() as root:
            writer = RawWriter()
            try:
                writer.submit(Path(root)/"a.npz", {"a": np.arange(3)}, {"start": "start_000"})
                with self.assertRaises(RuntimeError):
                    writer.submit(Path(root)/"b.npz", {}, {})
                result = writer.collect()
                self.assertEqual(result["start"], "start_000")
                self.assertGreater(result["raw_bytes"], 0)
                self.assertIsNone(writer.collect())
            finally:
                writer.close()

    def test_disk_preflight_rejects_insufficient_space(self):
        with mock.patch("posefree_omega_io.shutil.disk_usage", return_value=types.SimpleNamespace(free=10)):
            with self.assertRaises(RuntimeError):
                require_disk_budget(Path("unused"), 100, 179)


class ReferenceTests(unittest.TestCase):
    def test_cached_intrinsics_match_original_for_crop_and_padding(self):
        ns = {"np": np, "Image": Image}
        load_functions(LOCATIONS.omega_repo/"vggt_omega/utils/load_fn.py", ["_balanced_target_shape"], ns)
        load_functions(LOCATIONS.omega_repo/"tools/export_ufo_pose_override.py", ["transform_intrinsics_to_ufo"], ns)
        cache = IntrinsicsGeometryCache(Image, ns["_balanced_target_shape"])
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for i, shape in enumerate(((240, 160), (640, 80), (80, 640), (640, 360))):
                path = Path(root)/f"{i}.png"
                Image.new("RGB", shape).save(path)
                paths.append(str(path))
            for order in ([0, 1, 2, 3], [3, 0], [2, 1], [0]):
                selected = [paths[i] for i in order]
                k = np.repeat(np.array([[[501.5, 0, 238.25], [0, 487.0, 196.75], [0, 0, 1]]], dtype=np.float32), len(order), axis=0)
                expected, _ = ns["transform_intrinsics_to_ufo"](k, selected, [160, 240], image_resolution=512)
                np.testing.assert_array_equal(cache.transform(k, selected, [160, 240]), expected)
                with mock.patch.object(cache.Image, "open", side_effect=AssertionError("unexpected image reopen")):
                    np.testing.assert_array_equal(cache.transform(k, selected, [160, 240]), expected)

    def test_manifest_batch_matches_original_exporter(self):
        root = LOCATIONS.source_ufo_root
        ns = {"argparse": argparse, "json": json, "sys": sys, "Path": Path, "np": np,
              "dataclass": dataclasses.dataclass, "CONTEXT_STRIDE": 5}
        exec(compile((root/"ufo/dataset/constants.py").read_text(), "constants.py", "exec"), ns)
        load_functions(root/"ufo/paper_contract.py", ["FrameProtocol", "split_context_supervision"], ns)
        source = root/"tools/export_rgb_only_manifest.py"
        load_functions(source, ["parse_args", "image_path", "main"], ns)
        exporter = types.SimpleNamespace(__file__=str(source), main=ns["main"])
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            cfg = temp/"config.json"
            cfg.write_text(json.dumps(dict(timespan=0.5, num_target_chunks=4, num_max_cameras=3, input_size=[160, 240])))
            scene = dict(dataset="waymo", fps=10, num_timesteps=22, scene_id="0", scene_name="test_scene",
                         relative_image_path={c: [f"test_scene/images/{i}_{c}.jpg" for i in range(22)] for c in ("0", "1", "2")})
            (temp/"scene.json").write_text(json.dumps(scene))
            (temp/"list.txt").write_text("scene.json\n")
            common = ["--config", str(cfg), "--data-root", str(temp), "--annotation-file", str(temp/"list.txt"), "--scene-index", "0"]
            jobs = [{"start_index": i, "output": str(temp/f"start_{i}.json")} for i in (0, 1, 2)]
            old = sys.argv
            with contextlib.redirect_stdout(io.StringIO()):
                export_jobs(exporter, common, jobs)
                self.assertIs(sys.argv, old)
                for job in jobs:
                    try:
                        sys.argv = [str(source), *common, "--start-index", str(job["start_index"]), "--output", str(temp/"reference.json")]
                        exporter.main()
                    finally:
                        sys.argv = old
                    actual = json.loads(Path(job["output"]).read_text())
                    self.assertEqual(actual, json.loads((temp/"reference.json").read_text()))
                    self.assertEqual(len(actual["images"]), 60)
                    self.assertEqual(len({(e["frame_id"], e["camera_id"]) for e in actual["images"]}), 60)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *REST])
