"""Import the existing RGB-only manifest exporter once per scene."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def export_jobs(exporter, common, jobs):
    previous_argv = sys.argv
    try:
        for job in jobs:
            target = Path(job["output"])
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".json", delete=False) as f:
                temporary = Path(f.name)
            try:
                sys.argv = [str(exporter.__file__), *common, "--start-index", str(job["start_index"]),
                            "--output", str(temporary)]
                exporter.main()
                manifest = json.loads(temporary.read_text())
                if manifest["start_index"] != job["start_index"] or not manifest["images"]:
                    raise RuntimeError(f"invalid exported manifest: {target}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    finally:
        sys.argv = previous_argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ufo-root", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--scene-index", required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    source = a.source_ufo_root / "tools/export_rgb_only_manifest.py"
    spec = importlib.util.spec_from_file_location("r9_original_manifest_exporter", source)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    jobs = json.loads(a.jobs.read_text())
    common = ["--config", a.config, "--data-root", a.data_root,
              "--annotation-file", a.annotation_file, "--scene-index", a.scene_index]
    export_jobs(exporter, common, jobs)
    print(f"[manifest/done] scene={a.scene_index} windows={len(jobs)} "
          f"wall_s={time.perf_counter() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
