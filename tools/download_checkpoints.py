#!/usr/bin/env python3
"""Get upstream checkpoints on a fresh machine; Omega requires authorized HF access."""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from import_server_assets import ROOT, copy_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--omega-revision", default="main")
    p.add_argument("--moge-revision", default="main")
    a = p.parse_args()
    targets = [("facebook/VGGT-Omega", "vggt_omega_1b_512.pt", a.omega_revision, "checkpoints/vggt_omega_1b_512.pt"),
               ("Ruicheng/moge-2-vitl", "model.pt", a.moge_revision, "checkpoints/moge-2-vitl/model.pt")]
    import json
    fingerprints = {}
    for repo, filename, revision, target in targets:
        local = hf_hub_download(repo_id=repo, filename=filename, revision=revision)
        fingerprints[target] = dict(copy_file(Path(local), ROOT / target), repository=repo, requested_revision=revision)
    (ROOT / "checkpoints/fingerprints.json").write_text(json.dumps(fingerprints, indent=2)+"\n")
    print("CHECKPOINTS=PASS; see checkpoints/fingerprints.json")


if __name__ == "__main__":
    main()
