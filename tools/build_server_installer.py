#!/usr/bin/env python3
"""Build a self-contained installer from this checkout's standalone additions."""
import base64
import hashlib
import io
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = r'''#!/usr/bin/env python3
"""Create yx/adagent_posefree, install the standalone entry, copy installed assets.
Does not run inference/training or modify the original model/training directories.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

PAYLOAD = "__PAYLOAD__"
SHA256 = "__SHA256__"
BASE_COMMIT = "73aab6cfb8be904779713ade9565009cbc63077c"
REMOTE = "https://github.com/AKkzzzz/adagent_posefree.git"


def run(args, cwd=None, capture=False):
    return subprocess.run(args, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx"))
    p.add_argument("--source-root", type=Path)
    p.add_argument("--skip-weights", action="store_true")
    p.add_argument("--push", action="store_true")
    a = p.parse_args()
    target = a.base.resolve() / "adagent_posefree"
    source = (a.source_root or a.base / "ufoposefree").resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("source and new checkout must be independent directories")
    for path in (source/"vggt-omega/vggt_omega", source/"moge/moge"):
        if not path.is_dir():
            raise FileNotFoundError(path)
    payload = base64.b64decode(PAYLOAD)
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise RuntimeError("installer payload checksum mismatch")
    archive = zipfile.ZipFile(io.BytesIO(payload))
    files = archive.namelist()
    if any(Path(n).is_absolute() or ".." in Path(n).parts for n in files):
        raise RuntimeError("invalid installer member")
    if not target.exists():
        run(["git", "clone", "--branch", "main", REMOTE, str(target)])
    if not (target/".git").is_dir():
        raise RuntimeError(f"{target} exists but is not a Git checkout; no files changed")
    origin = run(["git", "remote", "get-url", "origin"], target, True).strip()
    if origin not in (REMOTE, REMOTE[:-4], "git@github.com:AKkzzzz/adagent_posefree.git"):
        raise RuntimeError("destination has an unexpected Git remote")
    if run(["git", "branch", "--show-current"], target, True).strip() != "main":
        raise RuntimeError("destination must be on main")
    receipt = target / "local/install_receipt.json"
    current = run(["git", "rev-parse", "HEAD"], target, True).strip()
    status = run(["git", "status", "--porcelain", "--untracked-files=normal"], target, True).strip()
    if receipt.is_file():
        if json.loads(receipt.read_text()).get("payload_sha256") != SHA256:
            raise RuntimeError("a different standalone installer was used; review the checkout first")
        if any(not (target/n).is_file() or (target/n).read_bytes() != archive.read(n) for n in files):
            raise RuntimeError("installed wrapper files changed; refusing to overwrite your edits")
    elif current != BASE_COMMIT or status:
        raise RuntimeError("destination must be a clean snapshot at 73aab6c; no reset/overwrite is performed")
    # All code writes target only the independent checkout.
    for name in files:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".install-tmp")
        temporary.write_bytes(archive.read(name))
        os.replace(temporary, path)
    receipt.parent.mkdir(exist_ok=True)
    receipt.write_text(json.dumps(dict(payload_sha256=SHA256)) + "\n")
    command = [sys.executable, str(target/"tools/import_server_assets.py"), "--source-root", str(source)]
    if not a.skip_weights:
        command.append("--copy-weights")
    run(command, target)
    # Archive only explicitly installed wrapper files and allowlisted vendor sources.
    run(["git", "add", "--", *files, "vendor"], target)
    staged = run(["git", "diff", "--cached", "--name-only"], target, True).splitlines()
    for name in staged:
        if name not in files and not name.startswith("vendor/"):
            raise RuntimeError("unrelated staged files found; refusing to include them in the installer commit")
        if name.startswith(("checkpoints/", "data/", "outputs/", "local/")) or Path(name).suffix in (".pt", ".pth", ".npz", ".safetensors"):
            raise RuntimeError("unexpected large/private asset staged; inspect before committing")
    if staged:
        run(["git", "-c", "user.name=ADAgent setup", "-c", "user.email=snapshot@localhost",
             "commit", "--quiet", "-m", "Add standalone RGB camera preprocessing and vendor installed model sources"], target)
    print(f"STANDALONE_INSTALL=PASS\nDIRECTORY={target}", flush=True)
    if a.push:
        if shutil.which("gh"):
            account = run(["gh", "api", "--hostname", "github.com", "user", "--jq", ".login"], target, True).strip()
            if account.lower() != "akkzzzz":
                raise RuntimeError("GitHub CLI must be logged in as AKkzzzz; local installation and commit remain available")
            run(["git", "config", "--local", "--replace-all", "credential.https://github.com.helper", ""], target)
            run(["git", "config", "--local", "--add", "credential.https://github.com.helper", "!gh auth git-credential"], target)
        pushurls = subprocess.run(["git", "config", "--get-all", "remote.origin.pushurl"], cwd=target, capture_output=True, text=True)
        if pushurls.returncode == 0:
            raise RuntimeError("unexpected origin pushurl; inspect before publishing")
        run(["git", "push", "--set-upstream", "origin", "main"], target)
        local = run(["git", "rev-parse", "HEAD"], target, True).strip()
        remote = run(["git", "ls-remote", "origin", "refs/heads/main"], target, True).split()[0]
        if local != remote:
            raise RuntimeError("remote commit did not match local HEAD")
        print(f"PUSH=PASS COMMIT={local}", flush=True)
    print("No GPU job was started. See SERVER_SETUP.md for the small-scene smoke run.", flush=True)


if __name__ == "__main__":
    main()
'''


def main():
    selected = [".gitignore", "README.md", "docs_snapshot_README.md", "THIRD_PARTY_NOTICES.md",
                "SERVER_SETUP.md", "pyproject.toml", "requirements-inference.txt",
                "tools/import_server_assets.py", "tools/download_checkpoints.py", "tools/build_server_installer.py"]
    for directory in ("adagent_posefree", "configs", "examples", "tests"):
        selected += [str(p.relative_to(ROOT)) for p in (ROOT/directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".md")]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(selected):
            archive.writestr(name, (ROOT/name).read_bytes())
    payload = stream.getvalue()
    installer = TEMPLATE.replace("__PAYLOAD__", base64.b64encode(payload).decode()).replace("__SHA256__", hashlib.sha256(payload).hexdigest())
    destination = ROOT / "releases/install_adagent_posefree.py"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(installer)
    print(destination, "bytes=", destination.stat().st_size)


if __name__ == "__main__":
    main()
