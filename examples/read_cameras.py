"""Usage: python examples/read_cameras.py outputs/demo/scenes/scene_000/cameras.npz"""
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as cameras:
    lookup = {(int(f), str(c)): i for i, (f, c) in enumerate(zip(cameras["frame_ids"], cameras["camera_ids"]))}
    first = next(iter(lookup))
    i = lookup[first]
    c2w = cameras["c2w"][i]
    w2c = np.linalg.inv(c2w)
    K = cameras["K"][i]
    print("frame,camera:", first, "image:", cameras["image_paths"][i])
    print("K image size [H,W]:", cameras["image_size"])
    print("camera-to-world:\n", c2w)
    print("world-to-camera:\n", w2c)
    print("intrinsics (pixels):\n", K)
