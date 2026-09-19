#!/usr/bin/env python3
"""Run RGB camera prediction from an image directory; manage inputs automatically."""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(
        description="输入图片目录，自动生成清单、预测相机并检查结果。中断后重复同一命令续跑。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=Path, required=True,
                   help="图片根目录，结构为：场景/相机/整数帧号.jpg")
    p.add_argument("--layout", choices=("scenes", "scene"), default="scenes",
                   help="scenes：目录内有多个场景；scene：目录本身就是一个场景")
    p.add_argument("--gpus", type=int, default=1, help="使用的 GPU 数量，多卡按场景分配")
    p.add_argument("--fps", type=float, help="图片序列实际帧率，默认 10；Waymo 从数据读取")
    p.add_argument("--output", type=Path, help="结果目录；默认仓库 outputs/图片目录名")
    p.add_argument("--format", choices=("images", "waymo"), default="images",
                   help="普通图片目录用 images；已有 UFO 格式 Waymo 数据用 waymo")
    p.add_argument("--scene-index", type=int,
                   help="仅 Waymo：选择训练列表中的零基场景编号，省略则处理全部场景")
    p.add_argument("--config", type=Path, help="可选的高级配置文件，通常无需设置")
    p.add_argument("--check-only", action="store_true", help="只检查输入及运行条件，不启动推理")
    return p


def input_scenes(data, fmt, fps, scene_index, layout="scenes"):
    from adagent_posefree.data import index_folders, index_waymo, load_manifest

    # Only the runtime writes to the output directory. A temporary input avoids
    # leaving an unowned manifest in a new output or replacing a resumed input.
    with tempfile.TemporaryDirectory(prefix="adagent-input-") as temp:
        work = Path(temp)
        if fmt == "images":
            if layout == "scene":
                # Reuse the same indexer without scanning any sibling scenes.
                # Its paths resolve to the original images, not temporary links.
                parent = work / "single"
                parent.mkdir()
                (parent / (data.name or "scene")).symlink_to(data, target_is_directory=True)
                payload = index_folders(parent, fps)
            else:
                payload = index_folders(data, fps)
        else:
            annotation = data / "scene_list/waymo_train.txt"
            if scene_index is not None:
                entries = [s.strip() for s in annotation.read_text().splitlines() if s.strip()]
                if not 0 <= scene_index < len(entries):
                    raise ValueError(f"Waymo 场景编号应为 0..{len(entries)-1}，收到 {scene_index}")
                entry = Path(entries[scene_index])
                annotation = work / "selected_scene.txt"
                annotation.write_text(str(entry if entry.is_absolute() else data / entry) + "\n")
            payload = index_waymo(data, annotation)
        manifest = work / "input.json"
        manifest.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        return load_manifest(manifest)


def run(a):
    if a.gpus < 1:
        raise ValueError("--gpus 必须为正整数")
    if a.format == "images" and a.scene_index is not None:
        raise ValueError("--scene-index 仅用于 --format waymo")
    if a.format == "waymo" and a.fps is not None:
        raise ValueError("Waymo 会从数据读取帧率，请去掉 --fps")
    if a.format == "waymo" and a.layout != "scenes":
        raise ValueError("Waymo 单场景请使用 --scene-index，普通图片才使用 --layout scene")
    fps = 10.0 if a.fps is None else a.fps
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("--fps 必须是有限的正数")
    data = a.data.expanduser().resolve()
    if not data.is_dir():
        raise FileNotFoundError(f"图片目录不存在或未挂载：{data}")
    name = data.name or "rgb"
    if a.format == "waymo":
        name = "waymo_full" if a.scene_index is None else f"waymo_scene_{a.scene_index:04d}"
    output = (a.output.expanduser() if a.output is not None else ROOT / "outputs" / name).resolve()
    if output == data or output in data.parents or data in output.parents:
        raise ValueError("输入与输出目录不能相互包含；请用 --output 指定独立结果目录")
    print(f"输入：{data}\n结果：{output}\n使用 GPU 数量：{a.gpus}", flush=True)
    scenes = input_scenes(data, a.format, fps, a.scene_index, a.layout)
    frames = sum(len(s["frames"]) for s in scenes)
    images = sum(len(s["frames"]) * len(s["camera_ids"]) for s in scenes)
    print(f"已找到 {len(scenes)} 个场景、{frames} 个时刻、{images} 张图片。", flush=True)

    from adagent_posefree.config import load_config
    from adagent_posefree.runtime import prepare

    config = load_config(a.config)
    print("开始检查运行条件。" if a.check_only else "开始相机预测，已完成的结果会自动跳过。", flush=True)
    prepare(scenes, config, output, a.gpus, a.check_only)
    if a.check_only:
        print("检查通过，尚未启动推理。去掉 --check-only 即可运行。", flush=True)
        return

    from adagent_posefree.output import complete

    contract = json.loads((output / ".posefree_contract.json").read_text())
    pending = [s["scene_name"] for s in scenes if not complete(output, s, contract["run_signature"])]
    if pending:
        raise RuntimeError(f"还有 {len(pending)} 个场景的结果不完整，请用同一命令续跑：{pending[:5]}")
    print(f"POSEFREE_DONE {len(scenes)}/{len(scenes)}\n相机结果：{output}/scenes/<场景名>/cameras.npz", flush=True)


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        run(a)
    except KeyboardInterrupt:
        print("\n已中断。再次执行同一命令即可续跑。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        message = str(exc)
        if "different input/config/code/assets run" in message:
            message = "输入、配置、代码或权重与已有结果不一致。恢复原设置续跑，或用 --output 指定新目录。"
        elif "camera frame IDs are not synchronized" in message:
            message = "同一场景各相机的帧号不一致。请检查缺帧，并确保同号图片对应同一时刻。"
        elif "image stems must be distinct integer frame IDs" in message:
            message = "图片文件名应为不重复的整数帧号，例如 000000.jpg、000001.jpg。"
        elif "no camera image folders" in message:
            message = "未找到相机图片目录。目录内是一批场景用 --layout scenes；目录本身是一个场景用 --layout scene。"
        print(f"运行失败：{message}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
