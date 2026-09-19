# adagent posefree

把连续图片转换为每张图对应的相机外参和内参，供 UFO、3DGS 等程序使用。环境和权重准备好后，只运行一个脚本；无需自己准备 manifest。

## 输入怎么放

批量输入按 `图片根目录/场景名/相机名/整数帧号.jpg` 组织，也支持 PNG。

```text
rgb/scene_001/front/000000.jpg
rgb/scene_001/front/000001.jpg
rgb/scene_001/right/000000.jpg
rgb/scene_001/right/000001.jpg
rgb/scene_002/front/000000.jpg
rgb/scene_002/front/000001.jpg
```

每个场景是一段连续序列；单相机只放一个相机目录。多相机帧号需一致，同号图片对应同一时刻。实际放入完整序列，首次可用 20 至 30 个同步时刻。无需 GT 相机、深度或 SAM，不直接读取视频。

## 运行哪个脚本

在仓库目录、已激活的推理环境里执行。把 `/data/rgb` 换成实际路径：

```bash
# 批量处理目录下所有场景，例如 729 个场景，使用 4 张卡
python run_posefree.py --data /data/rgb --layout scenes --gpus 4

# 只处理一个场景，目录下直接是 front、right 等相机目录
python run_posefree.py --data /data/rgb/scene_001 --layout scene --gpus 1
```

这两条命令各自都会完成输入整理、相机预测和结果检查，按需选一条运行。

| 参数 | 含义 |
| --- | --- |
| `--data` | 要处理的图片目录，必填 |
| `--layout scenes` | data 下是一批场景，全部处理；这是默认值 |
| `--layout scene` | data 本身就是一个场景 |
| `--gpus` | 使用几张卡，默认 1 |
| `--fps` | 实际帧率，默认 10；20 FPS 加 `--fps 20` |
| `--output` | 结果目录，默认仓库 `outputs/输入目录名` |

多卡按场景并行，场景数量不限于 729 或 798；单个场景仍只用一张卡。4090/H200 使用同一入口。中断后重复原命令即可续跑，卡数可以改变。换图片或帧率时，用 `--output` 指定新目录。

## 输出是什么

输入 `/data/rgb` 的默认结果：

```text
outputs/rgb/scenes/scene_001/cameras.npz
outputs/rgb/scenes/scene_001/metadata.json
outputs/rgb/global_aligned/scene_001/omega_pose_override.npz
```

每个场景都有自己的结果目录。主要读取 `cameras.npz`：

| 字段 | 内容 |
| --- | --- |
| `c2w` | 每张图的相机到世界坐标外参，OpenCV 相机轴 |
| `K` | 每张图的内参，默认对应高 160、宽 240 |
| `frame_ids`、`camera_ids` | 对应的帧号与相机名称 |
| `image_paths` | 对应的输入图片路径 |

```bash
python examples/read_cameras.py outputs/rgb/scenes/scene_001/cameras.npz
```

默认 K 要配原图直接缩放到 160×240 使用；需要 world-to-camera 时对 c2w 求逆。每个场景有独立坐标系，尺度由模型估计。UFO 兼容文件位于 `global_aligned`。工具输出相机参数，UFO/3DGS 训练另行启动。

终端出现 `POSEFREE_DONE N/N` 表示结果检查通过。成功后自动清理大体积中间缓存，保留最终相机与日志；中断时保留可续跑的数据。

## 新服务器首次安装

需要 Linux、Python 3.10，以及支持 CUDA/BF16 的 NVIDIA GPU。先用有仓库权限的账号克隆并安装环境。下面使用已有实跑的 PyTorch 版本组合，驱动需兼容 CUDA 12.8：

```bash
git clone https://github.com/AKkzzzz/adagent_posefree.git
cd adagent_posefree
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
TORCH_INDEX=https://download.pytorch.org/whl/cu128
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url "$TORCH_INDEX"
python -m pip install -r requirements-inference.txt
python -m pip install -e .
hf auth login
python tools/download_checkpoints.py
```

模型源码已在 Git 中，两份权重另外下载或复制到以下位置：

| 权重 | 仓库内位置 |
| --- | --- |
| [VGGT Omega](https://huggingface.co/facebook/VGGT-Omega)，约 4.26 GiB | `checkpoints/vggt_omega_1b_512.pt` |
| [MoGe 2 ViT L](https://huggingface.co/Ruicheng/moge-2-vitl)，约 1.22 GiB | `checkpoints/moge-2-vitl/model.pt` |

Omega 下载需先取得 Hugging Face 模型访问权限；遵守上游许可。安装完成后直接运行 `run_posefree.py`。组内服务器已有环境和权重，直接看 [SERVER_SETUP.md](SERVER_SETUP.md)。

现有推理流程已在服务器 4090 上完成场景 621；新的一键脚本已通过 CPU 流程测试，尚未在服务器实际启动。H200 独立版本仍待实机验证。开发测试命令：`python -m unittest discover -s tests -p test_simple_runner.py -v`。来源与许可证见 `docs_snapshot_README.md`、`THIRD_PARTY_NOTICES.md`。
