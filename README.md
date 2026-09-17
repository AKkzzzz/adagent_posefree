# adagent_posefree

独立的 RGB-only 离线相机预处理工具：**VGGT-Omega → MoGe-2/GCA 尺度 → 重叠窗口 SE(3) 对齐**。
入口不依赖完整 UFO 项目。4090 和 H200 使用同一个 CUDA 推理后端，用 `--num-gpus` 控制卡数。
支持单/多场景、任意字符串相机 ID、可配置参考相机；包含现有 Waymo/UFO 适配器。

当前状态：独立调度与接口已完成 CPU 回归验证；真实 Omega/MoGe 推理沿用实跑实现，
独立入口仍需在你的 4090/H200 上执行 smoke。不是已经实测所有 GPU/场景质量的发行版。
旧代码快照保留在 `h200_adapter/`、`source_ufo/`、`omega_adapter/`，来源说明见
`docs_snapshot_README.md`。新运行入口不导入这些旧目录，也不会启动 UFO 训练。

## 1. 初次整理已有服务器

在新的 Git 工作目录运行：

```bash
python tools/import_server_assets.py --source-root /path/to/ufoposefree --copy-weights
```

它把安装中的 `vggt-omega/vggt_omega/`、`moge/moge/` 和许可证复制到本仓库 `vendor/`，
保存每个源码文件的 SHA-256；把约 5.5 GiB 的两份权重复制到 `checkpoints/`。
复制的是实跑工作文件，包含模型代码的未提交修改。不会复制环境、数据、旧输出或旧 .git。
同名不同内容的文件会报错；重复运行接受相同内容，不覆盖人工改动。

随后将 `vendor/` 随独立源码一起提交 Git。`checkpoints/`、数据和输出被 .gitignore 排除。
这样其他机器 clone 后拥有全部项目/模型源码，环境与权重仍需准备。

## 2. 全新机器

```bash
git clone https://github.com/AKkzzzz/adagent_posefree.git
cd adagent_posefree
# 先安装适合本机驱动的 CUDA torch + 匹配 torchvision；不在这里更改驱动。
python -m pip install -r requirements-inference.txt
python -m pip install -e .
# 可选：从上游下载权重，也可以按下文位置手动复制已有权重。
python tools/download_checkpoints.py
python -m adagent_posefree doctor --num-gpus 1
```

请在 `vendor/omega/vggt_omega` 和 `vendor/moge/moge` 已提交的版本使用这套命令。
使用源码 checkout 或 editable install；当前不承诺独立 wheel 含 vendor。
Omega 权重来自 `facebook/VGGT-Omega`，需按上游要求申请访问并在本机登录 Hugging Face；
MoGe 使用 `Ruicheng/moge-2-vitl/model.pt`。不会自动接受上游条款或绕过访问限制。
下载器记录权重 SHA-256；精确复现实跑版本应使用服务器复制的权重与 fingerprints。
环境快照记录 torch 2.10.0 / torchvision 0.25.0；完整旧环境清单不是最小依赖列表。

默认权重位置（可在 JSON 中改，相对路径以**仓库根目录**为基准）：

```text
checkpoints/vggt_omega_1b_512.pt
checkpoints/moge-2-vitl/model.pt
```

`doctor` 验证文件、模型导入和 CUDA/BF16 能力，**不等于已加载权重并完成推理**。
真正验证模型和权重必须执行一个小场景。H200 容器必须能看到仓库、权重、RGB、输出以及
推理 Python；只有 HDD3 上的路径可见性不会被程序自动解决，虚拟环境不要跨路径硬复制。

## 3. 通用图像输入

建议目录（名称自定，图片文件名必须是整数帧号）：

```text
input/scene_000/front/000000.jpg
input/scene_000/front/000001.jpg
input/scene_000/right/000000.jpg
input/scene_000/right/000001.jpg
input/scene_001/front/000000.jpg
...
```

每个 scene 是独立连续序列；不同相机相同编号表示同一时刻。各相机的帧号集合必须相同，
每个场景至少有两张图。第一版不自动处理异步相机或缺失帧。单相机也可使用同一结构。
序列需要足够视觉重叠；相机估计质量不由结构校验保证。

```bash
python -m adagent_posefree index --input /path/to/input --fps 10 \
  --reference-camera front --output local/input.json
python -m adagent_posefree prepare --manifest local/input.json \
  --config configs/default.json --output outputs/demo --num-gpus 4
```

只改 `--num-gpus 4` 就能换卡数。想选特定物理卡/UUID：

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m adagent_posefree prepare \
  --manifest local/input.json --output outputs/demo --num-gpus 2
```

`index` 不搬动图片，输出绝对路径。也可手写 manifest，并用相对 manifest 文件位置的图片路径：

```json
{
  "schema_version": 1,
  "scenes": [{
    "scene_name": "scene_000",
    "fps": 10,
    "camera_ids": ["front"],
    "reference_camera": "front",
    "frames": [
      {"frame_id": 0, "timestamp": 0.0, "images": {"front": "../input/scene_000/front/000000.jpg"}},
      {"frame_id": 1, "timestamp": 0.1, "images": {"front": "../input/scene_000/front/000001.jpg"}}
    ]
  }]
}
```

上例用于说明有效结构；两张图不足以承诺可靠相机质量。输出 K 默认对应 160×240；
无需事先把输入缩到这个尺寸，Omega 内部仍用 512 级别的 crop/resize/pad。
如果下游用其他尺寸，修改 `output_image_size: [H,W]`；下游应对原输入 RGB 做对应的直接 resize。
无需提供 GT pose/K、深度、LiDAR、SAM 或训练 checkpoint。所有提供的窗口 RGB 都参与
估计（all_rgb），这是离线处理，不是 context-only 或因果评估。

## 4. 已有 Waymo 数据

```bash
python -m adagent_posefree index-waymo \
  --data-root /path/to/UFO_paper \
  --annotation /path/to/UFO_paper/scene_list/waymo_train.txt \
  --output local/waymo.json
python -m adagent_posefree prepare --manifest local/waymo.json \
  --output outputs/waymo --num-gpus 4
```

读取 scene_name、fps、num_timesteps、RGB 路径；忽略 GT 几何字段。
保持旧版 `images → images_4` 映射和相机顺序 1/0/2，参考相机 0。
scene list 可以只有一个场景，也可以有 798 个，不要求完整 UFO 安装。
默认窗口 20 帧、步长 1；10 FPS 下与旧实验窗口一致。其他 FPS 需自己明确窗口帧数，
新入口不会把 `timespan=0.5` 自动乘入不同 FPS。

## 5. 输出与使用

```text
outputs/demo/scenes/scene_000/cameras.npz
outputs/demo/scenes/scene_000/metadata.json
outputs/demo/scenes/scene_000/alignment_report.json
outputs/demo/scenes/scene_000/done.json
outputs/demo/global_aligned/scene_000/omega_pose_override.npz
outputs/demo/logs/scene_000/batch_0000_omega.log
outputs/demo/logs/scene_000/batch_0000_gca.log
outputs/demo/logs/scene_000/batch_0000_metric.log
```

`cameras.npz`：

| 字段 | 含义 |
|---|---|
| frame_ids[N], camera_ids[N] | 用这对键定位相机，不能假设数组等同于输入相机顺序 |
| c2w[N,4,4] | OpenCV 相机轴到该场景世界坐标；右、下、前 |
| K[N,3,3] | 像素内参；对应 image_size，而非原始输入图片尺寸 |
| camera_to_world_dataset[N,4,4] | 数据集相机轴版本；普通输入默认与 c2w 相同 |
| timestamps[N], image_paths[N] | 对应输入时间戳和图片路径 |
| image_size[2] | [height,width]，默认 [160,240] |

```bash
python examples/read_cameras.py outputs/demo/scenes/scene_000/cameras.npz
python -m adagent_posefree status --output outputs/demo
python -m adagent_posefree validate --output outputs/demo
```

渲染器需要 w2c 时用 `np.linalg.inv(c2w)`；K 与 RGB 的裁剪/缩放必须一致。
普通输入以选定参考相机定义场景轴，Waymo 另输出其固定相机轴变换。每个场景独立对齐，
不是 GPS 坐标，也不自动把多个场景合成一个世界。米制尺度来自 MoGe/GCA 预测，存在误差。
最终文件不含 RGB、SAM、深度图、mesh 或 Gaussian；下游仍需原图片和它自己的重建模型。

接 UFO 时，外参和内参 override 都指向 `outputs/waymo/global_aligned`，读取原来的
`omega_camera_to_world_global_metric` 和 `predicted_intrinsics_ufo`。
现有 Full-Waymo UFO 训练校验器仍要求 798 场景、相机 1/0/2；通用数据不会因此自动变成
可供 UFO 训练的数据集。这个仓库不包含完整 UFO 训练工程。

## 6. 运行、恢复与缓存

每卡一个场景 worker，共享任务队列；一个场景不会拆到多张卡。同阶段模型在一组窗口内
常驻；`window_batch_size=179` 是模型复用/磁盘缓存组大小，推理 batch 仍是 1。
第一版仍会在场景/阶段切换时重载模型。

Raw 和 scale 在该批 metric 文件校验后删除；窗口文件在整个场景两种最终输出校验通过后
删除。`keep_intermediates=true` 可保留诊断文件。日志、最终相机和 metadata 一直保留。
中断时保留尚未完成的数据，重复运行相同命令恢复；允许改变卡数和 CPU 线程数。
更换输入、图像 size/mtime、模型文件 size/mtime、源码或数值配置需使用新输出目录。
这是恢复一致性检查，不是文件内容的完整加密认证；模型另有复制时的 SHA-256 记录。

旧的 `posefree_camera_full` 缓存可以继续供旧训练读取；**不要把新入口指向正在运行的旧缓存**。
新入口要求自己的输出契约和完成标记，不会自动接管旧作业。没有理由为了测试新入口
重算当前已经完成的 798 场景缓存。

`--check-only` 做配置、输入、空间、GPU 和依赖检查，创建空输出目录但不启动推理。
预留磁盘默认每卡 37.5 GiB，raw 写入另做按图像尺寸估算的检查；这不是平台项目配额查询。
显存不足时缩短 window_frames 会改变估计问题；修改 window_batch_size 主要改变磁盘/复用，
不能认为它会减少每次前向的图像数。4090/H200 不追求 bitwise 一致，跨设备需实测相机误差。

## 7. 验证与来源

```bash
python -m unittest discover -s tests -v
```

CPU 测试覆盖：同步输入校验、Waymo RGB-only 适配、CUDA_VISIBLE_DEVICES 映射、
任意参考相机、单窗口短序列、失败后恢复、输出校验/清理、改变卡数恢复、拒绝混用配置，
以及相机尺度转换和重叠对齐与原实现的合成数据数值一致性。
这些测试不冒充真实模型 GPU 推理；请按 `SERVER_SETUP.md` 运行 smoke。

保留第三方许可证；参见 `THIRD_PARTY_NOTICES.md`、`vendor/provenance.json`。
