# 组内服务器直接使用

仓库、环境和权重已经准备好。先安装含 `run_posefree.py` 的更新，再运行下面命令。

## 用现成数据跑 Waymo 621

在分配了 GPU、能看到 hdd3 和 global_user 的容器里整段执行：

```bash
bash <<'BASH'
set -euo pipefail
BASE=/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx
GLOBAL=/inspire/hdd/global_user/guoluosong-253108120129/yx-ufo
PY="$BASE/ufoposefree/.venv-moge2/bin/python"
cd "$BASE/adagent_posefree"
"$PY" run_posefree.py --data "$GLOBAL/R9UFO/data/UFO_paper" \
    --format waymo --scene-index 621 --gpus 1
BASH
```

结果在仓库 `outputs/waymo_scene_0621/`，与此前已跑通的 `outputs/scene_0621/` 分开。此前结果仍可直接读取，无需为使用新脚本重算。换场景只改 `--scene-index`。Waymo 从数据读取帧率，不加 `--fps`。

## 自己的图片

在上面的仓库和 PY 环境中，按输入情况选择一条：

```bash
# data 下是一批场景，全部处理
"$PY" run_posefree.py --data /你的图片根目录 --layout scenes --gpus 4

# data 本身是一个场景，下面直接是相机目录
"$PY" run_posefree.py --data /你的图片根目录/scene_001 --layout scene --gpus 1
```

默认 10 FPS，其他帧率加 `--fps`。中断后重复同一命令续跑，无需操作 manifest。不要为测试新入口重复启动原来的 798 场景全量任务。

## 现成路径

以下以 `BASE=/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx` 为前缀：

| 内容 | 路径 |
| --- | --- |
| 仓库 | `$BASE/adagent_posefree` |
| Python 环境 | `$BASE/ufoposefree/.venv-moge2/bin/python` |
| Omega 权重 | `$BASE/adagent_posefree/checkpoints/vggt_omega_1b_512.pt` |
| MoGe 权重 | `$BASE/adagent_posefree/checkpoints/moge-2-vitl/model.pt` |
| 已跑通的 621 结果根目录 | `$BASE/adagent_posefree/outputs/scene_0621` |

H200 容器若看不到 hdd3，把仓库和权重放到该容器可见的 global_user 目录，并安装推理环境。运行脚本相同，输入输出都须使用容器内可见路径。
