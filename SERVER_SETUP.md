# Yang 服务器：独立目录安装与 smoke

目标目录：`/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/adagent_posefree`。
旧源码来源：同级 `ufoposefree`。安装器只读取旧源码/权重，写入新的独立 Git 目录。
不会移动文件、停止任务、启动模型或修改 R9UFO 训练目录。

## 安装和推送

将 `install_adagent_posefree.py` 上传到 yx 后，在能看到 HDD3 的容器整段执行：

```bash
bash <<'BASH'
set -euo pipefail
BASE=/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx
PY="$BASE/ufoposefree/.venv-moge2/bin/python"
"$PY" "$BASE/install_adagent_posefree.py" --base "$BASE" \
  --source-root "$BASE/ufoposefree" --push
git -C "$BASE/adagent_posefree" log -1 --oneline
BASH
```

安装器会 clone `AKkzzzz/adagent_posefree`，添加独立入口，复制模型 package 和许可证，
再复制约 5.5 GiB 权重。源码与来源哈希提交并推送 main，权重排除在 Git 外。
只安装不推送可省略 `--push`。GitHub 认证沿用已登录的 gh，使用兼容 gh 2.4 的 API。
若远端已有新提交，普通 push 可能被拒绝；本地目录/提交保留，不会 force push。
如果新目录已存在且不是本安装器管理的清洁快照，脚本停止，不覆盖用户代码。
复制权重涉及实际磁盘写入，filesystem free 检查不能确认 GPFS 项目配额。

看到 `STANDALONE_INSTALL=PASS` 表示本地复制完成；`PUSH=PASS` 表示远端提交核对成功。
两者都不代表真实 GPU 推理已验证。

## 小场景推理（在有空闲 GPU 的容器执行）

复用现有推理环境，不修改其包版本。新程序加载的模型源码来自新目录 vendor，
不导入旧 UFO 或旧模型 checkout。下面选原 Waymo 列表第一场景的前 21 帧：

```bash
bash <<'BASH'
set -euo pipefail
BASE=/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx
APP="$BASE/adagent_posefree"
PY="$BASE/ufoposefree/.venv-moge2/bin/python"
DATA=/inspire/hdd/global_user/guoluosong-253108120129/yx-ufo/R9UFO/data/UFO_paper
cd "$APP"
mkdir -p local
head -n 1 "$DATA/scene_list/waymo_train.txt" > local/smoke_scene_list.txt
if [ ! -f local/smoke_full_manifest.json ]; then
  "$PY" -m adagent_posefree index-waymo --data-root "$DATA" \
    --annotation local/smoke_scene_list.txt --output local/smoke_full_manifest.json
fi
"$PY" - <<'PY'
import json
from pathlib import Path
p = Path('local/smoke_manifest.json')
if not p.exists():
    manifest = json.loads(Path('local/smoke_full_manifest.json').read_text())
    manifest['scenes'][0]['frames'] = manifest['scenes'][0]['frames'][:21]
    p.write_text(json.dumps(manifest, indent=2) + '\n')
PY
# 根据空闲设备设置；若此容器仅分配一张卡，其编号通常为 0。
export CUDA_VISIBLE_DEVICES=0
"$PY" -m adagent_posefree prepare --manifest local/smoke_manifest.json \
  --config configs/default.json --output outputs/smoke --num-gpus 1
"$PY" -m adagent_posefree validate --output outputs/smoke
BASH
```

默认20帧窗口，21帧产生两个窗口。查看 `outputs/smoke/logs/<场景名>/` 的 omega/gca/metric 日志；
结束时有 `PREPARE_DONE`，验证应输出 `validated_final: 1, pending_or_invalid: 0`。
它是运行和结构 smoke，不代表相机精度已经通过 GT 对比或可视化验收。

原四卡任务继续完成旧缓存，不要立即用新入口再跑一遍全部数据。
后续新任务生成自己的 manifest 与独立输出，卡数只改 `--num-gpus`。

## H200 可见路径

新仓库的程序没有 HDD3 默认依赖。准备 H200 时，把含 vendor 的仓库 clone 到 H200 可见的
global_user 目录；把两份权重复制到那份仓库的 checkpoints（或在 config 写可见路径）。
在 H200 容器安装推理环境，重新生成指向 H200 可见 RGB 路径的 manifest，再运行同一个入口。
不要把带旧绝对路径的 .venv 当成可移植环境，也不要假设现有 UFO 训练环境已具备全部依赖。

## 当前验证边界

开发机没有 CUDA GPU/真实权重。已进行 CPU 回归与模拟多场景端到端检查，包括与旧 metric
转换、SE(3) 对齐的数值对照。4090/H200 实际模型导入、权重加载、显存峰值和结果质量
由上面的真实 smoke 验证；不把模拟模型测试当作 GPU 测试。
