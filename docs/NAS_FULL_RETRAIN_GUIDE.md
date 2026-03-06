# NAS 全量重训练指南（UCMerced_LandUse）

本文档对应脚本：`scripts/nas/retrain_candidate.py`  
目标：把 NAS 搜索出的候选架构做完整训练，并输出可复现实验结果。

## 1. 全量重训练逻辑

脚本每次只重训练 **一个架构**（通过 `--arch_json` 指定），流程如下：

1. 读取架构 JSON（通常来自 `runs/nas_search/<run_id>/best_arch.json`）。
2. 构建 train/valid dataloader（`ImageFolder` 目录结构）。
3. 初始化 `ChannelAwareClassifier`（可开关动态码率、信道条件、预训练骨干）。
4. 进入 epoch 循环：
5. 训练阶段：每个 batch 随机采样 `(channel_type, snr)` 进行训练。
6. 验证阶段：在 `channel_type × SNR` 网格上评估，得到 `mean_acc/worst_acc/acc_map`。
7. 用 `ReduceLROnPlateau` 按 `mean_valid_loss` 自适应降学习率。
8. 以 `mean_acc` 保存最佳权重 `best_model.pth`，同时保存最后一轮 `final_model.pth`。
9. 写出 `summary.json`（含每轮历史和网格评估结果）。

## 2. 参数总表（逐项说明）

| 参数 | 类型 | 默认值 | 作用 | 何时调整 |
| --- | --- | --- | --- | --- |
| `--arch_json` | str | 必填 | 要重训的架构 JSON 路径 | 必填，通常传 `best_arch.json` |
| `--dataset_name` | str | `UCMerced_LandUse` | 结果目录命名标识 | 多数据集实验时区分 run 名称 |
| `--train_dir` | str | `data/UCMerced_LandUse/UCMerced_LandUse-train` | 训练集目录（ImageFolder） | 换数据集时改 |
| `--valid_dir` | str | `data/UCMerced_LandUse/UCMerced_LandUse-valid` | 验证集目录（ImageFolder） | 换数据集时改 |
| `--output_dir` | str | `runs/nas_retrain` | 重训输出根目录 | 想按实验分组时改 |
| `--device` | str | `cuda:0` | 训练设备 | 无 GPU 时用 `cpu` |
| `--seed` | int | `42` | 随机种子 | 做多 seed 复现实验时改 |
| `--batch_size` | int | `64` | 训练/验证 batch size | 显存不足时减小 |
| `--num_workers` | int | `0` | dataloader 并行进程数 | I/O 慢时增大（如 4/8） |
| `--image_size` | int | `256` | 输入图像缩放尺寸 | 显存不足可降到 224 |
| `--epochs` | int | `120` | 总训练轮数 | 快速验证可先 20~40 |
| `--lr` | float | `3e-4` | Adam 初始学习率 | 收敛慢可小幅增大/减小 |
| `--weight_decay` | float | `1e-4` | Adam 权重衰减 | 过拟合时可增大 |
| `--eval_channel_types` | str | `AWGN,Fading,Combined_channel` | 验证信道类型列表（逗号分隔） | 只看单信道可简化 |
| `--eval_snr_list` | str | `0,4,8,12,16,20,24,28` | 验证 SNR 列表（逗号分隔） | 要更细鲁棒性曲线可加密 |
| `--disable_dynamic_rate` | flag | 关闭（默认启用动态码率） | 关闭后退化为固定 `arch.cr` | 消融实验 |
| `--disable_channel_condition` | flag | 关闭（默认启用信道条件） | 关闭后仅语义门控 | 消融实验 |
| `--min_dynamic_cr` | float | `0.3` | 动态码率下界 | 控制最小传输率 |
| `--max_dynamic_cr` | float | `1.0` | 动态码率上界 | 控制最大传输率 |
| `--rate_blend_alpha` | float | `0.7` | 动态码率与基础 `cr` 混合系数 | 想更偏动态预测时调高 |
| `--disable_pretrained_backbone` | flag | 关闭（默认启用预训练） | 关闭后骨干从零训练 | 纯从零训练对照实验 |
| `--disable_tensorboard` | flag | 关闭（默认启用 TensorBoard） | 关闭后不写入 tensorboard 日志 | 只想最小化 I/O 开销时使用 |

## 3. 训练命令模板（按场景）

### 3.1 标准全量重训练（推荐）

```bash
python scripts/nas/retrain_candidate.py \
  --arch_json runs/nas_search/<run_id>/best_arch.json \
  --dataset_name UCMerced_LandUse \
  --train_dir data/UCMerced_LandUse/UCMerced_LandUse-train \
  --valid_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --output_dir runs/nas_retrain \
  --device cuda:0 \
  --epochs 120 \
  --batch_size 64 \
  --num_workers 6
```

### 3.2 低显存模式（OOM 时）

```bash
python scripts/nas/retrain_candidate.py \
  --arch_json runs/nas_search/<run_id>/best_arch.json \
  --dataset_name UCMerced_LandUse \
  --train_dir data/UCMerced_LandUse/UCMerced_LandUse-train \
  --valid_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --output_dir runs/nas_retrain \
  --device cuda:0 \
  --image_size 224 \
  --batch_size 16 \
  --num_workers 4 \
  --epochs 120
```

### 3.3 CPU 快速冒烟测试（先检查流程）

```bash
python scripts/nas/retrain_candidate.py \
  --arch_json runs/nas_search/<run_id>/best_arch.json \
  --dataset_name UCMerced_LandUse \
  --train_dir data/UCMerced_LandUse/UCMerced_LandUse-train \
  --valid_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --output_dir runs/nas_retrain \
  --device cpu \
  --epochs 2 \
  --batch_size 8 \
  --eval_channel_types AWGN \
  --eval_snr_list 0,8
```

### 3.4 消融：关闭动态码率

```bash
python scripts/nas/retrain_candidate.py \
  --arch_json runs/nas_search/<run_id>/best_arch.json \
  --dataset_name UCMerced_LandUse \
  --train_dir data/UCMerced_LandUse/UCMerced_LandUse-train \
  --valid_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --output_dir runs/nas_retrain \
  --device cuda:0 \
  --disable_dynamic_rate
```

### 3.5 消融：关闭信道条件

```bash
python scripts/nas/retrain_candidate.py \
  --arch_json runs/nas_search/<run_id>/best_arch.json \
  --dataset_name UCMerced_LandUse \
  --train_dir data/UCMerced_LandUse/UCMerced_LandUse-train \
  --valid_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --output_dir runs/nas_retrain \
  --device cuda:0 \
  --disable_channel_condition
```

### 3.6 消融：同时关闭动态码率 + 信道条件

```bash
python scripts/nas/retrain_candidate.py \
  --arch_json runs/nas_search/<run_id>/best_arch.json \
  --dataset_name UCMerced_LandUse \
  --train_dir data/UCMerced_LandUse/UCMerced_LandUse-train \
  --valid_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --output_dir runs/nas_retrain \
  --device cuda:0 \
  --disable_dynamic_rate \
  --disable_channel_condition
```

### 3.7 更细鲁棒性评估（更密 SNR 网格）

```bash
python scripts/nas/retrain_candidate.py \
  --arch_json runs/nas_search/<run_id>/best_arch.json \
  --dataset_name UCMerced_LandUse \
  --train_dir data/UCMerced_LandUse/UCMerced_LandUse-train \
  --valid_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --output_dir runs/nas_retrain \
  --device cuda:0 \
  --eval_snr_list 0,2,4,6,8,10,12,16,20,24,28
```

### 3.8 从零训练骨干网络（无预训练）

```bash
python scripts/nas/retrain_candidate.py \
  --arch_json runs/nas_search/<run_id>/best_arch.json \
  --dataset_name UCMerced_LandUse \
  --train_dir data/UCMerced_LandUse/UCMerced_LandUse-train \
  --valid_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --output_dir runs/nas_retrain \
  --device cuda:0 \
  --disable_pretrained_backbone
```

### 3.9 批量重训练 Top-3（Bash）

说明：`retrain_candidate.py` 一次只训一个架构，下面命令会从 `ranked_results.json` 提取 Top-3 并循环训练。

```bash
RUN_DIR="runs/nas_search/<run_id>"
OUT_ROOT="runs/nas_retrain_top3"
mkdir -p "${OUT_ROOT}/arch_jsons"

python - <<'PY'
import json
from pathlib import Path
run_dir = Path("runs/nas_search/<run_id>")
rows = json.loads((run_dir / "ranked_results.json").read_text(encoding="utf-8"))
out = Path("runs/nas_retrain_top3/arch_jsons")
out.mkdir(parents=True, exist_ok=True)
for i, row in enumerate(rows[:3], start=1):
    (out / f"top{i}_{row['arch_tag']}.json").write_text(
        json.dumps(row["arch"], indent=2), encoding="utf-8"
    )
PY

for arch in ${OUT_ROOT}/arch_jsons/top*.json; do
  python scripts/nas/retrain_candidate.py \
    --arch_json "$arch" \
    --dataset_name UCMerced_LandUse \
    --train_dir data/UCMerced_LandUse/UCMerced_LandUse-train \
    --valid_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
    --output_dir "${OUT_ROOT}" \
    --device cuda:0 \
    --epochs 120 \
    --batch_size 64 \
    --num_workers 6
done
```

## 4. 输出文件说明

每次重训练会在：

`runs/nas_retrain/<dataset>_<timestamp>/`

生成：

- `arch.json`：本次实际重训的架构配置副本。
- `best_model.pth`：按 `mean_acc` 最优的模型权重。
- `final_model.pth`：最后一个 epoch 的模型权重。
- `summary.json`：完整训练历史（每轮 loss/acc/lr/acc_map/per_channel_mean_acc/per_snr_mean_acc/mean_cr 等），并记录 tensorboard 开关与日志目录。
- `tensorboard/`：TensorBoard 事件文件（默认开启）。

## 5. 实验建议

## 6. TensorBoard 记录项（当前实现）

默认开启（可用 `--disable_tensorboard` 关闭），日志目录为：

`runs/nas_retrain/<dataset>_<timestamp>/tensorboard/`

每个 epoch 会写入：

- `train/loss`、`train/mean_cr`、`train/std_cr`
- `valid/mean_loss`、`valid/mean_acc`、`valid/worst_acc`、`valid/robust_gap`
- `valid/mean_cr`、`valid/std_cr`
- `optim/lr`
- `best/mean_acc_so_far`
- `valid/channel_acc/<channel_type>`（分信道均值准确率）
- `valid/snr_acc/<snr>dB`（分 SNR 均值准确率）

运行开始时还会写入：

- `run/arch`（当前架构 JSON）
- `run/config`（关键训练配置）

## 7. 模型对比评测

统一对比评测脚本已放在独立目录：

- `scripts/benchmark/compare_models.py`

使用说明见：

- `docs/MODEL_BENCHMARK_GUIDE.md`

- 论文/报告阶段建议至少重训 Top-3，再比较最终 `best_mean_acc` 和鲁棒性。
- 若显存足够，优先保持 `image_size=256` 与 NAS 搜索阶段一致。
- 当前脚本不支持断点续训；中断后建议重新发起该架构的重训任务。
