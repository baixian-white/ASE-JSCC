# NAS 代码快速开始

本文档说明如何运行当前的信道感知多目标 NAS 流程。

## 1. 搜索架构

```bash
python scripts/nas/search_channel_aware.py \
  --dataset_name UCMerced_LandUse \
  --train_dir data/UCMerced_LandUse/UCMerced_LandUse-train \
  --valid_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --output_dir runs/nas_search \
  --num_samples 20 \
  --search_epochs 2 \
  --batch_size 32 \
  --eval_channel_types AWGN,Fading,Combined_channel \
  --eval_snr_list 0,4,8,12,16,20,24,28
```

如果希望对整个离散搜索空间做全量 proxy 评估，而不是随机采样，可以使用：

```bash
python scripts/nas/search_channel_aware.py \
  --dataset_name UCMerced_LandUse \
  --train_dir data/UCMerced_LandUse/UCMerced_LandUse-train \
  --valid_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --output_dir runs/nas_search \
  --exhaustive_search \
  --search_epochs 2 \
  --batch_size 32 \
  --eval_channel_types AWGN,Fading,Combined_channel \
  --eval_snr_list 0,4,8,12,16,20,24,28
```

可选的动态选择器控制项：

```bash
--disable_dynamic_rate                # 回退为固定 arch.cr
--disable_channel_condition           # 只使用普通语义门控
--min_dynamic_cr 0.3
--max_dynamic_cr 1.0
--rate_blend_alpha 0.7
--target_cr 0.7
--lambda_cr 0.2
```

输出文件：

- `search_results.jsonl`：所有已评估架构的结果。
- `best_arch.json`：按加权分数选出的最优架构。
- `topk_arches.json`：top-k 架构列表。
- `summary.json`：整体汇总信息和动态选择器配置。

## 2. 重训练选中的架构

```bash
python scripts/nas/retrain_candidate.py \
  --arch_json runs/nas_search/<run_id>/best_arch.json \
  --dataset_name UCMerced_LandUse \
  --train_dir data/UCMerced_LandUse/UCMerced_LandUse-train \
  --valid_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --output_dir runs/nas_retrain \
  --epochs 120 \
  --batch_size 64
```

可选的动态选择器控制项与搜索阶段相同。

如果需要查看完整重训练流程、全部参数说明和按场景整理的命令模板，请参考：

- `docs/NAS全量重训练指南.md`

输出文件：

- `best_model.pth`：按验证集平均精度保存的最佳 checkpoint。
- `final_model.pth`：最后一个 epoch 的 checkpoint。
- `summary.json`：训练历史以及 `mean_cr/std_cr` 统计信息。

## 3. 计算 Pareto 前沿

```bash
python scripts/nas/evaluate_pareto.py \
  --results_jsonl runs/nas_search/<run_id>/search_results.jsonl \
  --top_k 20
```

输出文件：

- `pareto_front.json`
- `pareto_topk.json`
- `pareto_summary.md`

## 4. 搜索目标

当前使用的加权分数为：

```text
score = mean_acc
      - lambda_param * param_m
      - lambda_tx * tx_norm
      - lambda_robust * robust_gap
      - lambda_cr * max(0, mean_cr - target_cr)
```

其中：

- `mean_acc`：越大越好
- `param_m`：越小越好
- `tx_cost` / `tx_norm`：越小越好
- `robust_gap`：越小越好
- `mean_cr`：超过目标 `cr` 的部分会被惩罚

## 5. 实用建议

- 所有脚本都默认输入为 `ImageFolder` 格式的 `train/valid` 目录。
- 如果在 CPU 上运行，建议适当减小：
  - `--num_samples`
  - `--search_epochs`
  - `--max_train_batches`
  - `--max_eval_batches`
