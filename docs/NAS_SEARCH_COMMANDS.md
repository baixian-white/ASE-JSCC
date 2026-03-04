# NAS 搜索命令汇总

本文档只整理 **搜索阶段** 命令（`search_channel_aware.py`）。

## 1. 环境检查（可选）

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 2. 快速试跑（先验证流程）

```bash
python scripts/nas/search_channel_aware.py --dataset_name Soya --train_dir data/SoyaHealthVision/train --valid_dir data/SoyaHealthVision/valid --output_dir runs/nas_search --device cuda:0 --image_size 256 --batch_size 64 --num_workers 4 --num_samples 8 --search_epochs 1 --max_train_batches 80 --max_eval_batches 20 --top_k 5
```

## 3. 推荐正式搜索（默认优先）

```bash
python scripts/nas/search_channel_aware.py --dataset_name Soya --train_dir data/SoyaHealthVision/train --valid_dir data/SoyaHealthVision/valid --output_dir runs/nas_search --device cuda:0 --image_size 256 --batch_size 80 --num_workers 6 --num_samples 36 --search_epochs 3 --top_k 8 --lambda_param 0.01 --lambda_tx 0.01 --lambda_robust 0.2 --lambda_rate 0.2 --target_rate 0.7 --min_dynamic_cr 0.3 --max_dynamic_cr 1.0 --rate_blend_alpha 0.7
```

## 4. 过夜强搜索（算力允许时）

```bash
python scripts/nas/search_channel_aware.py --dataset_name Soya --train_dir data/SoyaHealthVision/train --valid_dir data/SoyaHealthVision/valid --output_dir runs/nas_search --device cuda:0 --image_size 256 --batch_size 96 --num_workers 8 --num_samples 80 --search_epochs 4 --top_k 10 --lambda_param 0.01 --lambda_tx 0.01 --lambda_robust 0.2 --lambda_rate 0.2 --target_rate 0.7 --min_dynamic_cr 0.3 --max_dynamic_cr 1.0 --rate_blend_alpha 0.7
```

## 5. 低显存保守搜索（OOM 时使用）

```bash
python scripts/nas/search_channel_aware.py --dataset_name Soya --train_dir data/SoyaHealthVision/train --valid_dir data/SoyaHealthVision/valid --output_dir runs/nas_search --device cuda:0 --image_size 224 --batch_size 32 --num_workers 4 --num_samples 20 --search_epochs 2 --max_train_batches 120 --max_eval_batches 40 --top_k 5
```

## 6. 消融搜索命令

## 6.1 关闭动态码率（固定 cr）

```bash
python scripts/nas/search_channel_aware.py --dataset_name Soya --train_dir data/SoyaHealthVision/train --valid_dir data/SoyaHealthVision/valid --output_dir runs/nas_search --device cuda:0 --batch_size 80 --num_workers 6 --num_samples 36 --search_epochs 3 --top_k 8 --disable_dynamic_rate
```

## 6.2 关闭信道条件筛选（仅语义门控）

```bash
python scripts/nas/search_channel_aware.py --dataset_name Soya --train_dir data/SoyaHealthVision/train --valid_dir data/SoyaHealthVision/valid --output_dir runs/nas_search --device cuda:0 --batch_size 80 --num_workers 6 --num_samples 36 --search_epochs 3 --top_k 8 --disable_channel_condition
```

## 6.3 同时关闭动态码率 + 信道条件筛选（退化到简化筛选）

```bash
python scripts/nas/search_channel_aware.py --dataset_name Soya --train_dir data/SoyaHealthVision/train --valid_dir data/SoyaHealthVision/valid --output_dir runs/nas_search --device cuda:0 --batch_size 80 --num_workers 6 --num_samples 36 --search_epochs 3 --top_k 8 --disable_dynamic_rate --disable_channel_condition
```

## 7. 目标偏好搜索命令

## 7.1 更强调通信预算（更省传输）

```bash
python scripts/nas/search_channel_aware.py --dataset_name Soya --train_dir data/SoyaHealthVision/train --valid_dir data/SoyaHealthVision/valid --output_dir runs/nas_search --device cuda:0 --batch_size 80 --num_workers 6 --num_samples 36 --search_epochs 3 --top_k 8 --lambda_tx 0.03 --lambda_rate 0.4 --target_rate 0.6
```

## 7.2 更强调鲁棒性（跨 SNR 稳定）

```bash
python scripts/nas/search_channel_aware.py --dataset_name Soya --train_dir data/SoyaHealthVision/train --valid_dir data/SoyaHealthVision/valid --output_dir runs/nas_search --device cuda:0 --batch_size 80 --num_workers 6 --num_samples 36 --search_epochs 3 --top_k 8 --lambda_robust 0.4 --eval_snr_list 0,2,4,6,8,12,16,20,24,28
```

## 7.3 更强调精度（弱化开销惩罚）

```bash
python scripts/nas/search_channel_aware.py --dataset_name Soya --train_dir data/SoyaHealthVision/train --valid_dir data/SoyaHealthVision/valid --output_dir runs/nas_search --device cuda:0 --batch_size 80 --num_workers 6 --num_samples 36 --search_epochs 3 --top_k 8 --lambda_param 0.005 --lambda_tx 0.005 --lambda_rate 0.1
```

## 8. 多随机种子批量搜索（PowerShell）

```powershell
$seeds = 42, 43, 44
foreach ($s in $seeds) {
  python scripts/nas/search_channel_aware.py --dataset_name Soya --train_dir data/SoyaHealthVision/train --valid_dir data/SoyaHealthVision/valid --output_dir runs/nas_search --device cuda:0 --seed $s --batch_size 80 --num_workers 6 --num_samples 24 --search_epochs 2 --top_k 6
}
```

## 9. 常用参数说明

- `--num_samples`：采样架构数量，越大越慢。
- `--search_epochs`：每个架构的短训练轮数，越大越准但更慢。
- `--max_train_batches/--max_eval_batches`：用于控时；`0` 表示不限制。
- `--target_rate`：目标平均动态码率阈值。
- `--lambda_rate`：超过目标码率的惩罚强度。

## 10. 结果文件

每次搜索会在 `runs/nas_search/<dataset>_<timestamp>/` 产出：

- `search_results.jsonl`
- `best_arch.json`
- `topk_arches.json`
- `summary.json`

