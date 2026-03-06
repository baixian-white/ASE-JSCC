# Model Benchmark Guide

This guide explains how to benchmark two checkpoints with scripts isolated in:

- `scripts/benchmark/`

## 1. Script

- `scripts/benchmark/compare_models.py`

Supported model types:

- `legacy`: checkpoint from `scripts/train/ASE-JSCCtrain.py`
- `nas`: checkpoint from `scripts/nas/retrain_candidate.py`

Outputs (under `runs/model_benchmark/<run_name>/`):

- `benchmark_results.json`
- `benchmark_table.csv`
- `benchmark_summary.md`
- `fig_channel_acc_compare.png`
- `fig_delta_ci95.png`

## 2. Typical command (Legacy vs NAS)

```bash
python scripts/benchmark/compare_models.py \
  --data_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --output_dir runs/model_benchmark \
  --run_name legacy_vs_nas_best \
  --device cuda:0 \
  --batch_size 64 \
  --num_workers 6 \
  --image_size 256 \
  --mc_runs 5 \
  --channel_types AWGN,Fading,Combined_channel \
  --model_a_name legacy_baseline \
  --model_a_type legacy \
  --model_a_ckpt checkpoint/your_legacy_best.pth \
  --model_a_legacy_cr 0.8 \
  --model_b_name nas_best \
  --model_b_type nas \
  --model_b_ckpt runs/nas_retrain/<run_id>/best_model.pth \
  --model_b_arch_json runs/nas_search/<search_run_id>/best_arch.json
```

## 3. NAS vs NAS command

```bash
python scripts/benchmark/compare_models.py \
  --data_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --run_name nas_top1_vs_top2 \
  --device cuda:0 \
  --mc_runs 5 \
  --model_a_name nas_top1 \
  --model_a_type nas \
  --model_a_ckpt runs/nas_retrain/<top1_run>/best_model.pth \
  --model_a_arch_json runs/nas_search/<search_run_id>/best_arch.json \
  --model_b_name nas_top2 \
  --model_b_type nas \
  --model_b_ckpt runs/nas_retrain/<top2_run>/best_model.pth \
  --model_b_arch_json runs/nas_retrain_top3/arch_jsons/top2_<tag>.json
```

## 4. Notes

- For strict fairness, use the same `data_dir`, `batch_size`, and `image_size`.
- `mc_runs` controls Monte-Carlo repeats per channel type; larger is more stable.
- NAS forward uses random SNR per batch in `[nas_eval_snr_min, nas_eval_snr_max]`.
- Legacy model samples SNR internally, so repeated runs are required for stable estimates.
- Legacy benchmark path now dynamically loads `scripts/train/ASE-JSCCtrain.py` and reuses
  `SatelliteClassifierWithAttention` directly (including its channel simulation logic).
