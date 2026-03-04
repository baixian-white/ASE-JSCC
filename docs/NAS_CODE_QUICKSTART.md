# NAS Code Quickstart

This guide explains how to run the current channel-aware multi-objective NAS pipeline.

## 1. Search architectures

```bash
python scripts/nas/search_channel_aware.py \
  --dataset_name Soya \
  --train_dir data/SoyaHealthVision/train \
  --valid_dir data/SoyaHealthVision/valid \
  --output_dir runs/nas_search \
  --num_samples 20 \
  --search_epochs 2 \
  --batch_size 32 \
  --eval_channel_types AWGN,Fading,Combined_channel \
  --eval_snr_list 0,4,8,12,16,20,24,28
```

Optional dynamic-selector controls:

```bash
--disable_dynamic_rate                # fallback to fixed arch.cr
--disable_channel_condition           # use plain semantic gate only
--min_dynamic_cr 0.3
--max_dynamic_cr 1.0
--rate_blend_alpha 0.7
--target_rate 0.7
--lambda_rate 0.2
```

Outputs:

- `search_results.jsonl`: all evaluated architectures.
- `best_arch.json`: best architecture by weighted score.
- `topk_arches.json`: top-k architecture list.
- `summary.json`: overall summary and dynamic-selector settings.

## 2. Retrain a selected architecture

```bash
python scripts/nas/retrain_candidate.py \
  --arch_json runs/nas_search/<run_id>/best_arch.json \
  --dataset_name Soya \
  --train_dir data/SoyaHealthVision/train \
  --valid_dir data/SoyaHealthVision/valid \
  --output_dir runs/nas_retrain \
  --epochs 120 \
  --batch_size 64
```

Optional dynamic-selector controls are the same as search.

Outputs:

- `best_model.pth`: best checkpoint by mean validation accuracy.
- `final_model.pth`: last epoch checkpoint.
- `summary.json`: training history + `mean_cr/std_cr` stats.

## 3. Compute Pareto front

```bash
python scripts/nas/evaluate_pareto.py \
  --results_jsonl runs/nas_search/<run_id>/search_results.jsonl \
  --top_k 20
```

Outputs:

- `pareto_front.json`
- `pareto_topk.json`
- `pareto_summary.md`

## 4. Search objective

Current weighted score:

```text
score = mean_acc
      - lambda_param * param_m
      - lambda_tx * tx_norm
      - lambda_robust * robust_gap
      - lambda_rate * max(0, mean_cr - target_rate)
```

Where:

- `mean_acc`: maximize
- `param_m`: minimize
- `tx_cost`/`tx_norm`: minimize
- `robust_gap`: minimize
- `mean_cr` beyond target rate: penalized

## 5. Practical tips

- All scripts expect ImageFolder train/valid directories.
- If you run on CPU, reduce:
  - `--num_samples`
  - `--search_epochs`
  - `--max_train_batches`
  - `--max_eval_batches`
