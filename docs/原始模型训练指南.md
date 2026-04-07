# Original Model Training Guide

This guide is for the baseline model script:

- `scripts/train/ASE-JSCCtrain.py`

## 1. Recommended command (UCMerced_LandUse)

```bash
python scripts/train/ASE-JSCCtrain.py \
  --task train \
  --cr 0.8 \
  --num_epochs 120 \
  --channel_type Combined_channel \
  --dataset_name UCMerced_LandUse \
  --train_dir data/UCMerced_LandUse/UCMerced_LandUse-train \
  --valid_dir data/UCMerced_LandUse/UCMerced_LandUse-valid \
  --batch_size 64 \
  --num_workers 6 \
  --output_dir runs/original_train
```

Optional:

- `--run_name <name>` to fix a custom run folder name.

## 2. Output structure

Each run is saved to:

- `runs/original_train/<run_name_or_auto_name>/`

Key files:

- `checkpoint/best_*.pth`: best checkpoint by validation loss
- `checkpoint/classifier_*.pth`: final checkpoint
- `tensorboard/events.out.*`: TensorBoard events
- `tensorboard/*.txt`: plain-text training log
- `run_summary.md`: markdown summary for alignment with NAS retrain reports

## 3. Alignment note

For fair comparison with NAS retrain runs:

- Use the same train/valid splits
- Use the same image size / batch size as much as possible
- Use `scripts/benchmark/compare_models.py` for final unified evaluation
