# Original Model Training Summary

## Run
- task: `train`
- run_dir: `/mnt/e/CODE/ASE-JSCC/runs/original_train/ucmerced_original_120ep`
- start_time: `2026-03-06 09:57:49`
- end_time: `2026-03-06 12:17:16`

## Data & Config
- dataset_name: `UCMerced_LandUse`
- train_dir: `/mnt/e/CODE/ASE-JSCC/data/UCMerced_LandUse/UCMerced_LandUse-train`
- valid_dir: `/mnt/e/CODE/ASE-JSCC/data/UCMerced_LandUse/UCMerced_LandUse-valid`
- num_classes: `21`
- channel_type: `Combined_channel`
- cr: `0.8`
- num_epochs: `120`
- batch_size: `64`
- num_workers: `6`
- device: `cuda:0`

## Outputs
- best_checkpoint: `/mnt/e/CODE/ASE-JSCC/runs/original_train/ucmerced_original_120ep/checkpoint/best_classifier_attention_auto_UCMerced_LandUse_Combined_channel_ResNet18_120epoch_0.8.pth`
- final_checkpoint: `/mnt/e/CODE/ASE-JSCC/runs/original_train/ucmerced_original_120ep/checkpoint/classifier_attention_auto_UCMerced_LandUse_Combined_channel_ResNet18_120epoch_0.8.pth`
- tensorboard_dir: `/mnt/e/CODE/ASE-JSCC/runs/original_train/ucmerced_original_120ep/tensorboard`
- txt_log: `/mnt/e/CODE/ASE-JSCC/runs/original_train/ucmerced_original_120ep/tensorboard/classifier_attention_auto_UCMerced_LandUse_Combined_channel_ResNet18_120epoch_0.8.txt`

## Metrics
- final_valid_accuracy: `0.928571`
- final_valid_loss: `0.844684`
- best_valid_loss: `0.776387`
- best_epoch: `78`
