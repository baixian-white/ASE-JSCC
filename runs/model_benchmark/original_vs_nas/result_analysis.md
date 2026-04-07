# Original vs NAS Retrain 结果分析

## 1. 评测设置

- 评测时间：2026-03-14 09:45（见 `benchmark_results.json`）
- 数据集：`data/UCMerced_LandUse/UCMerced_LandUse-valid`（210 张图像）
- 设备：`cuda:0`（NVIDIA GeForce RTX 5070 Ti）
- 模型 A（baseline）：`original`（legacy）
- 模型 B（candidate）：`nas_retrain`（nas）
- 统一评测协议：
  - `mc_runs = 10`
  - 信道类型：`AWGN`、`Fading`、`Combined_channel`
  - NAS 模型 SNR 采样范围：`0~28 dB`
  - 使用同一数据预处理与统一脚本进行对比

## 2. 核心结论

在本次统一协议下，`nas_retrain` 相比 `original` 在所有信道上均取得稳定提升，且提升幅度明显高于统计波动范围。

- 总体均值精度（`overall_mean`）从 `0.9187` 提升到 `0.9817`
- 绝对提升：`+0.0630`（约 `+6.30` 个百分点）
- 95% 置信区间：`±0.0009`（远小于提升幅度）

按 210 张验证图估算，`+0.0630` 约等价于平均多分对 `13` 张图像。

## 3. 结果明细

| 评测维度 | original (A) | nas_retrain (B) | Delta (B-A) | Delta 95% CI |
| --- | ---: | ---: | ---: | ---: |
| AWGN | 0.9190 | 0.9762 | +0.0571 | ±0.0000 |
| Fading | 0.9190 | 0.9833 | +0.0643 | ±0.0016 |
| Combined_channel | 0.9181 | 0.9857 | +0.0676 | ±0.0019 |
| overall_mean | 0.9187 | 0.9817 | +0.0630 | ±0.0009 |

对应到 210 张图像的大致增益：

- AWGN：约多分对 12 张
- Fading：约多分对 13-14 张
- Combined_channel：约多分对 14 张

## 4. 稳定性与鲁棒性解读

- `nas_retrain` 的总体标准差更小（`std=0.0008`），说明多次随机评测下表现稳定。
- `worst_channel_acc` 从约 `0.9181` 提升到约 `0.9762`，说明最差信道条件下也显著改善。
- 在三类信道上均为正增益，说明提升不是某单一信道“拉高”的结果。

## 5. 结果边界与注意事项

- 本结论基于 `valid` 集，不是最终 `test` 集结论。
- 当前指标是统一 benchmark 的 Monte Carlo 随机 SNR 协议，不等同于 NAS 训练日志中的网格 `mean_acc`（口径不同，但趋势一致）。
- 95% CI 来自 10 次 MC 采样统计，已足以支持“提升显著且稳定”的工程结论。

## 6. 建议的下一步

1. 在 `UCMerced_LandUse-test` 上复现同协议对比，作为最终报告主结果。
2. 增加分类别分析（混淆矩阵/每类准确率），确认提升是否均匀分布于各类别。
3. 固定 checkpoint 后再做 3~5 组不同 seed 的重复实验，补充跨随机种子的稳定性证据。

