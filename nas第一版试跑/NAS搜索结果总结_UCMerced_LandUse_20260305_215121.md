# NAS 搜索结果总结（UCMerced_LandUse_20260305_215121）

## 1. 运行概览

- 运行 ID：`UCMerced_LandUse_20260305_215121`
- 数据集：`UCMerced_LandUse`（21 类）
- 训练/验证规模：`1050 / 210`
- 搜索空间总规模：`2880`（本次随机采样 `36` 个候选）
- 代理训练轮数：`3`（`search_epochs=3`）
- 搜索耗时：`1022.51s`（约 `17.04` 分钟）
- 设备：`cuda:0`（`NVIDIA GeForce RTX 5070 Ti`）
- 随机种子：`42`

## 2. 多目标打分配置（本次实参）

本次运行使用的权重如下（来自 `run_config.json`）：

- `lambda_param = 0.01`
- `lambda_tx = 0.01`
- `lambda_robust = 0.2`
- `lambda_rate = 0.2`
- `target_rate = 0.7`
- 动态 CR：`min_dynamic_cr=0.3`，`max_dynamic_cr=1.0`，`rate_blend_alpha=0.7`

## 3. 最优架构（Rank-1）

- `arch_tag`: `s4_r32_cr0.4_cb24_d2_k1_skip`
- `score`: `0.8350`
- `mean_acc`: `0.9552`
- `worst_acc`: `0.9476`
- `robust_gap`: `0.0190`
- `mean_cr`: `0.5566`
- `std_cr`: `0.0021`
- `param_m`: `11.548M`
- `tx_cost`: `855.0`

对应结构参数：

```json
{
  "insertion_stage": 4,
  "se_ratio": 32,
  "cr": 0.4,
  "bottleneck_channels": 24,
  "ae_depth": 2,
  "kernel_size": 1,
  "use_skip": true
}
```

## 4. Top-8 结果

| rank | arch_tag | score | mean_acc | worst_acc | mean_cr | param_m | tx_cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | s4_r32_cr0.4_cb24_d2_k1_skip | 0.8350 | 0.9552 | 0.9476 | 0.5566 | 11.548 | 855.0 |
| 2 | s4_r32_cr1.0_cb32_d2_k1_skip | 0.8121 | 0.9407 | 0.9333 | 0.7477 | 11.557 | 1531.4 |
| 3 | s4_r4_cr0.6_cb48_d2_k3_skip | 0.7952 | 0.9423 | 0.9381 | 0.6383 | 14.317 | 1960.9 |
| 4 | s4_r4_cr0.4_cb48_d2_k3_skip | 0.7712 | 0.9208 | 0.9095 | 0.5475 | 14.317 | 1681.9 |
| 5 | s4_r4_cr0.6_cb16_d3_k3_skip | 0.7665 | 0.9278 | 0.9238 | 0.6549 | 15.875 | 670.6 |
| 6 | s3_r8_cr0.8_cb48_d4_k5_skip | 0.7423 | 0.9234 | 0.9048 | 0.6890 | 16.407 | 8466.3 |
| 7 | s3_r8_cr0.6_cb24_d3_k3_skip | 0.7353 | 0.8827 | 0.8381 | 0.6437 | 12.439 | 3954.8 |
| 8 | s3_r32_cr1.0_cb24_d2_k1_skip | 0.7338 | 0.8798 | 0.8286 | 0.7654 | 11.298 | 4702.6 |

## 5. 全体分布与 Pareto 结果

### 5.1 全体 36 个候选分布

- `mean_acc` 范围：`0.0526 ~ 0.9552`，平均 `0.5477`
- `score` 范围：`-0.0920 ~ 0.8350`，平均 `0.3747`
- `score > 0` 的候选：`29/36`（`80.6%`）
- Top-5 平均：
  - `mean_acc = 0.9373`
  - `worst_acc = 0.9305`
  - `mean_cr = 0.6290`
  - `score = 0.7960`

### 5.2 Pareto Front

- Pareto 前沿规模：`14/36`
- 前沿头部（按 score）基本由高精度 + 低传输代价候选构成，最优点即 Rank-1。
- 前沿也保留了低 `tx_cost` 但精度较低的候选，说明搜索结果能覆盖不同“精度-代价”偏好。

## 6. 结果解读（是否合理）

- Top-8 中 `use_skip` 全为 `True`，说明在本次设置下残差旁路对稳定性/精度有明显帮助。
- Top-8 中 `insertion_stage=4` 占 `5/8`，后段插入在本数据集上更有优势。
- Rank-1 的 `mean_cr=0.5566 < target_rate=0.7`，无超标惩罚；同时 `tx_cost` 也处于较低水平。
- Rank-1 的 `worst_acc=0.9476` 且 `robust_gap=0.0190`，表明跨信道/SNR 条件鲁棒性较好。

结论：本次搜索结果整体合理，最佳候选在精度、鲁棒性、传输代价三者上达到了较均衡解。

## 7. 建议的全量重训候选

建议优先重训 Top-3，并保留 Top-5 做补充对比：

- Top-1：`s4_r32_cr0.4_cb24_d2_k1_skip`（当前综合最优）
- Top-2：`s4_r32_cr1.0_cb32_d2_k1_skip`（更高 CR，检验高码率上限）
- Top-3：`s4_r4_cr0.6_cb48_d2_k3_skip`（高精度且鲁棒 gap 最小之一）
- Top-4/5：作为结构对照，验证 `cr` 与 `bottleneck_channels` 组合影响

## 8. 结果文件索引

本次运行目录：`runs/nas_search/UCMerced_LandUse_20260305_215121`

- `summary.json`：完整汇总（包含 best、top_k、每条件 acc_map 等）
- `run_config.json`：本次运行参数与环境
- `ranked_results.json/csv`：36 个候选完整排序
- `topk_summary.md`：Top-K 简表
- `pareto_front.json` / `pareto_topk.json` / `pareto_summary.md`：Pareto 分析
- `figures/`：四张图（散点、Pareto、Top-K 对比、最优架构 SNR 曲线）
- `tensorboard/`：搜索阶段 TensorBoard 日志
