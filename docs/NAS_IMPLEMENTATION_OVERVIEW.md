# ASE-JSCC 现有 NAS 实现说明

本文档说明当前仓库里已经落地的 `Channel-Aware Multi-Objective NAS` 代码实现。

## 1. 当前版本已经具备的能力

- 信道感知搜索：在 `AWGN / Fading / Combined_channel` + 多个 SNR 条件下评估结构。
- 多目标优化：联合考虑精度、参数量、传输代价、鲁棒性。
- 动态特征筛选（已实现）：
  - `ChannelConditionedSelector`：门控会显式接收 `channel_type` 和 `snr` 条件。
  - `RateController`：预测样本级动态压缩率 `cr_i`，不再只依赖固定 `cr`。
- 可复现流程：搜索 -> 候选重训练 -> Pareto 分析。

## 2. 代码结构

- `scripts/nas/search_space.py`：架构搜索空间定义。
- `scripts/nas/searchable_model.py`：可搜索模型实现（含动态筛选与动态码率）。
- `scripts/nas/nas_utils.py`：工具函数（数据加载、路径、指标、序列化）。
- `scripts/nas/search_channel_aware.py`：搜索入口。
- `scripts/nas/retrain_candidate.py`：候选重训练入口。
- `scripts/nas/evaluate_pareto.py`：Pareto 前沿筛选。

## 3. 搜索空间

当前架构变量：

- `insertion_stage`: `{3, 4}`
- `se_ratio`: `{4, 8, 16, 32}`
- `cr`: `{0.4, 0.6, 0.8, 1.0}`（作为基础码率 prior）
- `bottleneck_channels`: `{16, 24, 32, 48, 64}`
- `ae_depth`: `{2, 3, 4}`
- `kernel_size`: `{1, 3, 5}`
- `use_skip`: `{False, True}`

## 4. 关键机制

## 4.1 信道条件门控（已实现）

`searchable_model.py` 中新增：

- `ChannelConditionEncoder`
- `ChannelConditionedSelector`

门控权重由“语义向量 + 信道类型嵌入 + SNR 嵌入”共同决定。

## 4.2 动态码率控制（已实现）

`RateController` 预测样本级 `cr_i`，并与架构基础 `cr` 融合：

- `min_dynamic_cr <= cr_i <= max_dynamic_cr`
- 由 `rate_blend_alpha` 控制“动态值 vs 基础值”的权重。

## 4.3 日志统计（已实现）

模型每次前向会记录：

- `mean_cr`
- `std_cr`
- `min_cr`
- `max_cr`

搜索和重训练脚本都会把这些统计写入结果。

## 5. 搜索评分函数

当前搜索脚本使用加权分数：

```text
score = mean_acc
      - lambda_param * param_m
      - lambda_tx * tx_norm
      - lambda_robust * robust_gap
      - lambda_rate * max(0, mean_cr - target_rate)
```

目标方向：

- 最大化：`mean_acc`
- 最小化：`param_m`, `tx_cost`, `robust_gap`
- 约束：`mean_cr` 不超过 `target_rate`（超出会惩罚）

## 6. 三个入口脚本职责

## 6.1 `search_channel_aware.py`

- 采样架构并进行短训练。
- 多信道多 SNR 网格验证。
- 输出 `search_results.jsonl`、`best_arch.json`、`topk_arches.json`、`summary.json`。

## 6.2 `retrain_candidate.py`

- 读取 `best_arch.json`（或手动架构 json）。
- 全量训练并保存最佳模型。
- 输出中包含 `train_mean_cr/std_cr`、`valid_mean_cr/std_cr`。

## 6.3 `evaluate_pareto.py`

- 从搜索结果中计算非支配解。
- 输出 `pareto_front.json`、`pareto_topk.json`、`pareto_summary.md`。

## 7. 当前边界

- 搜索器目前是“随机采样 + 短训练”的 MVP，不是 DARTS/进化超网。
- 复杂度指标以参数量和等效 `tx_cost` 为主，尚未接入真实延迟（latency）测量。
- 端到端实验需要可用的 `torch` 运行环境。

## 8. 论文可用一句话

我们实现了一个信道感知多目标 NAS 系统，并进一步引入“信道条件门控 + 动态码率控制”，实现了任务相关语义特征的自适应传输与鲁棒优化。
