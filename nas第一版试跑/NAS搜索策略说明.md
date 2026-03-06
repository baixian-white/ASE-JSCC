# NAS 搜索策略说明（第一版试跑）

本文档说明当前第一版 NAS 实际采用的搜索策略，对应脚本：

- `scripts/nas/search_channel_aware.py`

---

## 1. 策略总览

第一版采用的是**离散空间随机采样 + 代理训练 + 多目标打分排序**，不是 DARTS 或进化算法。

整体流程：

1. 从离散搜索空间随机采样 `num_samples` 个候选架构。
2. 每个候选做少量 `quick_train`（代理训练，不追求完全收敛）。
3. 在 `channel_type × SNR` 验证网格上评估准确率与鲁棒性。
4. 结合精度、参数量、传输代价、鲁棒性、码率约束做单标量打分。
5. 按分数排序，导出 `best_arch` 与 `top-k` 结果。

---

## 2. 候选架构如何产生

搜索空间由 `SearchSpace.default()` 定义，当前规模为 2880 组合：

- `insertion_stage`: `[3, 4]`
- `se_ratio`: `[4, 8, 16, 32]`
- `cr`: `[0.4, 0.6, 0.8, 1.0]`
- `bottleneck_channels`: `[16, 24, 32, 48, 64]`
- `ae_depth`: `[2, 3, 4]`
- `kernel_size`: `[1, 3, 5]`
- `use_skip`: `[False, True]`

第一版搜索中每一步使用 `search_space.sample(rng)` 随机采样一个候选，不做全遍历。

---

## 3. 每个候选如何评估（search_once）

每个候选会执行一次 `search_once(...)`，包含四部分：

### 3.1 构建模型

- 按当前候选架构实例化 `ChannelAwareClassifier`。
- 可通过开关做消融：
  - `--disable_dynamic_rate`
  - `--disable_channel_condition`
  - `--disable_pretrained_backbone`

### 3.2 代理训练（quick_train）

核心逻辑：

- 训练轮数：`search_epochs`
- 优化器：`Adam(lr=search_lr, weight_decay=1e-4)`
- 损失：`CrossEntropyLoss(label_smoothing=0.1)`
- 每个 batch 随机采样 `(channel_type, snr)`（默认 3 种信道近似均匀，SNR 在 `[0, 28]`）
- 可用 `--max_train_batches` 截断每轮训练 batch 数加速试跑

这一阶段目标是**低成本粗筛**，不是最终模型训练。

### 3.3 网格评估（evaluate_model）

在验证集上执行：

- 遍历 `eval_channel_types × eval_snr_list`
- 统计每个条件点准确率 `acc_map`
- 汇总 `mean_acc / worst_acc / robust_gap / mean_cr / std_cr`
- 可用 `--max_eval_batches` 截断验证 batch 数加速

mean_acc：在所有 channel_type × SNR 条件点上的平均准确率（越高越好）
worst_acc：所有条件点里最低的准确率（最差场景表现，越高越好）
robust_gap：性能波动幅度，计算是 max(acc) - min(acc)（越小越稳）
mean_cr：评估过程中样本级动态压缩率 CR 的平均值（反映平均压缩/传输强度）
std_cr：CR 的标准差（反映动态码率波动大小，越大说明自适应幅度越强）

### 3.4 多目标打分

当前打分函数（分数越高越好）：

```text
score = mean_acc
        - lambda_param * param_m
        - lambda_tx    * (tx_cost / 10000)
        - lambda_robust* robust_gap
        - lambda_rate  * max(0, mean_cr - target_rate)
```

其中：

- `param_m`：参数量（百万）
- `tx_cost`：估计传输代价
- `robust_gap`：跨信道/SNR 性能波动（越小越稳）
- `mean_cr` 超过 `target_rate` 时触发码率惩罚

---

## 4. 搜索结束后的选择策略

全部候选评估完成后：

1. 按 `score` 降序排序。
2. `ranked_results.json/csv` 保存完整排名。
3. 取前 `top_k` 保存到 `topk_arches.json`，并生成 `topk_summary.md`。
4. `best_arch.json` 保存第一名架构。

---

## 5. 结果与日志产物

每次搜索输出目录：

- `runs/nas_search/<dataset>_<timestamp>/`

关键文件：

- `search_results.jsonl`：每个候选完整结果（逐条落盘）
- `search_progress.jsonl`：搜索进度日志
- `best_arch.json`：最佳架构
- `ranked_results.json` / `ranked_results.csv`：完整排序
- `topk_arches.json` / `topk_summary.md`：Top-K 汇总
- `summary.json`：整次搜索摘要
- `run_config.json`：运行配置与搜索空间快照
- `tensorboard/`：可视化日志（默认开启）

---

## 6. 第一版策略的定位

第一版的策略重点是：

- 先验证“信道感知 + 动态码率 + 多目标约束”是否有效；
- 在可控算力预算下快速迭代；
- 输出可复现、可解释、可排序的候选架构。

它的优势是工程落地快，代价是搜索效率和最优性上限暂不如进化搜索/贝叶斯优化等更高级策略。
