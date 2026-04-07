# Channel-Aware Multi-Objective NAS 方案设计（ASE-JSCC）

## 1. 目标与定位

本方案面向当前基线模型（ResNet18 + SE + Autoencoder + 信道仿真），提出一个可用于论文主方法的创新方向：

- 核心目标：在**不同信道条件（AWGN / Fading / Combined）与随机 SNR** 下，自动搜索更优网络结构。
- 优化目标不是单一精度，而是多目标联合：
  - 分类性能（平均与低信噪比鲁棒性）
  - 计算复杂度（FLOPs / Params）
  - 传输代价（语义特征压缩后的码率或等效维度）

可命名为：

- `CA-MO-NAS`（Channel-Aware Multi-Objective NAS）


## 2. 与基线的关系（创新边界）

基线固定结构：

`ResNet18特征 -> SE通道筛选 -> Autoencoder压缩/解压 -> 信道扰动 -> 分类头`

本方案创新点：

1. 将“固定结构”升级为“可搜索结构”；
2. 将“单一准确率最优”升级为“多目标 Pareto 最优”；
3. 将“单信道/单SNR设定”升级为“信道分布期望鲁棒最优”。


## 3. 方法总览

### 3.1 形式化定义

给定架构 \(a \in \mathcal{A}\)、参数 \(\theta\)、信道条件 \(c\)、SNR \(s\)：

\[
\min_{a \in \mathcal{A}} \quad \mathcal{J}(a)
=
\Big[
\mathbb{E}_{c,s}\,\mathcal{L}_{cls}(a,\theta;c,s),\;
\Omega_{comp}(a),\;
\Omega_{tx}(a)
\Big]
\]

其中：

- \(\mathcal{L}_{cls}\)：分类损失（可转为准确率最大化）
- \(\Omega_{comp}\)：计算开销（FLOPs/Params/Latency）
- \(\Omega_{tx}\)：传输代价（压缩后特征长度、激活通道比例、等效 bpp）

最终输出不止一个模型，而是一组 Pareto 前沿候选；论文主结果选其中“性能-开销平衡最优点”。


## 4. 搜索空间设计（贴合你现有代码）

## 4.1 Backbone 插入位置（离散搜索）

- `stage_pos ∈ {after_layer3, after_layer4}`
- 含义：SE+JSCC模块插在 ResNet 的 layer3 或 layer4 后。

> 基线是 `after_layer4`，加入 `after_layer3` 可形成显著结构对比。

## 4.2 SE / 通道选择模块搜索

- SE reduction ratio：`r ∈ {4, 8, 16, 32}`
- 通道保留率（替代固定 cr=0.8）：`cr ∈ {0.4, 0.6, 0.8, 1.0}`
- Mask 方式：`{hard_topk, gumbel_topk, sigmoid_threshold}`

建议论文主线：

- 搜索阶段用 `gumbel_topk`（可微）
- 重训练部署阶段转为 `hard_topk`

## 4.3 Autoencoder（JSCC模块）结构搜索

- 编码器层数：`L_e ∈ {2, 3, 4}`
- 解码器层数：`L_d ∈ {2, 3, 4}`
- bottleneck通道：`C_b ∈ {16, 24, 32, 48, 64}`
- 卷积核大小：`k ∈ {1, 3, 5}`
- 是否使用残差/跳连：`skip ∈ {on, off}`

等效传输维度：

\[
D_{tx} = C_b \times H' \times W' \times \rho
\]

其中 \(\rho\) 是通道激活比例（由 cr 或门控决定）。

## 4.4 信道训练策略作为“环境搜索条件”

- 训练时按概率采样 `AWGN/Fading/Combined`
- SNR 采用分布采样（例如 Uniform[0, 28]）
- 可引入“难例偏置”采样：低SNR占比更高（提高鲁棒性）


## 5. 多目标函数与优化策略

## 5.1 建议目标函数

可用加权形式（实现简单）：

\[
\mathcal{L}_{total}
=
\mathcal{L}_{cls}
 \lambda_1 \cdot \widetilde{\Omega}_{comp}
 \lambda_2 \cdot \widetilde{\Omega}_{tx}
 \lambda_3 \cdot \mathcal{L}_{robust}
\]

其中：

- \(\mathcal{L}_{robust}\)：跨 SNR 方差或 worst-k SNR 损失项；
- 带 `~` 的项表示归一化到同量纲。

也可用 Pareto 进化（论文更“NAS正统”）：

- 不将目标硬加权，直接保留非支配解；
- 最后报告 Pareto 曲线。

## 5.2 鲁棒性项推荐

- `mean loss over channel/SNR` + `beta * std(loss over channel/SNR)`
- 或 `CVaR`：重点优化最差 20% 信道样本


## 6. 搜索算法（推荐两阶段，计算更可控）

## 6.1 Stage-A：可微超网搜索（粗搜索）

思路：

- 构建 supernet，把候选操作并联；
- 用 architecture parameters（\(\alpha\)）控制各候选权重；
- 训练时交替更新：
  - 权重参数 \(w\)（训练集）
  - 架构参数 \(\alpha\)（验证集）

优点：比纯进化快，适合你现阶段先拿可发表结果。

## 6.2 Stage-B：Pareto 细搜索（精筛）

做法：

1. 从 Stage-A 导出 Top-K 架构（按精度或联合分数）；
2. 对 Top-K 进行小规模进化/网格细化；
3. 完整重训练，得到最终 Pareto 前沿。

## 6.3 伪代码（简化）

```text
Initialize supernet weights w and architecture params alpha
for epoch in search_epochs:
  sample channel type c and SNR s
  update w on train set with fixed alpha
  update alpha on valid set with fixed w
  compute multi-objective score (acc, flops, tx_cost, robustness)
Export top-K candidate architectures from alpha
Retrain each candidate from scratch under channel-aware training
Select Pareto-optimal models and report
```


## 7. 实验设计（论文可直接用）

## 7.1 数据集与划分

- SoyaHealthVision（当前主线）
- UCMerced / AID（泛化验证）
- 使用你项目已有分割脚本生成 train/valid/test

## 7.2 信道设置

- Channel type：`AWGN`, `Fading`, `Combined`
- 训练 SNR：`Uniform[0, 28]`
- 测试 SNR：离散点 `0, 4, 8, 12, 16, 20, 24, 28`

## 7.3 指标

- `Top-1 Acc`（平均）
- `Worst-SNR Acc`（最低 SNR 点）
- `Robust Gap`（高低 SNR 精度差）
- `FLOPs`, `Params`
- `Tx-Cost`（等效传输维度或估算码率）

## 7.4 对比组

1. 固定基线（当前 ASE-JSCC）
2. 单目标 NAS（只优化精度）
3. 多目标 NAS（不加信道感知）
4. **CA-MO-NAS（完整方法）**

## 7.5 消融实验

- 去掉鲁棒性项 \(\mathcal{L}_{robust}\)
- 固定 channel（不随机采样）
- 固定 cr（不搜索）
- 固定 bottleneck（不搜索 \(C_b\)）
- 去掉 Pareto，只保留加权和


## 8. 代码落地方案（建议目录）

```text
scripts/
  nas/
    search_channel_aware.py        # supernet搜索入口
    retrain_candidates.py          # 重训练Top-K
    evaluate_pareto.py             # Pareto评估与导出
  train/
    ASE-JSCCtrain.py               # 保留为baseline训练入口

configs/
  nas/
    search.yaml
    retrain.yaml
```

关键实现模块（建议抽离）：

- `SearchableSEBlock`
- `SearchableAutoencoder`
- `MultiObjectiveEvaluator`（acc/flops/tx/robust）
- `ChannelSampler`


## 9. 训练与计算预算建议

- 粗搜索：20~40 epochs（proxy size，较小输入或子集）
- 精筛重训练：Top-5 到 Top-10，每个 80~150 epochs
- 每个候选至少 3 个随机种子报告均值和标准差

如果算力紧张：

- 先在 Soya 跑完整方法
- UCM/AID 只做迁移验证（缩减候选数量）


## 10. 论文写作结构建议

### 方法章节

1. 问题定义（信道不确定 + 资源约束）
2. 搜索空间（SE + JSCC + 插入位置）
3. 多目标优化（精度/复杂度/传输）
4. 信道感知训练机制（channel & SNR sampling）
5. 搜索与重训练流程

### 实验章节

1. 主结果（精度-开销 Pareto）
2. 跨信道与跨SNR鲁棒性
3. 消融与可视化（Pareto曲线、SNR-Acc曲线）


## 11. 风险与规避

- 风险1：搜索不稳定  
  - 规避：先固定部分空间（如先不搜 stage_pos），逐步放开
- 风险2：计算量过大  
  - 规避：两阶段搜索 + proxy训练 + Top-K重训
- 风险3：多目标权重敏感  
  - 规避：同时报告 weighted 和 Pareto 两套结果
- 风险4：复现性不足  
  - 规避：固定随机种子、统一数据划分、记录 channel/SNR 采样配置


## 12. 最小可执行版本（MVP）

第一版建议只做这三件事：

1. 搜索 `cr` + `bottleneck C_b`（其余先固定）  
2. 目标函数：`valid loss + λ1*FLOPs + λ2*TxCost`  
3. 信道训练：`Combined + Uniform[0,28] SNR`

这样可以最快产出“有创新且可解释”的第一批结果，再扩展到完整 CA-MO-NAS。


## 13. 一句话总结

该方案的核心价值是：把你现有 ASE-JSCC 从“固定手工结构”升级为“面向信道不确定性的自动结构优化系统”，并以多目标 Pareto 结果体现工程与学术价值。

## 14. 围绕“自动筛选特征传输”的创新增强（新增）

你提到的第二个优势是“模型能根据下游任务自动筛选特征进行传输”。
在现有方案上，可以将这一点升级为更强的论文创新主线：

### 14.1 从固定筛选升级为“任务-信道联合动态筛选”

当前基线的 `cr` 往往是固定超参数（例如 0.8）。
建议改为样本级动态码率：

- 设计 `RateController` 预测每个样本的 `cr_i`；
- 输入包含：语义特征统计 + 当前信道类型 + 当前 SNR；
- 输出 `cr_i`（范围可约束在 `[cr_min, cr_max]`）。

这样模型会自动学习：

- 好信道下传更多细节；
- 差信道下传更稳健的核心语义；
- 简单样本低码率、困难样本高码率。

### 14.2 将信道状态显式注入特征筛选器

在 SE / gating 模块中加入信道条件嵌入：

- `channel embedding`：AWGN / Fading / Combined 的可学习向量；
- `snr embedding`：将标量 SNR 映射到特征向量；
- 与全局池化语义向量拼接后，共同生成通道门控权重。

可命名为：

- `Channel-Conditioned Feature Selector (CCFS)`

### 14.3 从硬阈值到可微稀疏门控

为了提升训练稳定性与可搜索性，建议：

- 搜索阶段：`Gumbel-TopK` 或 `Hard-Concrete`；
- 部署阶段：转为硬 `TopK` 掩码。

收益：

- 更适合与 NAS 联合优化；
- 梯度可传，训练更稳定；
- 可直接控制稀疏度与传输代价。

### 14.4 目标函数中显式约束“传输预算”

在损失中加入可微传输代价项：

\[
\mathcal{L} = \mathcal{L}_{cls} + \lambda_{tx}\,\mathcal{L}_{tx} + \lambda_{rob}\,\mathcal{L}_{robust}
\]

其中：

- \(\mathcal{L}_{tx}\)：与激活通道数、瓶颈维度、平均 `cr_i` 相关；
- \(\mathcal{L}_{robust}\)：跨信道/跨SNR鲁棒项（方差、CVaR、worst-k）。

这能把“自动筛选”从结构描述提升为“可优化目标”。

### 14.5 对应可验证实验（建议作为论文关键消融）

新增对比组：

1. 固定 `cr` + 无信道条件筛选（基线）
2. 动态 `cr` + 无信道条件
3. 固定 `cr` + 信道条件筛选
4. 动态 `cr` + 信道条件筛选（完整方法）

建议报告：

- 平均精度、Worst-SNR 精度；
- 平均传输代价（或等效码率）；
- 精度-传输代价 Pareto 曲线；
- `cr_i` 在不同 SNR 下的分布可视化。

### 14.6 代码落地点（与现有实现对应）

可在当前 NAS 代码基础上增加：

- `scripts/nas/searchable_model.py`
  - 新增 `RateController`
  - 新增 `ChannelConditionedSelector`
  - 将 `cr` 从固定值改为前向预测 `cr_i`
- `scripts/nas/search_channel_aware.py`
  - 评分函数加入 `avg_cr` 或 `tx_budget_violation`
- `scripts/nas/retrain_candidate.py`
  - 训练日志记录 `mean_cr`、`std_cr`、按SNR分桶的 `cr` 统计

### 14.7 一句话总结这个创新点

将“按任务自动筛选特征传输”升级为“按任务与信道状态联合决定传输内容和传输码率”，
可同时提升鲁棒性、通信效率和方法新颖性。
