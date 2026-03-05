from __future__ import annotations

"""
NAS 搜索空间定义。

这个文件只做一件事：定义“可搜索的架构超参数”以及它们的采样方式。
核心对象：
1. ArchitectureConfig：一个具体架构（单个候选）的参数集合。
2. SearchSpace：所有候选的参数取值范围，并提供随机采样/网格枚举。

当前搜索空间（default）现状说明：
- 维度数：7
- 各维取值数：
  insertion_stage(2) * se_ratio(4) * cr(4) * bottleneck_channels(5)
  * ae_depth(3) * kernel_size(3) * use_skip(2)
- 全组合规模：2 * 4 * 4 * 5 * 3 * 3 * 2 = 2880
- 复杂度判断：属于“轻量级宏观结构搜索空间”，适合快速验证思路；
  若要进一步提升 NAS 上限，可扩展到更细粒度算子级搜索。
"""

from dataclasses import dataclass, asdict
from itertools import product
from random import Random
from typing import Dict, Iterable, List


@dataclass
class ArchitectureConfig:
    # JSCC 插入位置：
    # 3 -> 接在 ResNet layer3 后（特征通道 256）
    # 4 -> 接在 ResNet layer4 后（特征通道 512）
    insertion_stage: int
    # SE Block 的缩减比（越大表示中间隐藏层越小）
    se_ratio: int
    # 注意力通道保留比例基准值。
    # 若启用动态压缩率，则该值是动态 CR 的“先验基准”。
    cr: float
    # 自编码器瓶颈通道数（越小压缩越强）
    bottleneck_channels: int
    # 编码器/解码器深度（卷积层组数）
    ae_depth: int
    # 自编码器卷积核大小（通常 1/3/5）
    kernel_size: int
    # 是否为 JSCC 块添加输入输出残差旁路
    use_skip: bool

    def to_dict(self) -> Dict[str, object]:
        """将 dataclass 转成可序列化字典，便于写入 JSON。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ArchitectureConfig":
        """从 JSON 字典还原为 ArchitectureConfig。"""
        return cls(
            insertion_stage=int(payload["insertion_stage"]),
            se_ratio=int(payload["se_ratio"]),
            cr=float(payload["cr"]),
            bottleneck_channels=int(payload["bottleneck_channels"]),
            ae_depth=int(payload["ae_depth"]),
            kernel_size=int(payload["kernel_size"]),
            use_skip=bool(payload["use_skip"]),
        )

    @property
    def tag(self) -> str:
        """
        生成用于日志/文件名的短标签。

        示例：
        s4_r16_cr0.8_cb32_d3_k3_skip
        """
        skip_tag = "skip" if self.use_skip else "noskip"
        return (
            f"s{self.insertion_stage}_r{self.se_ratio}_cr{self.cr}"
            f"_cb{self.bottleneck_channels}_d{self.ae_depth}"
            f"_k{self.kernel_size}_{skip_tag}"
        )


@dataclass
class SearchSpace:
    """定义 NAS 搜索空间各维度可选值。"""

    insertion_stage: List[int]
    se_ratio: List[int]
    cr: List[float]
    bottleneck_channels: List[int]
    ae_depth: List[int]
    kernel_size: List[int]
    use_skip: List[bool]

    @classmethod
    def default(cls) -> "SearchSpace":
        """项目默认搜索空间（经验范围）。"""
        return cls(
            insertion_stage=[3, 4],
            se_ratio=[4, 8, 16, 32],
            cr=[0.4, 0.6, 0.8, 1.0],
            bottleneck_channels=[16, 24, 32, 48, 64],
            ae_depth=[2, 3, 4],
            kernel_size=[1, 3, 5],
            use_skip=[False, True],
        )

    def sample(self, rng: Random) -> ArchitectureConfig:
        """随机采样一个候选架构（离散均匀采样）。"""
        return ArchitectureConfig(
            insertion_stage=rng.choice(self.insertion_stage),
            se_ratio=rng.choice(self.se_ratio),
            cr=rng.choice(self.cr),
            bottleneck_channels=rng.choice(self.bottleneck_channels),
            ae_depth=rng.choice(self.ae_depth),
            kernel_size=rng.choice(self.kernel_size),
            use_skip=rng.choice(self.use_skip),
        )

    def grid(self) -> Iterable[ArchitectureConfig]:
        """
        枚举完整笛卡尔积搜索空间。

        注意：组合数可能非常大，实际使用时通常只做小规模验证或调试。
        """
        for values in product(
            self.insertion_stage,
            self.se_ratio,
            self.cr,
            self.bottleneck_channels,
            self.ae_depth,
            self.kernel_size,
            self.use_skip,
        ):
            yield ArchitectureConfig(
                insertion_stage=values[0],
                se_ratio=values[1],
                cr=values[2],
                bottleneck_channels=values[3],
                ae_depth=values[4],
                kernel_size=values[5],
                use_skip=values[6],
            )
