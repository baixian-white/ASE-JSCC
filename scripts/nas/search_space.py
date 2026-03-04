from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from random import Random
from typing import Dict, Iterable, List


@dataclass
class ArchitectureConfig:
    # Where to insert the JSCC block: after layer3 (256 channels) or layer4 (512 channels).
    insertion_stage: int
    # Reduction ratio of SE block.
    se_ratio: int
    # Base channel keep ratio in attention gating.
    # If dynamic rate is enabled, this value is used as the prior/base rate.
    cr: float
    # Bottleneck channels of autoencoder.
    bottleneck_channels: int
    # Number of encoder/decoder conv blocks.
    ae_depth: int
    # Kernel size for autoencoder convolutions.
    kernel_size: int
    # Whether to add residual skip between JSCC input/output.
    use_skip: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ArchitectureConfig":
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
        skip_tag = "skip" if self.use_skip else "noskip"
        return (
            f"s{self.insertion_stage}_r{self.se_ratio}_cr{self.cr}"
            f"_cb{self.bottleneck_channels}_d{self.ae_depth}"
            f"_k{self.kernel_size}_{skip_tag}"
        )


@dataclass
class SearchSpace:
    insertion_stage: List[int]
    se_ratio: List[int]
    cr: List[float]
    bottleneck_channels: List[int]
    ae_depth: List[int]
    kernel_size: List[int]
    use_skip: List[bool]

    @classmethod
    def default(cls) -> "SearchSpace":
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
