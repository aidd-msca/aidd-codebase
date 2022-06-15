from typing import Callable, Optional

import torch.nn as nn

from ...models.modules.highway_units import (
    ConstantScalingUnit,
    ConvolutionShortcutUnit,
    DropoutShortcutUnit,
    HighwayUnit,
)
from aidd_codebase.utils.typescripts import Tensor


def unit_switch(unit: str) -> Callable:
    switch = {
        "highway": HighwayUnit,
        "constant_scaling": ConstantScalingUnit,
        "dropout_shortcut": DropoutShortcutUnit,
        "convolution_shortcut": ConvolutionShortcutUnit,
    }
    return switch.get(unit)


class HighwayRHCNN(nn.Module):
    def __init__(
        self,
        highway_type: str,
        n_layers: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: Optional[float] = 0.1,
    ) -> None:
        super().__init__()
        assert highway_type in [
            "highway",
            "constant_scaling",
            "dropout_shortcut",
            "convolution_shortcut",
        ], (
            "unit type must be highway, constant_scaling, dropout_shortcut"
            + f"or convolution_shortcut, got {highway_type}"
        )
        unit = unit_switch(highway_type)
        if highway_type == "dropout_shortcut":
            self.highway = [
                unit(in_channels, out_channels, kernel_size, stride, dropout)
                for _ in range(n_layers)
            ]
        else:
            self.highway = [
                unit(in_channels, out_channels, kernel_size, stride)
                for _ in range(n_layers)
            ]

    def forward(self, x: Tensor) -> Tensor:
        for unit in self.highway:
            x = unit(x)
        return x
