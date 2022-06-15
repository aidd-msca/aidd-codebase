import pytorch_lightning as pl
import torch.nn as nn

from aidd_codebase.utils.typescripts import Tensor


class _HighwayUnit(pl.LightningModule):
    """Abstract concept of a highway unit where T(x) is the transform gate
    and H(x) is the transform.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.H = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
        )
        self.T = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        transform_gate = self.T(x)  # prevent double backprop?
        y = transform_gate * self.H(x) + x * (1 - transform_gate)
        return y


class HighwayUnit(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        y = x + self.relu1(self.conv1(x)) * self.relu2(self.conv2(x))
        return y


class ConstantScalingUnit(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        y = 0.5 * x + 0.5 * self.relu1(self.conv1(x)) * self.relu2(
            self.conv2(x)
        )
        return y


class DropoutShortcutUnit(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.dropout(x) + self.relu1(self.conv1(x)) * self.relu2(
            self.conv2(x)
        )
        return y


class ConvolutionShortcutUnit(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv3 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        y = self.relu3(self.conv3(x)) + self.relu1(self.conv1(x)) * self.relu2(
            self.conv2(x)
        )
        return y
