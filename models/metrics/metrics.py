from dataclasses import dataclass, field
from typing import List

import torch
import torchmetrics


@dataclass
class Metric:
    name: str = "metric"
    values: List[float] = field(default_factory=list)
    running_total: float = 0.0
    num_updates: float = 0.0
    average: float = 0.0

    def update(self, value: float, size: int = 1):
        self.values.append(value)
        self.running_total += value * size
        self.num_updates += size
        # weighted average
        self.average = self.running_total / self.num_updates


# example of RMSE in a torch (lightning) metric #TODO
class TorchMetric(torchmetrics.Metric):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.add_state("values", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_observations", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.values += torch.sum((preds - target) ** 2)
        self.n_observations += preds.numel()

    def cumpute(self):
        return torch.sqrt(self.RMSE / self.n_observations)


# Example of usage of torchmetrics
# def __init__(self):
#     ...
#     self.train_acc = torchmetrics.Accuracy()
#     self.valid_acc = torchmetrics.Accuracy()

# def training_step(self, batch, batch_idx):
#     x, y = batch
#     preds = self(x)
#     ...
#     self.train_acc(preds,y)
#     self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
#     ...

# def validation_step(self, batch, batch_idx):
#     x, y = batch
#     logits = self(x)
#     ...
#     self.valid_acc(logits, y)
#     self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
