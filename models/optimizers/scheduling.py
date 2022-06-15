from typing import Protocol

import numpy as np
from torch import optim


class SchedulerProtocol(Protocol):
    def get_lr(self) -> float:
        """Returns the next learning rate."""

    def step(self, epoch: int) -> None:
        """Instantiates the optimizer with the lr of the epoch"""


class Scheduler:  # optim.lr_sheduler._LRScheduler
    pass


class LinearPiecewiseScheduler(Scheduler):
    pass


class CosineWarmupScheduler(Scheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def step(self, epoch: int) -> None:
        return self.get_lr(epoch=epoch)

    def get_lr(self, epoch: int):
        lr_factor = self.get_lr_factor(epoch=epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
