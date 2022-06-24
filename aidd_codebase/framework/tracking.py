from enum import Enum, auto
from typing import Protocol


class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


class StagePL(Enum):
    FIT = "fit"
    TEST = "test"
    PREDICT = "predict"


class ExperimentTracker(Protocol):
    def set_stage(self, stage: Stage):
        """Sets the current stage of the experiment."""

    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric."""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging a epoch-level metric."""
