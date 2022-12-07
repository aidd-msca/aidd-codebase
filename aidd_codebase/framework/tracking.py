from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from aidd_codebase.framework.metrics import MetricsChoice
from aidd_codebase.utils.typescripts import Tensor


class Stage(Enum):
    train = "train"
    validation = "validation"
    test = "test"
    predict = "predict"


@dataclass
class Metric:
    values: List[float] = field(default_factory=list)
    running_total: float = 0.0
    num_updates: float = 0.0
    last_size: int = 0

    def update(self, value: float, size: int = 1) -> None:
        self.values.append(value)
        self.running_total += value * size
        self.num_updates += size
        self.last_size = size

    def last_update(self) -> float:
        return self.values[-1]

    def weighted_average(self) -> float:
        return self.running_total / self.num_updates


class FrameworkMetric:
    def __init__(
        self,
        name: str,
        stage: Optional[List[str]] = None,
        call_type: str = "match",
        **kwargs,
    ) -> None:
        self.stage = ["train", "validation", "test", "predict"] if not stage else stage
        self.name = name
        self.call_type = call_type

        metric_call = MetricsChoice.get_choice(name)
        self.metric = metric_call(**kwargs)

        self.set_call(call_type)

    def set_call(self, call_type: str) -> None:
        if call_type == "logit":
            self.call_method = self.logit_call
        elif call_type == "match":
            self.call_method = self.match_call
        else:
            raise ValueError("Metric type not found, options are 'logit' and 'match'," + f"got {self.call_type}")

    def __call__(self, prediction: Tensor, target: Tensor):
        return self.call_method(prediction, target)

    @staticmethod
    def call_method(self, prediction: Tensor, target: Tensor):
        raise AssertionError("Call function not changed")

    def logit_call(self, logits: Tensor, tgt_out: Tensor):
        return self.metric(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

    def match_call(self, pred: Tensor, actual: Tensor):
        return self.metric(pred, actual)


class ExperimentTracker:
    stage: str
    metrics: Dict[str, FrameworkMetric]
    metric_list: Dict[str, Metric]
    call_types: List[str] = []

    def __init__(self, cfg, metrics: List[str]) -> None:
        self.set_stage("train")
        self.metrics = {name: FrameworkMetric(**cfg["loss"]) for name in metrics}
        self.metric_list: Dict[str, Metric] = {}

    def set_stage(self, stage: str) -> None:
        """Sets the current stage of the experiment."""
        self.stage = stage

    def init_metrics(self) -> None:
        """Initializes metrics for a stage of the experiment."""
        self.metric_list = {metric.name: Metric() for metric in self.metrics.values() if self.stage in metric.stage}
        stage_metrics = {metric.name: metric for metric in self.metrics.values() if self.stage in metric.stage}
        self.call_types = list(set([metric.call_type for metric in stage_metrics.values()]))

    def update_metric(self, name: str, y_hat: Tensor, y_true: Tensor, size: int) -> None:
        """Updates metrics after a step of the experiment."""
        self.metric_list[name].update(self.metrics[name](y_hat, y_true), size)

    def return_batch_metric(self, name: str) -> float:
        """Returns last update for batch-level logging."""
        return self.metric_list[name].last_update()

    def return_epoch_metric(self, name: str) -> float:
        """Returns weighted average for epoch-level logging."""
        return self.metric_list[name].weighted_average()

    def reset(self) -> None:
        """Resets the metrics."""
        self.init_metrics()
