from typing import Any, Callable, List, Optional, Union

import pytorch_lightning as pl
import torch.nn as nn
from aidd_codebase.framework.optimizers import OptimizerChoice
from aidd_codebase.framework.tracking import ExperimentTracker
from aidd_codebase.models.modelchoice import ModelChoice
from aidd_codebase.utils.config import Config
from aidd_codebase.utils.initiator import ParameterInitialization

# from aidd_codebase.framework.loggers import LoggerPL
# from aidd_codebase.framework.scheduling import Scheduler


class ModelFramework(pl.LightningModule):
    def __init__(
        self,
        config: Config,
        model: str,
        loss: str,
        optimizer: str,
        metric_list: Optional[List[str]] = None,
        scheduler: Optional[str] = None,
        initialize_model: Optional[str] = None,
    ) -> None:
        super().__init__()

        # saving parameters
        self.save_hyperparameters(ignore=["config"])

        # Set Model
        model_call = ModelChoice.get_choice(model)
        self.model: Union[nn.Module, pl.LightningModule] = model_call(
            config.return_dataclass("model").__dict__
        )
        if initialize_model:
            param_init = ParameterInitialization(method=initialize_model)
            self.model = param_init.initialize_model(self.model)

        optimizer_call = OptimizerChoice.get_choice(optimizer)
        self.optimizer = optimizer_call(
            self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
        )

        self.scheduler = scheduler

        # Experiment tracker
        self.loss_name = loss
        metrics = [loss] + metric_list if metric_list else [loss]
        self.tracker = ExperimentTracker(config, metrics)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.scheduler:
            self.scheduler.step()

    def configure_optimizers(self):
        optimizer = self.optimizer

        if self.scheduler:
            self.scheduler = self.scheduler.set_optimizer(optimizer)

        return optimizer

    def import_loop_model(self) -> None:
        """Checks if the model supplied has the pl loops
        and sets them if true.
        """
        if callable(getattr(self.model, "training_step", None)):
            self.training_step_imported = self.model.training_step
        if callable(getattr(self.model, "validation_step", None)):
            self.training_step_imported = self.model.validation_step
        if callable(getattr(self.model, "test_step", None)):
            self.training_step_imported = self.model.test_step
        if callable(getattr(self.model, "predict_step", None)):
            self.training_step_imported = self.model.predict_step

    def set_loop(self, loop: Callable, stage: Optional[str] = None) -> None:
        """Allows you to set the training steps externally"""
        if not stage:
            self.training_step_imported = loop
            self.validation_step_imported = loop
            self.test_step_imported = loop
            self.predict_step_imported = loop
        else:
            if stage == "train":
                self.training_step_imported = loop
            elif stage == "validation":
                self.validation_step_imported = loop
            elif stage == "test":
                self.test_step_imported = loop
            elif stage == "predict":
                self.predict_step_imported = loop
            else:
                raise ValueError(f"{stage} stage not found")

    def training_step(self, batch: Any, batch_idx: int):
        self.tracker.init_metrics()
        return self.training_step_imported(self, batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int):
        self.tracker.init_metrics()
        return self.validation_step_imported(self, batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int):
        self.tracker.init_metrics()
        return self.test_step_imported(self, batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int):
        return self.predict_step_imported(self, batch, batch_idx)

    def log_batch(self, stage: str) -> None:
        for key, metric in self.tracker.metrics.items():
            self.log(
                f"{stage}/batch/{metric.name}",
                self.tracker.return_batch_metric(key),
            )

    def log_epoch(self, stage: str) -> None:
        for key, metric in self.tracker.metrics.items():
            self.log(
                f"{stage}/epoch/{metric.name}",
                self.tracker.return_epoch_metric(key),
            )

    def on_training_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.log_batch("train")

    def training_epoch_end(self, training_step_outputs) -> None:
        self.log_epoch("train")

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.log_batch("validation")

    def validation_epoch_end(self, training_step_outputs) -> None:
        self.log_epoch("validation")

    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.log_batch("test")

    def test_epoch_end(self, training_step_outputs) -> None:
        self.log_epoch("test")

    def on_predict_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.log_batch("predict")

    def predict_epoch_end(self, training_step_outputs) -> None:
        self.log_epoch("predict")
