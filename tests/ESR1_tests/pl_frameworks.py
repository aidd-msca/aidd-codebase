from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from aidd_codebase.models.metrics.metrics import Metric
from aidd_codebase.models.seq2seq import Seq2Seq
from aidd_codebase.utils.typescripts import Tensor


class TASK(Enum):
    REGRESSION = auto()
    CLASSIFICATION = auto()
    TRANSLATION = auto()
    GENERATION = auto()


class TopNAccuracy:
    def __init__(self, k: int = 3) -> None:
        self.k = k

    def __call__(
        self,
        logits: Tensor,
        tgt_out: Tensor,
    ) -> Tensor:
        """Calculates the top n accuracy for a logit and a target tensor."""
        tgt_padding_mask = Seq2Seq.create_padding_mask(tgt_out, 0)

        n_valid_tokens = torch.sum(~tgt_padding_mask.flatten())
        probs = torch.nn.functional.softmax(logits, dim=-1)
        _, top_tokens = torch.topk(
            probs, k=self.k, dim=-1
        )  # (values, indices)
        top_n_accuracy = torch.sum(
            torch.logical_and(
                torch.any(
                    top_tokens == tgt_out.unsqueeze(-1).tile(1, 1, self.k),
                    dim=-1,
                ),
                ~tgt_padding_mask.transpose(1, 0),
            )
            / n_valid_tokens
        )
        return top_n_accuracy


class LogitLoss:
    def __init__(self, function: Any) -> None:
        self.criterium = function

    def __call__(self, logits: Tensor, tgt_out: Tensor):
        return self.criterium(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
        )  # .item()


class ModelFramework_old(pl.LightningModule):
    def __init__(
        self,
        task: TASK,
        optimizer: Any,
        metrics: Dict[str, Any] = {
            "loss": LogitLoss(nn.CrossEntropyLoss(reduction="mean")),
            "accuracy": TopNAccuracy(1),
            "accuracy_top3": TopNAccuracy(3),
        },
        lr: Optional[float] = 1e-4,
        lr_scheduler: Optional[Any] = None,
        lr_warmup: Optional[int] = None,
        lr_max_iters: Optional[int] = None,
    ) -> None:
        super().__init__()

        # saving parameters
        self.save_hyperparameters(ignore=["model"])

        # setting parameters
        self.set_task(task)
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_warmup = lr_warmup
        self.lr_max_iters = lr_max_iters

    def set_model(self, model: pl.LightningModule) -> None:
        self.model = model

    def set_task(self, task: TASK) -> None:
        self.task = task

    def initialize_model(self, method: str = "xavier") -> None:
        if method == "xavier":
            for p in self.model.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
        elif method == "kaiming":
            for p in self.model.parameters():
                if p.dim() > 1:
                    torch.nn.init.kaiming_normal_(p, mode="fan_out")
        else:
            raise ValueError(f"initialization {method} is not recognized.")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def init_metrics(self, metrics: List[str] = ["loss"]) -> Dict[str, Metric]:
        return {k: Metric() for k in metrics}

    def log_dict(  # type: ignore
        self, dict: dict, batch_size: Optional[Tensor] = None
    ) -> None:
        """Logs everything in a dict."""
        for key, value in dict.items():
            self.log(key, value, batch_size=batch_size)

    def on_train_start(self) -> None:
        self.train_metrics = self.init_metrics(metrics=self.metrics.keys())

    def on_training_epoch_start(self) -> None:
        """Called before every training epoch."""
        self.train_metrics = self.init_metrics(metrics=self.metrics.keys())

    def training_step(self, batch, batch_idx):
        """Performs the training step, assumed dims [L, B]"""
        x, y = batch
        batch_size = x.shape[1]  # assumed dims are [L, B]

        if self.task == TASK.REGRESSION or self.task == TASK.CLASSIFICATION:
            y_hat = self(x)
            y_hat = y_hat.squeeze(1)

            for key in self.train_metrics.keys():
                self.train_metrics[key].update(
                    self.metrics[key](y, y_hat), batch_size
                )
        elif self.task == TASK.TRANSLATION:
            tgt_in = y[:-1, :]
            logits = self(x, tgt_in)

            for key in self.train_metrics.keys():
                self.train_metrics[key].update(
                    self.metrics[key](logits, y[1:, :]), batch_size
                )

        outputs = {
            "loss": self.train_metrics["loss"].values[-1],
            "batch_size": batch_size,
        }
        return outputs

    def on_training_batch_end(
        self,
        outputs: Union[Tensor, Dict[str, Any], None],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.log_dict(
            {
                f"train_{k}": v.values[-1]
                for k, v in self.train_metrics.items()
            },
            torch.tensor(outputs["batch_size"], device=self.device),
        )

    def training_epoch_end(self, outputs: List[Union[Tensor, Dict[str, Any]]]):
        """This function is called after every epoch"""
        # logging using tensorboard logger
        for k in self.train_metrics.keys():
            self.logger[0].experiment.add_scalar(
                f"{k}/train", self.train_metrics[k].average, self.current_epoch
            )

    def on_validation_epoch_start(self) -> None:
        """Called before every validation epoch."""
        self.val_metrics = self.init_metrics(metrics=self.metrics.keys())

    def validation_step(self, batch, batch_idx):
        """Performs the validation step, assumed dims [L, B]"""
        x, y = batch
        batch_size = x.shape[1]  # assumed dims are [L, B]

        if self.task == TASK.REGRESSION or self.task == TASK.CLASSIFICATION:
            y_hat = self(x)
            y_hat = y_hat.squeeze(1)
            for key in self.val_metrics.keys():
                self.val_metrics[key].update(
                    self.metrics[key](y, y_hat), batch_size
                )
        elif self.task == TASK.TRANSLATION:
            tgt_in = y[:-1, :]
            logits = self(x, tgt_in)
            for key in self.val_metrics.keys():
                self.val_metrics[key].update(
                    self.metrics[key](logits, y[1:, :]), batch_size
                )

        outputs = {
            "val_loss": self.val_metrics["loss"].values[-1],
            "batch_size": batch_size,
        }
        return outputs

    def on_validation_batch_end(
        self,
        outputs: Union[Tensor, Dict[str, Any], None],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.log_dict(
            {f"val_{k}": v.values[-1] for k, v in self.val_metrics.items()},
            torch.tensor(outputs["batch_size"], device=self.device),
        )

    def validation_epoch_end(
        self, outputs: List[Union[Tensor, Dict[str, Any]]]
    ):
        """This function is called after every epoch"""
        # logging using tensorboard logger
        for k in self.val_metrics.keys():
            self.logger[0].experiment.add_scalar(
                f"{k}/validation",
                self.val_metrics[k].average,
                self.current_epoch,
            )

    def on_test_epoch_start(self) -> None:
        """Called before every validation epoch."""
        self.test_metrics = self.init_metrics(metrics=self.metrics.keys())

    def test_step(self, batch, batch_idx):
        """Performs the test step, assumed dims [L, B]"""
        x, y = batch
        batch_size = x.shape[1]  # assumed dims are [L, B]

        if self.task == TASK.REGRESSION or self.task == TASK.CLASSIFICATION:
            y_hat = self(x)
            y_hat = y_hat.squeeze(1)
            for key in self.test_metrics.keys():
                self.test_metrics[key].update(
                    self.metrics[key](y, y_hat), batch_size
                )
        elif self.task == TASK.TRANSLATION:
            tgt_in = y[:-1, :]
            logits = self(x, tgt_in)
            for key in self.test_metrics.keys():
                self.test_metrics[key].update(
                    self.metrics[key](logits, y[1:, :]), batch_size
                )

        outputs = {
            "test_loss": self.test_metrics["loss"].values[-1],
            "batch_size": batch_size,
        }
        return outputs

    def on_test_batch_end(
        self,
        outputs: Union[Tensor, Dict[str, Any], None],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.log_dict(
            {f"test_{k}": v.values[-1] for k, v in self.test_metrics.items()},
            torch.tensor(outputs["batch_size"], device=self.device),
        )

    def test_epoch_end(self, outputs: List[Union[Tensor, Dict[str, Any]]]):
        """This function is called after every epoch"""
        # logging using tensorboard logger
        for k in self.test_metrics.keys():
            self.logger[0].experiment.add_scalar(
                f"{k}/test", self.test_metrics[k].average, self.current_epoch
            )

    def configure_optimizers(self):
        optimizer = self.optimizer

        # #TODO We don't return the lr scheduler
        # because we need to apply it per iteration, not per epoch
        if self.lr_scheduler:
            self.lr_scheduler = self.lr_scheduler(
                optimizer,
                warmup=self.lr_warmup,
                max_iters=self.lr_max_iters,
            )
        # lr_scheduler = self.lr_scheduler

        return optimizer
