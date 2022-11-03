# from pytorch_lightning.loops import FitLoop
from typing import Any, Optional

import pytorch_lightning as pl
from aidd_codebase.framework.framework import ModelFramework
from aidd_codebase.framework.loopchoice import LoopChoice
from aidd_codebase.utils.metacoding import CreditType
from aidd_codebase.utils.typescripts import Tensor


@LoopChoice.register_choice(
    call_name="pl_seq2seq_loops",
    author="Peter Hartog",
    github_handle="PeterHartog",
    credit_type=CreditType.NONE,
)
class Seq2SeqLoop(pl.LightningModule):
    """Performs the training step, assumed dims [L, B]"""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def training_step(self: ModelFramework, batch, batch_idx):
        """Steps for training and validation"""
        stage = "train"

        x, y = batch
        batch_size = x.shape[1]  # assumed dims are [L, B]

        tgt_in = y[:-1, :]
        logits = self(x, tgt_in)

        if "match" in self.tracker.call_types:
            y_hat = self.model.beam_search_predict(x, alg="greedy", k=1)
        else:
            y_hat = None

        Seq2SeqLoop.calc_metrics(self, stage, batch_size, y, logits, y_hat)
        outputs = {"loss": self.tracker.return_batch_metric(self.loss_name)}
        self.log("loss", outputs["loss"], batch_size=batch_size)
        return outputs

    @staticmethod
    def validation_step(self: ModelFramework, batch, batch_idx):
        """Steps for testing and prediction"""
        stage = "validation"

        x, y = batch
        batch_size = x.shape[1]  # assumed dims are [L, B]

        tgt_in = y[:-1, :]
        logits = self(x, tgt_in)

        if "match" in self.tracker.call_types:
            y_hat = self.model.beam_search_predict(x, alg="greedy", k=1)
        else:
            y_hat = None

        Seq2SeqLoop.calc_metrics(self, stage, batch_size, y, logits, y_hat)
        outputs = {
            "val_loss": self.tracker.return_batch_metric(self.loss_name)
        }
        self.log("val_loss", outputs["val_loss"])  # , batch_size=batch_size)
        return outputs

    @staticmethod
    def test_step(self, batch, batch_idx):
        """Steps for testing and prediction"""
        stage = "test"

        x, y = batch
        batch_size = x.shape[1]  # assumed dims are [L, B]

        tgt_in = y[:-1, :]
        logits = self(x, tgt_in)

        if "match" in self.tracker.call_types:
            y_hat = self.model.beam_search_predict(x, alg="greedy", k=1)
        else:
            y_hat = None

        Seq2SeqLoop.calc_metrics(self, stage, batch_size, y, logits, y_hat)
        outputs = {
            "test_loss": self.tracker.return_batch_metric(self.loss_name)
        }
        self.log(
            "test_batch/loss", outputs["test_loss"], batch_size=batch_size
        )
        return outputs

    @staticmethod
    def predict_step(self, batch, batch_idx):
        """Steps for prediction"""
        x = batch
        return self.model.beam_search_predict(x, alg="greedy", k=1)

    @staticmethod
    def calc_metrics(
        self: ModelFramework,
        stage: str,
        batch_size: Tensor,
        y: Tensor,
        logits: Optional[Tensor] = None,
        y_hat: Optional[Tensor] = None,
    ):
        for key, metric in self.tracker.metrics.items():
            if metric.call_type == "logit":
                self.tracker.update_metric(key, logits, y[1:, :], batch_size)
            elif metric.call_type == "match":
                self.tracker.update_metric(key, y_hat, y[1:, :], batch_size)

            # self.log(
            #     f"{stage}_batch/{metric.name}",
            #     self.tracker.return_batch_metric(key),
            #     batch_size=batch_size,
            # )


@LoopChoice.register_choice(
    call_name="pl_transformer_loops",
    author="Peter Hartog",
    github_handle="PeterHartog",
    credit_type=CreditType.NONE,
)
class TransformerLoops(pl.LightningModule):
    """Performs the training step, assumed dims [L, B]"""

    def __init__(self) -> None:
        super().__init__()

    def basic_loop(self, batch: Any, batch_idx: int, stage: str) -> Tensor:
        # x, y = batch                      moved to training_step
        # y_hat = model(x)                  moved to training_step
        # loss = loss_function(y_hat, y)    moved to training_step
        pass

    @staticmethod
    def translation(self, batch, batch_idx, stage):
        x, y = batch
        batch_size = x.shape[1]  # assumed dims are [L, B]

        tgt_in = y[:-1, :]
        logits = self(x, tgt_in)

        loss_key = self.get_loss_key(stage)
        for i, key in enumerate(self.metrics_list[stage].keys()):
            if key == loss_key:
                self.metrics_list[stage][key].update(
                    self.loss(logits, y[1:, :]), batch_size
                )
            else:
                self.metrics_list[stage][key].update(
                    self.metrics[i](logits, y[1:, :]), batch_size
                )

        outputs = {loss_key: self.metrics_list[stage][loss_key].values[-1]}
        return outputs

    @staticmethod
    def regression(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[1]  # assumed dims are [L, B]

        y_hat = self(x)
        y_hat = y_hat.squeeze(1)

        for key in self.metrics_list.keys():
            self.train_metrics[key].update(
                self.metrics[key](y, y_hat), batch_size
            )

        outputs = {
            "loss": self.train_metrics["loss"].values[-1],
            "batch_size": batch_size,
        }
        return outputs

    @staticmethod
    def classification(self, batch, batch_idx):
        return self.regression(batch, batch_idx)


@LoopChoice.register_choice(
    call_name="basic_qsar_loops",
    author="Peter Hartog",
    github_handle="PeterHartog",
    credit_type=CreditType.NONE,
)
class QSARLoop(pl.LightningModule):
    """Performs the training step, assumed dims [B, X_size]"""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def basic_loop(self, batch: Any, batch_idx: int, stage: str) -> Tensor:
        x, y = batch

        batch_size = x.shape[0]  # assumed dims are [B, X_size]

        y_hat = self.model(x)

        loss_key = self.get_loss_key(stage)
        for i, key in enumerate(self.metrics_list[stage].keys()):
            if key == loss_key:
                self.metrics_list[stage][key].update(
                    self.loss(y_hat, y), batch_size
                )
            else:
                self.metrics_list[stage][key].update(
                    self.metrics[i](y_hat, y), batch_size
                )

        outputs = {loss_key: self.metrics_list[stage][loss_key].values[-1]}
        return outputs


# class MyLoop(FitLoop):
#     def advance(self):
#         """Advance from one iteration to the next."""

#     def on_advance_end(self):
#         """Do something at the end of an iteration."""

#     def on_run_end(self):
#         """Do something when the loop ends."""


# class DefaultLoop(Loop):
#     def advance(self, batch, i):
#         loss = lightning_module.training_step(batch, i)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     def run(self, dataloader):
#         for i, batch in enumerate(dataloader):
#             self.advance(batch, i)


# from pytorch_lightning.loop import Loop


# class CustomLoop(Loop):
#     @property
#     def done(self) -> bool:
#         """Provide a condition to stop the loop."""

#     @property
#     def skip(self) -> bool:
#         """Determine whether to return immediately from the call to run()."""
#         pass

#     def reset(self) -> None:
#         """Resets the internal state of the loop at the beginning of each
#         call to run. Use when run is called multiple times."""
#         self.current_iteration = 0
#         self.outputs = []

#     def advance(self, *args, **kwargs):
#         """
#         Accepts all arguments passed to run.
#         Access your dataloader/s in whatever way you want.
#         Do your fancy optimization things.
#         Call the LightningModule methods at your leisure.
#         """
#         batch = next(iterator)
#         loss = self.trainer.lightning_module.training_step(batch, batch_idx)
#         ...

#     def run(self, *args, **kwargs):
#         if self.skip:
#             return self.on_skip()

#         self.reset()
#         self.on_run_start(*args, **kwargs)

#         while not self.done:
#             self.advance(*args, **kwargs)

#         output = self.on_run_end()
#         return output


# class ExampleCustomLoop(Loop):
#     @property
#     def done(self) -> bool:
#         return self.trainer.global_step >= self.trainer.max_steps

#     @property
#     def skip(self) -> bool:
#         return len(self.trainer.train_dataloader) == 0
