# from pytorch_lightning.loops import FitLoop
from typing import Any

import pytorch_lightning as pl

from aidd_codebase.utils.typescripts import Tensor


class Seq2SeqLoop(pl.LightningModule):
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
