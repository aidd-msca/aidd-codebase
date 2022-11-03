from dataclasses import dataclass
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from abstract_codebase.accreditation import CreditInfo, CreditType
from aidd_codebase.registries import ModelRegistry
from aidd_codebase.utils.typescripts import Tensor
from torch.nn import functional as F
from torchmetrics.classification.accuracy import Accuracy


@ModelRegistry.register_arguments(call_name="net_classifier")
@dataclass
class MLPArguments:
    input_size: int = 1024
    hidden_size: int = 512
    dropout_rate: float = 0.25
    output_size: int = 1
    learning_rate: float = 0.001
    gamma: float = 0.7


@ModelRegistry.register(
    key="net_classifier",
    credit=CreditInfo(author="Peter Hartog", github_handle="PeterHartog",),
    credit_type=CreditType.NONE,
)
class NetClassifier(pl.LightningModule):
    def __init__(self, model_args: Dict):
        super().__init__()

        self.save_hyperparameters()

        args = MLPArguments(**model_args)

        self.lr = args.learning_rate
        self.gamma = args.gamma

        self.test_acc = Accuracy()
        self.val_acc = Accuracy()

        # model
        self.fc1 = nn.Linear(args.input_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc_out = nn.Linear(args.hidden_size, args.output_size)

        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.ln2 = nn.LayerNorm(args.hidden_size)

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc_out(x)

        output = F.log_softmax(out, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)

        loss = F.binary_cross_entropy(logits.squeeze(), y.squeeze())
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # type: ignore
        x, y = batch
        logits = self.forward(x)
        loss = F.binary_cross_entropy(logits.squeeze(), y.squeeze())
        self.val_acc(logits, y.long())
        self.log("val_acc", self.val_acc)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.binary_cross_entropy(logits.squeeze(), y.squeeze())
        self.test_acc(logits, y.long())
        self.log("test_acc", self.test_acc)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        logits = self.forward(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)]
