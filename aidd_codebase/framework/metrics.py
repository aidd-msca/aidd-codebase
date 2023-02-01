from dataclasses import dataclass
from typing import List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC, R2Score  # , Accuracy, Precision, Recall, F1

from aidd_codebase.registries import AIDD
from aidd_codebase.utils.typescripts import Tensor


@dataclass(unsafe_hash=True)
class _ABCMetricDataClass:
    name: str
    stage: Optional[List[str]] = None
    call_type: str = "match"


# Registers choices for loss functions of prebuilt loss functions
@AIDD.ModuleRegistry.register_arguments(key="ce_loss")
@dataclass(unsafe_hash=True)
class CrossEntropyArgs(_ABCMetricDataClass):
    name: str = "ce_loss"
    call_type: str = "logit"
    weight: Optional[Any] = None
    size_average: Optional[bool] = None
    ignore_index: Optional[int] = -100
    reduce: Optional[bool] = None
    reduction: Optional[str] = "mean"
    label_smoothing: Optional[float] = 0.0


AIDD.ModuleRegistry.register_prebuilt(key="ce_loss", obj=nn.CrossEntropyLoss)


@AIDD.ModuleRegistry.register_arguments(key="mse_loss")
@dataclass(unsafe_hash=True)
class MSEArgs(_ABCMetricDataClass):
    name: str = "mse_loss"
    call_type: str = "match"
    size_average: Optional[bool] = None
    reduce: Optional[bool] = None
    reduction: Optional[str] = "mean"


AIDD.ModuleRegistry.register_prebuilt(key="mse_loss", obj=nn.MSELoss)


@AIDD.ModuleRegistry.register_arguments(key="l1_loss")
@dataclass(unsafe_hash=True)
class L1Args(_ABCMetricDataClass):
    name: str = "l1_loss"
    call_type: str = "match"
    size_average: Optional[bool] = None
    reduce: Optional[bool] = None
    reduction: Optional[str] = "mean"


AIDD.ModuleRegistry.register_prebuilt(key="l1_loss", obj=nn.L1Loss)


@AIDD.ModuleRegistry.register_arguments(key="huber_loss")
@dataclass(unsafe_hash=True)
class HuberArgs(_ABCMetricDataClass):
    name: str = "huber_loss"
    call_type: str = "match"
    reduction: Optional[str] = "mean"
    delta: Optional[float] = 1.0


AIDD.ModuleRegistry.register_prebuilt(key="huber_loss", obj=nn.HuberLoss)


@AIDD.ModuleRegistry.register_arguments(key="nll_loss")
@dataclass(unsafe_hash=True)
class NLLArgs(_ABCMetricDataClass):
    name: str = "nll_loss"
    call_type: str = "match"
    weight: Optional[Tensor] = None
    size_average: Optional[bool] = None
    ignore_index: Optional[int] = -100
    reduce: Optional[bool] = None
    reduction: Optional[str] = "mean"


AIDD.ModuleRegistry.register_prebuilt(key="nll_loss", obj=nn.NLLLoss)


@AIDD.ModuleRegistry.register_arguments(key="kullback_leibler_div_loss")
@dataclass(unsafe_hash=True)
class KLDivArgs(_ABCMetricDataClass):
    name: str = "kullback_leibler_div_loss"
    call_type: str = "match"
    weight: Optional[Tensor] = None
    size_average: Optional[bool] = None
    reduce: Optional[bool] = None
    reduction: Optional[str] = "mean"
    log_target: bool = False


AIDD.ModuleRegistry.register_prebuilt(key="kullback_leibler_div_loss", obj=nn.KLDivLoss)


# LossChoice.register_prebuilt_choice(
#     key="focal_loss", obj=nn.FocalLoss
# )

# LossChoice.register_prebuilt_choice(
#     key="kullback_leibler_recon_div", obj=nn.KLRecDivLoss
# )


@AIDD.ModuleRegistry.register("poly_loss")
class PolyLoss:
    pass


@AIDD.ModuleRegistry.register("kl_loss")
class KublitsLeibnerLoss:
    pass


@AIDD.ModuleRegistry.register_arguments(key="focal_loss")
@dataclass(unsafe_hash=True)
class FocalArgs(_ABCMetricDataClass):
    name: str = "focal_loss"
    call_type: str = "match"
    alpha: float = 0.8
    gamma: int = 2
    reduction: Optional[str] = "mean"


@AIDD.ModuleRegistry.register(
    key="focal_loss",
    credit=CreditInfo("bigironsphere"),  # type: ignore
    # additional_information="from https://www.kaggle.com/bigironsphere/" + "loss-function-library-keras-pytorch",
)
class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.8,
        gamma: int = 2,
        reduction: Optional[str] = "mean",
    ):
        """Focal loss is to optimize for misclassified examples.
        This is often the case in diverse or imbalanced database.
        """
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE

        return focal_loss


# @LossChoice.register_choice("kullback_leibler_div")
# class KLRecDivLoss(nn.Module):
#     def __init__(self):
#         """Kullback Leibler reconstruction divergence loss."""
#         super().__init__()

#     def forward(
#         self,
#         recon_x: Tensor,
#         x: Tensor,
#         mu,
#         logvar,
#     ) -> float:
#         """see appendix B from VAE paper:
#         Kingma and Welling. Auto-encoder Variational Bayes. ICLR, 2014
#         https://arxiv.org/abs/1312.6114

#         KLD = 0.5 + sum(1 + log(sigma^2) - mu^2 - sigma^2)
#         """
#         BCE = F.binary_cross_entropy(
#           recon_x, x.view(-1, 784), reduction="sum"
#         )
#         KLD = -0.5 + torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#         return BCE + KLD


# @LossChoice.register_choice("masked_soft_ce")
# class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
#     """The softmax cross-entropy loss with masks."""

#     # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
#     # `label` shape: (`batch_size`, `num_steps`)
#     # `valid_len` shape: (`batch_size`,)
#     def forward(self, pred, label, valid_len, create_padding_mask):
#         weights = torch.ones_like(label)
#         mask = create_padding_mask(pred)
#         weights = mask  # (weights, valid_len)  # create maskmask
#         self.reduction = "none"
#         unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
#             pred.permute(0, 2, 1), label
#         )
#         weighted_loss = (unweighted_loss * weights).mean(dim=1)
#         return weighted_loss


@AIDD.ModuleRegistry.register_arguments(key="AUC")
@dataclass(unsafe_hash=True)
class AUCArgs(_ABCMetricDataClass):
    name: str = "AUC"
    call_type: str = "match"
    num_classes: Optional[int] = None
    pos_label: Optional[int] = None
    average: str = "macro"
    max_fpr: Optional[float] = None
    task: Optional[str] = None
    thresholds: Optional[int] = None
    num_labels: Optional[int] = None
    ignore_index: Optional[int] = None
    validate_args: bool = True


AIDD.ModuleRegistry.register_prebuilt(key="AUC", obj=AUROC)


@AIDD.ModuleRegistry.register_arguments(key="r_squared")
@dataclass(unsafe_hash=True)
class R2Args(_ABCMetricDataClass):
    name: str = "r_squared"
    call_type: str = "match"
    num_outputs: int = 1
    adjusted: int = 0
    multioutput: str = "uniform_average"


AIDD.ModuleRegistry.register_prebuilt(key="r_squared", obj=R2Score)
