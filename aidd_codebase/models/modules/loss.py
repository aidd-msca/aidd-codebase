import torch
import torch.nn as nn
import torch.nn.functional as F

from aidd_codebase.utils.metacoding import DictChoiceFactory
from aidd_codebase.utils.typescripts import Tensor


class LossChoice(DictChoiceFactory):
    pass


LossChoice.register_prebuilt_choice(
    call_name="cross_entropy", callable_cls=nn.CrossEntropyLoss
)
# Registers choices for loss functions of prebuilt loss functions
# LossChoice.register_prebuilt_choice(
#     call_name="focal_loss", callable_cls=nn.FocalLoss
# )
# LossChoice.register_prebuilt_choice(
#     call_name="kullback_leibler_div", callable_cls=nn.KLDivLoss
# )
# LossChoice.register_prebuilt_choice(
#     call_name="kullback_leibler_recon_div", callable_cls=nn.KLRecDivLoss
# )


# from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
@LossChoice.register_choice("focal_loss")
class FocalLoss(nn.Module):
    def __init__(self):
        """Focal loss is to optimize for misclassified examples.
        This is often the case in diverse or imbalanced database.
        """
        super().__init__()

    def forward(
        self,
        inputs: Tensor,
        targets: Tensor,
        alpha: float = 0.8,
        gamma: int = 2,
    ):
        # comment out if your model contains a sigmoid or
        # equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


@LossChoice.register_choice("kullback_leibler_div")
class KLRecDivLoss(nn.Module):
    def __init__(self):
        """Kullback Leibler reconstruction divergence loss."""
        super().__init__()

    def forward(
        self,
        recon_x: Tensor,
        x: Tensor,
        mu,
        logvar,
    ) -> float:
        """see appendix B from VAE paper:
        Kingma and Welling. Auto-encoder Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114

        KLD = 0.5 + sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
        KLD = -0.5 + torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


@LossChoice.register_choice("masked_soft_ce")
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks."""

    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len, create_padding_mask):
        weights = torch.ones_like(label)
        mask = create_padding_mask(pred)
        weights = mask  # (weights, valid_len)  # create maskmask
        self.reduction = "none"
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label
        )
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
