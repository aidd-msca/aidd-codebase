import torch.nn as nn

from abstract_codebase.registration import RegistryFactory
from aidd_codebase.utils.typescripts import Tensor


class ActivationChoice(RegistryFactory):
    pass


# Registers choices for loss functions of prebuilt loss functions
ActivationChoice.register_prebuilt(key="relu", obj=nn.ReLU)
ActivationChoice.register_prebuilt(key="kullback_leibler_div", obj=nn.KLDivLoss)
ActivationChoice.register_prebuilt(key="kullback_leibler_recon_div", obj=nn.KLRecDivLoss)


@ActivationChoice.register(key="custom_relu")
class ReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(x: Tensor) -> Tensor:
        return x.clamp_min(0.0).requires_grad_(True)


@ActivationChoice.register(key="relu_alt")
class AlternativeReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(x: Tensor) -> Tensor:
        return x.clamp_min(0.0).requires_grad_(True) - 0.5  # To get better distr after relu
