import torch.nn as nn

from aidd_codebase.utils.metacoding import DictChoiceFactory
from aidd_codebase.utils.typescripts import Tensor


class ActivationChoice(DictChoiceFactory):
    pass


# Registers choices for loss functions of prebuilt loss functions
ActivationChoice.register_prebuilt_choice(
    call_name="relu", callable_cls=nn.ReLU
)
ActivationChoice.register_prebuilt_choice(
    call_name="kullback_leibler_div", callable_cls=nn.KLDivLoss
)
ActivationChoice.register_prebuilt_choice(
    call_name="kullback_leibler_recon_div", callable_cls=nn.KLRecDivLoss
)


@ActivationChoice.register_choice("custom_relu")
class ReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(x: Tensor) -> Tensor:
        return x.clamp_min(0.0).requires_grad_(True)


@ActivationChoice.register_choice("relu_alt")
class AlternativeReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(x: Tensor) -> Tensor:
        return (
            x.clamp_min(0.0).requires_grad_(True) - 0.5
        )  # To get better distr after relu
