from typing import Any

from aidd_codebase.utils.metacoding import DictChoiceFactory
from aidd_codebase.utils.typescripts import Tensor


class LossChoice(DictChoiceFactory):
    pass


# Registers choices for loss functions of prebuilt loss functions
LossChoice.register_prebuilt_choice(call_name="adam", callable_cls=optim.Adam)


@LossChoice.register_choice("logit_loss")
class LogitLoss:
    def __init__(self, function: Any) -> None:
        self.criterium = function

    def __call__(self, logits: Tensor, tgt_out: Tensor):
        return self.criterium(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
        )  # .item()  #(use loss.item to reserve memory)


@LossChoice.register_choice("focal_loss")
class FocalLoss:
    pass


@LossChoice.register_choice("ce_loss")
class CrossEntropyLoss:
    pass


@LossChoice.register_choice("poly_loss")
class PolyLoss:
    pass


@LossChoice.register_choice("kl_loss")
class KublitsLeibnerLoss:
    pass
