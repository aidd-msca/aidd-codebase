import torch.nn as nn

from aidd_codebase.utils.tools import DictChoiceFactory


class SwitchChoice(DictChoiceFactory):
    pass


@SwitchChoice.register_choice("excl_switch")
class ExclusiveSwitch(nn.Module):
    pass


@SwitchChoice.register_choice("moe_switch")
class MoESwitch(nn.Module):
    pass
