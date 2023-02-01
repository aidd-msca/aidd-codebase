import torch
import torch.nn as nn

from aidd_codebase.registries import AIDD


@AIDD.ModuleRegistry.register(key="xavier_init")
def xavier_initialization(model: nn.Module) -> nn.Module:
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model


@AIDD.ModuleRegistry.register(key="kaiming_init")
def kaiming_initialization(model: nn.Module) -> nn.Module:
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.kaiming_normal_(p, mode="fan_out")
    return model
