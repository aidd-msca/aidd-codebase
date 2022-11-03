import torch
import torch.nn as nn


class ParameterInitialization:
    def __init__(self, method: str = "xavier") -> None:
        self.initialize_method = method
        if method not in ["xavier", "kaiming"]:
            raise ValueError(f"initialization {method} is not recognized.")

    def initialize_model(self, model: nn.Module) -> nn.Module:
        if self.initialize_method == "xavier":
            return self.xavier_initialization(model)
        elif self.initialize_method == "kaiming":
            return self.kaiming_initialization(model)

    @staticmethod
    def xavier_initialization(model: nn.Module) -> nn.Module:
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        return model

    @staticmethod
    def kaiming_initialization(model: nn.Module) -> nn.Module:
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p, mode="fan_out")
        return model
