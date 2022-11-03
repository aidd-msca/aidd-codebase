import torch.nn as nn

from aidd_codebase.utils.metacoding import DictChoiceFactory
from aidd_codebase.utils.typescripts import Tensor


class MLPChoice(DictChoiceFactory):
    pass


@MLPChoice.register_choice("mlp")
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dim_feedforward: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        add_skip_connections: bool = True,
    ) -> None:
        super().__init__()
        self.linear_net = nn.ModuleList(
            [nn.Linear(input_dim, dim_feedforward)]
        )
        self.linear_net.extend(
            [
                nn.Sequential(
                    nn.Linear(dim_feedforward, dim_feedforward),
                    nn.Dropout(dropout),
                    nn.ReLU(inplace=True),
                )
                for _ in range(1, num_layers - 1)
            ]
        )
        self.linear_net.append(nn.Linear(dim_feedforward, output_dim))

        def forward() -> Tensor:
            pass

        def residual_forward() -> Tensor:
            linear_out = self.linear_net(x)
            x = x + self.dropout(linear_out)
            x = self.norm2(x)
