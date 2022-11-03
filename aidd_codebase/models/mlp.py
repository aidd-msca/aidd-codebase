from typing import Dict
from abstract_codebase.accreditation import CreditInfo, CreditType
import torch.nn as nn
from dataclasses import dataclass
from aidd_codebase.registries import ModelRegistry
from aidd_codebase.utils.typescripts import Tensor

@ModelRegistry.register_arguments(call_name="3_layer_mlp")
@dataclass
class MLPArguments:
    input_size: int = 1024
    hidden_size: int = 512
    dropout_rate: float = 0.80
    output_size: int = 1
    learning_rate: float = 0.001


@ModelRegistry.register(
    key="3_layer_mlp",
    credit=CreditInfo(
    author="Peter Hartog",
    github_handle="PeterHartog",),
    credit_type=CreditType.NONE,
)
class MLP(nn.Module):
    def __init__(self, model_args: Dict) -> None:
        super().__init__()
        
        args = MLPArguments(**model_args)
        
        # Three layers and a output layer
        self.fc1 = nn.Linear(args.input_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc_out = nn.Linear(args.hidden_size, args.output_size)
        
        # Layer normalization for faster training
        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.ln2 = nn.LayerNorm(args.hidden_size)
        self.ln3 = nn.LayerNorm(args.hidden_size)    
            
        # ReLU will be used as the activation function
        self.activation = nn.ReLU()
        
        #Dropout for regularization
        self.dropout = nn.Dropout(args.dropout_rate)
        
        self.model = nn.Sequential(
            self.fc1,
            self.ln1,
            self.activation,
            self.dropout,
            self.fc2,
            self.ln2,
            self.activation,
            self.dropout,
            self.fc3,
            self.ln3,
            self.activation,
            self.dropout,
        )
        
     
    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        return self.fc_out(out)