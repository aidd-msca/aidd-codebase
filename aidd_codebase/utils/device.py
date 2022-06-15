from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Device:
    device: str
    accelerator: Optional[str] = None
    precision: int = 32
    multi_gpu: bool = False
    gpus: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.device == "cpu" or not torch.cuda.is_available():
            self.torch_device = torch.device("cpu")
            self.accelerator = "cpu"
            self.precision = 32
            self.gpus = None
        elif self.device.startswith("cuda") and torch.cuda.is_available():
            self.torch_device = torch.device("cuda")
            self.accelerator = "gpu"

            if torch.cuda.device_count() == 1:
                self.multi_gpu = False

            if ":" in self.device:
                self.gpus = [
                    int(i) for i in self.device.split(":")[-1].split(",")
                ]
                if not self.multi_gpu:
                    self.gpus = [self.gpus[0]]
            else:
                self.gpus = [i for i in range(torch.cuda.current_device())]
                if not self.multi_gpu:
                    self.gpus = [0]

    def display(self) -> None:
        print(f"\nGPU is available: {torch.cuda.is_available()}")
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            with self.torch_device:
                print(
                    f"Using device {str(torch.cuda.current_device())}"
                    + f"/{str(torch.cuda.device_count())}, "
                    + f"name: {str(torch.cuda.get_device_name(0))}."
                )
