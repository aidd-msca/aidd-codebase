import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pathlib

import torch
import pytorch_lightning as pl

from ESR1.adapted_swiss_knife.seq2seq import main
from example_ESR.project1.arguments import EnvironmentArguments, DataArguments, ModelArguments
from coding_framework.utils.config import Config

def main():
    """
    Directory structure:
    ESR
    |-- project
      |-- data
        |-- ...
      |-- arguments.py
      |-- config.yaml
      |-- main.py
      |-- datamodule.py
      |-- pl_framework.py
    """

    dataclasses = {
        "env": EnvironmentArguments(),
        "data": DataArguments(),
        "model": ModelArguments(),
    }
    
    if not dataclasses["data"].HOME_DIR:
       dataclasses["data"].HOME_DIR = str(pathlib.Path(__file__).parent.resolve())
    dataclasses["token"].calc_dependents()
    
    config = Config()
    
    config.dataclass_config_override(dataclasses.values())
    config.yaml_config_override(f"{config.HOME_DIR}/config.yaml")
    config.flag_config_override()
    config.store_config_yaml(config.HOME_DIR)
    config.print_arguments()

    pl.seed_everything(config.SEED)

    DEVICE = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nGPU is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        with torch.cuda.device(DEVICE):
            print(
                f"Using device {str(torch.cuda.current_device())}"
                + f"/{str(torch.cuda.device_count())}, "
                + f"name: {str(torch.cuda.get_device_name(0))}."
            )
    
    ...


# Example: python ESR1/main.py --DEVICE cuda:0 --NAME my_run
# or modify and run: bash run.sh
if __name__ == "__main__":
    main()