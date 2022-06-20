from aidd_codebase import __version__
import torch
import pytorch_lightning as pl
from dataclasses import dataclass

from aidd_codebase.utils.config import Config, _ABCDataClass
from aidd_codebase.utils.device import Device
from aidd_codebase.datamodules.datachoice import DataChoice
from aidd_codebase.models.modelchoice import ModelChoice

from aidd_codebase.framework.framework import ModelFramework
from aidd_codebase.models.modules.loss import LossChoice
from aidd_codebase.models.optimizers.optimizers import OptimizerChoice
from aidd_codebase.utils.initiator import ParameterInitialization


@dataclass
class EnvironmentArguments(_ABCDataClass):
    DEBUG: bool = False
    STRICT: bool = True

    SEED: int = 1234
    PORT: int = 6006

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    LOG_TENSORBOARD: bool = True
    LOG_WANDB: bool = False
    LOG_MLFLOW: bool = False


def test_version():
    assert __version__ == '0.1.0'

    args = {
        "env": EnvironmentArguments(),
        "data": DataChoice.get_arguments('smiles'),
        "model": ModelChoice.get_arguments('pl_seq2seq')
    }

    args['data'].HOME_DIR = ''
    args['data'].DATA_PATH = '/home/emma/data/PavelsReactions/raw/retrosynthesis-all.smi'

    config = Config()

    config.dataclass_config_override(args.values())
    config.yaml_config_override(f"{args['data'].HOME_DIR}/config.yaml")
    config.store_config_yaml(args['data'].HOME_DIR)
    config.print_arguments()

    # Set seed
    pl.seed_everything(args['env'].SEED)

    # Init SMILES data
    data_choice = DataChoice.get_choice('smiles')
    datamodule = data_choice(args['data'])

    # Init Seq2seq model
    model_choice = ModelChoice.get_choice('pl_seq2seq')
    model = model_choice(args['model'])

    param_init = ParameterInitialization(method="xavier")
    model = param_init.initialize_model(model)

    loss = LossChoice.get_choice("ce_loss")
    optimizer = OptimizerChoice.get_choice("adam")

