import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

import aidd_codebase.models.seq2seq

# import pytorch_lightning as pl
import torch

# from aidd_codebase.datamodules.datachoice import DataChoice
from aidd_codebase.models.modelchoice import ModelChoice
from aidd_codebase.utils.config import Config, _ABCDataClass

# from aidd_codebase.utils.device import Device


@dataclass
class EnvironmentArguments(_ABCDataClass):
    DEBUG: bool = False
    STRICT: bool = True

    SEED: int = 1234
    PORT: int = 6006
    HOME_DIR: Optional[str] = str(pathlib.Path(__file__).parent.resolve())

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    LOG_TENSORBOARD: bool = True
    LOG_WANDB: bool = False
    LOG_MLFLOW: bool = False


@dataclass
class DataArguments(_ABCDataClass):
    NUM_WORKERS: int = 8
    PERSISTENT_WORKERS: bool = True
    REDO_DATA_PROCESSING: bool = False

    DATA_DIR: Optional[str] = f"{pathlib.Path(__file__).parent.resolve()}/data"
    DATA_LOAD_DIR: Optional[
        str
    ] = f"{pathlib.Path(__file__).parent.resolve()}/data/saved"

    REMOVE_MISSING: bool = True
    REMOVE_DUPLICATES: bool = True
    CANONICALIZATION: bool = True
    ENUMERATION: int = 10
    ENUM_OVERSAMPLE: int = 15


@dataclass
class TrainerArguments(_ABCDataClass):
    NAME: str = "run_x"

    NUM_EPOCHS: int = 100
    BATCH_SIZE: int = 128
    LOAD_PREVIOUS: bool = False
    MODEL_LOAD_PATH: Optional[str] = None


@dataclass
class TransformerCNNArguments(_ABCDataClass):
    NAME: str = "transformer_cnn"

    SHARE_WEIGHT: bool = False
    FREEZE_ENCODER: bool = True

    EMB_SIZE: int = 512

    NHEAD: int = 8
    DROPOUT: float = 0.1
    FFN_HID_DIM: int = 512
    NUM_ENCODER_LAYERS: int = 3

    KERNEL_SIZES: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    )
    FILTERS: List[int] = field(
        default_factory=lambda: [
            100,
            200,
            200,
            200,
            200,
            100,
            100,
            100,
            100,
            100,
            160,
            160,
        ]
    )
    STRIDE: int = 1
    NUM_CLASSES: int = 1


@dataclass
class TokenArguments(_ABCDataClass):
    # Define special symbols and indices
    PAD_IDX: int = 0  # Padding
    BOS_IDX: int = 1  # Beginning of Sequence
    EOS_IDX: int = 2  # End of Sequence
    UNK_IDX: int = 3  # Unknown Value
    MSK_IDX: Optional[int] = None  # Mask

    # Our vocabulary
    VOCAB: str = (
        " ^$?#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy"
    )
    MAX_SEQ_LEN: int = 110


def main(model="model_x"):  # , data="data_x"):
    dataclasses = {
        # "env_args": EnvironmentChoice.get_arguments(env),
        # "data_args": DataChoice.get_arguments(data),
        "model_args": ModelChoice.get_arguments(model),
    }

    config = Config()

    config.dataclass_config_override(dataclasses.values())
    # config.yaml_config_override(f"{config.HOME_DIR}/config.yaml")
    # config.store_config_yaml(config.HOME_DIR)
    config.print_arguments()

    # env_args, data_args, model_args = config.return_dataclasses()
    # config.environment

    # Set device
    # device = Device(
    #     device=env_args.DEVICE,
    #     multi_gpu=env_args.MULTI_GPU,
    #     precision=env_args.PRECISION,
    # )
    # device.display()

    # # Set seed
    # pl.seed_everything(env_args.SEED)

    # datamodule = DataChoice.get_choice(data)
    # datamodule = datamodule(data_args)

    model = ModelChoice.get_choice(model)
    model = model(dataclasses["model_args"])

    accreditations = ModelChoice.view_accreditations()
    print(accreditations)


if __name__ == "__main__":
    main("pl_seq2seq")
