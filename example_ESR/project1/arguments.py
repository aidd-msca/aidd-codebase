import pathlib
from dataclasses import dataclass
from typing import Optional
from coding_framework.utils.config import _ABCDataClass

import torch



@dataclass
class EnvironmentArguments(_ABCDataClass):
    DEBUG: bool = False
    STRICT: bool = True

    SEED: int = 1234
    PORT: int = 6006

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    

@dataclass
class DataArguments(_ABCDataClass):
    NUM_WORKERS: int = 8
    PERSISTENT_WORKERS: bool = True
    REDO_DATA_PROCESSING: bool = False

    HOME_DIR: Optional[str] = str(pathlib.Path(__file__).parent.resolve())
    DATA_DIR: Optional[str] = f"{pathlib.Path(__file__).parent.resolve()}/data"
    

@dataclass
class ModelArguments(_ABCDataClass):
    NAME: str = "model"

    NUM_EPOCHS: int = 100
    BATCH_SIZE: int = 128
    LOAD_PREVIOUS: bool = False
    MODEL_LOAD_PATH: Optional[str] = None
