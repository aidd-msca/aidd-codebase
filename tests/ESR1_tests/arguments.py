import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from aidd_codebase.utils.config import _ABCDataClass


@dataclass
class EnvironmentArguments(_ABCDataClass):
    DEBUG: bool = False
    STRICT: bool = True
    HOME_DIR: Optional[str] = str(pathlib.Path(__file__).parent.resolve())


    SEED: int = 1234
    PORT: int = 6006

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MULTI_GPU: bool = False
    PRECISION: Optional[int] = 16

    LOG_TENSORBOARD: bool = True
    LOG_WANDB: bool = False
    LOG_MLFLOW: bool = False


@dataclass
class DataArguments(_ABCDataClass):
    NUM_WORKERS: int = 8
    PERSISTENT_WORKERS: bool = True
    REDO_DATA_PROCESSING: bool = False

    HOME_DIR: Optional[str] = str(pathlib.Path(__file__).parent.resolve())
    DATA_DIR: Optional[str] = f"{pathlib.Path(__file__).parent.resolve()}/data"
    DATA_LOAD_DIR: Optional[
        str
    ] = f"{pathlib.Path(__file__).parent.resolve()}/data/saved"

    REMOVE_MISSING: bool = True
    REMOVE_DUPLICATES: bool = False
    CANONICALIZATION: bool = True
    ENUMERATION: int = 0  # 10
    ENUM_OVERSAMPLE: int = 15


@dataclass
class ModelArguments(_ABCDataClass):
    NAME: str = "model"

    NUM_EPOCHS: int = 100
    BATCH_SIZE: int = 128
    LOAD_PREVIOUS: bool = False
    MODEL_LOAD_PATH: Optional[str] = None


@dataclass
class Seq2SeqArguments(ModelArguments):
    NAME: str = "seq2seq"

    SHARE_WEIGHT: bool = False

    EMB_SIZE: int = 512

    NHEAD: int = 8
    DROPOUT: float = 0.1
    FFN_HID_DIM: int = 512
    NUM_ENCODER_LAYERS: int = 3
    NUM_DECODER_LAYERS: int = 3


@dataclass
class TransformerCNNArguments(ModelArguments):
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
