import pickle
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn


def save_model(
    model: nn.Module, name: str, path: str, save_weights: bool = True
) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(model, f"{path}/{name}.pth")
    if save_weights:
        torch.save(model.state_dict(), f"{path}/{name}_weights.pth")


def load_model_weights(model: nn.Module, weight_path: str) -> nn.Module:
    return model.load_state_dict(torch.load(weight_path))


def load_model(path: str) -> nn.Module:
    return torch.load(path)


def checkpoint_load(
    object: pl.LightningModule, checkpoint_path: str
) -> pl.LightningModule:
    return object.load_from_checkpoint(checkpoint_path)


def pickle_save(file_path: str, object: Any):
    with open(file_path, "wb") as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path: str) -> Any:
    with open(file_path, "rb") as handle:
        object = pickle.load(handle)
    return object
