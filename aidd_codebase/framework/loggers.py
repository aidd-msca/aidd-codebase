import warnings
from typing import Dict, List, Optional

from pytorch_lightning.loggers import (
    MLFlowLogger,
    TensorBoardLogger,
    WandbLogger,
)

from aidd_codebase.utils.directories import Directories


class LoggerPL:
    pass


class PL_Loggers:
    loggers: Dict

    def __init__(
        self,
        directories: Directories,
        tensorboard: bool,
        wandb: bool,
        mlflow: bool,
    ) -> None:
        self.loggers = {}
        self.dirs = directories
        if tensorboard:
            self.init_tensorboard()
        if wandb:
            self.init_wandb()
        if mlflow:
            self.init_mlflow()

    def init_tensorboard(self) -> None:
        logger = TensorBoardLogger(
            save_dir=self.dirs.LOG_DIR, name=self.dirs.TIME
        )
        self.loggers["tensorboard"] = logger

    def init_wandb(self) -> None:
        logger = WandbLogger(
            save_dir=self.dirs.LOG_DIR,
            name=self.dirs.DATE,
            project=self.dirs.PROJECT,
        )
        self.loggers["wandb"] = logger

    def init_mlflow(self) -> None:
        logger = MLFlowLogger(
            experiment_name=self.dirs.PROJECT, tracking_uri="file:./ml-runs"
        )
        self.loggers["mlflow"] = logger

    def return_loggers(self) -> Optional[List]:
        if not self.loggers.values():
            warnings.warn("No loggers recorded")
            return None
        else:
            return list(self.loggers.values())
