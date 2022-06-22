import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import pathlib
from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
import tests.ESR1_tests.datamodule
import torch
from aidd_codebase.datamodules.datachoice import DataChoice
from aidd_codebase.framework.framework import ModelFramework
from aidd_codebase.framework.loggers import PL_Loggers
from aidd_codebase.framework.loopchoice import LoopChoice
from aidd_codebase.models.metrics.loss import LossChoice
from aidd_codebase.models.modelchoice import ModelChoice
from aidd_codebase.models.optimizers.optimizers import OptimizerChoice
from aidd_codebase.utils.config import Config, _ABCDataClass
from aidd_codebase.utils.device import Device
from aidd_codebase.utils.directories import Directories
from aidd_codebase.utils.initiator import ParameterInitialization
from aidd_codebase.utils.metacoding import DictChoiceFactory
from pytorch_lightning.callbacks import ModelCheckpoint
from tests.ESR1_tests.pl_frameworks import LogitLoss

# TODO refactor tokenizer into object with class variables


class EnvironmentChoice(DictChoiceFactory):
    pass


@EnvironmentChoice.register_arguments(call_name="ESR1_example")
@dataclass(unsafe_hash=True)
class EnvironmentArguments(_ABCDataClass):
    NAME: str
    DEBUG: bool = True
    HOME_DIR: Optional[str] = str(pathlib.Path(__file__).parent.resolve())

    NUM_EPOCHS: int = 100
    BATCH_SIZE: int = 128
    LOAD_PREVIOUS: bool = False
    MODEL_LOAD_PATH: Optional[str] = None

    SEED: int = 1234
    PORT: int = 6006

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MULTI_GPU: bool = False
    PRECISION: Optional[int] = 16

    LOG_TENSORBOARD: bool = True
    LOG_WANDB: bool = False
    LOG_MLFLOW: bool = False


def main():
    dataclasses = {
        "env": EnvironmentChoice.get_arguments("ESR1_example"),
        "token": DataChoice.get_arguments("sequence_tokenizer"),
        "data": DataChoice.get_arguments("retrosynthesis_pavel"),
        "model": ModelChoice.get_arguments("pl_seq2seq"),
    }

    config = Config()
    config.dataclass_config_override(data_classes=dataclasses)
    config.yaml_config_override(f"{config.env['HOME_DIR']}/config.yaml")
    config.print_arguments()

    env_args = config.return_dataclass("env")
    token_args = config.return_dataclass("token")
    data_args = config.return_dataclass("data")
    model_args = config.return_dataclass("model")

    # Set seed
    pl.seed_everything(env_args.SEED)

    # Set device
    device = Device(
        device=env_args.DEVICE,
        multi_gpu=env_args.MULTI_GPU,
        precision=env_args.PRECISION,
    )
    device.display()

    # Load data
    tokenizer = DataChoice.get_choice("sequence_tokenizer")
    tokenizer = tokenizer(token_args)

    datamodule = DataChoice.get_choice("retrosynthesis_pavel")
    datamodule = datamodule(tokenizer=tokenizer, data_args=data_args)

    # Load Model
    model = ModelChoice.get_choice("pl_seq2seq")
    model = model(model_args=model_args)

    param_init = ParameterInitialization(method="xavier")
    model = param_init.initialize_model(model)

    # Setup Framework
    loss = LossChoice.get_choice("cross_entropy")
    loss = LogitLoss(loss(reduction="mean", ignore_index=tokenizer.pad_idx))
    optimizer = OptimizerChoice.get_choice("adam")
    optimizer = optimizer(
        model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
    framework = ModelFramework(
        loss=loss,
        metrics=None,
        optimizer=optimizer,
        scheduler=None,
        loggers=None,
    )
    framework.set_model(model)

    loop = LoopChoice.get_choice("pl_seq2seq_loops")
    framework.set_loop(loop.translation)

    directories = Directories(
        PROJECT=env_args.NAME, HOME_DIR=env_args.HOME_DIR
    )
    pl_loggers = PL_Loggers(
        directories=directories,
        tensorboard=env_args.LOG_TENSORBOARD,
        wandb=env_args.LOG_WANDB,
        mlflow=env_args.LOG_MLFLOW,
    )
    loggers = pl_loggers.return_loggers()

    # saves a file like:
    # .../runs/name/checkpoints/date/time/model-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=directories.CHECKPOINT_DIR,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=-1,
        mode="min",
        auto_insert_metric_name=True,
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        fast_dev_run=env_args.DEBUG,
        max_epochs=env_args.NUM_EPOCHS + 1,
        accelerator=device.accelerator,
        gpus=device.gpus,
        precision=device.precision,
        logger=loggers,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        weights_save_path=directories.WEIGHTS_DIR,
    )

    # train the model
    trainer.fit(framework, datamodule)

    # test model
    trainer.test(model=framework, datamodule=datamodule)

    # get accreditation
    DictChoiceFactory.view_accreditations()


# Example: python main.py --DEVICE cuda:0 --NAME my_run
# or modify and run: bash run.sh
if __name__ == "__main__":
    main()
