import os
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
from aidd_codebase.utils.directories import Directories
from aidd_codebase.framework.loggers import PL_Loggers
from aidd_codebase.utils.callbacks import CustomSavingCallback


@dataclass
class EnvironmentArguments(_ABCDataClass):
    debug: bool = False
    strict: bool = True

    seed: int = 1234
    port: int = 6006

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    name = 'Codebase Example'
    log_tensorboard: bool = True
    log_wandb: bool = False
    log_mlflow: bool = False

    num_epochs = 100


def main():
    args = {
        "env": EnvironmentArguments(),
        "data": DataChoice.get_arguments('smiles'),
        "model": ModelChoice.get_arguments('pl_seq2seq')
    }

    args['data'].home_path = ''
    args['data'].data_path = '/home/emma/data/PavelsReactions/raw/retrosynthesis-all.smi'

    config = Config()

    config.dataclass_config_override(args.values())
    #config.yaml_config_override(os.path.join(args['data'].HOME_DIR, '/config.yaml'))
    #config.store_config_yaml(args['data'].HOME_DIR)
    config.print_arguments()

    device = Device(device=args['env'].device)
    device.display()

    # Set seed
    pl.seed_everything(args['env'].seed)

    # Init SMILES data
    data_choice = DataChoice.get_choice('smiles')
    datamodule = data_choice(args['data'])

    # Init Seq2seq model
    model_choice = ModelChoice.get_choice('pl_seq2seq')
    model = model_choice(args['model'])
    param_init = ParameterInitialization(method="xavier")
    model = param_init.initialize_model(model)

    loss = LossChoice.get_choice("cross_entropy")
    optimizer = OptimizerChoice.get_choice("adam")

    framework = ModelFramework(
        loss=loss,
        metrics=None,
        optimizer=optimizer,
        scheduler=None,
        loggers=None,
    )
    framework.set_model(model)

    directories = Directories(PROJECT=args['env'].name, HOME_DIR=args['data'].home_path)
    pl_loggers = PL_Loggers(
        directories=directories,
        tensorboard=args['env'].log_tensorboard,
        wandb=args['env'].log_wandb,
        mlflow=args['env'].log_mlflow,
    )
    loggers = pl_loggers.return_loggers()

    # saves a file like:
    # .../runs/name/checkpoints/date/time/model-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=directories.CHECKPOINT_DIR,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=-1,
        mode="min",
        auto_insert_metric_name=True,
        every_n_epochs=1,
    )

    saving_callback = CustomSavingCallback(
        model_save_path=directories.MODEL_DIR
    )

    trainer = pl.Trainer(
        fast_dev_run=args['env'].debug,
        max_epochs=args['env'].num_epochs + 1,
        accelerator="gpu",
        gpus=[0],
        precision=16,
        logger=loggers,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, saving_callback],
        weights_save_path=directories.WEIGHTS_DIR,
    )

    trainer.fit(model=framework, datamodule=datamodule)

    trainer.test(model=framework, datamodule=datamodule)


if __name__ == "__main__":
    main()

