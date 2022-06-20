import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import pathlib

import pytorch_lightning as pl
import torch
from aidd_codebase.utils.device import Device
from aidd_codebase.datamodules.datachoice import DataChoice
from aidd_codebase.data_utils.tokenizer import Tokenizer
from aidd_codebase.framework.loggers import PL_Loggers
from aidd_codebase.utils.callbacks import CustomSavingCallback
from aidd_codebase.utils.config import Config
from aidd_codebase.utils.directories import Directories
from aidd_codebase.utils.initiator import ParameterInitialization
from aidd_codebase.models.modelchoice import ModelChoice
from tests.ESR1_tests.arguments import (
    EnvironmentArguments,
    TokenArguments,
)
import tests.ESR1_tests.datamodule
from tests.ESR1_tests.pl_frameworks import (
    TASK,
    LogitLoss,
    ModelFramework_old,
    TopNAccuracy,
)
from pytorch_lightning.callbacks import ModelCheckpoint



def main():
    """
    Directory structure:
    ESR
    |-- project
        |-- arguments.py
        |-- config.yaml
        |-- main.py
    """

    dataclasses = {
        "env": EnvironmentArguments(),
        "data": DataChoice.get_arguments("retrosynthesis_pavel"),
        "model": ModelChoice.get_arguments("pl_seq2seq"),
        "token": TokenArguments(),
    }

    config = Config()

    config.dataclass_config_override(dataclasses.values())
    # config.yaml_config_override(f"{config.HOME_DIR}/config.yaml")
    # config.store_config_yaml(config.HOME_DIR)
    config.print_arguments()

    # Set seed
    pl.seed_everything(config.SEED)

    # Set device
    device = Device(
        device=config.DEVICE,
        multi_gpu=config.MULTI_GPU,
        precision=config.PRECISION,
    )
    device.display()

    # Load data
    tokenizer = Tokenizer(
        vocab=config.VOCAB,
        pad_idx=config.PAD_IDX,
        bos_idx=config.BOS_IDX,
        eos_idx=config.EOS_IDX,
        max_seq_len=250, #config.MAX_SEQ_LEN,
    )

    datamodule = DataChoice.get_choice("retrosynthesis_pavel")
    datamodule = datamodule(tokenizer=tokenizer, data_args = dataclasses["data"])
    
    model = ModelChoice.get_choice("pl_seq2seq")
    model = model(model_args = dataclasses["model"])
    param_init = ParameterInitialization(method="xavier")
    model = param_init.initialize_model(model)

    if config.LOAD_PREVIOUS:
        model = torch.load(config.MODEL_LOAD_PATH)

    framework = ModelFramework_old(
        task=TASK.TRANSLATION,
        optimizer=torch.optim.Adam(
            model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
        ),
        metrics={
            "loss": LogitLoss(
                torch.nn.CrossEntropyLoss(
                    reduction="mean", ignore_index=config.PAD_IDX
                )
            ),
            "accuracy": TopNAccuracy(1),
            "accuracy_top3": TopNAccuracy(3),
        },
        lr=0.0001,
    )
    framework.set_model(model)

    directories = Directories(PROJECT=config.NAME, HOME_DIR=config.HOME_DIR)
    pl_loggers = PL_Loggers(
        directories=directories,
        tensorboard=config.LOG_TENSORBOARD,
        wandb=config.LOG_WANDB,
        mlflow=config.LOG_MLFLOW,
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

    saving_callback = CustomSavingCallback(
        model_save_path=directories.MODEL_DIR
    )

    trainer = pl.Trainer(
        fast_dev_run=True,
        max_epochs=config.NUM_EPOCHS + 1,
        accelerator="gpu",
        gpus=[0],
        precision=16,
        # auto_scale_batch_size="binsearch",  # doesn't work yet
        logger=loggers,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, saving_callback],
        weights_save_path=directories.WEIGHTS_DIR,
    )

    # tune the trainer parameters
    if trainer.auto_scale_batch_size is not None:
        trainer.tune(framework)

    # train the model
    trainer.fit(framework, datamodule)

    # test model
    trainer.test(model=framework, datamodule=datamodule)


# Example: python main.py --DEVICE cuda:0 --NAME my_run
# or modify and run: bash run.sh
if __name__ == "__main__":
    main()
