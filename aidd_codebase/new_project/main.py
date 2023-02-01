import hydra
from omegaconf import DictConfig
from aidd_codebase.utils.setup import setup
from aidd_codebase.utils.config_checker import ConfigChecker
from aidd_codebase.registries import AIDD
import pytorch_lightning as pl

# don't forget to add the imports to the __init__.py file if you wish to use them in the main.py file


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger, device = setup(cfg)

    # your code here
    model = None
    datamodule = None

    # set up the trainer
    trainer = pl.Trainer(
        fast_dev_run=cfg.debug,
        max_epochs=cfg.epochs,
        accelerator=device.accelerator,
        gpus=device.gpus,
        precision=device.precision,
        logger=None,
        log_every_n_steps=1,
        callbacks=None,
        weights_save_path=None,
    )

    # train the model
    trainer.fit(model=model, datamodule=datamodule)

    # test the model
    trainer.test(model=model, datamodule=datamodule)

    # get accreditation
    logger.info("Getting accreditation")
    AIDD.view_accreditations()


if __name__ == "__main__":
    ConfigChecker()
    main()
