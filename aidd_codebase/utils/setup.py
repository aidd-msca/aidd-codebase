import logging
from typing import Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from aidd_codebase.utils.device import Device
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError


def setup(cfg: DictConfig, verbose: bool = True) -> Tuple[logging.Logger, Optional[WandbLogger], Device]:
    logger = logging.getLogger(__name__)
    try:
        if cfg.debug:
            logger.setLevel(logging.DEBUG)
    except ConfigAttributeError:
        pass

    logging.info("Logging is set up")
    logging.debug(OmegaConf.to_yaml(cfg))

    logging.info("setting seed")
    pl.seed_everything(cfg.random_state)

    logging.info("setting up device")
    device = Device(
        device=cfg.device,
        multi_gpu=cfg.multi_gpu,
        precision=cfg.precision,
        gpus=cfg.gpus,
    )

    pl_loggers = None
    if "wandb" in cfg.logger:
        pl_loggers = WandbLogger(
            name=cfg.run_name,
            project=cfg.name,
            reinit=True,
        )

    if verbose:
        print(OmegaConf.to_yaml(cfg))
        device.display()

    if cfg.config_setup:
        raise Exception("Config setup done!")

    return logger, pl_loggers, device
