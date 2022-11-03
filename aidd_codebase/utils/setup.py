import logging
from typing import Tuple

import pytorch_lightning as pl
from aidd_codebase.utils.device import Device
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError


def setup(cfg: DictConfig, verbose: bool = True) -> Tuple[logging.Logger, Device]:
    logger = logging.getLogger(__name__)
    try:
        if cfg.debug:
            logger.setLevel(logging.DEBUG)
    except ConfigAttributeError:
        pass

    logging.info("Logging is set up")
    logging.debug(OmegaConf.to_yaml(cfg))

    logging.info("setting seed")
    pl.seed_everything(cfg.seed)

    logging.info("setting up device")
    device = Device(device=cfg.device, multi_gpu=cfg.multi_gpu, precision=cfg.precision,)

    if verbose:
        print(OmegaConf.to_yaml(cfg))
        device.display()

    return logger, device
