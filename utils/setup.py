import logging
from typing import Tuple

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError

from aidd_codebase.utils.device import Device


def setup(cfg: DictConfig, verbose: bool = True) -> Tuple[logging.Logger, Device]:
    logger = logging.getLogger(__name__)
    try:
        if cfg.debug:
            logger.setLevel(logging.DEBUG)
    except ConfigAttributeError:
        pass

    logging.info("Logging is set up")
    logging.debug(OmegaConf.to_yaml(cfg))
    
    logging.info("setting up device")
    device = Device(cfg.device)

    if verbose:
        print(OmegaConf.to_yaml(cfg))
        device.display()

    return logger, device
