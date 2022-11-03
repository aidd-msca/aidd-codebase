import hydra
from omegaconf import DictConfig
from aidd_codebase.utils.setup import setup
from aidd_codebase.utils.config_checker import ConfigChecker

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger, device = setup(cfg)
    # your code here

if __name__ == "__main__":
    ConfigChecker()
    main()
