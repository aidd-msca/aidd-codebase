import os
import pathlib

from aidd_codebase.utils.directories import validate_or_create_dir
from aidd_codebase.registries import AIDD
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


class ConfigChecker:
    def __init__(self, conf_path: str = os.path.join(pathlib.Path(__file__).parent.resolve(), "conf")):
        self.cs = ConfigStore.instance()
        self.conf_path = conf_path
        self.add_defaults_config()

    def _check_file_exists(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist")

    def add_defaults_config(self):
        """Check if the default yaml files from config exist."""
        conf = OmegaConf.load(f"{self.conf_path}/config.yaml")
        for default in conf.defaults:
            if default != "_self_":
                for key, value in default.items():
                    if not key.startswith("override"):
                        filepath = os.path.join(self.conf_path, key, f"{value}.yaml")
                        if not os.path.exists(filepath):
                            self.create_config(key, value)

    def create_config(self, group: str, name: str) -> None:
        conf = AIDD.get_subclass_arguments(selection={f"{group}registry": name})
        OmegaConf.save(
            config=conf[f"{group}"],
            f=os.path.join(validate_or_create_dir(os.path.join(self.conf_path, group)), f"{name}.yaml"),
        )
