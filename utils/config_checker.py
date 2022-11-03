import os
import pathlib

from abstract_codebase.directories import validate_or_create_dir
from abstract_codebase.registration import RegistryFactory
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


class ConfigChecker:
    def __init__(self):
        self.cs = ConfigStore.instance()
        self.conf_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "conf")
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
                    filepath = os.path.join(self.conf_path, key, f"{value}.yaml")
                    if not os.path.exists(filepath):
                        self.create_config(key, value)

    def create_config(self, group: str, name: str) -> None:
        OmegaConf.save(
            config=RegistryFactory.get_arguments(name),
            f=os.path.join(validate_or_create_dir(os.path.join(self.conf_path, group)), f"{name}.yaml"),
        )
