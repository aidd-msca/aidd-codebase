import os
import pathlib
from typing import Dict, Optional

from aidd_codebase.utils.directories import validate_or_create_dir
from aidd_codebase.registries import AIDD
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


class ConfigChecker:
    def __init__(self, conf_path: Optional[str] = None):
        self.cs = ConfigStore.instance()
        self.conf_path = (
            conf_path if conf_path is not None else os.path.join(pathlib.Path(__file__).parent.absolute(), "conf")
        )
        self.add_defaults_config()

    def _check_file_exists(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist")

    def create_config(self, file_path: str, registry: str, key: str, key_dict: Dict) -> None:
        conf = AIDD.get_arguments(registry, key, key_dict)
        OmegaConf.save(config=conf, f=validate_or_create_dir(file_path))

    def import_config(self) -> None:

        for registry, item in AIDD.keys():
            for hash_key, key_dict in item.items():
                key, key_dict = hash_key
                key_dict_values = [value for value in key_dict.values()] if key_dict is not None else "default"
                filepath = os.path.join(self.conf_path, key, f"{''.join(key_dict_values)}.yaml")
                if not os.path.exists(filepath):
                    self.create_config(filepath, registry, key, key_dict)

                for subgroup, name in key_dict.items():
                    self._import_config(registry, item, subgroup, name)

            self._import_config(registry, item)

        filepath = os.path.join(self.conf_path, group, f"{name}.yaml")
        self._check_file_exists(filepath)
        self.cs.store(name=f"group={group}", node=OmegaConf.load(filepath))

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

    # def create_config(self, group: str, name: str) -> None:
    #     conf = AIDD.get_subclass_arguments(selection={f"{group}": name})
    #     OmegaConf.save(
    #         config=conf[f"{group}"],
    #         f=os.path.join(validate_or_create_dir(os.path.join(self.conf_path, group)), f"{name}.yaml"),
    #     )
