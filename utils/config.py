import os
import warnings
from argparse import ArgumentParser
from ast import literal_eval
from collections import ChainMap
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from .tools import convert_dict_names_to_upper


@dataclass
class _ABCDataClass:
    pass


@dataclass
class _GeneratedDataClass:
    pass


@dataclass
class Config:
    FRAMEWORK_VERSION: str = "v001.000.001"  # v WORKING_IT.N_BRANCH.N_UPDATES

    def dataclass_config_override(
        self, DataClasses: List[_ABCDataClass]
    ) -> None:
        for Class in DataClasses:
            self.__dict__.update(Class.__dict__)

    def _read_config_yaml(self, file: str) -> Dict:
        with open(file) as file:
            yaml_config = yaml.safe_load(file)
        return yaml_config

    def yaml_config_override(self, file: str) -> None:
        if os.path.exists(file):
            self.__dict__.update(
                **dict(
                    ChainMap(
                        convert_dict_names_to_upper(
                            self._read_config_yaml(file)
                        )
                    )
                )
            )
        else:
            warnings.warn("No config file found.")

    def store_config_yaml(
        self, dir: Optional[str] = None, name: Optional[str] = None
    ) -> None:
        name = "config.yaml" if not name else name
        path = os.path.join(dir, name) if dir else name
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, sort_keys=False)

    def flag_config_override(self) -> None:
        """Parses all arguments from the input file that start with '--'"""

        parser = ArgumentParser()

        for k, v in self.__dict__.items():
            v_type = (
                literal_eval
                if type(v) == bool or type(v) == type(None)
                else type(v)
            )
            parser.add_argument(
                f"--{k}",
                help=k,
                default=v,
                type=v_type,
            )

        args, _ = parser.parse_known_args()

        self.__dict__.update(args.__dict__)

    def print_arguments(self) -> None:
        """Prints all arguments in a dataclass."""
        for k, v in self.__dict__.items():
            print(f"Arg {k} = {v}")

    def override_dataclasses(
        self, dataclasses: Dict[str, _ABCDataClass]
    ) -> Dict[str, _ABCDataClass]:
        output_dict = {}
        ticker = 0
        for key, value in self.__dict__.keys():
            if type(value) == dict:
                for name, dataclass in dataclasses:
                    if name.lower() == dataclass.__class__.__name__.lower():
                        output_dict[name] = dataclass(value)
                    else:
                        output_dict[f"gen_{ticker}"] = _GeneratedDataClass(
                            value
                        )
            else:
                pass

        def generate_dataclass(dict: Dict) -> _ABCDataClass:
            pass

        def init_arguments(self) -> None:
            pass


class TypeDict(dict):
    def __init__(self, t, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(t, type):
            raise TypeError("t must be a type")

        self._type = t

    @property
    def type(self):
        return self._type


class ConfigDataclassOverride:
    pass
