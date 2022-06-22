import os
import warnings
from dataclasses import dataclass, field, fields, make_dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Tuple

import yaml
from aidd_codebase import __version__


@dataclass(unsafe_hash=True)
class _ABCDataClass:
    def __reduce__(self):
        state = self.__dict__.copy()
        return (
            _NestedClassGetter(),
            (
                _ABCDataClass,
                self.__class__.__name__,
            ),
            state,
        )


class _NestedClassGetter(object):
    """
    When called with the containing class as the first argument,
    and the name of the nested class as the second argument,
    returns an instance of the nested class.
    """

    def __call__(self, containing_class, class_name):
        nested_class = getattr(containing_class, class_name)

        # make an instance of a simple object (this one will do), for which we
        # can change the __class__ later on.
        nested_instance = _NestedClassGetter()

        # set the class of the instance, the __init__ will never be called on
        # the class but the original state will be set later on by pickle.
        nested_instance.__class__ = nested_class
        return nested_instance


@dataclass(unsafe_hash=True)
class Config:
    codebase_version: str = __version__
    _dataclasses: Dict[str, _ABCDataClass] = field(default_factory=lambda: {})

    def dataclass_config_override(
        self, data_classes: Dict[str, _ABCDataClass]
    ) -> None:
        for k, data_class in data_classes.items():
            self._dataclasses[k] = data_class  # original Dataclasses
            self.__dict__[k] = data_class.__dict__  # mutable objects

    def _read_config_yaml(self, file: str) -> Dict:
        with open(file) as file:
            yaml_config = yaml.safe_load(file)
        return yaml_config

    def yaml_config_override(self, file: str) -> None:
        if os.path.exists(file) and self._read_config_yaml(file):
            yaml_file = self._read_config_yaml(file)
            for argument, group in yaml_file.items():
                if not isinstance(group, Dict):
                    warnings.warn(f"{group} not in dictionary.")
                else:
                    if argument in self._dataclasses.keys():
                        dataclass_fields = [
                            field.name
                            for field in fields(self._dataclasses[argument])
                        ]
                        for k in group.keys():
                            if k not in dataclass_fields:
                                warnings.warn(f"Unknown key found: {k}.")

                        group = {
                            k: v
                            for k, v in group.items()
                            if k in dataclass_fields
                        }
                        self._dataclasses[argument] = self._dataclasses[
                            argument
                        ](**group)
                        self.__dict__[argument] = self._dataclasses[
                            argument
                        ].__dict__
                        for k, v in group.items():
                            if (
                                k
                                not in self._dataclasses[
                                    argument
                                ].__dict__.keys()
                            ):
                                warnings.warn(f"Unknown key found: {k}.")
                    else:
                        self.__dict__[argument] = group

    def print_dataclass_arguments(self, data_class: _ABCDataClass) -> None:
        for k, v in data_class.items():
            if not k.startswith("__"):
                print(f"Arg {k} = {v}")

    def print_arguments(self) -> None:
        """Prints all arguments in a dataclass."""
        for k, v in self.__dict__.items():
            if not isinstance(v, (MappingProxyType, Dict)):
                if not k.startswith("__"):
                    print(f"Arg {k} = {v}")
            else:
                self.print_dataclass_arguments(v)

    def return_dataclass(self, call_name: str) -> dataclass:
        def choose_field(key: str, field_value: Any) -> Tuple:
            if isinstance(field_value, (Dict, List)):
                return (
                    key,
                    "typing.Any",
                    field(default_factory=lambda: field_value),
                )
            elif field_value is not None:
                return (key, "typing.Any", field(default=field_value))
            else:
                return (key, "typing.Optional", None)

        dataclass_fields = [
            choose_field(key, field_value)
            for key, field_value in self.__dict__[call_name].items()
            if not key.startswith("__")
        ]
        new_dataclass = make_dataclass(call_name, dataclass_fields)
        return new_dataclass()
