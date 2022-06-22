import functools
from typing import Any, Callable, Dict


def compose(*functions: Callable) -> Callable:
    """Helper function to compose together sequential operations."""
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


def convert_dict_names_to_lower(dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k.lower(): v for k, v in dict.items()}


def convert_dict_names_to_upper(dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k.upper(): v for k, v in dict.items()}
