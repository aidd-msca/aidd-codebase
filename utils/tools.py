import functools
from typing import Callable, Dict


def compose(*functions: Callable) -> Callable:
    """Helper function to compose together sequential operations."""
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


def convert_dict_names_to_upper(dict: Dict) -> Dict:
    return {k.upper(): v for k, v in dict.items()}
