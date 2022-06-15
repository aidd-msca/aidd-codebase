import warnings
from functools import update_wrapper

import torch.nn as nn

from aidd_codebase.utils.typescripts import Tensor


class ResidualConnection:
    """Adds skip connections between layers."""

    module: nn.Module

    def __new__(cls, sequential: nn.Module):
        instance = super().__new__(cls)
        instance.module = sequential

        func = getattr(sequential, "forward")

        wrapped = ResidualConnection._residual_wrap(func)
        setattr(instance, "forward", wrapped)

        return instance

    def _residual_wrap(func):
        """
        Wraps *func* with additional code.
        """

        def wrapped(*args, **kwargs):
            if "residual" in kwargs:
                residual = kwargs.get("residual")

            output = func(*args, **kwargs)

            if type(output) == Tensor or (
                type(output) == tuple and type(output[0]) == Tensor
            ):
                output = (
                    residual + output
                    if type(output) != tuple
                    else residual + output[0]
                )
            else:
                warnings.warn("couldn't add residual connection")
            return output

        # Use "update_wrapper" to keep docstrings and other function metadata
        # intact
        update_wrapper(wrapped, func)

        return wrapped
