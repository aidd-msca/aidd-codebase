import torch.nn as nn

from aidd_codebase.utils.metacoding import DictChoiceFactory


class AdapterChoice(DictChoiceFactory):
    pass


# class AdapterABC(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(
#         self, x: Tensor, mask: Tensor = None
#     ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
#         raise NotImplementedError


class InputAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        raise NotImplementedError

    @property
    def positional_encoding(self):
        raise NotImplementedError


class SequenceAdapter(InputAdapter):
    # TODO
    def __init__(self) -> None:
        super().__init__()

    def positional_encoding(self):
        return super().positional_encoding()  # identical? think not

    def forward(self):
        pass


class ImageAdapter(InputAdapter):
    # TODO
    def __init__(self) -> None:
        super().__init__()

    def positional_encoding(self):
        return super().positional_encoding()

    def forward(self):
        pass


class GraphAdapter(InputAdapter):
    # TODO
    def __init__(self) -> None:
        super().__init__()

    def positional_encoding(self):
        return super().positional_encoding()

    def forward(self):
        pass
