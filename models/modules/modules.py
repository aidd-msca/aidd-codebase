from typing import Optional

import torch.nn as nn

from aidd_codebase.utils.typescripts import Tensor


class WeightSharingEncoder(nn.TransformerEncoder):
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module],
    ) -> None:
        super().__init__(encoder_layer, num_layers=1, norm=norm)
        self.total_layers = num_layers  # prevent conflict with super()

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_paddin_mask: Optional[Tensor] = None,
    ):
        x = src
        for _ in range(self.total_layers):
            x = self.layers[0](
                x,
                mask,
                src_key_paddin_mask,
            )

        if self.norm:
            x = self.norm(x)

        return x


class WeightSharingDecoder(nn.TransformerDecoder):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module],
    ) -> None:
        super().__init__(decoder_layer, num_layers=1, norm=norm)
        self.total_layers = num_layers  # prevent conflict with super()

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        x = tgt
        for _ in range(self.total_layers):
            x = self.layers[0](
                x,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )

        if self.norm:
            x = self.norm(x)

        return x
