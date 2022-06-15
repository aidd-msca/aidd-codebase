from typing import List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from aidd_codebase.utils.typescripts import Tensor


class Collate:
    def __init__(self, pad_idx: Optional[int]) -> None:
        self.pad_idx = pad_idx

    def seq_collate_fn(
        self, batch: Tuple[List[Tensor], List[Tensor]]
    ) -> Tuple[Tensor, Tensor]:
        """Function to collate data samples into batch tensors"""
        src_batch, tgt_batch = [], []
        for src, tgt in batch:
            src_sample = src
            tgt_sample = tgt

            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)

        src_batch = pad_sequence(src_batch, padding_value=self.pad_idx)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.pad_idx)
        return src_batch, tgt_batch

    def simple_collate_fn(
        self, batch: Tuple[List[Tensor], List[Tensor]]
    ) -> Tuple[Tensor, Tensor]:
        """Stacks input tensors into source and target tensors.
        Assumes all tensors of equal length.
        """
        src_batch, tgt_batch = zip(*batch)
        return torch.t(torch.stack(src_batch)), torch.t(torch.stack(tgt_batch))
