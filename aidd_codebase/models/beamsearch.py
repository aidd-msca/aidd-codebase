from dataclasses import dataclass
from typing import Any

import torch
from aidd_codebase.models.modelchoice import ModelChoice
from aidd_codebase.utils.config import _ABCDataClass
from aidd_codebase.utils.metacoding import CreditType
from aidd_codebase.utils.typescripts import Tensor


@ModelChoice.register_arguments(call_name="beamsearch")
@dataclass(unsafe_hash=True)
class BeamSearchArguments(_ABCDataClass):
    decoder: Any
    fc_out: Any
    max_seq_len: int = 201
    device: str = "cuda:0"
    pad_idx: int = 0
    bos_idx: int = 1
    eos_idx: int = 2
    unk_idx: int = 3


@ModelChoice.register_choice(
    call_name="beamsearch",
    author="Paula Torren Peraire",
    github_handle="PTPeraire",
    credit_type=CreditType.ACKNOWLEDGEMENT,
)
class BeamSearch:
    def __init__(
        self,
        decoder: Any,
        fc_out: Any,
        max_seq_len: int = 201,
        device: str = "cuda:0",
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
        unk_idx: int = 3,
    ) -> None:

        self.decoder = decoder
        self.fc_out = fc_out
        self.pad_idx = pad_idx
        self.device = device
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx
        self.max_seq_len = max_seq_len

        self.soft_max = torch.nn.Softmax(dim=-1)

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf with zeros on diag."""
        return torch.triu(
            torch.ones(sz, sz, device=self.device) * float("-inf"), diagonal=1
        )

    def create_tgt_mask(self, tgt: Tensor) -> Tensor:
        return self.generate_square_subsequent_mask(tgt.shape[0])

    def create_eos_mask(self, mask: Tensor, dim: int = 0) -> Tensor:
        mask = torch.any(mask, dim=dim).to(self.device)
        return mask

    def create_padding_mask(self, tgt: Tensor) -> Tensor:
        mask = torch.where(
            (tgt == self.pad_idx) | (tgt == self.eos_idx), True, False
        )
        return mask.transpose(0, 1)

    def _step(self, memory, tgt, tgt_mask, tgt_padding_mask):
        x = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=None,
        )
        logits = self.fc_out(x)
        output = self.soft_max(logits)
        output = torch.log10(output).to(self.device)
        return output

    def reorder_scores(self, interim_score_, interim_idx_):
        topk = torch.topk(interim_score_, self.k)
        topk_idx_ = topk.indices.unsqueeze(-1).expand(
            [topk.indices.shape[0], self.k, 2]
        )
        topk_idx_ = torch.gather(interim_idx_, 1, topk_idx_)
        return topk.values, topk_idx_

    def _predict_k(self, ll, eos_padding_mask, k_):
        topk = torch.topk(ll, self.k)
        k_idx_ = topk.indices
        k_idx_[eos_padding_mask] = self.pad_idx
        k_idx = torch.full(topk.indices.shape, k_).to(self.device)
        k_idx = torch.stack([k_idx_, k_idx], dim=-1)
        k_score = topk.values
        k_score[eos_padding_mask] = 0
        return k_idx, k_score

    def eos_score_mask(self, max_score_k_masked):
        max_score_k_masked[:, 1:] = -torch.abs(
            max_score_k_masked[:, 1:]
        ) * float("inf")
        return max_score_k_masked

    def get_max_score_k(self, max_score, eos_padding_mask, k_):
        max_score_k = max_score[:, k_]
        max_score_k = max_score_k.unsqueeze(-1).expand(max_score.shape)
        max_score_k = max_score_k.clone()
        max_score_k[eos_padding_mask] = self.eos_score_mask(
            max_score_k[eos_padding_mask]
        )
        return max_score_k

    def update_beams(self, beam_tokens, idx, i):
        beam_tokens[:, :i, :] = torch.gather(
            beam_tokens[:, :i, :],
            0,
            idx[:, :, 1].transpose(0, 1).unsqueeze(1).repeat(1, i, 1),
        )
        beam_tokens[:, i, :] = idx[:, :, 0].transpose(0, 1)
        return beam_tokens

    def _beam_step(self, memory, beam_tokens, i, max_score, batch_size):
        score = torch.full(
            (batch_size, self.k),
            -float("inf"),
            dtype=torch.float32,
            device=self.device,  # TODO: initializing with -inf to avoid if else of k = 0; its quite slow so best to optimize
        )
        idx = torch.full(
            (batch_size, self.k, 2),
            self.pad_idx,
            dtype=torch.int64,
            device=self.device,
        )
        tgt_mask = torch.triu(
            torch.ones(i, i, device=self.device) * float("-inf"), diagonal=1
        )  # TODO: REFACTOR THIS
        for k_ in range(self.k):
            tgt = beam_tokens[k_]
            tgt_padding_mask = self.create_padding_mask(tgt[:i, :])
            eos_padding_mask = self.create_eos_mask(tgt_padding_mask, dim=1)
            interim_score = torch.empty(score.shape[0], self.k * 2).to(
                self.device
            )
            interim_idx = torch.empty(
                idx.shape[0], self.k * 2, 2, dtype=torch.int64
            ).to(self.device)

            ll = self._step(
                memory, tgt[:i, :], tgt_mask, tgt_padding_mask[:, :i]
            )
            ll = ll[-1, :, :]
            k_idx, k_score = self._predict_k(ll, eos_padding_mask, k_)
            max_score_k = self.get_max_score_k(max_score, eos_padding_mask, k_)

            interim_score = torch.cat((score, max_score_k + k_score), dim=-1)
            interim_idx = torch.cat((idx, k_idx), dim=1)
            score, idx = self.reorder_scores(interim_score, interim_idx)
            if i == 1:
                break  # Initialize beams before diverging
        beam_tokens = self.update_beams(beam_tokens, idx, i)
        return beam_tokens, score

    def greedy_search(self, memory: Tensor) -> Tensor:
        batch_size = memory.shape[1]
        greedy_tokens = torch.full(
            (self.max_seq_len, batch_size), self.pad_idx, device=self.device
        )
        greedy_tokens[0, :] = self.bos_idx
        for i in range(1, self.max_seq_len - 1):
            tgt_padding_mask = self.create_padding_mask(greedy_tokens[:i, :])
            eos_mask = ~self.create_eos_mask(tgt_padding_mask, dim=1)
            if torch.all(~eos_mask):
                break

            ll = self._step(
                tgt=greedy_tokens[:i, eos_mask],
                memory=memory[:, eos_mask],
                tgt_mask=self.create_tgt_mask(greedy_tokens[:i, eos_mask]),
                tgt_padding_mask=self.create_padding_mask(
                    greedy_tokens[:i, eos_mask]
                ),
            )
            ll = ll[-1, :, :]
            greedy_tokens[i, eos_mask] = torch.argmax(ll, dim=-1)
        return greedy_tokens[1:, :]

    def beam_search(self, memory, k):
        self.k = k
        batch_size = memory.shape[1]
        score = torch.ones([batch_size, self.k]).to(self.device)
        beam_tokens = torch.full(
            (self.k, self.max_seq_len, batch_size),
            self.pad_idx,
            device=self.device,
        )
        beam_tokens[:, 0, :] = self.bos_idx
        for i in range(1, self.max_seq_len):
            beam_tokens, score = self._beam_step(
                memory, beam_tokens, i, score, batch_size
            )
            eos = torch.where(beam_tokens == self.eos_idx, True, False)
            eos = torch.any(eos, dim=1).to(self.device)
            if torch.all(eos):
                break
        return beam_tokens[:, 1:, :]

    def sample_search(self, memory, alg="greedy", k=1):
        if alg == "greedy":
            return self.greedy_search(memory)
        elif alg == "beam":
            return self.beam_search(memory, k)
