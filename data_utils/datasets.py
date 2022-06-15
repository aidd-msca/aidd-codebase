from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from aidd_codebase.utils.typescripts import Tensor


class _AbsDataset(Dataset):
    """Dataset setup imported from the AstraZenica GitHub
    MolecularAI/Chemformer/blob/main/molbart/data/datasets.py"""

    def __len__(self) -> int:
        """Get the length of the dataset."""
        raise NotImplementedError()

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[Tensor, Tensor],
        Tuple[pd.DataFrame, pd.DataFrame],
    ]:
        """Retrieve a tuple of source and target for a specific index."""
        raise NotImplementedError()


class DummyDataset(_AbsDataset):
    def __init__(self, data: List[zip]) -> None:
        self.data = [*data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[Tensor, Tensor],
        Tuple[pd.DataFrame, pd.DataFrame],
    ]:
        inp, out = self.data[idx]
        return inp, out


class BasicDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        self.data = data
        self.input = (
            self.data.loc[:, (slice(None), "input")]
            .droplevel(level=1, axis=1)
            .squeeze()
            .to_numpy()
        )
        self.output = (
            self.data.loc[:, (slice(None), "output")]
            .droplevel(level=1, axis=1)
            .squeeze()
            .to_numpy()
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (self.input[idx], self.output[idx])
