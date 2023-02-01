from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from aidd_codebase.utils.typescripts import Tensor
from torch.utils.data import Dataset


class _AbsDataset(Dataset):
    """Dataset setup imported from the AstraZenica GitHub
    MolecularAI/Chemformer/blob/main/molbart/data/datasets.py"""

    def __len__(self) -> int:
        """Get the length of the dataset."""
        raise NotImplementedError()

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor], Tuple[pd.DataFrame, pd.DataFrame]]:
        """Retrieve a tuple of source and target for a specific index."""
        raise NotImplementedError()


class DummyDataset(_AbsDataset):
    def __init__(self, data: List[zip]) -> None:
        self.data = [*data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor], Tuple[pd.DataFrame, pd.DataFrame]]:
        inp, out = self.data[idx]
        return inp, out


class DataFrameDataset(Dataset):
    def __init__(self, data: pd.DataFrame, x_cols: List[str], y_cols: List[str]) -> None:
        self.data = data
        self.X = self.data.loc[:, x_cols].squeeze().to_numpy()
        self.Y = self.data.loc[:, y_cols].squeeze().to_numpy()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.X[idx], self.Y[idx]


class PredictDataFrameDataset(Dataset):
    def __init__(self, data: pd.DataFrame, x_cols: List[str]) -> None:
        self.data = data
        self.X = self.data.loc[:, x_cols].squeeze().to_numpy()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tensor:
        return self.X[idx]


class BasicDataset(Dataset):
    def __init__(self, data: pd.DataFrame, x_cols: List[str], y_cols: List[str]) -> None:
        self.data = data
        self.X = self.data.loc[:, x_cols].squeeze().to_numpy()
        self.Y = self.data.loc[:, y_cols].squeeze().to_numpy()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.X[idx], self.Y[idx]
