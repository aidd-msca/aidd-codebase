from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, List, Optional, Union

import pandas as pd
import torch

from aidd_codebase.data_utils.augmentation import Converter, Enumerator


class DataType(Enum):
    SEQUENCE = auto()
    VALUE = auto()
    CLASS = auto()


class ReturnOptions(Enum):
    Molecule = auto()
    CLEANED = auto()
    Original = auto()


class _AbsProcessor(ABC):  # Should be protocol -> change
    def __init__(self) -> None:
        pass

    @abstractmethod
    def load_data(self, data_path: str):
        pass

    @abstractmethod
    def set_data(self, data: Union[pd.DataFrame, pd.Series]) -> None:
        pass

    @abstractmethod
    def inspect_data(self):
        pass

    @abstractmethod
    def clean_data(self):
        pass

    @abstractmethod
    def augment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def clean_report(n_original: int, n_cleaned: int, n_missing: int, n_constrained: int) -> None:
        print("Clean Report:")
        print(f"\tOriginal: {n_original} datapoints, ", end="")
        print(f"Current: {n_cleaned}")
        print(f"\tMissing: {n_missing}")
        print(f"\tConstrained: {n_constrained}")

    @abstractmethod
    def return_data(self):
        pass
    

class CategoricalProcessor(_AbsProcessor):
    """Placeholder"""
    pass


class NumericalProcessor(_AbsProcessor):
    def __init__(
        self,
        type: DataType,
        return_type: ReturnOptions,
        remove_missing: bool = True,
        constrains: Optional[List[Callable]] = None,
        augmentations: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__(type)

        self.return_type = return_type
        assert isinstance(
            return_type, ReturnOptions
        ), "return_type should be in ReturnOptions class"

        self.remove_missing = remove_missing
        self.constrains = constrains
        self.augmentations = augmentations

    def load_data(self, data_path: str) -> None:
        self.data = pd.read_csv(data_path, header=None)

    def set_data(self, data: Union[pd.DataFrame, pd.Series]):
        self.data = data

    def inspect_data(self, data: Union[pd.DataFrame, pd.Series]) -> None:
        print(data.head())

    def clean_data(self, data: Union[pd.DataFrame, pd.Series]) -> None:
        # Original
        n_original = len(data)
        cleaned_data = data.copy()

        # Missing
        if self.remove_missing:
            cleaned_data = cleaned_data.dropna()
            n_missing = n_original - len(cleaned_data)

        # Constrains
        n_constrained = 0
        if self.constrains:
            for constrain in self.constrains:
                cleaned_data = cleaned_data[constrain(cleaned_data)].dropna()
            n_constrained = n_original - n_missing - len(cleaned_data)
        
        # Final
        n_cleaned = len(cleaned_data)
        self.cleaned_data = cleaned_data
        
        self.clean_report(n_original, n_cleaned, n_missing, n_constrained)

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.applymap(lambda x: torch.tensor(x))

    def augment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.augmentations:
            for augmentation in self.augmentations:
                data = augmentation(data)
        return data

    def return_data(self) -> pd.DataFrame:
        if self.return_type == ReturnOptions.CLEANED:
            return self.cleaned_data
        elif self.return_type == ReturnOptions.Original:
            return self.data.loc[self.cleaned_data.index, :]


class SmilesProcessor(_AbsProcessor):
    def __init__(
        self,
        remove_missing: bool = True,
        remove_duplicates: bool = True,
        canonicalize_smiles: bool = True,
        limit_seq_len: Optional[int] = None,
        constrains: Optional[List[Callable]] = None,
        augmentations: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__()
        
        self.remove_missing = remove_missing
        self.remove_duplicates = remove_duplicates
        self.canonicalize_smiles = canonicalize_smiles
        self.limit_seq_len = limit_seq_len
        self.constrains = constrains
        self.augmentations = augmentations

    def load_data(self, data_path: str) -> None:
        self.data = pd.read_csv(data_path, header=None)

    def set_data(self, data: Union[pd.DataFrame, pd.Series]):
        self.data = data

    def inspect_data(self) -> None:
        print(self.data.head())
        print("Length breakdown:", self.data.nunique(dropna=False))

    def clean_data(self) -> pd.DataFrame:
        self.original_len = len(self.data)
        canonical_smiles = self.data.copy() if self.canonicalize_smiles else self.data
        canonical_smiles = canonical_smiles.applymap(Converter.smile2canonical)

        # Duplicates
        if self.remove_duplicates:
            canonical_smiles = canonical_smiles.drop_duplicates()
            self.nduplicates = self.original_len - len(canonical_smiles)
        else:
            self.nduplicates = 0

        # Missing
        if self.remove_missing:
            canonical_smiles = canonical_smiles.dropna()
            self.nmissing = (
                self.original_len - self.nduplicates - len(canonical_smiles)
            )
        else:
            self.nmissing = 0

        # Constrains
        if self.limit_seq_len and self.limit_seq_len != 0:
            canonical_smiles = canonical_smiles[
                canonical_smiles.applymap(len) <= self.limit_seq_len
            ].dropna()
            
        if self.constrains is not None:
            for constrain in self.constrains:
                canonical_smiles = canonical_smiles[
                    constrain(canonical_smiles)
                ].dropna()
            self.nconstrain = (
                self.original_len
                - self.nduplicates
                - self.nmissing
                - len(canonical_smiles)
            )
        else:
            self.nconstrain = 0

        self.canonical_smiles = canonical_smiles

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.applymap(lambda x: self.tokenizer.smile_prep(x))

    def augment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.augmentations:
            original = len(data)
            print(len(data))
            for augmentation in self.augmentations:
                data = augmentation(data)
            print(
                f"\tAugmentation: from {original} datapoints,"
                + f"to {len(data)} datapoints."
            )
        return data

    def clean_report(self) -> None:
        print("Clean Report:")
        print(f"\tOriginal: {self.original_len} datapoints, ", end="")
        print(f"Current: {len(self.canonical_smiles)}")
        print(f"\tDuplicates: {self.nduplicates}")
        print(f"\tMissing: {self.nmissing}")
        print(f"\tConstrained: {self.nconstrain}")

    def return_data(self) -> pd.DataFrame:
        if self.canonicalize_smiles:
            return self.canonical_smiles
        else:
            return self.data.loc[self.canonical_smiles.index, :]


# ==============
# 1.1 Inspection
# ==============

# Detect unexpeted, incorrect or inconsistent data.

# Data profiling (summary statistics)
# Visualizations (outlier detection, ranges stds and the unexpected)


# ============
# 1.2 Cleaning
# ============

# Fix or remove the anomalies discovered.

# Irrelevant data: remove data unnessecary to our current goals
# Duplicates: combine or remove based on identical identifiers
# Type conversion: make sure tensors are tensors, string are strings etc.
# Syntax errors: padding remove/add, fix typos (nspectfor low-frequence values)
# Standardize: all units the same, all strings should have similar syntax
# Scaling/Transformation: Skaling 0-100 to 0-1
# Normalization: nomrally distributed data
# Missing Data: Remove all missing i.e. NA, 0, Null, etc., flag or impute
# Outliers: all values outside the 1.5 sigma away should be considered outliers
#           (but not neccessarily guilty)


# =============
# 1.3 Verifying
# =============

# After cleaning, ther results should be verified for correctness

# Redo Inspection en checks
# Verify that the data can be split correctly


# =============
# 1.4 Reporting
# =============

# A report about made changes and the qualitify of current data should be made

# Report all alterations, removals and final statistics of the dataset
