from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Callable, List, Optional, Union

import pandas as pd
import torch

from .augmentation import Converter
from .tokenizer import Tokenizer


class DataType(Enum):
    SEQUENCE = auto()
    VALUE = auto()
    CLASS = auto()


class ReturnOptions(Enum):
    Molecule = auto()
    CLEANED = auto()
    Original = auto()


class _AbsProcessor(ABC):  # Should be protocol -> change
    def __init__(self, type: DataType) -> None:
        self.type = type

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

    @abstractmethod
    def clean_report(self):
        pass

    @abstractmethod
    def return_data(self):
        pass


class FloatProcessor(_AbsProcessor):
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

    def inspect_data(self) -> None:
        print(self.data.head())

    def clean_data(self) -> None:
        self.original_len = len(self.data)
        self.original_len = len(self.data)

        cleaned_data = self.data.copy()

        # Missing
        if self.remove_missing:
            cleaned_data = cleaned_data.dropna()
            self.nmissing = self.original_len - len(cleaned_data)

        # Constrains
        if self.constrains:
            for constrain in self.constrains:
                cleaned_data = cleaned_data[constrain(cleaned_data)].dropna()
        self.nconstrain = (
            self.original_len - self.nmissing - len(cleaned_data)
            if self.constrains
            else 0
        )

        self.cleaned_data = cleaned_data

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.applymap(lambda x: torch.tensor(x))

    def augment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.augmentations:
            for augmentation in self.augmentations:
                data = augmentation(data)
        return data

    def clean_report(self) -> None:
        print("Clean Report:")
        print(f"\tOriginal: {self.original_len} datapoints, ", end="")
        print(f"Current: {len(self.cleaned_data)}")
        print(f"\tMissing: {self.nmissing}")
        print(f"\tConstrained: {self.nconstrain}")

    def return_data(self) -> pd.DataFrame:
        if self.return_type == ReturnOptions.CLEANED:
            return self.cleaned_data
        elif self.return_type == ReturnOptions.Original:
            return self.data.loc[self.cleaned_data.index, :]


class SmilesProcessor(_AbsProcessor):
    def __init__(
        self,
        type: DataType,
        return_type: ReturnOptions,
        tokenizer: Tokenizer,
        remove_duplicates: bool = True,
        remove_missing: bool = True,
        constrains: Optional[List[Callable]] = None,
        augmentations: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__(type)

        self.return_type = return_type
        assert isinstance(
            return_type, ReturnOptions
        ), "return_type should be in ReturnOptions class"

        self.tokenizer = tokenizer
        self.remove_duplicates = remove_duplicates
        self.remove_missing = remove_missing
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
        # canonical_smiles = self.data.applymap(Converter.smile2canonical)
        canonical_smiles = self.data.copy()

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
        if self.return_type == ReturnOptions.CLEANED:
            return self.canonical_smiles
        elif self.return_type == ReturnOptions.Original:
            return self.data.loc[self.canonical_smiles.index, :]
        elif self.return_type == ReturnOptions.Molecule:
            return self.data.loc[self.canonical_smiles.index, :].applymap(
                Converter.smile2mol
            )


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
