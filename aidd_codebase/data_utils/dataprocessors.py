from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Callable, List, Optional, Union
from tqdm import tqdm
import logging

import numpy as np
import pandas as pd
import torch
from aidd_codebase.data_utils.augmentation import Converter
from aidd_codebase.utils.tools import compose

from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize


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

    @abstractmethod
    def clean_report(self) -> None:
        pass

    @abstractmethod
    def return_data(self):
        pass


class CategoricalProcessor(_AbsProcessor):
    """Placeholder"""

    pass


class NumericalProcessor(_AbsProcessor):
    def __init__(
        self,
        remove_missing: bool = True,
        constrains: Optional[List[Callable]] = None,
        augmentations: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__()

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

    def clean_data(self) -> None:
        # Original
        n_original = len(self.data)
        cleaned_data = self.data.copy()

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

        self.cleaned_length = n_cleaned
        self.original_length = n_original
        self.missing_length = n_missing
        self.constrained_length = n_constrained

        # self.clean_report(n_original, n_cleaned, n_missing, n_constrained)

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.astype(float).applymap(lambda x: torch.tensor(x))

    def augment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.augmentations:
            for augmentation in self.augmentations:
                data = augmentation(data)
        return data

    def return_data(self) -> pd.DataFrame:
        if self.remove_missing:
            return self.cleaned_data
        else:
            return self.data.loc[self.cleaned_data.index, :]


class SmilesProcessor(_AbsProcessor):
    def __init__(
        self,
        return_original: bool = False,
        verbose: bool = True,
        logger: Optional[logging.Logger] = None,
        sanitize: bool = True,
        remove_salts: bool = True,
        remove_stereo: bool = True,
        remove_metal_atoms: bool = False,
        keep_largest_fragment: bool = True,
        neutralize_mol: bool = False,
        standardize_tautomers: bool = True,
        remove_duplicates: bool = True,
        canonicalize_smiles: bool = True,
        limit_seq_len: Optional[int] = None,
        constrains: Optional[List[Callable]] = None,
        augmentations: Optional[List[Callable]] = None,
        transformations: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__()

        tqdm.pandas()

        self.return_original = return_original
        self.verbose = verbose
        self.logger = logger

        self.sanitize = sanitize
        self.remove_salts = remove_salts
        self.remove_stereo = remove_stereo
        self.remove_metal_atoms = remove_metal_atoms
        self.keep_largest_fragment = keep_largest_fragment
        self.neutralize_mol = neutralize_mol
        self.standardize_tautomers = standardize_tautomers
        self.canonicalize_smiles = canonicalize_smiles
        self.remove_duplicates = remove_duplicates
        self.limit_seq_len = limit_seq_len
        self.constrains = constrains

        self.augmentations = augmentations
        self.transformations = transformations

        self.original_length: Optional[int] = None
        self.duplicated_length: Optional[int] = None
        self.constrained_length: Optional[int] = None
        self.cleaned_length: Optional[int] = None
        self.augmented_length: Optional[int] = None

    def load_data(self, data_path: str) -> None:
        self.data = pd.read_csv(data_path, header=None)

    def set_data(self, data: Union[pd.DataFrame, pd.Series]):
        self.data = data

    def inspect_data(self) -> None:
        print(self.data.head())
        print("Length breakdown:", self.data.nunique(dropna=False))

    def _remove_salts(self, mols: pd.DataFrame) -> pd.DataFrame:
        def __remove_salts(mol):
            if mol is None:
                return None
            try:
                remover = SaltRemover.SaltRemover()
                return remover.StripMol(mol)
            except ValueError:
                pass
            return mol

        self.log("Stripping salts...")
        mols = mols.progress_applymap(__remove_salts)
        return mols

    def _remove_stereo(self, mols: pd.DataFrame) -> pd.DataFrame:
        def __remove_stereo(mol):
            if mol is None:
                return None
            try:
                Chem.RemoveStereochemistry(mol)
                return mol
            except ValueError:
                pass
            return mol

        self.log("Removing stereochemistry...")
        mols.progress_applymap(__remove_stereo)
        return mols

    def _assign_stereo(self, mols: pd.DataFrame) -> pd.DataFrame:
        def __assign_stereo(mol):
            if mol is None:
                return None
            try:
                Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
                return mol
            except ValueError:
                pass
            return mol

        self.log("Assigning stereochemistry...")
        mols.progress_applymap(__assign_stereo)
        return mols

    def _remove_metal_atoms(self, mols: pd.DataFrame) -> pd.DataFrame:
        self.log("Removing metal atoms...")
        mols = mols.progress_applymap(rdMolStandardize.MetalDisconnector().Disconnect)
        return mols

    def _keep_largest_fragment(self, mols: pd.DataFrame) -> pd.DataFrame:
        self.log("Keeping largest fragment...")
        mols = mols.progress_applymap(rdMolStandardize.FragmentParent)
        return mols

    def _neutralize_mol(self, mols: pd.DataFrame) -> pd.DataFrame:
        self.log("Neutralizing molecules...")
        mols = mols.progress_applymap(rdMolStandardize.Uncharger().uncharge)
        return mols

    def _tautomers(self, mols: pd.DataFrame) -> pd.DataFrame:
        self.log("Enumerating tautomers...")
        mols = mols.progress_applymap(rdMolStandardize.TautomerEnumerator().Canonicalize)
        return mols

    def clean_data(self) -> pd.DataFrame:
        self.original_length = len(self.data)

        self.log("Converting SMILES to molecules...")
        mols: pd.DataFrame = self.data.copy().progress_applymap(Converter.smile2mol).dropna()
        # mols: pd.DataFrame = self.data.copy().progress_applymap(Converter.smile2reaction).dropna()
        # rdChemReactions.ChemicalReaction.Initialize(mols.iloc[0, 0])

        def sanitize(mol):
            try:
                Chem.SanitizeMol(mol)
                mol = Chem.RemoveHs(mol)
                mol = rdMolStandardize.Normalize(mol)
                mol = rdMolStandardize.Reionize(mol)
            except ValueError:
                return None
            return mol

        if self.sanitize:
            self.log("Sanitizing molecules...")
            mols = mols.progress_applymap(sanitize).dropna()

        if self.remove_salts:
            mols = self._remove_salts(mols).dropna()

        if self.remove_stereo:
            mols = self._remove_stereo(mols).dropna()
        else:
            mols = self._assign_stereo(mols).dropna()
            # add mixed data option
            # check non-smile symbols

        if self.remove_metal_atoms:
            mols = self._remove_metal_atoms(mols).dropna()

        if self.keep_largest_fragment:
            mols = self._keep_largest_fragment(mols).dropna()

        if self.neutralize_mol:
            mols = self._neutralize_mol(mols).dropna()

        if self.standardize_tautomers:
            mols = self._tautomers(mols).dropna()

        if self.canonicalize_smiles:
            self.log("Converting molecules to canonical SMILES...")
            cleaned_smiles: pd.DataFrame = (
                mols.progress_applymap(Converter.mol2canonical).replace(r"^\s*$", np.nan, regex=True).dropna()
            )
        else:
            self.log("Converting molecules to SMILES...")
            cleaned_smiles = mols.progress_applymap(Chem.MolToSmiles).replace(r"^\s*$", np.nan, regex=True).dropna()
        self.cleaned_length = self.original_length - len(cleaned_smiles)

        if self.remove_duplicates:
            self.log("Removing duplicates...")
            cleaned_smiles = cleaned_smiles.drop_duplicates().dropna()
            self.duplicated_length = self.original_length - len(cleaned_smiles)

        if self.limit_seq_len and self.limit_seq_len > 0:
            self.log("Removing SMILES with length > {}...".format(self.limit_seq_len))
            cleaned_smiles = cleaned_smiles[cleaned_smiles.applymap(len) <= self.limit_seq_len].dropna()

        if self.constrains is not None:
            self.log("Applying constrains...")
            for constrain in self.constrains:
                cleaned_smiles = cleaned_smiles[constrain(cleaned_smiles)].dropna()
            self.constrained_length = self.original_length - len(cleaned_smiles)

        self.cleaned = cleaned_smiles

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.transformations:
            transforms = compose(*(transform for transform in self.transformations))
        return data.applymap(lambda x: transforms(x))

    def augment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.augmentations is not None:
            original = len(data)
            for augmentation in self.augmentations:
                data = augmentation(data)
            self.log(f"\tAugmentation: from {original} datapoints," + f"to {len(data)} datapoints.")
        return data

    def clean_report(self) -> None:
        self.log("Clean Report:")
        self.log(f"\tOriginal: {self.original_length} datapoints, Current: {len(self.data)}")
        self.log(f"\tCleaned: {self.cleaned_length}")
        self.log(f"\tDuplicates: {self.duplicated_length}")
        self.log(f"\tConstrained: {self.constrained_length}")

    def return_data(self) -> pd.DataFrame:
        if not self.return_original:
            return self.cleaned
        else:
            return self.data.loc[self.cleaned.index, :]

    def __len__(self) -> int:
        return len(self.cleaned)

    def log(self, msg: str) -> None:
        if not self.verbose:
            pass
        elif self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)


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


# params = rdMolStandardize.CleanupParameters(preferOrganic = preferOrganic)

# print("Default acidbaseFile: %s" % params.acidbaseFile)
# doCanonical: bool = True
# largestFragmentChooserCountHeavyAtomsOnly: bool = False
# largestFragmentChooserUseAtomCount: bool = True
# preferOrganic: bool = True
# print("Default normalizationsFile: %s" % params.normalizationsFile)
# print("Default fragmentFile: %s" % params.fragmentFile)
# print("Default maxRestarts: %s" % params.maxRestarts)
# print("Default preferOrganic: %s" % params.preferOrganic)
