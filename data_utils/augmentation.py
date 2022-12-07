from typing import List, Optional, Union

import pandas as pd
from rdkit import Chem


class Enumerator:
    def __init__(
        self,
        enumerations: int,
        seed: int,
        oversample: int = 0,
        max_len: Optional[int] = None,
        keep_original: bool = True,
    ) -> None:
        self.enumerations = enumerations
        self.oversample = oversample
        self.max_len = max_len
        self.keep_original = keep_original

        self.seed = seed

    def smiles_enumeration(
        self,
        smile: str,
    ) -> Union[List[str], None]:
        try:
            enum_smiles = Chem.MolToRandomSmilesVect(
                Chem.MolFromSmiles(smile),
                self.enumerations + self.oversample,
                randomSeed=self.seed,
            )
        except Exception:
            return None

        if self.keep_original:
            unique_enum_smiles = list(set([smile, *enum_smiles]))
            max_n = self.enumerations + 1
        else:
            unique_enum_smiles = list(set(enum_smiles))
            max_n = self.enumerations

        if self.max_len:
            unique_enum_smiles = [
                smile
                for smile in unique_enum_smiles
                if len(smile) <= self.max_len
            ]

        return (
            unique_enum_smiles[:max_n]
            if len(unique_enum_smiles) >= max_n
            else None
        )

    def dataframe_enumeration(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.applymap(lambda x: self.smiles_enumeration(x))
        data = data.dropna()
        return data.apply(pd.Series.explode)


class Converter:
    @staticmethod
    def smile2mol(smile: str) -> Chem:
        return Chem.MolFromSmiles(smile)

    @staticmethod
    def mol2canonical(mol: Chem) -> Union[str, None]:
        try:
            return Chem.MolToSmiles(
                mol,
                isomericSmiles=True,
                kekuleSmiles=False,
                canonical=True,
                allBondsExplicit=False,
                allHsExplicit=False,
            )
        except Exception:
            return None

    @staticmethod
    def smile2canonical(smile: str) -> Union[str, None]:
        try:
            return Chem.MolToSmiles(
                Chem.MolFromSmiles(smile),
                isomericSmiles=True,
                kekuleSmiles=False,
                canonical=True,
                allBondsExplicit=False,
                allHsExplicit=False,
            )
        except Exception:
            return None
