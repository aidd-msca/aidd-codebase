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
        add_canonical: bool = True,
    ) -> None:
        self.seed = seed

        self.enumerations = enumerations + 1 if add_canonical else enumerations
        self.samples = enumerations + oversample
        self.max_len = max_len
        self.add_canonical = add_canonical

    def smiles_enumeration(self, smile: str) -> Optional[List[str]]:
        try:
            mol = Chem.MolFromSmiles(smile)
            enum_smiles = Chem.MolToRandomSmilesVect(mol, self.samples, randomSeed=self.seed)
            if self.add_canonical:
                canon_smile = Chem.MolToSmiles(
                    mol, isomericSmiles=True, kekuleSmiles=False, canonical=True, allBondsExplicit=False
                )
                unique_enum_smiles = list(set([canon_smile, *enum_smiles]))
            else:
                unique_enum_smiles = list(set(*enum_smiles))

        except Exception:
            return None

        if self.max_len:
            unique_enum_smiles = [smile for smile in unique_enum_smiles if len(smile) <= self.max_len]

        return unique_enum_smiles[: self.enumerations] if len(unique_enum_smiles) >= self.enumerations else None

    def dataframe_enumeration(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.applymap(lambda x: self.smiles_enumeration(x))
        data = data.dropna()
        return data.apply(pd.Series.explode)


class Converter:
    @staticmethod
    def smile2mol(smile: str) -> Chem:
        try:
            return Chem.MolFromSmiles(smile)
        except Exception:
            return None

    @staticmethod
    def smile2reaction(smile: str) -> Chem:
        try:
            return Chem.rdChemReactions.ReactionFromSmarts(smile, useSmiles=True)
        except Exception:
            return None

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
