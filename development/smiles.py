import random
from abc import ABC, abstractclassmethod
from typing import Optional

from rdkit import Chem


class _ABCGenerator(ABC):
    def __init__(self, max_iter: int) -> None:
        self.n = 0
        self.max_iter = max_iter

    @abstractclassmethod
    def __iter__(self):
        pass

    @abstractclassmethod
    def __next__(self):
        pass

    def continue_iter(self) -> bool:
        return self.n < self.max_iter


class SmilesEnumerator(_ABCGenerator):
    def __init__(
        self,
        mol: str,
        seed: int,
        max_iter: int = 100,
        batch_size: Optional[int] = None,
    ) -> None:
        super().__init__(max_iter)
        self.max_iter = max_iter
        self.batch_size = batch_size if batch_size else 1
        
        self.mol = Chem.MolFromSmiles(mol)
        assert self.mol, "Not a valid input smile."
        
        self.random_state = random.Random()
        self.random_state.seed(seed)
        

    def __iter__(self) -> _ABCGenerator:
        return self

    def __next__(self) -> str:
        if not self.continue_iter():
            raise StopIteration
        self.n += 1
        
        return Chem.MolToRandomSmilesVect(
            self.mol, self.batch_size, randomSeed=self.random_state.randint(0, int(10e8)),
        )
