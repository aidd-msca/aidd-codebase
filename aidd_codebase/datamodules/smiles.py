from abc import ABC

import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datachoice import DataChoice
from ..utils.config import _ABCDataClass
from ..utils.metacoding import CreditType
from ..data_utils.datasets import BasicDataset
from ..data_utils.collate import Collate


@dataclass
class DataArguments(_ABCDataClass):
    NUM_WORKERS: int = 8
    PERSISTENT_WORKERS: bool = True
    REDO_DATA_PROCESSING: bool = False
    BATCH_SIZE = 32

    REACTIONS: bool = True
    HOME_DIR: Optional[str] = '/content/'
    DATA_PATH: Optional[str] = '/content/retrosynthesis-all.smi'

    REMOVE_MISSING: bool = True
    REMOVE_DUPLICATES: bool = False
    CANONICALIZATION: bool = True
    ENUMERATION: int = 0
    ENUM_OVERSAMPLE: int = 15

    SEED = 10
    PARTITIONS = [0.8, 0.1, 0.1]


@DataChoice.register_choice("smiles", "Emma Svensson", CreditType.NONE)
class SmilesDataModule(pl.LightningDataModule):
    def __init__(self, data_args):
        super().__init__()
        self.data_path = data_args.DATA_PATH
        self.reactions = data_args.REACTIONS

        self.partitions = data_args.PARTITIONS
        self.datasplit: Dict[str, pd.DataFrame] = {}

        self.seed = data_args.SEED
        self.batch_size = data_args.BATCH_SIZE
        self.num_workers = data_args.NUM_WORKERS
        self.persistent_workers = data_args.PERSISTENT_WORKERS

        self.collator = Collate(0)

    def prepare_data(self, **kwargs) -> None:
        '''
        Called by PyTorch-Lightning.
        '''
        print("Starting data loading and preparing...")

        data, target = self.read_data()
        self.split_data(data, target)

        print("Finished preparing data!")

    def read_data(self):
        '''
        Helper to prepare data.
        '''
        df = pd.read_csv(self.data_path, header=None)

        df = df[0].str.split(">>", n=1, expand=True)
        data = df.iloc[:, [0]]

        if self.reactions:
            target = df.iloc[:, [1]]
        else:
            target = data.copy()

        return data, target

    def split_data(self, data: pd.DataFrame, target: pd.DataFrame) -> None:
        '''
        Helper to prepare data.
        '''
        assert len(data) == len(target), "The input length must match the output length."
        assert (sum(self.partitions.values()) <= 1), "Combined partitions should be less than or equal to 1."

        input.columns = pd.MultiIndex.from_product([input.columns, ["input"]])
        target.columns = pd.MultiIndex.from_product(
            [target.columns, ["output"]]
        )
        df = pd.merge(data, target, left_index=True, right_index=True)

        datasets: List = []
        sampled_idxs: List = []
        for frac in list(self.partitions.values()):
            fraction_dataset = df.drop(sampled_idxs).sample(
                frac=frac, random_state=self.seed
            )
            datasets.append(fraction_dataset)
            sampled_idxs.extend(fraction_dataset.index)

        self.datasplit.update(zip(list(self.partitions.keys()), datasets))

        print(
            ", ".join(
                [f"{k.upper()}: {len(v)}" for k, v in self.datasplit.items()]
            )
        )

    def setup(self, stage: Optional[str] = None) -> None:
        '''
        Called by PyTorch-Lightning.
        '''
        print("Starting Data Module Setup...")

        if stage in ('fit', None):
            self.train = self.datasplit['train']
            self.val = self.datasplit['valid']

        if stage in ('test', None):
            self.test = self.datasplit['test']

        print("Finished Data Module!")

    def train_dataloader(self):
        cuda = torch.cuda.is_available()
        return DataLoader(
            BasicDataset(self.train),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers if cuda else 0,
            persistent_workers=self.persistent_workers if cuda else False,
            pin_memory=cuda,
        )

    def val_dataloader(self):
        cuda = torch.cuda.is_available()
        return DataLoader(
            BasicDataset(self.val),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers if cuda else 0,
            persistent_workers=self.persistent_workers if cuda else False,
            pin_memory=cuda,
        )

    def test_dataloader(self):
        cuda = torch.cuda.is_available()
        return DataLoader(
            BasicDataset(self.test),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers if cuda else 0,
            persistent_workers=self.persistent_workers if cuda else False,
            pin_memory=cuda,
        )

    def predict_dataloader(self):
        '''
        Shadows val_dataloader.
        '''
        return self.val_dataloader()

