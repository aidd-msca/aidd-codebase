from abc import ABC

import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from aidd_codebase.utils.config import _ABCDataClass
from aidd_codebase.utils.metacoding import CreditType
from aidd_codebase.datamodules.datachoice import DataChoice
from aidd_codebase.data_utils.datasets import BasicDataset
from aidd_codebase.data_utils.tokenizer import Tokenizer
from aidd_codebase.data_utils.collate import Collate
from aidd_codebase.data_utils.dataprocessors import _AbsProcessor, DataType, ReturnOptions, SmilesProcessor
from aidd_codebase.data_utils.augmentation import Enumerator


@DataChoice.register_arguments(call_name="smiles")
class SmilesArguments(_ABCDataClass):
    num_workers: int = 8
    persistent_workers: bool = True
    redo_data_processing: bool = False
    batch_size = 32

    reactions: bool = True
    home_path: Optional[str] = '/content/'
    data_path: Optional[str] = '/content/retrosynthesis-all.smi'

    seed = 10
    partitions: Dict[str, float] = {'train': 0.8, 'valid': 0.1, 'test': 0.1}

    remove_missing: bool = True
    remove_duplicates: bool = False
    canonicalization: bool = True
    enumeration: int = 0  # 10
    enumeration_oversample: int = 0  # 15


@DataChoice.register_choice(call_name="smiles", author="Emma Svensson", github_handle="emmas96", credit_type=CreditType.NONE)
class SmilesDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_args: SmilesArguments
    ) -> None:
        super().__init__()

        self.data_path = data_args.data_path
        self.reactions = data_args.reactions

        self.partitions = data_args.partitions
        self.datasplit: Dict[str, pd.DataFrame] = {}

        self.seed = data_args.seed
        self.batch_size = data_args.batch_size
        self.num_workers = data_args.num_workers
        self.persistent_workers = data_args.persistent_workers

        tokenizer = Tokenizer(
            vocab=(" ^$?#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy"),
            pad_idx=0,
            bos_idx=1,
            eos_idx=2,
            max_seq_len=250
        )

        self.collator = Collate(tokenizer.PAD_IDX).simple_collate_fn

        enumerator = Enumerator(
            enumerations=data_args.enumeration,
            seed=data_args.seed,
            oversample=data_args.enumeration_oversample,
            max_len=tokenizer.MAX_SEQ_LEN,
            keep_original=data_args.canonicalization,
        )
        if data_args.enumeration > 0:
            augmentations = [lambda x: enumerator.dataframe_enumeration(x)]
        else:
            augmentations = None

        self.processors = {source: SmilesProcessor(
            type=DataType.SEQUENCE,
            return_type=ReturnOptions.CLEANED if source == 'data' and data_args.canonicalization else ReturnOptions.Original,
            tokenizer=tokenizer,
            remove_duplicates=data_args.remove_duplicates,
            remove_missing=data_args.remove_missing,
            constrains=[lambda x: x.applymap(len) <= tokenizer.MAX_SEQ_LEN],
            augmentations=augmentations,
        ) for source in ['data', 'target']}

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
        data = self.process_data(data, self.processors['data'])

        if self.reactions:
            target = df.iloc[:, [1]]
            target = self.process_data(target, self.processors['target'])
        else:
            target = data.copy()

        return data, target

    def process_data(self, data: pd.DataFrame, data_processor: _AbsProcessor) -> pd.DataFrame:
        '''
        Helper to read data.
        '''
        data_processor.set_data(data)
        data_processor.inspect_data()
        data_processor.clean_data()
        data_processor.clean_report()
        return data_processor.return_data()

    def split_data(self, data: pd.DataFrame, target: pd.DataFrame) -> None:
        '''
        Helper to prepare data.
        '''
        assert len(data) == len(target), "The input length must match the output length."
        assert (sum(self.partitions.values()) <= 1), "Combined partitions should be less than or equal to 1."

        fractions = list(self.partitions.values())
        corrected_fractions = [frac / (1.0 - sum(fractions[:idx])) for idx, frac in enumerate(fractions)]
        corrected_fractions = [frac if frac <= 1.0 else 1.0 for frac in corrected_fractions]

        data.columns = pd.MultiIndex.from_product([data.columns, ["input"]])
        target.columns = pd.MultiIndex.from_product([target.columns, ["output"]])
        df = pd.merge(data, target, left_index=True, right_index=True)

        sampled_idxs: List = []
        for split, frac in zip(self.partitions.keys(), corrected_fractions):
            fraction_dataset = df.drop(sampled_idxs).sample(frac=frac, random_state=self.seed)
            sampled_idxs.extend(fraction_dataset.index)
            self.datasplit[split] = fraction_dataset

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

        print("Finished setting up Data Module!")

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

