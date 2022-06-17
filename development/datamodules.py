from typing import Callable, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ..data_utils.dataprocessors import _AbsProcessor
from ..data_utils.datasets import BasicDataset
from ..utils.typescripts import Tensor


class ABCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_file: str,
        output_file: str,
        collate_fn: Callable,
        input_processor: _AbsProcessor,
        output_processor: _AbsProcessor,
        seed: int,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.input_file = input_file
        self.output_file = output_file

        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.input_processor = input_processor
        self.output_processor = output_processor

    def process_data(
        self, data_file: str, data_processor: _AbsProcessor
    ) -> pd.DataFrame:
        data_processor.load_data(data_file)
        data_processor.inspect_data()
        data_processor.clean_data()
        data_processor.clean_report()
        data_processor.transform_data()
        return data_processor.return_data()

    def split_data(self):
        raise NotImplementedError

    def convert_to_tensor(self):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        print("Starting Data Module Setup...")
        # Optional: Load Data
        # Process Data
        self.input = self.process_data(self.input_file)
        self.output = self.process_data(self.output_file)

        train, val, test = self.split_data()

        if stage == "fit" or stage is None:
            self.train = self.convert_to_tensors(train)

        if stage == "fit" or stage is None:
            self.val = self.convert_to_tensors(val)

        if stage == "test" or stage is None:
            self.test = self.convert_to_tensors(test)

        print("Finished Data Module!")

    def train_dataloader(self):
        return DataLoader(
            BasicDataset(self.train),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            BasicDataset(self.val),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            BasicDataset(self.test),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
        )
