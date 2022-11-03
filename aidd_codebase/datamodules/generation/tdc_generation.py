import pathlib
from dataclasses import dataclass, field
from os import listdir
from os.path import isfile, join
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from aidd_codebase.data_utils.augmentation import Enumerator
from aidd_codebase.data_utils.collate import Collate
from aidd_codebase.data_utils.dataprocessors import (
    SmilesProcessor,
    _AbsProcessor,
)
from aidd_codebase.data_utils.datasets import BasicDataset2
from aidd_codebase.datamodules.datachoice import DataChoice
from aidd_codebase.datamodules.tokenizer import Tokenizer
from aidd_codebase.utils.config import _ABCDataClass
from aidd_codebase.utils.directories import validate_or_create_dir
from aidd_codebase.utils.metacoding import CreditType
from tdc.generation import MolGen
from torch.utils.data import DataLoader


@DataChoice.register_arguments(call_name="chembl")
@dataclass(unsafe_hash=True)
class DataArguments(_ABCDataClass):
    dataset: str = "chembl_V29"
    seed: int = 1234
    batch_size: int = 128
    num_workers: int = 8
    persistent_workers: bool = True

    override_prepared_data: bool = True
    prepared_data_dir: Optional[
        str
    ] = f"{pathlib.Path(__file__).parent.resolve()}/data/saved"

    partitions: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )

    remove_missing: bool = True
    remove_duplicates: bool = False
    canonicalization: bool = True
    enumeration: int = 10
    enumeration_oversample: int = 15


@DataChoice.register_choice(
    call_name="chembl",
    author="Peter Hartog",
    github_handle="PeterHartog",
    credit_type=CreditType.NONE,
)
class ChemblDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: Tokenizer,
        data_args: DataArguments,
    ):
        super().__init__()

        args = data_args
        # self.data_args = DataArguments(**data_args)

        self.datasplit: Dict[str, pd.DataFrame] = {}

        if args.prepared_data_dir:
            self.prepared_data_dir = validate_or_create_dir(
                args.prepared_data_dir
            )
        self.override_prepared_data = args.override_prepared_data

        self.seed = args.seed
        self.partitions = {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        }  # args.partitions
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.persistent_workers = args.persistent_workers
        self.collate_fn = Collate(tokenizer.pad_idx).simple_collate_fn

        self.tokenizer = tokenizer

        enumerator = Enumerator(
            enumerations=args.enumeration,
            seed=args.seed,
            oversample=args.enumeration_oversample,
            max_len=tokenizer.max_seq_len,
            keep_original=args.canonicalization,
        )
        if args.enumeration > 0:
            augmentations = [lambda x: enumerator.dataframe_enumeration(x)]
        else:
            augmentations = None

        self.input_processor = SmilesProcessor(
            remove_missing=args.remove_missing,
            remove_duplicates=args.remove_duplicates,
            canonicalize_smiles=args.canonicalization,
            limit_seq_len=tokenizer.max_seq_len,
            constrains=None,
            augmentations=augmentations,
        )
        self.output_processor = SmilesProcessor(
            remove_missing=args.remove_missing,
            remove_duplicates=args.remove_duplicates,
            canonicalize_smiles=args.canonicalization,
            limit_seq_len=tokenizer.max_seq_len,
            constrains=None,
            augmentations=None,
        )

    def process_data(
        self, data: pd.DataFrame, data_processor: _AbsProcessor
    ) -> pd.DataFrame:
        data_processor.set_data(data)
        data_processor.inspect_data()
        data_processor.clean_data()
        return data_processor.return_data()

    def random_split_data(
        self,
        df: pd.DataFrame,
        partitions: Dict[str, float],
    ) -> None:
        assert (
            sum(partitions.values()) <= 1
        ), "Combined partitions should be less than or equal to 1"

        np.random.seed(self.seed)
        df_len = len(df)
        splits = np.split(
            df.sample(frac=1, random_state=self.seed),
            [
                int(frac * df_len)
                for frac in np.cumsum(list(partitions.values()))
            ],
        )
        self.datasplit.update(zip(list(partitions.keys()), splits))

        print(
            ", ".join(
                [f"{k.upper()}: {len(v)}" for k, v in self.datasplit.items()]
            )
        )

    def prepare_data(self, **kwargs) -> None:
        if not (
            self.check_prepared_files(
                self.prepared_data_dir, list(self.partitions.keys())
            )
            and self.override_prepared_data
        ):
            print("Starting input data download...")
            smiles = MolGen(name="ChEMBL_V29").get_data()
            smiles = smiles.iloc[0:5000, :]
            smiles = self.process_data(smiles, self.input_processor)
            self.data = self.combine_input_output(
                smiles.rename(columns={"smiles": "smiles_in"}),
                smiles.rename(columns={"smiles": "smiles_out"}),
            )
            self.random_split_data(self.data, partitions=self.partitions)

    def save_prepared_data(self, data: pd.DataFrame, stage: str) -> None:
        print(f"Saving {stage}...")
        data.to_csv(f"{self.prepared_data_dir}/{stage}.csv", index=False)

    def load_prepared_data(self, stage: str) -> pd.DataFrame:
        print(f"Loading {stage}...")
        return pd.read_csv(f"{self.prepared_data_dir}/{stage}.csv")

    def transform_data(self, data: pd.DataFrame, stage: str) -> pd.DataFrame:
        print(f"Transforming {stage}...")
        return data.applymap(lambda x: self.tokenizer.smile_prep(x))

    def augment_data(self, data: pd.DataFrame, stage: str) -> pd.DataFrame:
        print(f"Augmenting {stage}...")
        input = self.input_processor.augment_data(data[["smiles_in"]])
        output = self.output_processor.augment_data(data[["smiles_out"]])
        return self.combine_input_output(input, output)

    def setup_data(self, data: pd.DataFrame, stage: str) -> pd.DataFrame:
        print(f"Setting up {stage} stage.")
        data = self.augment_data(data, stage)
        self.save_prepared_data(data, stage)
        return data.reset_index(drop=True)

    def get_data(self, partition: str) -> pd.DataFrame:
        if (not self.override_prepared_data) and self.check_prepared_files(
            dir=self.prepared_data_dir, stages=[partition]
        ):
            partition_data = self.load_prepared_data(partition)
        else:
            if partition not in self.datasplit.keys():
                raise ValueError(f"Partition {partition} not in partitions.")
            partition_data = self.setup_data(
                self.datasplit[partition], partition
            )
        partition_data = self.transform_data(partition_data, partition)
        return partition_data

    def setup(self, stage: Optional[str] = None) -> None:
        print("Starting Data Module Setup...")

        if stage in ("fit", None):
            self.train = self.get_data(partition="train")
            self.val = self.get_data(partition="val")

        if stage in ("test", None):
            self.test = self.get_data(partition="test")

        if stage in ("predict", None):
            self.predict = self.get_data(partition="predict")

        print("Finished Data Module!")

    def train_dataloader(self):
        return DataLoader(
            BasicDataset2(
                self.train, x_cols=["smiles_in"], y_cols=["smiles_out"]
            ),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers if torch.cuda.is_available() else 0,
            persistent_workers=self.persistent_workers
            if torch.cuda.is_available()
            else False,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            BasicDataset2(
                self.val, x_cols=["smiles_in"], y_cols=["smiles_out"]
            ),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers if torch.cuda.is_available() else 0,
            persistent_workers=self.persistent_workers
            if torch.cuda.is_available()
            else False,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            BasicDataset2(
                self.test, x_cols=["smiles_in"], y_cols=["smiles_out"]
            ),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers if torch.cuda.is_available() else 0,
            persistent_workers=self.persistent_workers
            if torch.cuda.is_available()
            else False,
            pin_memory=torch.cuda.is_available(),
        )

    def predict_dataloader(self):
        return DataLoader(
            BasicDataset2(
                self.predict, x_cols=["smiles_in"], y_cols=["smiles_out"]
            ),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers if torch.cuda.is_available() else 0,
            persistent_workers=self.persistent_workers
            if torch.cuda.is_available()
            else False,
            pin_memory=torch.cuda.is_available(),
        )

    @staticmethod
    def combine_input_output(
        input: pd.DataFrame, output: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.merge(input, output, left_index=True, right_index=True)

    @staticmethod
    def check_prepared_files(dir: str, stages: List[str]) -> bool:
        files = [f for f in listdir(dir) if isfile(join(dir, f))]
        required_files = [f"{stage}.csv" for stage in stages]
        return all(file in files for file in required_files)
