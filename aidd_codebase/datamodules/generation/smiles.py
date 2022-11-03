from os import listdir
from os.path import isfile, join
import pathlib
from typing import Any, Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from dataclasses import dataclass, field
from aidd_codebase.data_utils.augmentation import Enumerator
from aidd_codebase.data_utils.collate import Collate
from aidd_codebase.utils.config import _ABCDataClass
from aidd_codebase.datamodules.datachoice import DataChoice
from aidd_codebase.utils.metacoding import CreditType
from aidd_codebase.data_utils.dataprocessors import _AbsProcessor, DataType, ReturnOptions, SmilesProcessor
from aidd_codebase.data_utils.datasets import BasicDataset
from aidd_codebase.utils.directories import validate_or_create_dir
from aidd_codebase.utils.typescripts import Tensor
from torch.utils.data import DataLoader


@DataChoice.register_arguments(call_name="retrosynthesis_pavel")
@dataclass(unsafe_hash=True)
class DataArguments(_ABCDataClass):
    seed: int = 1234
    batch_size: int = 128
    num_workers: int = 8
    persistent_workers: bool = True

    override_prepared_data: bool = False
    prepared_data_dir: Optional[str] = f"{pathlib.Path(__file__).parent.resolve()}/data/saved"

    #partitions: Dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})

    remove_missing: bool = True
    remove_duplicates: bool = False
    canonicalization: bool = True
    enumeration: int = 0  # 10
    enumeration_oversample: int = 0  # 15


@DataChoice.register_choice(call_name="retrosynthesis_pavel", author="Peter Hartog", github_handle="PeterHartog",
                            credit_type=CreditType.NONE)
class SmilesDataModule(pl.LightningDataModule):
    def __init__(
            self,
            tokenizer: Any,
            data_args: DataArguments,
    ):
        super().__init__()

        self.datasplit: Dict[str, pd.DataFrame] = {}

        self.prepared_data_dir = validate_or_create_dir(data_args.prepared_data_dir)
        self.override_prepared_data = data_args.override_prepared_data

        self.seed = data_args.seed
        self.partitions = {"train": 0.8, "val": 0.1, "test": 0.1} #data_args.partitions
        self.batch_size = data_args.batch_size
        self.num_workers = data_args.num_workers
        self.persistent_workers = data_args.persistent_workers
        self.collate_fn = Collate(tokenizer.pad_idx).simple_collate_fn

        enumerator = Enumerator(
            enumerations=data_args.enumeration,
            seed=data_args.seed,
            oversample=data_args.enumeration_oversample,
            max_len=tokenizer.max_seq_len,
            keep_original=data_args.canonicalization,
        )
        if data_args.enumeration > 0:
            augmentations = [lambda x: enumerator.dataframe_enumeration(x)]
        else:
            augmentations = None

        self.input_processor = SmilesProcessor(
            type=DataType.SEQUENCE,
            return_type=ReturnOptions.CLEANED
            if data_args.canonicalization
            else ReturnOptions.Original,
            tokenizer=tokenizer,
            remove_duplicates=data_args.remove_duplicates,
            remove_missing=data_args.remove_missing,
            constrains=[lambda x: x.applymap(len) <= tokenizer.max_seq_len],
            augmentations=augmentations,
        )
        self.output_processor = SmilesProcessor(
            type=DataType.SEQUENCE,
            return_type=ReturnOptions.Original,
            tokenizer=tokenizer,
            remove_duplicates=data_args.remove_duplicates,
            remove_missing=data_args.remove_missing,
            constrains=[lambda x: x.applymap(len) <= tokenizer.max_seq_len],
            augmentations=None,
        )

    def process_data(
            self, data: pd.DataFrame, data_processor: _AbsProcessor
    ) -> pd.DataFrame:
        data_processor.set_data(data)
        data_processor.inspect_data()
        data_processor.clean_data()
        data_processor.clean_report()
        return data_processor.return_data()

    def split_data(
            self,
            input: pd.DataFrame,
            output: pd.DataFrame,
            partitions: Dict[str, float],
    ) -> None:
        assert len(input) == len(
            output
        ), "The input length must match the output length."
        assert (
                sum(partitions.values()) <= 1
        ), "Combined partitions should be less than or equal to 1"

        df = self.combine_input_output(input, output)
        corrected_fractions = self.correct_fractions(list(partitions.values()))

        datasets: List = []
        sampled_idxs: List = []
        for frac in corrected_fractions:
            fraction_dataset = df.drop(sampled_idxs).sample(
                frac=frac, random_state=self.seed
            )
            datasets.append(fraction_dataset)
            sampled_idxs.extend(fraction_dataset.index)

        self.datasplit.update(zip(list(partitions.keys()), datasets))

        print(
            ", ".join(
                [f"{k.upper()}: {len(v)}" for k, v in self.datasplit.items()]
            )
        )

    def prepare_data(self, **kwargs) -> None:
        print("Starting input data download...")
        url = 'https://raw.githubusercontent.com/bigchem/retrosynthesis/master/data/retrosynthesis-all.smi'
        reactions = pd.read_csv(url, header=None, names=["reaction"])
        reactions = reactions.applymap(lambda x: x.split(">>"))
        reactions = pd.DataFrame(reactions["reaction"].to_list(), columns=['product', 'reactants'])

        self.input = reactions.loc[:, ["product"]]
        print(self.input.head())
        self.input = self.process_data(self.input, self.input_processor)

        self.target = reactions.loc[:, ["reactants"]]
        print(self.target.head())
        self.output = self.process_data(self.target, self.output_processor)

        self.split_data(
            self.input, self.output, partitions=self.partitions
        )

    def save_prepared_data(self, data: pd.DataFrame, stage: str) -> None:
        input = data.loc[:, (slice(None), "input")].droplevel(level=1, axis=1)
        output = data.loc[:, (slice(None), "output")].droplevel(
            level=1, axis=1
        )
        input.to_csv(
            f"{self.prepared_data_dir}/{stage}_input.csv", index=False
        )
        output.to_csv(
            f"{self.prepared_data_dir}/{stage}_output.csv", index=False
        )

    def load_prepared_data(self, stage: str) -> pd.DataFrame:
        print(f"Loading {stage}...")
        input = pd.read_csv(f"{self.prepared_data_dir}/{stage}_input.csv")
        output = pd.read_csv(f"{self.prepared_data_dir}/{stage}_output.csv")
        return self.combine_input_output(input, output)

    def transform_data(
            self,
            data: pd.DataFrame,
    ) -> pd.DataFrame:
        input = data.loc[:, (slice(None), "input")].droplevel(level=1, axis=1)
        output = data.loc[:, (slice(None), "output")].droplevel(
            level=1, axis=1
        )
        input = self.input_processor.transform_data(input)
        output = self.output_processor.transform_data(output)
        return self.combine_input_output(input, output)

    def augment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        input = data.loc[:, (slice(None), "input")].droplevel(level=1, axis=1)
        output = data.loc[:, (slice(None), "output")].droplevel(
            level=1, axis=1
        )
        input = self.input_processor.augment_data(input)
        output = self.output_processor.augment_data(output)
        return self.combine_input_output(input, output)

    def setup_data(self, data: pd.DataFrame, stage: str) -> pd.DataFrame:
        print(f"Setting up {stage} stage.")
        print("Augmenting data...")
        data = self.augment_data(data)
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
        partition_data = self.transform_data(partition_data)
        return partition_data

    def setup(self, stage: Optional[str] = None) -> None:
        print("Starting Data Module Setup...")

        if stage in ("fit", None):
            self.train = self.get_data(partition="train")
            self.val = self.get_data(partition="val")

        if stage in ("test", None):
            self.test = self.get_data(partition="test")

        print("Finished Data Module!")

    def train_dataloader(self):
        return DataLoader(
            BasicDataset(self.train),
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
            BasicDataset(self.val),
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
            BasicDataset(self.test),
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
        input.columns = pd.MultiIndex.from_product([input.columns, ["input"]])
        output.columns = pd.MultiIndex.from_product(
            [output.columns, ["output"]]
        )
        return pd.merge(input, output, left_index=True, right_index=True)

    @staticmethod
    def check_prepared_files(dir: str, stages: List[str]) -> bool:
        files = [f for f in listdir(dir) if isfile(join(dir, f))]
        opties = ["input.csv", "output.csv"]
        required_files = ["_".join([s, o]) for s in stages for o in opties]
        return all(file in files for file in required_files)

    @staticmethod
    def correct_fractions(fractions: List[float]) -> List[float]:
        """Corrects the fractions to the fraction
        of the data that is left after previous fractions are removed.
        """
        corrected_fractions = [
            frac / (1.0 - sum(fractions[:idx]))
            for idx, frac in enumerate(fractions)
        ]
        # to prevent weird error of 0.1/0.1 > 1.0
        corrected_fractions = [
            frac if frac <= 1.0 else 1.0 for frac in corrected_fractions
        ]
        return corrected_fractions
