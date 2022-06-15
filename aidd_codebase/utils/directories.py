from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def validate_or_create_dir(dir_path: str) -> str:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


@dataclass
class Directories:
    """Creates some generic directories for a specific run."""

    PROJECT: str
    HOME_DIR: str
    TIME: str = datetime.now().strftime("%H:%M")
    DATE: str = datetime.now().strftime("%Y-%m-%d")

    def __post_init__(self):
        self.DATE_TIME = f"{self.DATE}/{self.TIME}"
        self.RUN_DIR: str = self.dir(f"{self.HOME_DIR}/runs/{self.PROJECT}")
        self.LOG_DIR: str = self.dir(f"{self.RUN_DIR}/logs")
        self.MODEL_DIR: str = self.dir(
            f"{self.RUN_DIR}/models/{self.DATE_TIME}"
        )
        self.CHECKPOINT_DIR: str = self.dir(
            f"{self.RUN_DIR}/checkpoint/{self.DATE_TIME}"
        )
        self.WEIGHTS_DIR: str = self.dir(
            f"{self.RUN_DIR}/weights/{self.DATE_TIME}"
        )

    @staticmethod
    def dir(dir: str) -> str:
        return validate_or_create_dir(dir)
