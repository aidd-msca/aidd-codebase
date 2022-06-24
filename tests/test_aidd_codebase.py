import pytorch_lightning
import rdkit
import torch
from aidd_codebase import __version__


def test_version():
    assert __version__ == "0.1.4"


def test_dependencies():
    # TODO fix
    assert rdkit.__version__ == "0.1.1"
    assert torch.__version__ == "0.1.1"
    assert pytorch_lightning.__version__ == "1.6.0"
