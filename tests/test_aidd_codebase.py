import pytorch_lightning
import rdkit
import torch
import registry_factory
from aidd_codebase import __version__


def test_version():
    assert __version__ == "0.1.4"


def test_dependencies():
    # TODO fix
    assert registry_factory.__version__ == "0.1.1"
    assert rdkit.__version__ == "0.1.1"
    assert torch.__version__ == "0.1.1"
    assert pytorch_lightning.__version__ == "1.6.0"
