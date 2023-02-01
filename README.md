# AIDD Codebase

![PyPI](https://img.shields.io/pypi/v/aidd-codebase)
![PyPI](https://img.shields.io/pypi/pyversions/aidd-codebase)
![PyPI](https://img.shields.io/github/license/aidd-msca/aidd-codebase)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jlyEd1yxhvFCN82YqEFI82q2n0k_y06F?usp=sharing)

A high-level codebase for deep learning development in drug discovery applications using PyTorch-Lightning.

## Dependencies

The codebase requires the following additional dependencies
- CUDA >= 11.4
- PyTorch >= 1.9
- Pytorch-Lightning >= 1.5 
- RDKit 
- Optionally supports: tensorboard and/or wandb


## Installation

The codebase can be installed from PyPI using `pip`, or your package manager of choice, with

```bash
$ pip install aidd-codebase
```

## Usage

The codebase is designed to be used in a modular fashion. The main components are the `DataModule`, `Model`, and `Trainer` classes. The `DataModule` is responsible for loading and preprocessing data, the `Model` is responsible for defining the model architecture, and the `Trainer` is responsible for training the model. The `Trainer` is a subclass of `pytorch_lightning.Trainer` and can be used as such. The `DataModule` and `Model` classes are designed to be used with the `Trainer` class, but can be used independently if desired.

### Starting a new project
```bash
$ python -m aidd_codebase.start_project name dir_path
```
This will create a new project folder with the following structure:

```
name
├── conf
│   └── config.yaml
├── src
└── main.py
```

The `conf` folder contains the configuration file for the project. The `src` folder contains the source code for the project. The `main.py` file is the entry point for the project.


## Contributors

All fellows of the AIDD consortium have contributed to the packaged.

## Code of Conduct

Everyone interacting in the codebase, issue trackers, chat rooms, and mailing lists is expected to follow the [PyPA Code of Conduct](https://www.pypa.io/en/latest/code-of-conduct/).
