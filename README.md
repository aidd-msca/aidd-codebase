# AIDD Codebase

![PyPI](https://img.shields.io/pypi/v/flake8-markdown.svg)
![PyPI](https://img.shields.io/pypi/pyversions/flake8-markdown.svg)
![PyPI](https://img.shields.io/github/license/AIDD-ESR1/AIDD-codebase)

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

```
pip install aidd-codebase
```

## Usage

1. __*Configuration*__: The coding framework has a number of argument dataclasses in the file *arguments.py*. This file contains all standard arguments for each of the models. Because they are dataclasses, you can easily adapt them to your own needs. 
<br> 
Does your Seq2Seq adaptation need an extra argument? Import the Seq2SeqArguments from arguments.py, create your own dataclass which inherits it and add your extra argument. <br> <br>
*It is important to note that the order of supplying arguments to a script goes as follows:* <br>
- --flags override config.yaml <br>
- config.yaml overrides default values in arguments.py <br>
- default values from arguments.py are used when no other values are supplied<br>
At the end, it stores all arguments in config.yaml
<br><br>

2. __*Use*__: The coding framework has four main parts: <br>
- utils
- data_utils
- models
- interpretation

These parts should be used 
&nbsp; 

3. __*File Setup*__: The setup of the files in the system is best used as followed:<br>
coding_framework<br> 
|-- ..<br> 
ESR X<br> 
|-- project 1<br> 
  |-- data<br> 
    |-- ..<br> 
  |-- Arguments.py<br> 
  |-- config.yaml<br> 
  |-- main.py<br>
  |-- datamodule.py<br>
  |-- pl_framework.py<br>

## Contributors

All fellows of the AIDD consortium have contributed to the packaged.

## Code of Conduct

Everyone interacting in the codebase, issue trackers, chat rooms, and mailing lists is expected to follow the [PyPA Code of Conduct](https://www.pypa.io/en/latest/code-of-conduct/).

 