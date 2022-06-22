from typing import Callable, Dict

from aidd_codebase.utils.config import _ABCDataClass
from aidd_codebase.utils.metacoding import Accreditation, DictChoiceFactory


class LoopChoice(DictChoiceFactory):
    pass


class IndependentLoopChoice(DictChoiceFactory):
    dict_choice: Dict[str, Callable] = {}
    choice_arguments: Dict[str, _ABCDataClass] = {}
    accreditation: Accreditation = Accreditation()
