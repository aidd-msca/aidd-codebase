from typing import Callable, Dict
from aidd_codebase.utils.metacoding import Accreditation, DictChoiceFactory
from aidd_codebase.utils.config import _ABCDataClass

class LoopChoice(DictChoiceFactory):
    dict_choice: Dict[str, Callable] = {}
    choice_arguments: Dict[str, _ABCDataClass] = {}
    accreditation: Accreditation = Accreditation()
