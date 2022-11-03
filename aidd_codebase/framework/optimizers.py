from torch import optim

from aidd_codebase.utils.metacoding import DictChoiceFactory


class OptimizerChoice(DictChoiceFactory):
    pass


# Registers choices for loss functions of prebuilt loss functions
OptimizerChoice.register_prebuilt_choice(
    call_name="adam", callable_cls=optim.Adam
)
