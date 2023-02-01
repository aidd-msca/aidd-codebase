from torch import optim

from aidd_codebase.registries import AIDD


# Registers choices for loss functions of prebuilt loss functions
AIDD.ModuleRegistry.register_prebuilt(key="adam", obj=optim.Adam)


# class KarpovOptimizer(optim):
#     def __init__(self, params, lr=0.0001, factor=20, warmup_steps=1600):
#         super().__init__()
#         self.base_lr = lr
#         self.factor = factor
#         self.warmup_steps = warmup_steps

#     def return_optimizer(self, step: int):
#         return max(self.base_lr, self.factor * (min(1.0, step / self.warmup_steps) / max(step, self.warmup_steps)))
# add stochastic ridge regression
# to do snapshot ensembling
# Possibly add simmulated annealing to determine lr


class SnapshotEnsembling:
    """Uses cosine cyclic sheduling, ~20-40 epochs per cycle."""

    pass


class FastGeometricEnsembling:
    """Uses linear piecewise cyclic sheduling, ~2-4 epochs per cycle."""

    pass


class StochasticWeightAveraging:
    """Creates two models:
    - one stores running average of model weights.
    - the other traversing weight space by cyclical learning rate.
    """

    pass
