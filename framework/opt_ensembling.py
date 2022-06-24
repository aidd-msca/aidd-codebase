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
