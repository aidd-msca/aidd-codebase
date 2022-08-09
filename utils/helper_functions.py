from typing import Any

import dill


def detect_unpickle(object: Any) -> None:
    print(dill.detect.baditems(object))


def detect_pickle_error(object: Any) -> None:
    dill.detect.trace(True)
    print(dill.detect.errors(object))
