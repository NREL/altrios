from pathlib import Path

import numpy as np
from altrios.loaders.powertrain_components import _res_from_excel


from altrios.utilities import set_param_from_path  # noqa: F401
from altrios import utilities as utils

import logging

# make everything in altrios_core_py available here
from altrios.altrios_core_py import *


# Set up logging
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d | %(filename)s#%(lineno)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
utils.enable_logging()


def package_root() -> Path:
    """
    Returns the package root directory.
    """
    path = Path(__file__).parent
    return path


def resources_root() -> Path:
    """
    Returns the resources root directory.
    """
    path = package_root() / "resources"
    return path


def __array__(self):
    return np.array(self.tolist())


setattr(ReversibleEnergyStorage, "from_excel", classmethod(_res_from_excel))  # noqa: F405
setattr(Pyo3VecWrapper, "__array__", __array__)  # noqa: F405
setattr(Pyo3Vec2Wrapper, "__array__", __array__)
setattr(Pyo3Vec3Wrapper, "__array__", __array__)
setattr(Pyo3VecBoolWrapper, "__array__", __array__)
