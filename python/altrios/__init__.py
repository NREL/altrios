from pkg_resources import get_distribution
__version__ = get_distribution("altrios").version

from pathlib import Path
import numpy as np
import logging

from altrios.loaders.powertrain_components import _res_from_excel
from altrios.utilities import set_param_from_path  # noqa: F401
from altrios.utilities import copy_demo_files  # noqa: F401
from altrios import utilities as utils
from altrios.utilities import package_root, resources_root
# make everything in altrios_pyo3 available here
from altrios.altrios_pyo3 import *


# Set up logging
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d | %(filename)s#%(lineno)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
utils.enable_logging()

def __array__(self):
    return np.array(self.tolist())


setattr(ReversibleEnergyStorage, "from_excel", classmethod(_res_from_excel))  # noqa: F405
setattr(Pyo3VecWrapper, "__array__", __array__)  # noqa: F405
setattr(Pyo3Vec2Wrapper, "__array__", __array__)
setattr(Pyo3Vec3Wrapper, "__array__", __array__)
setattr(Pyo3VecBoolWrapper, "__array__", __array__)
