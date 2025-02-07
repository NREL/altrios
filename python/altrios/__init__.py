from importlib.metadata import version
__version__ = version("altrios")

from pathlib import Path
import re
import numpy as np
import logging
import inspect
from typing import List, Union, Dict, Optional
from typing_extensions import Self
import pandas as pd
import polars as pl

from altrios.loaders.powertrain_components import _res_from_excel
from altrios.utilities import set_param_from_path  # noqa: F401
from altrios.utilities import copy_demo_files  # noqa: F401
from altrios import utilities as utils
from altrios.utilities import package_root, resources_root
# make everything in altrios_pyo3 available here
from altrios.altrios_pyo3 import *
from altrios import *

DEFAULT_LOGGING_CONFIG = dict(
    format="%(asctime)s.%(msecs)03d | %(filename)s:%(lineno)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set up logging
logging.basicConfig(**DEFAULT_LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def __array__(self):
    return np.array(self.tolist())

# creates a list of all python classes from rust structs that need python-side serde helpers
ACCEPTED_RUST_STRUCTS = [
    attr for attr in altrios_pyo3.__dir__() if not attr.startswith("__") and isinstance(getattr(altrios_pyo3, attr), type) and
    attr[0].isupper() 
]

# TODO connect to crate features
data_formats = [
    'yaml',
    'msg_pack',
    # 'toml',
    'json',
]

def to_pydict(self, data_fmt: str = "msg_pack", flatten: bool = False) -> Dict:
    """
    Returns self converted to pure python dictionary with no nested Rust objects
    # Arguments
    - `flatten`: if True, returns dict without any hierarchy
    - `data_fmt`: data format for intermediate conversion step
    """
    data_fmt = data_fmt.lower()
    assert data_fmt in data_formats, f"`data_fmt` must be one of {data_formats}"
    match data_fmt:
        case "msg_pack":
            import msgpack
            pydict = msgpack.loads(self.to_msg_pack())
        case "yaml":
            from yaml import load
            try:
                from yaml import CLoader as Loader
            except ImportError:
                from yaml import Loader
            pydict = load(self.to_yaml(), Loader=Loader)
        case "json":
            from json import loads
            pydict = loads(self.to_json())

    if not flatten:
        return pydict
    else:
        return next(iter(pd.json_normalize(pydict, sep=".").to_dict(orient='records')))

@classmethod
def from_pydict(cls, pydict: Dict, data_fmt: str = "msg_pack", skip_init: bool = False) -> Self:
    """
    Instantiates Self from pure python dictionary 
    # Arguments
    - `pydict`: dictionary to be converted to ALTRIOS object
    - `data_fmt`: data format for intermediate conversion step  
    - `skip_init`: passed to `SerdeAPI` methods to control whether initialization
      is skipped
    """
    data_fmt = data_fmt.lower()
    assert data_fmt in data_formats, f"`data_fmt` must be one of {data_formats}"
    match data_fmt.lower():
        case "yaml":
            import yaml
            obj = cls.from_yaml(yaml.dump(pydict), skip_init=skip_init)
        case "msg_pack":
            import msgpack
            obj = cls.from_msg_pack(
                msgpack.packb(pydict), skip_init=skip_init)
        case "json":
            from json import dumps
            obj = cls.from_json(dumps(pydict), skip_init=skip_init)

    return obj

def to_dataframe(self, pandas: bool = False, allow_partial: bool = False) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Returns time series results from fastsim object as a Polars or Pandas dataframe.

    # Arguments
    - `pandas`: returns pandas dataframe if True; otherwise, returns polars dataframe by default
    - `allow_partial`: returns dataframe of length equal to solved time steps if simulation fails early
    """
    obj_dict = self.to_pydict(flatten=True)
    history_dict = {}
    history_keys = [
        'history',
        'speed_trace',
        'power_trace'
    ]
    for k, v in obj_dict.items():
        if any([hk in k for hk in history_keys]):
            history_dict[k] = v
    try:
        hd_len = len(next(iter(history_dict.values())))
    except StopIteration as err:
        raise Exception(f"{err}\n`history_dict` should not be empty by this point")
  
    for k, v in obj_dict.items():
        if ("__len__" in dir(v)) and (len(v) == hd_len):
            history_dict[k] = v

    if allow_partial:
        cutoff = min([len(val) for val in history_dict.values()])

        if not pandas:
            df = pl.DataFrame({col: val[:cutoff]
                              for col, val in history_dict.items()})
        else:
            df = pd.DataFrame({col: val[:cutoff]
                              for col, val in history_dict.items()})
    else:
        if not pandas:
            try:
                df = pl.DataFrame(history_dict)
            except Exception as err:
                raise (
                    f"{err}\nTry passing `allow_partial=True` to `to_dataframe`")
        else:
            try:
                df = pd.DataFrame(history_dict)
            except Exception as err:
                raise (
                    f"{err}\nTry passing `allow_partial=True` to `to_dataframe`")
    return df

# adds variable_path_list() and history_path_list() as methods to all classes in
# ACCEPTED_RUST_STRUCTS
for item in ACCEPTED_RUST_STRUCTS:
    setattr(getattr(altrios_pyo3, item), "to_pydict", to_pydict)
    setattr(getattr(altrios_pyo3, item), "from_pydict", from_pydict)
    setattr(getattr(altrios_pyo3, item), "to_dataframe", to_dataframe)

setattr(ReversibleEnergyStorage, "from_excel", classmethod(_res_from_excel))  # noqa: F405
setattr(Pyo3VecWrapper, "__array__", __array__)  # noqa: F405
setattr(Pyo3Vec2Wrapper, "__array__", __array__)
setattr(Pyo3Vec3Wrapper, "__array__", __array__)
setattr(Pyo3VecBoolWrapper, "__array__", __array__)
