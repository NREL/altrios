from pkg_resources import get_distribution
__version__ = get_distribution("altrios").version

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

# creates a list of all python classes from rust structs that need variable_path_list() and
# history_path_list() added as methods
ACCEPTED_RUST_STRUCTS = [
    attr for attr in altrios_pyo3.__dir__() if not attr.startswith("__") and isinstance(getattr(altrios_pyo3, attr), type) and
    attr[0].isupper() 
]

def variable_path_list(self, element_as_list:bool=False) -> List[str]:
    """
    Returns list of key paths to all variables and sub-variables within
    dict version of `self`. See example usage in `altrios/demos/
    demo_variable_paths.py`.

    # Arguments:  
    - `element_as_list`: if True, each element is itself a list of the path elements
    """
    return variable_path_list_from_py_objs(self.to_pydict(), element_as_list=element_as_list)
                                        
def variable_path_list_from_py_objs(
    obj: Union[Dict, List], 
    pre_path:Optional[str]=None,
    element_as_list:bool=False,
) -> List[str]:
    """
    Returns list of key paths to all variables and sub-variables within
    dict version of class. See example usage in `altrios/demos/
    demo_variable_paths.py`.

    # Arguments:  
    - `obj`: altrios object in dictionary form from `to_pydict()`
    - `pre_path`: This is used to call the method recursively and should not be
        specified by user.  Specifies a path to be added in front of all paths
        returned by the method.
    - `element_as_list`: if True, each element is itself a list of the path elements
    """
    key_paths = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            # check for nested dicts and call recursively
            if isinstance(val, dict):
                key_path = f"['{key}']" if pre_path is None else pre_path + f"['{key}']"
                key_paths.extend(variable_path_list_from_py_objs(val, key_path))
            # check for lists or other iterables that do not contain numeric data
            elif "__iter__" in dir(val) and (len(val) > 0) and not(isinstance(val[0], float) or isinstance(val[0], int)):
                key_path = f"['{key}']" if pre_path is None else pre_path + f"['{key}']"
                key_paths.extend(variable_path_list_from_py_objs(val, key_path))
            else:
                key_path = f"['{key}']" if pre_path is None else pre_path + f"['{key}']"
                key_paths.append(key_path)
                
    elif isinstance(obj, list):
        for key, val in enumerate(obj):
            # check for nested dicts and call recursively
            if isinstance(val, dict):
                key_path = f"[{key}]" if pre_path is None else pre_path + f"[{key}]"
                key_paths.extend(variable_path_list_from_py_objs(val, key_path))
            # check for lists or other iterables that do not contain numeric data
            elif "__iter__" in dir(val) and (len(val) > 0) and not(isinstance(val[0], float) or isinstance(val[0], int)):
                key_path = f"[{key}]" if pre_path is None else pre_path + f"[{key}]"
                key_paths.extend(variable_path_list_from_py_objs(val, key_path))
            else:
                key_path = f"[{key}]" if pre_path is None else pre_path + f"[{key}]"
                key_paths.append(key_path)
    if element_as_list:
        re_for_elems = re.compile("\\[('(\\w+)'|(\\w+))\\]")
        for i, kp in enumerate(key_paths):
            kp: str
            groups = re_for_elems.findall(kp)
            selected = [g[1] if len(g[1]) > 0 else g[2] for g in groups]
            key_paths[i] = selected
    
    return key_paths

def history_path_list(self, element_as_list:bool=False) -> List[str]:
    """
    Returns a list of relative paths to all history variables (all variables
    that contain history as a subpath). 
    See example usage in `altrios/demo_data/demo_variable_paths.py`.

    # Arguments
    - `element_as_list`: if True, each element is itself a list of the path elements
    """
    item_str = lambda item: item if not element_as_list else ".".join(item)
    history_path_list = [
        item for item in self.variable_path_list(
            element_as_list=element_as_list) if "history" in item_str(item)
    ]
    return history_path_list
            
def to_pydict(self) -> Dict:
    """
    Returns self converted to pure python dictionary with no nested Rust objects
    """
    from yaml import load
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    pydict = load(self.to_yaml(), Loader = Loader)
    return pydict

@classmethod
def from_pydict(cls, pydict: Dict) -> Self:
    """
    Instantiates Self from pure python dictionary 
    """
    import yaml
    return cls.from_yaml(yaml.dump(pydict),skip_init=False)

def to_dataframe(self, pandas:bool=False) -> [pd.DataFrame, pl.DataFrame, pl.LazyFrame]:
    """
    Returns time series results from altrios object as a Polars or Pandas dataframe.

    # Arguments
    - `pandas`: returns pandas dataframe if True; otherwise, returns polars dataframe by default
    """
    obj_dict = self.to_pydict()
    history_paths = self.history_path_list(element_as_list=True)   
    cols = [".".join(hp) for hp in history_paths]
    vals = []
    for hp in history_paths:
        obj:Union[dict|list] = obj_dict
        for elem in hp:
            try: 
                obj = obj[elem]
            except:
                obj = obj[int(elem)]
        vals.append(obj)
    if not pandas:
        df = pl.DataFrame({col: val for col, val in zip(cols, vals)})
    else:
        df = pd.DataFrame({col: val for col, val in zip(cols, vals)})
    return df

# adds variable_path_list() and history_path_list() as methods to all classes in
# ACCEPTED_RUST_STRUCTS
for item in ACCEPTED_RUST_STRUCTS:
    setattr(getattr(altrios_pyo3, item), "variable_path_list", variable_path_list)
    setattr(getattr(altrios_pyo3, item), "history_path_list", history_path_list)
    setattr(getattr(altrios_pyo3, item), "to_pydict", to_pydict)
    setattr(getattr(altrios_pyo3, item), "from_pydict", from_pydict)
    setattr(getattr(altrios_pyo3, item), "to_dataframe", to_dataframe)

setattr(ReversibleEnergyStorage, "from_excel", classmethod(_res_from_excel))  # noqa: F405
setattr(Pyo3VecWrapper, "__array__", __array__)  # noqa: F405
setattr(Pyo3Vec2Wrapper, "__array__", __array__)
setattr(Pyo3Vec3Wrapper, "__array__", __array__)
setattr(Pyo3VecBoolWrapper, "__array__", __array__)
