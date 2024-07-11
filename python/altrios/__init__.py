from pkg_resources import get_distribution
__version__ = get_distribution("altrios").version

from pathlib import Path
import numpy as np
import logging
import inspect

import altrios as alt
from altrios.loaders.powertrain_components import _res_from_excel
from altrios.utilities import set_param_from_path  # noqa: F401
from altrios.utilities import copy_demo_files  # noqa: F401
from altrios import utilities as utils
from altrios.utilities import package_root, resources_root
# make everything in altrios_pyo3 available here
from altrios.altrios_pyo3 import *

DEFAULT_LOGGING_CONFIG = dict(
    format="%(asctime)s.%(msecs)03d | %(filename)s:%(lineno)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set up logging
logging.basicConfig(**DEFAULT_LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def __array__(self):
    return np.array(self.tolist())


setattr(ReversibleEnergyStorage, "from_excel", classmethod(_res_from_excel))  # noqa: F405
setattr(Pyo3VecWrapper, "__array__", __array__)  # noqa: F405
setattr(Pyo3Vec2Wrapper, "__array__", __array__)
setattr(Pyo3Vec3Wrapper, "__array__", __array__)
setattr(Pyo3VecBoolWrapper, "__array__", __array__)

# creates a list of all python classes from rust structs that need variable_path_list() and
# history_path_list() added as methods
ACCEPTED_RUST_STRUCTS = [attr for attr in alt.__dir__() if not\
                         attr.startswith("__") and\
                            isinstance(getattr(alt,attr), type) and\
                                attr[0].isupper() and\
                                    ("altrios" in str(inspect.getmodule(getattr(alt,attr))))]

def variable_path_list(self, path = "") -> list[str]:
    """Returns list of relative paths to all variables and sub-variables within
    class (relative to the class the method was called on).  
    See example usage in param_paths_demo.py.
    Arguments:
    ----------
    path : Defaults to empty string. This is mainly used within the method in
    order to call the method recursively and should not be specified by user. 
    Specifies a path to be added in front of all paths returned by the method.
    """
    variable_paths = []
    if "tolist" in self.__dir__():
        print(type(self.tolist()))
        if len(self.tolist()) != 0:
            for counter in range(len(self.tolist())):
                print(self.tolist()[counter])
                if self.tolist()[counter] == None:
                    if path == "":
                        variable_path = "tolist()[" + str(counter) + "]"
                    else:
                        variable_path = path + ".tolist()[" + str(counter) + "]"
                    variable_paths.append(variable_path)
                elif not type(self.tolist()[counter]).__name__ in ACCEPTED_RUST_STRUCTS:
                    if path == "":
                        variable_path = "tolist()[" + str(counter) + "]"
                    else:
                        variable_path = path + ".tolist()[" + str(counter) + "]"
                    variable_paths.append(variable_path)
                else:            
                    if path == "":
                        variable_path = "tolist()[" + str(counter) + "]"
                    else:
                        variable_path = path + ".tolist()[" + str(counter) + "]"
                    variable_paths.extend(self.tolist()[counter].variable_path_list(path=variable_path))
        else:
            variable_path = path + ".to_list()"
            variable_paths.append(variable_path)
    else:
        variable_list = [attr for attr in self.__dir__() if not attr.startswith("__")]
        # getattr(self,attr) I think returns a runtime error if the attr is an option that turns out to be 
        # None -- how to I get around this??
        variable_list = [attr for attr in variable_list if not getattr(self,attr) is None]
        variable_list = [attr for attr in variable_list if not callable(getattr(self,attr))]
        # variable_list = [attr for attr in variable_list if not attr.startswith("__")\
        #                  and not getattr(self,attr) == None and not callable(getattr(self,attr))]
        for variable in variable_list:
            if not type(getattr(self,variable)).__name__ in ACCEPTED_RUST_STRUCTS:
                if path == "":
                    variable_path = variable
                else:
                    variable_path = path + "." + variable
                variable_paths.append(variable_path)
            # elif "tolist" in getattr(self,variable).__dir__():
            #     print(variable, "dir with tolist:", getattr(self,variable).__dir__())
            #     for counter in 0..variable.tolist().len():
            #         if not type(getattr(self,variable).tolist()[counter]).__name__ in ACCEPTED_RUST_STRUCTS:
            #             if path == "":
            #                 variable_path = variable
            #             else:
            #                 variable_path = path + "." + variable
            #             variable_paths.append(variable_path)
            #         else:            
            #             if path == "":
            #                 variable_path = variable + ".tolist()[" + str(counter) + "]"
            #             else:
            #                 variable_path = path + "." + variable + ".tolist()[" + str(counter) + "]"
            #             variable_paths.extend(getattr(self,variable).tolist()[counter].variable_path_list(path=variable_path))
            # elif len([attr for attr in getattr(self,variable).__dir__() if not attr.startswith("__")\
            #         and not callable(getattr(getattr(self,variable),attr))]) == 0\
            #               and "tolist" not in getattr(self,variable).__dir__():
            #     if path == "":
            #         variable_path = variable
            #     else:
            #         variable_path = path + "." + variable
            #     variable_paths.append(variable_path)    
            else:
                if path == "":
                    variable_path = variable
                else:
                    variable_path = path + "." + variable
                print(variable_path, "got through to variable_path_list iteration.")
                variable_paths.extend(getattr(self,variable).variable_path_list(path=variable_path))
    return variable_paths

def history_path_list(self) -> list[str]:
    """Returns a list of relative paths to all history variables (all variables
    that contain history as a subpath). 
    See example usage in param_paths_demo.py."""
    return [item for item in self.variable_path_list() if "history" in item]
            

# adds variable_path_list() and history_path_list() as methods to all classes in
# ACCEPTED_RUST_STRUCTS
for item in ACCEPTED_RUST_STRUCTS:
    setattr(getattr(alt, item), "variable_path_list", variable_path_list)
    setattr(getattr(alt, item), "history_path_list", history_path_list)
