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

# for param_path_list() method to identify something as a struct so that it
# checks for sub-variables and sub-structs, it must be added to this list:
ACCEPTED_RUST_STRUCTS = ['FuelConverter', 
                         'FuelConverterState', 
                         'FuelConverterStateHistoryVec',
                         'Generator',
                         'GeneratorState',
                         'GeneratorStateHistoryVec',
                         'ReversibleEnergyStorage',
                         'ReversibleEnergyStorageState',
                         'ReversibleEnergyStorageStateHistoryVec',
                         'ElectricDrivetrain',
                         'ElectricDrivetrainState',
                         'ElectricDrivetrainStateHistoryVec',
                         'Locomotive',
                         'LocoParams',
                         'ConventionalLoco',
                         'HybridLoco',
                         'BatteryElectricLoco',
                         'DummyLoco',
                         'LocomotiveState',
                         'LocomotiveStateHistoryVec',
                         'LocomotiveSimulation',
                         'PowerTrace',
                         'Consist',
                         'ConsistState',
                         'ConsistStateHistoryVec',
                         'ConsistSimulation',
                         'SpeedTrace',
                         'SetSpeedTrainSim',
                         'SpeedLimitTrainSim',
                         'LinkIdx',
                         'LinkIdxTime',
                         'TimedLinkPath',
                         'LinkPoint',
                         'Link',
                         'Elev',
                         'Heading',
                         'Location',
                         'Network',
                         'LinkPath',
                         'SpeedSet',
                         'InitTrainState',
                         'TrainState',
                         'TrainStateHistoryVec',
                         'TrainConfig',
                         'TrainType',
                         'TrainParams',
                         'RailVehicle',
                         'TrainSimBuilder',
                         'SpeedLimitTrainSimVec',
                         'EstTimeNet',
                         'Pyo3VecWrapper',
                         'Pyo3Vec2Wrapper',
                         'Pyo3Vec3Wrapper',
                         'Pyo3VecBoolWrapper']

def param_path_list(self, path = "", param_path_list = []) -> list[str]:
    """Returns list of relative paths to all variables and sub-variables within
    class (relative to the class the method was called on) 
    See example usage in param_paths_demo.py.
    Arguments:
    ----------
    path : Defaults to empty string. This is mainly used within the method in
    order to call the method recursively and does not need to be specified by
    user. Specifies a path to be added in front of all paths returned by the
    method.
    param_path_list : Defaults to empty list.  This is mainly used within the
    method in order to call the method recursively and does not need to be
    specified by user. Specifies a list of paths to be appended to the list
    returned by the method.
    """
    variable_list = [attr for attr in self.__dir__() if not attr.startswith("__")\
         and not callable(getattr(self,attr))]
    for variable in variable_list:
        if not type(getattr(self,variable)).__name__ in ACCEPTED_RUST_STRUCTS:
            if path == "":
                variable_path = variable
            else:
                variable_path = path + "." + variable
            param_path_list.append(variable_path)
        elif len([attr for attr in getattr(self,variable).__dir__() if not attr.startswith("__")\
                 and not callable(getattr(getattr(self,variable),attr))]) == 0:
            if path == "":
                variable_path = variable
            else:
                variable_path = path + "." + variable
            param_path_list.append(variable_path)    
        else:
            if path == "":
                variable_path = variable
            else:
                variable_path = path + "." + variable
            param_path_list = getattr(self,variable).param_path_list(
                path=variable_path,
                param_path_list=param_path_list,
            )
    return param_path_list

def history_path_list(self) -> list[str]:
    """Returns a list of relative paths to all history variables (all variables
    that contain history as a subpath). 
    See example usage in param_paths_demo.py."""
    return [item for item in self.param_path_list() if "history" in item]
            

# add param_path_list as an attribute for all Rust structs
setattr(FuelConverter, "param_path_list", param_path_list)
setattr(FuelConverterState, "param_path_list", param_path_list)
setattr(FuelConverterStateHistoryVec, "param_path_list", param_path_list)
setattr(Generator, "param_path_list", param_path_list)
setattr(GeneratorState, "param_path_list", param_path_list)
setattr(GeneratorStateHistoryVec, "param_path_list", param_path_list)
setattr(ReversibleEnergyStorage, "param_path_list", param_path_list)
setattr(ReversibleEnergyStorageState, "param_path_list", param_path_list)
setattr(ReversibleEnergyStorageStateHistoryVec, "param_path_list", param_path_list)
setattr(ElectricDrivetrain, "param_path_list", param_path_list)
setattr(ElectricDrivetrainState, "param_path_list", param_path_list)
setattr(ElectricDrivetrainStateHistoryVec, "param_path_list", param_path_list)

setattr(Locomotive, "param_path_list", param_path_list)
setattr(LocoParams, "param_path_list", param_path_list)
setattr(ConventionalLoco, "param_path_list", param_path_list)
setattr(HybridLoco, "param_path_list", param_path_list)
setattr(BatteryElectricLoco, "param_path_list", param_path_list)
setattr(DummyLoco, "param_path_list", param_path_list)
setattr(LocomotiveState, "param_path_list", param_path_list)
setattr(LocomotiveStateHistoryVec, "param_path_list", param_path_list)
setattr(LocomotiveSimulation, "param_path_list", param_path_list)
setattr(PowerTrace, "param_path_list", param_path_list)

setattr(Consist, "param_path_list", param_path_list)
setattr(ConsistState, "param_path_list", param_path_list)
setattr(ConsistStateHistoryVec, "param_path_list", param_path_list)
setattr(ConsistSimulation, "param_path_list", param_path_list)

setattr(SpeedTrace, "param_path_list", param_path_list)
setattr(SetSpeedTrainSim, "param_path_list", param_path_list)
setattr(SpeedLimitTrainSim, "param_path_list", param_path_list)
setattr(LinkIdx, "param_path_list", param_path_list)
setattr(LinkIdxTime, "param_path_list", param_path_list)
setattr(TimedLinkPath, "param_path_list", param_path_list)
setattr(LinkPoint, "param_path_list", param_path_list)
setattr(Link, "param_path_list", param_path_list)
setattr(Elev, "param_path_list", param_path_list)
setattr(Heading, "param_path_list", param_path_list)
setattr(Location, "param_path_list", param_path_list)
setattr(Network, "param_path_list", param_path_list)
setattr(LinkPath, "param_path_list", param_path_list)
setattr(SpeedSet, "param_path_list", param_path_list)

setattr(InitTrainState, "param_path_list", param_path_list)
setattr(TrainState, "param_path_list", param_path_list)
setattr(TrainStateHistoryVec, "param_path_list", param_path_list)

setattr(TrainConfig, "param_path_list", param_path_list)
setattr(TrainType, "param_path_list", param_path_list)
setattr(TrainParams, "param_path_list", param_path_list)
setattr(RailVehicle, "param_path_list", param_path_list)
setattr(TrainSimBuilder, "param_path_list", param_path_list)
setattr(SpeedLimitTrainSimVec, "param_path_list", param_path_list)
setattr(EstTimeNet, "param_path_list", param_path_list)

setattr(Pyo3VecWrapper, "param_path_list", param_path_list)
setattr(Pyo3Vec2Wrapper, "param_path_list", param_path_list)
setattr(Pyo3Vec3Wrapper, "param_path_list", param_path_list)
setattr(Pyo3VecBoolWrapper, "param_path_list", param_path_list)

# add history_path_list as an attribute for all Rust structs
setattr(FuelConverter, "history_path_list", history_path_list)
setattr(FuelConverterState, "history_path_list", history_path_list)
setattr(FuelConverterStateHistoryVec, "history_path_list", history_path_list)
setattr(Generator, "history_path_list", history_path_list)
setattr(GeneratorState, "history_path_list", history_path_list)
setattr(GeneratorStateHistoryVec, "history_path_list", history_path_list)
setattr(ReversibleEnergyStorage, "history_path_list", history_path_list)
setattr(ReversibleEnergyStorageState, "history_path_list", history_path_list)
setattr(ReversibleEnergyStorageStateHistoryVec, "history_path_list", history_path_list)
setattr(ElectricDrivetrain, "history_path_list", history_path_list)
setattr(ElectricDrivetrainState, "history_path_list", history_path_list)
setattr(ElectricDrivetrainStateHistoryVec, "history_path_list", history_path_list)

setattr(Locomotive, "history_path_list", history_path_list)
setattr(LocoParams, "history_path_list", history_path_list)
setattr(ConventionalLoco, "history_path_list", history_path_list)
setattr(HybridLoco, "history_path_list", history_path_list)
setattr(BatteryElectricLoco, "history_path_list", history_path_list)
setattr(DummyLoco, "history_path_list", history_path_list)
setattr(LocomotiveState, "history_path_list", history_path_list)
setattr(LocomotiveStateHistoryVec, "history_path_list", history_path_list)
setattr(LocomotiveSimulation, "history_path_list", history_path_list)
setattr(PowerTrace, "history_path_list", history_path_list)

setattr(Consist, "history_path_list", history_path_list)
setattr(ConsistState, "history_path_list", history_path_list)
setattr(ConsistStateHistoryVec, "history_path_list", history_path_list)
setattr(ConsistSimulation, "history_path_list", history_path_list)

setattr(SpeedTrace, "history_path_list", history_path_list)
setattr(SetSpeedTrainSim, "history_path_list", history_path_list)
setattr(SpeedLimitTrainSim, "history_path_list", history_path_list)
setattr(LinkIdx, "history_path_list", history_path_list)
setattr(LinkIdxTime, "history_path_list", history_path_list)
setattr(TimedLinkPath, "history_path_list", history_path_list)
setattr(LinkPoint, "history_path_list", history_path_list)
setattr(Link, "history_path_list", history_path_list)
setattr(Elev, "history_path_list", history_path_list)
setattr(Heading, "history_path_list", history_path_list)
setattr(Location, "history_path_list", history_path_list)
setattr(Network, "history_path_list", history_path_list)
setattr(LinkPath, "history_path_list", history_path_list)
setattr(SpeedSet, "history_path_list", history_path_list)

setattr(InitTrainState, "history_path_list", history_path_list)
setattr(TrainState, "history_path_list", history_path_list)
setattr(TrainStateHistoryVec, "history_path_list", history_path_list)

setattr(TrainConfig, "history_path_list", history_path_list)
setattr(TrainType, "history_path_list", history_path_list)
setattr(TrainParams, "history_path_list", history_path_list)
setattr(RailVehicle, "history_path_list", history_path_list)
setattr(TrainSimBuilder, "history_path_list", history_path_list)
setattr(SpeedLimitTrainSimVec, "history_path_list", history_path_list)
setattr(EstTimeNet, "history_path_list", history_path_list)

setattr(Pyo3VecWrapper, "history_path_list", history_path_list)
setattr(Pyo3Vec2Wrapper, "history_path_list", history_path_list)
setattr(Pyo3Vec3Wrapper, "history_path_list", history_path_list)
setattr(Pyo3VecBoolWrapper, "history_path_list", history_path_list)
