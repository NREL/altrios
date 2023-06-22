"""Module for general functions and classes."""

import re
import numpy as np
from typing import Tuple, Union, Optional, Dict, Any, TYPE_CHECKING
import pandas as pd
import datetime
import numpy.typing as npt
import logging
from pathlib import Path
import datetime


from altrios.altrios_core_py import (
    SetSpeedTrainSim,
    ConsistSimulation,
    Consist,
    LocomotiveSimulation,
    Locomotive,
    FuelConverter,
    ReversibleEnergyStorage,
    Generator,
    ElectricDrivetrain,
    PowerTrace,
)

MPS_PER_MPH = 1.0 / 2.237
N_PER_LB = 4.448
KG_PER_LB = 1.0 / 2.20462
W_PER_HP = 745.7
KG_PER_TON = KG_PER_LB * 2000.0
CM_PER_IN = 2.54
CM_PER_FT = CM_PER_IN * 12.0
M_PER_FT = CM_PER_FT / 100.0
LITER_PER_M3 = 1.0e3
G_PER_TONNE = 1.0e6
GALLONS_PER_LITER = 1.0 / 3.79
KWH_PER_MJ = 0.277778 # https://www.eia.gov/energyexplained/units-and-calculators/energy-conversion-calculators.php
MWH_PER_MJ = KWH_PER_MJ / 1.0e3

WARM_START_DAYS = 7


def print_dt():
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def cumutrapz(x, y):
    """
    Returns cumulative trapezoidal integral array for:
    Arguments:
    ----------
    x: array of monotonically increasing values to integrate over
    y: array of values being integrated
    """
    assert len(x) == len(y)
    z = np.zeros(len(x))
    z[0] = 0
    for i in np.arange(1, len(x)):
        z[i] = z[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return z


R_air = np.float64(287)  # J/(kg*K)


def get_rho_air(temperature_degC, elevation_m=180):
    """Returns air density [kg/m**3] for given elevation and temperature.
    Source: https://www.grc.nasa.gov/WWW/K-12/rocket/atmosmet.html
    Arguments:
    ----------
    temperature_degC : ambient temperature [°C]
    elevation_m : elevation above sea level [m].
        Default 180 m is for Chicago, IL"""
    #     T = 15.04 - .00649 * h
    #     p = 101.29 * [(T + 273.1)/288.08]^5.256
    T_standard = 15.04 - 0.00649 * elevation_m  # nasa [degC]
    p = 101.29e3 * ((T_standard + 273.1) / 288.08) ** 5.256  # nasa [Pa]
    rho = p / (R_air * (temperature_degC + 273.15))  # [kg/m**3]

    return rho


def set_param_from_path_dict(mod_dict: dict, path: str, value: float) -> Dict:
    cur_mod_dict = mod_dict
    path_list = path.split(".")

    cur_ptr = cur_mod_dict

    for step in path_list[:-1]:
        cur_ptr = cur_ptr[step]

    cur_ptr[path_list[-1]] = value

    return cur_mod_dict


def set_param_from_path(
    model: Union[
        SetSpeedTrainSim,
        ConsistSimulation,
        Consist,
        LocomotiveSimulation,
        Locomotive,
        FuelConverter,
        ReversibleEnergyStorage,
        Generator,
        ElectricDrivetrain,
        PowerTrace,
    ],
    path: str,
    value: Any,
) -> Union[
    SetSpeedTrainSim,
    ConsistSimulation,
    Consist,
    LocomotiveSimulation,
    Locomotive,
    FuelConverter,
    ReversibleEnergyStorage,
    Generator,
    ElectricDrivetrain,
    PowerTrace,
]:
    """
    Set parameter `value` on `model` for `path` to parameter

    Example usage:
    todo
    """
    path_list = path.split(".")

    def _get_list(path_elem, container):
        list_match = re.match(r"([\w\d]+)\[(\d+)\]", path_elem)
        if list_match is not None:
            list_name = list_match.group(1)
            index = int(list_match.group(2))
            l = container.__getattribute__(list_name).tolist()
            return l, list_name, index
        else:
            return None, None, None

    containers = [model]
    lists = [None] * len(path_list)
    has_list = [False] * len(path_list)
    for i, path_elem in enumerate(path_list):
        container = containers[-1]

        list_attr, list_name, list_index = _get_list(path_elem, container)
        if list_attr is not None:
            attr = list_attr[list_index]
            # save for when we repack the containers
            lists[i] = (list_attr, list_name, list_index)
        else:
            attr = container.__getattribute__(path_elem)

        if i < len(path_list) - 1:
            containers.append(attr)

    prev_container = value

    # iterate through remaining containers, inner to outer
    for list_tuple, container, path_elem in zip(
        lists[-1::-1], containers[-1::-1], path_list[-1::-1]
    ):
        if list_tuple is not None:
            list_attr, list_name, list_index = list_tuple
            list_attr[list_index] = prev_container

            container.__setattr__(list_name, list_attr)
        else:
            container.__setattr__(f"__{path_elem}", prev_container)

        prev_container = container

    return model


def resample(
    df: pd.DataFrame,
    dt_new: Optional[float] = 1.0,
    time_col: Optional[str] = "Time[s]",
    rate_vars: Optional[Tuple[str]] = [],
    hold_vars: Optional[Tuple[str]] = [],
) -> pd.DataFrame:
    """
    Resamples dataframe `df`.
    Arguments:
    - df: dataframe to resample
    - dt_new: new time step size, default 1.0 s
    - time_col: column for time in s
    - rate_vars: list of variables that represent rates that need to be time averaged
    - hold_vars: vars that need zero-order hold from previous nearest time step
        (e.g. quantized variables like current gear)
    """

    new_dict = dict()

    new_time = np.arange(
        0, np.floor(df[time_col].to_numpy()[-1] / dt_new) *
        dt_new + dt_new, dt_new
    )

    for col in df.columns:
        if col in rate_vars:
            # calculate average value over time step
            cumu_vals = (df[time_col].diff().fillna(0) * df[col]).cumsum()
            new_dict[col] = (
                np.diff(
                    np.interp(
                        x=new_time, xp=df[time_col].to_numpy(), fp=cumu_vals),
                    prepend=0,
                )
                / dt_new
            )

        elif col in hold_vars:
            assert col not in rate_vars
            pass  # this may need to be fleshed out

        else:
            # just interpolate -- i.e. state variables like temperatures
            new_dict[col] = np.interp(
                x=new_time, xp=df[time_col].to_numpy(), fp=df[col].to_numpy()
            )

    return pd.DataFrame(new_dict)


def smoothen(signal: npt.ArrayLike, period: int = 9) -> npt.ArrayLike:
    """
    Apply smoothing to signal, assuming 1 Hz data collection.
    """
    new_signal = np.convolve(
        np.concatenate(
            [
                np.full(((period + 1) // 2) - 1, signal[0]),
                signal,
                np.full(period // 2, signal[-1]),
            ]
        ),
        np.ones(period) / period,
        mode="valid",
    )
    return new_signal


def print_dt():
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def set_log_level(level: Union[str, int]):
    # Map string name to logging level
    if isinstance(level, str):
        level = logging._nameToLevel[level]
    # Set logging level
    logging.getLogger("").setLevel(level)


def disable_logging():
    set_log_level(logging.CRITICAL + 1)


def enable_logging():
    set_log_level(logging.WARNING)


