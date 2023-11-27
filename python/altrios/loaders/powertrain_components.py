from pathlib import Path
from typing import List, Union

import pandas as pd
import numpy as np

from altrios.altrios_core_py import ReversibleEnergyStorage


def _res_from_excel(
    cls: ReversibleEnergyStorage,
    file: Union[str, Path],
    temps: List[int] = [23, 30, 45, 55],
) -> ReversibleEnergyStorage:
    """
    Loads a ReversibleEnergyStorage from an Excel file.
    This function expects the general config to be in a sheet name "config"
    This function expects a sheet for each temperature named "temp_<temp>".

    Args:
        file: The path to the Excel file.
        temps: The temperatures to look for in the file.
    """
    path = Path(file)
    if not path.is_file():
        raise FileNotFoundError(f"could not locate res file: {path}")

    if path.suffix == ".xlsx":
        try:
            import openpyxl
        except ImportError:
            raise ImportError("openpyxl is required for loading Excel files")
        engine = "openpyxl"
    elif path.suffix == ".xls":
        try:
            import xlrd
        except ImportError:
            raise ImportError("xlrd is required for loading Excel files")
        engine = "xlrd"
    else:
        raise ValueError(
            f"unsupported file extension for from_excel: {path.suffix}")

    # pull general config parameters
    config = pd.read_excel(file, sheet_name="config", engine=engine).set_index(
        "parameter"
    )

    pwr_out_max_watts = float(config.loc["pwr_out_max_watts"].value)
    energy_capacity_joules = float(config.loc["energy_capacity_joules"].value)
    min_soc = float(config.loc["min_soc"].value)
    max_soc = float(config.loc["max_soc"].value)
    save_interval = int(config.loc["save_interval"].value)
    initial_soc = float(config.loc["initial_soc"].value)
    initial_temp_c = float(config.loc["initial_temp_c"].value)

    # build a lookup table for the interpolation values
    interp = []
    for t in temps:
        df = pd.read_excel(
            file,
            sheet_name=f"temp_{t}",
            engine="openpyxl",
        )
        udf = df.set_index("c_rate").unstack()
        udf.index = udf.index.rename(["soc", "c_rate"])
        udf = udf.rename("efficiency").reset_index()
        udf["temperature_c"] = t
        interp.append(udf)

    interp_df = pd.concat(interp)

    lookup = interp_df.set_index(["temperature_c", "soc", "c_rate"])

    temp_interp_grid = list(map(float, interp_df.temperature_c.unique()))
    soc_interp_grid = list(map(float, interp_df.soc.unique()))
    c_rate_interp_grid = list(map(float, interp_df.c_rate.unique()))

    tg, sg, cg = np.meshgrid(
        temp_interp_grid, soc_interp_grid, c_rate_interp_grid, indexing="ij"
    )

    eta_interp_values = np.zeros(tg.shape).tolist()

    for i, tup in enumerate(zip(tg, sg, cg)):
        t, s, c = tup
        for j, tup1 in enumerate(zip(t, s, c)):  # type: ignore
            t1, s1, c1 = tup1
            for k, tup2 in enumerate(zip(t1, s1, c1)):
                t2, s2, c2 = tup2
                value = float(lookup.loc[t2, s2, c2].efficiency)
                eta_interp_values[i][j][k] = value

    res = ReversibleEnergyStorage(
        temp_interp_grid,
        soc_interp_grid,
        c_rate_interp_grid,
        eta_interp_values,
        pwr_out_max_watts,
        energy_capacity_joules,
        min_soc,
        max_soc,
        initial_soc,
        initial_temp_c,
        save_interval,
    )

    return res
