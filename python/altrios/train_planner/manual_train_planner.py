import pandas as pd
import polars as pl
from typing import Dict, Tuple, List
import numpy as np

import altrios as alt
from altrios.train_planner.planner_config import TrainPlannerConfig
from altrios.train_planner import (
    data_prep,
    schedulers,
    planner_config,
    train_demand_generators,
)


def manual_train_planner(
    train_consist_plan: pl.DataFrame,
    loco_map: Dict[str, str],
    config: TrainPlannerConfig = TrainPlannerConfig(),
) -> Tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    List[alt.SpeedLimitTrainSim],
    List[alt.EstTimeNet],
]:
    """
    Generate train simulations from historical data

    # Arguments
    - `train_consist_plan`:
        train_consist_plan = pl.DataFrame(schema=
            {'Train_ID': pl.Int64,
             'Train_Type': pl.Utf8,
             'Locomotive_ID': pl.UInt32,
             'Locomotive_Type': pl.Categorical,
             'Origin_ID': pl.Utf8,
             'Destination_ID': pl.Utf8,
             'Cars_Loaded': pl.Float64,
             'Cars_Empty': pl.Float64,
             'Containers_Loaded': pl.Float64,
             'Containers_Empty': pl.Float64,
             'Departure_SOC_J': pl.Float64,
             'Departure_Time_Planned_Hr': pl.Float64,
             'Arrival_Time_Planned_Hr': pl.Float64})
    - `loco_map`: mapping of test data locomotive types to ALTRIOS locomotive
        types with real world locomotive types as keys and ALTRIOS models as values
    """

    # Append additional locomotive info to configuration.  Not sure why this is important, but I'm copying the regular train planner.
    config.loco_info = data_prep.append_loco_info(config.loco_info)

    # this is a list of all the unique destinations in the consist plan
    node_list = (
        pl.concat(
            [
                train_consist_plan.get_column("Origin_ID"),
                train_consist_plan.get_column("Destination_ID"),
            ]
        )
        .unique()
        .sort()
    )

    loco_pool = build_loco_pool_from_model_list(
        train_consist_plan,
    )

    # build refueling or recharging infrastructure at all nodes
    refuelers = data_prep.build_refuelers(
        node_list,
        loco_pool,
        config.refueler_info,
        config.refuelers_per_incoming_corridor,
    )

    return (
        train_consist_plan,
        loco_pool,
        refuelers,
        speed_limit_train_sims,
        est_time_nets,
    )


def build_loco_pool_from_model_list(
    train_consist_plan: pl.DataFrame,
) -> pl.DataFrame:

    loco_numbers = train_consist_plan["Locomotive_ID"].unique()
    types = train_consist_plan["Locomotive_Type"]
    # TODO: check this line
    sorted_nodes = train_consist_plan["Origin_ID"]
    rows = initial_size * num_nodes  # number of locomotives in total

    loco_pool = pl.DataFrame(
        {
            "Locomotive_ID": pl.Series(loco_numbers, dtype=pl.UInt32),
            "Locomotive_Type": pl.Series(types, dtype=pl.Categorical),
            "Node": pl.Series(sorted_nodes, dtype=pl.Categorical),
            "Arrival_Time": pl.Series(np.zeros(rows), dtype=pl.Float64),
            "Servicing_Done_Time": pl.Series(np.zeros(rows), dtype=pl.Float64),
            "Refueling_Done_Time": pl.Series(np.tile(0, rows), dtype=pl.Float64),
            "Status": pl.Series(np.tile("Ready", rows), dtype=pl.Categorical),
            "SOC_Target_J": pl.Series(np.zeros(rows), dtype=pl.Float64),
            "Refuel_Duration": pl.Series(np.zeros(rows), dtype=pl.Float64),
            "Refueler_J_Per_Hr": pl.Series(np.zeros(rows), dtype=pl.Float64),
            "Refueler_Efficiency": pl.Series(np.zeros(rows), dtype=pl.Float64),
            "Port_Count": pl.Series(np.zeros(rows), dtype=pl.UInt32),
        }
    )

    loco_info_pl = pl.from_pandas(
        config.loco_info.drop(labels="Rust_Loco", axis=1),
        schema_overrides={
            "Locomotive_Type": pl.Categorical,
            "Fuel_Type": pl.Categorical,
        },
    )

    loco_pool = loco_pool.join(loco_info_pl, on="Locomotive_Type")

    return loco_pool
