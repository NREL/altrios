import pandas as pd
import polars as pl
from typing import Dict, Tuple, List

import altrios as alt
from altrios import defaults
from altrios.train_planner.planner_config import TrainPlannerConfig


def manual_train_planner(
    train_consist_plan: pl.DataFrame,
    loco_map: Dict[str, str],
    config: TrainPlannerConfig = TrainPlannerConfig(),
) -> Tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    List[alt.SpeedLimitTrainSim],
    List[alt.EstTimeNet]
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

    node_list = pl.concat(
        [train_consist_plan.get_column("Origin_ID"),
        train_consist_plan.get_column("Destination_ID")]).unique().sort()
    loco_pool = build_loco_pool_from_model_list() # TODO: make this function

    refuelers = data_prep.build_refuelers(
        node_list, 
        loco_pool,
        config.refueler_info, 
        config.refuelers_per_incoming_corridor)
    
    return (
        train_consist_plan,
        loco_pool,
        refuelers,
        speed_limit_train_sims,
        est_time_nets
    )
