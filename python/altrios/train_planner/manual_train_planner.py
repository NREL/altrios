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
    consist_plan: pl.DataFrame,
    loco_map: Dict[str, str],
    rail_vehicles: List[alt.RailVehicle],
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
    - `consist_plan`:
        consist_plan = pl.DataFrame(schema=
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
                consist_plan.get_column("Origin_ID"),
                consist_plan.get_column("Destination_ID"),
            ]
        )
        .unique()
        .sort()
    )

    loco_pool = build_loco_pool_from_model_list(
        consist_plan, config
    )

    # build refueling or recharging infrastructure at all nodes
    refuelers = data_prep.build_refuelers(
        node_list,
        loco_pool,
        config.refueler_info,
        config.refuelers_per_incoming_corridor,
    )
    
    # Create mapping from freight types to car types
    freight_type_to_car_type = {}
    for rv in rail_vehicles:
        # Check for duplicate mappings (should not happen)
        if rv.freight_type in freight_type_to_car_type:
            assert f"More than one rail vehicle car type for freight type {rv.freight_type}"
        else:
            # Map this freight type to its car type
            freight_type_to_car_type[rv.freight_type] = rv.car_type

    for this_train in consist_plan.unique(subset=["Train_ID"], maintain_order=True, keep='first').iter_rows(named=True):
    # Create train simulation builder
                    # Calculate drag coefficient if configured
                    if config.drag_coeff_function is not None:
                        # Apply drag coefficient function based on number of cars
                        cd_area_vec = config.drag_coeff_function(
                            int(this_train["Number_of_Cars"])
                        )
                    else:
                        # No custom drag coefficient
                        cd_area_vec = None

                    # Configure rail vehicles for this train
                    rv_to_use, n_cars_by_type = data_prep.configure_rail_vehicles(
                        this_train, rail_vehicles, freight_type_to_car_type
                    )

                    # Create train configuration
                    train_config = alt.TrainConfig(
                        rail_vehicles=rv_to_use,
                        n_cars_by_type=n_cars_by_type,
                        train_type=this_train["Train_Type"],
                        cd_area_vec=cd_area_vec,
                    )

                    tsb = alt.TrainSimBuilder(
                        train_id=this_train[0],
                        origin_id=this_train["Origin_ID"],
                        destination_id=this_train["Destination_ID"],
                        train_config=train_config,
                        loco_con=loco_con,
                        init_train_state=init_train_state,
                    )

                    # Create speed-limited train simulation
                    slts = tsb.make_speed_limit_train_sim(
                        location_map=location_map,
                        save_interval=None,
                        simulation_days=config.simulation_days,
                        scenario_year=scenario_year,
                    )

    return (
        consist_plan,
        loco_pool,
        refuelers,
        speed_limit_train_sims,
        est_time_nets,
    )


def build_loco_pool_from_model_list(
    consist_plan: pl.DataFrame,
    config: TrainPlannerConfig,
) -> pl.DataFrame:

    unique_consist_plan_locos = consist_plan.unique(subset=["Locomotive_ID", "Locomotive_Type"], maintain_order=True, keep='first')
    loco_pool = unique_consist_plan_locos[["Locomotive_ID", "Locomotive_Type", "Origin_ID"]]
    loco_pool = loco_pool.rename({'Origin_ID':'Node'})
    
    # TODO: check this line


    loco_pool = loco_pool.with_columns(pl.lit(0.0).alias("Arrival_Time"), 
                                       pl.lit(0.0).alias("Servicing_Done_Time"),
                                       pl.lit(0.0).alias("Refueling_Done_Time"),
                                       pl.lit("Ready").alias("Status"),
                                       pl.lit(0.0).alias("SOC_Target_J"),
                                       pl.lit(0.0).alias("Refuel_Duration"),
                                       pl.lit(0.0).alias("Refueler_J_Per_Hr"),
                                       pl.lit(0.0).alias("Refueler_Efficiency"),
                                       pl.lit(0.0).alias("Port_Count"),
                                       )
    
    loco_pool = loco_pool.cast({"Locomotive_Type" : pl.Categorical,
                                "Node" : pl.Categorical,
                                })
        # [
        #     "": pl.Series(np.zeros(rows), dtype=pl.Float64),
        #     "Servicing_Done_Time": pl.Series(np.zeros(rows), dtype=pl.Float64),
        #     "Refueling_Done_Time": pl.Series(np.tile(0, rows), dtype=pl.Float64),
        #     "Status": pl.Series(np.tile("Ready", rows), dtype=pl.Categorical),
        #     "SOC_Target_J": pl.Series(np.zeros(rows), dtype=pl.Float64),
        #     "Refuel_Duration": pl.Series(np.zeros(rows), dtype=pl.Float64),
        #     "Refueler_J_Per_Hr": pl.Series(np.zeros(rows), dtype=pl.Float64),
        #     "Refueler_Efficiency": pl.Series(np.zeros(rows), dtype=pl.Float64),
        #     "Port_Count": pl.Series(np.zeros(rows), dtype=pl.UInt32),
        # ]
    

    loco_info_pl = pl.from_pandas(
        config.loco_info.drop(labels="Rust_Loco", axis=1),
        schema_overrides={
            "Locomotive_Type": pl.Categorical,
            "Fuel_Type": pl.Categorical,
        },
    )

    loco_pool = loco_pool.join(loco_info_pl, on="Locomotive_Type")

    return loco_pool
