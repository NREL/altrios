"""
Module for getting the output of the Train Consist Planner and Meet Pass Planner to run 3 week simulation.
"""

import polars as pl
from typing import Any, Union, Dict, List, Optional, Tuple
from pathlib import Path
import time
from altrios import defaults


# from __future__ import annotations # TODO: uncomment and propagate
import altrios as alt
from altrios import train_planner as planner
from altrios import metric_calculator as metrics

def main(
    rail_vehicle_map: Dict[str, alt.RailVehicle],
    location_map: Dict[str, List[alt.Location]],
    network: List[alt.Link],
    simulation_days: int = defaults.SIMULATION_DAYS,
    scenario_year: int = defaults.BASE_ANALYSIS_YEAR,
    target_bel_share: float = 0.5,
    debug: bool = False,
    loco_pool: pl.DataFrame = None,
    refuelers: pl.DataFrame = None,
    grid_emissions_factors: pl.DataFrame = None,
    nodal_energy_prices: pl.DataFrame = None,
    train_planner_config: planner.TrainPlannerConfig = planner.TrainPlannerConfig(),
    demand_file_path: str = str(defaults.DEMAND_FILE),
    network_charging_guidelines: pl.DataFrame = None
) -> Tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    alt.SpeedLimitTrainSimVec,
    List[List[alt.LinkIdxTime]],
]:
    """
    # Return
    ```
    return (
        train_consist_plan,
        loco_pool,
        refuelers,
        grid_emissions_factors,
        nodal_energy_prices,
        train_sims,
        timed_paths,
    )
    ```

    """

    if debug:
        print("Entering `sim_manager` module.")
        alt.utils.print_dt()
        print(demand_file_path)

    for loc_name in location_map:
        for loc in location_map[loc_name]:
            if loc.link_idx.idx >= len(network):
                raise ValueError("Location " + loc.location_id + " with link index " +
                                 str(loc.link_idx.idx) + " is invalid for network!")
            
    train_planner_config.loco_info = metrics.add_battery_costs(train_planner_config.loco_info, scenario_year)
            
    if loco_pool is None: loco_pool = planner.build_locopool(
            config = train_planner_config,
            method="shares_twoway",
            shares=[1-target_bel_share, target_bel_share],
            demand_file=demand_file_path
            )

    t0_ptc = time.perf_counter()
    (
        train_consist_plan, 
        loco_pool, 
        refuelers, 
        speed_limit_train_sims, 
        est_time_nets
    ) = planner.run_train_planner(
        rail_vehicle_map = rail_vehicle_map,
        location_map = location_map,
        network = network,
        loco_pool= loco_pool,
        refuelers = refuelers,
        simulation_days=simulation_days + 2 * defaults.WARM_START_DAYS,
        scenario_year = scenario_year,
        config = train_planner_config,
        demand_file_path = demand_file_path,
        network_charging_guidelines = network_charging_guidelines,
    )
    t1_ptc = time.perf_counter()

    if grid_emissions_factors is None:
        grid_emissions_factors = metrics.import_emissions_factors_cambium(
            location_map = location_map,
            scenario_year = scenario_year
        )
    if nodal_energy_prices is None:
        nodal_energy_prices = metrics.import_energy_prices_eia(
            location_map = location_map,
            scenario_year = scenario_year
        )

    if debug:
        print(
            f"Elapsed time to plan train consist for year {scenario_year}: {t1_ptc - t0_ptc:.3g} s"
        )

    t0_disp = time.perf_counter()
    timed_paths = alt.run_dispatch(
        network, 
        alt.SpeedLimitTrainSimVec(speed_limit_train_sims), 
        est_time_nets, 
        False, 
        False,
    )
    timed_paths: List[List[alt.LinkIdxTime]] = [
        tp.tolist() for tp in timed_paths
    ]

    t1_disp = time.perf_counter()
    if debug:
        print(
            f"Elapsed time to run dispatch for year {scenario_year}: {t1_disp - t0_disp:.3g} s"
        )

    train_times = pl.DataFrame(
        {'Train_ID': pl.Series([sim.train_id for sim in speed_limit_train_sims], dtype=pl.Int32).cast(pl.UInt32),
         'Origin_ID': pl.Series([sim.origs[0].location_id for sim in speed_limit_train_sims], dtype=str),
         'Destination_ID': pl.Series([sim.dests[0].location_id for sim in speed_limit_train_sims], dtype=str),
         'Departure_Time_Actual_Hr': pl.Series([this[0].time_hours for this in timed_paths], dtype=pl.Float64),
         'Arrival_Time_Actual_Hr': pl.Series([this[len(this)-1].time_hours for this in timed_paths], dtype=pl.Float64)}
    )

    train_consist_plan = train_consist_plan.join(train_times,on=["Train_ID","Origin_ID","Destination_ID"],how="left")

    train_consist_plan = train_consist_plan.filter(
        (pl.col("Departure_Time_Actual_Hr") >= pl.lit(24*alt.defaults.WARM_START_DAYS)) & 
        (pl.col("Departure_Time_Actual_Hr") < pl.lit(24*(simulation_days+alt.defaults.WARM_START_DAYS)))
    )

    train_consist_plan = train_consist_plan.with_columns((pl.col("Train_ID").rank("dense")-1).alias("TrainSimVec_Index"))

     #speed_limit_train_sims is 0-indexed but Train_ID starts at 1
    to_keep = train_consist_plan.unique(subset=['Train_ID']).to_series().sort()
    for sim in speed_limit_train_sims: 
        alt.set_param_from_path(sim, "simulation_days", simulation_days)
    train_sims = alt.SpeedLimitTrainSimVec([speed_limit_train_sims[i-1] for i in to_keep])
    timed_paths = [timed_paths[i-1] for i in to_keep]

    return (
        train_consist_plan,
        loco_pool,
        refuelers,
        grid_emissions_factors,
        nodal_energy_prices,
        train_sims,
        timed_paths,
    )
