"""
Module for getting the output of the Train Consist Planner and Meet Pass Planner to run 3 week simulation.
"""

import pandas as pd
import polars as pl
from typing import Any, Union, Dict, List, Optional, Tuple
from pathlib import Path
import time


# from __future__ import annotations # TODO: uncomment and propagate
import altrios as alt
from altrios import train_planner as planner
from altrios import metric_calculator as metrics


def make_train_sim_builders(
    df_train_consist_plan: pd.DataFrame,
    loco_cons: List[alt.Consist],
) -> List[alt.TrainSimBuilder]:
    tsbs = []

    for idx, train_row in df_train_consist_plan.drop_duplicates(["Train ID"]).iterrows():
        train_summary = alt.TrainSummary(
            train_row["Train Type"],
            int(train_row["Number of Empty Railcars"]),
            int(train_row["Number of Loaded Railcars"]),
            None,
            None,
            None,
        )
        init_train_state = alt.InitTrainState(
            time_seconds=train_row["Planned Departure Time(hr)"] * 3600,
        )
        tsbs.append(
            alt.TrainSimBuilder(
                train_id=str(idx),
                origin_id=train_row["Origin ID"],
                destination_id=train_row["Destination ID"],
                train_summary=train_summary,
                loco_con=loco_cons[train_row["Train ID"] - 1],
                init_train_state=init_train_state,
            )
        )
    return tsbs


def main(
    rail_vehicle_map: Dict[str, alt.RailVehicle],
    location_map: Dict[str, List[alt.Location]],
    network: List[alt.Link],
    simulation_days: int = 7,
    scenario_year: int = 2020,
    target_bel_share: float = 0.5,
    debug: bool = False,
    loco_pool: pl.DataFrame = None,
    refuel_facilities: pl.DataFrame = None,
    grid_emissions_factors: pl.DataFrame = None,
    train_planner_config: planner.TrainPlannerConfig = planner.TrainPlannerConfig(),
    demand_file_path: str = str(alt.resources_root() / "Default Demand.csv"),
) -> Tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    alt.SpeedLimitTrainSimVec,
    List[List[alt.LinkIdxTime]],
]:
    """
    do the module!
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
            
    if loco_pool is None: loco_pool = planner.build_locopool(
            config = train_planner_config,
            method="shares_twoway",
            shares=[1-target_bel_share, target_bel_share],
            demand_file=demand_file_path
            )

    t0_ptc = time.perf_counter()
    train_consist_plan, loco_pool, refuel_facilities, speed_limit_train_sims, est_time_nets = planner.run_train_planner(
        rail_vehicle_map = rail_vehicle_map,
        location_map = location_map,
        network = network,
        loco_pool= loco_pool,
        refuel_facilities = refuel_facilities,
        simulation_days=simulation_days + 2 * alt.utils.WARM_START_DAYS,
        config = train_planner_config,
        demand_file_path=demand_file_path,
    )
    t1_ptc = time.perf_counter()

    if grid_emissions_factors is None:
        grid_emissions_factors = metrics.import_emissions_factors_cambium(
            location_map = location_map,
            scenario_year = scenario_year
        )

    if debug:
        print(
            f"Elapsed time to plan train consist for year {scenario_year}: {t1_ptc - t0_ptc:.3g} s"
        )

    t0_disp = time.perf_counter()
    timed_paths = alt.run_dispatch(
        network, alt.SpeedLimitTrainSimVec(speed_limit_train_sims), est_time_nets, False, False)

    t1_disp = time.perf_counter()
    if debug:
        print(
            f"Elapsed time to run dispatch for year {scenario_year}: {t1_disp - t0_disp:.3g} s"
        )

    train_times = pl.DataFrame(
        {'Train ID': pl.Series([sim.train_id for sim in speed_limit_train_sims], dtype=pl.Int32).cast(pl.UInt32),
         'Origin ID': pl.Series([sim.origs[0].location_id for sim in speed_limit_train_sims], dtype=str),
         'Destination ID': pl.Series([sim.dests[0].location_id for sim in speed_limit_train_sims], dtype=str),
         'Actual Departure Time(hr)': pl.Series([this[0].time_hours for this in timed_paths], dtype=pl.Float64),
         'Actual Arrival Time(hr)': pl.Series([this[len(this)-1].time_hours for this in timed_paths], dtype=pl.Float64)}
    )

    train_consist_plan = train_consist_plan.join(train_times,on=["Train ID","Origin ID","Destination ID"],how="left")

    train_consist_plan = train_consist_plan.filter(
        (pl.col("Actual Departure Time(hr)") >= pl.lit(24*alt.utilities.WARM_START_DAYS)) & 
        (pl.col("Actual Departure Time(hr)") < pl.lit(24*(simulation_days+alt.utilities.WARM_START_DAYS)))
    )

    train_consist_plan = train_consist_plan.with_columns((pl.col("Train ID").rank("dense")-1).alias("TrainSimVec Index"))

     #speed_limit_train_sims is 0-indexed but Train IDs start at 1
    to_keep = train_consist_plan.unique(subset=['Train ID']).to_series().sort()
    train_sims = alt.SpeedLimitTrainSimVec([speed_limit_train_sims[i-1] for i in to_keep])
    timed_paths = [timed_paths[i-1] for i in to_keep]

    return (
        train_consist_plan,
        loco_pool,
        refuel_facilities,
        grid_emissions_factors,
        train_sims,
        timed_paths,
    )
