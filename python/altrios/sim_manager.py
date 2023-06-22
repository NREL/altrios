"""
Module for getting the output of the Train Consist Planner and Meet Pass Planner to run 3 week simulation.
"""

import pandas as pd
from typing import Any, Union, Dict, List, Optional, Tuple
from pathlib import Path
import time


# from __future__ import annotations # TODO: uncomment and propagate
import altrios as alt
from altrios import train_planner


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
            time_seconds=train_row["Origin Departure Time(hr)"] * 3600,
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
) -> Tuple[
    pd.DataFrame,
    alt.SpeedLimitTrainSimVec,
    List[List[alt.LinkIdxTime]],
]:
    """
    do the module!
    """

    if debug:
        print("Entering `sim_manager` module.")
        alt.utils.print_dt()

    for loc_name in location_map:
        for loc in location_map[loc_name]:
            if loc.link_idx.idx >= len(network):
                raise ValueError("Location " + loc.location_id + " with link index " +
                                 str(loc.link_idx.idx) + " is invalid for network!")

    t0_ptc = time.perf_counter()
    df_train_consist_plan, speed_limit_train_sims, est_time_nets = train_planner.run_train_planner(
        rail_vehicle_map,
        location_map,
        network,
        df_pool=train_planner.build_locopool(method="shares_twoway",
                                             shares=[1-target_bel_share, target_bel_share]),
        simulation_days=simulation_days + 2 * alt.utils.WARM_START_DAYS,
    )
    t1_ptc = time.perf_counter()

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

    to_keep = df_train_consist_plan[(df_train_consist_plan['Origin Departure Time(hr)'] >= 24*alt.utilities.WARM_START_DAYS) &
                                    (df_train_consist_plan['Origin Departure Time(hr)'] < 24*(simulation_days+alt.utilities.WARM_START_DAYS))].index
    df_train_consist_plan = df_train_consist_plan.filter(items=to_keep, axis=0)
    to_keep = df_train_consist_plan.drop_duplicates(["Train ID"])["Train ID"]
    train_sims = alt.SpeedLimitTrainSimVec(
        [speed_limit_train_sims[i] for i in to_keep])
    timed_paths = [timed_paths[i] for i in to_keep]

    return (
        df_train_consist_plan,
        train_sims,
        timed_paths,
    )
