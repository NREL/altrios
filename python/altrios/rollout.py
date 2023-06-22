from altrios import sim_manager
from altrios import metric_calculator
from altrios.metric_calculator import ScenarioInfo

import altrios as alt
import numpy as np
import time
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path


DEBUG = False


def simulate_prescribed_rollout(
    max_bel_share: float,
    number_of_years: int,
    # If you do not have this path, please add it! We are trying to keep confidential documents out of the git repo
    network_filename_path: str = alt.package_root()
    / "resources/networks/Scenic - ALTRIOS Confidential.yaml",
    save_interval: Optional[int] = None,
    parallelize: Optional[bool] = True,
) -> Tuple[List[ScenarioInfo], pd.DataFrame]:
    years = list(range(2020, 2020 + number_of_years))
    target_bel_shares = np.zeros(len(years))
    if len(years) == 1:
        target_bel_shares[0] = max_bel_share
    else:
        for idx, _ in enumerate(target_bel_shares):
            target_bel_shares[idx] = ((idx+1) / (len(years))) * max_bel_share

    if parallelize:
        print("`build_and_run_speed_limit_train_sims` is parallelized.  Set to `False` if memory issues happen.")

    rail_vehicle_map = alt.import_rail_vehicles(
        str(alt.resources_root() / "rolling_stock/rail_vehicles.csv")
    )
    location_map = alt.import_locations(
        str(alt.resources_root() / "networks/default_locations.csv")
    )
    network = alt.import_network(
        str(alt.resources_root() / "networks/Taconite.yaml"))

    sim_days = 7
    scenarios = []
    for idx, scenario_year in enumerate(years):
        t0 = time.perf_counter()
        (
            df_train_consist_plan, speed_limit_train_sims, timed_paths
        ) = sim_manager.main(
            network=network,
            rail_vehicle_map=rail_vehicle_map,
            location_map=location_map,
            simulation_days=sim_days,
            scenario_year=scenario_year,
            target_bel_share=target_bel_shares[idx],
            debug=True,
        )

        t1 = time.perf_counter()
        if DEBUG:
            print(f"Elapsed time to run `sim_manager.main()`: {t1-t0:.3g} s")

        speed_limit_train_sims.set_save_interval(save_interval)
        sims = alt.run_speed_limit_train_sims(
            speed_limit_train_sims=speed_limit_train_sims,
            timed_paths=timed_paths,
            network=network,
            parallelize=parallelize
        )

        scenarios.append(ScenarioInfo(
            sims, scenario_year, df_train_consist_plan))

        t2 = time.perf_counter()
        if DEBUG:
            print(
                f"Elapsed time to run `build_and_run_speed_limit_train_sims()`: {t2-t1:.3g} s")

    metrics = metric_calculator.main(scenarios)

    t3 = time.perf_counter()
    if DEBUG:
        print(f"Elapsed time to run metric_calculator.main(): {t3-t2:.3g} s")

    return scenarios, metrics
