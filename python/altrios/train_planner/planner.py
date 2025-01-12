from __future__ import annotations
from pathlib import Path
from altrios.train_planner import demand_generators, schedulers, planner_config
import numpy as np
from scipy.stats import rankdata
import pandas as pd
import polars as pl
import polars.selectors as cs
import math
from typing import Union, Tuple, List, Dict, Callable, Optional
from itertools import repeat
from dataclasses import dataclass, field
import altrios as alt
from altrios import defaults, utilities

def demand_loader(
    demand_table: Union[pl.DataFrame, Path, str]
) -> Tuple[pl.DataFrame, pl.Series, int]:
    """
    Load the user input csv file into a dataframe for later processing
    Arguments:
    ----------
    user_input_file: path to the input csv file that user import to the module
    Example Input:
        Origin	Destination	Train_Type	Number_of_Cars	Number_of_Containers
        Barstow	Stockton	Unit	    2394	        0
        Barstow	Stockton	Manifest	2588	        0
        Barstow	Stockton	Intermodal	2221	        2221

    Outputs:
    ----------
    df_annual_demand: dataframe with all pair information including:
    origin, destination, train type, number of cars
    node_list: List of origin or destination demand nodes
    """
    if type(demand_table) is not pl.DataFrame:
        demand_table = pl.read_csv(demand_table, dtypes = {"Number_of_Cars": pl.UInt32, "Number_of_Containers": pl.UInt32})

    nodes = pl.concat(
        [demand_table.get_column("Origin"),
        demand_table.get_column("Destination")]).unique().sort()
    return demand_table, nodes

def append_loco_info(loco_info: pd.DataFrame) -> pd.DataFrame:
    if all(item in loco_info.columns for item in [
        'HP','Loco_Mass_Tons','SOC_J','SOC_Min_J','SOC_Max_J','Capacity_J'
        ]
    ): return loco_info
    get_hp = lambda loco: loco.pwr_rated_kilowatts * 1e3 / alt.utils.W_PER_HP
    get_mass_ton = lambda loco: 0 if not loco.mass_kg else loco.mass_kg / alt.utils.KG_PER_TON
    get_starting_soc = lambda loco: defaults.DIESEL_TANK_CAPACITY_J if not loco.res else loco.res.state.soc * loco.res.energy_capacity_joules
    get_min_soc = lambda loco: 0 if not loco.res else loco.res.min_soc * loco.res.energy_capacity_joules
    get_max_soc = lambda loco: defaults.DIESEL_TANK_CAPACITY_J if not loco.res else loco.res.max_soc * loco.res.energy_capacity_joules
    get_capacity = lambda loco: defaults.DIESEL_TANK_CAPACITY_J if not loco.res else loco.res.energy_capacity_joules
    loco_info.loc[:,'HP'] = loco_info.loc[:,'Rust_Loco'].apply(get_hp) 
    loco_info.loc[:,'Loco_Mass_Tons'] = loco_info.loc[:,'Rust_Loco'].apply(get_mass_ton) 
    loco_info.loc[:,'SOC_J'] = loco_info.loc[:,'Rust_Loco'].apply(get_starting_soc) 
    loco_info.loc[:,'SOC_Min_J'] = loco_info.loc[:,'Rust_Loco'].apply(get_min_soc) 
    loco_info.loc[:,'SOC_Max_J'] = loco_info.loc[:,'Rust_Loco'].apply(get_max_soc) 
    loco_info.loc[:,'Capacity_J'] = loco_info.loc[:,'Rust_Loco'].apply(get_capacity) 
    return loco_info

def build_locopool(
    config: planner_config.TrainPlannerConfig,
    demand_file: Union[pl.DataFrame, Path, str],
    method: str = "tile",
    shares: List[float] = [],
    locomotives_per_node: int = None
) -> pl.DataFrame:
    """
    Generate default locomotive pool
    Arguments:
    ----------
    demand_file: Path to a file with origin-destination demand
    method: Method to determine each locomotive's type ("tile" or "shares_twoway" currently implemented)
    shares: List of shares for each locomotive type in loco_info (implemented for two-way shares only)
    Outputs:
    ----------
    loco_pool: Locomotive pool containing all locomotives' information that are within the system
    """
    config.loco_info = append_loco_info(config.loco_info)
    loco_types = list(config.loco_info.loc[:,'Locomotive_Type'])
    demand, node_list = demand_loader(demand_file)
    
    num_nodes = len(node_list)
    if locomotives_per_node is None:
        num_ods = demand.height
        cars_per_od = demand.get_column("Number_of_Cars").mean()
        if config.single_train_mode:
            initial_size = math.ceil(cars_per_od / config.cars_per_locomotive["Default"]) 
            rows = initial_size
        else:
            num_destinations_per_node = num_ods*1.0 / num_nodes*1.0
            initial_size = math.ceil((cars_per_od / config.cars_per_locomotive["Default"]) *
                                    num_destinations_per_node)  # number of locomotives per node
            rows = initial_size * num_nodes  # number of locomotives in total
    else:
        initial_size = locomotives_per_node
        rows = locomotives_per_node * num_nodes

    if config.single_train_mode:
        sorted_nodes = np.tile([demand.select(pl.col("Origin").first()).item()],rows).tolist()
        engine_numbers = range(0, rows)
    else:
        sorted_nodes = np.sort(np.tile(node_list, initial_size)).tolist()
        engine_numbers = rankdata(sorted_nodes, method="dense") * 1000 + \
            np.tile(range(0, initial_size), num_nodes)

    if method == "tile":
        repetitions = math.ceil(rows/len(loco_types))
        types = np.tile(loco_types, repetitions).tolist()[0:rows]
    elif method == "shares_twoway":
        # TODO: this logic can be replaced (and generalized to >2 types) using altrios.utilities.allocateItems
        if((len(loco_types) != 2) | (len(shares) != 2)):
            raise ValueError(
                f"""2-way prescribed locopool requested but number of locomotive types is not 2.""")

        idx_1 = np.argmin(shares)
        idx_2 = 1 - idx_1
        share_type_one = shares[idx_1]
        label_type_one = loco_types[idx_1]
        label_type_two = loco_types[idx_2]

        num_type_one = round(initial_size * share_type_one)
        if 0 == num_type_one:
            types = pd.Series([label_type_two] * initial_size)
        elif initial_size == num_type_one:
            types = pd.Series([label_type_one] * initial_size)
        else:
            # Arrange repeated sequences of type 1 + {type_two_per_type_one, type_two_per_type_one+1} type 2
            # so as to match the required total counts of each.
            type_two_per_type_one = (
                initial_size - num_type_one) / num_type_one
            # Number of type 1 + {type_two_per_bel+1} type 2 sequences needed
            num_extra_type_two = round(
                num_type_one * (type_two_per_type_one % 1.0))
            series_fewer_type_two = pd.Series(
                [label_type_one] + [label_type_two] * math.floor(type_two_per_type_one))
            series_more_type_two = pd.Series(
                [label_type_one] + [label_type_two] * math.ceil(type_two_per_type_one))
            types = np.concatenate((
                np.tile(series_more_type_two, num_extra_type_two),
                np.tile(series_fewer_type_two, num_type_one-num_extra_type_two)),
                axis=None)
        types = np.tile(types, num_nodes).tolist()
    else:
        raise ValueError(
            f"""Locopool build method '{method}' invalid or not implemented.""")

    loco_pool = pl.DataFrame(
        {'Locomotive_ID': pl.Series(engine_numbers, dtype=pl.UInt32),
         'Locomotive_Type': pl.Series(types, dtype=pl.Categorical),
         'Node': pl.Series(sorted_nodes, dtype=pl.Categorical),
         'Arrival_Time': pl.Series(np.zeros(rows), dtype=pl.Float64),
         'Servicing_Done_Time': pl.Series(np.zeros(rows), dtype=pl.Float64),
         'Refueling_Done_Time': pl.Series(np.tile(0, rows), dtype=pl.Float64),
         'Status': pl.Series(np.tile("Ready", rows), dtype=pl.Categorical),
         'SOC_Target_J': pl.Series(np.zeros(rows), dtype=pl.Float64),
         'Refuel_Duration': pl.Series(np.zeros(rows), dtype=pl.Float64),
         'Refueler_J_Per_Hr': pl.Series(np.zeros(rows), dtype=pl.Float64), 
         'Refueler_Efficiency': pl.Series(np.zeros(rows), dtype=pl.Float64), 
         'Port_Count': pl.Series(np.zeros(rows), dtype=pl.UInt32)}
    )

    loco_info_pl = pl.from_pandas(config.loco_info.drop(labels='Rust_Loco',axis=1),
        schema_overrides={'Locomotive_Type': pl.Categorical,
                          'Fuel_Type': pl.Categorical}
    )

    loco_pool = loco_pool.join(loco_info_pl, on="Locomotive_Type")
    return loco_pool

def build_refuelers(
    node_list: pd.Series,
    loco_pool: pl.DataFrame,
    refueler_info: pd.DataFrame,
    refuelers_per_incoming_corridor: int,
) -> pl.DataFrame:
    """
    Build the default set of refueling facilities.
    Arguments:
    ----------
    node_list: List of origin or destination demand nodes
    loco_pool: Locomotive pool
    refueler_info: DataFrame with information for each type of refueling infrastructure to use
    refuelers_per_incoming_corridor: Queue size per corridor arriving at each node.
    Outputs:
    ----------
    refuelers: Polars dataframe of facility county by node and type of fuel
    """
    ports_per_node = (loco_pool
        .group_by(pl.col("Locomotive_Type", "Fuel_Type").cast(pl.Utf8))
        .agg([(pl.lit(refuelers_per_incoming_corridor) * pl.len() / pl.lit(loco_pool.height))
              .ceil()
              .alias("Ports_Per_Node")])
        .join(pl.from_pandas(refueler_info),
              on=["Locomotive_Type", "Fuel_Type"], 
              how="left")
    )

    locations = pd.DataFrame(data={
        'Node': np.tile(node_list, ports_per_node.height)})
    locations = locations.sort_values(by=['Node']).reset_index(drop=True)

    refuelers = pl.DataFrame({
        'Node': pl.Series(locations['Node'], dtype=pl.Categorical).cast(pl.Categorical),
        'Refueler_Type': pl.Series(np.tile(
            ports_per_node.get_column("Refueler_Type").to_list(), len(node_list)), 
            dtype=pl.Categorical).cast(pl.Categorical),
        'Locomotive_Type': pl.Series(np.tile(
            ports_per_node.get_column("Locomotive_Type").to_list(), len(node_list)), 
            dtype=pl.Categorical).cast(pl.Categorical),
        'Fuel_Type': pl.Series(np.tile(
            ports_per_node.get_column("Fuel_Type").to_list(), len(node_list)), 
            dtype=pl.Categorical).cast(pl.Categorical),
        'Refueler_J_Per_Hr': pl.Series(np.tile(
            ports_per_node.get_column("Refueler_J_Per_Hr").to_list(), len(node_list)), 
            dtype=pl.Float64),
        'Refueler_Efficiency': pl.Series(np.tile(
            ports_per_node.get_column("Refueler_Efficiency").to_list(), len(node_list)), 
            dtype=pl.Float64),
        'Lifespan_Years': pl.Series(np.tile(
            ports_per_node.get_column("Lifespan_Years").to_list(), len(node_list)), 
            dtype=pl.Float64),
        'Cost_USD': pl.Series(np.tile(
            ports_per_node.get_column("Cost_USD").to_list(), len(node_list)), 
            dtype=pl.Float64),
        'Port_Count': pl.Series(np.tile(
            ports_per_node.get_column("Ports_Per_Node").to_list(), len(node_list)), 
            dtype=pl.UInt32)})
    return refuelers

def append_charging_guidelines(
    refuelers: pl.DataFrame,
    loco_pool: pl.DataFrame,
    demand: pl.DataFrame,
    network_charging_guidelines: pl.DataFrame
) -> pl.DataFrame:
    active_ods = demand.select(["Origin","Destination"]).unique()
    network_charging_guidelines = (network_charging_guidelines
        .join(active_ods, on=["Origin","Destination"], how="inner")
        .group_by(pl.col("Origin"))
        .agg(pl.col("Allowable_Battery_Headroom_MWh").min() * 1e6 / utilities.MWH_PER_MJ)
        .rename({"Allowable_Battery_Headroom_MWh": "Battery_Headroom_J"})
        .with_columns(pl.col("Origin").cast(pl.Categorical)))
    refuelers = (refuelers
        .join(network_charging_guidelines, left_on="Node", right_on="Origin", how="left")
        .with_columns(pl.when(pl.col("Fuel_Type")=="Electricity")
            .then(pl.col("Battery_Headroom_J"))
            .otherwise(0)
            .fill_null(0)
            .alias("Battery_Headroom_J")
            ))
    loco_pool = (loco_pool
        .join(network_charging_guidelines, left_on="Node", right_on="Origin", how="left")
        .with_columns(pl.when(pl.col("Fuel_Type")=="Electricity")
            .then(pl.col("Battery_Headroom_J"))
            .otherwise(0)
            .fill_null(0)
            .alias("Battery_Headroom_J"))
        .with_columns(pl.max_horizontal([pl.col('SOC_Max_J')-pl.col('Battery_Headroom_J'), pl.col('SOC_Min_J')]).alias("SOC_J")))
    return refuelers, loco_pool

def dispatch(
    dispatch_time: int,
    origin: str,
    loco_pool: pl.DataFrame,
    train_tonnage: float,
    hp_required: float,
    total_cars: float,
    config: planner_config.TrainPlannerConfig, 
) -> pl.Series:
    """
    Update the locomotive pool by identifying the desired locomotive to dispatch and assign to the
    new location (destination) with corresponding updated ready time
    Arguments:
    ----------
    dispatch_time: time that a train is due
    origin: origin node name of the train
    loco_pool: locomotive pool dataframe containing all locomotives in the network
    hp_required: Horsepower required for this train type on this origin-destination corridor
    total_cars: Total number of cars (loaded, empty, or otherwise) included on the train
    config: TrainPlannerConfig object
    Outputs:
    ----------
    selected: Indices of selected locomotives
    """
    hp_per_ton = hp_required / train_tonnage
    # Candidate locomotives at the right place that are ready
    candidates = loco_pool.select((pl.col("Node") == origin) &
                                (pl.col("Status") == "Ready")).to_series()
    if not candidates.any():
        message = f"""No available locomotives at node {origin} at hour {dispatch_time}."""
        waiting_counts = (loco_pool
            .filter(
                pl.col("Status").is_in(["Servicing","Refuel_Queue"]),
                pl.col("Node") == origin
            )
            .group_by(['Locomotive_Type']).agg(pl.len())
        )
        if waiting_counts.height == 0:
            message = message + f"""\nNo locomotives are currently located there. Instead, they are at:"""
            locations = loco_pool.group_by("Node").agg(pl.len())
            for row in locations.iter_rows(named = True):
                message = message + f"""
                {row['Node']}: {row['count']}"""
        else:
            message = message + f"""Count of locomotives refueling or waiting to refuel at {origin} are:"""
            for row in waiting_counts.iter_rows(named = True):
                message = message + f"""\n{row['Locomotive_Type']}: {row['count']}"""

        raise ValueError(message)

    # Running list of downselected candidates
    selected = candidates
    diesel_to_require_hp = 0
    if config.require_diesel:
        # First available diesel (in order of loco_pool) will be moved from candidates to selected
        # TODO gracefully handle cases when there is no diesel locomotive to be dispatched
        # (ex: hold the train until enough diesels are present)
        diesel_filter = pl.col("Fuel_Type").cast(pl.Utf8).str.contains("(?i)diesel")
        diesel_candidates = loco_pool.select(pl.lit(candidates) & diesel_filter).to_series()
        if not diesel_candidates.any():
            refueling_diesel_count = loco_pool.filter(
                pl.col("Node") == origin,
                pl.col("Status").is_in(["Servicing","Refuel_Queue"]),
                diesel_filter
            ).select(pl.len())[0, 0]
            message = f"""No available diesel locomotives at node {origin} at hour {dispatch_time}, so
                    the one-diesel-per-consist rule cannot be satisfied. {refueling_diesel_count} diesel locomotives at
                    {origin} are servicing, refueling, or queueing."""
            if refueling_diesel_count > 0:
                diesel_port_count = loco_pool.filter(
                    pl.col("Node") == origin,
                    diesel_filter
                ).select(pl.col("Port_Count").min()).item()
                message += f""" (queue capacity {diesel_port_count})."""
            else:
                message += "."
            raise ValueError(message)

        diesel_to_require = diesel_candidates.eq(True).cum_sum().eq(1).arg_max()
        diesel_to_require_hp = loco_pool.filter(diesel_filter).select(pl.first("HP"))
        # Need to mask this so it's not double-counted on next step
        candidates[diesel_to_require] = False

    message = ""
    if config.cars_per_locomotive_fixed:
        # Get as many available locomotives as are needed (in order of loco_pool)
        enough_locomotives = loco_pool.select(
            (pl.lit(1.0) * pl.lit(candidates)).cumsum() >= total_cars).to_series()
        if not enough_locomotives.any():
            message = f"""Locomotives needed ({total_cars}) at {origin} at hour {dispatch_time}
                is more than the available locomotives ({candidates.sum()}).
                Count of locomotives servicing, refueling, or queueing at {origin} are:"""
    else:
        # Get running sum, including first diesel, of hp of the candidates (in order of loco_pool)
        enough_hp = loco_pool.select((
            (
                (pl.col("HP") - (pl.col("Loco_Mass_Tons") * pl.lit(hp_per_ton))) * pl.lit(candidates)
            ).cum_sum() + pl.lit(diesel_to_require_hp)) >= hp_required).to_series()
        if not enough_hp.any():
            available_hp = loco_pool.select(
                (
                    (pl.col("HP") - (pl.col("Loco_Mass_Tons") * pl.lit(hp_per_ton))) * pl.lit(candidates)
                ).cum_sum().max())[0, 0]
            message = f"""Outbound horsepower needed ({hp_required}) at {origin} at hour {dispatch_time}
                is more than the available horsepower ({available_hp}).
                Count of locomotives servicing, refueling, or queueing at {origin} are:"""

    if not enough_hp.any():
        # Hold the train until enough diesels are present (future development)
        waiting_counts = (loco_pool
            .filter(
                pl.col("Node") == origin,
                pl.col("Status").is_in(["Servicing","Refuel_Queue"])
            )
            .group_by(['Locomotive_Type'])
                .agg(pl.count().alias("count"))
        )
        for row in waiting_counts.iter_rows(named = True):
            message = message + f"""
            {row['Locomotive_Type']}: {row['count']}"""
        # Hold the train until enough locomotives are present (future development)
        raise ValueError(message)

    last_row_to_use = enough_hp.eq(True).cum_sum().eq(1).arg_max()
    # Set false all the locomotives that would add unnecessary hp
    selected[np.arange(last_row_to_use+1, len(selected))] = False

    if config.require_diesel:
        # Add first diesel (which could come after last_row_to_use) to selection list
        selected[diesel_to_require] = True
    return selected

def update_refuel_queue(
        loco_pool: pl.DataFrame,
        refuelers: pl.DataFrame,
        current_time: float,
        event_tracker: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Update the locomotive pool by identifying the desired locomotive to dispatch and assign to the
    new location (destination) with corresponding updated ready time
    Arguments:
    ----------
    loco_pool: locomotive pool dataframe containing all locomotives in the network
    refuelers: refuelers dataframe containing all refueling ports in the network
    current_time:
    event_tracker:
    hp_per_ton: Horsepower per ton required for this train type on this origin-destination corridor
    Outputs:
    ----------
    loco_pool: Locomotive pool with updates made to servicing, refueling, or queued locomotives
    """

    # If any trains arrived, add arrivals to the service queue
    arrived = loco_pool.select((pl.col("Status") == "Dispatched") &
                                 (pl.col("Arrival_Time") <= current_time)).to_series()
    if(arrived.sum() > 0):
        loco_pool = (loco_pool
            .drop(['Refueler_J_Per_Hr','Port_Count','Battery_Headroom_J'])
            .join(
                refuelers.select(["Node","Locomotive_Type","Fuel_Type","Refueler_J_Per_Hr","Port_Count",'Battery_Headroom_J']), 
                on=["Node", "Locomotive_Type" ,"Fuel_Type"],
                how="left")
            .with_columns(
                pl.when(arrived)
                    .then(pl.lit("Refuel_Queue"))
                    .otherwise(pl.col("Status")).alias("Status"),
                pl.when(arrived)
                    .then(pl.max_horizontal([pl.col('SOC_Max_J')-pl.col('Battery_Headroom_J'), pl.col('SOC_J')]))
                    .otherwise(pl.col("SOC_Target_J")).alias("SOC_Target_J"))
            .with_columns(
                pl.when(arrived)
                    .then((pl.col("SOC_Target_J")-pl.col("SOC_J"))/pl.col("Refueler_J_Per_Hr"))
                    .otherwise(pl.col("Refuel_Duration")).alias("Refuel_Duration"))
            .sort("Node", "Locomotive_Type", "Fuel_Type", "Arrival_Time", "Locomotive_ID", descending = False, nulls_last = True))
        charger_type_breakouts = (loco_pool
            .filter(
                pl.col("Status") == "Refuel_Queue",
                (pl.col("Refueling_Done_Time") >= current_time) | (pl.col("Refueling_Done_Time").is_null())
            )
            .partition_by(["Node","Locomotive_Type"])
        )
        charger_type_list = []
        for charger_type in charger_type_breakouts:
            loco_ids = charger_type.get_column("Locomotive_ID")
            arrival_times = charger_type.get_column("Arrival_Time")
            refueling_done_times = charger_type.get_column("Refueling_Done_Time")
            refueling_durations = charger_type.get_column("Refuel_Duration")
            port_counts = charger_type.get_column("Port_Count")
            for i in range(0, refueling_done_times.len()):
                if refueling_done_times[i] is not None: continue
                next_done = refueling_done_times.filter(
                    (refueling_done_times.is_not_null()) & 
                    (refueling_done_times.rank(method='ordinal', descending = True).eq(port_counts[i])))
                if next_done.len() == 0: next_done = arrival_times[i]
                else: next_done = max(next_done[0], arrival_times[i])
                refueling_done_times[i] = next_done + refueling_durations[i]
            charger_type_list.append(pl.DataFrame([loco_ids, refueling_done_times]))
        all_queues = pl.concat(charger_type_list, how="diagonal")
        loco_pool = (loco_pool
            .join(all_queues, on="Locomotive_ID", how="left", suffix="_right")
            .with_columns(pl.when(pl.col("Refueling_Done_Time_right").is_not_null())
                            .then(pl.col("Refueling_Done_Time_right"))
                            .otherwise(pl.col("Refueling_Done_Time"))
                            .alias("Refueling_Done_Time"))
            .drop("Refueling_Done_Time_right"))
        
    # Remove locomotives that are done refueling from the refuel queue
    refueling_finished = loco_pool.select(
        (pl.col("Status") == "Refuel_Queue") & (pl.col("Refueling_Done_Time") <= current_time)
    ).to_series()
    refueling_finished_count = refueling_finished.sum()
    if(refueling_finished_count > 0):
        # Record the refueling event
        new_rows = pl.DataFrame([
            np.concatenate([
                np.tile('Refueling_Start', refueling_finished_count),
                np.tile('Refueling_End', refueling_finished_count)]),
            np.concatenate([
                loco_pool.filter(refueling_finished).select(pl.col('Refueling_Done_Time') - pl.col("Refuel_Duration")).to_series(),
                loco_pool.filter(refueling_finished).get_column('Refueling_Done_Time')]),
            np.tile(loco_pool.filter(refueling_finished).get_column('Locomotive_ID'), 2)],
            schema=event_tracker.columns,
            orient="col")
        event_tracker = pl.concat([event_tracker, new_rows])
        
        loco_pool = loco_pool.with_columns(
            pl.when(refueling_finished)
                .then(pl.col("SOC_Target_J"))
                .otherwise(pl.col('SOC_J'))
                .alias("SOC_J"),
            pl.when(refueling_finished)
                .then(pl.lit(None))
                .otherwise(pl.col('Refueling_Done_Time'))
                .alias("Refueling_Done_Time"),
            pl.when(pl.lit(refueling_finished) & (pl.col("Servicing_Done_Time") <= current_time))
                .then(pl.lit("Ready"))
                .when(pl.lit(refueling_finished) & (pl.col("Servicing_Done_Time") > current_time))
                .then(pl.lit("Servicing"))
                .otherwise(pl.col('Status'))
                .alias("Status"))
        
    servicing_finished = loco_pool.select(
        (pl.col("Status") == "Servicing") & (pl.col("Servicing_Done_Time") <= current_time)).to_series()
    if(servicing_finished.sum() > 0):
        loco_pool = loco_pool.with_columns(
            pl.when(servicing_finished)
                .then(pl.lit("Ready"))
                .otherwise(pl.col('Status'))
                .alias("Status"),
            pl.when(servicing_finished)
                .then(pl.lit(None))
                .otherwise(pl.col("Servicing_Done_Time"))
                .alias("Servicing_Done_Time")
        )
    return loco_pool.sort("Locomotive_ID"), event_tracker
    
def run_train_planner(
    rail_vehicles: List[alt.RailVehicle],
    location_map: Dict[str, List[alt.Location]],
    network: List[alt.Link],
    loco_pool: pl.DataFrame,
    refuelers: pl.DataFrame,
    simulation_days: int,
    scenario_year: int,
    train_type: alt.TrainType = alt.TrainType.Freight, 
    config: planner_config.TrainPlannerConfig = planner_config.TrainPlannerConfig(),
    demand_file: Union[pl.DataFrame, Path, str] = defaults.DEMAND_FILE,
    network_charging_guidelines: pl.DataFrame = None,
) -> Tuple[
    pl.DataFrame, 
    pl.DataFrame, 
    pl.DataFrame, 
    List[alt.SpeedLimitTrainSim], 
    List[alt.EstTimeNet]
]:
    """
    Run the train planner
    Arguments:
    ----------
    rail_vehicles:
    location_map:
    network:
    loco_pool:
    refuelers:
    simulation_days:
    config: Object storing train planner configuration paramaters
    demand_file: 
    Outputs:
    ----------
    """
    config.loco_info = append_loco_info(config.loco_info)
    demand, node_list = demand_loader(demand_file)

    if refuelers is None: 
        refuelers = build_refuelers(
            node_list, 
            loco_pool,
            config.refueler_info, 
            config.refuelers_per_incoming_corridor)
        
    if network_charging_guidelines is None: 
        network_charging_guidelines = pl.read_csv(alt.resources_root() / "networks" / "network_charging_guidelines.csv")

    refuelers, loco_pool = append_charging_guidelines(refuelers, loco_pool, demand, network_charging_guidelines)
    if config.single_train_mode:
        demand = demand_generators.generate_demand_trains(demand, 
                                        demand_returns = pl.DataFrame(), 
                                        demand_rebalancing = pl.DataFrame(), 
                                        rail_vehicles = rail_vehicles, 
                                        config = config)
        dispatch_schedule = (demand
            .with_row_index(name="index")
            .with_columns(pl.col("index").mul(24.0).alias("Hour"))
            .drop("index")
        )
    else:
        demand_returns = demand_generators.generate_return_demand(demand, config)
        demand_rebalancing = pl.DataFrame()
        if demand.filter(pl.col("Train_Type").str.contains("Manifest")).height > 0:
            demand_origin_manifest = demand_generators.generate_origin_manifest_demand(demand, node_list, config)
            demand_rebalancing = demand_generators.balance_trains(demand_origin_manifest)
        demand = demand_generators.generate_demand_trains(demand, demand_returns, demand_rebalancing, rail_vehicles, config)
        if config.dispatch_scheduler is None:
            config.dispatch_scheduler = schedulers.generate_dispatch_details
        dispatch_schedule = config.dispatch_scheduler(demand, rail_vehicles, simulation_days, config)

    final_departure = dispatch_schedule.get_column("Hour").max()
    train_consist_plan = pl.DataFrame(schema=
        {'Train_ID': pl.Int64, 
         'Train_Type': pl.Utf8, 
         'Locomotive_ID': pl.UInt32, 
         'Locomotive_Type': pl.Categorical, 
         'Origin_ID': pl.Utf8, 
         'Destination_ID': pl.Utf8, 
         'Cars_Loaded': pl.Float64, 
         'Cars_Empty': pl.Float64, 
         'Departure_SOC_J': pl.Float64, 
         'Departure_Time_Planned_Hr': pl.Float64, 
         'Arrival_Time_Planned_Hr': pl.Float64})
    event_tracker = pl.DataFrame(schema=[
        ("Event_Type", pl.Utf8), 
        ("Time_Hr", pl.Float64), 
        ("Locomotive_ID", pl.UInt32)])
        
    train_id_counter = 1
    speed_limit_train_sims = []
    est_time_nets = []

    done = False
    # start at first departure time
    current_time = dispatch_schedule.get_column("Hour").min()
    while not done:
        # Dispatch new train consists
        current_dispatches = dispatch_schedule.filter(pl.col("Hour") == current_time)
        if(current_dispatches.height > 0):
            loco_pool, event_tracker = update_refuel_queue(loco_pool, refuelers, current_time, event_tracker)

            for this_train in current_dispatches.iter_rows(named = True):
                if this_train['Tons_Per_Train'] > 0:
                    train_id=str(train_id_counter)
                    if config.single_train_mode:
                        selected = loco_pool.select(pl.col("Locomotive_ID").is_not_null().alias("selected")).to_series()
                        dispatched = loco_pool
                    else:
                        selected = dispatch(
                            current_time,
                            this_train['Origin'],
                            loco_pool,
                            this_train['Tons_Per_Train'],
                            this_train['HP_Required'],
                            this_train['Cars_Loaded'] + this_train['Cars_Empty'],
                            config
                        )
                        dispatched = loco_pool.filter(selected)

                    if config.drag_coeff_function is not None:
                        cd_area_vec = config.drag_coeff_function(
                             this_train['Number_of_Cars'], 
                             gap_size = defaults.DEFAULT_GAP_SIZE
                         )
                    else:
                        cd_area_vec = None

                    train_types = []
                    n_cars_by_type = {}
                    this_train_type = this_train['Train_Type']
                    if this_train['Cars_Loaded'] > 0:
                        train_types.append(f'{this_train_type}_Loaded')
                        n_cars_by_type[f'{this_train_type}_Loaded'] = int(this_train['Cars_Loaded'])
                    if this_train['Cars_Empty'] > 0:
                        train_types.append(f'{this_train_type}_Empty')
                        n_cars_by_type[f'{this_train_type}_Empty'] = int(this_train['Cars_Empty'])

                    train_config = alt.TrainConfig(
                        rail_vehicles = [vehicle for vehicle in rail_vehicles if vehicle.car_type in train_types],
                        n_cars_by_type = n_cars_by_type,
                        train_type = train_type,
                        cd_area_vec = cd_area_vec
                    )

                    loco_start_soc_j = dispatched.get_column("SOC_J")
                    dispatch_order =  (dispatched.select(
                        pl.col('Locomotive_ID')
                        .rank().alias('rank').cast(pl.UInt32)
                        ).with_row_count().sort('row_nr'))
                    dispatched = dispatched.sort('Locomotive_ID')
                    loco_start_soc_pct = dispatched.select(pl.col('SOC_J') / pl.col('Capacity_J')).to_series()
                    locos = [
                        config.loco_info[config.loco_info['Locomotive_Type']==loco_type]['Rust_Loco'].to_list()[0].clone() 
                        for loco_type in dispatched.get_column('Locomotive_Type')
                    ]
                    [alt.set_param_from_path(
                        locos[i], 
                        "res.state.soc", 
                        loco_start_soc_pct[i]
                    ) for i in range(len(locos)) if dispatched.get_column('Fuel_Type')[i] == 'Electricity']

                    loco_con = alt.Consist(
                        loco_vec=locos,
                        save_interval=None,
                    )

                    init_train_state = alt.InitTrainState(
                        time_seconds=current_time * 3600
                    )
                    tsb = alt.TrainSimBuilder(
                        train_id=train_id,
                        origin_id=this_train['Origin'],
                        destination_id=this_train['Destination'],
                        train_config=train_config,
                        loco_con=loco_con,
                        init_train_state=init_train_state,
                    )
                    
                    slts = tsb.make_speed_limit_train_sim(
                        location_map=location_map, 
                        save_interval=None, 
                        simulation_days=simulation_days, 
                        scenario_year=scenario_year
                    )

                    (est_time_net, loco_con_out) = alt.make_est_times(slts, network)
                    travel_time = (
                        est_time_net.get_running_time_hours()
                        * config.dispatch_scaling_dict["time_mult_factor"] 
                        + config.dispatch_scaling_dict["hours_add"]
                    )
                    
                    locos = loco_con_out.loco_vec.tolist()
                    energy_use_locos = [loco.res.state.energy_out_chemical_joules if loco.res else loco.fc.state.energy_fuel_joules if loco.fc else 0 for loco in locos]
                    energy_use_j = np.zeros(len(loco_pool))
                    energy_use_j[selected] = [energy_use_locos[i-1] for i in dispatch_order.get_column('rank').to_list()] 
                    energy_use_j *= config.dispatch_scaling_dict["energy_mult_factor"]
                    energy_use_j = pl.Series(energy_use_j)
                    speed_limit_train_sims.append(slts)
                    est_time_nets.append(est_time_net)
                    loco_pool = loco_pool.with_columns(
                        pl.when(selected)
                            .then(pl.lit(this_train['Destination']))
                            .otherwise(pl.col('Node')).alias("Node"),
                        pl.when(selected)
                            .then(pl.lit(current_time + travel_time))
                            .otherwise(pl.col('Arrival_Time')).alias("Arrival_Time"),
                        pl.when(selected)
                            .then(pl.lit(current_time + travel_time) + pl.col('Min_Servicing_Time_Hr'))
                            .otherwise(pl.col('Servicing_Done_Time')).alias("Servicing_Done_Time"),
                        pl.when(selected)
                            .then(None)
                            .otherwise(pl.col('Refueling_Done_Time')).alias("Refueling_Done_Time"),
                        pl.when(selected)
                            .then(pl.lit("Dispatched"))
                            .otherwise(pl.col('Status')).alias("Status"),
                        pl.when(selected)
                        .then(pl.max_horizontal(
                                pl.col('SOC_Min_J'),
                                pl.min_horizontal(
                                    pl.col('SOC_J') - pl.lit(energy_use_j), 
                                    pl.col('SOC_Max_J'))))
                        .otherwise(pl.col('SOC_J')).alias("SOC_J")
                    )

                    # Populate the output dataframe with the dispatched trains
                    new_row_count = selected.sum()
                    new_rows = pl.DataFrame([
                        pl.Series(repeat(train_id_counter, new_row_count)),
                        pl.Series(repeat(this_train['Train_Type'], new_row_count)),
                        loco_pool.filter(selected).get_column('Locomotive_ID'),
                        loco_pool.filter(selected).get_column('Locomotive_Type'),
                        pl.Series(repeat(this_train['Origin'], new_row_count)),
                        pl.Series(repeat(this_train['Destination'], new_row_count)),
                        pl.Series(repeat(this_train['Cars_Loaded'], new_row_count)),
                        pl.Series(repeat(this_train['Cars_Empty'], new_row_count)),
                        loco_start_soc_j,
                        pl.Series(repeat(current_time, new_row_count)),
                        pl.Series(repeat(current_time + travel_time, new_row_count))],
                        schema = train_consist_plan.columns,
                        orient="col")
                    train_consist_plan = pl.concat([train_consist_plan, new_rows], how="diagonal_relaxed")
                    train_id_counter += 1

        if current_time >= final_departure:
            current_time = float("inf")
            loco_pool, event_tracker = update_refuel_queue(loco_pool, refuelers, current_time, event_tracker)
            done = True
        else:
            current_time = dispatch_schedule.filter(pl.col("Hour").gt(current_time)).get_column("Hour").min()

    train_consist_plan = (train_consist_plan
        .with_columns(
            cs.categorical().cast(str),
            pl.col("Train_ID", "Locomotive_ID").cast(pl.UInt32)
        )
        .sort(["Locomotive_ID", "Train_ID"], descending=False)
    )
    loco_pool = loco_pool.with_columns(cs.categorical().cast(str))
    refuelers = refuelers.with_columns(cs.categorical().cast(str))
    
    event_tracker = event_tracker.sort(["Locomotive_ID","Time_Hr","Event_Type"])
    service_starts = (event_tracker
        .filter(pl.col("Event_Type") == "Refueling_Start")
        .get_column("Time_Hr")
        .rename("Refuel_Start_Time_Planned_Hr"))
    service_ends = (event_tracker
        .filter(pl.col("Event_Type") == "Refueling_End")
        .get_column("Time_Hr")
        .rename("Refuel_End_Time_Planned_Hr"))
        
    train_consist_plan = train_consist_plan.with_columns(
        service_starts, service_ends
    )                    
    
    return train_consist_plan, loco_pool, refuelers, speed_limit_train_sims, est_time_nets


if __name__ == "__main__":

    rail_vehicles=[alt.RailVehicle.from_file(vehicle_file) 
                for vehicle_file in Path(alt.resources_root() / "rolling_stock/").glob('*.yaml')]

    location_map = alt.import_locations(
        str(alt.resources_root() / "networks/default_locations.csv")
    )
    network = alt.Network.from_file(
        str(alt.resources_root() / "networks/Taconite-NoBalloon.yaml")
    )
    config = planner_config.TrainPlannerConfig()
    loco_pool = build_locopool(config, defaults.DEMAND_FILE)
    demand, node_list = demand_loader(defaults.DEMAND_FILE)
    refuelers = build_refuelers(
        node_list, 
        loco_pool,
        config.refueler_info, 
        config.refuelers_per_incoming_corridor)

    output = run_train_planner(
        rail_vehicles=rail_vehicles, 
        location_map=location_map, 
        network=network,
        loco_pool=loco_pool,
        refuelers=refuelers,
        simulation_days=defaults.SIMULATION_DAYS + 2 * defaults.WARM_START_DAYS,
        scenario_year=defaults.BASE_ANALYSIS_YEAR,
        config=config)
