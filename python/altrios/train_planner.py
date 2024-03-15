from pathlib import Path
from typing import Union
import numpy as np
from scipy.stats import rankdata
import pandas as pd
import polars as pl
import polars.selectors as cs
import math
from typing import Tuple, List, Dict
from itertools import repeat
import altrios as alt
from altrios import defaults, utilities

pl.enable_string_cache(True)

class TrainPlannerConfig:
    def __init__(self, 
                 min_cars_per_train: int = 60,
                 target_cars_per_train: int = 180,
                 manifest_empty_return_ratio: float = 0.6,
                 cars_per_locomotive: int = 70,
                 refuelers_per_incoming_corridor: int = 4,
                 hp_required_per_ton: Dict = {
                     "Default": {
                        "Unit": 2.0,
                        "Manifest": 1.5,
                        "Intermodal": 2.0 + 2.0,
                        "Unit_Empty": 2.0,
                        "Manifest_Empty": 1.5,
                        "Intermodal_Empty": 2.0 + 2.0,
                        }                         
                     }, 
                 dispatch_scaling_dict: Dict = {
                     "time_mult_factor": 1.4,
                     "hours_add": 2,
                     "energy_mult_factor": 1.25
                     },
                 loco_info = pd.DataFrame({
                    "Diesel_Large": {
                        "Capacity_Cars": 20,
                        "Min_Servicing_Time_Hr": 3.0,
                        "Rust_Loco": alt.Locomotive.default(),
                        "Cost_USD": defaults.DIESEL_LOCO_COST_USD,
                        "Lifespan_Years": defaults.LOCO_LIFESPAN
                        },
                    "BEL": {
                        "Capacity_Cars": 20,
                        "Min_Servicing_Time_Hr": 3.0,
                        "Rust_Loco": alt.Locomotive.default_battery_electric_loco(),
                        "Cost_USD": defaults.BEL_MINUS_BATTERY_COST_USD,
                        "Lifespan_Years": defaults.LOCO_LIFESPAN
                        }
                    }).transpose().reset_index(names='Locomotive_Type'),
                 refueler_info = pd.DataFrame({
                    "Diesel_Fueler": {
                        "Locomotive_Type": "Diesel_Large",
                        "Refueler_J_Per_Hr": defaults.DIESEL_REFUEL_RATE_J_PER_HR,
                        "Refueler_Efficiency": defaults.DIESEL_REFUELER_EFFICIENCY,
                        "Cost_USD": defaults.DIESEL_REFUELER_COST_USD,
                        "Lifespan_Years": defaults.LOCO_LIFESPAN
                    },
                    "BEL_Charger": {
                        "Locomotive_Type": "BEL",
                        "Refueler_J_Per_Hr": defaults.BEL_CHARGE_RATE_J_PER_HR,
                        "Refueler_Efficiency": defaults.BEL_CHARGER_EFFICIENCY,
                        "Cost_USD": defaults.BEL_CHARGER_COST_USD,
                        "Lifespan_Years": defaults.LOCO_LIFESPAN
                    }
                }).transpose().reset_index(names='Refueler_Type')
    ):
        """
        Constructor for train planner configuration objects
        Arguments:
        ----------
        min_cars_per_train: the minimum length in number of cars to form a train
        target_cars_per_train: the standard train length in number of cars
        manifest_empty_return_ratio: Desired railcar reuse ratio to calculate the empty manifest car demand, (E_ij+E_ji)/(L_ij+L_ji)
        cars_per_locomotive: Heuristic scaling factor used to size number of locomotives needed based on demand.
        refuelers_per_incoming_corridor: 
        hp_required_per_ton:
        dispatch_scaling_dict:
        loco_info:
        refueler_info:
        """
        self.min_cars_per_train = min_cars_per_train
        self.target_cars_per_train = target_cars_per_train
        self.manifest_empty_return_ratio = manifest_empty_return_ratio
        self.cars_per_locomotive = cars_per_locomotive
        self.refuelers_per_incoming_corridor = refuelers_per_incoming_corridor
        self.hp_required_per_ton = hp_required_per_ton
        self.dispatch_scaling_dict = dispatch_scaling_dict
        self.loco_info = loco_info
        self.refueler_info = refueler_info

def demand_loader(
    user_input_file: Union[Path, str]
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
    demand = pl.read_csv(user_input_file, dtypes = {"Number_of_Cars": pl.Float64})

    nodes = pl.concat(
        [demand.get_column("Origin"),
        demand.get_column("Destination")]).unique().sort()
    return demand, nodes


def generate_return_demand(
    demand: pl.DataFrame,
    config: TrainPlannerConfig
) -> pl.DataFrame:
    """
    Create a dataframe for additional demand needed for empty cars of the return trains
    Arguments:
    ----------
    df_annual_demand: The user_input file loaded by previous functions
    that contains laoded demand for each demand pair.
    config: Object storing train planner configuration paramaters
    Outputs:
    ----------
    df_return_demand: The demand generated by the need
    of returning the empty cars to their original nodes
    """
    return (demand
        .with_columns(
            pl.col("Origin").alias("Destination"),
            pl.col("Destination").alias("Origin"),
            pl.concat_str([pl.col("Train_Type"),pl.lit("_Empty")]).alias("Train_Type"),
            pl.when(pl.col("Train_Type") == pl.lit("Manifest"))
                .then((pl.col("Number_of_Cars") * config.manifest_empty_return_ratio).floor())
                .otherwise(pl.col("Number_of_Cars"))
                .alias("Number_of_Cars"))            
        .drop("Number_of_Containers"))

def generate_origin_manifest_demand(
    demand: pl.DataFrame,
    node_list: List[str],
    config: TrainPlannerConfig
) -> pl.DataFrame:
    """
    Create a dataframe for summarized view of all origins' manifest demand
    in number of cars and received cars, both with loaded and empty counts
    Arguments:
    ----------
    demand: The user_input file loaded by previous functions
    that contains laoded demand for each demand pair.
    node_list: A list containing all the names of nodes in the system    
    config: Object storing train planner configuration paramaters

    Outputs:
    ----------
    origin_manifest_demand: The dataframe that summarized all the manifest demand
    originated from each node by number of loaded and empty cars
    with additional columns for checking the unbalance quantity and serve as check columns
    for the manifest empty car rebalancing function
    """
    manifest_demand = (demand
        .filter(pl.col("Train_Type") == "Manifest")
        .select(["Origin", "Destination","Number_of_Cars"])
        .rename({"Number_of_Cars": "Manifest"})
        .unique())
    
    origin_volume = manifest_demand.groupby("Origin").agg(pl.col("Manifest").sum())
    destination_volume = manifest_demand.groupby("Destination").agg(pl.col("Manifest").sum().alias("Manifest_Reverse"))
    origin_manifest_demand = (pl.DataFrame({"Origin": node_list})
        .join(origin_volume, left_on="Origin", right_on="Origin", how="left")
        .join(destination_volume, left_on="Origin", right_on="Destination", how="left")
        .with_columns(
            (pl.col("Manifest_Reverse") * config.manifest_empty_return_ratio).floor().alias("Manifest_Empty"))
        .with_columns(
            (pl.col("Manifest") + pl.col("Manifest_Empty")).alias("Manifest_Dispatched"),
            (pl.col("Manifest_Reverse") + pl.col("Manifest") * config.manifest_empty_return_ratio).floor().alias("Manifest_Received"))
        .drop("Manifest_Reverse")
        .filter((pl.col("Manifest").is_not_null()) | (pl.col("Manifest_Empty").is_not_null()))
    )

    return origin_manifest_demand


def balance_trains(
    demand_origin_manifest: pl.DataFrame
) -> pl.DataFrame:
    """
    Update the manifest demand, especially the empty car demand to maintain equilibrium of number of
    cars dispatched and received at each node for manifest
    Arguments:
    ----------
    demand_origin_manifest: Dataframe that summarizes empty and loaded 
    manifest demand dispatched and received for each node by number cars
    Outputs:
    ----------
    demand_origin_manifest: Updated demand_origin_manifest with additional
    manifest empty car demand added to each node
    df_balance_storage: Documented additional manifest demand pairs and corresponding quantity for
    rebalancing process
    """
    df_balance_storage = pd.DataFrame(np.zeros(shape=(0, 4)))
    df_balance_storage = df_balance_storage.rename(
        columns={0: "Origin", 
                 1: "Destination", 
                 2: "Train_Type", 
                 3: "Number_of_Cars"})
    
    train_type = "Manifest_Empty"
    demand = demand_origin_manifest.to_pandas()[
        ["Origin","Manifest_Received","Manifest_Dispatched","Manifest_Empty"]]
    demand = demand.rename(columns={"Manifest_Received": "Received", 
                            "Manifest_Dispatched": "Dispatched",
                            "Manifest_Empty": "Empty"})

    step = 0
    # Calculate the number of iterations needed
    max_iter = len(demand) * (len(demand)-1) / 2
    while (~np.isclose(demand["Received"], demand["Dispatched"])).any() and (step <= max_iter):
        rows_def = demand[demand["Received"] < demand["Dispatched"]]
        rows_sur = demand[demand["Received"] > demand["Dispatched"]]
        if((len(rows_def) == 0) | (len(rows_sur) == 0)): 
            break
        # Find the first node that is in deficit of cars because of the empty return
        row_def = rows_def.index[0]
        # Find the first node that is in surplus of cars
        row_sur = rows_sur.index[0]
        surplus = demand.loc[row_sur, "Received"] - demand.loc[row_sur, "Dispatched"]
        df_balance_storage.loc[len(df_balance_storage.index)] = \
            [demand.loc[row_sur, "Origin"],
            demand.loc[row_def, "Origin"],
            train_type,
            surplus]
        demand.loc[row_def, "Received"] += surplus
        demand.loc[row_sur, "Dispatched"] = demand.loc[row_sur, "Received"]
        step += 1
        
    if (~np.isclose(demand["Received"], demand["Dispatched"])).any():
        raise Exception("While loop didn't converge")
    return pl.from_pandas(df_balance_storage)

def generate_demand_trains(
    demand: pl.DataFrame,
    demand_returns: pl.DataFrame,
    demand_rebalancing: pl.DataFrame,
    rail_vehicle_map: Dict[str, alt.RailVehicle],
    config: TrainPlannerConfig
) -> pl.DataFrame:
    """
    Generate a tabulated demand pair to indicate the final demand
    for each demand pair for each train type in number of trains
    Arguments:
    ----------
    demand: Tabulated demand for each demand pair for each train type in number of cars

    demand: The user_input file loaded and prepared by previous functions
    that contains loaded car demand for each demand pair.
    demand_returns: The demand generated by the need 
    of returning the empty cars to their original nodes
    demand_rebalancing: Documented additional manifest demand pairs and corresponding quantity for
    rebalancing process

    config: Object storing train planner configuration paramaters
    Outputs:
    ----------
    demand: Tabulated demand for each demand pair in terms of number of cars and number of trains
    """
    
    demand = pl.concat([
        demand.drop("Number_of_Containers"),
        demand_returns,
        demand_rebalancing],
        how="diagonal")
    #MPrepare ton_per_car requirements to merge onto the demand DataFrame
    def get_kg_empty(veh):
        return veh.mass_static_empty_kilograms + veh.axle_count * veh.mass_extra_per_axle_kilograms
    def get_kg_loaded(veh):
        return veh.mass_static_loaded_kilograms + veh.axle_count * veh.mass_extra_per_axle_kilograms
    vehicle_types = [
        vt.replace("_Empty", "") for vt in demand.get_column("Train_Type").unique().to_list()
    ]
    kg_empty = [get_kg_empty(rail_vehicle_map[veh_type]) for veh_type in vehicle_types]
    kg_loaded = [get_kg_loaded(rail_vehicle_map[veh_type]) for veh_type in vehicle_types]
    ton_per_car = (
        pl.DataFrame({"Train_Type": vehicle_types,
                        "KG_Empty": kg_empty,
                        "KG_Loaded": kg_loaded})
            .with_columns(pl.when(pl.col("Train_Type").str.contains("_Empty"))
                                .then(pl.col("KG_Empty") / utilities.KG_PER_TON)
                                .otherwise(pl.col("KG_Loaded") / utilities.KG_PER_TON)
                                .alias("Tons_Per_Car"))
            .drop(["KG_Empty","KG_Loaded"])
    )
    demand = demand.join(ton_per_car, on="Train_Type", how="left")
    #Prepare hp_per_ton requirements to merge onto the demand DataFrame
    hp_per_ton = (
        pl.DataFrame(pd.DataFrame(config.hp_required_per_ton).reset_index(names="Train_Type"))
        .melt(id_vars="Train_Type",variable_name="O_D",value_name="HP_Required_Per_Ton")
        .with_columns(pl.col("O_D").str.split("_").list.first().alias("Origin"),
                      pl.col("O_D").str.split("_").list.last().alias("Destination"))
    )
    
    #Get Intermodal_Empty and Manifest_Empty car counts and tonnage ready to merge onto demand DataFrame
    demand_empty = (demand
        .filter(pl.col("Train_Type").is_in(["Intermodal_Empty","Manifest_Empty"]))
        .with_columns(pl.col("Train_Type").str.replace("_Empty","").alias("Train_Type"))
        .groupby(["Origin","Destination","Train_Type"])
        .agg(pl.col("Number_of_Cars").sum(), pl.col("Tons_Per_Car").mean())
    )

    demand = (demand
        # First, merge on OD-specific hp_per_ton if the user specified any
        .join(hp_per_ton.drop("O_D"), 
            on=["Origin","Destination","Train_Type"], 
            how="left")
        # Second, merge on defaults per train type
        .join(hp_per_ton.filter((pl.col("O_D") =="Default")).drop(["O_D","Origin","Destination"]),
            on=["Train_Type"],
            how="left",
            suffix="_Default")
        # Fill in defaults per train type wherever the user didn't specify OD-specific hp_per_ton
        .with_columns(pl.when(pl.col("HP_Required_Per_Ton").is_not_null())
                    .then(pl.col("HP_Required_Per_Ton"))
                    .otherwise(pl.col("HP_Required_Per_Ton_Default"))
                    .alias("HP_Required_Per_Ton"))
        .drop("HP_Required_Per_Ton_Default")
    )

    demand = (demand
        # Merge Intermodal_Empty and Manifest_Empty car counts and tonnage
        .filter(~pl.col("Train_Type").is_in(["Intermodal_Empty","Manifest_Empty"]))
        .join(demand_empty, 
              on = ["Origin","Destination","Train_Type"],
              how = "left",
              suffix = "_Empty")
        # Fix up Unit and Unit_Empty to have information in the correct column
        .with_columns(
            pl.when(pl.col("Train_Type")=="Unit_Empty")
                .then(pl.col("Number_of_Cars"))
                .otherwise(pl.col("Number_of_Cars_Empty"))
                .alias("Number_of_Cars_Empty"),
            pl.when(pl.col("Train_Type")=="Unit_Empty")
                .then(pl.col("Tons_Per_Car"))
                .otherwise(pl.col("Tons_Per_Car_Empty"))
                .alias("Tons_Per_Car_Empty"),
            pl.when(pl.col("Train_Type")=="Unit_Empty")
                .then(pl.col("Number_of_Cars_Empty"))
                .otherwise(pl.col("Number_of_Cars"))
                .alias("Number_of_Cars"),
            pl.when(pl.col("Train_Type")=="Unit_Empty")
                .then(pl.col("Tons_Per_Car_Empty"))
                .otherwise(pl.col("Tons_Per_Car"))
                .alias("Tons_Per_Car"))
        # Replace nulls with zero
        .with_columns(cs.numeric().fill_null(0))
        # Calculate total number of cars and total aggregate tonnage
        .with_columns(
            (pl.col("Number_of_Cars") + pl.col("Number_of_Cars_Empty")).alias("Number_of_Cars_Total"),
            (pl.col("Number_of_Cars") * pl.col("Tons_Per_Car") + \
                pl.col("Number_of_Cars_Empty") * pl.col("Tons_Per_Car_Empty")).alias("Tons_Aggregate"))
        # Convert total number of cars to total number of trains
        .with_columns(
            pl.when(pl.col("Number_of_Cars_Total") == pl.lit(0))
                .then(0)
                .otherwise(
                    pl.max([1,
                        (pl.when(pl.col("Number_of_Cars_Total").mod(pl.lit(config.target_cars_per_train)).gt(pl.lit(config.min_cars_per_train)))
                            .then(pl.col("Number_of_Cars_Total").floordiv(pl.lit(config.target_cars_per_train)))
                            .otherwise(pl.col("Number_of_Cars_Total").floordiv(pl.lit(config.target_cars_per_train))+1))
                        ])
                )
                .alias("Number_of_Trains"))
        # Calculate per-train car counts and tonnage
        .with_columns(
            pl.col("Number_of_Cars").floordiv(pl.col("Number_of_Trains")).alias("Cars_Per_Train_Loaded"),
            pl.col("Number_of_Cars_Empty").floordiv(pl.col("Number_of_Trains")).alias("Cars_Per_Train_Empty"),
            pl.col("Tons_Aggregate").truediv(pl.col("Number_of_Trains")).ceil().alias("Tons_Per_Train_Total"))
    )
    return demand


def calculate_dispatch_times(
    demand: pl.DataFrame,
    hours: int
) -> pl.DataFrame:
    """
    Generate a tabulated demand pair to indicate the expected dispatching interval
    and actual dispatching timesteps after rounding
    Arguments:
    ----------
    config: Object storing train planner configuration paramaters
    demand_train: Dataframe of demand (number of trains) for each OD pair for each train type
    hours: Number of hours in the simulation time period
    Outputs:
    ----------
    dispatch_times: Tabulated dispatching time for each demand pair for each train type
    in hours
    """
    return (demand
        .filter(pl.col("Number_of_Trains") > 0)
        .select(["Origin","Destination","Train_Type","Number_of_Trains",
            "Cars_Per_Train_Loaded","Cars_Per_Train_Empty",
            "Tons_Per_Train_Total","HP_Required_Per_Ton"])
        .with_columns(
            (hours / pl.col("Number_of_Trains")).alias("Interval"),
            pl.col("Number_of_Trains").cast(pl.Int32).alias("Number_of_Trains"))
        .select(pl.exclude("Number_of_Trains").repeat_by("Number_of_Trains").explode())
        .with_columns(
            ((pl.col("Interval").cumcount().over(["Origin","Destination","Train_Type"]) + 1) \
             * pl.col("Interval")).alias("Hour"))
        .drop("Interval")
        .sort(["Hour","Origin","Destination","Train_Type"])
    )

def build_locopool(
    config: TrainPlannerConfig,
    demand_file: str,
    method: str = "tile",
    shares: List[float] = [],
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
    num_ods = demand.height
    cars_per_od = demand.get_column("Number_of_Cars").mean()
    num_destinations_per_node = num_ods*1.0 / num_nodes*1.0
    initial_size = math.ceil((cars_per_od / config.cars_per_locomotive) *
                             num_destinations_per_node)  # number of locomotives per node

    rows = initial_size * num_nodes  # number of locomotives in total
    sorted_nodes = np.sort(np.tile(node_list, initial_size)).tolist()
    engine_numbers = rankdata(sorted_nodes, method="dense") * 1000 + \
        np.tile(range(0, initial_size), num_nodes)

    if method == "tile":
        repetitions = math.ceil(rows/len(loco_types))
        types = np.tile(loco_types, repetitions).tolist()[0:rows]
    elif method == "shares_twoway":
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
        schema_overrides={'Locomotive_Type': pl.Categorical}
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
        .groupby(pl.col("Locomotive_Type").cast(pl.Utf8))
        .agg([(pl.lit(refuelers_per_incoming_corridor) * pl.count() / pl.lit(loco_pool.height))
              .ceil()
              .alias("Ports_Per_Node")])
        .join(pl.from_pandas(refueler_info),
              on="Locomotive_Type", 
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
        .groupby(pl.col("Origin"))
        .agg(pl.col("Allowable Battery Headroom (MWh)").min() * 1e6 / utilities.MWH_PER_MJ)
        .rename({"Allowable Battery Headroom (MWh)": "Battery_Headroom_J"})
        .with_columns(pl.col("Origin").cast(pl.Categorical)))
    refuelers = (refuelers
        .join(network_charging_guidelines, left_on="Node", right_on="Origin", how="left")
        .with_columns(pl.when(pl.col("Locomotive_Type")=="BEL")
            .then(pl.col("Battery_Headroom_J"))
            .otherwise(0)
            .fill_null(0)
            .alias("Battery_Headroom_J")
            ))
    loco_pool = (loco_pool
        .join(network_charging_guidelines, left_on="Node", right_on="Origin", how="left")
        .with_columns(pl.when(pl.col("Locomotive_Type")=="BEL")
            .then(pl.col("Battery_Headroom_J"))
            .otherwise(0)
            .fill_null(0)
            .alias("Battery_Headroom_J"))
        .with_columns(pl.max([pl.col('SOC_Max_J')-pl.col('Battery_Headroom_J'), pl.col('SOC_Min_J')]).alias("SOC_J")))
    return refuelers, loco_pool

def append_loco_info(loco_info: pd.DataFrame) -> pd.DataFrame:
    if all(item in loco_info.columns for item in [
        'HP','Loco_Mass_Tons','SOC_J','SOC_Min_J','SOC_Max_J','Capacity_J'
        ]
    ): return loco_info
    
    get_hp = lambda loco: loco.pwr_rated_kilowatts * 1e3 / alt.utils.W_PER_HP
    get_mass_ton = lambda loco: loco.mass_kg / alt.utils.KG_PER_TON
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

def dispatch(
    dispatch_time: int,
    ton: float,
    origin: str,
    loco_pool: pl.DataFrame,
    hp_per_ton: float,
) -> pl.Series:
    """
    Update the locomotive pool by identifying the desired locomotive to dispatch and assign to the
    new location (destination) with corresponding updated ready time
    Arguments:
    ----------
    dispatch_time: time that a train is due
    ton: required tonnage in the train
    origin: origin node name of the train
    loco_pool: locomotive pool dataframe containing all locomotives in the network
    hp_per_ton: Horsepower per ton required for this train type on this origin-destination corridor
    Outputs:
    ----------
    selected: Indices of selected locomotives
    """
    hp_required = ton * hp_per_ton

    # Candidate locomotives at the right place that are ready

    candidates = loco_pool.select((pl.col("Node") == origin) &
                                (pl.col("Status") == "Ready")).to_series()
    if not candidates.any():
        message = f"""No available locomotives at node {origin} at hour {dispatch_time}."""
        waiting_counts = loco_pool.filter((pl.col("Status").is_in(["Servicing","Refuel_Queue"])) & (pl.col("Node") == origin)
                                        ).groupby(['Locomotive_Type']).count()
        if waiting_counts.height == 0:
            message = message + f"""\nNo locomotives are currently located there. Instead, they are at:"""
            locations = loco_pool.groupby("Node").agg(pl.count())
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
    # First available diesel (in order of loco_pool) will be moved from candidates to selected
    # TODO gracefully handle cases when there is no diesel locomotive to be dispatched
    # (ex: hold the train until enough diesels are present)
    diesel_candidates = loco_pool.select(pl.lit(candidates) &
                                       pl.col("Locomotive_Type").cast(pl.Utf8).str.contains("(?i)diesel")).to_series()
    if not diesel_candidates.any():
        refueling_diesel_count = loco_pool.filter(
            (pl.col("Node") == origin) &
            (pl.col("Status").is_in(["Servicing","Refuel_Queue"])) &
            (pl.col("Locomotive_Type").cast(pl.Utf8).str.contains("(?i)diesel"))
        ).select(pl.count())[0, 0]
        message = f"""No available diesel locomotives at node {origin} at hour {dispatch_time}, so
                the one-diesel-per-consist rule cannot be satisfied. {refueling_diesel_count} diesel locomotives at
                {origin} are servicing, refueling, or queueing."""
        if refueling_diesel_count > 0:
            diesel_port_count = loco_pool.filter(
                (pl.col("Node") == origin) &
                (pl.col("Locomotive_Type").cast(pl.Utf8).str.contains("(?i)diesel"))
            ).select(pl.col("Port_Count").min())[0, 0]
            message += f""" (queue capacity {diesel_port_count})."""
        else:
            message += "."
        raise ValueError(message)

    diesel_to_require = diesel_candidates.eq(True).cumsum().eq(1).arg_max()
    diesel_to_require_hp = loco_pool.filter(
        pl.col("Locomotive_Type") == "Diesel_Large").select(pl.first("HP"))
    # Need to mask this so it's not double-counted on next step
    candidates[diesel_to_require] = False
    # Get running sum, including first diesel, of hp of the candidates (in order of loco_pool)
    enough_hp = loco_pool.select((
        (
            (pl.col("HP") - (pl.col("Loco_Mass_Tons") * pl.lit(hp_per_ton))) * pl.lit(candidates)
        ).cumsum() + pl.lit(diesel_to_require_hp)) >= hp_required).to_series()
    if not enough_hp.any():
        available_hp = loco_pool.select(
            (
                (pl.col("HP") - (pl.col("Loco_Mass_Tons") * pl.lit(hp_per_ton))) * pl.lit(candidates)
            ).cumsum().max())[0, 0]
        message = f"""Outbound horsepower needed ({hp_required}) at {origin} at hour {dispatch_time}
            is more than the available horsepower ({available_hp}).
            Count of locomotives servicing, refueling, or queueing at {origin} are:"""
        # Hold the train until enough diesels are present (future development)
        waiting_counts = loco_pool.filter(
            (pl.col("Node") == origin) & (pl.col("Status").is_in(["Servicing","Refuel_Queue"]))
        ).select("Locomotive_Type").groupby(['Locomotive_Type']).count()
        for row in waiting_counts.iter_rows(named = True):
            message = message + f"""
            {row['Locomotive_Type']}: {row['count']}"""
        # Hold the train until enough locomotives are present (future development)
        raise ValueError(message)

    last_row_to_use = enough_hp.eq(True).cumsum().eq(1).arg_max()
    # Set false all the locomotives that would add unnecessary hp
    selected[np.arange(last_row_to_use+1, len(selected))] = False
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
                refuelers.select(["Node","Locomotive_Type","Refueler_J_Per_Hr","Port_Count",'Battery_Headroom_J']), 
                left_on=["Node", "Locomotive_Type"],
                right_on=["Node", "Locomotive_Type"],
                how="left")
            .with_columns(
                pl.when(arrived)
                    .then(pl.lit("Refuel_Queue"))
                    .otherwise(pl.col("Status")).alias("Status"),
                pl.when(arrived)
                    .then(pl.max([pl.col('SOC_Max_J')-pl.col('Battery_Headroom_J'), pl.col('SOC_J')]))
                    .otherwise(pl.col("SOC_Target_J")).alias("SOC_Target_J"))
            .with_columns(
                pl.when(arrived)
                    .then((pl.col("SOC_Target_J")-pl.col("SOC_J"))/pl.col("Refueler_J_Per_Hr"))
                    .otherwise(pl.col("Refuel_Duration")).alias("Refuel_Duration"))
            .sort("Node", "Locomotive_Type","Arrival_Time", "Locomotive_ID", descending = False, nulls_last = True))
        charger_type_breakouts = (loco_pool.filter(
            (pl.col("Status") == "Refuel_Queue") &
            ((pl.col("Refueling_Done_Time") >= current_time) 
                | (pl.col("Refueling_Done_Time").is_null())))
            .partition_by(["Node","Locomotive_Type"]))
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
        (pl.col("Status") == "Refuel_Queue") & (pl.col("Refueling_Done_Time") <= current_time)).to_series()
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
    rail_vehicle_map: Dict[str, alt.RailVehicle],
    location_map: Dict[str, List[alt.Location]],
    network: List[alt.Link],
    loco_pool: pl.DataFrame,
    refuelers: pl.DataFrame,
    simulation_days: int,
    scenario_year: int,
    train_type: alt.TrainType = alt.TrainType.Freight, 
    config: TrainPlannerConfig = TrainPlannerConfig(),
    demand_file_path=defaults.DEMAND_FILE,
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
    rail_vehicle_map:
    location_map:
    network:
    loco_pool:
    refuelers:
    simulation_days:
    config: Object storing train planner configuration paramaters
    demand_file_path: 
    Outputs:
    ----------
    """

    config.loco_info = append_loco_info(config.loco_info)
    demand, node_list = demand_loader(demand_file_path)
    if refuelers is None: 
        refuelers = build_refuelers(
            node_list, 
            loco_pool,
            config.refueler_info, 
            config.refuelers_per_incoming_corridor)
        
    if network_charging_guidelines is None: 
        network_charging_guidelines = pl.read_csv(alt.resources_root() / "networks" / "network_charging_guidelines.csv")

    refuelers, loco_pool = append_charging_guidelines(refuelers, loco_pool, demand, network_charging_guidelines)
    demand_returns = generate_return_demand(demand, config)
    demand_rebalancing = pl.DataFrame()
    if demand.filter(pl.col("Train_Type").str.contains("Manifest")).height > 0:
        demand_origin_manifest = generate_origin_manifest_demand(demand, node_list, config)
        demand_rebalancing = balance_trains(demand_origin_manifest)
    demand = generate_demand_trains(demand, demand_returns, demand_rebalancing, rail_vehicle_map, config)
    dispatch_times = calculate_dispatch_times(demand, simulation_days * 24)
    final_departure = dispatch_times.get_column("Hour").max()
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
    current_time = dispatch_times.get_column("Hour").min()
    while not done:
        # Dispatch new train consists
        current_dispatches = dispatch_times.filter(pl.col("Hour") == current_time)
        if(current_dispatches.height > 0):
            loco_pool, event_tracker = update_refuel_queue(loco_pool, refuelers, current_time, event_tracker)

            for this_train in current_dispatches.iter_rows(named = True):
                if this_train['Tons_Per_Train_Total'] > 0:
                    train_id=str(train_id_counter)
                    selected = dispatch(
                        current_time,
                        this_train['Tons_Per_Train_Total'],
                        this_train['Origin'],
                        loco_pool,
                        this_train['HP_Required_Per_Ton']
                    )
                    train_config = alt.TrainConfig(
                        cars_empty = int(this_train['Cars_Per_Train_Empty']),
                        cars_loaded = int(this_train['Cars_Per_Train_Loaded']),
                        rail_vehicle_type = this_train['Train_Type'],
                        train_type = train_type,
                    )
                    
                    dispatched = loco_pool.filter(selected)
                    loco_start_soc_j = dispatched.get_column("SOC_J")
                    dispatch_order =  (dispatched.select(
                        pl.col('Locomotive_ID')
                        .rank().alias('rank').cast(pl.UInt32)
                        ).with_row_count().sort('row_nr'))
                    dispatched = dispatched.sort('Locomotive_ID')
                    loco_types = dispatched.get_column('Locomotive_Type')
                    loco_start_soc_pct = dispatched.select(pl.col('SOC_J') / pl.col('Capacity_J')).to_series()
                    locos = [
                        config.loco_info[config.loco_info['Locomotive_Type']==loco_type]['Rust_Loco'].to_list()[0].clone() 
                        for loco_type in loco_types
                    ]
                    [alt.set_param_from_path(
                        locos[i], 
                        "res.state.soc", 
                        loco_start_soc_pct[i]
                    ) for i in range(len(locos)) if loco_types[i] == 'BEL']

                    loco_con = alt.Consist(
                        loco_vec=locos,
                        save_interval=None,
                    )

                    init_train_state = alt.InitTrainState(
                        time_seconds=current_time * 3600,
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
                        rail_vehicle_map[train_config.rail_vehicle_type], 
                        location_map, 
                        None, 
                        simulation_days, 
                        scenario_year
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
                        .then(pl.max(
                                pl.col('SOC_Min_J'),
                                pl.min(
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
                        pl.Series(repeat(this_train['Cars_Per_Train_Loaded'], new_row_count)),
                        pl.Series(repeat(this_train['Cars_Per_Train_Empty'], new_row_count)),
                        loco_start_soc_j,
                        pl.Series(repeat(current_time, new_row_count)),
                        pl.Series(repeat(current_time + travel_time, new_row_count))],
                        schema = train_consist_plan.columns,
                        orient="col")
                    
                    train_consist_plan = pl.concat([train_consist_plan, new_rows])
                    train_id_counter += 1

        if current_time >= final_departure:
            current_time = float("inf")
            loco_pool, event_tracker = update_refuel_queue(loco_pool, refuelers, current_time, event_tracker)
            done = True
        else:
            current_time = dispatch_times.filter(pl.col("Hour").gt(current_time)).get_column("Hour").min()

    train_consist_plan = (train_consist_plan.with_columns(
        pl.col("Train_Type").cast(str),
        pl.col("Locomotive_Type").cast(str),
        pl.col("Origin_ID").cast(str),
        pl.col("Destination_ID").cast(str),
        pl.col("Train_ID").cast(pl.UInt32),
        pl.col("Locomotive_ID").cast(pl.UInt32))
        .sort(["Locomotive_ID", "Train_ID"], descending=False))
    loco_pool = loco_pool.with_columns(
        pl.col("Locomotive_Type").cast(str),
        pl.col("Node").cast(str),
    )
    refuelers = refuelers.with_columns(
        pl.col("Refueler_Type").cast(str),
        pl.col("Locomotive_Type").cast(str),
        pl.col("Node").cast(str),
    )
    
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
    rail_vehicle_map = alt.import_rail_vehicles(
        str(alt.resources_root() / "rolling_stock/rail_vehicles.csv")
    )
    location_map = alt.import_locations(
        str(alt.resources_root() / "networks/default_locations.csv")
    )
    network = alt.Network.from_file(
        str(alt.resources_root() / "networks/Taconite.yaml")
    )
    config = TrainPlannerConfig()
    loco_pool = build_locopool(config, defaults.DEMAND_FILE)
    demand, node_list = demand_loader(defaults.DEMAND_FILE)
    refuelers = build_refuelers(
        node_list, 
        loco_pool,
        config.refueler_info, 
        config.refuelers_per_incoming_corridor)

    output = run_train_planner(
        rail_vehicle=rail_vehicle_map, 
        location_map=location_map, 
        network=network,
        loco_pool=loco_pool,
        refuelers=refuelers,
        simulation_days=defaults.SIMULATION_DAYS + 2 * defaults.WARM_START_DAYS,
        scenario_year=defaults.BASE_ANALYSIS_YEAR,
        config=config)
    # print(output)
