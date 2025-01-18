import math
from typing import List, Union
from collections import defaultdict
import polars as pl
import altrios as alt
from altrios import utilities
from altrios.train_planner import planner_config

def calculate_waiting_time_single_dispatch(
    cumulative_demand_control: int, 
    last_dispatch: int, 
    demand_hourly: pl.DataFrame, 
    dispatch_hour: int, 
    remaining_demand_list: pl.DataFrame, 
    remaining_demand_list_control: pl.DataFrame, 
    search_range: int,
    od_pair_loop: str,
    min_num_cars_per_train: int,
    target_num_cars_per_train: int
    
) -> tuple:
    """
    Calculate the waiting time for a single dispatch using Polars DataFrames.
    """
    # Initialize variables
    direction_demand = demand_hourly.filter(pl.col("OD_Pair") == od_pair_loop)
    remaining_demand = 0
    total_waiting_time = 0
    total_waiting_time_before_dispatch = 0
    dispatched = 0
    cumulative_demand = cumulative_demand_control
    remaining_demand_list = remaining_demand_list_control.clone() #Copy to avoid modifying original
    remaining_demand_tem = []
    # Calculate cumulative demand up to the dispatch hour
    end_hour = min(dispatch_hour + 1, len(demand_hourly))
    hourly_demand = direction_demand.slice(last_dispatch, end_hour - last_dispatch)
    cumulative_demand += hourly_demand["Number_of_Containers"].sum()
    
    if remaining_demand_list.is_empty():
        hourly_demand = hourly_demand.with_columns(
            ((pl.col("Number_of_Containers") * (dispatch_hour - pl.col("Hour"))).alias("Waiting_Time"))
        )
        total_waiting_time_before_dispatch = hourly_demand["Waiting_Time"].sum()
        total_waiting_time += hourly_demand["Waiting_Time"].sum()

        # If there is remaining demand, calculate waiting time for new and remaining demand
    else:
        hourly_demand = hourly_demand.with_columns(
            ((pl.col("Number_of_Containers") * (dispatch_hour - pl.col("Hour"))).alias("Waiting_Time"))
        )
        total_waiting_time_before_dispatch = hourly_demand["Waiting_Time"].sum()
        total_waiting_time += hourly_demand["Waiting_Time"].sum()

        # Calculate waiting time for each entry in the remaining demand list
        remaining_waiting_times = remaining_demand_list.with_columns(
            (pl.col("Remaining_Demand") * (dispatch_hour - pl.col("Hour"))).alias("Remaining_Waiting_Time")
        )
        total_waiting_time_before_dispatch += remaining_waiting_times["Remaining_Waiting_Time"].sum()
        total_waiting_time += remaining_waiting_times["Remaining_Waiting_Time"].sum()
        

    # Handle remaining demands if cumulative demand exceeds thresholds
    if cumulative_demand >= min_num_cars_per_train:
        if cumulative_demand >= target_num_cars_per_train:
            dispatched = target_num_cars_per_train
            dispatched_split = target_num_cars_per_train
            remaining_demand = cumulative_demand - dispatched_split  # Carry over remaining demand
            # Update remaining demand list if there's no prior remaining demand
            if remaining_demand_list.height == 0:
                #remaining_demand_tem = []
                for row in hourly_demand.iter_rows():
                    # Number_of_Containers is located at the 4th column
                    if row[3] > 0:
                        if dispatched_split >= row[3]:
                            dispatched_split -= row[3]
                        else:
                            remaining_demand_for_hour = row[3] - dispatched_split
                            # Hour is located at the 5th column
                            remaining_demand_tem.append((remaining_demand_for_hour,row[4]))
                            # dispatched_split stop working from this hour to the end of this loop behavior
                            dispatched_split = 0
                # Filter `remaining_demand_tem` to include only positive Remaining_Demand values
                filtered_remaining_demand_tem = [(rd[0], rd[1]) for rd in remaining_demand_tem if rd[0] > 0]

                # Construct the DataFrame directly from the filtered list
                remaining_demand_list = pl.DataFrame({
                    "Remaining_Demand": [rd[0] for rd in filtered_remaining_demand_tem],
                    "Hour": [rd[1] for rd in filtered_remaining_demand_tem]
                })
                #remaining_demand_list = pl.DataFrame({"Remaining_Demand": [rd[0] for rd in remaining_demand_tem],"Hour": [rd[1] for rd in remaining_demand_tem]}).filter(pl.col("Remaining_Demand") > 0)
                cumulative_demand = remaining_demand            
            else:                
                # Prepare the cumulative transformation approach
                for row in remaining_demand_list.iter_rows():
                    if dispatched_split >= row[0]:
                        dispatched_split -= row[0]                       
                        list_tem = list(row)
                        list_tem[0] = 0
                        row = tuple(list_tem)                     
                    else:
                        row[0] -= dispatched_split
                        dispatched_split = 0
                    remaining_demand_list = remaining_demand_list.filter(pl.col("Remaining_Demand") > 0)          
                # If there is still dispatched capacity left, apply it to new demand within the range
                if dispatched_split > 0:
                    for row in hourly_demand.iter_rows():
                        if row[3] > 0:
                            if dispatched_split >= row[3]:
                                dispatched_split -= row[3]
                            else:
                                remaining_demand_for_hour = row[3] - dispatched_split
                                remaining_demand_tem.append((remaining_demand_for_hour,row[4]))
                                dispatched_split = 0
                remaining_demand_list = pl.DataFrame({"Remaining_Demand": [rd[0] for rd in remaining_demand_tem],"Hour": [rd[1] for rd in remaining_demand_tem]}).filter(pl.col("Remaining_Demand") > 0)
                cumulative_demand = remaining_demand
        else:
            dispatched = cumulative_demand
            cumulative_demand = 0  # Reset cumulative demand if all is dispatched

    # Accumulate demand if below minimum threshold
    else:
        cumulative_demand = cumulative_demand

    # Filter demand for future hours in the specified search range
    future_demand = demand_hourly.filter(
        (pl.col("Hour") > dispatch_hour) & (pl.col("Hour") < min(last_dispatch + search_range, len(demand_hourly)))
    )

    # Calculate waiting time for each future hour
    future_demand = future_demand.with_columns(
        ((last_dispatch + search_range - 1 - pl.col("Hour")) * pl.col("Number_of_Containers")).alias("Waiting_Time")
    )
    
    # Sum up all waiting times for future demand
    total_waiting_time += future_demand["Waiting_Time"].sum()

    # Calculate waiting time for remaining demand from previous hours, if any
    if not remaining_demand_list.is_empty():
        # Add waiting time for remaining demand entries
        remaining_waiting = remaining_demand_list.with_columns(
            ((last_dispatch + search_range - 1 - pl.col("Hour")) * pl.col("Remaining_Demand")).alias("Remaining_Waiting_Time")
        )

        # Sum up the waiting times from remaining demand list
        total_waiting_time += remaining_waiting["Remaining_Waiting_Time"].sum()
# Return results
    return total_waiting_time_before_dispatch, total_waiting_time, remaining_demand_list, cumulative_demand, dispatched

def find_minimum_waiting_time(
    num_iterations: int, 
    demand_hourly: pl.DataFrame, 
    border_time_list: list, 
    target_num_cars_per_train: int
) -> pl.DataFrame:
    """
    Find the minimum waiting time for dispatches using Polars DataFrame.
    """
    for i in range(len(border_time_list)):
        od_pair_loop = border_time_list[i][0]
        reverse_pair = "-".join(od_pair_loop.split("-")[::-1])
        directional_total_containers = demand_hourly.filter(pl.col("OD_Pair") == od_pair_loop)["Number_of_Containers"].sum()
        reverse_total_containers = demand_hourly.filter(pl.col("OD_Pair") == reverse_pair)["Number_of_Containers"].sum()
        if directional_total_containers > reverse_total_containers:
            empty_containers = directional_total_containers - reverse_total_containers
            empty_containers_o_d = reverse_pair
        else:
            empty_containers = reverse_total_containers - directional_total_containers
            empty_containers_o_d = od_pair_loop
        print(f"total containers for {od_pair_loop} is {directional_total_containers}")
        print(f"reverse_total_containers for {reverse_pair} is {reverse_total_containers}")
        #print(f"empty_containers for {empty_containers_o_d} is {empty_containers}")
    print(f"empty_containers for {empty_containers_o_d} is {empty_containers}")
    print(f"There are {len(border_time_list[0])-1} trains to dispatch")
    empty_return = []
    final_dispatch_rows = []
    for j in range(len(border_time_list)):
        start_hour = 0
        total_dispatched = 0
        dispatch_time = []
        waiting_time_total = 0
        waiting_time_total_before_dispatch = 0
        cumulative_demand_control = 0
        last_dispatch = 0
        remaining_demand_list_control = pl.DataFrame({
            "Remaining_Demand": pl.Series([], dtype=pl.Int64),
            "Hour": pl.Series([], dtype=pl.Float64),
        })
        dispatched_list = []
        #print(f"border_time_list[j][0] is {border_time_list[j][0]}")
        od_pair_loop = border_time_list[j][0]
        #print(f"od_pair_loop is {od_pair_loop}")
        origin, destination = border_time_list[j][0].split('-')
        #print(f"origin is {origin}")
        #print(f"destination is {destination}")
        #print(demand)
        total_containers = demand_hourly.filter(pl.col("OD_Pair") == od_pair_loop)["Number_of_Containers"].sum()
        for i in range(2, num_iterations):
            if total_containers - total_dispatched == 0:
                dispatched_list.append(0.0)
            search_range = border_time_list[j][i] - start_hour
            # DataFrame to accumulate dispatch hour info
            total_waiting_time_demand_list = pl.DataFrame({
                "Dispatch_Hour": pl.Series([], dtype=pl.Int64),
                "Waiting_Before_Dispatch": pl.Series([], dtype=pl.Float64),
                "Total_Waiting": pl.Series([], dtype=pl.Float64),
                "Remaining_Demand_List": pl.Series([], dtype=pl.Object),
                "Cumulative_Demand": pl.Series([], dtype=pl.Float64),
                "Dispatched": pl.Series([], dtype=pl.Float64)
            }) 
            for dispatch_hour in range(start_hour, start_hour + search_range):
                total_waiting_time_before_dispatch, total_waiting_time, remaining_demand_list, cumulative_demand, dispatched = calculate_waiting_time_single_dispatch(
                    cumulative_demand_control, last_dispatch, demand_hourly, dispatch_hour, remaining_demand_list_control, remaining_demand_list_control.clone(), search_range,od_pair_loop,min_num_cars_per_train, target_num_cars_per_train
                )
                # Append data for each dispatch hour, ensuring consistent types
                new_row = pl.DataFrame({
                    "Dispatch_Hour": [dispatch_hour],
                    "Waiting_Before_Dispatch": [float(total_waiting_time_before_dispatch)],
                    "Total_Waiting": [float(total_waiting_time)],
                    "Remaining_Demand_List": [remaining_demand_list],
                    "Cumulative_Demand": [float(cumulative_demand)],
                    "Dispatched": [float(dispatched)]
                })
                total_waiting_time_demand_list = total_waiting_time_demand_list.vstack(new_row)
                
            # Find the row with the minimum "Total_Waiting"
            min_waiting_row = total_waiting_time_demand_list.sort("Total_Waiting").head(1)
            min_waiting_time_hour = min_waiting_row[0, "Dispatch_Hour"]
            min_waiting_time_before_dispatch = min_waiting_row[0, "Waiting_Before_Dispatch"]
            min_waiting_time = min_waiting_row[0, "Total_Waiting"]
            remaining_demand_list = min_waiting_row[0, "Remaining_Demand_List"]
            cumulative_demand = min_waiting_row[0, "Cumulative_Demand"]
            dispatched = min_waiting_row[0, "Dispatched"]


            # Track dispatched containers for each dispatch hour
            dispatched_list.append(dispatched)
            # Reset remaining demand if cumulative demand is zero
            if cumulative_demand == 0:
                remaining_demand_list = pl.DataFrame(schema=["Remaining_Demand_List"])

            # Update control values for the next iteration
            remaining_demand_list_control = remaining_demand_list
            cumulative_demand_control = cumulative_demand
            last_dispatch = min_waiting_time_hour + 1
            start_hour = min_waiting_time_hour + 1
            total_dispatched += dispatched
            waiting_time_total += min_waiting_time

            # Accumulate total waiting time for before dispatch and overall
            if i == num_iterations - 1:
                waiting_time_total_before_dispatch += min_waiting_time
            else:
                waiting_time_total_before_dispatch += min_waiting_time_before_dispatch

            # Add the dispatch hour to the list
            dispatch_time.append(min_waiting_time_hour) 
                    
        remaining_to_dispatch = total_containers - total_dispatched
        final_waiting_time = remaining_to_dispatch * (168 - start_hour)
        waiting_time_total += final_waiting_time
        dispatch_time.append(167)  # Assuming final dispatch at the end of the period
        dispatched_list.append(remaining_to_dispatch)  
        dispatch_df_row = []
        print(dispatched_list)
        for i in range(len(dispatched_list)):
            dispatch_df_row.append({
                "Origin": origin,
                "Destination": destination,
                "Train_Type": "Intermodal",
                "Cars_Per_Train_Loaded": dispatched_list[i],
                "Cars_Per_Train_Empty": 0.0,
                "Target_Cars_Per_Train": float(target_num_cars_per_train),
                "Number_of_Cars_Total": dispatched_list[i],
                "Tons_Per_Train_Total": dispatched_list[i]*98.006382,
                "HP_Required_Per_Ton": 4.0,
                "Hour":float(dispatch_time[i])
            })
        if od_pair_loop == empty_containers_o_d:
            dispatch_df_row = []
            for i in range(len(dispatched_list)):
                if empty_containers > target_num_cars_per_train-dispatched_list[i]:
                    empty_return.append(target_num_cars_per_train-dispatched_list[i])
                    empty_containers -= (target_num_cars_per_train-dispatched_list[i])
                else:
                    empty_return.append(empty_containers)
                    empty_containers = 0.0
            for i in range(len(dispatched_list)):
                dispatch_df_row.append({
                    "Origin": origin,
                    "Destination": destination,
                    "Train_Type": "Intermodal",
                    "Cars_Per_Train_Loaded": dispatched_list[i],
                    "Cars_Per_Train_Empty": empty_return[i],
                    "Target_Cars_Per_Train": float(target_num_cars_per_train),
                    "Number_of_Cars_Total": dispatched_list[i]+empty_return[i],
                    "Tons_Per_Train_Total": dispatched_list[i]*98.006382 + empty_return[i]*39.106152 ,
                    "HP_Required_Per_Ton": 4.0,
                    "Hour":float(dispatch_time[i])
                }) 
        #print(f"dispatch_df_row is {dispatch_df_row}")
        final_dispatch_rows.extend(dispatch_df_row) 
    #print(f"final_dispatch_rows is {final_dispatch_rows}")
    dispatch_times = pl.DataFrame(final_dispatch_rows)
    dispatch_times = dispatch_times.sort("Hour")
    print(dispatch_times)
    print(f"empty_return is {empty_return}")
    return dispatch_times

def formatScheduleColumns(
    df: Union[pl.DataFrame, pl.LazyFrame],
    config: planner_config.TrainPlannerConfig
) -> Union[pl.DataFrame, pl.LazyFrame]: 
    containers_per_car = 0.0
    if config.stack_type == "single":
        containers_per_car = 1.0
    elif config.stack_type == "double":
        containers_per_car = 2.0
    else:
        assert(f'Unhandled container stack type: {config.stack_type}')
    return (df
        .with_columns(
            (pl.col("Tons_Per_Car_Loaded").mul("Number_of_Cars_Loaded") + pl.col("Tons_Per_Car_Empty").mul("Number_of_Cars_Empty")).alias("Tons_Per_Train"),
            (pl.col("HP_Required_Per_Ton_Loaded").mul("Tons_Per_Car_Loaded").mul("Number_of_Cars_Loaded") + 
                pl.col("HP_Required_Per_Ton_Empty").mul("Tons_Per_Car_Empty").mul("Number_of_Cars_Empty")
                ).alias("HP_Required"),
            pl.when(pl.col("Train_Type").str.contains("Intermodal"))
                .then(pl.col("Number_of_Cars_Loaded").mul(containers_per_car))
                .otherwise(0)
                .alias("Containers_Loaded"),
            pl.when(pl.col("Train_Type").str.contains("Intermodal"))
                .then(pl.col("Number_of_Cars_Empty").mul(containers_per_car))
                .otherwise(0)
                .alias("Containers_Empty"),
        )
        .select("Hour", "Origin", "Destination", "Train_Type", 
                "Number_of_Cars", "Number_of_Cars_Loaded", "Number_of_Cars_Empty", "Containers_Empty", "Containers_Loaded",
                "Tons_Per_Train", "HP_Required"
        )
        .rename({"Number_of_Cars_Loaded": "Cars_Loaded", 
                "Number_of_Cars_Empty": "Cars_Empty"})
        .sort(["Hour","Origin","Destination","Train_Type"])
    )


def calculate_dispatch_data(total_cars, target_num_cars, label, cars, max_min_cars):
    remaining_cars = total_cars % target_num_cars
    min_num_cars = (
        total_cars // target_num_cars + (1 if remaining_cars > 0 else 0)
    )
    
    ## Apply the maximum min_num_cars value
    min_num_cars = int(max(min_num_cars, max_min_cars))

    border_list = []
    border_time_list = []
    dispatched_list = []

    if remaining_cars == 0:
        border = total_cars / min_num_cars
    else:
        border = math.ceil(total_cars / min_num_cars)
        
    containers_left = total_cars
    for _ in range(min_num_cars):
        if containers_left >= border:
            border_list.append(border)
            containers_left -= border
        else:
            border_list.append(containers_left)
            break

    cumulate_demand = 0
    p = 0
    for i, car in enumerate(cars):
        cumulate_demand += car
        if p == len(border_list) - 1:
            border_time_list.append(len(cars) - 1)
            break
        if cumulate_demand >= border_list[p]:
            border_time_list.append(i)
            dispatched_list.append(cumulate_demand)
            total_cars -= cumulate_demand
            cumulate_demand = 0
            p += 1
            if p >= len(border_list):
                break
    if len(border_time_list) < len(border_list):
        for _ in range(len(border_list) - len(border_time_list)):
            border_time_list.append(border_time_list[-1]+_+1)
    dispatched_list.append(total_cars)
    if len(dispatched_list) < len(border_list):
        for _ in range(len(border_list) - len(dispatched_list)):
            dispatched_list.append(0)
    return {
        "Border": border_list,
        "Border_Times": border_time_list,
        "Dispatched": dispatched_list,
    }


def calculate_dispatches_deterministic_hourly(
    demand: pl.DataFrame,
    rail_vehicles: List[alt.RailVehicle],
    simulation_days: int,
    config: planner_config.TrainPlannerConfig
) -> pl.DataFrame:
    """
    Converts a table of demand into a dispatch plan where trains depart from each origin in uniformly spaced intervals.
    Arguments:
    ----------
    demand: `DataFrame` or `LazyFrame` representing origin-destination demands (number of trains).
    rail_vehicles: List of `altrios.RailVehicle` objects.
    simulation_days: Number of days of simulation to run.
    config: `TrainPlannerConfig` object.
    Outputs:
    ----------
    Updated demand `DataFrame` or `LazyFrame` representing dispatches, each defined with an origin, destination, train type, number of (loaded and empty) cars, tonnage, and HP per ton requirement.
    """
    # Define the main function to generate demand trains with the updated rule
    def generate_trains(
        demand_hourly: pl.DataFrame,
        target_num_cars_per_train:int
    ) -> pl.DataFrame:
        demand_hourly = demand_hourly.with_columns((pl.col("Origin") + "-" + pl.col("Destination")).alias("OD_Pair"))
        grouped_data = (demand_hourly
            .group_by("OD_Pair")
                .agg(
                    pl.col("Number_of_Cars").sum().alias("Total_Cars"),
                    pl.col("Number_of_Cars").sum().mod(target_num_cars_per_train).alias("Remaining_Cars")
                )
            .with_columns(
                pl.col("Total_Cars").truediv(target_num_cars_per_train).round().add(pl.col("Remaining_Cars").gt(0)).alias("Max_Min_Cars")
            )
        )

        # Prepare a list to collect the results for all OD pairs
        all_dispatch_data = []
        # Step 4: Loop through each unique OD pair to calculate dispatch data
        for row in grouped_data.iter_rows(named=True):
            od_pair = row['OD_Pair']  # This is the "OD_Pair" string like "Origin-Destination"
            total_cars = row['Total_Cars']  # This is the "Total_Cars" value

            # Filter cars list for the specific OD pair
            cars = demand_hourly.filter(pl.col("OD_Pair") == od_pair).select("Number_of_Cars").to_series().to_list()

            # Retrieve the max min_num_cars for this OD pair
            max_min_cars = row['Max_Min_Cars']

            # Calculate dispatch data for the current OD pair with the updated rule
            dispatch_data = calculate_dispatch_data(total_cars, target_num_cars_per_train, od_pair, cars, max_min_cars)
            
            # Add the OD pair label to each entry in the dispatch data
            dispatch_data_df = pl.DataFrame(dispatch_data, strict=False)
            dispatch_data_df = dispatch_data_df.with_columns(pl.lit(od_pair).alias("OD_Pair"))

            # Ensure the "Border" column exists and is cast to Float64
            if "Border" in dispatch_data_df.columns:
                dispatch_data_df = dispatch_data_df.with_columns(
                    pl.col("Border").cast(pl.Float64)
                )

            # Append the result to the list
            all_dispatch_data.append(dispatch_data_df)

        # Combine all dispatch data into a single DataFrame
        final_dispatch_df = pl.concat(all_dispatch_data, how="diagonal_relaxed")
    
        return final_dispatch_df

    target_num_cars_per_train = config.target_cars_per_train['Default'] #TODO make this flexible
    dispatch_df = generate_trains(demand,target_num_cars_per_train)
    od_border_time = []
    border_time = dispatch_df["Border_Times"].to_list() 
    od_pair_list = dispatch_df["OD_Pair"].to_list()
    for i in range(len(od_pair_list)):
        od_border_list_sub =[]
        od_border_list_sub.append(od_pair_list[i])
        od_border_list_sub.append(border_time[i])
        od_border_time.append(od_border_list_sub)
    grouped_data = defaultdict(list)
    for od_pair, value in od_border_time:
        grouped_data[od_pair].append(value)
    border_time_list= [[key] + values for key, values in grouped_data.items()]
    num_iterations = len(border_time_list[0])
    schedule = find_minimum_waiting_time(num_iterations=num_iterations,
        demand_hourly=demand,
        border_time_list=border_time_list,
        target_num_cars_per_train=target_num_cars_per_train 
    )
    return schedule.pipe(formatScheduleColumns)

def generate_dispatch_details(
    demand: pl.DataFrame,
    rail_vehicles: List[alt.RailVehicle],
    simulation_days: int,
    config: planner_config.TrainPlannerConfig
) -> pl.DataFrame:
    """
    Generate a tabulated demand pair to indicate the expected dispatching interval
    and actual dispatching timesteps after rounding, with departures from each terminal
    spaced as evenly as possible
    Arguments:
    ----------
    demand: `DataFrame` or `LazyFrame` representing origin-destination demands (number of trains).
    rail_vehicles: List of `altrios.RailVehicle` objects.
    simulation_days: Number of days of simulation to run.
    config: `TrainPlannerConfig` object.
    Outputs:
    ----------
    schedule: Tabulated dispatching time for each demand pair for each train type
    in hours
    """
    hours = simulation_days * 24
    grouping_vars = ["Origin", "Destination", "Train_Type"]
    return (demand
        .select(pl.exclude("Number_of_Trains").repeat_by("Number_of_Trains").explode())
        .pipe(utilities.allocateIntegerEvenly, target = "Number_of_Cars_Loaded", grouping_vars = grouping_vars)
        .drop("Percent_Within_Group_Cumulative")
        .pipe(utilities.allocateIntegerEvenly, target = "Number_of_Cars_Empty", grouping_vars = grouping_vars)
        .drop("Percent_Within_Group_Cumulative")
        .group_by(pl.exclude("Number_of_Cars_Empty", "Number_of_Cars_Loaded"))
            .agg(pl.col("Number_of_Cars_Empty", "Number_of_Cars_Loaded"))
        .with_columns(pl.col("Number_of_Cars_Loaded").list.sort(descending=True),
                      pl.col("Number_of_Cars_Empty").list.sort(descending=False))
        .explode("Number_of_Cars_Empty", "Number_of_Cars_Loaded")
        .with_columns((pl.col("Number_of_Cars_Empty") + pl.col("Number_of_Cars_Loaded")).alias("Number_of_Cars"))
        #TODO: space out trains with slightly more vs less demand, instead of ordering randomly
        .with_columns(pl.int_range(pl.len()).shuffle().alias("random_int"))
        .sort("Origin", "Destination", "Train_Type", "random_int")
        .drop("random_int")
        .with_columns(
            (hours * 1.0 / pl.len().over("Origin", "Destination")).alias("Interval")
        )
        .with_columns(
            ((pl.col("Interval").cum_count().over(["Origin","Destination"])) \
             * pl.col("Interval")).alias("Hour")
        )
        .pipe(formatScheduleColumns, config = config)
    )