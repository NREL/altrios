from altrios import sim_manager
import altrios as alt
from altrios.train_planner import planner_config
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
ALTRIOS_demo_result_root = alt.resources_root().parent/"demos/results/case study/Traffic Level Sensitivity Result.csv"
result_output = pd.read_csv(ALTRIOS_demo_result_root)
num_train_per_day = 5
Nodes = ["Allouez","Hibbing"]
Node_1 = Nodes[0]
Node_2 = Nodes[1]
Train_types = ["Ironore"]
Train_types = ["Grain"]
travel_time = 10 #hr
num_car_per_train = 100
percent_of_loaded = 0.5
sets_of_dispatches = 21
loco_per_train = 4
trains_per_day_per_node = (len(Nodes) - 1) * len(Train_types) * num_train_per_day
num_of_trains = len(Nodes) * (len(Nodes) - 1) * len(Train_types) * num_train_per_day * sets_of_dispatches
print("Expected number of trains to be simulated: ", num_of_trains)
loco_consist = "mixed" # Diesel or mixed # Diesel_Large and BEL are the loco names
columns = ["Train_ID", 
        "Train_Type", 
        "Locomotive_ID", 
        "Locomotive_Type", 
        "Origin_ID", 
        "Destination_ID", 
        "Cars_Loaded", 
        "Cars_Empty", 
        "Containers_Loaded", 
        "Containers_Empty", 
        "Departure_SOC_J", 
        "Departure_Time_Planned_Hr", 
        "Arrival_Time_Planned_Hr", 
        "Refuel_Start_Time_Planned_Hr", 
        "Refuel_End_Time_Planned_Hr", 
        "Departure_Time_Actual_Hr", 
        "Arrival_Time_Actual_Hr", 
        "TrainSimVec_Index"]
train_plan = pd.DataFrame(columns=columns)
train_id = 0
for iteration in range(sets_of_dispatches):
    print("Iteration: ", iteration)
    for train_cycle in range(num_train_per_day):
        for origin in Nodes:
            destination_nodes = [node for node in Nodes if node != origin]
            for destination in destination_nodes:
                for train_type in Train_types:
                    train_id += 1
                    for loco_num in range(loco_per_train):
                        row_number = len(train_plan)
                        loco_id = int(train_id * 10 +loco_num)
                        if (loco_num == loco_per_train - 1) and (loco_consist == "mixed"):
                            loco_type = "BEL"
                        else:
                            loco_type = "Diesel_Large"
                        cars_loaded = np.floor(num_car_per_train * percent_of_loaded) if train_type == "Manifest" else num_car_per_train
                        cars_empty = num_car_per_train - cars_loaded if train_type == "Manifest" else 0
                        containers_loaded = cars_loaded if train_type == "Intermodal" else 0
                        containers_empty = cars_empty if train_type == "Intermodal" else 0
                        departure_soc_j = 8.28691E+11
                        departure_time_planned_hr = (train_id * (24 / trains_per_day_per_node / len(Nodes)) + round(np.random.uniform(-1, 1),1) if train_id != 0 else 0) if loco_num == 0 else train_plan.loc[row_number - 1, "Departure_Time_Planned_Hr"]
                        arrival_time_planned_hr = departure_time_planned_hr + travel_time + round(np.random.uniform(-1, 1),1) if loco_num == 0 else train_plan.loc[row_number - 1, "Arrival_Time_Planned_Hr"]
                        refuel_start_time_planned_hr = arrival_time_planned_hr + np.random.uniform(0, 1) if loco_num == 0 else train_plan.loc[row_number - 1, "Refuel_Start_Time_Planned_Hr"]
                        refuel_end_time_planned_hr = refuel_start_time_planned_hr + np.random.uniform(0, 1) if loco_num == 0 else train_plan.loc[row_number - 1, "Refuel_End_Time_Planned_Hr"]
                        departure_time_actual_hr = departure_time_planned_hr + np.random.uniform(-0.5, 0.5) if loco_num == 0 else train_plan.loc[row_number - 1, "Departure_Time_Actual_Hr"]
                        arrival_time_actual_hr = arrival_time_planned_hr + np.random.uniform(-0.5, 0.5) if loco_num == 0 else train_plan.loc[row_number - 1, "Arrival_Time_Actual_Hr"]
                        train_plan.loc[row_number, :] = [train_id, train_type, loco_id, loco_type, origin, destination, cars_loaded, cars_empty, containers_loaded, containers_empty, departure_soc_j, departure_time_planned_hr, arrival_time_planned_hr, refuel_start_time_planned_hr, refuel_end_time_planned_hr, departure_time_actual_hr, arrival_time_actual_hr, train_id]
                    # print(departure_time_planned_hr)
print(train_plan)
cutoff_point_0 = 24 * 7
cutoff_point_1 = 24 * 14
train_plan = train_plan.loc[train_plan["Departure_Time_Planned_Hr"] >= cutoff_point_0]
train_plan = train_plan.loc[train_plan["Arrival_Time_Planned_Hr"] < cutoff_point_1]
print(train_plan)
train_plan.to_csv(alt.resources_root() / "demo_data/manual_train_planner_demo/demo_train_consist_plan_test.csv", index=False)

t0_total = time.perf_counter()

SHOW_PLOTS = alt.utils.show_plots()

plot_dir = Path() / "plots"
# make the dir if it doesn't exist
plot_dir.mkdir(exist_ok=True)

# %%
# Use the same public network, locations, and rolling stock as `sim_manager_demo.py`
# so this demo does not depend on any proprietary datasets.
print("Loading `rail_vehicles`")
rail_vehicles = [
    alt.RailVehicle.from_file(vehicle_file, skip_init=False)
    for vehicle_file in Path(alt.resources_root() / "rolling_stock/").glob("*.yaml")
]

print("Importing `location_map`")
location_map = alt.import_locations(
    alt.resources_root() / "networks/default_locations.csv"
)

print("Loading `network`")
network = alt.Network.from_file(
    alt.resources_root() / "networks/Taconite-NoBalloon.yaml"
)

print("Loading `train_planner_config`")
train_planner_config = planner_config.TrainPlannerConfig()

# %%
# `manual_train_planner_demo` demonstrates replaying an already-planned train
# consist plan via `sim_manager.main(consist_plan_in=...)`. The example consist
# plan below was generated with the standard planner on the public Taconite
# network (the same inputs as `sim_manager_demo.py`) and is committed under
# `resources/demo_data/` so this demo stays self-contained and free of any
# proprietary data. In a production workflow `consist_plan_in` would instead come
# from your own pre-planned source.
print("Loading example `consist_plan_in`")
consist_plan_in = pl.read_csv(
    alt.resources_root() / "demo_data/manual_train_planner_demo/demo_train_consist_plan_test.csv"
)

# %%
print("Running `sim_manager.main` in manual mode (replaying `consist_plan_in`)")
t0_main = time.perf_counter()
# Because `consist_plan_in` is provided, the planner skips demand generation and
# scheduling and instead replays the pre-planned trains.
(
    consist_plan_out,
    loco_pool,
    refuel_facilities,
    grid_emissions_factors,
    nodal_energy_prices,
    speed_limit_train_sims,
    timed_paths,
    consist_plan_out_untrimmed,
) = sim_manager.main(
    network=network,
    rail_vehicles=rail_vehicles,
    location_map=location_map,
    consist_plan_in=consist_plan_in,
    train_planner_config=train_planner_config,
    debug=True,
)


# %%
t1_main = time.perf_counter()
print(f"Elapsed time to run `sim_manager.main()`: {t1_main - t0_main:.3g} s")

# %%
t0_train_sims = time.perf_counter()
speed_limit_train_sims.set_save_interval(100)
(sims, refuel_sessions) = alt.run_speed_limit_train_sims(
    speed_limit_train_sims=speed_limit_train_sims,
    network=network,
    train_consist_plan_py=consist_plan_out,
    loco_pool_py=loco_pool,
    refuel_facilities_py=refuel_facilities,
    timed_paths=[alt.TimedLinkPath.from_pydict(tp) for tp in timed_paths],
)
t1_train_sims = time.perf_counter()
print(f"Elapsed time to run train sims: {t1_train_sims - t0_train_sims:.3g} s")
t_train_time = sum([sim["state"]["time_seconds"] for sim in sims.to_pydict()])
t_train_time = sum([sim["history"]["time_seconds"][-1] - sim["history"]["time_seconds"][0]for sim in sims.to_pydict()])
print(f"Total number of train sims simulated: {len(sims.to_pydict())}")
print(f"Total train-seconds simulated: {t_train_time} s")
print(f"Total train-hours simulated: {t_train_time / 3600} hr")
print(f"Average train-second per train sim: {t_train_time / len(sims.to_pydict())} s")
print(f"Average train-hour per train sim: {t_train_time / len(sims.to_pydict()) / 3600} hr")
row_number = len(result_output)
result_output.loc[row_number, "Node 1"] = Node_1
result_output.loc[row_number, "Node 2"] = Node_2
result_output.loc[row_number, "num_train_per_day"] = num_train_per_day
result_output.loc[row_number, "Total Train Time (s)"] = t_train_time
result_output.loc[row_number, "Total Train Time (hr)"] = t_train_time / 3600
result_output.loc[row_number, "Average Train Time (s)"] = t_train_time / len(sims.to_pydict())
result_output.loc[row_number, "Average Train Time (hr)"] = t_train_time / len(sims.to_pydict())/ 3600
result_output.to_csv(ALTRIOS_demo_result_root, index=False)

# %%
t0_summary_sims = time.perf_counter()
speed_limit_train_sims.set_save_interval(None)
(summary_sims, summary_refuel_sessions) = alt.run_speed_limit_train_sims(
    speed_limit_train_sims=speed_limit_train_sims,
    network=network,
    train_consist_plan_py=consist_plan_out,
    loco_pool_py=loco_pool,
    refuel_facilities_py=refuel_facilities,
    timed_paths=[alt.TimedLinkPath.from_pydict(tp) for tp in timed_paths],
)
t1_summary_sims = time.perf_counter()
print(
    f"Elapsed time to build and run summary sims: {t1_summary_sims - t0_summary_sims:.3g} s"
)

# %%
t0_tolist = time.perf_counter()
sims_list = sims.to_pydict()
t1_tolist = time.perf_counter()
print(f"Elapsed time to run `to_pydict()`: {t1_tolist - t0_tolist:.3g} s")

if not sims_list:
    raise RuntimeError("No train simulations were returned; cannot build study-period results.")

sim0 = sims_list[0]

# Export one representative trip's speed profile in both tabular and plotted form.
# Times are relative to the beginning of this trip so the profile is easy to reuse.
speed_profile = pd.DataFrame(
    {
        "Time_Hr": (
            np.asarray(sim0["history"]["time_seconds"])
            - sim0["history"]["time_seconds"][0]
        )
        / 3_600,
        "Speed_MPH": np.asarray(sim0["history"]["speed_meters_per_second"])
        / 0.44704,
        "Speed_Limit_MPH": np.asarray(
            sim0["history"]["speed_limit_meters_per_second"]
        )
        / 0.44704,
    }
)
speed_profile_path = plot_dir / f"train_{sim0['train_id']}_speed_profile.csv"
speed_profile.to_csv(speed_profile_path, index=False)

fig, ax = plt.subplots()
ax.plot(speed_profile["Time_Hr"], speed_profile["Speed_MPH"], label="Actual")
ax.plot(
    speed_profile["Time_Hr"],
    speed_profile["Speed_Limit_MPH"],
    label="Speed limit",
)
ax.set(
    xlabel="Trip time [hr]",
    ylabel="Speed [mph]",
    title=f"Train {sim0['train_id']} speed profile",
)
ax.legend()
fig.tight_layout()
speed_profile_plot_path = plot_dir / f"train_{sim0['train_id']}_speed_profile.png"
fig.savefig(speed_profile_plot_path, dpi=200)
plt.close(fig)
print(f"Speed profile data written to: {speed_profile_path.resolve()}")
print(f"Speed profile plot written to: {speed_profile_plot_path.resolve()}")


# %%

t0_main = time.perf_counter()
e_total_fuel_joules = summary_sims.get_energy_fuel_joules(annualize=False)
t1_main = time.perf_counter()

print(f"Elapsed time to get total fuel energy: {t1_main - t0_main:.3g} s")
e_total_fuel_mwh = e_total_fuel_joules / 3.6e9
print(f"Total diesel energy consumed: {e_total_fuel_mwh:.3g} MWh")

# Net onboard battery energy consumed over all train trips.  A positive value
# means the batteries ended the trips with less energy than they started with;
# regenerative charging can make an individual trip (or the total) negative.
e_total_battery_joules = 0.0
for sim in sims_list:
    for loco in sim["loco_con"]["loco_vec"]:
        loco_type = next(iter(loco["loco_type"].values()))
        res = loco_type.get("res")
        if res is None:
            continue
        soc_history = res["history"]["soc"]
        if soc_history:
            e_total_battery_joules += (
                soc_history[0] - soc_history[-1]
            ) * res["energy_capacity_joules"]

e_total_battery_mwh = e_total_battery_joules / 3.6e9
print(f"Total net battery electricity consumed: {e_total_battery_mwh:.3g} MWh")

result_output.loc[row_number, "Total Diesel Energy Consumption (MWh)"] = (
    e_total_fuel_mwh
)
result_output.loc[row_number, "Total Electricity Consumption (MWh)"] = (
    e_total_battery_mwh
)
result_output.loc[row_number, "Consist Configuration"] = loco_consist
result_output.loc[row_number, "Diesel_Large Locomotives per Consist"] = (
    loco_per_train - 1 if loco_consist == "mixed" else loco_per_train
)
result_output.loc[row_number, "BEL Locomotives per Consist"] = (
    1 if loco_consist == "mixed" else 0
)
result_output.loc[row_number, "Speed Profile Train ID"] = sim0["train_id"]
result_output.loc[row_number, "Speed Profile CSV"] = str(speed_profile_path.resolve())
result_output.loc[row_number, "Speed Profile Plot"] = str(
    speed_profile_plot_path.resolve()
)
print(f"Total elapsed time: {time.perf_counter() - t0_total} s")
result_output.to_csv(ALTRIOS_demo_result_root, index=False)
