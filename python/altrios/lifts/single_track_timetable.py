import pandas as pd
import random
import math
from altrios.lifts.distances_single_track import *
from altrios.lifts.single_track_parameters import *

# User-input
replicate_times = 1
daily_throughput = 1000 * replicate_times
train_batch_size = 300
simulation_duration = 24 * replicate_times
start_time = 0

# layout_file = "C:/Users/mbruchon/Documents/Repos/NREL/altrios/python/altrios/lifts/single_track_input/layout.xlsx"
layout_file = "/Users/qianqiantong/PycharmProjects/altrios-private/altrios/python/altrios/lifts/single_track_input/layout.xlsx"
df_layout = pd.read_excel(layout_file)

if train_batch_size not in df_layout["train batch (k)"].values:
    raise ValueError("train_batch_size doesn't exist on layout.xlsx.")

layout_params = df_layout[df_layout["train batch (k)"] == train_batch_size].iloc[0]
M, N, n_t, n_p, n_r = layout_params[["rows (M)", "cols (N)", "trainlanes (n_t)", "parknglanes (n_p)", "blocklen (n_r)"]]

# recourse calculations
print(f"d_h: {uniform_mean(d_h_min, d_h_max)} ft")
print(f"d_r: {uniform_mean(d_r_min, d_r_max)} ft")
print(f"crane number: {state.CRANE_NUMBER}")
print(f"average travel distance for hostler: {(2 * uniform_mean(d_h_min, d_h_max) + uniform_mean(d_r_min, d_r_max)) / 3.2} meters.")
print(f"average hostler speed: {2.9 * 3600} m/hr.")
hostler_cycle_time = state.CRANE_NUMBER * ((2 * uniform_mean(d_h_min, d_h_max) + uniform_mean(d_r_min, d_r_max)) / 3.2 ) / (2.9 * 3600)   # (ft -> m) / (m/s -> m/hr)
print(f"hostler moving cycle time: {hostler_cycle_time} hr")
crane_loading_time = 2/60 # hr
hostler_num = (hostler_cycle_time + crane_loading_time) / (state.CONTAINERS_PER_CRANE_MOVE_MEAN)   # hr/hr
print(f"numbers of hostler: {math.ceil(hostler_num)}")

# train numbers
num_trains = math.ceil(daily_throughput / (replicate_times * train_batch_size)) * replicate_times
print(f"number of trains: {num_trains}")
full_cars_list = [train_batch_size] * (num_trains - 1)
full_cars_list.append(daily_throughput - sum(full_cars_list))
train_ids = random.sample(range(1, 1000), num_trains)

arrival_times = [start_time + i * (simulation_duration - start_time) / num_trains for i in range(num_trains)]

train_timetable = []
for i in range(num_trains):
    train_id = train_ids[i]
    arrival_time = arrival_times[i]
    if i < num_trains - 1:
        departure_time = arrival_times[i + 1]
    else:
        departure_time = min(arrival_time, simulation_duration)
    full_cars = full_cars_list[i] / 2
    oc_number = full_cars
    truck_number = max(full_cars, oc_number)

    train_timetable.append({
        "train_id": train_id,
        "arrival_time": int(round(arrival_time, 2)),
        "departure_time": int(round(departure_time, 2)),
        "empty_cars": 0,
        "full_cars": int(full_cars),
        "oc_number": int(oc_number),
        "truck_number": int(truck_number)
    })

print(train_timetable)