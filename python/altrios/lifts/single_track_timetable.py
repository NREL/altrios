import pandas as pd
import random
import math

# User-input
daily_throughput = 2000
train_batch_size = 300
simulation_duration = 24
start_time = 10


layout_file = "C:/Users/mbruchon/Documents/Repos/NREL/altrios/python/altrios/lifts/single_track_input/layout.xlsx"
df_layout = pd.read_excel(layout_file)

if train_batch_size not in df_layout["train batch (k)"].values:
    raise ValueError("train_batch_size doesn't exist on layout.xlsx.")

layout_params = df_layout[df_layout["train batch (k)"] == train_batch_size].iloc[0]
M, N, n_t, n_p, n_r = layout_params[["rows (M)", "cols (N)", "trainlanes (n_t)", "parknglanes (n_p)", "blocklen (n_r)"]]

num_trains = math.ceil(daily_throughput / train_batch_size)

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
    full_cars = full_cars_list[i]
    empty_cars = random.randint(0, 10)
    oc_number = full_cars + empty_cars
    truck_number = max(full_cars, oc_number)

    train_timetable.append({
        "train_id": train_id,
        "arrival_time": round(arrival_time, 2),
        "departure_time": round(departure_time, 2),
        "empty_cars": empty_cars,
        "full_cars": full_cars,
        "oc_number": oc_number,
        "truck_number": truck_number
    })

print(train_timetable)