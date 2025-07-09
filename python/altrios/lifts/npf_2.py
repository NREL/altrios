import math
import random
import pandas as pd
import subprocess
from altrios.lifts.distances_single_track import *
from altrios.lifts.single_track_parameters import *


simulation_days = 100    # simulation
total_simulation_length = 24 * simulation_days  # including warm-up and cool-down
replicate_times = 3     # timetable generation
simulation_duration = replicate_times * 24  # train generation

daily_throughput_range = range(200, 2001, 50)
train_batch_size_range = [200]  # range(200, 201, 50)
layout_file = "single_track_input/layout.xlsx"

df_layout = pd.read_excel(layout_file)
performance_matrix = []


def generate_train_timetable(daily_throughput, train_batch_size, replicate_times):
    train_timetable = []
    train_id_counter = 1

    trains_per_day = math.ceil((daily_throughput / 2) / train_batch_size)
    simulation_days = replicate_times

    for day in range(simulation_days):
        base_time = day * 24 + 12
        arrival_times = []

        for _ in range(trains_per_day):
            randomness = 12 * random.uniform(-1, 1)
            arrival_time = round(base_time + randomness, 2)
            arrival_times.append(arrival_time)

        arrival_times.sort()

        for i in range(trains_per_day):
            arrival = arrival_times[i]
            if i < trains_per_day - 1:
                departure = arrival_times[i + 1]
            else:
                departure = (day + 1) * 24

            train = {
                "train_id": train_id_counter,
                "arrival_time": round(arrival, 2),
                "departure_time": round(departure, 2),
                "empty_cars": 0,
                "full_cars": train_batch_size,
                "oc_number": train_batch_size,
                "truck_number": train_batch_size
            }

            train_timetable.append(train)
            train_id_counter += 1

    return train_timetable


for daily_throughput in daily_throughput_range:
    for train_batch_size in train_batch_size_range:

        train_timetable = generate_train_timetable(daily_throughput, train_batch_size, replicate_times)

        with open("train_timetable.json", "w") as f:
            json.dump(train_timetable, f)

        layout_params = df_layout[df_layout["train batch (k)"] == train_batch_size * 2].iloc[0]
        M, N, n_t, n_p, n_r = layout_params[
            ["rows (M)", "cols (N)", "trainlanes (n_t)", "parknglanes (n_p)", "blocklen (n_r)"]]

        d_h = uniform_mean(d_h_min, d_h_max)
        d_r = uniform_mean(d_r_min, d_r_max)

        hostler_cycle_time = state.CRANE_NUMBER * ((2 * uniform_mean(d_h_min, d_h_max) + uniform_mean(d_r_min, d_r_max)) / 3.2) / (2.9 * 3600)  # (ft -> m) / (m/s -> m/hr)
        crane_loading_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN
        hostler_num = math.ceil((hostler_cycle_time + crane_loading_time) / (state.CONTAINERS_PER_CRANE_MOVE_MEAN))

        config = {
            "layout": {
                "K": daily_throughput,
                "k": train_batch_size,
                "M": int(M),
                "N": int(N),
                "n_t": int(n_t),
                "n_p": int(n_p),
                "n_r": int(n_r)
            },
            "vehicles": {
                "simulation_duration": total_simulation_length, # extend length of simulation (including warm-up & cool-down)
                "CRANE_NUMBER": 1,
                "HOSTLER_NUMBER": 1
            }
        }

        with open("sim_config.json", "w") as f:
            json.dump(config, f)

        # single_track_simulation.py
        # pass new parameters
        state.CRANE_NUMBER = config["vehicles"]["CRANE_NUMBER"]
        state.HOSTLER_NUMBER = config["vehicles"]["HOSTLER_NUMBER"]

        print("-" * 100)
        print(f"current throughput: {daily_throughput}, batch size: {train_batch_size}, simulation duration: {total_simulation_length}")
        subprocess.run(["python", "single_track_simulation.py"], check=True)