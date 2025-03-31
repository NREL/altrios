import pandas as pd
import random
import math
import subprocess
import json
from demo import save_to_excel
from altrios.lifts.distances_single_track import *
from altrios.lifts.single_track_parameters import *

replicate_times = 3
daily_throughput_range = range(200, 2001, 10)
train_batch_size_range = range(150, 301, 10)
simulation_duration = 24 * replicate_times
layout_file = "layout.xlsx"

df_layout = pd.read_excel(layout_file)
performance_matrix = []

for daily_throughput in daily_throughput_range:
    for train_batch_size in train_batch_size_range:
        if train_batch_size not in df_layout["train batch (k)"].values:
            continue

        layout_params = df_layout[df_layout["train batch (k)"] == train_batch_size].iloc[0]
        M, N, n_t, n_p, n_r = layout_params[
            ["rows (M)", "cols (N)", "trainlanes (n_t)", "parknglanes (n_p)", "blocklen (n_r)"]]

        d_h = uniform_mean(d_h_min, d_h_max)
        d_r = uniform_mean(d_r_min, d_r_max)

        crane_number = 4
        hostler_speed = 2.9 * 3600  # m/hr
        crane_rate = state.CONTAINERS_PER_CRANE_MOVE_MEAN
        hostler_cycle_time = crane_number * ((2 * d_h + d_r) / 3.2) / hostler_speed
        hostler_num = math.ceil(hostler_cycle_time / crane_rate)

        num_trains = math.ceil(daily_throughput / (replicate_times * train_batch_size)) * replicate_times
        full_cars_list = [train_batch_size] * (num_trains - 1)
        full_cars_list.append(daily_throughput - sum(full_cars_list))
        train_ids = random.sample(range(1, 1000), num_trains)
        arrival_times = [i * (simulation_duration / num_trains) for i in range(num_trains)]

        train_timetable = [
            {
                "train_id": train_ids[i],
                "arrival_time": int(round(arrival_times[i], 2)),
                "departure_time": int(round(arrival_times[i + 1], 2)) if i < num_trains - 1 else simulation_duration,
                "empty_cars": 0,
                "full_cars": int(full_cars_list[i] / 2),
                "oc_number": int(full_cars_list[i] / 2),
                "truck_number": max(int(full_cars_list[i] / 2), int(full_cars_list[i] / 2))
            }
            for i in range(num_trains)
        ]

        with open("train_timetable.json", "w") as f:
            json.dump(train_timetable, f)

        subprocess.run(["python", "demo.py", str(hostler_num)], check=True)

        performance = save_to_excel()

        current_summary = {
            "Layout parameters": [M, N, n_t, n_p, n_r],
            "Vehicle parameters": {"cranes": crane_number, "hostlers": hostler_num},
            "Processing time": performance["Total processing time"],
            "Energy consumption": performance["Energy consumption vector"]
        }
        performance_matrix.append([daily_throughput, train_batch_size, current_summary])

performance_df = pd.DataFrame(performance_matrix, columns=["K", "k", "Performance Summary"])
performance_df.to_excel("performance_results.xlsx", index=False)