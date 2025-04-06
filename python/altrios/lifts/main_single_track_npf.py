import pandas as pd
import random
import math
import subprocess
import json
from altrios.lifts.distances_single_track import *
from altrios.lifts.single_track_parameters import *
from demo import save_vehicle_and_performance_metrics

replicate_times = 2
daily_throughput_range = range(200, 251, 10)
train_batch_size_range = range(150, 201, 10)
simulation_duration = 24 * replicate_times
layout_file = "single_track_input/layout.xlsx"

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

        crane_number = 2
        hostler_speed = 2.9 * 3600  # m/hr
        crane_rate = 1/30 # hr
        hostler_cycle_time = crane_number * ((2 * d_h + d_r) / 3.2) / hostler_speed
        hostler_num = math.ceil(hostler_cycle_time / crane_rate)

        config = {
            "layout": {
                "k": daily_throughput,
                "M": int(M),
                "N": int(N),
                "n_t": int(n_t),
                "n_p": int(n_p),
                "n_r": int(n_r)
            },
            "vehicles": {
                "CRANE_NUMBER": crane_number,
                "HOSTLER_NUMBER": hostler_num
            }
        }

        with open("sim_config.json", "w") as f:
            json.dump(config, f)

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


        # demo.py
        subprocess.run(["python", "single_track_simulation.py"], check=True)

        single_run = save_vehicle_and_performance_metrics(state)

        if single_run is None:
            print("[Warning] Performance extraction failed, skipping this run.")
            continue

        ic_time, oc_time, total_time, ic_energy, oc_energy, total_energy = single_run

        current_summary = {
            "Layout parameters": [int(M), int(N), int(n_t), int(n_p), int(n_r)],
            "Vehicle parameters": {"cranes": crane_number, "hostlers": hostler_num},
            "Processing time (avg)": {
                "IC": round(ic_time, 2),
                "OC": round(oc_time, 2),
                "Total": round(total_time, 2)
            },
            "Energy consumption (avg)": {
                "IC": round(ic_energy, 2),
                "OC": round(oc_energy, 2),
                "Total": round(total_energy, 2)
            }
        }
        performance_matrix.append([daily_throughput, train_batch_size, current_summary])



data = []
for entry in performance_matrix:
    daily_throughput, train_batch_size, summary = entry

    M, N, n_t, n_p, n_r = summary["Layout parameters"]
    cranes = summary["Vehicle parameters"]["cranes"]
    hostlers = summary["Vehicle parameters"]["hostlers"]

    ic_proc = summary["Processing time (avg)"]["IC"]
    oc_proc = summary["Processing time (avg)"]["OC"]
    total_proc = summary["Processing time (avg)"]["Total"]

    ic_energy = summary["Energy consumption (avg)"]["IC"]
    oc_energy = summary["Energy consumption (avg)"]["OC"]
    total_energy = summary["Energy consumption (avg)"]["Total"]

    data.append([
        daily_throughput, train_batch_size,
        M, N, n_t, n_p, n_r,
        cranes, hostlers,
        ic_proc, oc_proc, total_proc,
        ic_energy, oc_energy, total_energy
    ])


columns = [
    "daily_throughput(K)", "train_batch_size(k)",
    "M", "N", "n_t", "n_p", "n_r",
    "crane_numbers", "hostler_numbers",
    "ic_avg_processing_time", "oc_avg_processing_time", "total_avg_processing_time",
    "ic_avg_energy", "oc_avg_energy", "total_avg_energy"
]

data = pd.DataFrame(data, columns=columns)
data.to_excel("npf_performance_results.xlsx", index=False)
print("Done!")
