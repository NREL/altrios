import math
import random
import pandas as pd
import subprocess
import json
from altrios.lifts.distances_single_track import *
from altrios.lifts.single_track_parameters import *

simulation_days = 20
total_simulation_length = 24 * simulation_days
replicate_times = 3
simulation_duration = replicate_times * 24

train_batch_sizes = list(range(100, 101, 100))
num_trains_per_day = [1]
containers_per_train = 1
layout_file = "single_track_input/layout.xlsx"

df_layout = pd.read_excel(layout_file)
performance_matrix = []

num_cranes_list = [100]
num_hostlers_list = [100]

def generate_train_timetable(train_batch_size, trains_per_day, simulation_days):
    train_timetable = []
    train_id_counter = 1

    for day in range(simulation_days):
        base_time = day * 24 + 12  # Noon
        arrival_times = []

        for _ in range(trains_per_day):
            randomness = 12 * random.uniform(-1, 1)
            arrival_time = round(base_time + randomness, 2)
            arrival_times.append(arrival_time)

        arrival_times.sort()

        for i in range(trains_per_day):
            arrival = arrival_times[i]
            departure = arrival_times[i + 1] if i < trains_per_day - 1 else (day + 1) * 24

            train = {
                "train_id": train_id_counter,
                "arrival_time": arrival,
                "departure_time": departure,
                "empty_cars": 0,
                "full_cars": train_batch_size,
                "oc_number": train_batch_size,
                "truck_number": train_batch_size
            }

            train_timetable.append(train)
            train_id_counter += 1

    return train_timetable

for train_batch_size in train_batch_sizes:
    for trains_per_day in num_trains_per_day:
        for cranes in num_cranes_list:
            for hostlers in num_hostlers_list:

                daily_throughput = 2 * trains_per_day * train_batch_size * containers_per_train

                train_timetable = generate_train_timetable(train_batch_size, trains_per_day, replicate_times)
                with open("train_timetable.json", "w") as f:
                    json.dump(train_timetable, f)

                layout_params = df_layout[df_layout["train batch (k)"] == train_batch_size * 2].iloc[0]
                M, N, n_t, n_p, n_r = layout_params[["rows (M)", "cols (N)", "trainlanes (n_t)", "parknglanes (n_p)", "blocklen (n_r)"]]

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
                        "simulation_duration": total_simulation_length,
                        "CRANE_NUMBER": cranes,
                        "HOSTLER_NUMBER": hostlers
                    }
                }

                with open("sim_config.json", "w") as f:
                    json.dump(config, f)

                # update state
                state.CRANE_NUMBER = cranes
                state.HOSTLER_NUMBER = hostlers

                print("-" * 100)
                print(f"Throughput: {daily_throughput}, batch size: {train_batch_size}, cranes: {cranes}, hostlers: {hostlers}")

                subprocess.run(["python", "single_track_simulation.py"], check=True)

                with open("performance_matrix.json", "r") as f:
                    performance_data = json.load(f)

                ic_avg_time = performance_data["ic_avg_time"]
                ic_avg_delay = performance_data["ic_avg_delay"]
                total_ic_avg_time = performance_data["total_ic_avg_time"]

                oc_avg_time = performance_data["oc_avg_time"]
                oc_avg_delay = performance_data["oc_avg_delay"]
                total_oc_avg_time = performance_data["total_oc_avg_time"]

                ic_energy = performance_data["ic_energy"]
                oc_energy = performance_data["oc_energy"]
                total_avg_energy = performance_data["total_energy"]

                performance_matrix.append([
                    daily_throughput, train_batch_size, trains_per_day,
                    M, N, n_t, n_p, n_r,
                    cranes, hostlers,
                    ic_avg_delay, oc_avg_delay,
                    ic_avg_time, oc_avg_time,
                    total_ic_avg_time, total_oc_avg_time,
                    ic_energy, oc_energy, total_avg_energy
                ])

columns = [
    "daily_throughput(K)", "train_batch_size(k)", "num_trains",
    "M", "N", "n_t", "n_p", "n_r",
    "crane_numbers", "hostler_numbers",
    "ic_avg_delay", "oc_avg_delay",
    "ic_processing_time", "oc_processing_time",
    "total_ic_time", "total_oc_time",
    "ic_avg_energy", "oc_avg_energy", "total_avg_energy"
]

df = pd.DataFrame(performance_matrix, columns=columns)
df.to_excel("npf_performance_results.xlsx", index=False)
print("Done!")
