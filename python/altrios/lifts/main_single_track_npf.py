import subprocess
import random
import math
from altrios.lifts.distances_single_track import *
from altrios.lifts.single_track_parameters import *

simulation_days = 7    # simulation
total_simulation_length = 24 * simulation_days  # including warm-up and cool-down

replicate_times = 3     # timetable generation
simulation_duration = replicate_times * 24  # train generation

daily_throughput_range = range(200, 2001, 50)
train_batch_size_range = range(70, 221, 50)
layout_file = "single_track_input/layout.xlsx"

df_layout = pd.read_excel(layout_file)
performance_matrix = []


def generate_train_timetable(daily_throughput, train_batch_size, replicate_times, simulation_duration):
    train_timetable = []
    global_train_id_counter = 1

    time_slot = simulation_duration / replicate_times
    for rep in range(replicate_times):
        num_trains = math.ceil(daily_throughput / (train_batch_size * 2))

        full_cars_list = [train_batch_size] * (num_trains - 1)
        last_train_load = (daily_throughput - (train_batch_size * 2 * (num_trains - 1))) / 2
        full_cars_list.append(last_train_load)

        interval = time_slot / num_trains
        base_times = [rep * time_slot + i * interval for i in range(num_trains)]

        jitter = interval * 0.9
        arrival_times = [
            round(random.uniform(max(rep * time_slot, t - jitter), min((rep + 1) * time_slot, t + jitter)), 2)
            for t in base_times
        ]
        arrival_times.sort()

        for i in range(num_trains):
            arrival = arrival_times[i]
            departure = arrival_times[i + 1] if i < num_trains - 1 else (rep + 1) * time_slot

            train = {
                "train_id": global_train_id_counter,
                "arrival_time": round(arrival, 2),
                "departure_time": round(departure, 2),
                "empty_cars": 0,
                "full_cars": int(full_cars_list[i]),
                "oc_number": int(full_cars_list[i]),
                "truck_number": int(full_cars_list[i])
            }
            train_timetable.append(train)
            global_train_id_counter += 1

    return train_timetable


for daily_throughput in daily_throughput_range:
    for train_batch_size in train_batch_size_range:
        if train_batch_size not in df_layout["train batch (k)"].values:
            continue

        layout_params = df_layout[df_layout["train batch (k)"] == train_batch_size * 2].iloc[0]
        M, N, n_t, n_p, n_r = layout_params[["rows (M)", "cols (N)", "trainlanes (n_t)", "parknglanes (n_p)", "blocklen (n_r)"]]

        d_h = uniform_mean(d_h_min, d_h_max)
        d_r = uniform_mean(d_r_min, d_r_max)

        hostler_cycle_time = state.CRANE_NUMBER * ((2 * uniform_mean(d_h_min, d_h_max) + uniform_mean(d_r_min, d_r_max)) / 3.2 ) / (2.9 * 3600)   # (ft -> m) / (m/s -> m/hr)
        crane_loading_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN
        hostler_num = math.ceil((hostler_cycle_time + crane_loading_time) / (state.CONTAINERS_PER_CRANE_MOVE_MEAN))

        config = {
            "layout": {
                "K": daily_throughput,
                "k":train_batch_size,
                "M": int(M),
                "N": int(N),
                "n_t": int(n_t),
                "n_p": int(n_p),
                "n_r": int(n_r)
            },
            "vehicles": {
                "simulation_duration": total_simulation_length, # extend length of simulation (including warm-up & cool-down)
                "CRANE_NUMBER": state.CRANE_NUMBER,
                "HOSTLER_NUMBER": hostler_num
            }
        }

        with open("sim_config.json", "w") as f:
            json.dump(config, f)

        train_timetable = generate_train_timetable(daily_throughput, train_batch_size, replicate_times, simulation_duration)

        with open("train_timetable.json", "w") as f:
            json.dump(train_timetable, f)

        # single_track_simulation.py
        # pass new parameters
        state.CRANE_NUMBER = config["vehicles"]["CRANE_NUMBER"]
        state.HOSTLER_NUMBER = config["vehicles"]["HOSTLER_NUMBER"]

        print("-" * 100)
        print(f"current throughput: {daily_throughput}, batch size: {train_batch_size}, simulation duration: {total_simulation_length}")
        subprocess.run(["python", "single_track_simulation.py"], check=True)

        with open("performance_matrix.json", "r") as f:
            performance_data = json.load(f)

        num_trains = math.ceil(daily_throughput / (train_batch_size * 2))

        # === Layout & Vehicle Config ===
        daily_throughput = config["layout"]["K"]
        M = config["layout"]["M"]
        N = config["layout"]["N"]
        n_t = config["layout"]["n_t"]
        n_p = config["layout"]["n_p"]
        n_r = config["layout"]["n_r"]

        cranes = config["vehicles"]["CRANE_NUMBER"]
        hostlers = config["vehicles"]["HOSTLER_NUMBER"]

        # === Performance Data Extraction ===
        ic_avg_time = performance_data["ic_avg_time"]
        ic_avg_delay = performance_data["ic_avg_delay"]
        total_ic_avg_time = performance_data["total_ic_avg_time"]

        oc_avg_time = performance_data["oc_avg_time"]
        oc_avg_delay = performance_data["oc_avg_delay"]
        total_oc_avg_time = performance_data["total_oc_avg_time"]

        ic_energy = performance_data["ic_energy"]
        oc_energy = performance_data["oc_energy"]
        total_avg_energy = performance_data["total_energy"]

        # === Summary Output ===
        current_summary = {
            "Layout parameters": [int(M), int(N), int(n_t), int(n_p), int(n_r)],
            "Vehicle parameters": {
                "cranes": cranes,
                "hostlers": hostlers
            },
            "Delay time (avg)": {
                "IC": round(ic_avg_delay, 4),
                "OC": round(oc_avg_delay, 4)
            },
            "Processing time (avg, no delay)": {
                "IC": round(ic_avg_time, 4),
                "OC": round(oc_avg_time, 4)
            },
            "Processing time (avg, with delay)": {
                "IC": round(total_ic_avg_time, 4),
                "OC": round(total_oc_avg_time, 4)
            },
            "Energy consumption (avg)": {
                "IC": round(ic_energy, 4),
                "OC": round(oc_energy, 4),
                "Total": round(total_avg_energy, 4)
            }
        }

        # === Performance Matrix Row ===
        performance_matrix.append([
            daily_throughput, train_batch_size, num_trains,
            M, N, n_t, n_p, n_r,
            cranes, hostlers,
            ic_avg_delay, oc_avg_delay,
            ic_avg_time, oc_avg_time,
            total_ic_avg_time, total_oc_avg_time,
            ic_energy, oc_energy, total_avg_energy
        ])

        # === Column Headers ===
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