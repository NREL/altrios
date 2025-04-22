import subprocess
from altrios.lifts.distances_single_track import *
from altrios.lifts.single_track_parameters import *
from single_track_simulation import save_vehicle_and_performance_metrics

replicate_times = 1
daily_throughput_range = range(100, 2001, 20)
train_batch_size_range = range(20, 201, 20)
simulation_duration = 24 * replicate_times
layout_file = "single_track_input/layout.xlsx"

df_layout = pd.read_excel(layout_file)
performance_matrix = []


def generate_train_timetable(daily_throughput, train_batch_size, replicate_times, simulation_duration):
    import math, random
    train_timetable = []

    for rep in range(replicate_times):
        num_trains_each_replicate = math.ceil(daily_throughput / (train_batch_size*2))

        full_cars_list_each_replicate = [train_batch_size] * (num_trains_each_replicate - 1)
        last_train_load = (daily_throughput - (train_batch_size * 2 * (num_trains_each_replicate - 1))) / 2   # just for oc/ic
        full_cars_list_each_replicate.append(last_train_load)

        train_ids_each_replicate = random.sample(range(1, 100), num_trains_each_replicate)
        time_slot = simulation_duration / replicate_times
        arrival_times = [rep * time_slot + i * (time_slot / num_trains_each_replicate) for i in range(num_trains_each_replicate)]

        for i in range(num_trains_each_replicate):
            train = {
                "train_id": train_ids_each_replicate[i],
                "arrival_time": int(round(arrival_times[i], 2)),
                "departure_time": int(round(arrival_times[i + 1], 2)) if i < num_trains_each_replicate - 1 else int(rep * time_slot + time_slot),
                "empty_cars": 0,
                "full_cars": int(full_cars_list_each_replicate[i]),
                "oc_number": int(full_cars_list_each_replicate[i]),
                "truck_number": max(int(full_cars_list_each_replicate[i]), int(full_cars_list_each_replicate[i]))
            }
            train_timetable.append(train)

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
                "simulation_duration": simulation_duration,
                "CRANE_NUMBER": state.CRANE_NUMBER,
                "HOSTLER_NUMBER": hostler_num
            }
        }

        with open("sim_config.json", "w") as f:
            json.dump(config, f)

        train_timetable = generate_train_timetable(daily_throughput, train_batch_size, replicate_times, simulation_duration)
        print(train_timetable)

        with open("train_timetable.json", "w") as f:
            json.dump(train_timetable, f)

        # single_track_simulation.py
        # pass new parameters
        state.CRANE_NUMBER = config["vehicles"]["CRANE_NUMBER"]
        state.HOSTLER_NUMBER = config["vehicles"]["HOSTLER_NUMBER"]

        subprocess.run(["python", "single_track_simulation.py"], check=True)

        with open("delay_output.json", "r") as f:
            delay_data = json.load(f)

        total_delay_time = delay_data["total_delay_time"]
        average_container_delay_time = delay_data["average_container_delay_time"]

        single_run = save_vehicle_and_performance_metrics(state, average_container_delay_time)

        if single_run is None:
            print("[Warning] Performance extraction failed, skipping this run.")
            continue

        avg_container_delay_time, ic_time, oc_time, total_time, ic_energy, oc_energy, total_energy = single_run

        num_trains = math.ceil(daily_throughput / (train_batch_size * 2))

        current_summary = {
            "Layout parameters": [int(M), int(N), int(n_t), int(n_p), int(n_r)],
            "Vehicle parameters": {"cranes": state.CRANE_NUMBER, "hostlers": hostler_num},
            "Total delay time": round(avg_container_delay_time,4),
            "Processing time (avg)": {
                "IC": round(ic_time, 4) + round(avg_container_delay_time, 4),
                "OC": round(oc_time, 4) + round(avg_container_delay_time, 4),
                "Total": round(total_time, 4) + round(avg_container_delay_time, 4),
            },
            "Energy consumption (avg)": {
                "IC": round(ic_energy, 4),
                "OC": round(oc_energy, 4),
                "Total": round(total_energy, 4)
            }
        }
        performance_matrix.append([daily_throughput, train_batch_size, num_trains, current_summary])


data = []
for entry in performance_matrix:
    daily_throughput, train_batch_size, num_trains, summary = entry

    M, N, n_t, n_p, n_r = summary["Layout parameters"]
    cranes = summary["Vehicle parameters"]["cranes"]
    hostlers = summary["Vehicle parameters"]["hostlers"]

    delay_time = summary["Total delay time"]

    ic_proc = summary["Processing time (avg)"]["IC"]
    oc_proc = summary["Processing time (avg)"]["OC"]
    total_proc = summary["Processing time (avg)"]["Total"]

    ic_energy = summary["Energy consumption (avg)"]["IC"]
    oc_energy = summary["Energy consumption (avg)"]["OC"]
    total_energy = summary["Energy consumption (avg)"]["Total"]

    data.append([
        daily_throughput, train_batch_size, num_trains,
        M, N, n_t, n_p, n_r,
        cranes, hostlers, delay_time,
        ic_proc, oc_proc, total_proc,
        ic_energy, oc_energy, total_energy
    ])


columns = [
    "daily_throughput(K)", "train_batch_size(k)", "num_trains",
    "M", "N", "n_t", "n_p", "n_r",
    "crane_numbers", "hostler_numbers", "avg_delay_time",
    "ic_avg_processing_time", "oc_avg_processing_time", "total_avg_processing_time",
    "ic_avg_energy", "oc_avg_energy", "total_avg_energy"
]

df = pd.DataFrame(data, columns=columns)

# mask = df["avg_delay_time"] == 0
# df_delay0 = df[mask]
# # IC
# df.loc[mask, "smoothed_ic_avg_processing_time"] = (
#     df_delay0.groupby("num_trains")["ic_avg_processing_time"]
#     .transform("mean").values
# )
#
# # OC
# df.loc[mask, "smoothed_oc_avg_processing_time"] = (
#     df_delay0.groupby("num_trains")["oc_avg_processing_time"]
#     .transform("mean").values
# )
#
# # Total
# df.loc[mask, "smoothed_total_avg_processing_time"] = (
#     df_delay0.groupby("num_trains")["total_avg_processing_time"]
#     .transform("mean").values
# )

df.to_excel("npf_performance_results.xlsx", index=False)
print("Done!")