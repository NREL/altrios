import pandas as pd
import random
import math
from altrios.lifts.distances_single_track import *
from altrios.lifts.single_track_parameters import *

# User-input
replicate_times = 3
daily_throughput = 1000
train_batch_size = 30
simulation_duration = 24 * replicate_times
start_time = 0

# layout_file = "C:/Users/mbruchon/Documents/Repos/NREL/altrios/python/altrios/lifts/single_track_input/layout.xlsx"
layout_file = "/Users/qianqiantong/PycharmProjects/altrios-private/altrios/python/altrios/lifts/single_track_input/layout.xlsx"
df_layout = pd.read_excel(layout_file)

if train_batch_size not in df_layout["train batch (k)"].values:
    raise ValueError("train_batch_size doesn't exist on layout.xlsx.")

layout_params = df_layout[df_layout["train batch (k)"] == train_batch_size * 2].iloc[0]
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


# def generate_train_timetable(daily_throughput, train_batch_size, replicate_times, simulation_duration):
#     import math, random
#     train_timetable = []
#
#     for rep in range(replicate_times):
#         num_trains_each_replicate = math.ceil(daily_throughput / (train_batch_size*2))
#
#         full_cars_list_each_replicate = [train_batch_size] * (num_trains_each_replicate - 1)
#         last_train_load = (daily_throughput - (train_batch_size * 2 * (num_trains_each_replicate - 1))) / 2   # just for oc/ic
#         full_cars_list_each_replicate.append(last_train_load)
#
#         train_ids_each_replicate = random.sample(range(1, 100), num_trains_each_replicate)
#         time_slot = simulation_duration / replicate_times
#         arrival_times = [rep * time_slot + i * (time_slot / num_trains_each_replicate) for i in
#                          range(num_trains_each_replicate)]
#
#         for i in range(num_trains_each_replicate):
#             train = {
#                 "train_id": train_ids_each_replicate[i],
#                 "arrival_time": int(round(arrival_times[i], 2)),
#                 "departure_time": int(round(arrival_times[i + 1], 2)) if i < num_trains_each_replicate - 1 else int(
#                     rep * time_slot + time_slot),
#                 "empty_cars": 0,
#                 "full_cars": int(full_cars_list_each_replicate[i]),
#                 "oc_number": int(full_cars_list_each_replicate[i]),
#                 "truck_number": max(int(full_cars_list_each_replicate[i]), int(full_cars_list_each_replicate[i]))
#             }
#             train_timetable.append(train)
#
#     return train_timetable

def generate_train_timetable(daily_throughput, train_batch_size, replicate_times, simulation_duration):
    import math, random
    train_timetable = []
    max_per_train = train_batch_size * 2  # 每列 train 最大容纳量

    for rep in range(replicate_times):
        total_container = daily_throughput
        num_trains = total_container // max_per_train
        remainder = total_container % max_per_train

        # 初始满载列车
        full_cars_list = [max_per_train] * num_trains

        # 若有余数，添加一列列车处理
        if remainder > 0:
            full_cars_list.append(remainder)
            num_trains += 1

            # 如果最后一列很小，平摊给前面几列
            if remainder < 0.5 * max_per_train and num_trains > 1:
                redistribute = remainder
                full_cars_list.pop()  # 删除最后一列
                for i in range(num_trains - 1, 0, -1):
                    if redistribute == 0:
                        break
                    give = min(5, redistribute)
                    full_cars_list[i - 1] += give
                    redistribute -= give

        # 确保 train 数量更新为列表长度
        num_trains = len(full_cars_list)

        # 在 full_cars_list 确定后，再生成 train_ids 和 arrival_times
        train_ids = random.sample(range(1, 100), num_trains)
        time_slot = simulation_duration / replicate_times
        arrival_times = [rep * time_slot + i * (time_slot / num_trains) for i in range(num_trains)]

        for i in range(num_trains):
            full_cars = int(full_cars_list[i] / 2)  # 分成一半为 IC，一半为 OC
            train = {
                "train_id": train_ids[i],
                "arrival_time": int(round(arrival_times[i], 2)),
                "departure_time": int(round(arrival_times[i + 1], 2)) if i < num_trains - 1 else int(rep * time_slot + time_slot),
                "empty_cars": 0,
                "full_cars": full_cars,
                "oc_number": full_cars,
                "truck_number": full_cars  # 当前逻辑：一集卡对应一个 container
            }
            train_timetable.append(train)

    return train_timetable


timetable = generate_train_timetable(daily_throughput, train_batch_size, replicate_times, simulation_duration)
print(timetable)
print("len of train timetable:", len(timetable))