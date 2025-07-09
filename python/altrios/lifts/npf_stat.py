import os
import re
import math
import pandas as pd

def calculate_processing_time(container_excel_path, num_trains, train_batch_size, daily_throughput):
    df = pd.read_excel(container_excel_path, engine='openpyxl')

    obs_time_start = 24
    calculation_days = 20   # extend cool down for unprocessed containers
    obs_time_end = 24 * calculation_days
    timetable_days = 3  # rely on timetable durations, keep the same as replicate_times on npf.py

    df_ic = df[(df['type'] == 'IC') & (df['train_arrival'] > obs_time_start) & (df['train_arrival'] < obs_time_end)].copy()
    df_oc = df[(df['type'] == 'OC') & (df['hostler_pickup'] > obs_time_start) & (df['hostler_pickup'] < obs_time_end)].copy()

    ic_avg_process_time = df_ic['container_processing_time'].mean()
    df_ic["ic_delay_time"] = df_ic["train_arrival"] - df_ic["train_arrival_expected"]
    ic_avg_delay_time = df_ic["ic_delay_time"].sum() / len(df_ic)

    oc_avg_process_time = df_oc['container_processing_time'].mean()
    df_oc["ic_delay_time"] = df_oc["train_depart"] - df_ic["first_oc_pickup_time"]
    oc_avg_delay_time = df_ic["ic_delay_time"].sum() / len(df_oc)

    # over all simulation days
    all_processed_ic_df = df[df['type'] == 'IC'].copy()
    total_unprocessed_ic = math.ceil(timetable_days * (daily_throughput / 2) - len(all_processed_ic_df))

    all_processed_oc_df = df[df['type'] == 'OC'].copy()
    total_unprocessed_oc = math.ceil(timetable_days * (daily_throughput / 2) - len(all_processed_oc_df))

    return ic_avg_process_time, ic_avg_delay_time, oc_avg_process_time, oc_avg_delay_time, total_unprocessed_ic, total_unprocessed_oc


def process_all_files(folder_path):
    results = []

    for filename in os.listdir(folder_path):
        match = re.search(r'(\d+)C-(\d+)H_container_throughput_(\d+)_batch_size_(\d+)', filename)
        if match:
            cranes = int(match.group(1))
            hostlers = int(match.group(2))
            daily_throughput = int(match.group(3))
            train_batch_size = int(match.group(4))

            num_trains = math.ceil(daily_throughput / (train_batch_size * 2))
            file_path = os.path.join(folder_path, filename)

            try:
                ic_avg_process_time, ic_avg_delay_time, oc_avg_process_time, oc_avg_delay_time, total_unprocessed_ic, total_unprocessed_oc = calculate_processing_time(file_path, num_trains, train_batch_size, daily_throughput)

                results.append({
                    'cranes': cranes,
                    'hostlers': hostlers,
                    'daily_throughput': daily_throughput,
                    'train_batch_size': train_batch_size,
                    'num_trains': num_trains,
                    'total_unprocessed_ic': total_unprocessed_ic,
                    'ic_avg_time(no delay)': ic_avg_process_time,
                    'ic_avg_delay_time': ic_avg_delay_time,
                    'ic_avg_time': ic_avg_process_time + ic_avg_delay_time,
                    'total_unprocessed_oc': total_unprocessed_oc,
                    'oc_avg_time(no delay)': oc_avg_process_time,
                    'oc_avg_delay_time': oc_avg_delay_time,
                    'oc_avg_time': oc_avg_process_time + oc_avg_delay_time,
                })

                print(f"Done: {filename}")

            except Exception as e:
                print(f"Warning {filename}: {e}")
                continue

    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    folder_path = '/Users/qianqiantong/PycharmProjects/altrios-private/altrios/python/altrios/lifts/demos/single_track_results'

    final_results = process_all_files(folder_path)
    print(final_results)
    final_results.to_excel('container_processing_summary.xlsx', index=False)