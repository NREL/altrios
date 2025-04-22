import pandas as pd
import utilities
from single_track_parameters import *
from pathlib import Path
import altrios.lifts.distances_single_track as layout

K, k, M, N, n_t, n_p, n_r= layout.load_layout_config_from_json()

package_root = utilities.package_root()
out_path = package_root / 'demos' / 'single_track_results'
out_path.mkdir(parents=True, exist_ok=True)

# Dictionary to store vehicle events
vehicle_events = {
    'crane': [],
    'hostler': [],
    'truck': []
}

def record_vehicle_event(vehicle_category, vehicle, action, state, move, time, emission, event_type, timestamp):
    vehicle_events[vehicle_category].append({
        'vehicle_id': str(vehicle),
        'action': action,
        'state': state, # loaded or empty
        'move': move, # load or trip
        'time': time,
        'emission': emission,
        'event_type': event_type,
        'timestamp': timestamp
    })

def calculate_performance(dataframes, ic_count, oc_count):
    summary = {vehicle: {'IC Emissions': 0, 'OC Emissions': 0, 'Total Emissions': 0,
                         'IC Time': 0, 'OC Time': 0, 'Total Time': 0} for vehicle in dataframes.keys()}

    for vehicle, df in dataframes.items():
        for _, row in df.iterrows():
            emission = row['emission']
            time = row['time']
            action = row['action']

            if 'OC' in action:
                summary[vehicle]['OC Emissions'] += emission
                summary[vehicle]['OC Time'] += time
            else:
                summary[vehicle]['IC Emissions'] += emission
                summary[vehicle]['IC Time'] += time

            summary[vehicle]['Total Emissions'] = summary[vehicle]['IC Emissions'] + summary[vehicle]['OC Emissions']
            summary[vehicle]['Total Time'] = summary[vehicle]['IC Time'] + summary[vehicle]['OC Time']

    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={'index': 'Vehicle'}, inplace=True)

    total_row = summary_df.sum(numeric_only=True).to_dict()
    total_row['Vehicle'] = 'Total'
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

    # Append the average row directly below total row
    average_row = {
        'Vehicle': 'Average',
        'IC Emissions': total_row['IC Emissions'] / ic_count if ic_count else 0,
        'OC Emissions': total_row['OC Emissions'] / oc_count if oc_count else 0,
        'Total Emissions': total_row['Total Emissions'] / (ic_count + oc_count) if (ic_count + oc_count) else 0,
        'IC Time': total_row['IC Time'] / ic_count if ic_count else 0,
        'OC Time': total_row['OC Time'] / oc_count if oc_count else 0,
        'Total Time': total_row['Total Time'] / (ic_count + oc_count) if (ic_count + oc_count) else 0
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([average_row])], ignore_index=True)
    return summary_df

def save_energy_to_excel(state):
    df_logs = {}
    for vehicle_type, events in vehicle_events.items():
        df_logs[vehicle_type] = pd.DataFrame(events)

    ic_count = state.IC_NUM
    oc_count = state.OC_NUM

    # Calculate summary statistics
    performance_df = calculate_performance(df_logs, ic_count, oc_count)
    performance_df.rename(columns={'IC Emissions': 'IC Energy Consumption',
                               'OC Emissions': 'OC Energy Consumption',
                               'Total Emissions': 'Total Energy Consumption',
                               'IC Time': 'IC Processing Time',
                               'OC Time': 'OC Processing Time',
                               'Total Time': 'Total Processing Time'}, inplace=True)

    file_name = f"vehicle_throughput_{K}_batch_size_{k}.xlsx"
    file_path = out_path / file_name

    with pd.ExcelWriter(file_path) as writer:
        for vehicle_type, df in df_logs.items():
            df.to_excel(writer, sheet_name=vehicle_type, index=False)
        performance_df.to_excel(writer, sheet_name='performance', index=True)


# def calculate_container_processing_time(container_excel_path):
#     df = pd.read_excel(container_excel_path)
#     df['type'] = df['container_id'].apply(lambda x: 'IC' if str(x).isdigit() else 'OC' if str(x).startswith('OC-') else 'Unknown')
#     # df = df[(df["train_arrival"] >= 24) & (df["train_arrival"] < 144)]
#
#     ic_df = df[df['type'] == 'IC']
#     oc_df = df[df['type'] == 'OC']
#     # ic_df = df[(df['type'] == 'IC') & (df["train_arrival"] >= 24) & (df["train_arrival"] < 144)]
#     # oc_df = df[(df['type'] == 'OC') & (df["truck_arrival"] >= 24) & (df["truck_arrival"] < 144)]
#
#     ic_avg_time = ic_df['container_processing_time'].mean() if not ic_df.empty else 0
#     oc_avg_time = oc_df['container_processing_time'].mean() if not oc_df.empty else 0
#     total_avg_time = df['container_processing_time'].mean() if not df.empty else 0
#
#     return ic_avg_time, oc_avg_time, total_avg_time


import pandas as pd

def calculate_container_processing_time(container_excel_path, num_trains, train_batch_size, daily_throughput):
    df = pd.read_excel(container_excel_path)

    # 标记类型
    df['type'] = df['container_id'].apply(
        lambda x: 'IC' if str(x).isdigit() else 'OC' if str(x).startswith('OC-') else 'Unknown')

    full_load_limit = (num_trains - 1) * train_batch_size

    # 处理 IC
    ic_df = df[df['type'] == 'IC'].copy()
    ic_df.loc[:, 'numeric_id'] = ic_df['container_id'].astype(int)
    ic_full = ic_df[ic_df['numeric_id'] <= full_load_limit]
    ic_remaining = ic_df[ic_df['numeric_id'] > full_load_limit]
    ic_full_avg = ic_full['container_processing_time'].mean()
    ic_remain_avg = ic_remaining['container_processing_time'].mean()
    ic_avg_time = (ic_full_avg if not ic_full.empty else 0) + (ic_remain_avg if not ic_remaining.empty else 0)

    print(f"full_ic_avg_time: {ic_full_avg}, remaining_ic_avg_time: {ic_remain_avg}")

    # 处理 OC
    oc_df = df[df['type'] == 'OC'].copy()
    oc_df.loc[:, 'numeric_id'] = oc_df['container_id'].str.replace('OC-', '', regex=False).astype(int)
    oc_full = oc_df[oc_df['numeric_id'] <= full_load_limit]
    oc_remaining = oc_df[oc_df['numeric_id'] > full_load_limit]
    oc_full_avg = oc_full['container_processing_time'].mean()
    oc_remain_avg = oc_remaining['container_processing_time'].mean()
    oc_avg_time = (oc_full_avg if not oc_full.empty else 0) + (oc_remain_avg if not oc_remaining.empty else 0)

    print(f"full_oc_avg_time: {oc_full_avg}, remaining_oc_avg_time: {oc_remain_avg}")

    total_avg_time = (ic_avg_time + oc_avg_time)/2

    return ic_avg_time, oc_avg_time, total_avg_time



def calculate_vehicle_energy(vehicle_excel_path):
    df = pd.read_excel(vehicle_excel_path, sheet_name="performance")
    average_row = df[df['Vehicle'] == 'Average']

    ic_energy = average_row['IC Energy Consumption'].values[0]
    oc_energy = average_row['OC Energy Consumption'].values[0]
    total_energy = average_row['Total Energy Consumption'].values[0]

    return ic_energy, oc_energy, total_energy


def print_and_save_metrics(average_container_delay_time, ic_time, oc_time, total_time, ic_energy, oc_energy, total_energy):
    print("*" * 100)
    print("Intermodal Terminal Performance Matrix")
    print(f"Average Container Delay Time: {average_container_delay_time:.4f}")
    print(f"IC average processing time: {ic_time:.4f}")
    print(f"OC average processing time: {oc_time:.4f}")
    print(f"Average container processing time: {total_time:.4f}")
    print(f"IC average energy consumption: {ic_energy:.4f}")
    print(f"OC average energy consumption: {oc_energy:.4f}")
    print(f"Average container energy consumption: {total_energy:.4f}")
    print("*" * 100)

    single_run = [average_container_delay_time, ic_time, oc_time, total_time, ic_energy, oc_energy, total_energy]
    return single_run