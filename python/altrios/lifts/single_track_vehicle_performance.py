import pandas as pd
import utilities
from single_track_parameters import *
from pathlib import Path

package_root = utilities.package_root()
out_path = package_root / 'demos' / 'single_track_results'
out_path.mkdir(parents=True, exist_ok=True)

# Dictionary to store vehicle events
vehicle_events = {
    'crane': [],
    'hostler': [],
    'truck': []
}

def record_vehicle_event(vehicle_category, vehicle, action, state, time, emission, event_type, timestamp):
    vehicle_events[vehicle_category].append({
        'vehicle_id': str(vehicle),
        'action': action,
        'state': state,
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

    total_row = summary_df.sum(numeric_only=True)
    total_row['Vehicle'] = 'Total'
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

    # Calculate average row
    average_row = {
        'Vehicle': 'Average',
        'IC Energy Consumption': total_row['IC Emissions'] / ic_count if ic_count else 0,
        'OC Energy Consumption': total_row['OC Emissions'] / oc_count if oc_count else 0,
        'Total Energy Consumption': total_row['Total Emissions'] / (ic_count + oc_count) if (ic_count + oc_count) else 0,
        'IC Processing Time': total_row['IC Time'] / ic_count if ic_count else 0,
        'OC Processing Time': total_row['OC Time'] / oc_count if oc_count else 0,
        'Total Processing Time': total_row['Total Time'] / (ic_count + oc_count) if (ic_count + oc_count) else 0
    }
    average_row_list = list(average_row.values())
    summary_df.loc[len(summary_df)] = average_row_list

    return summary_df

def save_to_excel(state):
    df_logs = {}
    for vehicle_type, events in vehicle_events.items():
        df_logs[vehicle_type] = pd.DataFrame(events)

    ic_count = state.IC_NUM - 1
    oc_count = state.OC_NUM - 1

    # Calculate summary statistics
    performance_df = calculate_performance(df_logs, ic_count, oc_count)
    performance_df.rename(columns={'IC Emissions': 'IC Energy Consumption',
                               'OC Emissions': 'OC Energy Consumption',
                               'Total Emissions': 'Total Energy Consumption',
                               'IC Time': 'IC Processing Time',
                               'OC Time': 'OC Processing Time',
                               'Total Time': 'Total Processing Time'}, inplace=True)

    file_name = f"vehicle_log_crane_{state.CRANE_NUMBER}_hostler_{state.HOSTLER_NUMBER}.xlsx"
    file_path = out_path / file_name

    with pd.ExcelWriter(file_path) as writer:
        for vehicle_type, df in df_logs.items():
            df.to_excel(writer, sheet_name=vehicle_type, index=False)
        performance_df.to_excel(writer, sheet_name='performance', index=True)

    # print(f"total processed oc {state.OC_NUM}; total processed ic {state.IC_NUM}")
    print("*" * 100)
    print("Intermodal Terminal Performance Matrix")
    print(performance_df)
    print("*" * 100)

if __name__ == "__main__":
    global state
    save_to_excel(state)
