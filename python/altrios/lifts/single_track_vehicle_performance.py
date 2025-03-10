import pandas as pd
import utilities
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


def record_vehicle_event(vehicle_category, vehicle, action, state, emission, event_type, timestamp):
    """Records an event related to a vehicle."""
    vehicle_events[vehicle_category].append({
        'vehicle_id': str(vehicle),
        'action': action,
        'state': state,
        'emission': emission,
        'event_type': event_type,
        'timestamp': timestamp
    })


def calculate_average_times():
    """Calculates average duration between 'start' and 'end' events for each vehicle category."""
    averages = {}
    for vehicle_type, events in vehicle_events.items():
        total_time = 0
        count = 0
        start_time = None

        for event in events:
            if event['event_type'] == 'start':
                start_time = event['timestamp']
            elif event['event_type'] == 'end' and start_time is not None:
                total_time += (event['timestamp'] - start_time)
                count += 1
                start_time = None

        averages[vehicle_type] = total_time / count if count > 0 else 0
    return averages


def save_to_excel(state):
    """Saves vehicle event logs and summary statistics to an Excel file."""
    # Convert vehicle event logs to DataFrames
    df_logs = {}
    for vehicle_type, events in vehicle_events.items():
        df_logs[vehicle_type] = pd.DataFrame(events)

    # Calculate summary statistics
    averages = calculate_average_times()
    summary_df = pd.DataFrame.from_dict(averages, orient='index', columns=['Average Time'])

    # Define output file name
    file_name = f"vehicle_log_crane_{state.CRANE_NUMBER}_hostler_{state.HOSTLER_NUMBER}.xlsx"
    file_path = out_path / file_name

    # Save to Excel
    with pd.ExcelWriter(file_path) as writer:
        for vehicle_type, df in df_logs.items():
            df.to_excel(writer, sheet_name=vehicle_type, index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=True)


if __name__ == "__main__":
    global state
    save_to_excel(state)