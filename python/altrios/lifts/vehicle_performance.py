import pandas as pd

vehicle_events = {
    'crane': [],
    'hostler': [],
    'truck': []
}

def record_vehicle_event(vehicle_category, vehicle, action, state, emission, event_type, timestamp):
    vehicle_events[vehicle_category].append({
        'vehicle_id': vehicle[0] if isinstance(vehicle, tuple) else vehicle.id,
        # 'vehicle_id': vehicle.to_string(),
        'action': action,
        'state': state,
        'emission': emission,
        'event_type': event_type,
        'timestamp': timestamp
    })

def calculate_average_times():
    averages = {}
    for vehicle_type, events in vehicle_events.items():
        total_time = 0
        count = 0
        for event in events:
            if event['event_type'] == 'start':
                start_time = event['timestamp']
            elif event['event_type'] == 'end':
                end_time = event['timestamp']
                total_time += (end_time - start_time)
                count += 1
        if count > 0:
            averages[vehicle_type] = total_time / count
        else:
            averages[vehicle_type] = 0
    return averages

def save_average_times():
    averages = calculate_average_times()
    with open("vehicle_average_times.txt", "w") as f:
        for vehicle_type, avg_time in averages.items():
            f.write(f"{vehicle_type}: {avg_time}\n")

def save_vehicle_logs():
    for vehicle_type, events in vehicle_events.items():
        log_file = f"{vehicle_type}_work_log.txt"
        with open(log_file, "w") as f:
            for event in events:
                f.write(f"Vehicle ID: {event['vehicle_id']}, "
                        f"Action: {event['action']}, "
                        f"State: {event['state']}, "
                        f"Event Type: {event['event_type']}, "
                        f"Emission: {event['emission']}, "
                        f"Timestamp: {event['timestamp']}\n")

if __name__ == "__main__":
    save_average_times()
