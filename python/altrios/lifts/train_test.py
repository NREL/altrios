import simpy

def process_train(env, train_schedule, train_departed_event, next_departed_event):
    """
    Train arrival according to timetable: Handle the arrival, processing, and departure of a single train.
    """
    # Wait for the previous train to depart if the event exists
    if not train_departed_event.triggered:  # Only wait if the event hasn't succeeded
        yield train_departed_event
        print(f"Train {train_schedule['train_id']} starts after previous train departs at {env.now}")

    # Wait for the train to arrive
    yield env.timeout(train_schedule["arrival_time"] - env.now)
    print(f"Train {train_schedule['train_id']} arrived at {env.now}")

    # Simulate train processing (loading/unloading containers)
    processing_time = 100  # Example fixed processing time
    yield env.timeout(processing_time)
    print(f"Train {train_schedule['train_id']} completed processing at {env.now}")

    # Wait for the train to depart
    departure_delay = train_schedule["departure_time"] - env.now
    if departure_delay > 0:
        yield env.timeout(departure_delay)
    print(f"Train {train_schedule['train_id']} departed at {env.now}")

    # Trigger the departure event for the current train
    next_departed_event.succeed()
    print(f"Train {train_schedule['train_id']} departure event succeeded at {env.now}")


# Main simulation logic
env = simpy.Environment()

# Train timetable with arrival, departure, and processing details
train_timetable = [
    {"train_id": 19, "arrival_time": 187, "departure_time": 300, "empty_cars": 3, "full_cars": 5, "oc_number": 2,
     "truck_number": 5},
    {"train_id": 12, "arrival_time": 300, "departure_time": 800, "empty_cars": 5, "full_cars": 3, "oc_number": 4,
     "truck_number": 4},
]

# Initialize the departure event for the first train
train_departed_event = env.event()  # The first train starts immediately
train_departed_event.succeed()  # Trigger the event for the first train to start without waiting

# Create train processes based on the timetable
for i, train_schedule in enumerate(train_timetable):
    next_departed_event = env.event()  # Create a new departure event for the next train
    env.process(process_train(env, train_schedule, train_departed_event, next_departed_event))
    train_departed_event = next_departed_event  # Update the departure event for the next train

# Run the simulation
env.run()