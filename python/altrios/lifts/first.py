import simpy
import random
import polars as pl
from altrios.lifts import utilities
from altrios.lifts.demo_parameters import *
from altrios.lifts.distances import *
from altrios.lifts.dictionary import *
from altrios.lifts.schedule import *
from altrios.lifts.single_track_vehicle_performance import record_vehicle_event, save_average_times, save_vehicle_logs


class Terminal:
    def __init__(self, env, truck_capacity, chassis_count):
        self.env = env
        self.cranes = simpy.Resource(env, state.CRANE_NUMBER)  # Crane source: numbers
        self.in_gates = simpy.Resource(env, state.IN_GATE_NUMBERS)  # In-gate source: numbers
        self.out_gates = simpy.Resource(env, state.OUT_GATE_NUMBERS)  # Out-gate source: numbers
        self.truck_store = simpy.Store(env, capacity=truck_capacity)  # Truck source: numbers
        self.oc_store = simpy.Store(env)  # Outbound container source: numbers
        self.chassis = simpy.FilterStore(env, capacity=chassis_count)  # Chassis source: numbers
        self.hostlers = simpy.Store(env, capacity=state.HOSTLER_NUMBER)  # Hostler source: numbers
        for hostler_id in range(1, state.HOSTLER_NUMBER + 1):
            self.hostlers.put(hostler_id)


def record_event(container_id, event_type, timestamp):
    global state
    if container_id not in state.container_events:
        state.container_events[container_id] = {}
    state.container_events[container_id][event_type] = timestamp


def truck_entry(env, terminal, truck_id, oc_id):
    global state
    with terminal.in_gates.request() as gate_request:
        yield gate_request
        print(f"Time {env.now}: Truck {truck_id} passed the in-gate and is entering the terminal")
        yield env.timeout(state.TRUCK_INGATE_TIME + random.uniform(0,state.TRUCK_INGATE_TIME_DEV))  # truck passing gate time: 1 sec (demo_parameters.TRUCK_INGATE_TIME and TRUCK_INGATE_TIME_DEV)

        # Assign IDs for OCs
        terminal.oc_store.put(f"OC-{oc_id}")
        # Each truck drops an OC to the closest parking lot according to triangular distribution
        d_t_dist = create_triang_distribution(d_t_min, d_t_avg, d_t_max).rvs()
        yield env.timeout(d_t_dist / (2 * state.TRUCK_SPEED_LIMIT))
        print(f"Time {env.now}: Truck {truck_id} placed OC {oc_id} at parking slot.")


def container_process(env, terminal, train_schedule, all_trucks_arrived_event, all_oc_prepared):
    global state
    truck_number = train_schedule["truck_number"]
    arrival_rate = 1  # Truck arrival rate: poisson distribution, arrival rate depends on the gap between last train departure and next train arrival

    # Trucks enter according to truck schedule (truck_numbers)
    for truck_id in range(1, truck_number + 1):
        print("truck_id", truck_id)
        oc_id = truck_id
        yield env.timeout(random.expovariate(arrival_rate))  # Assume truck arrives according to the poisson distribution
        terminal.truck_store.put(truck_id)
        # Truck drops off an OC
        env.process(truck_entry(env, terminal, truck_id, oc_id))
        # Hostler picks up an OC
        if len(terminal.oc_store.items) >= 0:
            hostler_id = yield terminal.hostlers.get()
            print(f'Hostler assigned for the next OC {oc_id}: {hostler_id}')

            # Hostler picks up the OC
            oc_id = yield terminal.oc_store.get()
            travel_time_to_oc = state.HOSTLER_TRANSPORT_CONTAINER_TIME
            yield env.timeout(travel_time_to_oc)
            print(f"Time {env.now}: Hostler {hostler_id} picked up OC {oc_id} and heading to drop off OC")
            record_event(oc_id, 'hostler_pickup', env.now)

            # Hostler drops off the OC to chassis
            yield env.timeout(travel_time_to_oc)
            yield terminal.chassis.put(oc_id)
            print(f"Time {env.now}: Hostler {hostler_id} dropped OC {oc_id} and returned to the terminal")
            record_event(oc_id, 'hostler_dropoff', env.now)

            yield env.timeout(travel_time_to_oc)
            yield terminal.hostlers.put(hostler_id)
            print(f"Time {env.now}: Hostler {hostler_id} has returned to the platoon")

    yield env.timeout(state.TRUCK_TO_PARKING)  # truck travel time of placing OC at parking slot (demo_parameters.TRUCK_TO_PARKING)
    all_trucks_arrived_event.succeed()  # if all_trucks_arrived_event is triggered, train is allowed to enter

    if len(terminal.chassis.items) == truck_number:
        print("Chassis are fully loaded, and the train is ready to depart")
        all_oc_prepared.succeed()


def crane_load_process(env, terminal, load_time, start_load_event, end_load_event):
    global state
    yield start_load_event
    print(f"Time {env.now}: Starting loading process onto the train.")
    print("items on chassis", terminal.chassis.items)

    while len([item for item in terminal.chassis.items if isinstance(item, str) and "OC" in item]) > 0:  # if there still has OC on chassis
        with terminal.cranes.request() as request:
            yield request
            oc = yield terminal.chassis.get(lambda x: isinstance(x, str) and "OC" in x)  # obtain an OC from chassis
            # print("chassis:",terminal.chassis.items)
            print(f"Time {env.now}: Crane starts loading {oc} onto the train")
            yield env.timeout(load_time)  # loading time, depends on container parameter**
            record_event(oc, 'crane_load', env.now)
            print(f"Time {env.now}: Crane finished loading {oc} onto the train")

    print(f"Time {env.now}: All OCs loaded. Train is fully loaded and ready to depart!")
    print("Containers on chassis (after loading OCs):", terminal.chassis.items)
    end_load_event.succeed()


def process_train_arrival(env, terminal, train_departed_event, train_schedule, next_departed_event):
    global state
    # yield train_departed_event
    if train_departed_event is not None:
        yield train_departed_event

    train_id = train_schedule["train_id"]
    arrival_time = train_schedule["arrival_time"]
    departure_time = train_schedule["departure_time"]
    oc_needed = train_schedule["oc_number"]
    total_ic = train_schedule["full_cars"]

    print(f"------------------- The current train is {train_id}: scheduled arrival time {arrival_time}, OC {oc_needed}, IC {total_ic} -------------------")

    # create events as processing conditions
    all_trucks_arrived_event = env.event()  # condition for train arrival
    all_oc_prepared = env.event()  # condition 2 for crane loading
    start_load_event = env.event()  # condition 1 for train departure
    end_load_event = env.event()  # condition 2 for train departure

    # Initialize dictionary
    delay_list = {}

    # Train is waiting to load OCs
    yield env.timeout(arrival_time - env.now)
    print(f"Time {env.now}: [In Time] Train {train_schedule['train_id']} has arrived.")

    # All trucks arrive
    env.process(container_process(env, terminal, train_schedule, all_trucks_arrived_event, all_oc_prepared))

    # Check loading conditions
    yield all_trucks_arrived_event & all_oc_prepared
    print(f"Time {env.now}: All {oc_needed} OCs are ready on chassis.")
    start_load_event.succeed()  # condition of chassis loading

    # crane loading process
    env.process(crane_load_process(env, terminal, load_time=2,start_load_event=start_load_event, end_load_event=end_load_event))

    yield end_load_event

    # train departs
    if env.now <= departure_time:
        yield env.timeout(departure_time - env.now)
        print(f"Time {env.now}: [In Time] Train {train_schedule['train_id']} departs.")
    else:
        delay_time = env.now - departure_time
        if f"train_id_{train_id}" not in delay_list:
            delay_list[f"train_id_{train_id}"] = {}
        delay_list[f"train_id_{train_id}"]["departure"] = delay_time
        print(f"Time {env.now}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours.")

    print(f"Time {env.now}: Train is departing the terminal.")

    for oc_id in range(state.OC_NUM, state.OC_NUM + train_schedule['oc_number']):
        record_event(f"OC-{oc_id}", 'train_depart',env.now)  # loop: assign container_id range(current_oc, current_oc + train_schedule['full_cars'])

    state.time_per_train[train_schedule['train_id']] = env.now - arrival_time

    # Update various parameters to track inbound and outbound containers
    state.IC_NUM = state.IC_NUM + train_schedule['full_cars']
    state.OC_NUM = state.OC_NUM + train_schedule['oc_number']

    # # Trigger the departure event for the current train
    next_departed_event.succeed()  # the terminal now is clear and ready to accept the next train


def run_simulation(train_consist_plan: pl.DataFrame, terminal: str, out_path=None):
    global state
    state.terminal = terminal
    state.train_consist_plan = train_consist_plan
    state.initialize()

    # Create environment
    env = simpy.Environment()

    # Train timetable: only one train
    # For initializations, only load OCs, so full_cars = 0
    # Trucks prepare according to truck_number, and truck_number = oc_number = empty_time
    train_timetable = [
        {"train_id": 2, "arrival_time": 0, "departure_time": 50, "empty_cars": 8, "full_cars": 0, "oc_number": 8,
         "truck_number": 8},  # test: ic > oc
        ]

    # Initialize train status in the terminal
    train_departed_event = env.event()
    train_departed_event.succeed()

    truck_number = max([entry['truck_number'] for entry in train_timetable])
    chassis_count = max([entry['empty_cars'] + entry['full_cars'] for entry in train_timetable])
    terminal = Terminal(env, truck_capacity=truck_number, chassis_count=chassis_count)

    # Trains arrive according to timetable
    for i, train_schedule in enumerate(train_timetable):
        print(train_schedule)
        next_departed_event = env.event()  # Create a new departure event for the next train
        env.process(process_train_arrival(env, terminal, train_departed_event, train_schedule, next_departed_event))
        train_departed_event = next_departed_event  # Update the departure event for the next train

    # Simulation hyperparameters
    env.run(until=state.sim_time)

    # Create DataFrame for container events
    print("Total simulation time (hr):", state.sim_time)

    # container_data = (
    #     pl.from_dicts(
    #         [dict(event, **{'container_id': container_id}) for container_id, event in state.container_events.items()]
    #     )
    #     .with_columns(
    #         pl.when(
    #             pl.col("truck_exit").is_not_null() & pl.col("train_arrival").is_not_null()
    #         )
    #         .then(
    #             pl.col("truck_exit") - pl.col("train_arrival")
    #         )
    #         .when(
    #             pl.col("train_depart").is_not_null()
    #         )
    #         .then(
    #             pl.col("crane_load") - pl.col("truck_arrival")
    #         )
    #         .otherwise(None)
    #         .alias("container_processing_time")
    #     )
    #     .sort("container_id")
    #     .select(pl.col("container_id"), pl.all().exclude("container_id"))
    # )
    # if out_path is not None:
    #     container_data.write_excel(
    #         out_path / f"first_station_simulation_crane_{state.CRANE_NUMBER}_hostler_{state.HOSTLER_NUMBER}.xlsx")

    # Use save_average_times and save_vehicle_logs for vehicle related logs
    # save_average_times()
    # save_vehicle_logs()

    print("Done!")
    # return container_data


if __name__ == "__main__":
    run_simulation(
        train_consist_plan=pl.read_csv(utilities.package_root() / 'demos' / 'starter_demo' / 'train_consist_plan.csv'),
        terminal="Allouez",
        out_path=utilities.package_root() / 'demos' / 'starter_demo' / 'results'
    )