import simpy
import random
from altrios.lifts.single_track_timetable import *
import polars as pl
from altrios.lifts import utilities
from altrios.lifts.single_track_parameters import *
from altrios.lifts.distances import *
from altrios.lifts.dictionary import *
from altrios.lifts.schedule import *
from altrios.lifts.single_track_vehicle_performance import *


class Terminal:
    def __init__(self, env, truck_capacity, chassis_count):
        self.env = env
        self.tracks = simpy.Store(env, capacity=state.TRACK_NUMBER)
        for track_id in range(1, state.TRACK_NUMBER + 1):
            self.tracks.put(track_id)
        self.cranes = simpy.Store(env, state.CRANE_NUMBER)
        self.in_gates = simpy.Resource(env, state.IN_GATE_NUMBERS)
        self.out_gates = simpy.Resource(env, state.OUT_GATE_NUMBERS)
        self.oc_store = simpy.Store(env)
        self.parking_slots = simpy.Store(env)
        self.chassis = simpy.FilterStore(env, capacity=chassis_count)
        self.hostlers = simpy.Store(env, capacity=state.HOSTLER_NUMBER)
        self.truck_store = simpy.Store(env, capacity=truck_capacity)

        # resource setting
        # crane
        num_diesel_crane = round(state.CRANE_NUMBER * state.CRANE_DIESEL_PERCENTAGE)
        num_hybrid_crane = state.CRANE_NUMBER - num_diesel_crane
        cranes = [(i, "diesel") for i in range(num_diesel_crane)] + \
                 [(i + num_diesel_crane, "hybrid") for i in range(num_hybrid_crane)]
        for crane_id in cranes:
            self.cranes.put(crane_id)
        print(f"crane {cranes}")

        # hostler
        num_diesel = round(state.HOSTLER_NUMBER * state.HOSTLER_DIESEL_PERCENTAGE)
        num_electric = state.HOSTLER_NUMBER - num_diesel
        hostlers = [(i, "diesel") for i in range(num_diesel)] + \
                 [(i + num_diesel, "electric") for i in range(num_electric)]
        for hostler_id in hostlers:
            self.hostlers.put(hostler_id)
        print(f"hostler {hostlers}")


def record_event(container_id, event_type, timestamp):
    global state
    if container_id not in state.container_events:
        state.container_events[container_id] = {}
    state.container_events[container_id][event_type] = timestamp


def emission_calculation(status, vehicle, id, travel_time):
    global state
    vehicle = vehicle.capitalize()
    vehicle_type = id[1]
    vehicle_type = vehicle_type.capitalize()
    if status == 'loaded':
        emission_unit = state.FULL_EMISSIONS_RATES[vehicle][vehicle_type]
    if status == 'empty':
        emission_unit = state.IDLE_EMISSIONS_RATES[vehicle][vehicle_type]
    emissions = emission_unit * travel_time

    return emissions


def truck_entry(env, train_schedule, terminal, truck_id):
    global state
    with terminal.in_gates.request() as gate_request:
        yield gate_request
        print(f"Time {env.now}: Truck {truck_id} passed the in-gate and is entering the terminal")

        oc_id = yield terminal.oc_store.get()
        record_event(oc_id, 'truck_arrival', env.now)
        # Calculate truck speed according to the current density
        truck_travel_time = simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
        emissions = emission_calculation('loaded', 'truck', truck_id, truck_travel_time)
        record_vehicle_event('truck', truck_id, f'entry_{oc_id}', 'loaded', truck_travel_time, emissions, 'end', env.now)

        # Assume each truck takes 1 OC, and drop OC to the closest parking lot according to triangular distribution
        # Assign IDs for OCs
        print(f"Time {env.now}: Truck {truck_id} placed {oc_id} at parking slot.")
        terminal.parking_slots.put(oc_id)
        record_event(oc_id, 'truck_dropoff', env.now)
        emissions = emission_calculation('loaded', 'truck', truck_id, truck_travel_time)
        record_vehicle_event('truck', truck_id, f'dropoff_{oc_id}', 'loaded', truck_travel_time, emissions, 'end', env.now)


def empty_truck(env, train_schedule, terminal, truck_id):
    global state
    with terminal.in_gates.request() as gate_request:
        yield gate_request
        print(f"Time {env.now}: Truck {truck_id} passed the in-gate with empty loading")
        # Note the arrival of empty trucks will not be recorded due to excel output dimensions
        truck_travel_time = simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
        emissions = emission_calculation('empty', 'truck', truck_id, truck_travel_time)
        record_vehicle_event('truck', truck_id, 'pass_gate', 'empty', truck_travel_time, emissions, 'end', env.now)


def truck_arrival(env, terminal, train_schedule, all_trucks_arrived_event):
    global state
    truck_number = train_schedule["truck_number"]
    num_diesel = round(truck_number * state.TRUCK_DIESEL_PERCENTAGE)
    num_electric = truck_number - num_diesel
    # arrival_rate = 0.1  # truck arrival rate: poisson distribution, arrival rate depends on the gap between last train departure and next train arrival

    trucks = [(i, "diesel") for i in range(num_diesel)] + \
             [(i + num_diesel, "electric") for i in range(num_electric)]
    random.shuffle(trucks)
    print(f"truck platoon has total {len(trucks)} including {trucks}")
    print(f"oc store has {terminal.oc_store.items}")

    for truck_id in trucks:
        yield env.timeout(0)  # Assume truck arrives not impact on system, not random.expovariate(arrival_rate)
        terminal.truck_store.put(truck_id)
        if len(terminal.oc_store.items)!= 0:    # Truck move OC from outside (oc_store) to terminal (parking_slots)
            print("len(terminal.oc_store.items)", len(terminal.oc_store.items))
            env.process(truck_entry(env, train_schedule, terminal, truck_id))
        else:
            env.process(empty_truck(env, train_schedule, terminal, truck_id))

    yield env.timeout(state.TRUCK_TO_PARKING)    # truck travel time of placing OC at parking slot (demo_parameters.TRUCK_TO_PARKING)
    all_trucks_arrived_event.succeed()  # if all_trucks_arrived_event is triggered, train is allowed to enter


def crane_unload_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, total_ic, all_ic_picked, all_ic_unload_event):
    global state
    ic_unloaded_count = 0

    for ic_id in range(state.IC_NUM, state.IC_NUM + total_ic):
        crane_id = yield terminal.cranes.get()
        print(f"Time {env.now}: Crane starts unloading IC {ic_id}")
        crane_unloading_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME)
        yield env.timeout(crane_unloading_time)
        record_event(ic_id, 'crane_unload', env.now)

        print(f"Time {env.now}: Crane {crane_id} finishes unloading IC {ic_id} onto chassis")
        emissions = emission_calculation('loaded', 'crane',crane_id, state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME))
        record_vehicle_event('crane', crane_id, f'unload_{ic_id}', 'unloaded', crane_unloading_time, emissions, 'end', env.now)
        terminal.chassis.put(ic_id)
        terminal.cranes.put(crane_id)
        print(f"Time {env.now}: chassis (loading ic):", terminal.chassis.items)
        print(f"Time {env.now}: hostler (loading ic):", terminal.hostlers.items)
        ic_unloaded_count += 1
        env.process(container_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, all_ic_unload_event, all_ic_picked))

        if ic_unloaded_count == total_ic:
            all_ic_unload_event.succeed()
            print(f"Time {env.now}: All ICs have been unloaded onto the chassis.")


def container_process(env, terminal,train_schedule, all_oc_prepared, oc_needed, all_ic_unload_event, all_ic_picked):
    global state
    '''
    It is designed to transfer both inbound and outbound containers.
    The main simulation process is as follows:
    1. A hostler picks up an IC, and drops off IC at parking slot.
    2. A truck picks up the IC, and leaves the gate
    3. The hostler picks up an OC, and drops off OC at the chassis.
    4. Once all OCs are prepared (all_oc_prepared), the crane starts loading (other function).
    '''
    hostler_id = yield terminal.hostlers.get()


    # Hostler puts IC to the closest parking lot
    current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    hostler_travel_time_to_parking = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
    yield env.timeout(hostler_travel_time_to_parking)
    # Hostler picks up IC from chassis
    ic_id = yield terminal.chassis.get(lambda x: isinstance(x, int))
    print(f"Time {env.now}: Hostler {hostler_id} picked up IC {ic_id} and is heading to parking slot.")
    print(f"Chassis: {terminal.chassis.items}")
    record_event(ic_id, 'hostler_pickup', env.now)
    emissions = emission_calculation('empty', 'hostler',hostler_id, hostler_travel_time_to_parking)
    record_vehicle_event('hostler', hostler_id, f'pickup_IC_{ic_id}', 'empty', hostler_travel_time_to_parking, emissions, 'end', env.now)

    # Prepare for crane loading: if chassis has no IC AND all_ic_picked (parking side) is not triggered => trigger all_ic_picked
    if sum(str(item).isdigit() for item in terminal.chassis.items) == 0 and all_ic_unload_event.triggered:
        all_ic_picked.succeed()
        print(f"Time {env.now}: All ICs are picked up by hostlers.")

    # Test: status for chassis
    print("Containers on chassis (hostler picking-up IC):", terminal.chassis.items)
    print("# of IC on chassis:", sum(str(item).isdigit() for item in terminal.chassis.items))

    # Hostler drop off IC to parking slot
    current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    hostler_travel_time_to_parking = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
    yield env.timeout(hostler_travel_time_to_parking)   # update: time calculated by density-travel_time function
    print(f"Time {env.now}: Hostler {hostler_id} dropped off IC {ic_id} at parking slot.")
    record_event(ic_id, 'hostler_dropoff', env.now)
    emissions = emission_calculation('loaded', 'hostler',hostler_id, hostler_travel_time_to_parking)
    record_vehicle_event('hostler', hostler_id, f'dropoff_IC_{ic_id}', 'loaded', hostler_travel_time_to_parking, emissions, 'end', env.now)

    # Assign a truck to pick up IC
    truck_id = yield terminal.truck_store.get()

    print(f"Time {env.now}: Truck {truck_id} is assigned to IC {ic_id} for exit.")
    record_event(ic_id, 'truck_pickup', env.now)
    truck_travel_time = simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
    yield env.timeout(truck_travel_time)
    emissions = emission_calculation('empty', 'truck',truck_id, truck_travel_time)
    record_vehicle_event('truck', truck_id, f'pickup_IC_{ic_id}', 'empty', truck_travel_time, emissions, 'end', env.now)

    # Truck queue and exit the gate
    env.process(truck_exit(env, terminal, truck_id, ic_id))

    # Assign a hostler to pick up an OC
    if len(terminal.parking_slots.items) > 0:
        oc = yield terminal.parking_slots.get()
        print(f"Time {env.now}: Hostler {hostler_id} is going to pick up {oc}")

        # Test: OC remaining before hostlers pick up OCs
        print(f"Hostlers: {terminal.hostlers.items}")
        print(f"OC remains (parking slots): {terminal.parking_slots.items}")

        # The hostler picks up an OC
        hostler_reposition_travel_time = simulate_reposition_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
        yield env.timeout(hostler_reposition_travel_time)
        print(f"Time {env.now}: Hostler {hostler_id} picked up {oc} and is returning to the terminal")
        record_event(oc, 'hostler_pickup', env.now)
        emissions = emission_calculation('empty', 'hostler',hostler_id, hostler_reposition_travel_time)
        record_vehicle_event('hostler', hostler_id, f'pickup_{oc}', 'empty', hostler_reposition_travel_time, emissions, 'end',env.now)

        # Test: Containers after hostler picking-up OC
        print("Containers on chassis (after hostler picking-up OC):", terminal.chassis.items)
        print("# of IC", sum(str(item).isdigit() for item in terminal.chassis.items))

        # The hostler drops off OC at the chassis
        current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
        hostler_travel_time_to_parking = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
        yield env.timeout(hostler_travel_time_to_parking)
        print(f"Time {env.now}: Hostler {hostler_id} dropped off {oc} onto chassis")
        yield terminal.chassis.put(oc)
        record_event(oc, 'hostler_dropoff', env.now)
        emissions = emission_calculation('loaded', 'hostler', hostler_id, hostler_reposition_travel_time)
        record_vehicle_event('hostler', hostler_id, f'dropoff_{oc}', 'loaded', hostler_travel_time_to_parking, emissions, 'end',env.now)

        # The hostler-truck-hostler process keeps going, until conditions are satisfied and then further trigger crane movement.
        # ICs are all picked up and OCs are prepared
        if sum(1 for item in terminal.chassis.items if "OC-" in str(item)) == oc_needed:
            all_oc_prepared.succeed()
            print("chassis (check if all oc prepared):", terminal.chassis.items)
            print(f"hostler (check if all oc prepared): {terminal.hostlers.items}")
            print("# of OC on chassis:", sum(1 for item in terminal.chassis.items if "OC-" in str(item)))
            print(f"Time {env.now}: All OCs are ready on chassis.")

    current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    hostler_travel_time_to_parking = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
    emissions = emission_calculation('empty', 'hostler', hostler_id, hostler_travel_time_to_parking)
    yield env.timeout(hostler_travel_time_to_parking)   # update: time calculated by density-travel_time function
    print(f"Time {env.now}: Hostler {hostler_id} return to parking slot.")
    yield terminal.hostlers.put(hostler_id)
    record_vehicle_event('hostler', hostler_id, f'back_parking', 'empty', hostler_travel_time_to_parking, emissions, 'end', env.now)

    # Test: check if all OCs on chassis
    print("chassis (all oc prepared):", terminal.chassis.items)
    print("# of IC on chassis:", sum(str(item).isdigit() for item in terminal.chassis.items))
    print(f"hostler (all oc prepared): {terminal.hostlers.items}")

    # IC < OC: ICs are all picked up and still have OCs remaining
    print("# of OCs in parking_slots:", len(terminal.parking_slots.items))
    if sum(str(item).isdigit() for item in terminal.chassis.items) == 0 and len(terminal.parking_slots.items) == train_schedule['oc_number'] - train_schedule['full_cars'] :
        print(f"ICs are prepared, but OCs remaining: {terminal.parking_slots.items}")
        remaining_oc = len(terminal.parking_slots.items)
        # Repeat hostler picking-up OC process only, until all remaining OCs are transported
        for i in range (1, remaining_oc + 1):
            oc = yield terminal.parking_slots.get()
            print(f"The OC is {oc}")

            pickup_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
            hostler_reposition_travel_time = simulate_reposition_travel(hostler_id, pickup_veh_num, total_lane_length, d_r_min, d_r_max)
            yield env.timeout(hostler_reposition_travel_time)
            print(f"Time {env.now}: Hostler {hostler_id} picked up {oc} and is returning to the terminal")
            record_event(oc, 'hostler_pickup', env.now)
            emissions = emission_calculation('empty', 'hostler',hostler_id, hostler_reposition_travel_time)
            record_vehicle_event('hostler', hostler_id, f'pickup_{oc}', 'empty', hostler_reposition_travel_time, emissions, 'end', env.now)

            dropoff_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
            hostler_travel_time = simulate_reposition_travel(hostler_id, dropoff_veh_num, total_lane_length, d_h_min, d_h_max)
            yield env.timeout(hostler_travel_time)
            print(f"Time {env.now}: Hostler {hostler_id} dropped off {oc} onto chassis")
            yield terminal.chassis.put(oc)
            record_event(oc, 'hostler_dropoff', env.now)
            emissions = emission_calculation('loaded', 'hostler',hostler_id, hostler_reposition_travel_time)
            record_vehicle_event('hostler', hostler_id, f'dropoff_{oc}', 'loaded', hostler_travel_time, emissions,'end', env.now)

            print("chassis (oc_remaining):", terminal.chassis.items)
            print(f"hostler (oc_remaining): {terminal.hostlers.items}")
            if sum(1 for item in terminal.chassis.items if isinstance(item, str) and "OC-" in str(item)) == oc_needed:
                all_oc_prepared.succeed()
                print("chassis (all_oc_prepared):", terminal.chassis.items)
                print(f"hostler (all_oc_prepared): {terminal.hostlers.items}")
                print("# of OC on chassis:", sum(1 for item in terminal.chassis.items if "OC-" in str(item)))
                print(f"Time {env.now}: All OCs are ready on chassis.")
            i += 1


def truck_exit(env, terminal, truck_id, ic_id):
    global state
    with terminal.out_gates.request() as out_gate_request:
        yield out_gate_request
        print(f"Time {env.now}: Truck {truck_id} with IC {ic_id} is passing through the out-gate and leaving the terminal")
        truck_travel_time = state.TRUCK_OUTGATE_TIME + random.uniform(0,state.TRUCK_OUTGATE_TIME_DEV)
        yield env.timeout(truck_travel_time)
        record_event(ic_id, 'truck_exit', env.now)
        emissions = emission_calculation('loaded', 'truck',truck_id, truck_travel_time)
        record_vehicle_event('truck', truck_id, f'leave_gate_IC_{ic_id}', 'loaded', truck_travel_time, emissions, 'end', env.now)
    yield terminal.truck_store.put(truck_id)


def crane_load_process(env, terminal, start_load_event, end_load_event):
    global state
    yield start_load_event
    print(f"Time {env.now}: Starting loading process onto the train.")

    while len([item for item in terminal.chassis.items if isinstance(item, str) and "OC" in item]) > 0:  # if there still has OC on chassis
        crane_id = yield terminal.cranes.get()
        oc = yield terminal.chassis.get(lambda x: isinstance(x, str) and "OC" in x)  # obtain an OC from chassis
        print(f"Time {env.now}: Crane {crane_id} starts loading {oc} onto the train.")
        crane_loading_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME)
        yield env.timeout(crane_loading_time)
        record_event(oc, 'crane_load', env.now)
        print(f"Time {env.now}: Crane finished loading {oc} onto the train.")
        terminal.cranes.put(crane_id)
        emissions = emission_calculation('loaded', 'crane', crane_id, state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME))
        record_vehicle_event('crane', crane_id, f'load_{oc}', 'loaded', crane_loading_time, emissions, 'end', env.now)

    print(f"Time {env.now}: All OCs loaded. Train is fully loaded and ready to depart.")
    print("Containers on chassis (after loading OCs):", terminal.chassis.items)
    end_load_event.succeed()


def process_train_arrival(env, terminal, train_departed_event, train_schedule, next_departed_event):
    global state
    # yield train_departed_event
    if train_departed_event is not None:
        yield train_departed_event

    train_id = train_schedule["train_id"]
    arrival_time = train_schedule["arrival_time"]
    oc_needed = train_schedule["oc_number"]
    total_ic = train_schedule["full_cars"]

    print(f"------------------- The current train is {train_id}: scheduled arrival time {arrival_time}, OC {oc_needed}, IC {total_ic} -------------------")

    # create events as processing conditions
    all_trucks_arrived_event = env.event()  # condition for train arrival
    all_ic_unload_event = env.event()       # condition for all ic picked
    all_ic_picked = env.event()             # condition 1 for crane loading
    all_oc_prepared = env.event()           # condition 2 for crane loading
    start_load_event = env.event()          # condition 1 for train departure
    end_load_event = env.event()            # condition 2 for train departure

    # Initialize dictionary
    delay_list = {}

    oc_id = state.OC_NUM
    print("start oc_id:", oc_id)
    for oc in range(state.OC_NUM, state.OC_NUM + train_schedule['oc_number']):
        terminal.oc_store.put(f"OC-{oc_id}")
        oc_id += 1
    print("oc has:", terminal.oc_store.items)

    # All trucks arrive before train arrives
    env.process(truck_arrival(env, terminal, train_schedule, all_trucks_arrived_event))

    # Track assignment for a train
    print("Current available track has ", terminal.tracks.items)
    track_id = yield terminal.tracks.get()
    if track_id is None:
        print(f"Time {env.now}: Train {train_id} is waiting for an available track.")
        terminal.waiting_trains.append(train_id)
        return

    # Wait train arriving
    if env.now <= arrival_time:
        yield env.timeout(arrival_time - env.now)
        print(f"Time {env.now}: [In Time] Train {train_schedule['train_id']} has arrived, waiting to be assigned to the track {track_id}.")
        delay_time = 0
    else:
        delay_time = env.now - arrival_time
        if f"train_id_{train_id}" not in delay_list:
            delay_list[f"train_id_{train_id}"] = {}
        delay_list[f"train_id_{train_id}"]["arrival"] = delay_time
        print(f"Time {env.now}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours, waiting to be assigned to the track {track_id}.")
    state.train_delay_time[train_schedule['train_id']] = delay_time

    for ic_id in range(state.IC_NUM, state.IC_NUM + train_schedule['full_cars']):
        record_event(ic_id, 'train_arrival',env.now)  # loop: assign container_id range(current_ic, current_ic + train_schedule['full_cars'])

    # crane unloading IC
    env.process(crane_unload_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, total_ic, all_ic_picked, all_ic_unload_event))

    # prepare all OC and pick up all IC before crane loading
    yield all_ic_picked & all_oc_prepared
    print(f"Time {env.now}: All {oc_needed} OCs are ready on chassis.")
    start_load_event.succeed()  # condition of chassis loading

    # crane loading process
    env.process(crane_load_process(env, terminal, start_load_event=start_load_event, end_load_event=end_load_event))
    yield end_load_event

    # train departs
    if env.now < train_schedule["departure_time"]:
        yield env.timeout(state.TRAIN_INSPECTION_TIME)
        print(f"Time {env.now}: [EARLY] Train {train_schedule['train_id']} departs from the track {track_id}.")
    elif env.now == train_schedule["departure_time"]:
        yield env.timeout(state.TRAIN_INSPECTION_TIME)
        print(f"Time {env.now}: [In Time] Train {train_schedule['train_id']} departs from the track {track_id}.")
    else:
        delay_time = env.now - train_schedule["departure_time"]
        if f"train_id_{train_id}" not in delay_list:
            delay_list[f"train_id_{train_id}"] = {}
        delay_list[f"train_id_{train_id}"]["departure"] = delay_time
        print(f"Time {env.now}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours from the track {track_id}..")

    yield terminal.tracks.put(track_id)
    print(f"Time {env.now}: Train is departing the terminal.")

    for oc_id in range(state.OC_NUM, state.OC_NUM + train_schedule['oc_number']):
        record_event(f"OC-{oc_id}", 'train_depart', env.now) # loop: assign container_id range(current_oc, current_oc + train_schedule['full_cars'])

    state.time_per_train[train_schedule['train_id']] = env.now - arrival_time

    # Update various parameters to track inbound and outbound containers
    state.IC_NUM = state.IC_NUM + train_schedule['full_cars']
    state.OC_NUM = state.OC_NUM + train_schedule['oc_number']

    # Trigger the departure event for the current train
    next_departed_event.succeed()   # the terminal now is clear and ready to accept the next train

def run_simulation(train_consist_plan: pl.DataFrame, terminal: str, out_path = None):
    global state
    state.terminal = terminal
    state.train_consist_plan = train_consist_plan
    state.initialize()

    # Create environment
    env = simpy.Environment()

    # # Train timetable
    # Truck_numbers is equal to max(full_cars, oc_numbers), to make sure each container has one truck to pick up
    # train_timetable = [
    #     {"train_id": 19, "arrival_time": 187, "departure_time": 250, "empty_cars": 3, "full_cars": 50, "oc_number": 20,
    #      "truck_number": 50},    # test: ic > oc
    #     {"train_id": 12, "arrival_time": 300, "departure_time": 500, "empty_cars": 5, "full_cars": 40, "oc_number": 40,
    #      "truck_number": 40},    # test: ic = oc
    #     {"train_id": 70, "arrival_time": 530, "departure_time": 800, "empty_cars": 5, "full_cars": 30, "oc_number": 50,
    #      "truck_number": 50},    # test: ic < oc
    # ]

    train_timetable = [
        {"train_id": 19, "arrival_time": 187, "departure_time": 250, "empty_cars": 3, "full_cars": 500, "oc_number": 200,
         "truck_number": 500},    # test: ic > oc
        {"train_id": 12, "arrival_time": 300, "departure_time": 500, "empty_cars": 5, "full_cars": 400, "oc_number": 400,
         "truck_number": 400},    # test: ic = oc
        {"train_id": 70, "arrival_time": 530, "departure_time": 800, "empty_cars": 5, "full_cars": 300, "oc_number": 500,
         "truck_number": 500},    # test: ic < oc
    ]

    # train_timetable = train_timetable

    # Initialize train status in the terminal
    train_departed_event = env.event()
    train_departed_event.succeed()
    
    truck_number = max([entry['truck_number'] for entry in train_timetable])
    chassis_count = max([entry['empty_cars'] + entry['full_cars'] for entry in train_timetable])
    terminal = Terminal(env, truck_capacity = truck_number, chassis_count=chassis_count)

    # Trains arrive according to timetable
    for i, train_schedule in enumerate(train_timetable):
        print(train_schedule)
        next_departed_event = env.event()  # Create a new departure event for the next train
        env.process(process_train_arrival(env, terminal, train_departed_event, train_schedule, next_departed_event))
        train_departed_event = next_departed_event  # Update the departure event for the next train

    # Simulation hyperparameters
    env.run(until = state.sim_time)

    # Performance Matrix
    # Train processing time
    avg_time_per_train = sum(state.time_per_train.values()) / len(state.time_per_train)
    print(f"Average train processing time: {sum(state.time_per_train) / len(state.time_per_train) if state.time_per_train else 0:.2f}")
    print("Simulation completed. ")
    with open("avg_time_per_train.txt", "w") as f:
       f.write(str(avg_time_per_train))

    # Create DataFrame for container events
    print(state.sim_time)

    container_data = (
        pl.from_dicts(
            [dict(event, **{'container_id': container_id}) for container_id, event in state.container_events.items()],
            infer_schema_length = None
        )
        .with_columns(
            pl.when(
                pl.col("truck_exit").is_not_null() & pl.col("train_arrival").is_not_null()
            )
            .then(
                pl.col("truck_exit") - pl.col("train_arrival")
            )
            .when(
                pl.col("train_depart").is_not_null()
            )
            .then(
                # pl.col("crane_load") - pl.col("truck_arrival")
                pl.col("crane_load") - pl.col("hostler_pickup")
            )
            .otherwise(None)
            .alias("container_processing_time")
        )
        .sort("container_id")
        .select(pl.col("container_id"), pl.all().exclude("container_id"))
    )
    if out_path is not None:
       container_data.write_excel(out_path / f"simulation_crane_{state.CRANE_NUMBER}_hostler_{state.HOSTLER_NUMBER}.xlsx")

    # Save vehicle related logs
    save_to_excel(state)


    print("Done!")
    return container_data


if __name__ == "__main__":
    run_simulation(
        train_consist_plan = pl.read_csv(utilities.package_root() / 'demos' / 'starter_demo' / 'train_consist_plan.csv'),
        terminal = "Allouez",
        out_path = utilities.package_root() / 'demos' / 'single_track_results'
    )