import simpy
import random
import json
import polars as pl
from altrios.lifts import utilities
# from altrios.lifts.main_single_track_npf import single_run
from altrios.lifts.single_track_parameters import *
from altrios.lifts.distances_single_track import *
from altrios.lifts.dictionary import *
from altrios.lifts.schedule import *
from altrios.lifts.single_track_vehicle_performance import *
import altrios.lifts.distances_single_track as layout
from altrios.lifts.single_track_vehicle_performance import vehicle_events

K, k, M, N, n_t, n_p, n_r= layout.load_layout_config_from_json()

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
        self.parking_slots = simpy.Store(env)   # todo: parking_slots capacity
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

        # hostler
        num_diesel = round(state.HOSTLER_NUMBER * state.HOSTLER_DIESEL_PERCENTAGE)
        num_electric = state.HOSTLER_NUMBER - num_diesel
        hostlers = [(i, "diesel") for i in range(num_diesel)] + \
                   [(i + num_diesel, "electric") for i in range(num_electric)]
        for hostler_id in hostlers:
            self.hostlers.put(hostler_id)
        # print(f"hostler {hostlers}")


def record_event(container_id, event_type, timestamp):
    global state
    if container_id not in state.container_events:
        state.container_events[container_id] = {}
    state.container_events[container_id][event_type] = timestamp

def handle_train_departure(env, train_schedule, train_id, track_id):
    global state

    if env.now < train_schedule["departure_time"]:
        # print(f"Time {env.now}: [EARLY] Train {train_schedule['train_id']} departs from the track {track_id}.")
        delay_time = 0
    elif env.now == train_schedule["departure_time"]:
        # print(f"Time {env.now}: [In Time] Train {train_schedule['train_id']} departs from the track {track_id}.")
        delay_time = 0
    else:
        delay_time = env.now - train_schedule["departure_time"]
        # print(f"Time {env.now}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours from the track {track_id}.")

    state.delay_list[train_id] = delay_time

    total_delay_time = sum(state.delay_list.values())

    print(f"Total delay time for all trains is {total_delay_time} hours.")

    return delay_time


def save_vehicle_and_performance_metrics(state, average_container_delay_time):
    out_path = utilities.package_root() / 'demos' / 'single_track_results'

    container_excel_path = out_path / f"container_throughput_{K}_batch_size_{k}.xlsx"
    vehicle_excel_path = out_path / f"vehicle_throughput_{K}_batch_size_{k}.xlsx"

    if not container_excel_path.exists():
        print(f"[Error] Container Excel not found: {container_excel_path}")
        return
    if not vehicle_excel_path.exists():
        print(f"[Error] Vehicle Excel not found: {vehicle_excel_path}")
        return

    ic_time, oc_time, total_time = calculate_container_processing_time(
        container_excel_path,
        train_batch_size=k,
        daily_throughput=K,
        num_trains=math.ceil(K/(k*2)),
    )

    ic_energy, oc_energy, total_energy = calculate_vehicle_energy(vehicle_excel_path)

    single_run = [average_container_delay_time, ic_time, oc_time, total_time, ic_energy, oc_energy, total_energy]

    return single_run


def emission_calculation(status, move, vehicle, id, travel_time):
    global state
    vehicle = vehicle.capitalize()
    vehicle_type = id[1]
    vehicle_type = vehicle_type.capitalize()
    if status == 'loaded' and move == 'load':
        emission_unit = state.FULL_LOAD_EMISSIONS_RATES[vehicle][vehicle_type]
        emissions = emission_unit
    elif status == 'empty' and move == 'load':
        emission_unit = state.IDLE_LOAD_EMISSIONS_RATES[vehicle][vehicle_type]
        emissions = emission_unit
    elif status == 'loaded' and move == 'trip':
        emission_unit = state.FULL_TRIP_EMISSIONS_RATES[vehicle][vehicle_type]
        emissions = emission_unit * travel_time
    elif status == 'empty' and move == 'trip':
        emission_unit = state.IDLE_TRIP_EMISSIONS_RATES[vehicle][vehicle_type]
        emissions = emission_unit * travel_time

    return emissions


def truck_entry(env, train_schedule, terminal, truck_id):
    global state
    with terminal.in_gates.request() as gate_request:
        yield gate_request
        # print(f"Time {env.now}: Truck {truck_id} passed the in-gate and is entering the terminal")

        oc_id = yield terminal.oc_store.get()
        record_event(oc_id, 'truck_arrival', env.now)
        # Calculate truck speed according to the current density
        truck_travel_time = simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
        emissions = emission_calculation('loaded', 'trip', 'truck', truck_id, truck_travel_time)
        record_vehicle_event('truck', truck_id, f'drop off {oc_id}', 'loaded', 'trip', truck_travel_time, emissions, 'start', env.now)

        truck_unload_time = 1/60 + random.uniform(0, 1/600)
        emissions = emission_calculation('loaded', 'load', 'truck', truck_id, truck_unload_time)
        record_vehicle_event('truck', truck_id, f'drop off {oc_id}', 'loaded', 'load', truck_unload_time, emissions, 'end', env.now)
        # print(f"Time {env.now}: Truck {truck_id} unloaded {oc_id} at parking slot.")

        # Truck resource parking
        terminal.parking_slots.put(oc_id)
        record_event(oc_id, 'truck_dropoff', env.now)
        emissions = emission_calculation('empty', 'trip', 'truck', truck_id, truck_travel_time)
        record_vehicle_event('truck', truck_id, f'trip {oc_id}', 'empty', 'trip', truck_travel_time, emissions, 'end', env.now)


def empty_truck(env, train_schedule, terminal, truck_id):
    global state
    with terminal.in_gates.request() as gate_request:
        yield gate_request
        # print(f"Time {env.now}: Truck {truck_id} passed the in-gate with empty loading")
        # Note the arrival of empty trucks will not be recorded due to excel output dimensions
        truck_travel_time = simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min,
                                                  d_t_max)
        emissions = emission_calculation('empty', 'trip','truck', truck_id, truck_travel_time)
        record_vehicle_event('truck', truck_id, f'entry', 'empty', 'trip', truck_travel_time, emissions, 'end', env.now)


def truck_arrival(env, terminal, train_schedule, all_trucks_arrived_event):
    global state
    truck_number = train_schedule["truck_number"]
    num_diesel = round(truck_number * state.TRUCK_DIESEL_PERCENTAGE)
    num_electric = truck_number - num_diesel

    trucks = [(i, "diesel") for i in range(num_diesel)] + \
             [(i + num_diesel, "electric") for i in range(num_electric)]

    # print(f"truck platoon has total {len(trucks)} including {trucks}")
    # print(f"oc store has {terminal.oc_store.items}")

    for truck_id in trucks:
        yield env.timeout(0)  # Assume truck arrives not impact on system, not random.expovariate(arrival_rate)
        terminal.truck_store.put(truck_id)
        if len(terminal.oc_store.items) != 0:  # Truck move OC from outside (oc_store) to terminal (parking_slots)
            env.process(truck_entry(env, train_schedule, terminal, truck_id))
        else:
            env.process(empty_truck(env, train_schedule, terminal, truck_id))

    all_trucks_arrived_event.succeed()  # if all_trucks_arrived_event is triggered, train is allowed to enter


def crane_unload_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, total_ic, all_ic_picked,
                         all_ic_unload_event):
    global state
    ic_unloaded_count = 0

    for ic_id in range(state.IC_NUM, state.IC_NUM + total_ic):
        crane_id = yield terminal.cranes.get()
        # print(f"Time {env.now}: Crane starts unloading IC {ic_id}")
        crane_unload_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME)
        yield env.timeout(crane_unload_time)
        record_event(ic_id, 'crane_unload', env.now)

        # print(f"Time {env.now}: Crane {crane_id} finishes unloading IC {ic_id} onto chassis")
        emissions = emission_calculation('loaded', 'load', 'crane', crane_id, crane_unload_time)
        record_vehicle_event('crane', crane_id, f'unload_{ic_id}', 'unloaded', 'load', crane_unload_time, emissions, 'end', env.now)

        terminal.chassis.put(ic_id)
        terminal.cranes.put(crane_id)

        ic_unloaded_count += 1
        env.process(container_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, all_ic_unload_event, all_ic_picked))

        if ic_unloaded_count == total_ic:
            all_ic_unload_event.succeed()


def container_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, all_ic_unload_event, all_ic_picked):
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

    # Hostler picks up IC from chassis
    ic_id = yield terminal.chassis.get(lambda x: isinstance(x, int))
    hostler_load_time = 1/60 + random.uniform(0, 1/600)
    record_event(ic_id, 'hostler_loaded', env.now)
    emissions = emission_calculation('loaded', 'load','hostler', hostler_id, hostler_load_time)
    record_vehicle_event('hostler', hostler_id, f'pick up IC-{ic_id}', 'loaded', 'load', hostler_load_time,
                         emissions, 'start', env.now)

    # Hostler taking IC to parking slots
    current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    hostler_travel_time_to_parking = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
    yield env.timeout(hostler_travel_time_to_parking)
    # print(f"Time {env.now}: Hostler {hostler_id} picked up IC {ic_id} and is heading to parking slot.")
    record_event(ic_id, 'hostler_pickup', env.now)
    emissions = emission_calculation('empty', 'trip','hostler', hostler_id, hostler_travel_time_to_parking)
    record_vehicle_event('hostler', hostler_id, f'pick up IC-{ic_id}', 'empty', 'trip', hostler_travel_time_to_parking,
                         emissions, 'doing', env.now)

    # Hostler drops off IC to the closest parking lot
    hostler_unload_time = 1/60 + random.uniform(0, 1/600)
    record_event(ic_id, 'hostler_loaded', env.now)
    emissions = emission_calculation('loaded', 'load','hostler', hostler_id, hostler_unload_time)
    record_vehicle_event('hostler', hostler_id, f'drop off IC-{ic_id}', 'loaded', 'load', hostler_travel_time_to_parking,
                         emissions, 'end', env.now)

    # Prepare for crane loading: if chassis has no IC AND all_ic_picked (parking side) is not triggered => trigger all_ic_picked
    if sum(str(item).isdigit() for item in terminal.chassis.items) == 0 and all_ic_unload_event.triggered:
        all_ic_picked.succeed()
        # print(f"Time {env.now}: All ICs are picked up by hostlers.")

    # Assign a truck to pick up IC
    truck_id = yield terminal.truck_store.get()
    # print(f"Time {env.now}: Truck {truck_id} is assigned to IC {ic_id} for exit.")

    # Truck going to pick up IC
    record_event(ic_id, 'truck_pickup', env.now)
    truck_travel_time = simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
    yield env.timeout(truck_travel_time)
    emissions = emission_calculation('empty', 'trip', 'truck', truck_id, truck_travel_time)
    record_vehicle_event('truck', truck_id, f'pickup_IC_{ic_id}', 'empty', 'trip', truck_travel_time, emissions, 'start', env.now)

    # Truck picks up IC
    truck_load_time = 1/60 + random.uniform(0, 1/600)
    emissions = emission_calculation('loaded', 'load', 'truck', truck_id, truck_load_time)
    record_vehicle_event('truck', truck_id, f'pickup_IC_{ic_id}', 'loaded', 'load', truck_load_time, emissions, 'doing',
                         env.now)

    # Truck going to leave the gate
    truck_travel_time = simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
    yield env.timeout(truck_travel_time)
    emissions = emission_calculation('loaded', 'trip', 'truck', truck_id, truck_travel_time)
    record_vehicle_event('truck', truck_id, f'pickup_IC_{ic_id}', 'loaded', 'trip', truck_travel_time, emissions, 'end',
                         env.now)

    # Truck queue and exit the gate
    env.process(truck_exit(env, terminal, truck_id, ic_id))

    # Assign a hostler to pick up an OC, if OC remains (balanced loop remains)
    if len(terminal.parking_slots.items) > 0:
        oc = yield terminal.parking_slots.get()
        # print(f"Time {env.now}: Hostler {hostler_id} is going to pick up {oc}")

        # The hostler going to pick up an OC after picking up an IC
        hostler_reposition_travel_time = simulate_reposition_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
        yield env.timeout(hostler_reposition_travel_time)
        # print(f"Time {env.now}: Hostler {hostler_id} picked up {oc} and is returning to the terminal")
        record_event(oc, 'hostler_pickup', env.now)
        emissions = emission_calculation('empty', 'trip','hostler', hostler_id, hostler_reposition_travel_time)
        record_vehicle_event('hostler', hostler_id, f'pick up {oc}', 'empty', 'trip', hostler_reposition_travel_time, emissions,
                             'end', env.now)

        # Hostler picks up OC
        hostler_load_time = 1 / 60 + random.uniform(0, 1 / 600)
        record_event(ic_id, 'hostler_loaded', env.now)
        emissions = emission_calculation('loaded', 'load', 'hostler', hostler_id, hostler_load_time)
        record_vehicle_event('hostler', hostler_id, f'picks up OC-{oc}', 'loaded', 'load',
                             hostler_travel_time_to_parking, emissions, 'start', env.now)

        # Hostler going to drop off OC
        current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
        hostler_travel_time_to_parking = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length,
                                                                 d_h_min, d_h_max)
        yield env.timeout(hostler_travel_time_to_parking)
        # print(f"Time {env.now}: Hostler {hostler_id} dropped off {oc} onto chassis")
        yield terminal.chassis.put(oc)
        record_event(oc, 'hostler_dropoff', env.now)

        # Hostler drops off OC
        hostler_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
        record_event(ic_id, 'hostler_unloaded', env.now)
        emissions = emission_calculation('loaded', 'load', 'hostler', hostler_id, hostler_unload_time)
        record_vehicle_event('hostler', hostler_id, f'drops off OC-{oc}', 'loaded', 'load',
                             hostler_travel_time_to_parking, emissions, 'end', env.now)

    # Hostler going back to resource parking
    current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    hostler_travel_time_to_parking = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
    emissions = emission_calculation('empty', 'trip','hostler', hostler_id, hostler_travel_time_to_parking)
    yield env.timeout(hostler_travel_time_to_parking)  # update: time calculated by speed-density function
    # print(f"Time {env.now}: Hostler {hostler_id} return to parking slot.")
    yield terminal.hostlers.put(hostler_id)
    record_vehicle_event('hostler', hostler_id, f'back_parking', 'empty', 'trip', hostler_travel_time_to_parking, emissions,
                         'end', env.now)

    # IC < OC: ICs are all picked up and still have OCs remaining
    # if sum(str(item).isdigit() for item in terminal.chassis.items) == 0 and len(terminal.parking_slots.items) != 0:
    if sum(str(item).isdigit() for item in terminal.chassis.items) == 0 and len(terminal.parking_slots.items) == \
            train_schedule['oc_number'] - train_schedule['full_cars']:
        # print(f"ICs are prepared, but OCs remaining: {terminal.parking_slots.items}")
        remaining_oc = len(terminal.parking_slots.items)
        # Repeat hostler picking-up OC process only, until all remaining OCs are transported
        for i in range(1, remaining_oc + 1):
            oc = yield terminal.parking_slots.get()
            # print(f"The OC is {oc}")

            # Hostler going to pick up OC from parking slots
            current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
            hostler_travel_time_to_parking = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length,
                                                                     d_h_min, d_h_max)
            yield env.timeout(hostler_travel_time_to_parking)
            yield terminal.chassis.put(oc)
            record_event(oc, 'hostler_dropoff', env.now)
            record_vehicle_event('hostler', hostler_id, f'pick up OC-{oc}', 'empty', 'travel',
                                 hostler_travel_time_to_parking, emissions, 'end', env.now)

            # Hostler picks up OC
            hostler_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
            record_event(ic_id, 'hostler_unloaded', env.now)
            emissions = emission_calculation('loaded', 'load', 'hostler', hostler_id, hostler_unload_time)
            record_vehicle_event('hostler', hostler_id, f'drops off OC-{oc}', 'loaded', 'load',
                                 hostler_travel_time_to_parking, emissions, 'end', env.now)
            # print(f"Time {env.now}: Hostler {hostler_id} picked up off {oc} from chassis")

            # Hostler going to drop off OC
            current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
            hostler_travel_time_to_parking = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length,
                                                                     d_h_min, d_h_max)
            yield env.timeout(hostler_travel_time_to_parking)
            # print(f"Time {env.now}: Hostler {hostler_id} dropped off {oc} onto chassis")
            yield terminal.chassis.put(oc)
            record_event(oc, 'hostler_dropoff', env.now)

            # Hostler drops off OC to chassis
            hostler_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
            record_event(ic_id, 'hostler_unloaded', env.now)
            emissions = emission_calculation('loaded', 'load', 'hostler', hostler_id, hostler_unload_time)
            record_vehicle_event('hostler', hostler_id, f'drops off OC-{oc}', 'loaded', 'load',
                                 hostler_travel_time_to_parking, emissions, 'end', env.now)
            yield terminal.chassis.put(oc)

            if sum(1 for item in terminal.chassis.items if isinstance(item, str) and "OC-" in str(item)) == oc_needed:
                all_oc_prepared.succeed()
                # print(f"Time {env.now}: All OCs are ready on chassis.")
            i += 1

    if sum(1 for item in terminal.chassis.items if isinstance(item, str) and "OC-" in str(item)) == oc_needed:
        all_oc_prepared.succeed()
        # print(f"Time {env.now}: All OCs are ready on chassis.")


def truck_exit(env, terminal, truck_id, ic_id):
    global state
    with terminal.out_gates.request() as out_gate_request:
        yield out_gate_request
        # print(f"Time {env.now}: Truck {truck_id} with IC {ic_id} is passing through the out-gate and leaving the terminal")
        truck_pass_time = state.TRUCK_OUTGATE_TIME + random.uniform(0, state.TRUCK_OUTGATE_TIME_DEV)
        yield env.timeout(truck_pass_time)
        record_event(ic_id, 'truck_exit', env.now)
    yield terminal.truck_store.put(truck_id)


def crane_load_process(env, terminal, start_load_event, end_load_event):
    global state
    yield start_load_event
    # print(f"Time {env.now}: Starting loading process onto the train.")

    while len([item for item in terminal.chassis.items if isinstance(item, str) and "OC" in item]) > 0:  # if there still has OC on chassis
        crane_id = yield terminal.cranes.get()
        oc = yield terminal.chassis.get(lambda x: isinstance(x, str) and "OC" in x)  # obtain an OC from chassis
        # print(f"Time {env.now}: Crane {crane_id} starts loading {oc} onto the train.")
        crane_load_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME)
        yield env.timeout(crane_load_time)
        record_event(oc, 'crane_load', env.now)
        # print(f"Time {env.now}: Crane finished loading {oc} onto the train.")
        terminal.cranes.put(crane_id)
        emissions = emission_calculation('loaded', 'load', 'crane', crane_id, crane_load_time)
        record_vehicle_event('crane', crane_id, f'load_{oc}', 'loaded', 'load', crane_load_time, emissions, 'end', env.now)

    # print(f"Time {env.now}: All OCs loaded. Train is fully loaded and ready to depart.")
    # print("Containers on chassis (after loading OCs):", terminal.chassis.items)
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
    all_ic_unload_event = env.event()  # condition for all ic picked
    all_ic_picked = env.event()  # condition 1 for crane loading
    all_oc_prepared = env.event()  # condition 2 for crane loading
    start_load_event = env.event()  # condition 1 for train departure
    end_load_event = env.event()  # condition 2 for train departure

    # Initialize dictionary
    delay_list = {}

    oc_id = state.OC_NUM
    # print("start oc_id:", oc_id)
    for oc in range(state.OC_NUM, state.OC_NUM + train_schedule['oc_number']):
        terminal.oc_store.put(f"OC-{oc_id}")
        oc_id += 1
    # print("oc has:", terminal.oc_store.items)

    # All trucks arrive before train arrives
    env.process(truck_arrival(env, terminal, train_schedule, all_trucks_arrived_event))

    # Track assignment for a train
    # print("Current available track has ", terminal.tracks.items)
    track_id = yield terminal.tracks.get()
    if track_id is None:
        # print(f"Time {env.now}: Train {train_id} is waiting for an available track.")
        terminal.waiting_trains.append(train_id)
        return

    # Wait train arriving
    if env.now <= arrival_time:
        yield env.timeout(arrival_time - env.now)
        # print(f"Time {env.now}: [In Time] Train {train_schedule['train_id']} has arrived, waiting to be assigned to the track {track_id}.")
        delay_time = 0
    else:
        delay_time = env.now - arrival_time
        if f"train_id_{train_id}" not in delay_list:
            delay_list[f"train_id_{train_id}"] = {}
        delay_list[f"train_id_{train_id}"]["arrival"] = delay_time
        # print(f"Time {env.now}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours, waiting to be assigned to the track {track_id}.")
    state.train_delay_time[train_schedule['train_id']] = delay_time

    for ic_id in range(state.IC_NUM, state.IC_NUM + train_schedule['full_cars']):
        record_event(ic_id, 'train_arrival', env.now)  # loop: assign container_id range(current_ic, current_ic + train_schedule['full_cars'])

    # crane unloading IC
    env.process(crane_unload_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, total_ic, all_ic_picked, all_ic_unload_event))

    # prepare all OC and pick up all IC before crane loading
    yield all_ic_picked & all_oc_prepared
    # print(f"Time {env.now}: All {oc_needed} OCs are ready on chassis.")
    start_load_event.succeed()  # condition of chassis loading

    # crane loading process
    env.process(crane_load_process(env, terminal, start_load_event=start_load_event, end_load_event=end_load_event))
    yield end_load_event

    # train departs & delay records
    yield env.timeout(state.TRAIN_INSPECTION_TIME)
    handle_train_departure(env, train_schedule, train_id, track_id)

    yield terminal.tracks.put(track_id)
    # print(f"Time {env.now}: Train {train_id} is departing the terminal.")

    for oc_id in range(state.OC_NUM, state.OC_NUM + train_schedule['oc_number']):
        record_event(f"OC-{oc_id}", 'train_depart', env.now)  # loop: assign container_id range(current_oc, current_oc + train_schedule['full_cars'])

    state.time_per_train[train_schedule['train_id']] = env.now - arrival_time

    # Update various parameters to track inbound and outbound containers
    state.IC_NUM = state.IC_NUM + train_schedule['full_cars']
    state.OC_NUM = state.OC_NUM + train_schedule['oc_number']

    # Trigger the departure event for the current train
    next_departed_event.succeed()  # the terminal now is clear and ready to accept the next train


def run_simulation(train_consist_plan: pl.DataFrame, terminal: str, out_path=None):
    global state
    state.terminal = terminal
    state.train_consist_plan = train_consist_plan
    state.initialize()

    random.seed(42)

    # Create environment
    env = simpy.Environment()

    # # Train timetable
    with open("train_timetable.json", "r") as f:
        train_timetable = json.load(f)

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

    container_data = (
        pl.from_dicts(
            [dict(event, **{'container_id': container_id}) for container_id, event in state.container_events.items()],
            infer_schema_length=None
        )
        .with_columns(
            pl.when(
                pl.col("truck_exit").is_not_null() & pl.col("train_arrival").is_not_null()
            )
            .then(
                pl.col("truck_exit") - pl.col("train_arrival")  # IC
            )
            .when(
                pl.col("train_depart").is_not_null()
            )
            .then(
                pl.col("train_depart") - pl.col("hostler_pickup")   # OC
            )
            .otherwise(None)
            .alias("container_processing_time")
        )
        .sort("container_id")
        .select(pl.col("container_id"), pl.all().exclude("container_id"))
    )
    if out_path is not None:
        container_data.write_excel(out_path / f"container_throughput_{K}_batch_size_{k}.xlsx")

    save_energy_to_excel(state)

    total_delay_time = sum(state.delay_list.values())
    num_processed_trains = len(state.delay_list)
    processed_containers = sum(train['full_cars'] + train['oc_number'] for train in train_timetable[:num_processed_trains])
    average_container_delay_time = total_delay_time / processed_containers

    single_run = save_vehicle_and_performance_metrics(state, average_container_delay_time)

    print("Done!")

    return total_delay_time, average_container_delay_time, single_run


if __name__ == "__main__":
    total_delay_time, average_container_delay_time, single_run = run_simulation(
        train_consist_plan=pl.read_csv(utilities.package_root() / 'demos' / 'starter_demo' / 'train_consist_plan.csv'),
        terminal="Allouez",
        out_path=utilities.package_root() / 'demos' / 'single_track_results'
    )
    print(f"Average delay time for each container is {average_container_delay_time} hours.")

    # Performance Matrix
    output = {
        "total_delay_time": total_delay_time,
        "average_container_delay_time": average_container_delay_time,
        "single_run": single_run,
        "ic_avg_time": single_run[1],
        "oc_avg_time": single_run[2],
        "total_avg_time": single_run[3],
        "ic_energy": single_run[4],
        "oc_energy": single_run[5],
        "total_energy": single_run[6]
    }

    with open("performance_matrix.json", "w") as f:
        json.dump(output, f)