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
        self.parking_slots = simpy.Store(env, K)   # todo: parking_slots capacity
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
        print(f"Time {env.now}: [EARLY] Train {train_schedule['train_id']} departs from the track {track_id}.")
        # delay_time = 0
    elif env.now == train_schedule["departure_time"]:
        print(f"Time {env.now}: [In Time] Train {train_schedule['train_id']} departs from the track {track_id}.")
        # delay_time = 0
    else:
        delay_time = env.now - train_schedule["departure_time"]
        print(f"Time {env.now}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours from the track {track_id}.")

    # state.delay_list[train_id] = delay_time
    # total_delay_time = sum(state.delay_list.values())

    # print(f"Total delay time for all trains is {total_delay_time} hours.")

    # return delay_time


def save_vehicle_and_performance_metrics(state, ic_avg_delay, oc_avg_delay):
    out_path = utilities.package_root() / 'demos' / 'single_track_results'

    container_excel_path = out_path / f"{state.CRANE_NUMBER}C-{state.HOSTLER_NUMBER}H_container_throughput_{K}_batch_size_{k}.xlsx"
    vehicle_excel_path = out_path / f"{state.CRANE_NUMBER}C-{state.HOSTLER_NUMBER}H_vehicle_throughput_{K}_batch_size_{k}.xlsx"

    if not container_excel_path.exists():
        print(f"[Error] Container Excel not found: {container_excel_path}")
        return
    if not vehicle_excel_path.exists():
        print(f"[Error] Vehicle Excel not found: {vehicle_excel_path}")
        return

    ic_avg_time, oc_avg_time, total_ic_avg_time, total_oc_avg_time = calculate_container_processing_time(
        container_excel_path,
        train_batch_size=k,
        daily_throughput=K,
        num_trains=math.ceil(K/(k*2)),
        ic_delay_time=ic_avg_delay,
        oc_delay_time=oc_avg_delay
    )

    ic_energy, oc_energy, total_energy = calculate_vehicle_energy(vehicle_excel_path)

    single_run = [ic_avg_time, ic_avg_delay, total_ic_avg_time, oc_avg_time, oc_avg_delay, total_oc_avg_time, ic_energy, oc_energy, total_energy]

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


# def crane_unload_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, total_ic, all_ic_picked, all_ic_unload_event):
#     global state
#     ic_unloaded_count = 0
#
#     for ic_id in range(state.IC_NUM, state.IC_NUM + total_ic):
#         crane_id = yield terminal.cranes.get()
#         # print(f"Time {env.now}: Crane starts unloading IC {ic_id}")
#         crane_unload_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME)
#         yield env.timeout(crane_unload_time)
#         record_event(ic_id, 'crane_unload', env.now)
#
#         # print(f"Time {env.now}: Crane {crane_id} finishes unloading IC {ic_id} onto chassis")
#         emissions = emission_calculation('loaded', 'load', 'crane', crane_id, crane_unload_time)
#         record_vehicle_event('crane', crane_id, f'unload_{ic_id}', 'unloaded', 'load', crane_unload_time, emissions, 'end', env.now)
#
#         terminal.chassis.put(ic_id)
#         terminal.cranes.put(crane_id)
#
#         ic_unloaded_count += 1
#         env.process(container_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, all_ic_unload_event, all_ic_picked))
#
#         if ic_unloaded_count == total_ic:
#             all_ic_unload_event.succeed()


def crane_unload_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, total_ic, all_ic_picked, all_ic_unload_event):
    global state

    ic_store = simpy.Store(env)
    for ic_id in range(state.IC_NUM, state.IC_NUM + total_ic):
        ic_store.put(ic_id)

    def unload_ic(env, ic_id):
        crane_id = yield terminal.cranes.get()

        crane_unload_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME)
        yield env.timeout(crane_unload_time)

        record_event(ic_id, 'crane_unload', env.now)
        emissions = emission_calculation('loaded', 'load', 'crane', crane_id, crane_unload_time)
        record_vehicle_event('crane', crane_id, f'unload_{ic_id}', 'unloaded', 'load', crane_unload_time, emissions, 'end', env.now)

        terminal.chassis.put(ic_id)
        terminal.cranes.put(crane_id)
        env.process(container_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, all_ic_unload_event, all_ic_picked))

        if len(ic_store.items) == 0 and not all_ic_unload_event.triggered:
            all_ic_unload_event.succeed()

    # start process for every ic
    while len(ic_store.items) > 0:
        ic_id = yield ic_store.get()
        env.process(unload_ic(env, ic_id))


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

    # Hostler picks up IC (load locking) from chassis
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
    if sum(str(item).isdigit() for item in terminal.chassis.items) == 0 and all_ic_unload_event.triggered and not all_ic_picked.triggered:
        all_ic_picked.succeed()
        # print(f"Time {env.now}: All ICs for train-{train_schedule['train_id']} are picked up by hostlers.")

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
        hostler_travel_time_to_parking = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
        yield env.timeout(hostler_travel_time_to_parking)
        # print(f"Time {env.now}: Hostler {hostler_id} dropped off {oc} onto chassis")
        yield terminal.chassis.put(oc)
        record_event(oc, 'hostler_dropoff', env.now)

        # Hostler drops off OC
        hostler_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
        record_event(ic_id, 'hostler_unloaded', env.now)
        emissions = emission_calculation('loaded', 'load', 'hostler', hostler_id, hostler_unload_time)
        record_vehicle_event('hostler', hostler_id, f'drops off OC-{oc}', 'loaded', 'load', hostler_travel_time_to_parking, emissions, 'end', env.now)

    # Hostler going back to resource parking
    current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    hostler_travel_time_to_parking = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
    emissions = emission_calculation('empty', 'trip','hostler', hostler_id, hostler_travel_time_to_parking)
    yield env.timeout(hostler_travel_time_to_parking)  # update: time calculated by speed-density function
    # print(f"Time {env.now}: Hostler {hostler_id} return to parking slot.")
    yield terminal.hostlers.put(hostler_id)
    record_vehicle_event('hostler', hostler_id, f'back_parking', 'empty', 'trip', hostler_travel_time_to_parking, emissions,'end', env.now)

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
            record_vehicle_event('hostler', hostler_id, f'pick up OC-{oc}', 'empty', 'travel', hostler_travel_time_to_parking, emissions, 'end', env.now)

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

    # if sum(1 for item in terminal.chassis.items if isinstance(item, str) and "OC-" in str(item)) == oc_needed:
    if (sum(1 for item in terminal.chassis.items if isinstance(item, str) and "OC-" in str(item)) == oc_needed
    and not all_oc_prepared.triggered):
        all_oc_prepared.succeed()
        print(f"Time {env.now}: All OCs are ready on chassis.")


def truck_exit(env, terminal, truck_id, ic_id):
    global state
    with terminal.out_gates.request() as out_gate_request:
        yield out_gate_request
        # print(f"Time {env.now}: Truck {truck_id} with IC {ic_id} is passing through the out-gate and leaving the terminal")
        truck_pass_time = state.TRUCK_OUTGATE_TIME + random.uniform(0, state.TRUCK_OUTGATE_TIME_DEV)
        yield env.timeout(truck_pass_time)
        record_event(ic_id, 'truck_exit', env.now)
    yield terminal.truck_store.put(truck_id)


# def crane_load_process(env, terminal, start_load_event, end_load_event):
#     global state
#     yield start_load_event
#     # print(f"Time {env.now}: Starting loading process onto the train.")
#
#     while len([item for item in terminal.chassis.items if isinstance(item, str) and "OC" in item]) > 0:  # if there still has OC on chassis
#         crane_id = yield terminal.cranes.get()
#         oc = yield terminal.chassis.get(lambda x: isinstance(x, str) and "OC" in x)  # obtain an OC from chassis
#         # print(f"Time {env.now}: Crane {crane_id} starts loading {oc} onto the train.")
#         crane_load_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME)
#         yield env.timeout(crane_load_time)
#         record_event(oc, 'crane_load', env.now)
#         # print(f"Time {env.now}: Crane finished loading {oc} onto the train.")
#         terminal.cranes.put(crane_id)
#         emissions = emission_calculation('loaded', 'load', 'crane', crane_id, crane_load_time)
#         record_vehicle_event('crane', crane_id, f'load_{oc}', 'loaded', 'load', crane_load_time, emissions, 'end', env.now)
#
#     # print(f"Time {env.now}: All OCs loaded. Train is fully loaded and ready to depart.")
#     # print("Containers on chassis (after loading OCs):", terminal.chassis.items)
#     end_load_event.succeed()

def crane_load_process(env, terminal, start_load_event, end_load_event):
    global state
    yield start_load_event

    def load_oc(env):
        while True:
            if not any(isinstance(item, str) and "OC" in item for item in terminal.chassis.items):
                if not end_load_event.triggered:
                    end_load_event.succeed()
                break

            crane_id = yield terminal.cranes.get()
            oc = yield terminal.chassis.get(lambda x: isinstance(x, str) and "OC" in x)

            crane_load_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME)
            yield env.timeout(crane_load_time)

            record_event(oc, 'crane_load', env.now)
            emissions = emission_calculation('loaded', 'load', 'crane', crane_id, crane_load_time)
            record_vehicle_event('crane', crane_id, f'load_{oc}', 'loaded', 'load', crane_load_time, emissions, 'end', env.now)

            # release crane
            terminal.cranes.put(crane_id)

    num_cranes = len(terminal.cranes.items)  # 如果 terminal.cranes 是 Store
    for _ in range(num_cranes):
        env.process(load_oc(env))


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
        # print(f"Time {env.now}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours, waiting to be assigned to the track {track_id}.")
    state.train_delay_time[train_schedule['train_id']] = delay_time

    for ic_id in range(state.IC_NUM, state.IC_NUM + train_schedule['full_cars']):
        record_event(ic_id, 'train_arrival_expected', train_schedule['arrival_time'])
        record_event(ic_id, 'train_arrival', env.now)  # loop: assign container_id range(current_ic, current_ic + train_schedule['full_cars'])

    # crane unloading IC
    env.process(crane_unload_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, total_ic, all_ic_picked, all_ic_unload_event))

    # prepare all OC and pick up all IC before crane loading
    yield all_ic_picked & all_oc_prepared
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
        record_event(f"OC-{oc_id}", 'train_depart_expected', train_schedule['departure_time'])
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

    # Simulation duration: time includes warm-up and cool down
    with open("sim_config.json", "r") as f:
        sim_config = json.load(f)
    total_simulation_length = sim_config["vehicles"]["simulation_duration"]
    env.run(until=total_simulation_length)

    # Data analysis
    # ==== 1. container log dataframe ====
    import pandas as pd
    container_records = []
    for container_id, event in state.container_events.items():
        record = {
            "container_id": container_id,
            "train_arrival": event.get("train_arrival"),
            "train_arrival_expected": event.get("train_arrival_expected"),
            "crane_unload": event.get("crane_unload"),
            "hostler_loaded": event.get("hostler_loaded"),
            "hostler_pickup": event.get("hostler_pickup"),
            "truck_pickup": event.get("truck_pickup"),
            "hostler_unloaded": event.get("hostler_unloaded"),
            "truck_exit": event.get("truck_exit"),
            "truck_arrival": event.get("truck_arrival"),
            "truck_dropoff": event.get("truck_dropoff"),
            "hostler_dropoff": event.get("hostler_dropoff"),
            "crane_load": event.get("crane_load"),
            "train_depart": event.get("train_depart"),
            "train_depart_expected": event.get("train_depart_expected"),
        }
        container_records.append(record)

    container_data = pd.DataFrame(container_records)

    # ==== 2. Label container type ====
    container_data["type"] = container_data["container_id"].astype(str).apply(
        lambda x: "OC" if x.startswith("OC-") else "IC" if x.isdigit() else "Unknown")

    # ==== 3. Add container processing time ====
    def compute_processing_time(row):
        if pd.notnull(row["truck_exit"]) and pd.notnull(row["crane_unload"]):
            return row["truck_exit"] - row["train_arrival"]  # IC
        elif pd.notnull(row["train_depart"]) and pd.notnull(row["hostler_pickup"]):
            return row["train_depart"] - row["hostler_pickup"]  # OC
        else:
            return None

    container_data["container_processing_time"] = container_data.apply(compute_processing_time, axis=1)

    # # ==== 4. Add full vs. remainder ====
    # config_path = "/Users/qianqiantong/PycharmProjects/altrios-private/altrios/python/altrios/lifts/sim_config.json"
    # with open(config_path, 'r') as f:
    #     config = json.load(f)
    #
    # daily_throughput = config['layout']['K']
    # train_batch_size = config['layout']['k']
    # simulation_hours = config['vehicles']['simulation_duration']
    # simulation_days = simulation_hours / 24
    #
    # num_ic = int(daily_throughput / 2)
    # num_trains = math.ceil(num_ic / train_batch_size)
    #
    # raw_ids = container_data["container_id"].astype(str)
    # id_nums = raw_ids.apply(lambda x: int(x.split("-")[-1]) if "-" in x else int(x))
    #
    # containers = int(daily_throughput / 2)
    # relative_id = id_nums % containers
    # bound_id = train_batch_size * (num_trains - 1)
    #
    # container_data["location"] = ["full" if rid <= bound_id else "remainder" for rid in relative_id]

    # ==== 4. Add first_oc_pickup_time for OC containers ====

    # 确保 container_id 是字符串且非空
    container_data = container_data[container_data["container_id"].notna()].copy()
    container_data["container_id"] = container_data["container_id"].astype(str)

    # 筛选 OC container
    df_oc = container_data[container_data['container_id'].str.startswith("OC-")].copy()

    # 找出每个 train_depart 中最早的 hostler_pickup 时间
    first_pickup_per_train = (
        df_oc.groupby("train_depart")["hostler_pickup"]
        .min()
        .reset_index()
        .rename(columns={"hostler_pickup": "first_oc_pickup_time"})
    )

    # 合并 first_oc_pickup_time 回 OC 数据
    df_oc = df_oc.merge(first_pickup_per_train, on="train_depart", how="left")

    # 再将该列合并回 container_data
    container_data = container_data.merge(
        df_oc[["container_id", "first_oc_pickup_time"]],
        on="container_id",
        how="left"
    )

    # ==== 5. Sort by numeric ID ====
    def extract_numeric_id(cid):
        digits = ''.join(filter(str.isdigit, str(cid)))
        return int(digits) if digits else -1

    container_data["container_id_numeric"] = container_data["container_id"].apply(extract_numeric_id)
    container_data = container_data.sort_values("container_id_numeric").drop(columns=["container_id_numeric"])

    # ==== 6. Save to excel ====
    if out_path is not None:
        container_data.to_excel(out_path / f"{state.CRANE_NUMBER}C-{state.HOSTLER_NUMBER}H_container_throughput_{K}_batch_size_{k}.xlsx", index=False)
    save_energy_to_excel(state)

    # ==== 7. Delay Calculations ====
    # Processing time for IC => avg processing time + delay time
    ic_df = container_data[container_data["type"] == "IC"].copy()
    ic_df["ic_delay_time"] = ic_df["train_arrival"] - ic_df["train_arrival_expected"]
    ic_avg_delay = ic_df["ic_delay_time"].mean()

    oc_df = container_data[container_data["type"] == "OC"].copy()
    oc_df["oc_delay_time"] = oc_df["train_depart"] - oc_df["first_oc_pickup_time"]
    oc_avg_delay = oc_df["oc_delay_time"].mean()

    single_run = save_vehicle_and_performance_metrics(state, ic_avg_delay, oc_avg_delay)
    print("Done!")

    return single_run


if __name__ == "__main__":
    single_run = run_simulation(
        train_consist_plan=pl.read_csv(utilities.package_root() / 'demos' / 'starter_demo' / 'train_consist_plan.csv'),
        terminal="Allouez",
        out_path=utilities.package_root() / 'demos' / 'single_track_results'
    )

    # Performance Matrix
    output = {
        "single_run": single_run,
        "ic_avg_time": single_run[0],
        "ic_avg_delay": single_run[1],
        "total_ic_avg_time": single_run[2],
        "oc_avg_time": single_run[3],
        "oc_avg_delay": single_run[4],
        "total_oc_avg_time": single_run[5],
        "ic_energy": single_run[6],
        "oc_energy": single_run[7],
        "total_energy": single_run[8]
    }

    with open("performance_matrix.json", "w") as f:
        json.dump(output, f)