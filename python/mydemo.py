import simpy
import random
import polars as pl
from altrios.lifts import utilities
from altrios.lifts.demo_parameters import *
from altrios.lifts.distances import *
from altrios.lifts.dictionary import *
from altrios.lifts.schedule import *
from altrios.lifts.vehicle_performance import record_vehicle_event, save_average_times, save_vehicle_logs


class Terminal:
    def __init__(self, env, truck_capacity, chassis_count):
        self.env = env
        self.tracks = simpy.Store(env, capacity=state.TRACK_NUMBER)
        for track_id in range(1, state.TRACK_NUMBER + 1):
            self.tracks.put(track_id)
        # Example of number of cranes per track
        cranes_per_track = {
            1: 1,
            2: 2
        }
        self.cranes = simpy.FilterStore(env, state.CRANE_NUMBER)

        self.train_ic_stores = {}
        self.train_ic_unload_events = {}

        for item in cranes_per_track.items():
            track_id = item[0]
            cranes_on_this_track = item[1]
            print("hybrid cranes:", cranes_on_this_track)
            # crane_hybrid = round(state.HOSTLER_NUMBER * state.CRANE_HYBRID_PERCENTAGE)
            # crane_electric = state.HOSTLER_NUMBER - crane_hybrid
            for crane_number in range(1, cranes_on_this_track+1):
                self.cranes.put((f'{track_id}-{crane_number}', 'Hybrid', track_id))

        self.in_gates = simpy.Resource(env, state.IN_GATE_NUMBERS)
        self.out_gates = simpy.Resource(env, state.OUT_GATE_NUMBERS)
        self.oc_store = simpy.FilterStore(env)
        self.chassis = simpy.FilterStore(env, capacity=chassis_count)
        self.hostlers = simpy.Store(env, capacity=state.HOSTLER_NUMBER)
        self.truck_store = simpy.Store(env, capacity=truck_capacity)
        # Hostler setup
        hostler_diesel = round(state.HOSTLER_NUMBER * state.HOSTLER_DIESEL_PERCENTAGE)
        hostler_electric = state.HOSTLER_NUMBER - hostler_diesel
        hostlers = [(i, "diesel") for i in range(hostler_diesel)] + \
                   [(i + hostler_diesel, "electric") for i in range(hostler_electric)]
        for hostler_id in hostlers:
            self.hostlers.put(hostler_id)
        print(f"Hostlers: {hostlers}")


def record_container_event(container, event_type, timestamp):
    global state
    if type(container) is str:
        container_string = container
    else:
        container_string = container.to_string()
    
    if container_string not in state.container_events:
        state.container_events[container_string] = {}
    state.container_events[container_string][event_type] = timestamp


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


def truck_entry(env, terminal, truck_id, oc, train_schedule):
    global state
    with terminal.in_gates.request() as gate_request:
        yield gate_request
        if state.log_level == loggingLevel.BASIC:
            print(f"Time {env.now}: Truck {truck_id} passed the in-gate and is entering the terminal")
        truck_travel_time = state.TRUCK_INGATE_TIME + random.uniform(0, state.TRUCK_INGATE_TIME_DEV)
        yield env.timeout(truck_travel_time)  # truck passing gate time: 1 sec (demo_parameters.TRUCK_INGATE_TIME and TRUCK_INGATE_TIME_DEV)
        emissions = emission_calculation('loaded', 'truck', truck_id, truck_travel_time)
        record_vehicle_event('truck', truck_id, f'entry_OC_{oc.id}', 'loaded',emissions, 'end', env.now)

        # Assume each truck takes 1 OC, and drop OC to the closest parking lot according to triangular distribution
        # Assign IDs for OCs
        if state.log_level == loggingLevel.BASIC:
            print(f"Time {env.now}: Truck {truck_id} placed OC {oc.id} at parking slot.")
        record_container_event(oc, 'truck_arrival', env.now)

        # Calculate truck speed according to the current density
        d_t_dist = create_triang_distribution(d_t_min, d_t_avg, d_t_max).rvs()
        # current_veh_num = train_schedule["truck_number"] - len(terminal.truck_store.items)
        # veh_density = current_veh_num / total_lane_length
        # truck_speed = speed_density(veh_density)
        # print("Current truck speed is:", truck_speed)
        # yield env.timeout(d_t_dist / (2 * truck_speed))
        yield env.timeout(d_t_dist / (2 * state.TRUCK_SPEED_LIMIT))
        record_container_event(oc, 'truck_dropoff', env.now)
        emissions = emission_calculation('loaded', 'truck', truck_id, truck_travel_time)
        record_vehicle_event('truck', truck_id, f'dropoff_OC_{oc.id}', 'loaded',
                             emissions, 'end', env.now)


def empty_truck(env, terminal, truck_id):
    global state
    with terminal.in_gates.request() as gate_request:
        yield gate_request
        if state.log_level == loggingLevel.BASIC:
            print(f"Time {env.now}: Truck {truck_id} passed the in-gate and is entering the terminal with empty loading")
        truck_travel_time = state.TRUCK_INGATE_TIME + random.uniform(0, state.TRUCK_INGATE_TIME_DEV)
        yield env.timeout(truck_travel_time)  # truck passing gate time: 1 sec (demo_parameters.TRUCK_INGATE_TIME and TRUCK_INGATE_TIME_DEV)
        # Note the arrival of empty trucks will not be recorded due to excel output dimensions
        emissions = emission_calculation('empty', 'truck', truck_id, truck_travel_time)
        record_vehicle_event('truck', truck_id, 'pass_gate', 'empty',
                             emissions, 'end', env.now)


def truck_arrival(env, terminal, train_schedule, all_trucks_arrived_event):
    global state
    truck_number = train_schedule["truck_number"]
    total_oc = train_schedule["oc_number"]
    num_diesel = round(truck_number * state.TRUCK_DIESEL_PERCENTAGE)
    num_electric = truck_number - num_diesel
    arrival_rate = 1  # truck arrival rate: poisson distribution, arrival rate depends on the gap between last train departure and next train arrival

    if state.log_level == loggingLevel.BASIC:
        print("Truck arrival: state.OC_NUM has", state.OC_NUM)
        print("total_oc", total_oc)

    trucks = [(i, "diesel", f"train-{train_schedule['train_id']}") for i in range(num_diesel)] + \
         [(i + num_diesel, "electric", f"train-{train_schedule['train_id']}") for i in range(num_electric)]

    random.shuffle(trucks)
    if state.log_level == loggingLevel.BASIC:
        print(f"truck is:", trucks)
    
    oc_id = state.OC_NUM
    if state.log_level == loggingLevel.BASIC:
        print("start oc_id:", oc_id)
    for oc in range(state.OC_NUM, state.OC_NUM + total_oc):
        terminal.oc_store.put(
            container(type='Outbound', id=oc_id, train_id=train_schedule['train_id'])
        )
        oc_id += 1
    print(f"Time {env.now}: oc from store has {terminal.oc_store.items}", )

    # for truck_id in range(1, truck_number + 1):
    for truck_id in trucks:
        yield env.timeout(random.expovariate(arrival_rate))  # Assume truck arrives according to the poisson distribution
        terminal.truck_store.put(truck_id)
        if len(terminal.truck_store.items) <= total_oc:
            # if truck_id <= total_oc:
            oc = terminal.oc_store.get(lambda x: x.train_id == train_schedule['train_id'])
            env.process(truck_entry(env, terminal, truck_id, oc.value, train_schedule))
        else:
            env.process(empty_truck(env, terminal, truck_id))

    yield env.timeout(state.TRUCK_TO_PARKING)  # truck travel time of placing OC at parking slot (demo_parameters.TRUCK_TO_PARKING)
    all_trucks_arrived_event.succeed()  # if all_trucks_arrived_event is triggered, train is allowed to enter


def crane_unload_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, total_ic, all_ic_picked,
                         all_ic_unload_event, track_id):
    global state
    ic_unloaded_count = 0

    for ic_id in range(state.IC_NUM, state.IC_NUM + total_ic):
        crane_item = terminal.cranes.get(lambda x: x[2] == track_id)
        print(f"Time {env.now}: Crane {crane_item.value[0]} starts unloading IC-{ic_id}-Train-{train_schedule['train_id']}")
        hostler_to_chassis_travel_time = state.CRANE_UNLOAD_CONTAINER_TIME_MEAN + random.uniform(0,state.CRANE_MOVE_DEV_TIME)
        yield env.timeout(hostler_to_chassis_travel_time)
        record_container_event(f"{ic_id}-Train-{train_schedule['train_id']}", 'crane_unload', env.now)
        record_vehicle_event('crane', (crane_item.value[0],crane_item.value[1]), f"unload_IC_{ic_id}-Train-{train_schedule['train_id']}", 'full',
                             hostler_to_chassis_travel_time * state.FULL_EMISSIONS_RATES['Crane'][crane_item.value[1]], 'end', env.now)
        print(f"Time {env.now}: Crane {crane_item.value[0]} finishes unloading IC-{ic_id}-Train-{train_schedule['train_id']} onto chassis")
        yield terminal.chassis.put(container(type='Inbound', id=ic_id, train_id=train_schedule['train_id']))
        yield terminal.cranes.put(crane_item.value)
        print(f"Time {env.now}: chassis (loading ic):", terminal.chassis.items)
        print(f"Time {env.now}: hostler (loading ic):", terminal.hostlers.items)
        ic_unloaded_count += 1
        env.process(container_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, all_ic_unload_event,
                                all_ic_picked))

        if ic_unloaded_count == total_ic:
            all_ic_unload_event.succeed()
            print(f"Time {env.now}: All ICs have been unloaded onto the chassis.")


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

    # Hostler puts IC to the closest parking lot
    # # TODO: Replace state.HOSTLER_TRANSPORT_CONTAINER_TIME from hostler speed to the current density
    # d_h_dist = create_triang_distribution(d_h_min, d_h_avg, d_h_max).rvs()
    # current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    # veh_density = current_veh_num / total_lane_length
    # hostler_speed = speed_density(veh_density)
    # print("Current hostler speed is:", hostler_speed)
    # yield env.timeout(d_h_dist / (2 * hostler_speed))
    travel_time_to_parking = state.HOSTLER_TRANSPORT_CONTAINER_TIME
    yield env.timeout(travel_time_to_parking)
    # Hostler picks up IC from chassis
    ic = terminal.chassis.get(lambda x: x.type=='Inbound' and x.train_id==train_schedule['train_id']).value
    print(f"    Time {env.now}: Hostler {hostler_id} picked up IC {ic.id}-Train-{train_schedule['train_id']} and is heading to parking slot.")
    print(f"Chassis: {terminal.chassis.items}")
    record_container_event(ic, 'hostler_pickup', env.now)
    emissions = emission_calculation('empty', 'hostler', hostler_id, travel_time_to_parking)
    record_vehicle_event('hostler', hostler_id, f"pickup_IC_{ic.id}-Train-{train_schedule['train_id']}", 'empty',
                         emissions, 'end', env.now)

    # Prepare for crane loading: if chassis has no IC AND all_ic_picked (parking side) is not triggered => trigger all_ic_picked
    if sum(item.type=='Inbound' for item in terminal.train_ic_stores[train_schedule['train_id']].items) == 0 and all_ic_unload_event.triggered:
        all_ic_picked.succeed()
        print(f"Time {env.now}: All ICs are picked up by hostlers.")

    # Test: status for chassis
    print("Containers on chassis (hostler picking-up IC):", terminal.chassis.items)
    print("# of IC on chassis:", sum(str(item).isdigit() for item in terminal.chassis.items))

    # Hostler drop off IC to parking slot
    # # TODO: Replace state.HOSTLER_TRANSPORT_CONTAINER_TIME from hostler speed to the current density
    # d_h_dist = create_triang_distribution(d_h_min, d_h_avg, d_h_max).rvs()
    # current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    # veh_density = current_veh_num / total_lane_length
    # hostler_speed = speed_density(veh_density)
    # print("Current hostler speed is:", hostler_speed)
    # yield env.timeout(d_h_dist / (2 * hostler_speed))
    travel_time_to_parking = state.HOSTLER_TRANSPORT_CONTAINER_TIME
    yield env.timeout(travel_time_to_parking)  # update: time calculated by density-travel_time function
    print(f"Time {env.now}: Hostler {hostler_id} dropped off IC {ic.id}-Train-{train_schedule['train_id']} at parking slot.")
    record_container_event(ic, 'hostler_dropoff', env.now)
    emissions = emission_calculation('loaded', 'hostler', hostler_id, travel_time_to_parking)
    record_vehicle_event('hostler', hostler_id, f"dropoff_IC_{ic.id}-Train-{train_schedule['train_id']}", 'loaded',
                         emissions, 'end', env.now)

    # Assign a truck to pick up IC
    truck_id = yield terminal.truck_store.get()
    print(f"Time {env.now}: Truck {truck_id} is assigned to IC {ic.id}-Train-{train_schedule['train_id']} for exit.")
    record_container_event(ic, 'truck_pickup', env.now)
    emissions = emission_calculation('empty', 'truck', truck_id, travel_time_to_parking)
    record_vehicle_event('truck', truck_id, f"pickup_IC_{ic.id}-Train-{train_schedule['train_id']}", 'empty',
                         emissions, 'end', env.now)

    # Truck queue and exit the gate
    env.process(truck_exit(env, terminal, truck_id, ic, train_schedule))

    # Assign a hostler to pick up an OC
    if len(terminal.oc_store.items) > 0:
        oc = yield terminal.oc_store.get()
        print(f"Time {env.now}: Hostler {hostler_id} is going to pick up OC {oc}")

        # Test: OC remaining before hostlers pick up OCs
        print(f"Hostlers: {terminal.hostlers.items}")
        print(f"OC remains (oc_store): {terminal.oc_store.items}")

        # The hostler picks up an OC
        # # TODO: Replace state.HOSTLER_FIND_CONTAINER_TIME from d_r / hostler speed (using the current density)
        # d_r_dist = create_triang_distribution(d_r_min, d_r_avg, d_r_max).rvs()
        # current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
        # veh_density = current_veh_num / total_lane_length
        # hostler_speed = speed_density(veh_density)
        # print("Current hostler speed is:", hostler_speed)
        # yield env.timeout(d_r_dist / (2 * hostler_speed))
        travel_time_to_oc = state.HOSTLER_FIND_CONTAINER_TIME
        yield env.timeout(travel_time_to_oc)
        print(f"Time {env.now}: Hostler {hostler_id} picked up OC {oc} and is returning to the terminal")
        record_container_event(oc, 'hostler_pickup', env.now)
        emissions = emission_calculation('empty', 'hostler', hostler_id, travel_time_to_oc)
        record_vehicle_event('hostler', hostler_id, f'pickup_OC_{oc}', 'empty',
                             emissions, 'end', env.now)

        # Test: Containers after hostler picking-up OC
        print("Containers on chassis (after hostler picking-up OC):", terminal.chassis.items)
        print("# of IC", sum(str(item).isdigit() for item in terminal.chassis.items))

        # The hostler drops off OC at the chassis
        # # TODO: Replace state.HOSTLER_TRANSPORT_CONTAINER_TIME from hostler speed to the current density
        # d_h_dist = create_triang_distribution(d_h_min, d_h_avg, d_h_max).rvs()
        # current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
        # veh_density = current_veh_num / total_lane_length
        # hostler_speed = speed_density(veh_density)
        # print("Current hostler speed is:", hostler_speed)
        # yield env.timeout(d_h_dist / (2 * hostler_speed))
        travel_time_to_chassis = state.HOSTLER_TRANSPORT_CONTAINER_TIME
        yield env.timeout(travel_time_to_chassis)
        print(f"Time {env.now}: Hostler {hostler_id} dropped off OC {oc} onto chassis")
        yield terminal.chassis.put(oc)
        record_container_event(oc, 'hostler_dropoff', env.now)
        record_vehicle_event('hostler', hostler_id, f'dropoff_OC_{oc}', 'loaded', travel_time_to_parking * 10, 'end',
                             env.now)

        # The hostler-truck-hostler process keeps going, until conditions are satisfied and then further trigger crane movement.
        # ICs are all picked up and OCs are prepared
        if sum(1 for item in terminal.chassis.items if "OC-" in str(item)) == oc_needed:
            all_oc_prepared.succeed()
            print("chassis (check if all oc prepared):", terminal.chassis.items)
            print(f"hostler (check if all oc prepared): {terminal.hostlers.items}")
            print("# of OC on chassis:", sum(1 for item in terminal.chassis.items if "OC-" in str(item)))
            print(f"Time {env.now}: All OCs are ready on chassis.")

    # # TODO: Replace state.HOSTLER_TRANSPORT_CONTAINER_TIME from hostler speed to the current density
    # d_h_dist = create_triang_distribution(d_h_min, d_h_avg, d_h_max).rvs()
    # current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    # veh_density = current_veh_num / total_lane_length
    # hostler_speed = speed_density(veh_density)
    # print("Current hostler speed is:", hostler_speed)
    # yield env.timeout(d_h_dist / (2 * hostler_speed))
    travel_time_to_parking = state.HOSTLER_TRANSPORT_CONTAINER_TIME
    yield env.timeout(travel_time_to_parking)  # update: time calculated by density-travel_time function
    print(f"Time {env.now}: Hostler {hostler_id} return to parking slot.")
    yield terminal.hostlers.put(hostler_id)
    record_vehicle_event('hostler', hostler_id, f'back_parking', 'empty', travel_time_to_parking * 8, 'end', env.now)

    # Test: check if all OCs on chassis
    print("chassis (all oc prepared):", terminal.chassis.items)
    print("# of IC on chassis:", sum(str(item).isdigit() for item in terminal.chassis.items))
    print(f"hostler (all oc prepared): {terminal.hostlers.items}")

    # IC < OC: ICs are all picked up and still have OCs remaining
    print("# of OCs in oc_store:", len(terminal.oc_store.items))
    # print("Containers gap:", train_schedule['oc_number'] - train_schedule['full_cars'])
    if sum(str(item).isdigit() for item in terminal.chassis.items) == 0 and len(terminal.oc_store.items) == \
            train_schedule['oc_number'] - train_schedule['full_cars']:
        print(f"ICs are prepared, but OCs remaining: {terminal.oc_store.items}")
        remaining_oc = len(terminal.oc_store.items)
        # Repeat hostler picking-up OC process only, until all remaining OCs are transported
        for i in range(1, remaining_oc + 1):
            oc = yield terminal.oc_store.get()
            print(f"The OC is {oc}")
            # # TODO: Replace state.HOSTLER_FIND_CONTAINER_TIME from d_r / hostler speed (using the current density)
            # d_r_dist = create_triang_distribution(d_r_min, d_r_avg, d_r_max).rvs()
            # current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
            # veh_density = current_veh_num / total_lane_length
            # hostler_speed = speed_density(veh_density)
            # print("Current hostler speed is:", hostler_speed)
            # yield env.timeout(d_r_dist / (2 * hostler_speed))
            travel_time_to_oc = state.HOSTLER_FIND_CONTAINER_TIME
            yield env.timeout(travel_time_to_oc)
            print(f"Time {env.now}: Hostler {hostler_id} picked up OC {oc} and is returning to the terminal")
            record_container_event(oc, 'hostler_pickup', env.now)
            emissions = emission_calculation('empty', 'hostler', hostler_id, travel_time_to_oc)
            record_vehicle_event('hostler', hostler_id, f'pickup_OC_{oc}', 'empty',
                                 emissions, 'end', env.now)

            travel_time_to_chassis = state.HOSTLER_TRANSPORT_CONTAINER_TIME
            yield env.timeout(travel_time_to_chassis)
            print(f"Time {env.now}: Hostler {hostler_id} dropped off OC {oc} onto chassis")
            yield terminal.chassis.put(oc)
            record_container_event(oc, 'hostler_dropoff', env.now)
            emissions = emission_calculation('loaded', 'hostler', hostler_id, travel_time_to_oc)
            record_vehicle_event('hostler', hostler_id, f'dropoff_OC_{oc}', 'loaded',
                                 emissions, 'end', env.now)

            print("chassis (oc_remaining):", terminal.chassis.items)
            print(f"hostler (oc_remaining): {terminal.hostlers.items}")
            if sum(1 for item in terminal.chassis.items if isinstance(item, str) and "OC-" in str(item)) == oc_needed:
                all_oc_prepared.succeed()
                print("chassis (all_oc_prepared):", terminal.chassis.items)
                print(f"hostler (all_oc_prepared): {terminal.hostlers.items}")
                print("# of OC on chassis:", sum(1 for item in terminal.chassis.items if "OC-" in str(item)))
                print(f"Time {env.now}: All OCs are ready on chassis.")
            i += 1


def truck_exit(env, terminal, truck_id, ic, train_schedule):
    global state
    with terminal.out_gates.request() as out_gate_request:
        yield out_gate_request
        print(f"Time {env.now}: Truck {truck_id} with IC-{ic.id}-Train-{train_schedule['train_id']} is passing through the out-gate and leaving the terminal")
        truck_travel_time = state.TRUCK_OUTGATE_TIME + random.uniform(0, state.TRUCK_OUTGATE_TIME_DEV)
        yield env.timeout(truck_travel_time)
        record_container_event(ic, 'truck_exit', env.now)
        emissions = emission_calculation('loaded', 'truck', truck_id, truck_travel_time)
        record_vehicle_event('truck', truck_id, f"leave_gate_IC_{ic.id}-Train-{train_schedule['train_id']}", 'loaded',
                             emissions, 'end', env.now)

    yield terminal.truck_store.put(truck_id)


def crane_load_process(env, terminal, load_time, start_load_event, end_load_event, track_id, train_schedule):
    global state
    yield start_load_event
    print(f"Time {env.now}: Starting loading process onto the train.")

    while len([item for item in terminal.chassis.items if
               isinstance(item, str) and "OC" in item]) > 0:  # if there still has OC on chassis
        crane_item = terminal.cranes.get(lambda x: x[2] == track_id)
        oc = yield terminal.chassis.get(lambda x: x.type=='Outbound' and x.train_id == train_schedule['train_id'])  # obtain an OC from chassis
        print("line 364 chassis:", terminal.chassis.items)
        print(f"Time {env.now}: Crane {crane_item.value[0]} starts loading {oc} onto the train.")
        yield env.timeout(load_time)  # loading time, depends on container parameter**
        record_container_event(oc, 'crane_load', env.now)
        record_vehicle_event('crane', (crane_item.value[0], crane_item.value[1]), f'load_{oc}', 'full',
                             load_time * state.FULL_EMISSIONS_RATES['Crane'][crane_item.value[1]], 'end', env.now)
        print(f"Time {env.now}: Crane {crane_item.value[0]} finished loading {oc} onto the train.")
        yield terminal.cranes.put(crane_item.value)


    print(f"Time {env.now}: All OCs loaded. Train is fully loaded and ready to depart.")
    print("Containers on chassis (after loading OCs):", terminal.chassis.items)
    end_load_event.succeed()


def process_train_arrival(env, terminal, train_schedule):
    global state

    train_id = train_schedule["train_id"]
    arrival_time = train_schedule["arrival_time"]
    departure_time = train_schedule["departure_time"]
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

    # All trucks arrive before train arrives
    env.process(truck_arrival(env, terminal, train_schedule, all_trucks_arrived_event))

    print("Current available track has ", terminal.tracks.items)

    # Wait train arriving
    if env.now <= arrival_time:
        yield env.timeout(arrival_time - env.now)
        print(f"Time {env.now}: [In Time] Train {train_schedule['train_id']} has arrived, waiting to be assigned to a track.")
        delay_time = 0
    else:
        delay_time = env.now - arrival_time
        if f"train_id_{train_id}" not in delay_list:
            delay_list[f"train_id_{train_id}"] = {}
        delay_list[f"train_id_{train_id}"]["arrival"] = delay_time
        print(f"Time {env.now}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours.")
    state.train_delay_time[train_schedule['train_id']] = delay_time

    track_id = yield terminal.tracks.get()
    if track_id is None:
        print(f"Time {env.now}: Train {train_id} is waiting for an available track.")
        terminal.waiting_trains.append(train_id)
        return
    print(f"Time {env.now}: Train {train_id} is assigned to Track {track_id}.")

    train_ic_store = simpy.FilterStore(env)
    for ic_id in range(state.IC_NUM, state.IC_NUM + train_schedule['full_cars']):
        ic = container(type='Inbound', id=ic_id, train_id = train_schedule['train_id'])
        train_ic_store.put(ic)
        record_container_event(ic, 'train_arrival', env.now)  # loop: assign container_id range(current_ic, current_ic + train_schedule['full_cars'])

    terminal.train_ic_stores[train_schedule['train_id']] = train_ic_store
    terminal.train_ic_unload_events[train_schedule['train_id']] = env.event()

    # crane unloading IC
    env.process(crane_unload_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, total_ic, all_ic_picked,
                                     all_ic_unload_event, track_id))

    # prepare all OC and pick up all IC before crane loading
    yield all_ic_picked & all_oc_prepared
    print(f"Time {env.now}: All {oc_needed} OCs are ready on chassis.")
    start_load_event.succeed()  # condition of chassis loading

    # crane loading process
    env.process(crane_load_process(env, terminal, load_time=2, start_load_event=start_load_event,
                                   end_load_event=end_load_event, track_id=track_id, train_schedule=train_schedule))

    yield end_load_event

    # train departs
    if env.now <= departure_time:
        yield env.timeout(departure_time - env.now)
        print(f"Time {env.now}: [In Time] Train {train_schedule['train_id']} departs from the track {track_id}.")
    else:
        delay_time = env.now - departure_time
        if f"train_id_{train_id}" not in delay_list:
            delay_list[f"train_id_{train_id}"] = {}
        delay_list[f"train_id_{train_id}"]["departure"] = delay_time
        print(f"Time {env.now}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours from the track {track_id}.")

    yield terminal.tracks.put(track_id)
    print(f"Time {env.now}: Train {train_schedule['train_id']} is departing from the terminal.")

    for oc_id in range(state.OC_NUM, state.OC_NUM + train_schedule['oc_number']):
        record_container_event(f"OC-{oc_id}-Train-{train_schedule['train_id']}", 'train_depart',env.now)  # loop: assign container_id range(current_oc, current_oc + train_schedule['full_cars'])

    state.time_per_train[train_schedule['train_id']] = env.now - arrival_time

    # Update various parameters to track inbound and outbound containers
    state.IC_NUM = state.IC_NUM + train_schedule['full_cars']
    state.OC_NUM = state.OC_NUM + train_schedule['oc_number']

    # Check available tracks
    print("Available tracks:", terminal.tracks.items)

def run_simulation(train_consist_plan: pl.DataFrame,terminal: str,out_path=None):
    global state
    state.terminal = terminal
    state.train_consist_plan = train_consist_plan
    state.initialize()

    # Create environment
    env = simpy.Environment()

    # Train timetable: shorter headway
    train_timetable = [
        {"train_id": 19, "arrival_time": 187, "departure_time": 200, "empty_cars": 3, "full_cars": 5, "oc_number": 2,
         "truck_number": 5},  # test: ic > oc
        {"train_id": 12, "arrival_time": 190, "departure_time": 220, "empty_cars": 5, "full_cars": 4, "oc_number": 4,
         "truck_number": 4},  # test: ic = oc
        {"train_id": 70, "arrival_time": 200, "departure_time": 300, "empty_cars": 5, "full_cars": 3, "oc_number": 5,
         "truck_number": 5},  # test: ic < oc
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
        env.process(process_train_arrival(env, terminal, train_schedule))

    # Simulation hyperparameters
    env.run(until=state.sim_time)

    # Performance Matrix
    # Train processing time
    # avg_time_per_train = sum(state.time_per_train.values()) / len(state.time_per_train)
    # print(f"Average train processing time: {sum(state.time_per_train) / len(state.time_per_train) if state.time_per_train else 0:.2f}")
    # print("Simulation completed. ")
    # with open("avg_time_per_train.txt", "w") as f:
    #   f.write(str(avg_time_per_train))

    # Create DataFrame for container events
    print(state.sim_time)

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
    #         out_path / f"simulation_crane_{state.CRANE_NUMBER}_hostler_{state.HOSTLER_NUMBER}.xlsx")

    # Use save_average_times and save_vehicle_logs for vehicle related logs
    save_vehicle_logs()

    print("Done!")
    return None


if __name__ == "__main__":
    run_simulation(
        train_consist_plan=pl.read_csv(utilities.package_root() / 'demos' / 'starter_demo' / 'train_consist_plan.csv'),
        terminal="Allouez",
        out_path=utilities.package_root() / 'demos' / 'starter_demo' / 'results'
    )