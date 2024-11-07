import simpy
import random
from parameters import *
from distances import *
from dictionary import *
from vehicle_performance import record_vehicle_event, save_average_times, save_vehicle_logs


# Test input
CRANE_NUMBER = 1
HOSTLER_NUMBER = 1
TRUCK_NUMBERS = 1000

def record_event(container_id, event_type, timestamp):
    if container_id not in container_events:
        container_events[container_id] = {}
    container_events[container_id][event_type] = timestamp


def handle_truck_arrivals(env, in_gate_resource, truck_numbers):
    global all_trucks_ready_event, truck_processed, start_oc_container_id, end_oc_container_id, TRUCK_ARRIVAL_MEAN, TRAIN_ARRIVAL_HR

    truck_id = 1
    truck_processed = 0
    TRUCK_ARRIVAL_MEAN = abs(TRAIN_ARRIVAL_HR - previous_train_departure) / max(INBOUND_CONTAINER_NUMBER, OUTBOUND_CONTAINER_NUMBER)
    print(f"current time is {env.now}")
    print(f"next TRAIN_ARRIVAL_HR:{TRAIN_ARRIVAL_HR}")
    print(f"TRUCK_ARRIVAL_MEAN IS {TRUCK_ARRIVAL_MEAN}")

    while truck_id <= TRUCK_NUMBERS:
        inter_arrival_time = random.expovariate(1 / TRUCK_ARRIVAL_MEAN)
        yield env.timeout(inter_arrival_time)
        truck_arrival_time.append(env.now)

        env.process(truck_through_gate(env, in_gate_resource, truck_id))
        truck_id += 1

    if truck_id > TRUCK_NUMBERS:
        # print(f"truck_id = {truck_id} vs TRUCK_NUM = {TRUCK_NUMBERS}")
        if not all_trucks_ready_event.triggered:
            all_trucks_ready_event.succeed()
            # print(f"{env.now}: All trucks arrived for the {TRAIN_ID} train.")


def truck_through_gate(env, in_gate_resource, truck_id):
    global last_leave_time, INBOUND_CONTAINER_NUMBER, OUTBOUND_CONTAINER_NUMBER

    with in_gate_resource.request() as request:
        yield request
        wait_time = max(0, truck_arrival_time[truck_id - 1] - last_leave_time)
        if wait_time <= 0:
            wait_time = 0  # first arriving trucks
            # print(f"Truck {truck_id} enters the gate without waiting")
        else:
            # print(f"Truck {truck_id} enters the gate and queued for {wait_time} hrs")
            truck_waiting_time.append(wait_time)

        yield env.timeout(TRUCK_INGATE_TIME + random.uniform(0, TRUCK_INGATE_TIME_DEV))

        # Case 1: Normal handling when OC >= IC (all trucks have containers)
        if OUTBOUND_CONTAINER_NUMBER >= INBOUND_CONTAINER_NUMBER:
            env.process(handle_container(env, truck_id))

        # Case 2: OC < IC, extra empty trucks are needed
        else:
            if truck_id <= OUTBOUND_CONTAINER_NUMBER:
                env.process(handle_container(env, truck_id))  # Loaded trucks
            else:
                env.process(empty_truck(env, truck_id))  # Empty trucks


def handle_container(env, truck_id):
    global outbound_container_id_counter, last_leave_time

    container_id = outbound_container_id_counter
    outbound_container_id_counter += 1
    record_event(container_id, 'truck_arrival', env.now)

    d_t_dist = create_triang_distribution(d_t_min, d_t_avg, d_t_max).rvs()
    yield env.timeout(d_t_dist / (2 * TRUCK_SPEED_LIMIT))

    record_event(container_id, 'truck_drop_off', env.now)
    # print(f"{env.now}: Truck {truck_id} drops outbound container {container_id}.")
    last_leave_time = env.now


def empty_truck(env, truck_id):
    global inbound_container_id_counter, last_leave_time

    d_t_dist = create_triang_distribution(d_t_min, d_t_avg, d_t_max).rvs()
    yield env.timeout(d_t_dist / (2 * TRUCK_SPEED_LIMIT))

    # print(f"{env.now}: Empty truck {truck_id} arrives.")
    last_leave_time = env.now


def train_arrival(env, train_timetable, train_processing, cranes, hostlers, chassis, in_gate_resource, outbound_containers_store, truck_store, out_gate_resource):
    global record_oc_label, train_id_counter, TRUCK_NUMBERS, INBOUND_CONTAINER_NUMBER, OUTBOUND_CONTAINER_NUMBER, TRAIN_DEPARTURE_HR, oc_chassis_filled_event, TRAIN_ID, inbound_container_id_counter, previous_train_departure

    for i, train in enumerate(train_timetable):
        TRAIN_ARRIVAL_HR = train['arrival_time']
        TRAIN_DEPARTURE_HR = train['departure_time']
        INBOUND_CONTAINER_NUMBER = train['full_cars']
        OUTBOUND_CONTAINER_NUMBER = train['oc_number']
        TRUCK_NUMBERS = train['truck_number']
        TRAIN_ID = train['train_id']

        print(f"---------- Next Train {TRAIN_ID} Is On the  Way ----------")
        print(f"IC {INBOUND_CONTAINER_NUMBER}")
        print(f"OC {OUTBOUND_CONTAINER_NUMBER}")

        outbound_containers_store.items.clear()
        for oc in range(record_oc_label, record_oc_label + OUTBOUND_CONTAINER_NUMBER):  # from 10001 to 10001 + OC
            # print("oc_number", oc)
            outbound_containers_store.put(oc)
            # yield outbound_containers_store.put(oc)
            # print(f"Current store contents after putting {oc}: {outbound_containers_store.items}")

        # print("outbound_containers_store is:", outbound_containers_store.items)

        # Trucks enter until the precious train departs, if not the first truck
        previous_train_departure = train_timetable[i-1]['departure_time'] if i > 0 else 0
        print(f"Schedule {TRUCK_NUMBERS} Trucks arriving between previous train departure at {previous_train_departure} and current train arrival at {TRAIN_ARRIVAL_HR}")
        env.process(handle_truck_arrivals(env, in_gate_resource, outbound_containers_store))

        # Trains arrive according to the timetable, fix negative delay bug
        delay = TRAIN_ARRIVAL_HR - env.now
        if delay <= 0:
            yield env.timeout(0)
        else:
            yield env.timeout(delay)

        train_id = train_id_counter
        print(f"Train {TRAIN_ID} ({train_id} in the dictionary) arrives at {env.now}")

        # for container_id in range(inbound_container_id_counter, inbound_container_id_counter + INBOUND_CONTAINER_NUMBER):
        for container_id in range(int(inbound_container_id_counter), int(inbound_container_id_counter) + int(INBOUND_CONTAINER_NUMBER)):  # fix float error

            record_event(container_id, 'train_arrival', env.now)

        with train_processing.request() as request:
            yield request
            oc_chassis_filled_event = env.event()
            yield env.process(process_train(env, train_id, cranes, hostlers, chassis, in_gate_resource, outbound_containers_store, truck_store, train_processing, oc_chassis_filled_event, out_gate_resource))
            train_id_counter += 1

        record_oc_label += OUTBOUND_CONTAINER_NUMBER
        # print("record_oc_label", record_oc_label)
        # print("oc_variance in train_process:", oc_variance)


def process_train(env, train_id, cranes, hostlers, chassis, in_gate_resource, outbound_containers_store, truck_store, train_processing, oc_chassis_filled_event, out_gate_resource):
    global oc_variance, time_per_train, train_series, INBOUND_CONTAINER_NUMBER, OUTBOUND_CONTAINER_NUMBER, record_oc_label, TRAIN_ID_FIXED

    start_time = env.now

    # Cranes unload all IC
    unload_processes = []
    chassis_inbound_ids = []  # To save chassis_id, current_inbound_id to hostler_transfer_IC_single_loop

    # if train_id < TRAIN_NUMBERS:
    for chassis_id in range(1, int(INBOUND_CONTAINER_NUMBER) + 1):
        unload_process = env.process(crane_and_chassis(env, train_id, 'unload', cranes, hostlers, chassis, truck_store, train_processing, outbound_containers_store, in_gate_resource, out_gate_resource, oc_chassis_filled_event))
        unload_processes.append(unload_process)

    # All IC are processed
    # print("Unload process is:", unload_processes)
    yield simpy.events.AllOf(env, unload_processes)
    results = yield simpy.events.AllOf(env, unload_processes)

    for container_id in range(int(inbound_container_id_counter), int(inbound_container_id_counter) + int(INBOUND_CONTAINER_NUMBER)):  # fix float error
        env.process(crane_movement(env, container_id, 'unload'))


    # To pass chassis_id, current_inbound_id to hostler_transfer_IC_single_loop as a list from calling chassis_inbound_ids
    for result in results.values():
        chassis_id, current_inbound_id = result
        chassis_inbound_ids.append((chassis_id, current_inbound_id))
        env.process(hostler_transfer(env, hostlers, 'inbound', chassis, chassis_id, current_inbound_id, truck_store, cranes,
                             train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event,
                             out_gate_resource))

    # # Once all OC are dropped by hostlers, crane start working
    # print("Chassis are filled with OC (-1) now. ")
    # print(f"Chassis status after OC processed is: {chassis_status}, where ")
    # print(f"there are {chassis_status.count(0)} chassis is filled with OC (0)")
    # print(f"there are {chassis_status.count(-1)} chassis is filled with empty (-1)")
    # print(f"there are {chassis_status.count(1)} chassis is filled with IC (1)")

    # Cranes move all OC to chassis
    load_processes = []
    for chassis_id in range(1, OUTBOUND_CONTAINER_NUMBER + 1):
        load_process = env.process(crane_and_chassis(env, train_id, 'load', cranes, hostlers, chassis, truck_store, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource, chassis_id=chassis_id))
        load_processes.append(load_process)

    yield simpy.events.AllOf(env, load_processes)

    for container_id in range(int(inbound_container_id_counter), int(inbound_container_id_counter) + int(INBOUND_CONTAINER_NUMBER)):  # fix float error
        env.process(crane_movement(env, container_id, 'load'))

    # Check if all outbound containers are loaded (all chassis is empty 0), the train departs
    if chassis_status.count(-1) == TRAIN_UNITS:
        # oc_chassis_filled_event.succeed()
        TRAIN_ID_FIXED = TRAIN_ID
        print(f"Train {TRAIN_ID_FIXED} is ready to depart at {env.now}.")
        env.process(train_departure(env, train_id))
        time_per_train.append(env.now - start_time)

    end_time = env.now
    time_per_train.append(end_time - start_time)
    train_series += 1
    oc_variance += OUTBOUND_CONTAINER_NUMBER

def crane_movement(env, container_id, action):
    global record_oc_label, crane_id_counter, chassis_status, inbound_container_id_counter, outbound_containers_mapping, outbound_container_id_counter, INBOUND_CONTAINER_NUMBER, OUTBOUND_CONTAINER_NUMBER

    # # Print before requesting crane resource
    if action == 'unload':
        crane_id = crane_id_counter
        crane_id_counter = (crane_id_counter % CRANE_NUMBER) + 1
        yield env.timeout(CRANE_UNLOAD_CONTAINER_TIME_MEAN + random.uniform(0, CRANE_MOVE_DEV_TIME))
        record_event(container_id, 'crane_unload', env.now)
        print(f"Crane unloads outbound container {container_id} to train {TRAIN_ID} at {env.now}")

    if action == 'load':
        for container_id in range(record_oc_label, record_oc_label + OUTBOUND_CONTAINER_NUMBER):
            yield env.timeout(CRANE_LOAD_CONTAINER_TIME_MEAN + random.uniform(0, CRANE_MOVE_DEV_TIME))
            # chassis_status[chassis_id - 1] = -1
            print(f"Crane loads outbound container {container_id} to train {TRAIN_ID} at {env.now}")
            record_event(container_id, 'crane_load', env.now)


def crane_and_chassis(env, train_id, action, cranes, hostlers, chassis, truck_store, train_processing, outbound_containers_store, in_gate_resource, out_gate_resource, oc_chassis_filled_event, chassis_id=None):
    global record_oc_label, crane_id_counter, chassis_status, inbound_container_id_counter, outbound_containers_mapping, outbound_container_id_counter, INBOUND_CONTAINER_NUMBER, OUTBOUND_CONTAINER_NUMBER

    with cranes.request() as request:
        yield request

        # # Print after acquiring crane resource
        # print(f"[{env.now}] Crane {crane_id_counter} acquired crane resource. Available cranes: {cranes.count}/{cranes.capacity}")

        start_time = env.now
        record_vehicle_event('crane', crane_id_counter, 'start', start_time)    # performance record: starting

        if action == 'unload':
            # crane_id = crane_id_counter
            # crane_id_counter = (crane_id_counter % CRANE_NUMBER) + 1

            chassis_id = ((inbound_container_id_counter - 1) % CHASSIS_NUMBER) + 1

            current_inbound_id = inbound_container_id_counter
            inbound_container_id_counter += 1
            # yield env.timeout(CRANE_UNLOAD_CONTAINER_TIME_MEAN + random.uniform(0, CRANE_MOVE_DEV_TIME))

            # for chassis_id in range(int(inbound_container_id_counter), int(inbound_container_id_counter) + int(INBOUND_CONTAINER_NUMBER)):
            chassis_status[chassis_id - 1] = 1

            end_time = env.now
            record_vehicle_event('crane', crane_id_counter, 'end', end_time)     # performance record: ending

            # hostler picks up IC
            env.process(hostler_transfer(env, hostlers, 'inbound', chassis, chassis_id, current_inbound_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource))

            return chassis_id, current_inbound_id

        elif action == 'load':
            if chassis_id not in outbound_containers_mapping:
                print(f"Notice: No outbound container mapped to chassis {chassis_id} at {env.now}")
                return

            container_id = outbound_containers_mapping[chassis_id]  # Retrieve container ID from mapping
            # print("outbound_containers_mapping in crane and chassis func:", outbound_containers_mapping)
            # print("container_id in crane and chassis func:", container_id)

            if CRANE_NUMBER == 1:
                crane_id = 1
            else:
                crane_id = (chassis_id % CRANE_NUMBER) + 1

            chassis_status[chassis_id - 1] = -1

            # for container_id in range(record_oc_label, record_oc_label + OUTBOUND_CONTAINER_NUMBER):
            #     yield env.timeout(CRANE_LOAD_CONTAINER_TIME_MEAN + random.uniform(0, CRANE_MOVE_DEV_TIME))
            #     chassis_status[chassis_id - 1] = -1
            #     print(f"Crane {crane_id} loads outbound container {container_id} to train {TRAIN_ID} at {env.now}")
            #     record_event(container_id, 'crane_load', env.now)


def hostler_transfer(env, hostlers, container_type, chassis, chassis_id, container_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource):
    global hostler_id_counter, chassis_status, outbound_containers_mapping, outbound_container_id, record_oc_label, HOSTLER_NUMBER

    with hostlers.request() as request:
        yield request

        start_time = env.now
        record_vehicle_event('hostler', hostler_id_counter, 'start', start_time)  # performance record

        hostler_id = hostler_id_counter
        hostler_id_counter = (hostler_id_counter % HOSTLER_NUMBER) + 1

        with chassis.request() as chassis_request:
            yield chassis_request

            if container_type == 'inbound' and chassis_status[chassis_id - 1] == 1:
                d_h_dist = create_triang_distribution(d_h_min, d_h_avg, d_h_max).rvs()
                HOSTLER_TRANSPORT_CONTAINER_TIME = d_h_dist / (2 * HOSTLER_SPEED_LIMIT)
                print(f"Hostler pick-up time is:{HOSTLER_TRANSPORT_CONTAINER_TIME}")
                yield env.timeout(HOSTLER_TRANSPORT_CONTAINER_TIME)
                record_event(container_id, 'hostler_pickup', env.now)
                print(f"Hostler {hostler_id} picks up inbound container {container_id} from chassis {chassis_id} and heads to parking area at {env.now}")

                chassis_status[chassis_id - 1] = -1

                # Hostler drop off: different route for picking-up and dropping-off
                d_h_dist = create_triang_distribution(d_h_min, d_h_avg, d_h_max).rvs()
                HOSTLER_TRANSPORT_CONTAINER_TIME = d_h_dist / (2 * HOSTLER_SPEED_LIMIT)
                print(f"Hostler drop-off time is:{HOSTLER_TRANSPORT_CONTAINER_TIME}")
                yield env.timeout(HOSTLER_TRANSPORT_CONTAINER_TIME)
                record_event(container_id, 'hostler_dropoff', env.now)
                print(f"Hostler {hostler_id} drops off inbound container {container_id} from chassis {chassis_id} and moves toward the assigned outbound container at {env.now}")

                end_time = env.now
                record_vehicle_event('hostler', hostler_id_counter, 'end', end_time)  # performance record

                # Process functions of notify_truck and handle_outbound_container simultaneously
                env.process(notify_truck(env, truck_store, container_id, out_gate_resource))

                # Assign outbound container and chassis_id for the hostler which drops off an inbound container
                chassis_id, outbound_container_id = yield env.process(outbound_container_decision_making(
                    env, hostlers, chassis, container_id, truck_store, cranes, train_processing,
                    outbound_containers_store,
                    in_gate_resource, oc_chassis_filled_event, out_gate_resource, chassis_status,
                    outbound_containers_mapping,
                    record_oc_label, outbound_container_id
                ))

                # Process outbound containers
                if chassis_id is not None and outbound_container_id is not None:
                    env.process(handle_outbound_container(env, hostler_id, chassis_id, outbound_container_id, truck_store,
                                                  cranes, train_processing, outbound_containers_store, in_gate_resource,
                                                  oc_chassis_filled_event))


# When OC are fully processed, but IC are not
def hostler_transfer_IC_single_loop(env, hostlers, container_type, chassis, chassis_id, container_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource):
    print(f"Starting single hostler transfer IC loop for chassis {chassis_id} at {env.now}")
    global hostler_id_counter

    print(f"Requesting hostler for IC at chassis {chassis_id} at {env.now}")

    with hostlers.request() as request:
        print(f"Request available hostlers: {hostlers.count} vs total hostlers {HOSTLER_NUMBER}, Hostlers capacity: {hostlers.capacity} at {env.now}")
        yield request

        start_time = env.now
        record_vehicle_event('hostler', hostler_id_counter, 'start', start_time)  # performance record

        hostler_id = hostler_id_counter
        hostler_id_counter = (hostler_id_counter % HOSTLER_NUMBER) + 1

        with chassis.request() as chassis_request:
            yield chassis_request

            if container_type == 'inbound' and chassis_status[chassis_id - 1] == 1:
                chassis_status[chassis_id - 1] = -1
                print(f"Single loop chassis status {chassis_status}")
                print(f"There are {chassis_status.count(1)} IC")
                print(f"There are {chassis_status.count(-1)} empty")
                print(f"There are {chassis_status.count(0)} OC")
                d_h_dist = create_triang_distribution(d_h_min, d_h_avg, d_h_max).rvs()
                HOSTLER_TRANSPORT_CONTAINER_TIME = d_h_dist / (2 * HOSTLER_SPEED_LIMIT)

                yield env.timeout(HOSTLER_TRANSPORT_CONTAINER_TIME)
                # hostler picks up the rest of IC from the chassis
                # chassis_status[chassis_id - 1] = -1
                record_event(container_id, 'hostler_pickup', env.now)
                print(f"Hostler {hostler_id} picks up inbound container {container_id} from chassis {chassis_id} to parking area at {env.now}")
                # hostler drops off the IC
                yield env.timeout(HOSTLER_TRANSPORT_CONTAINER_TIME)
                record_event(container_id, 'hostler_dropoff', env.now)
                print(f"Hostler {hostler_id} drops off inbound container {container_id} from chassis {chassis_id} to parking area at {env.now}")

                # Check if all chassis filled
                if chassis_status.count(0) == OUTBOUND_CONTAINER_NUMBER and chassis_status.count(
                        -1) == TRAIN_UNITS - OUTBOUND_CONTAINER_NUMBER and not oc_chassis_filled_event.triggered:
                    print(f"Chassis is fully filled with OC, and cranes start moving: {chassis_status}")
                    print(f"where there are {chassis_status.count(0)} chassis filled with OC (0)")
                    print(f"where there are {chassis_status.count(-1)} chassis filled with empty (-1)")
                    print(f"where there are {chassis_status.count(1)} chassis filled with IC (1)")
                    oc_chassis_filled_event.succeed()
                    return
                else:
                    print(f"Chassis is not fully filled: {chassis_status}")
                    print(f"where there are {chassis_status.count(0)} chassis filled with OC (0)")
                    print(f"where there are {chassis_status.count(-1)} chassis filled with empty (-1)")
                    print(f"where there are {chassis_status.count(1)} chassis filled with IC (1)")

                end_time = env.now
                record_vehicle_event('hostler', hostler_id, 'end', end_time)  # performance record

                # trucks pick up IC
                yield env.process(notify_truck(env, truck_store, container_id, out_gate_resource))


def outbound_container_decision_making(env, hostlers, chassis, current_inbound_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource, chassis_status, outbound_containers_mapping, record_oc_label, outbound_container_id):
    # Check if outbound_containers_store has outbound container
    if len(outbound_containers_store.items) > 0:
        outbound_container_id = yield outbound_containers_store.get()
        print(f"Outbound containers remaining: {len(outbound_containers_store.items)}")

        if -1 in chassis_status:
            chassis_id = chassis_status.index(-1) + 1  # find the first chassis
            # If chassis are not assigned with outbound container
            if chassis_id not in outbound_containers_mapping:
                # outbound_container_id += record_oc_label
                outbound_containers_mapping[chassis_id] = outbound_container_id
                chassis_status[chassis_id - 1] = 0  # already assigned outbound container
                print(f"OC mapping created: outbound container {outbound_container_id} assigned to chassis {chassis_id}")
            else:
                print(f"Chassis {chassis_id} is already mapped to an outbound container.")
        else:
            print("No empty chassis available for outbound container assignment.")

    # if outbound_containers_store is null, check if we need operate single loop
    else:
        chassis_id = None
        outbound_container_id = None
        # chassis_status = 1: inbound containers are not loaded
        if chassis_status.count(1) != 0:
            print(f"Haven't finished all IC yet at {env.now}. Starting single loop.")
            chassis_id = chassis_status.index(1) + 1
            chassis_status[chassis_id - 1] = 0  # assigned with IC
            # single loop takes rest inbound container
            yield env.process(hostler_transfer_IC_single_loop(env, hostlers, 'inbound', chassis, chassis_id, current_inbound_id,
                                                truck_store, cranes, train_processing,
                                                outbound_containers_store, in_gate_resource, oc_chassis_filled_event,
                                                out_gate_resource))
        else:
            print("All inbound containers have been processed.")

    return chassis_id, outbound_container_id


def handle_outbound_container(env, hostler_id, chassis_id, outbound_container_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event):
    global HOSTLER_FIND_CONTAINER_TIME

    d_h_dist = create_triang_distribution(d_h_min, d_h_avg, d_h_max).rvs()
    HOSTLER_TRANSPORT_CONTAINER_TIME = d_h_dist / (2 * HOSTLER_SPEED_LIMIT)

    d_r_dist = create_triang_distribution(d_r_min, d_r_avg, d_r_max).rvs()
    HOSTLER_FIND_CONTAINER_TIME = d_r_dist / (2 * TRUCK_SPEED_LIMIT)
    yield env.timeout(HOSTLER_FIND_CONTAINER_TIME)

    record_event(outbound_container_id, 'hostler_pickup', env.now)
    print(f"Hostler {hostler_id} picks up outbound container {outbound_container_id} from parking area to chassis {chassis_id} at {env.now}")

    yield env.timeout(HOSTLER_TRANSPORT_CONTAINER_TIME)

    record_event(outbound_container_id, 'hostler_dropoff', env.now)
    print(f"Hostler {hostler_id} drops off outbound container {outbound_container_id} to chassis {chassis_id} at {env.now}")


# truck pick up IC
def notify_truck(env, truck_store, container_id, out_gate_resource):
    truck_id = yield truck_store.get()
    yield env.timeout(TRUCK_INGATE_TIME)
    print(f"Truck {truck_id} arrives at parking area and prepare to pick up inbound container {container_id} at {env.now}")
    yield env.process(truck_transfer(env, truck_id, container_id, out_gate_resource))


def truck_transfer(env, truck_id, container_id, out_gate_resource):
    global TRUCK_INGATE_TIME, TRUCK_TRANSPORT_CONTAINER_TIME, outbound_container_id_counter

    start_time = env.now
    record_vehicle_event('truck', truck_id, 'start', start_time)  # performance record

    # Truck moves to the parking area
    yield env.timeout(TRUCK_TO_PARKING)
    record_event(container_id, 'truck_pickup', env.now)
    print(f"Truck {truck_id} picks up inbound container {container_id} at {env.now}")

    # Calculate the transport time for the truck
    d_t_dist = create_triang_distribution(d_t_min, d_t_avg, d_t_max).rvs()
    TRUCK_TRANSPORT_CONTAINER_TIME = d_t_dist / (2 * TRUCK_SPEED_LIMIT)
    yield env.timeout(TRUCK_TRANSPORT_CONTAINER_TIME)

    # Request out_gate_resource resource before the truck exits
    with out_gate_resource.request() as request:
        yield request

        # Simulate the time it takes for the truck to pass through the gate
        yield env.timeout(TRUCK_OUTGATE_TIME + random.uniform(0,TRUCK_OUTGATE_TIME_DEV))
        record_event(container_id, 'truck_exit', env.now)
        print(f"Truck {truck_id} exits gate with inbound container {container_id} at {env.now}")

    # End performance recording
    end_time = env.now
    record_vehicle_event('truck', truck_id, 'end', end_time)


def train_departure(env, train_id):
    global TRAIN_ID_FIXED, record_oc_label

    if env.now < TRAIN_DEPARTURE_HR:
        yield env.timeout(TRAIN_DEPARTURE_HR - env.now)
    yield env.timeout(TRAIN_INSPECTION_TIME)
    print(f"Train {TRAIN_ID_FIXED} ({train_id} in the dictionary) departs at {env.now}")

    for container_id in range(record_oc_label - OUTBOUND_CONTAINER_NUMBER, record_oc_label):
        record_event(container_id, 'train_depart', env.now)


def run_simulation():
    global all_trucks_ready_event, record_oc_label, TRUCK_NUMBERS, TRAIN_NUMBERS

    print(f"Starting simulation with No.{TRAIN_ID} trains, {HOSTLER_NUMBER} hostlers, {CRANE_NUMBER} cranes, and {TRUCK_NUMBERS} trucks.")
    env = simpy.Environment()

    # Resources
    train_processing = simpy.Resource(env, capacity=1)
    cranes = simpy.Resource(env, capacity=CRANE_NUMBER)
    chassis = simpy.Resource(env, capacity=CHASSIS_NUMBER)
    hostlers = simpy.Resource(env, capacity=HOSTLER_NUMBER)
    in_gate_resource = simpy.Resource(env, capacity=IN_GATE_NUMBERS)
    out_gate_resource = simpy.Resource(env, capacity=OUT_GATE_NUMBERS)
    outbound_containers_store = simpy.Store(env, capacity=100)
    truck_store = simpy.Store(env, capacity=100)

    # Initialize trucks
    truck_store.items.clear()
    # print("TRUCK_NUMBERS:",  TRUCK_NUMBERS)
    for truck_id in range(1, TRUCK_NUMBERS + 1):
        truck_store.put(truck_id)
    # print("TRUCK_STORE:", truck_store.items)

    all_trucks_ready_event = env.event()

    # # toy case
    # train_timetable = [
    #     {"train_id": 19, "arrival_time": 187, "departure_time": 200, "empty_cars": 3, "full_cars":7, "oc_number": 2, "truck_number":7 },
    #     {"train_id": 25, "arrival_time": 250, "departure_time": 350, "empty_cars": 4, "full_cars":6, "oc_number": 2, "truck_number":6 },
    #     {"train_id": 49, "arrival_time": 400, "departure_time": 600, "empty_cars": 5, "full_cars":5, "oc_number": 2, "truck_number":5 },
    #     {"train_id": 60, "arrival_time": 650, "departure_time": 750, "empty_cars": 6, "full_cars":4, "oc_number": 2, "truck_number":4 },
    #     {"train_id": 12, "arrival_time": 800, "departure_time": 1000, "empty_cars": 7, "full_cars":3, "oc_number": 4, "truck_number":4 },
    # ]

    # REAL TEST
    train_timetable = timetable(terminal)
    TRAIN_NUMBERS = len(timetable(terminal))

    # env.process(train_arrival(env, train_processing, cranes, in_gate_resource, outbound_containers_store, truck_store, train_timetable))
    env.process(train_arrival(env, train_timetable, train_processing, cranes, hostlers, chassis, in_gate_resource,
                  outbound_containers_store, truck_store, out_gate_resource))

    env.run(until=SIM_TIME)

    # Performance Matrix: train processing time
    avg_time_per_train = sum(time_per_train) / len(time_per_train)
    print(f"Average train processing time: {sum(time_per_train) / len(time_per_train) if time_per_train else 0:.2f}")
    print("Simulation completed. ")
    with open("avg_time_per_train.txt", "w") as f:
        f.write(str(avg_time_per_train))

    # Create DataFrame for container events
    container_data = []

    for container_id, events in sorted(container_events.items()):
        container_type = 'inbound' if container_id < 10001 else 'outbound'
        if container_type == 'inbound':
            container_process_time = events.get('truck_exit', '-') - events.get('train_arrival', '-') if 'truck_exit' in events and 'train_arrival' in events else '-'
        else:
            container_process_time = events.get('train_depart', '-') - events.get('truck_drop_off', '-') if 'train_depart' in events and 'truck_drop_off' in events else '-'

        container_data.append({
            'container_id': container_id,
            'container_type': container_type,
            'train_arrival': events.get('train_arrival', '-'),
            'truck_arrival': events.get('truck_arrival', '-'),
            'crane_unload': events.get('crane_unload', '-'),
            'hostler_pickup': events.get('hostler_pickup', '-'),
            'hostler_dropoff': events.get('hostler_dropoff', '-'),
            'truck_drop_off': events.get('truck_drop_off', '-'),
            'truck_pickup': events.get('truck_pickup', '-'),
            'truck_exit': events.get('truck_exit', '-'),
            'crane_load': events.get('crane_load', '-'),
            'train_depart': events.get('train_depart', '-'),
            'container_processing_time': container_process_time
        })

    df = pd.DataFrame(container_data)
    filename = f"C:/Users/Irena Tong/PycharmProjects/simulation_test/test/results/simulation_crane_{CRANE_NUMBER}_hostler_{HOSTLER_NUMBER}.xlsx"
    df.to_excel(filename, index=False)

    # Use save_average_times and save_vehicle_logs for vehicle related logs
    save_average_times()
    save_vehicle_logs()

    print("Done!")


if __name__ == "__main__":
    run_simulation()