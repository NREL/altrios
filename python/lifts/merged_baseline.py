import simpy
import random
import pandas as pd
from parameters import *
from distances import *
from dictionary import *
from vehicle_performance import record_vehicle_event, save_average_times, save_vehicle_logs


CRANE_NUMBER = 2
HOSTLER_NUMBER = 20

def record_event(container_id, event_type, timestamp):
    if container_id not in container_events:
        container_events[container_id] = {}
    container_events[container_id][event_type] = timestamp


def handle_truck_arrivals(env, in_gate_resource, truck_numbers):
    global all_trucks_ready_event, truck_processed

    truck_id = 1
    truck_processed = 0

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
            print(f"Truck {truck_id} enters the gate without waiting")
        else:
            print(f"Truck {truck_id} enters the gate and queued for {wait_time} hrs")
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

    container_id = outbound_container_id_counter + 1
    outbound_container_id_counter += 1
    record_event(container_id, 'truck_arrival', env.now)

    d_g_dist = create_triang_distribution(d_t_min, d_t_avg, d_t_max).rvs()
    yield env.timeout(d_g_dist / (2 * TRUCK_SPEED_LIMIT))

    record_event(container_id, 'truck_drop_off', env.now)
    print(f"{env.now}: Truck {truck_id} drops outbound container {container_id}.")
    last_leave_time = env.now


def empty_truck(env, truck_id):
    global inbound_container_id_counter, last_leave_time

    container_id = inbound_container_id_counter + OUTBOUND_CONTAINER_NUMBER + 1
    record_event(container_id, 'truck_arrival', env.now)

    d_g_dist = create_triang_distribution(d_t_min, d_t_avg, d_t_max).rvs()
    yield env.timeout(d_g_dist / (2 * TRUCK_SPEED_LIMIT))

    record_event(container_id, 'truck_drop_off', env.now)
    print(f"{env.now}: Empty truck {truck_id} arrives.")
    last_leave_time = env.now


def train_arrival(env, train_timetable, train_processing, cranes, hostlers, chassis, in_gate_resource, outbound_containers_store, truck_store, out_gate_resource):
# def train_arrival(env, train_processing, cranes, in_gate_resource, outbound_containers_store, truck_store, train_timetable):
    global train_id_counter, TRUCK_NUMBERS, INBOUND_CONTAINER_NUMBER, OUTBOUND_CONTAINER_NUMBER, TRAIN_DEPARTURE_HR


    for i, train in enumerate(train_timetable):
        TRAIN_ARRIVAL_HR = train['arrival_time']  # 获取列车到达时间
        TRAIN_DEPARTURE_HR = train['departure_time']  # 获取列车离开时间
        INBOUND_CONTAINER_NUMBER = train['full_cars']
        OUTBOUND_CONTAINER_NUMBER = train['oc_number']
        TRUCK_NUMBERS = train['truck_number']
        TRAIN_ID = train['train_id']

        print(f"---------- Next Train {TRAIN_ID} Is On the Way ----------")
        print(f"IC {INBOUND_CONTAINER_NUMBER}")
        print(f"OC {OUTBOUND_CONTAINER_NUMBER}")

        # 如果不是第一列火车，在上一列火车离开后安排卡车的到达
        previous_train_departure = train_timetable[i-1]['departure_time']
        print(f"Schedule {TRUCK_NUMBERS} Trucks arriving between previous train departure at {previous_train_departure} and current train arrival at {TRAIN_ARRIVAL_HR}")
        env.process(handle_truck_arrivals(env, in_gate_resource, outbound_containers_store))

        # 等待当前火车到达的时间
        yield env.timeout(TRAIN_ARRIVAL_HR - env.now)

        train_id = train_id_counter
        print(f"Train {train_id} arrives at {env.now}")

        with train_processing.request() as request:
            yield request
            oc_chassis_filled_event = env.event()
            # yield env.process(process_train(env, train_id, cranes, TRAIN_DEPARTURE_HR))
            yield env.process(process_train(env, train_id, cranes, hostlers, chassis, in_gate_resource, outbound_containers_store, truck_store, train_processing, oc_chassis_filled_event, out_gate_resource))
            train_id_counter += 1


def process_train(env, train_id, cranes, hostlers, chassis, in_gate_resource, outbound_containers_store, truck_store, train_processing, oc_chassis_filled_event, out_gate_resource):
    global time_per_train, train_series, INBOUND_CONTAINER_NUMBER, OUTBOUND_CONTAINER_NUMBER

    start_time = env.now

    # Cranes unload all IC
    unload_processes = []
    chassis_inbound_ids = []  # To save chassis_id, current_inbound_id to hostler_transfer_IC_single_loop
    for _ in range(INBOUND_CONTAINER_NUMBER):
        unload_process = env.process(crane_and_chassis(env, train_id, 'unload', cranes, hostlers, chassis, truck_store, train_processing, outbound_containers_store, in_gate_resource, out_gate_resource, oc_chassis_filled_event))
        unload_processes.append(unload_process)
    # All IC are processed
    results = yield simpy.events.AllOf(env, unload_processes)

    # To pass chassis_id, current_inbound_id to hostler_transfer_IC_single_loop as a list from calling chassis_inbound_ids
    for result in results.values():
        chassis_id, current_inbound_id = result
        chassis_inbound_ids.append((chassis_id, current_inbound_id))

    # Are all chassis filled with OC?
    # Once all OC are dropped by hostlers, crane start working
    print("Check before cranes start: Chassis filled with OC (-1) ? ")
    print(f"Chassis status after OC processed is: {chassis_status}, where ")
    print(f"there are {chassis_status.count(0)} chassis is filled with OC (0)")
    print(f"there are {chassis_status.count(-1)} chassis is filled with empty (-1)")
    print(f"there are {chassis_status.count(1)} chassis is filled with IC (1)")

    if chassis_status.count(1) != 0:    # IC is not fully processed
        print("Haven't finished all IC yet")
        # env.process(hostler_transfer_IC_single_loop(env, hostlers, 'inbound', chassis, chassis_id, current_inbound_id, truck_store, oc_chassis_filled_event, out_gate_resource))
        env.process(hostler_transfer_IC_single_loop(env, hostlers, 'inbound', chassis, chassis_id, current_inbound_id, truck_store, cranes, train_processing,
                                                    outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource))

    yield oc_chassis_filled_event

    # Cranes move all OC to chassis
    load_processes = []
    for chassis_id in range(1, OUTBOUND_CONTAINER_NUMBER + 1):
        load_process = env.process(crane_and_chassis(env, train_id, 'load', cranes, hostlers, chassis, truck_store, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource, chassis_id=chassis_id))
        load_processes.append(load_process)
    yield simpy.events.AllOf(env, load_processes)

    # Check if all outbound containers are loaded (all chassis is empty 0), the train departs
    if chassis_status.count(-1) == TRAIN_UNITS:
        oc_chassis_filled_event.succeed()
        print(f"Train {TRAIN_ID} is ready to depart.")
        env.process(train_departure(env, train_id))
        time_per_train.append(env.now - start_time)

    end_time = env.now
    time_per_train.append(end_time - start_time)
    train_series += 1


def crane_and_chassis(env, train_id, action, cranes, hostlers, chassis, truck_store, train_processing, outbound_containers_store, in_gate_resource, out_gate_resource, oc_chassis_filled_event, chassis_id=None):
    global crane_id_counter, chassis_status, inbound_container_id_counter, outbound_containers_mapping, outbound_container_id_counter, INBOUND_CONTAINER_NUMBER, OUTBOUND_CONTAINER_NUMBER

    with cranes.request() as request:
        yield request

        start_time = env.now
        record_vehicle_event('crane', crane_id_counter, 'start', start_time)    # performance record: starting

        if action == 'unload':
            crane_id = crane_id_counter
            crane_id_counter = (crane_id_counter % CRANE_NUMBER) + 1

            chassis_id = ((inbound_container_id_counter - 1) % CHASSIS_NUMBER) + 1

            current_inbound_id = inbound_container_id_counter
            inbound_container_id_counter += 1
            yield env.timeout(CRANE_UNLOAD_CONTAINER_TIME_MEAN + random.uniform(0, CRANE_MOVE_DEV_TIME))

            end_time = env.now
            record_vehicle_event('crane', crane_id_counter, 'end', end_time)     # performance record: ending

            # print(f"length of chassis status: {len(chassis_status)}")
            chassis_status[chassis_id - 1] = 1
            record_event(current_inbound_id, 'crane_unload', env.now)
            print(f"Crane {crane_id} unloads inbound container {current_inbound_id} at chassis {chassis_id} from train {train_id} at {env.now}")
            env.process(hostler_transfer(env, hostlers, 'inbound', chassis, chassis_id, current_inbound_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource))

            return chassis_id, current_inbound_id

        elif action == 'load':
            if chassis_id not in outbound_containers_mapping:
                print(f"Error: No outbound container mapped to chassis {chassis_id} at {env.now}")
                return

            container_id = outbound_containers_mapping[chassis_id]  # Retrieve container ID from mapping

            if CRANE_NUMBER == 1:
                crane_id = 1
            else:
                crane_id = (chassis_id % CRANE_NUMBER) + 1

            yield env.timeout(CRANE_LOAD_CONTAINER_TIME_MEAN + random.uniform(0, CRANE_MOVE_DEV_TIME))
            chassis_status[chassis_id - 1] = -1
            record_event(container_id, 'crane_load', env.now)
            print(f"Crane {crane_id} loads outbound container {container_id} from chassis {chassis_id} to train {TRAIN_ID} at {env.now}")


def hostler_transfer(env, hostlers, container_type, chassis, chassis_id, container_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource):
    global hostler_id_counter

    with hostlers.request() as request:
        yield request

        start_time = env.now
        record_vehicle_event('hostler', hostler_id_counter, 'start', start_time)  # performance record

        hostler_id = hostler_id_counter
        hostler_id_counter = (hostler_id_counter % HOSTLER_NUMBER) + 1

        with chassis.request() as chassis_request:
            yield chassis_request

            if container_type == 'inbound' and chassis_status[chassis_id - 1] == 1:
                d_t_dist = create_triang_distribution(d_t_min, d_t_avg, d_t_max).rvs()
                d_y_dist = create_triang_distribution(d_y_min, d_y_avg, d_y_max).rvs()
                HOSTLER_TRANSPORT_CONTAINER_TIME = (d_t_dist + d_y_dist) / (2 * HOSTLER_SPEED_LIMIT)
                yield env.timeout(HOSTLER_TRANSPORT_CONTAINER_TIME)
                chassis_status[chassis_id - 1] = -1
                record_event(container_id, 'hostler_pickup', env.now)
                print(f"Hostler {hostler_id} picks up inbound container {container_id} from chassis {chassis_id} to parking area at {env.now}")
                yield env.timeout(HOSTLER_TRANSPORT_CONTAINER_TIME)
                record_event(container_id, 'hostler_dropoff', env.now)
                print(f"Hostler {hostler_id} drops off inbound container {container_id} from chassis {chassis_id} to parking area at {env.now}")

                end_time = env.now
                record_vehicle_event('hostler', hostler_id_counter, 'end', end_time)  # performance record

                yield env.process(notify_truck(env, truck_store, container_id, out_gate_resource))

                # Check and process outbound container
                yield env.process(handle_outbound_container(env, hostlers, chassis, chassis_id, truck_store, cranes,
                                                            train_processing, outbound_containers_store,
                                                            in_gate_resource, oc_chassis_filled_event))

                # When all chassis are either filled with outbound container or empty, the cranes start loading
                if chassis_status.count(0) == OUTBOUND_CONTAINER_NUMBER and chassis_status.count(
                        -1) == TRAIN_UNITS - OUTBOUND_CONTAINER_NUMBER and not oc_chassis_filled_event.triggered:
                    print(f"Chassis is fully filled with OC, and cranes start moving: {chassis_status}")
                    print(f"where there are {chassis_status.count(0)} chassis is filled with OC (0)")
                    print(f"where there are {chassis_status.count(-1)} chassis is filled with empty (-1)")
                    print(f"where there are {chassis_status.count(1)} chassis is filled with IC (1)")
                    oc_chassis_filled_event.succeed()
                else:
                    print(f"Chassis is not fully filled: {chassis_status}")
                    print(f"where there are {chassis_status.count(0)} chassis is filled with OC (0)")
                    print(f"where there are {chassis_status.count(-1)} chassis is filled with empty (-1)")
                    print(f"where there are {chassis_status.count(1)} chassis is filled with IC (1)")
                    return


# When OC are fully processed, but IC are not
def hostler_transfer_IC_single_loop(env, hostlers, container_type, chassis, chassis_id, container_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource):
    print(f"Starting hostler_transfer_IC_single_loop for chassis {chassis_id} at {env.now}")
    global hostler_id_counter

    print(f"Requesting hostler for chassis {chassis_id} at {env.now}")

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
                d_t_dist = create_triang_distribution(d_t_min, d_t_avg, d_t_max).rvs()
                d_y_dist = create_triang_distribution(d_y_min, d_y_avg, d_y_max).rvs()
                HOSTLER_TRANSPORT_CONTAINER_TIME = (d_t_dist + d_y_dist) / (2 * HOSTLER_SPEED_LIMIT)

                yield env.timeout(HOSTLER_TRANSPORT_CONTAINER_TIME)
                # hostler picks up the rest of IC from the chassis
                chassis_status[chassis_id - 1] = -1
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


def handle_outbound_container(env, hostlers, chassis, chassis_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event):
    global HOSTLER_FIND_CONTAINER_TIME, HOSTLER_TRANSPORT_CONTAINER_TIME, chassis_status, hostler_id_counter, outbound_container_id_counter, outbound_containers_mapping

    hostler_id = hostler_id_counter
    hostler_id_counter = (hostler_id_counter % HOSTLER_NUMBER) + 1

    outbound_container_id = yield outbound_containers_store.get()

    if chassis_id not in outbound_containers_mapping:  # New mapping from outbound containers to chassis
        outbound_container_id = outbound_container_id
        outbound_containers_mapping[chassis_id] = outbound_container_id
        chassis_status[chassis_id - 1] = 0
        print(f"New mapping created: outbound container {outbound_container_id} to chassis {chassis_id} at {env.now}")

    outbound_container_id = outbound_containers_mapping[chassis_id]
    d_find_dist = create_triang_distribution(0, 0.5*(A+B), (A+B)).rvs()
    HOSTLER_FIND_CONTAINER_TIME = d_find_dist / (2 * TRUCK_SPEED_LIMIT)
    yield env.timeout(HOSTLER_FIND_CONTAINER_TIME)
    record_event(outbound_container_id, 'hostler_pickup', env.now)
    print(f"Hostler {hostler_id} brings back outbound container {outbound_container_id} from parking area to chassis {chassis_id} at {env.now}")
    yield env.timeout(HOSTLER_TRANSPORT_CONTAINER_TIME)
    record_event(outbound_container_id, 'hostler_dropoff', env.now)
    print(f"Hostler {hostler_id} drops off outbound container {outbound_container_id} from parking area to chassis {chassis_id} at {env.now}")


def notify_truck(env, truck_store, container_id, out_gate_resource):

    truck_id = yield truck_store.get()
    yield env.timeout(TRUCK_INGATE_TIME)
    print(f"Truck {truck_id} arrives at parking area at {env.now}")
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
    d_g_dist = create_triang_distribution(d_t_min, d_t_avg, d_t_max).rvs()
    TRUCK_TRANSPORT_CONTAINER_TIME = d_g_dist / (2 * TRUCK_SPEED_LIMIT)
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
    if env.now < TRAIN_DEPARTURE_HR:
        yield env.timeout(TRAIN_DEPARTURE_HR - env.now)
    yield env.timeout(TRAIN_INSPECTION_TIME)
    print(f"Train {train_id} departs at {env.now}")


def run_simulation():
    global all_trucks_ready_event

    print(f"Starting simulation with No.{TRAIN_ID} trains, {HOSTLER_NUMBER} hostlers, {CRANE_NUMBER} cranes, and {TRUCK_NUMBERS} trucks.")
    env = simpy.Environment()

    # Resources
    train_processing = simpy.Resource(env, capacity=1)
    cranes = simpy.Resource(env, capacity=CRANE_NUMBER)
    chassis = simpy.Resource(env, capacity=CHASSIS_NUMBER)
    hostlers = simpy.Resource(env, capacity=HOSTLER_NUMBER)
    in_gate_resource = simpy.Resource(env, capacity=IN_GATE_NUMBERS)
    out_gate_resource = simpy.Resource(env, capacity=OUT_GATE_NUMBERS)
    outbound_containers_store = simpy.Store(env, capacity=OUTBOUND_CONTAINER_NUMBER)
    truck_store = simpy.Store(env, capacity=TRUCK_NUMBERS)

    # Initialize trucks
    for truck_id in range(1, TRUCK_NUMBERS + 1):
        truck_store.put(truck_id)

    all_trucks_ready_event = env.event()

    train_timetable = [
        {"train_id": 19, "arrival_time": 187, "departure_time": 200, "empty_cars": 3, "full_cars":7, "oc_number": 2, "truck_number":7 },
        {"train_id": 25, "arrival_time": 250, "departure_time": 350, "empty_cars": 4, "full_cars":6, "oc_number": 2, "truck_number":6 },
        {"train_id": 49, "arrival_time": 400, "departure_time": 600, "empty_cars": 5, "full_cars":5, "oc_number": 2, "truck_number":5 },
        {"train_id": 60, "arrival_time": 650, "departure_time": 750, "empty_cars": 6, "full_cars":4, "oc_number": 2, "truck_number":4 },
        {"train_id": 12, "arrival_time": 800, "departure_time": 1000, "empty_cars": 7, "full_cars":3, "oc_number": 4, "truck_number":4 },
    ]

    # # REAL TEST
    # train_timetable = train_timetable(terminal)

    # env.process(train_arrival(env, train_processing, cranes, in_gate_resource, outbound_containers_store, truck_store, train_timetable))
    env.process(train_arrival(env, train_timetable, train_processing, cranes, hostlers, chassis, in_gate_resource,
                  outbound_containers_store, truck_store, out_gate_resource))

    env.run(until=SIM_TIME)

    print(f"Average train processing time: {sum(time_per_train) / len(time_per_train) if time_per_train else 0:.2f}")
    print("Simulation completed. ")

if __name__ == "__main__":
    run_simulation()