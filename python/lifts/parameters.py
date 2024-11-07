from schedule import *

# Simulation parameters
RANDOM_SEED = 42
SIM_TIME = 1100
terminal = 'Allouez'    # choose 'Hibbing' or 'Allouez'


# Counting vehicles
train_id_counter = 1
crane_id_counter = 1
hostler_id_counter = 1
truck_id_counter = 1
total_initial_oc_trucks = 1
empty_truck_id_counter = 1


# inbound container counting
inbound_containers_processed = 0    # trucks drop off OC to chassis
inbound_containers_hostler_processed = 0    # hostlers pick up IC from chassis
inbound_container_id_counter = 1

# outbound container counting
outbound_container_id_counter = 10001
outbound_containers_processed = 0   # trucks pick up IC from chassis
outbound_containers_hostler_processed = 0   # hostlers drop off OC to chassis
outbound_container_id = 0   # initialize OC id for chassis assignment
record_oc_label = 10001     # update outbound_containers_mapping among trains
oc_variance = 0     # record previous batch OC numbers cumulatively

# yield events or conditions
all_trucks_ready_event = None  # initialize trucks
train_has_arrived_event = None    # crane starts working after the train arrives
train_departure_event = None    # train arrives after the last train departs
oc_chassis_filled_event = None  # outbound containers fill available chassis before cranes load


# Trains
def train_arrival_parameters(terminal):
    global train_id_counter
    timetable = train_timetable(terminal)
    TRAIN_TIMETABLE = timetable.iloc[train_id_counter-1]

    return TRAIN_TIMETABLE


# TRAIN_UNITS = int(input("Enter the number of train units: "))
TRAIN_TIMETABLE = train_arrival_parameters(terminal)
TRAIN_ID = int(TRAIN_TIMETABLE['Train_ID'])
CARS_LOADED_ARRIVAL = int(float(TRAIN_TIMETABLE['Cars_Loaded']))
CARS_EMPTY_ARRIVAL = int(float(TRAIN_TIMETABLE['Cars_Empty']))
TRAIN_ARRIVAL_HR = TRAIN_TIMETABLE['Arrival_Time_Actual_Hr']
TRAIN_DEPARTURE_HR = TRAIN_TIMETABLE['Departure_Time_Actual_Hr']

TRAIN_UNITS = CARS_LOADED_ARRIVAL + CARS_EMPTY_ARRIVAL
TRAIN_SPOTS = TRAIN_UNITS
# TRAIN_ARRIVAL_MEAN = 10
TRAIN_INSPECTION_TIME = 10/60    # hr
previous_train_departure = 0
train_series = 0
time_per_train = []
train_delay_time = []


# Containers
CONTAINERS_PER_CAR = 1
CONTAINER_LEN = 20   # 1 TEU = 20 ft long, 8 ft wide, and 8.6 ft tall
CONTAINER_WID = 8
CONTAINER_TAL = 8.6
INBOUND_CONTAINER_NUMBER = CARS_LOADED_ARRIVAL
df = outbound_containers()
OUTBOUND_CONTAINER_NUMBER = df.loc[df['Train_ID'] == TRAIN_ID, 'Outbound_Num'].values[0]
container_events = {}   # Dictionary to store container event data


# Chassis
CHASSIS_NUMBER = TRAIN_UNITS
chassis_status = [-1] * CHASSIS_NUMBER  # -1 means empty, 1 means inbound container, 0 means outbound container


# Cranes
# CRANE_NUMBER = int(input("Enter the number of crane: "))
CRANE_NUMBER = 10
CONTAINERS_PER_CRANE_MOVE_MEAN = 600   # 10ft/min = 600 ft/hr, crane speed
CRANE_LOAD_CONTAINER_TIME_MEAN = (CONTAINERS_PER_CAR*(2*CONTAINER_TAL+CONTAINER_WID))/CONTAINERS_PER_CRANE_MOVE_MEAN   # hr
CRANE_UNLOAD_CONTAINER_TIME_MEAN = (CONTAINERS_PER_CAR*(2*CONTAINER_TAL+CONTAINER_WID))/CONTAINERS_PER_CRANE_MOVE_MEAN # hr
CRANE_MOVE_DEV_TIME = 5/60 # hr
outbound_containers_mapping = {}  # To keep track of outbound containers ID mapped to chassis


# Hostlers
# HOSTLER_NUMBER = int(input("Enter the number of hostler: "))
HOSTLER_NUMBER = 50
CONTAINERS_PER_HOSTLER = 1  # hostler capacity
HOSTLER_SPEED_LIMIT = 20*5280   # ft/hr
HOSTLER_TRANSPORT_CONTAINER_TIME = 0    # hr, triangular distribution
HOSTLER_FIND_CONTAINER_TIME = 0  # hr, triangular distribution
hostler_status = [-1] * HOSTLER_NUMBER   # 1 means trackside, 0 means parking side, -1 means hostler resources side


# Trucks
TRUCK_ARRIVAL_MEAN = 40/60   # hr, calculate by
TRUCK_INGATE_TIME = 1/60    # hr
TRUCK_OUTGATE_TIME = 2/60    # hr
TRUCK_INGATE_TIME_DEV = 1/60    # hr
TRUCK_OUTGATE_TIME_DEV = 1/60    # hr
TRUCK_TO_PARKING = 2/60    # hr
TRUCK_SPEED_LIMIT = 20*5280   # ft/hr
TRUCK_TRANSPORT_CONTAINER_TIME = 0  # hr, triangular distribution
TRUCK_NUMBERS = max(INBOUND_CONTAINER_NUMBER, OUTBOUND_CONTAINER_NUMBER)
IN_OUT_GAP = abs(INBOUND_CONTAINER_NUMBER - OUTBOUND_CONTAINER_NUMBER)



# Gate settings
IN_GATE_NUMBERS = 6  # test queuing module with 1; normal operations with 6
OUT_GATE_NUMBERS = 6
last_leave_time = 0
truck_arrival_time = []
truck_waiting_time = []