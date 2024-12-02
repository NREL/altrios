import polars
import simpy
from dataclasses import dataclass, field
from lifts.schedule import *

def train_arrival_parameters(train_consist_plan, terminal, train_id_counter):
    timetable = build_train_timetable(train_consist_plan, terminal, swap_arrive_depart = False, as_dicts = False)
    TRAIN_TIMETABLE = timetable.iloc[train_id_counter-1]

    return TRAIN_TIMETABLE

@dataclass
class LiftsState:
    # Simulation parameters
    random_seed: int = 42
    sim_time: int = 1100
    terminal: str = 'Allouez'    # choose 'Hibbing' or 'Allouez'
    # Counting vehicles
    train_id_counter: int = 1
    crane_id_counter: int  = 1
    hostler_id_counter: int  = 1
    truck_id_counter: int  = 1
    total_initial_oc_trucks: int  = 1
    empty_truck_id_counter: int  = 1
    # inbound container counting
    inbound_containers_processed: int  = 0    # trucks drop off OC to chassis
    inbound_containers_hostler_processed: int  = 0    # hostlers pick up IC from chassis
    inbound_container_id_counter: int  = 1
    # outbound container counting
    outbound_container_id_counter: int  = 10001
    outbound_containers_processed: int  = 0   # trucks pick up IC from chassis
    outbound_containers_hostler_processed: int  = 0   # hostlers drop off OC to chassis
    outbound_container_id: int  = 0   # initialize OC id for chassis assignment
    record_oc_label: int  = 10001     # update outbound_containers_mapping among trains
    oc_variance: int  = 0     # record previous batch OC numbers cumulatively
    # yield events or conditions
    all_trucks_ready_event: simpy.events.Event = None  # initialize trucks
    train_has_arrived_event: simpy.events.Event = None    # crane starts working after the train arrives
    train_departure_event: simpy.events.Event = None    # train arrives after the last train departs
    oc_chassis_filled_event: simpy.events.Event = None  # outbound containers fill available chassis before cranes load
    # Trains
    # TRAIN_UNITS = int(input("Enter the number of train units: "))
    # TRAIN_ARRIVAL_MEAN = 10
    TRAIN_INSPECTION_TIME: float = 10/60    # hr
    previous_train_departure: float = 0
    train_series: int = 0
    time_per_train: list[float] = field(default_factory = lambda: [])
    train_delay_time: list[float] = field(default_factory = lambda: [])
    # Containers
    CONTAINERS_PER_CAR: int = 1
    CONTAINER_LEN: float = 20   # 1 TEU = 20 ft long, 8 ft wide, and 8.6 ft tall
    CONTAINER_WID: float = 8
    CONTAINER_TAL: float = 8.6
    container_events: dict = field(default_factory = lambda: {})   # Dictionary to store container event data
    # Cranes
    # CRANE_NUMBER = int(input("Enter the number of crane: "))
    CRANE_NUMBER: int = 10
    CONTAINERS_PER_CRANE_MOVE_MEAN: float = 600   # 10ft/min = 600 ft/hr, crane speed
    CRANE_MOVE_DEV_TIME: float = 5/60 # hr
    outbound_containers_mapping: dict = field(default_factory = lambda: {})  # To keep track of outbound containers ID mapped to chassis
    # Hostlers
    # HOSTLER_NUMBER = int(input("Enter the number of hostler: "))
    HOSTLER_NUMBER: int = 50
    CONTAINERS_PER_HOSTLER: int = 1  # hostler capacity
    HOSTLER_SPEED_LIMIT: float = 20*5280   # ft/hr
    HOSTLER_TRANSPORT_CONTAINER_TIME: float = 0    # hr, triangular distribution
    HOSTLER_FIND_CONTAINER_TIME: float = 0  # hr, triangular distribution
    # Trucks
    TRUCK_ARRIVAL_MEAN: float = 40/60   # hr, calculate by
    TRUCK_INGATE_TIME: float = 1/60    # hr
    TRUCK_OUTGATE_TIME: float = 2/60    # hr
    TRUCK_INGATE_TIME_DEV: float = 1/60    # hr
    TRUCK_OUTGATE_TIME_DEV: float = 1/60    # hr
    TRUCK_TO_PARKING: float = 2/60    # hr
    TRUCK_SPEED_LIMIT: float = 20*5280   # ft/hr
    TRUCK_TRANSPORT_CONTAINER_TIME: float = 0  # hr, triangular distribution
    # Gate settings
    IN_GATE_NUMBERS: int = 6  # test queuing module with 1; normal operations with 6
    OUT_GATE_NUMBERS: int = 6
    last_leave_time: float = 0
    truck_arrival_time: list[float] = field(default_factory = lambda: [])
    truck_waiting_time: list[float] = field(default_factory = lambda: [])
    train_consist_plan: pl.DataFrame = field(default_factory = lambda: pl.DataFrame())

    def initialize_from_consist_plan(self, train_consist_plan):
        self.train_consist_plan = train_consist_plan
        self.TRAIN_TIMETABLE = train_arrival_parameters(self.train_consist_plan, self.terminal, self.train_id_counter)
        self.TRAIN_ID = int(self.TRAIN_TIMETABLE['train_id'])
        self.TRAIN_ID_FIXED = 0
        self.CARS_LOADED_ARRIVAL = int(float(self.TRAIN_TIMETABLE['full_cars']))
        self.CARS_EMPTY_ARRIVAL = int(float(self.TRAIN_TIMETABLE['empty_cars']))
        self.TRAIN_ARRIVAL_HR = self.TRAIN_TIMETABLE['arrival_time']
        self.TRAIN_DEPARTURE_HR = self.TRAIN_TIMETABLE['departure_time']
        self.TRAIN_UNITS = self.CARS_LOADED_ARRIVAL + self.CARS_EMPTY_ARRIVAL
        self.TRAIN_SPOTS = self.TRAIN_UNITS

        # Containers
        self.INBOUND_CONTAINER_NUMBER = self.CARS_LOADED_ARRIVAL
        #df = outbound_containers()
        # TODO: confirm expected source of Outbound_Num; expected input file not available
        self.OUTBOUND_CONTAINER_NUMBER = self.INBOUND_CONTAINER_NUMBER#df.loc[df['Train_ID'] == TRAIN_ID, 'Outbound_Num'].values[0]

        # Chassis
        self.CHASSIS_NUMBER = self.TRAIN_UNITS
        self.chassis_status = [-1] * self.CHASSIS_NUMBER  # -1 means empty, 1 means inbound container, 0 means outbound container

        # Trucks
        self.TRUCK_NUMBERS = max(self.INBOUND_CONTAINER_NUMBER, self.OUTBOUND_CONTAINER_NUMBER)
        self.IN_OUT_GAP = abs(self.INBOUND_CONTAINER_NUMBER - self.OUTBOUND_CONTAINER_NUMBER)

    def initialize(self):
        self.CRANE_LOAD_CONTAINER_TIME_MEAN = (self.CONTAINERS_PER_CAR*(2*self.CONTAINER_TAL+self.CONTAINER_WID))/self.CONTAINERS_PER_CRANE_MOVE_MEAN   # hr
        self.CRANE_UNLOAD_CONTAINER_TIME_MEAN = (self.CONTAINERS_PER_CAR*(2*self.CONTAINER_TAL+self.CONTAINER_WID))/self.CONTAINERS_PER_CRANE_MOVE_MEAN # hr
        self.hostler_status = [-1] * self.HOSTLER_NUMBER   # 1 means trackside, 0 means parking side, -1 means hostler resources side
        # Trains
        if self.train_consist_plan.height > 0:
            self.initialize_from_consist_plan()

    def __post_init__(self):
        self.initialize()

state = LiftsState()

