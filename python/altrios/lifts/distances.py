import numpy as np
from altrios.lifts.single_track_parameters import *
from scipy.stats import triang, uniform
import math

# Yard setting: optimal layout output
YARD_TYPE = 'parallel'  # choose 'perpendicular' or 'parallel'
M = 2 # decide the number of rows of parking blocks in the layout
N = 2 # decide the number of columns of parking blocks in the layout
n_t = 2 # decide the numbers of train side aisles per group
n_p = 2 # decide the numbers of parking area aisles per group
n_r = 6  # decide the number of spots within each parking block (10 * n_r = BL_l, the length of each parking block)

# Fixed yard parameters
P = 10  # fixed aisle width
BL_w = 80  # fixed block width

A = M * 10 * n_r + (M+1) * n_p * P  # the vertical width of the yard
B = N * 80 + (N+1) * n_p * P # the horizontal length of the yard

# Total length of yard lanes, used to estimate density (veh/ft)
total_lane_length = A * (N + 1) + B * (M + 1)  # total distances of lanes

def speed_density(count, vehicle_type):
    '''
    Unit of speed: ft/s
    '''
    if vehicle_type == 'hostler':
        speed = 3.80 * math.e ** (-3.34 * count)
    elif vehicle_type == 'truck':
        speed = 10.22 * math.e ** (-10.98 * count)
    else:
        raise ValueError("Invalid vehicle type. Choose 'hostler' or 'truck'.")
    return speed


from scipy.stats import uniform


def simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min, d_t_max):
    """
    Simulates the truck travel time based on uniform distribution and vehicle density.

    Parameters:
    - total_lane_length: Total lane length (ft)
    - d_t_min, d_t_max: Range for travel distance (ft)
    """

    # Generate truck travel distance from uniform distribution
    d_t_dist = uniform(loc=d_t_min, scale=(d_t_max - d_t_min)).rvs()

    # Calculate vehicle density
    current_veh_num = train_schedule["truck_number"] - len(terminal.truck_store.items)
    veh_density = current_veh_num / total_lane_length

    # Compute truck speed based on density
    truck_speed = speed_density(veh_density, 'truck')
    print(f"Current truck {truck_id} speed is {truck_speed} (m/s)")

    # Compute truck travel time in hours and convert to seconds
    truck_travel_time = (32 * d_t_dist) / (2 * truck_speed * 3600)  # ft/s
    print(f"Truck {truck_id} travel time {truck_travel_time} (hr)")

    return truck_travel_time


def simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max):
    global state
    # Generate hostler travel distance from uniform distribution
    d_h_dist = uniform(loc=d_h_min, scale=(d_h_max - d_h_min)).rvs()

    # Calculate vehicle density
    veh_density = current_veh_num / total_lane_length

    # Compute hostler speed based on density
    hostler_speed = speed_density(veh_density, 'hostler')
    print(f"Current hostler {hostler_id} speed is {hostler_speed} (m/s)")

    # Compute hostler travel time in hours and convert to seconds
    hostler_travel_time = (32 * d_h_dist) / (2 * hostler_speed * 3600)  # ft/s
    print(f"hostler {hostler_id} travel time {hostler_travel_time} (hr)")

    return hostler_travel_time

def simulate_reposition_travel(hostler_id, current_veh_num, total_lane_length, d_r_min, d_r_max):
    global state
    # Generate reposition travel distance from uniform distribution
    d_r_dist = uniform(loc=d_r_min, scale=(d_r_max - d_r_min)).rvs()

    # Calculate vehicle density
    veh_density = current_veh_num / total_lane_length

    # Compute reposition speed based on density
    hostler_speed = speed_density(veh_density, 'hostler')
    print(f"Current hostler {hostler_id} speed is {hostler_speed} (m/s)")

    # Compute hostler travel time in hours and convert to seconds
    hostler_reposition_travel_time = (32 * d_r_dist) / (2 * hostler_speed * 3600)  # ft/h
    print(f"hostler {hostler_id} travel time {hostler_reposition_travel_time} (hr)")

    return hostler_reposition_travel_time


def create_distribution(min_val, avg_val, max_val):
    c = (avg_val - min_val) / (max_val - min_val)
    return triang(c, loc=min_val, scale=(max_val - min_val))

def uniform_distribution(min_val, max_val):
    return uniform(loc=min_val, scale=(max_val - min_val))

def ugly_sigma(x):
    total_sum = 0
    for i in range(1, x):
        total_sum += 2 * i * (x - i)
    result = total_sum / (x ** 2)
    return result

def A(M, n_r, n_p):
    return M * 10 * n_r + (M+1) * n_p * P

def B(N, n_p):
    return N * 80 + (N+1) * n_p * P

# Distance estimation
if YARD_TYPE == 'parallel':
    # d_h: hostler distance
    d_h_min = n_t * P + 1.5 * n_p * P
    d_h_max = n_t * P + A(M, n_r, n_p) - n_p * P + B(N, n_p) - n_p * P

    # d_r: repositioning distance
    d_r_min = 5 * n_r + 40
    d_r_max = ugly_sigma(M) * (10 * n_r + n_p * P) + ugly_sigma(N) * (80 + n_p * P)

    # d_t: truck distance
    d_t_min = 0.5 * n_p * P
    d_t_max = B(N, n_p) - n_p * P + A(M, n_r, n_p) - n_p * P


elif YARD_TYPE == 'perpendicular':
    # d_h
    d_h_min = n_t * P + 1.5 * n_p * P
    d_h_avg = 10 * n_r * M + 80 * N + (M + N + 1.5) * n_p * P + 2 * n_t * P
    d_h_max = n_t * P + A(M, n_r, n_p) - n_p * P + B(N, n_p) - n_p * P

    # d_r
    d_r_min = 0
    d_r_avg = 5 * n_r + 40 + ugly_sigma(M) * (10 * n_r + n_p*P) + ugly_sigma(N) * (80 + n_p * P)
    d_r_max = 10 * n_r + 80 + A(M, n_r, n_p) - n_p * P + B(N, n_p) - n_p * P

    # d_t
    d_t_min = 1.5 * n_p * P
    d_t_avg = 0.5 * (B(N, n_p) + A(M, n_r, n_p) - 0.5 * n_p * P)
    d_t_max = B(N, n_p) + A(M, n_r, n_p) - 2 * n_p * P