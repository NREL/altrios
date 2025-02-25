import numpy as np
from scipy.stats import triang

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

def speed_density(count):
    speed = 10 / count  # density-speed function: calculate the current speed of the vehicle
    return speed

def create_triang_distribution(min_val, avg_val, max_val):
    c = (avg_val - min_val) / (max_val - min_val)
    return triang(c, loc=min_val, scale=(max_val - min_val))

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
    d_h_max = n_t * P + A(M, n_r, n_p) + B(N, n_p)
    d_h_avg = 0.5 * (d_h_min + d_h_max)

    # d_r: repositioning distance
    d_r_min = 0
    # d_r_avg = 5 * n_r + 40 + ugly_sigma(M) * (10 * n_r + n_p*P) + ugly_sigma(N) * (80 + n_p * P)
    d_r_max = 10 * n_r + 80 + A(M, n_r, n_p) - n_p * P + B(N, n_p) - n_p * P
    d_r_avg = 0.5 * (d_r_min + d_r_max)

    # d_t: truck distance
    d_t_min = 1.5 * n_p * P
    # d_t_avg = 0.5 * (B(N, n_p) + A(M, n_r, n_p) - 0.5 * n_p * P)
    d_t_max = B(N, n_p) + A(M, n_r, n_p) - 2 * n_p * P
    d_t_avg = 0.5 * (d_t_min + d_t_max)


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