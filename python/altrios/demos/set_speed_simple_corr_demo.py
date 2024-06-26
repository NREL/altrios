# %%
"""
SetSpeedTrainSim over a simple, hypothetical corridor
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import altrios as alt 
sns.set_theme()

SHOW_PLOTS = alt.utils.show_plots()

SAVE_INTERVAL = 1

# https://docs.rs/altrios-core/latest/altrios_core/train/struct.TrainConfig.html
train_config = alt.TrainConfig(
    cars_empty=50,
    cars_loaded=50,
    rail_vehicle_type="Manifest",
    train_type=None,
    train_length_meters=None,
    train_mass_kilograms=None,
)

bel: alt.Locomotive = alt.Locomotive.default_battery_electric_loco()

# construct a vector of one BEL and several conventional locomotives
loco_vec = [bel] + [alt.Locomotive.default()] * 7
# instantiate consist
loco_con = alt.Consist(
    loco_vec,
    SAVE_INTERVAL,
)

tsb = alt.TrainSimBuilder(
    train_id="0",
    train_config=train_config,
    loco_con=loco_con,
)

rail_vehicle_file = "rolling_stock/rail_vehicles.csv"
rail_vehicle_map = alt.import_rail_vehicles(alt.resources_root() / rail_vehicle_file)
rail_vehicle = rail_vehicle_map[train_config.rail_vehicle_type]

network = alt.Network.from_file(
    alt.resources_root() / 'networks/simple_corridor_network.yaml')
# This data in this file were generated by running 
# ```python
# [lp.link_idx.idx for lp in sim0.path_tpc.link_points]
# ``` 
# in sim_manager_demo.py.
link_path = alt.LinkPath.from_csv_file(
    alt.resources_root() / "demo_data/link_points_idx_simple_corridor.csv")


speed_trace = alt.SpeedTrace.from_csv_file(
    alt.resources_root() / "demo_data/speed_trace_simple_corridor.csv"
)

train_sim: alt.SetSpeedTrainSim = tsb.make_set_speed_train_sim(
    rail_vehicle=rail_vehicle,
    network=network,
    link_path=link_path,
    speed_trace=speed_trace,
    save_interval=SAVE_INTERVAL,
)

train_sim.set_save_interval(1)
t0 = time.perf_counter()
train_sim.walk()
t1 = time.perf_counter()
print(f'Time to simulate: {t1 - t0:.5g}')

# pull out solved locomotive for plotting convenience
loco0:alt.Locomotive = train_sim.loco_con.loco_vec.tolist()[0]

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    train_sim.history.pwr_whl_out_watts,
    label="tract pwr",
)
ax[0].set_ylabel('Power')
ax[0].legend()

ax[1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    train_sim.history.res_aero_newtons,
    label='aero',
)
ax[1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    train_sim.history.res_rolling_newtons,
    label='rolling',
)
ax[1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    train_sim.history.res_curve_newtons,
    label='curve',
)
ax[1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    train_sim.history.res_bearing_newtons,
    label='bearing',
)
ax[1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    train_sim.history.res_grade_newtons,
    label='grade',
)
ax[1].set_ylabel('Force [N]')
ax[1].legend()

ax[-1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    train_sim.speed_trace.speed_meters_per_second,
)
ax[-1].set_xlabel('Time [hr]')
ax[-1].set_ylabel('Speed [m/s]')

ax[2].plot(
    np.array(train_sim.history.time_seconds) / 3_600, 
    np.array(loco0.res.history.soc)
)

ax[2].set_ylabel('SOC')
ax[2].legend()

plt.suptitle("Set Speed Train Sim Demo")
plt.tight_layout()

if SHOW_PLOTS:
    fig.show()

# %%
