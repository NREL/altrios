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

# Build the train config
rail_vehicle_loaded = alt.RailVehicle.from_file(
    alt.resources_root() / "rolling_stock/Manifest_Loaded.yaml")
rail_vehicle_empty = alt.RailVehicle.from_file(
    alt.resources_root() / "rolling_stock/Manifest_Empty.yaml")

# https://docs.rs/altrios-core/latest/altrios_core/train/struct.TrainConfig.html
train_config = alt.TrainConfig(
    rail_vehicles=[rail_vehicle_loaded, rail_vehicle_empty],
    n_cars_by_type={
        "Manifest_Loaded": 50,
        "Manifest_Empty": 50,
    },
    train_length_meters=None,
    train_mass_kilograms=None,
)

# Build the locomotive consist model
# instantiate battery model
# https://docs.rs/altrios-core/latest/altrios_core/consist/locomotive/powertrain/reversible_energy_storage/struct.ReversibleEnergyStorage.html#
res = alt.ReversibleEnergyStorage.from_file(
    alt.resources_root() / "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
)

edrv = alt.ElectricDrivetrain(
    pwr_out_frac_interp=[0., 1.],
    eta_interp=[0.98, 0.98],
    pwr_out_max_watts=5e9,
    save_interval=SAVE_INTERVAL,
)

bel: alt.Locomotive = alt.Locomotive.from_pydict({
    "loco_type": {"BatteryElectricLoco": {
        "res": res.to_pydict(),
        "edrv": edrv.to_pydict(),
    }},
    "pwr_aux_offset_watts": 8.55e3,
    "pwr_aux_traction_coeff": 540.e-6,
    "force_max_newtons": 667.2e3,
    "mass_kilograms": alt.LocoParams.default().to_pydict()['mass_kilograms'],
    "save_interval": SAVE_INTERVAL,
})

# construct a vector of one BEL and several conventional locomotives
loco_vec = [bel.clone()] + [alt.Locomotive.default()] * 7
# instantiate consist
loco_con = alt.Consist(
    loco_vec
)

# Instantiate the intermediate `TrainSimBuilder`
tsb = alt.TrainSimBuilder(
    train_id="0",
    origin_id="A",
    destination_id="B",
    train_config=train_config,
    loco_con=loco_con,
)

# Load the network and construct the timed link path through the network.  
network = alt.Network.from_file(
    alt.resources_root() / 'networks/simple_corridor_network.yaml')

location_map = alt.import_locations(alt.resources_root() / "networks/simple_corridor_locations.csv")
train_sim: alt.SetSpeedTrainSim = tsb.make_speed_limit_train_sim(
    location_map=location_map,
    save_interval=1,
)
train_sim.set_save_interval(SAVE_INTERVAL)
est_time_net, _consist = alt.make_est_times(train_sim, network)

timed_link_path = alt.run_dispatch(
    network,
    alt.SpeedLimitTrainSimVec([train_sim]),
    [est_time_net],
    False,
    False,
)[0]

t0 = time.perf_counter()
train_sim.walk_timed_path(
    network=network,
    timed_path=timed_link_path,
)
t1 = time.perf_counter()
print(f'Time to simulate: {t1 - t0:.5g}')
assert len(train_sim.history) > 1

# pull out solved locomotive for plotting convenience
loco0:alt.Locomotive = train_sim.loco_con.loco_vec.tolist()[0]

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    np.array(train_sim.history.pwr_whl_out_watts) / 1e6,
    label="tract pwr",
)
ax[0].set_ylabel('Power [MW]')
ax[0].legend()

ax[1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    np.array(train_sim.history.res_aero_newtons) / 1e3,
    label='aero',
)
ax[1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    np.array(train_sim.history.res_rolling_newtons) / 1e3,
    label='rolling',
)
ax[1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    np.array(train_sim.history.res_curve_newtons) / 1e3,
    label='curve',
)
ax[1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    np.array(train_sim.history.res_bearing_newtons) / 1e3,
    label='bearing',
)
ax[1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    np.array(train_sim.history.res_grade_newtons) / 1e3,
    label='grade',
)
ax[1].set_ylabel('Force [MN]')
ax[1].legend()

ax[2].plot(
    np.array(train_sim.history.time_seconds) / 3_600, 
    np.array(loco0.res.history.soc)
)

ax[2].set_ylabel('SOC')
ax[2].legend()

ax[-1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    train_sim.history.speed_meters_per_second,
    label='achieved'
)
ax[-1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    train_sim.history.speed_limit_meters_per_second,
    label='limit'
)
ax[-1].set_xlabel('Time [hr]')
ax[-1].set_ylabel('Speed [m/s]')
ax[-1].legend()
plt.suptitle("Speed Limit Train Sim Demo")
if SHOW_PLOTS:
    plt.tight_layout()
    plt.show()

