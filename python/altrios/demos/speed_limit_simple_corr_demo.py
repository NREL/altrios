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
res = alt.ReversibleEnergyStorage.from_file(
    alt.resources_root() / "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
)
# https://docs.rs/altrios-core/latest/altrios_core/train/struct.TrainConfig.html
train_config = alt.TrainConfig(
    cars_empty=50,
    cars_loaded=50,
    rail_vehicle_type="Manifest",
    train_type=None, 
    train_length_meters=None,
    train_mass_kilograms=None,
)

edrv = alt.ElectricDrivetrain(
    pwr_out_frac_interp=[0., 1.],
    eta_interp=[0.98, 0.98],
    pwr_out_max_watts=5e9,
    save_interval=SAVE_INTERVAL,
)

bel: alt.Locomotive = alt.Locomotive.build_battery_electric_loco(
    reversible_energy_storage=res,
    drivetrain=edrv,
    loco_params=alt.LocoParams.from_dict(dict(
        pwr_aux_offset_watts=8.55e3,
        pwr_aux_traction_coeff_ratio=540.e-6,
        force_max_newtons=667.2e3,
)))

# construct a vector of one BEL and several conventional locomotives
loco_vec = [bel.clone()] + [alt.Locomotive.default()] * 7
# instantiate consist
loco_con = alt.Consist(
    loco_vec
)


tsb = alt.TrainSimBuilder(
    train_id="0",
    origin_id="A",
    destination_id="B",
    train_config=train_config,
    loco_con=loco_con,
)

rail_vehicle_file = "rolling_stock/rail_vehicles.csv"
rail_vehicle_map = alt.import_rail_vehicles(alt.resources_root() / rail_vehicle_file)
rail_vehicle = rail_vehicle_map[train_config.rail_vehicle_type]

network = alt.Network.from_file(
    alt.resources_root() / 'networks/simple_corridor_network.yaml')

location_map = alt.import_locations(alt.resources_root() / "networks/simple_corridor_locations.csv")
train_sim: alt.SetSpeedTrainSim = tsb.make_speed_limit_train_sim(
    rail_vehicle=rail_vehicle,
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

<<<<<<< HEAD
=======

#Downhill Simulation
#last recorded soc value
#singular BEL:
train_config = alt.TrainConfig(
    cars_empty=Num_car_empty,
    cars_loaded=Num_car_loaded,
    rail_vehicle_type="Unit",
    train_type=None, 
    train_length_meters=None,
    train_mass_kilograms=None,
)
if num_BEL != 0:
    uphill_soc = loco0.res.state.soc

    # manually update soc value
    alt.set_param_from_path(
        #item, parameter to be changed, modified value
        bel, "res.state.soc", uphill_soc
    )
loco_vec = [bel.clone()] * num_BEL + [alt.Locomotive.default()] * num_diesel
# instantiate consist
loco_con = alt.Consist(
    loco_vec
)

tsb = alt.TrainSimBuilder(
    train_id="0",
    origin_id="B",
    destination_id="A",
    train_config=train_config,
    loco_con=loco_con,
)


network = alt.Network.from_file(
    alt.resources_root() / network_file_name)
#Multiple BELs:
#for loco in range(len(loco_vec)):


location_map = alt.import_locations(alt.resources_root() / "networks/simple_corridor_locations_new_format.csv")
train_sim_down: alt.SetSpeedTrainSim = tsb.make_speed_limit_train_sim(
    rail_vehicle=rail_vehicle,
    location_map=location_map,
    save_interval=1,
)
train_sim_down.set_save_interval(SAVE_INTERVAL)
est_time_net, _consist = alt.make_est_times(train_sim_down, network)

timed_link_path = alt.run_dispatch(
    network,
    alt.SpeedLimitTrainSimVec([train_sim_down]),
    [est_time_net],
    False,
    False,
)[0]

t0 = time.perf_counter()
train_sim_down.walk_timed_path(
    network=network,
    timed_path=timed_link_path,
)
t1 = time.perf_counter()
print(f'Time to simulate: {t1 - t0:.5g}')
assert len(train_sim_down.history) > 1

# pull out solved locomotive for plotting convenience
if num_BEL != 0:
    loco_down:alt.Locomotive = train_sim_down.loco_con.loco_vec.tolist()[0]
    print("second leg battery usage: ",loco_down.res.state.energy_out_electrical_joules/(3.6*10**9) * num_BEL, " MWh")
    print("second leg battery usage from `loco_vec`: ", np.sum([
        0. if loco.res == None else loco.res.state.energy_out_electrical_joules for loco in train_sim_down.loco_con.loco_vec.tolist()
    ]) / (3.6*10**9), " MWh")

if num_diesel != 0:
    diesel_loco_down:alt.Locomotive = train_sim_down.loco_con.loco_vec.tolist()[-1]
    print("second leg diesel usage: ",train_sim_down.get_energy_fuel_joules(False)/(3.6*10**9), " MWh")
    print("second leg diesel usage from `loco_vec`: ", np.sum([
        0. if loco.fc == None else loco.fc.state.energy_fuel_joules for loco in train_sim_down.loco_con.loco_vec.tolist()
    ]) / (3.6*10**9), " MWh")

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(
    np.array(train_sim_down.history.offset_back_meters),
    np.array(train_sim_down.history.pwr_whl_out_watts) / 1e6,
    label="tract pwr",
)
ax[0].set_ylabel('Power [MW]')
ax[0].legend()

ax[1].plot(
    np.array(train_sim_down.history.offset_back_meters),
    np.array(train_sim_down.history.res_aero_newtons) / 1e3,
    label='aero',
)
ax[1].plot(
    np.array(train_sim_down.history.offset_back_meters),
    np.array(train_sim_down.history.res_rolling_newtons) / 1e3,
    label='rolling',
)
ax[1].plot(
    np.array(train_sim_down.history.offset_back_meters),
    np.array(train_sim_down.history.res_curve_newtons) / 1e3,
    label='curve',
)
ax[1].plot(
    np.array(train_sim_down.history.offset_back_meters),
    np.array(train_sim_down.history.res_bearing_newtons) / 1e3,
    label='bearing',
)
ax[1].plot(
    np.array(train_sim_down.history.offset_back_meters),
    np.array(train_sim_down.history.res_grade_newtons) / 1e3,
    label='grade',
)
ax[1].set_ylabel('Force [MN]')
ax[1].legend()
if num_BEL != 0:
    ax[2].plot(
        np.array(train_sim_down.history.offset_back_meters), 
        np.array(loco_down.res.history.soc)
    )

    ax[2].set_ylabel('SOC')
    ax[2].legend()

ax[-1].plot(
    np.array(train_sim_down.history.offset_back_meters),
    train_sim_down.history.speed_meters_per_second,
    label='achieved'
)
ax[-1].plot(
    np.array(train_sim_down.history.offset_back_meters),
    train_sim_down.history.speed_limit_meters_per_second,
    label='limit'
)
ax[-1].set_xlabel('Distance (m)')
ax[-1].set_ylabel('Speed [m/s]')
ax[-1].legend()
plt.suptitle(code_name + " Train Sim Demo")
if SHOW_PLOTS:
    plt.tight_layout()
    plt.show()


print(train_sim_down.state.mass_adj_kilograms/907.185)
>>>>>>> 426aec0f (energy calcs robust to loco_vec length)
# %%
