# %%
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

import altrios as alt 
sns.set_theme()

# Uncomment and run `maturin develop --release --features logging` to enable logging, 
# which is needed because logging bogs the CPU and is off by default.
# alt.utils.set_log_level("DEBUG")

SHOW_PLOTS = alt.utils.show_plots()
PYTEST = os.environ.get("PYTEST", "false").lower() == "true"

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
# instantiate electric drivetrain (motors and any gearboxes)
# https://docs.rs/altrios-core/latest/altrios_core/consist/locomotive/powertrain/electric_drivetrain/struct.ElectricDrivetrain.html
edrv = alt.ElectricDrivetrain(
    pwr_out_frac_interp=[0., 1.],
    eta_interp=[0.98, 0.98],
    pwr_out_max_watts=5e9,
    save_interval=SAVE_INTERVAL,
)

loco_type = alt.BatteryElectricLoco(res, edrv)

bel: alt.Locomotive = alt.Locomotive(
    loco_type=loco_type,
    loco_params=alt.LocoParams.from_dict(dict(
        pwr_aux_offset_watts=8.55e3,
        pwr_aux_traction_coeff_ratio=540.e-6,
        force_max_newtons=667.2e3,
)))

# construct a vector of one BEL and several conventional locomotives
loco_vec = [bel] + [alt.Locomotive.default()] * 7
# instantiate consist
loco_con = alt.Consist(
    loco_vec,
    SAVE_INTERVAL,
)

# Instantiate the intermediate `TrainSimBuilder`
tsb = alt.TrainSimBuilder(
    train_id="0",
    train_config=train_config,
    loco_con=loco_con,
)

# Load the network and link path through the network.  
network = alt.Network.from_file(
    alt.resources_root() / "networks/Taconite.yaml")
network.set_speed_set_for_train_type(alt.TrainType.Freight)
link_path = alt.LinkPath.from_csv_file(
    alt.resources_root() / "demo_data/link_points_idx.csv"
)

# load the prescribed speed trace that the train will follow
speed_trace = alt.SpeedTrace.from_csv_file(
    alt.resources_root() / "demo_data/speed_trace.csv"
)

train_sim: alt.SetSpeedTrainSim = tsb.make_set_speed_train_sim(
    network=network,
    link_path=link_path,
    speed_trace=speed_trace,
    save_interval=SAVE_INTERVAL,
)

alt.utils.set_log_level("WARNING")

train_sim.set_save_interval(SAVE_INTERVAL)
t0 = time.perf_counter()
train_sim.walk()
t1 = time.perf_counter()
print(f'Time to simulate: {t1 - t0:.5g}')

fig, ax = plt.subplots(3, 1, sharex=True)
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

ax[-1].plot(
    np.array(train_sim.history.time_seconds) / 3_600,
    train_sim.speed_trace.speed_meters_per_second,
)
ax[-1].set_xlabel('Time [hr]')
ax[-1].set_ylabel('Speed [m/s]')

plt.suptitle("Set Speed Train Sim Demo")

if SHOW_PLOTS:
    plt.tight_layout()
    plt.show()

if PYTEST:
    # to access these checks, run `SHOW_PLOTS=f PYTEST=true python set_speed_train_sim_demo.py`
    import json
    json_path = alt.resources_root() / "test_assets/set_speed_ts_demo.json"
    with open(json_path, 'r') as file:
        train_sim_reference = json.load(file)

    dist_msg = f"`train_sim.state.total_dist_meters`: {train_sim.state.total_dist_meters}\n" + \
        f"`train_sim_reference['state']['total_dist']`: {train_sim_reference['state']['total_dist']}"
    energy_whl_out_msg = f"`train_sim.state.energy_whl_out_joules`: {train_sim.state.energy_whl_out_joules}\n" + \
        f"`train_sim_reference['state']['energy_whl_out']`: {train_sim_reference['state']['energy_whl_out']}"
    train_sim_fuel = train_sim.loco_con.get_energy_fuel_joules()
    train_sim_reference_fuel = sum(
        loco['loco_type']['ConventionalLoco']['fc']['state']['energy_fuel'] if 'ConventionalLoco' in loco['loco_type'] else 0 
        for loco in train_sim_reference['loco_con']['loco_vec']
    )
    fuel_msg = f"`train_sim_fuel`: {train_sim_fuel}\n`train_sim_referenc_fuel`: {train_sim_reference_fuel}"
    train_sim_net_res = train_sim.loco_con.get_net_energy_res_joules()
    train_sim_reference_net_res = sum(
        loco['loco_type']['BatteryElectricLoco']['res']['state']['energy_out_chemical'] if 'BatteryElectricLoco' in loco['loco_type'] else 0 
        for loco in train_sim_reference['loco_con']['loco_vec']
    )
    net_res_msg = f"`train_sim_net_res`: {train_sim_net_res}\n`train_sim_referenc_net_res`: {train_sim_reference_net_res}"

    # check total distance
    assert train_sim.state.total_dist_meters == train_sim_reference["state"]["total_dist"], dist_msg

    # check total tractive energy
    assert train_sim.state.energy_whl_out_joules == train_sim_reference["state"]["energy_whl_out"], energy_whl_out_msg

    # check consist-level fuel usage
    assert train_sim_fuel == train_sim_reference_fuel, fuel_msg

    # check consist-level battery usage
