# %%
import time
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import os

import altrios as alt 
sns.set_theme()

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

loco_type = alt.BatteryElectricLoco.from_pydict({
    "res": res.to_pydict(),
    "edrv": edrv.to_pydict(),
})

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
    alt.resources_root() / "networks/Taconite-NoBalloon.yaml")
link_path = alt.LinkPath.from_csv_file(
    alt.resources_root() / "demo_data/link_path.csv"
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

df = train_sim.to_dataframe()

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
    np.array(train_sim.speed_trace.speed_meters_per_second)[::SAVE_INTERVAL],
)
ax[-1].set_xlabel('Time [hr]')
ax[-1].set_ylabel('Speed [m/s]')

plt.suptitle("Set Speed Train Sim Demo")

if SHOW_PLOTS:
    plt.tight_layout()
    plt.show()

# whether to run assertions, enabled by default
ENABLE_ASSERTS = os.environ.get("ENABLE_ASSERTS", "true").lower() == "true"
# whether to override reference files used in assertions, disabled by default
ENABLE_REF_OVERRIDE = os.environ.get("ENABLE_REF_OVERRIDE", "false").lower() == "true"
# directory for reference files for checking sim results against expected results
ref_dir = alt.resources_root() / "demo_data/set_speed_train_sim_demo/"

if ENABLE_REF_OVERRIDE:
    ref_dir.mkdir(exist_ok=True, parents=True)
    df:pl.DataFrame = train_sim.to_dataframe().lazy().collect()[-1]
    df.write_csv(ref_dir / "to_dataframe_expected.csv")
if ENABLE_ASSERTS:
    print("Checking output of `to_dataframe`")
    to_dataframe_expected = pl.scan_csv(ref_dir / "to_dataframe_expected.csv").collect()[-1]
    assert to_dataframe_expected.equals(train_sim.to_dataframe()[-1]), \
        f"to_dataframe_expected: \n{to_dataframe_expected}\ntrain_sim.to_dataframe()[-1]: \n{train_sim.to_dataframe()[-1]}" + \
            "\ntry running with `ENABLE_REF_OVERRIDE=True`"
    print("Success!")
