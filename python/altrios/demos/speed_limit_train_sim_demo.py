# %%
import altrios as alt
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
sns.set()

SHOW_PLOTS = alt.utils.show_plots()

SAVE_INTERVAL = 100

train_config = alt.TrainConfig(
    rail_vehicle_type="Manifest",
    cars_empty=50,
    cars_loaded=50,
    train_type=None,
    train_length_meters=None,
    train_mass_kilograms=None,
)

# instantiate battery model
res = alt.ReversibleEnergyStorage.from_file(
    str(alt.resources_root() / 
        "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
    )
)
# instantiate electric drivetrain (motors and any gearboxes)
edrv = alt.ElectricDrivetrain(
    pwr_out_frac_interp=[0., 1.],
    eta_interp=[0.98, 0.98],
    pwr_out_max_watts=5e9,
    save_interval=SAVE_INTERVAL,
)

bel = alt.Locomotive.build_battery_electric_loco(
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
    loco_vec,
)
init_train_state = alt.InitTrainState(
    # this corresponds to middle week of simulation period in sim_manager_demo.py
    time_seconds=604_800.0,
)

tsb = alt.TrainSimBuilder(
    train_id="0",
    origin_id="Minneapolis",
    destination_id="Superior",
    train_config=train_config,
    loco_con=loco_con,
    init_train_state=init_train_state,
)

# make sure rail_vehicle_map can be constructed from yaml file and such
rail_vehicle_file = "rolling_stock/" + train_config.rail_vehicle_type + ".yaml"
rail_vehicle = alt.RailVehicle.from_file(
    str(alt.resources_root() / rail_vehicle_file)
)

network = alt.import_network(
    str(alt.resources_root() / "networks/Taconite-NoBalloon.yaml"))

location_map = alt.import_locations(
    str(alt.resources_root() / "networks/default_locations.csv")
)

train_sim: alt.SpeedLimitTrainSim = tsb.make_speed_limit_train_sim(
    rail_vehicle=rail_vehicle,
    location_map=location_map,
    save_interval=1,
)
train_sim.set_save_interval(SAVE_INTERVAL)

timed_path = alt.LinkIdxTimeVec.from_file(
    str(alt.resources_root() / "demo_data/timed_path.yaml")
)

# %%

t0 = time.perf_counter()
train_sim.walk_timed_path(
    network=network,
    timed_path=timed_path,
)
t1 = time.perf_counter()
print(f'Time to simulate: {t1 - t0:.5g}')
assert len(train_sim.history) > 1

fig, ax = plt.subplots(3, 1, sharex=True)
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
    train_sim.history.speed_meters_per_second,
)
ax[-1].set_xlabel('Time [hr]')
ax[-1].set_ylabel('Speed [m/s]')
plt.suptitle("Speed Limit Train Sim Demo")
if SHOW_PLOTS:
    plt.tight_layout()
    plt.show()
# Impact of sweep of battery capacity

# %%
