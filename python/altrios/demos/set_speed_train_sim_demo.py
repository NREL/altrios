# %%
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import altrios as alt 
sns.set()

SHOW_PLOTS = alt.utils.show_plots()

SAVE_INTERVAL = 1

# Question: what happens to type, length, and mass when this is instantiated? TrainSummary does not
# have locos so mass and length may not be fully known yet.  If they are, in fact, not known, we
# should hide these fields from python
train_summary = alt.TrainSummary(
    rail_vehicle_type="Manifest", # maybe make it so that you could provide the rail vehicle file or the type
    cars_empty=50,
    cars_loaded=50,
    # what is `train_type` used for?  It has overlap with railcar type  
    # Geordie: this is here because Geordie couldn't specify the enum in python. It is only used
    # w.r.t. speed limits -- could go in rail vehicle csv file, but might create trouble if you have
    # multiple rail vehicle types per train, which is pretty much never a thing  
    # TODO: move `train_type` to rail vehicle file
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

# TODO: make it so that a BatteryElectricLocomotive can be instantiated here and then passed in to
# `alt.Locomotive(...)`
bel: alt.Locomotive = alt.Locomotive.build_battery_electric_loco(
    reversible_energy_storage=res,
    drivetrain=edrv,
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

# TODO: `dt` in this struct may not get used anywhere or may not be needed  
# Check whether this is just initial or the dt for the whole time and consider moving to TSB or TS
init_train_state = alt.InitTrainState()

tsb = alt.TrainSimBuilder(
    # TODO: make sure `train_id` is being used meaningfully
    train_id="0",
    # Question: what happens if we use arbitrary nonsense for `origin_id` and `destination_id`?
    origin_id="Minneapolis",
    destination_id="Superior",
    train_summary=train_summary,
    loco_con=loco_con,
    init_train_state=init_train_state,
)

# TODO: make sure rail_vehicle_map can be constructed from yaml file and such
rail_vehicle_file = "rolling_stock/rail_vehicles.csv"
rail_vehicle_map = alt.import_rail_vehicles(
    str(alt.resources_root() / rail_vehicle_file)
)

network = alt.import_network(str(alt.resources_root() / "networks/Taconite.yaml"))
# TODO: explain how this file was created from running `sim_manager_demo.py` and getting the first
# simulation
link_points = pd.read_csv(
    alt.resources_root() / "demo_data/link_points.csv")["link points"].tolist()
# TODO: if possible, make a way to generate this directly from `link_points`
link_path = [alt.LinkIdx(int(lp)) for lp in link_points]

# TODO: uncomment and fix
# speed_trace = alt.SpeedTrace.from_csv_file(
#     str(alt.resources_root() / "speed_trace.csv")
# )
df_speed_trace = pd.read_csv(alt.resources_root() / "demo_data/speed_trace.csv")
speed_trace = alt.SpeedTrace(
    df_speed_trace['time_seconds'],
    df_speed_trace['speed_meters_per_second'],
    None,
)

train_sim: alt.SetSpeedTrainSim = tsb.make_set_speed_train_sim(
    rail_vehicle_map=rail_vehicle_map,
    network=network,
    link_path=link_path,
    speed_trace=speed_trace,
    save_interval=SAVE_INTERVAL,
)
alt.set_param_from_path(train_sim, "state.time_seconds", 604_800.0)

train_sim.set_save_interval(1)
t0 = time.perf_counter()
train_sim.walk()
t1 = time.perf_counter()
print(f'Time to simulate: {t1 - t0:.5g}')

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
    train_sim.speed_trace.speed_meters_per_second,
)
ax[-1].set_xlabel('Time [hr]')
ax[-1].set_ylabel('Speed [m/s]')

if SHOW_PLOTS:
    plt.tight_layout()
    plt.show()
