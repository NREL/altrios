# %%
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

import altrios as alt 

# %%

SAVE_INTERVAL = 1


res = alt.ReversibleEnergyStorage.from_file(
    str(alt.resources_root() / 
        "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
    )
)
edrv = alt.ElectricDrivetrain(
    pwr_out_frac_interp=[0., 1.],
    eta_interp=[0.98, 0.98],
    pwr_out_max_watts=5e9,
    save_interval=SAVE_INTERVAL,
)

loco_params = alt.LocoParams.from_dict

loco_vec = [
    alt.Locomotive.build_battery_electric_loco(
        reversible_energy_storage=res,
        drivetrain=edrv,
        pwr_aux_offset_watts=8.55e3,
        pwr_aux_traction_coeff=540.e-6,
        force_max_newtons=None,
    )
]
loco_con = alt.Consist(
    loco_vec,
    SAVE_INTERVAL,
)
train_state = alt.TrainState.
speed_trace = alt.SpeedTrace.
train_res = alt.Train.
path_tpc = alt.

train_sim = alt.SetSpeedTrainSim.new(
    loco_con,
    train_state,
    speed_trace,
    train_res,
    path_tpc,
    SAVE_INTERVAL,
)
train_sim.set_save_interval(1)
t0 = time.perf_counter()
train_sim.walk()
t1 = time.perf_counter()
print(f'Time to simulate: {t1 - t0:.5g}')

# %%
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(
    train_sim.history.time_seconds,
    train_sim.history.pwr_whl_out_watts,
    label="tract pwr",
)
ax[0].set_ylabel('Power')
ax[0].legend()

ax[1].plot(
    train_sim.history.time_seconds,
    train_sim.history.res_aero_newtons,
    label='aero',
)
ax[1].plot(
    train_sim.history.time_seconds,
    train_sim.history.res_rolling_newtons,
    label='rolling',
)
ax[1].plot(
    train_sim.history.time_seconds,
    train_sim.history.res_curve_newtons,
    label='curve',
)
ax[1].plot(
    train_sim.history.time_seconds,
    train_sim.history.res_bearing_newtons,
    label='bearing',
)
ax[1].plot(
    train_sim.history.time_seconds,
    train_sim.history.res_grade_newtons,
    label='grade',
)
ax[1].set_ylabel('Force [N]')
ax[1].legend()

ax[-1].plot(
    train_sim.history.time_seconds,
    train_sim.speed_trace.speed_meters_per_second,
)
ax[-1].set_xlabel('Time [s]')
ax[-1].set_ylabel('Speed [m/s]')
# %%
