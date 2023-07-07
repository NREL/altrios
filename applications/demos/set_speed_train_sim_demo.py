# %%
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

import altrios as alt 

# %%

save_interval = 1

train_sim = alt.SetSpeedTrainSim.default()
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
