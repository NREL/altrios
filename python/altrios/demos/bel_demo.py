# %%
# Script for running the Wabtech BEL consist for sample data from Barstow to Stockton
# Consist comprises [2X Tier 4](https://www.wabteccorp.com/media/3641/download?inline)
# + [1x BEL](https://www.wabteccorp.com/media/466/download?inline)


import altrios as alt
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns

sns.set_theme()

SHOW_PLOTS = alt.utils.show_plots()


SAVE_INTERVAL = 1


pt = alt.PowerTrace.default()

res = alt.ReversibleEnergyStorage.from_file(
    alt.resources_root() / 
        "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
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

# instantiate battery model
t0 = time.perf_counter()
sim = alt.LocomotiveSimulation(bel, pt, SAVE_INTERVAL)
t1 = time.perf_counter()
print(f"Time to load: {t1-t0:.3g}")

# simulate
t0 = time.perf_counter()
sim.walk()
t1 = time.perf_counter()
print(f"Time to simulate: {t1-t0:.5g}")


bel_rslt = sim.loco_unit
t_s = np.array(sim.power_trace.time_seconds)

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 12))

# power
fontsize = 16

i = 0

ax[i].plot(
    t_s,
    np.array(bel_rslt.res.history.pwr_out_chemical_watts) * 1e-6,
    label="pwr_out_chem",
)
ax[i].plot(
    t_s,
    np.array(bel_rslt.history.pwr_out_watts) * 1e-6,
    label="loco pwr_out",
)
ax[i].plot(
    t_s,
    np.array(sim.power_trace.pwr_watts) * 1e-6,
    linestyle="--",
    label="power_trace",
)

ax[i].tick_params(labelsize=fontsize)

ax[i].set_ylabel("Power [MW]", fontsize=fontsize)
ax[i].legend(fontsize=fontsize)

i += 1
ax[i].plot(
    t_s,
    np.array(sim.loco_unit.history.pwr_out_watts),
)
ax[i].set_ylabel("Total Tractive\nEffort [MW]", fontsize=fontsize)

i += 1
ax[i].plot(t_s, np.array(bel_rslt.res.history.soc), label="SOC")
ax[i].set_ylabel("SOC", fontsize=fontsize)
ax[i].tick_params(labelsize=fontsize)

if SHOW_PLOTS:
    plt.tight_layout()
    plt.show()
