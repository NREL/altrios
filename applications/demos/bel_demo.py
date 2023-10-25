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

sns.set()

SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"


# %%
SAVE_INTERVAL = 1
# load hybrid consist
t0 = time.perf_counter()
res = alt.ReversibleEnergyStorage.default()
# uncomment to trigger error
# altpy.set_param_from_path(
#     res,
#     "pwr_out_max_watts",
#     res.pwr_out_max_watts / 10.
# )

# from_file(
#     str(altpy.resources_root() /
#         "powertrains" /
#         "reversible_energy_storages" /
#         "Kokam_NMC_75Ah_flx_drive.yaml")
# )

res = alt.set_param_from_path(res, "state.soc", 0.95)
edrv = alt.ElectricDrivetrain.default()

# .from_file(
#     str(altpy.resources_root() /
#         "powertrains" /
#         "electric_drivetrains" /
#         "edrv_default.yaml")
# )

loco_params = alt.LocoParams.from_dict({'pwr_aux_offset_watts':13e3,
        'pwr_aux_traction_coeff':1.1e-3,
        'force_max_newtons':667.2e3,
        'mass_kg': 1e6})

bel = alt.Locomotive.build_battery_electric_loco(
    reversible_energy_storage=res,
    drivetrain=edrv,
    loco_params=loco_params,
    save_interval=SAVE_INTERVAL,
)


# %%

pt = alt.PowerTrace.default()

sim = alt.LocomotiveSimulation(bel, pt, SAVE_INTERVAL)
t1 = time.perf_counter()

print(f"Time to load: {t1-t0:.3g}")

# simulate
t0 = time.perf_counter()
sim.walk()
t1 = time.perf_counter()
print(f"Time to simulate: {t1-t0:.5g}")


# %%


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
# %%
