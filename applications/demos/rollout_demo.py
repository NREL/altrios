# %%
import altrios
from altrios import rollout

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set()

# %

# %%


plot_dir = Path() / "plots"
# make the dir if it doesn't exist
plot_dir.mkdir(exist_ok=True)

scenario_infos, metrics = rollout.simulate_prescribed_rollout(
    max_bel_share=0.5, number_of_years=2
)

# %%
# Plotting code currently just plots the first year of a multi-year simulation.
to_plot = scenario_infos[0].sims.tolist()

for idx, sim in enumerate(to_plot[:10]):
    # sim = altc.SpeedLimitTrainSim.from_bincode(
    #     sim.to_bincode())  # to support linting
    fig, ax = plt.subplots(3, 1, sharex=True)

    loco0 = sim.loco_con.loco_vec.tolist()[0]

    # loco0 = altc.Locomotive.from_bincode(
    #     loco0.to_bincode())  # to support linting
    loco1 = sim.loco_con.loco_vec.tolist()[1]
    plt.suptitle(f"sim #: {idx}")

    if loco0.fc is not None:
        ax[0].plot(
            np.array(sim.history.time_seconds) / 3_600,
            np.array(loco0.fc.history.pwr_fuel_watts) / 1e6,
            # label='fuel'
        )
        # ax[0].plot(
        #     np.array(sim.history.time_seconds) / 3_600,
        #     np.array(loco0.history.pwr_out_watts) / 1e6,
        #     label='conv. loco. tractive'
        # )
        # ax[0].plot(
        #     np.array(sim.history.time_seconds) / 3_600,
        #     np.array(loco1.history.pwr_out_watts) / 1e6,
        #     label='BEL tractive'
        # )
        ax[0].set_ylabel("Single Loco.\nFuel Power [MW]")
        # ax[0].legend()

    if loco1.res is not None:
        ax[1].plot(np.array(sim.history.time_seconds) /
                   3_600, loco1.res.history.soc)
        ax[1].set_ylabel("SOC")

    ax[-1].plot(
        np.array(sim.history.time_seconds) / 3_600,
        sim.history.velocity_meters_per_second,
        label="actual",
    )
    ax[-1].plot(
        np.array(sim.history.time_seconds) / 3_600,
        sim.history.speed_limit_meters_per_second,
        label="limit",
    )
    # ax[-1].plot(
    #     sim.history.time_seconds,
    #     sim.history.speed_target_meters_per_second,
    #     label='target',
    #     linestyle="-."
    # )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [hr]")
    ax[-1].set_ylabel("Speed [m/s]")

    plt.tight_layout()
    plt.savefig(plot_dir / f"sim num {idx}.png")
    plt.savefig(plot_dir / f"sim num {idx}.svg")
    plt.close()

# %%
