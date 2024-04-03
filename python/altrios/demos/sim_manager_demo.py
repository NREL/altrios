# %%
from altrios import sim_manager
from altrios import utilities, defaults, train_planner
import altrios as alt
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from pathlib import Path

sns.set_theme()

SHOW_PLOTS = alt.utils.show_plots()
# %

plot_dir = Path() / "plots"
# make the dir if it doesn't exist
plot_dir.mkdir(exist_ok=True)


# %%

t0_import = time.perf_counter()
t0_total = time.perf_counter()

rail_vehicle_map = alt.import_rail_vehicles(alt.resources_root() / "rolling_stock/rail_vehicles.csv")
location_map = alt.import_locations(alt.resources_root() / "networks/default_locations.csv")
network = alt.Network.from_file(alt.resources_root() / "networks/Taconite-NoBalloon.yaml")

t1_import = time.perf_counter()
print(
    f"Elapsed time to import rail vehicles, locations, and network: {t1_import - t0_import:.3g} s"
)

train_planner_config = train_planner.TrainPlannerConfig(
            cars_per_locomotive=50,
            target_cars_per_train=90)

t0_main = time.perf_counter()

(
    train_consist_plan, 
    loco_pool, 
    refuel_facilities, 
    grid_emissions_factors, 
    nodal_energy_prices, 
    speed_limit_train_sims, 
    timed_paths
) = sim_manager.main(
    network=network,
    rail_vehicle_map=rail_vehicle_map,
    location_map=location_map,
    train_planner_config=train_planner_config,
    debug=True,
)

t1_main = time.perf_counter()
print(f"Elapsed time to run `sim_manager.main()`: {t1_main-t0_main:.3g} s")

# %%
t0_train_sims = time.perf_counter()
speed_limit_train_sims.set_save_interval(100)
(sims, refuel_sessions) = alt.run_speed_limit_train_sims(
    speed_limit_train_sims=speed_limit_train_sims,
    network=network,
    train_consist_plan_py=train_consist_plan,
    loco_pool_py=loco_pool,
    refuel_facilities_py=refuel_facilities,
    timed_paths=timed_paths
)
t1_train_sims = time.perf_counter()
print(f"Elapsed time to run train sims: {t1_train_sims-t0_train_sims:.3g} s")

# %%
t0_summary_sims = time.perf_counter()
speed_limit_train_sims.set_save_interval(None)
(summary_sims, summary_refuel_sessions) = alt.run_speed_limit_train_sims(
    speed_limit_train_sims=speed_limit_train_sims,
    network=network,
    train_consist_plan_py=train_consist_plan,
    loco_pool_py=loco_pool,
    refuel_facilities_py=refuel_facilities,
    timed_paths=timed_paths,
)
t1_summary_sims = time.perf_counter()
print(
    f"Elapsed time to build and run summary sims: {t1_summary_sims-t0_summary_sims:.3g} s"
)

# %%
t0_tolist = time.perf_counter()
sims_list = sims.tolist()
t1_tolist = time.perf_counter()
print(f"Elapsed time to run `tolist()`: {t1_tolist-t0_tolist:.3g} s")

sim0 = sims_list[0]
# sim0 = alt.SpeedLimitTrainSim.from_bincode(
#     sim0.to_bincode())  # to support linting


# %%

t0_main = time.perf_counter()
e_total_fuel_mj = summary_sims.get_energy_fuel_joules(annualize=False) / 1e9
t1_main = time.perf_counter()

print(f"Elapsed time to get total fuel energy: {t1_main-t0_main:.3g} s")
print(f"Total fuel energy used: {e_total_fuel_mj:.3g} GJ")

v_total_fuel_gal = summary_sims.get_energy_fuel_joules(annualize=False) / 1e3 / defaults.LHV_DIESEL_KJ_PER_KG / \
    defaults.RHO_DIESEL_KG_PER_M3 * utilities.LITER_PER_M3 * utilities.GALLONS_PER_LITER

print(f"Total fuel used: {v_total_fuel_gal:.3g} gallons")
print(f"Total elapsed time: {time.perf_counter() - t0_total} s")


# %%

if SHOW_PLOTS:
    for idx, sim in enumerate(sims_list[:10]):
        sim: alt.SpeedLimitTrainSim
        fig, ax = plt.subplots(3, 1, sharex=True)

        loco0 = sim.loco_con.loco_vec.tolist()[0]
        # loco0 = alt.Locomotive.from_bincode(
        #     loco0.to_bincode())  # to support linting
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
        #     np.array(loco1.history.pwr_out_watts) / 1e6
        #     label='BEL tractive'
        # )
        ax[0].set_ylabel("Single Loco.\nFuel Power [MW]")
        # ax[0].legend()

        # TODO: Figure out robust way to ensure one bel in demo consist
        if len(sim.loco_con.loco_vec.tolist()) > 1:
            loco1 = sim.loco_con.loco_vec.tolist()[1]
            if loco1.res is not None:
                ax[1].plot(
                    np.array(sim.history.time_seconds) / 3_600,
                    loco1.res.history.soc
                )
                ax[1].set_ylabel('SOC')

        ax[-1].plot(
            np.array(sim.history.time_seconds) / 3_600,
            sim.history.speed_meters_per_second,
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
