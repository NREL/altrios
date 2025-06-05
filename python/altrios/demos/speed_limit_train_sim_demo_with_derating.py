# %%
import os
import time
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

import altrios as alt

sns.set_theme()


SHOW_PLOTS = alt.utils.show_plots()

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
hel: alt.Locomotive = alt.Locomotive.default_hybrid_electric_loco()
hel_dict = hel.to_pydict()
hel_pt_cntrl = hel_dict["loco_type"]["HybridLoco"]["pt_cntrl"]["RGWDB"]
hel_pt_cntrl["speed_soc_disch_buffer_meters_per_second"] = 0
hel_pt_cntrl["speed_soc_regen_buffer_meters_per_second"] = 100
hel_dict["loco_type"]["HybridLoco"]["pt_cntrl"]["RGWDB"] = hel_pt_cntrl
hel = alt.Locomotive.from_pydict(hel_dict)

fc_with_derate = alt.FuelConverter.default()
fc_with_derate.set_default_elev_and_temp_derate()
fc_with_derate_dict = fc_with_derate.to_pydict()

hel_new_dict = copy(hel_dict)
hel_new_dict["loco_type"]["HybridLoco"]["fc"] = fc_with_derate_dict
hel_with_derate = alt.Locomotive.from_pydict(hel_new_dict)

conv_new_dict = alt.Locomotive.default().to_pydict()
conv_new_dict["loco_type"]["ConventionalLoco"]["fc"] = fc_with_derate_dict
conv_with_derate = alt.Locomotive.from_pydict(conv_new_dict)

# construct a vector of one BEL, one HEL, and several conventional locomotives
loco_vec = (
    [hel.clone()]
    + [alt.Locomotive.default()] * 1
)

# construct a vector of one BEL, one HEL, and several conventional locomotives
loco_vec_with_derating = (
    [hel_with_derate.clone()]
    + [conv_with_derate] * 1
)

# instantiate consist
loco_con = alt.Consist(
    loco_vec,
    SAVE_INTERVAL,
)

# instantiate consist
loco_con_with_derating = alt.Consist(
    loco_vec_with_derating,
    SAVE_INTERVAL,
)

# Instantiate the intermediate `TrainSimBuilder`
tsb = alt.TrainSimBuilder(
    train_id="0",
    origin_id="Minneapolis",
    destination_id="Superior",
    train_config=train_config,
    loco_con=loco_con,
)

# Instantiate the intermediate `TrainSimBuilder`
tsb_with_derating = alt.TrainSimBuilder(
    train_id="0",
    origin_id="Minneapolis",
    destination_id="Superior",
    train_config=train_config,
    loco_con=loco_con_with_derating,
)

# Load the network and construct the timed link path through the network.
network = alt.Network.from_file(
    alt.resources_root() / "networks/Taconite-NoBalloon.yaml")

location_map = alt.import_locations(
    alt.resources_root() / "networks/default_locations.csv")

train_sim: alt.SpeedLimitTrainSim = tsb.make_speed_limit_train_sim(
    location_map=location_map,
    save_interval=SAVE_INTERVAL,
)
train_sim.set_save_interval(SAVE_INTERVAL)

CELSIUS_TO_KELVIN = 273.15
temp_trace = alt.TemperatureTrace.from_pydict({
    "time_seconds": [0, 7_200],
    "temp_at_sea_level_degrees_Celsius": (np.asarray([22.0, 45.0]) + CELSIUS_TO_KELVIN).tolist(),
})
train_sim_with_derating: alt.SpeedLimitTrainSim = tsb_with_derating.make_speed_limit_train_sim(
    location_map=location_map,
    save_interval=SAVE_INTERVAL,
    temp_trace=temp_trace,
)
train_sim_with_derating.set_save_interval(SAVE_INTERVAL)

est_time_net, _consist = alt.make_est_times(train_sim, network)

est_time_net_with_derating, _consist = alt.make_est_times(
    train_sim_with_derating, network)

timed_link_path = next(iter(alt.run_dispatch(
    network,
    alt.SpeedLimitTrainSimVec([train_sim]),
    [est_time_net],
    False,
    False,
)))

timed_link_path_with_derating = next(iter(alt.run_dispatch(
    network,
    alt.SpeedLimitTrainSimVec([train_sim_with_derating]),
    [est_time_net_with_derating],
    False,
    False,
)))


# whether to override files used by set_speed_train_sim_demo.py
OVERRIDE_SSTS_INPUTS = os.environ.get(
    "OVERRIDE_SSTS_INPUTS", "false").lower() == "true"
if OVERRIDE_SSTS_INPUTS:
    print("Overriding files used by `set_speed_train_sim_demo.py`")
    link_path = alt.LinkPath([x.link_idx for x in timed_link_path.tolist()])
    link_path.to_csv_file(alt.resources_root() / "demo_data/link_path.csv")

t0 = time.perf_counter()
train_sim.walk_timed_path(
    network=network,
    timed_path=timed_link_path,
)
t1 = time.perf_counter()

print(f"Time to simulate: {t1 - t0:.5g}")
raw_fuel_gigajoules = train_sim.get_energy_fuel_joules(False) / 1e9
print(
    f"Total raw fuel used without engine derating in BEL and HEL: {raw_fuel_gigajoules:.6g} GJ")
corrected_fuel_gigajoules = train_sim.get_energy_fuel_soc_corrected_joules() / 1e9
print(
    f"Total SOC-corrected fuel used without engine derating in BEL and HEL: {corrected_fuel_gigajoules:.6g} GJ")
assert len(train_sim.history) > 1

t0 = time.perf_counter()
train_sim_with_derating.walk_timed_path(
    network=network,
    timed_path=timed_link_path_with_derating,
)
t1 = time.perf_counter()

print(f"\nTime to simulate with altitude and temperature derating of engine: {t1 - t0:.5g}")
raw_fuel_with_derating_gigajoules = train_sim_with_derating.get_energy_fuel_joules(False) / 1e9
print(
    f"Total raw fuel used with altitude and temperature derating in BEL and HEL: {raw_fuel_with_derating_gigajoules:.6g} GJ")
corrected_fuel_with_derating_gigajoules = train_sim_with_derating.get_energy_fuel_soc_corrected_joules() / 1e9
print(
    f"Total SOC-corrected fuel used with altitude and temperature derating in BEL and HEL: {corrected_fuel_with_derating_gigajoules:.6g} GJ")
assert len(train_sim_with_derating.history) > 1

fuel_increase = (raw_fuel_with_derating_gigajoules - raw_fuel_gigajoules) / raw_fuel_gigajoules * 100
print(f"\nRaw fuel increase due to derating: {fuel_increase:.5g}%")
fuel_increase_soc_corrected = (
     corrected_fuel_with_derating_gigajoules - corrected_fuel_gigajoules) / corrected_fuel_gigajoules * 100
print(f"SOC-corrected fuel increase due to derating: {fuel_increase_soc_corrected:.5g}%")


# Uncomment the following lines to overwrite `set_speed_train_sim_demo.py` `speed_trace`
def plot_train_level_powers(ts: alt.SpeedLimitTrainSim, mod_str: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Train Power " + mod_str)
    ax[0].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        np.asarray(ts.history.pwr_whl_out_watts) / 1e6,
        label="tract pwr",
    )
    ax[0].set_ylabel("Power [MW]")
    ax[0].legend()

    ax[1].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        np.asarray(ts.history.res_aero_newtons) / 1e3,
        label="aero",
    )
    ax[1].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        np.asarray(ts.history.res_rolling_newtons) / 1e3,
        label="rolling",
    )
    ax[1].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        np.asarray(ts.history.res_curve_newtons) / 1e3,
        label="curve",
    )
    ax[1].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        np.asarray(ts.history.res_bearing_newtons) / 1e3,
        label="bearing",
    )
    ax[1].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        np.asarray(ts.history.res_grade_newtons) / 1e3,
        label="grade",
    )
    ax[1].set_ylabel("Force [MN]")
    ax[1].legend()

    ax[-1].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        ts.history.speed_meters_per_second,
        label="achieved",
    )
    ax[-1].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        ts.history.speed_limit_meters_per_second,
        label="limit",
    )
    ax[-1].set_xlabel("Time [hr]")
    ax[-1].set_ylabel("Speed [m/s]")
    ax[-1].legend()

    return fig, ax


def plot_train_network_info(ts: alt.SpeedLimitTrainSim, mod_str: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Train Position in Network " + mod_str)
    ax[0].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        np.asarray(ts.history.offset_in_link_meters) / 1_000,
        label="current link",
    )
    ax[0].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        np.asarray(ts.history.offset_meters) / 1_000,
        label="overall",
    )
    ax[0].legend()
    ax[0].set_ylabel("Net Dist. [km]")

    ax[1].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        ts.history.link_idx_front,
        linestyle="",
        marker=".",
    )
    ax[1].set_ylabel("Link Idx Front")

    ax[-1].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        ts.history.speed_meters_per_second,
    )
    ax[-1].set_xlabel("Time [hr]")
    ax[-1].set_ylabel("Speed [m/s]")

    plt.tight_layout()

    return fig, ax


def plot_consist_pwr(ts: alt.SpeedLimitTrainSim, mod_str: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Loco. Consist " + mod_str)
    ax[0].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        np.asarray(ts.history.pwr_whl_out_watts) / 1e6,
        label="consist tract pwr",
    )
    ax[0].set_ylabel("Power [MW]")
    ax[0].legend()

    ax[1].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        np.asarray(ts.history.grade_front) * 100.,
    )
    ax[1].set_ylabel("Grade [%] at\nHead End")

    ax[-1].plot(
        np.asarray(ts.history.time_seconds) / 3_600,
        ts.history.speed_meters_per_second,
    )
    ax[-1].set_xlabel("Time [hr]")
    ax[-1].set_ylabel("Speed [m/s]")

    return fig, ax


hel_type = "HybridLoco"


def plot_hel_pwr_and_soc(ts: alt.SpeedLimitTrainSim, mod_str: str) -> tuple[plt.Figure, plt.Axes]:
    ts_dict = ts.to_pydict()
    hybrid_loco = ts_dict["loco_con"]["loco_vec"][0]
    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Hybrid Locomotive " + mod_str)

    ax_idx = 0
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.asarray(hybrid_loco["history"]["pwr_out_watts"]) / 1e3,
        label="tract. pwr.",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.asarray(
            hybrid_loco["loco_type"][hel_type]["res"]["history"]["pwr_disch_max_watts"],
        ) / 1e3,
        label="batt. max disch. pwr",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.asarray(
            hybrid_loco["loco_type"][hel_type]["res"]["history"]["pwr_charge_max_watts"],
        ) / 1e3,
        label="batt. max chrg. pwr",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.asarray(hybrid_loco["loco_type"][hel_type]["res"]
                 ["history"]["pwr_out_electrical_watts"]) / 1e3,
        label="batt. elec. pwr.",
    )
    pwr_gen_elect_out = np.asarray(hybrid_loco["loco_type"][hel_type]["gen"]["history"]["pwr_elec_prop_out_watts"]) \
        + np.asarray(hybrid_loco["loco_type"][hel_type]
                   ["gen"]["history"]["pwr_elec_aux_watts"])
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        pwr_gen_elect_out / 1e3,
        label="gen. elec. pwr.",
    )
    y_max = ax[ax_idx].get_ylim()[1]
    ax[ax_idx].set_ylim([-y_max, y_max])
    ax[ax_idx].set_ylabel("Power [kW]")
    ax[ax_idx].legend()

    ax_idx += 1
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        hybrid_loco["loco_type"][hel_type]["res"]["history"]["soc"],
        label="soc",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"][1:],
        hybrid_loco["loco_type"][hel_type]["res"]["history"]["soc_chrg_buffer"][1:],
        label="chrg buff",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"][1:],
        hybrid_loco["loco_type"][hel_type]["res"]["history"]["soc_disch_buffer"][1:],
        label="disch buff",
    )
    # TODO: add static min and max soc bounds to plots
    # TODO: make a plot util for any type of locomotive that will plot all the stuff
    ax[ax_idx].set_ylabel("[-]")
    ax[ax_idx].legend()

    ax_idx += 1
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        ts_dict["history"]["speed_meters_per_second"],
    )
    ax[ax_idx].set_ylabel("Speed [m/s]")
    ax[ax_idx].set_xlabel("Times [s]")
    plt.tight_layout()

    return fig, ax


bel_type = "BatteryElectricLoco"


def plot_bel_pwr_and_soc(ts: alt.SpeedLimitTrainSim, mod_str: str) -> tuple[plt.Figure, plt.Axes]:
    ts_dict = ts.to_pydict()
    batt_loco = ts_dict["loco_con"]["loco_vec"][0]
    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Battery Electric Locomotive " + mod_str)

    ax_idx = 0
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.asarray(batt_loco["history"]["pwr_out_watts"]) / 1e3,
        label="tract. pwr.",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.asarray(
            batt_loco["loco_type"][bel_type]["res"]["history"]["pwr_disch_max_watts"],
        ) / 1e3,
        label="batt. max disch. pwr",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.asarray(
            batt_loco["loco_type"][bel_type]["res"]["history"]["pwr_charge_max_watts"],
        ) / 1e3,
        label="batt. max chrg. pwr",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.asarray(batt_loco["loco_type"][bel_type]["res"]
                 ["history"]["pwr_out_electrical_watts"]) / 1e3,
        label="batt. elec. pwr.",
    )
    y_max = ax[ax_idx].get_ylim()[1]
    ax[ax_idx].set_ylim([-y_max, y_max])
    ax[ax_idx].set_ylabel("Power [kW]")
    ax[ax_idx].legend()

    ax_idx += 1
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        batt_loco["loco_type"][bel_type]["res"]["history"]["soc"],
        label="soc",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"][1:],
        batt_loco["loco_type"][bel_type]["res"]["history"]["soc_chrg_buffer"][1:],
        label="chrg buff",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"][1:],
        batt_loco["loco_type"][bel_type]["res"]["history"]["soc_disch_buffer"][1:],
        label="disch buff",
    )
    ax[ax_idx].set_ylabel("[-]")
    ax[ax_idx].legend()

    ax_idx += 1
    # TODO: add static min and max soc bounds to plots
    # TODO: make a plot util for any type of locomotive that will plot all the stuff
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        ts_dict["history"]["speed_meters_per_second"],
    )
    ax[ax_idx].set_ylabel("Speed [m/s]")
    ax[ax_idx].set_xlabel("Times [s]")
    ax[ax_idx].legend()
    plt.tight_layout()

    return fig, ax


df_no_derate = train_sim.to_dataframe()
df_with_derate = train_sim_with_derating.to_dataframe()

fig, ax = plt.subplots(3, 1, sharex=True)
fig.suptitle("Derate Comparison")
ax[0].plot(
    df_no_derate["history.time_seconds"],
    df_no_derate["loco_con.history.energy_fuel_joules"] / 1e9,
    label="no derating",
)
ax[0].plot(
    df_with_derate["history.time_seconds"],
    df_with_derate["loco_con.history.energy_fuel_joules"] / 1e9,
    label="with derate",
)
ax[0].legend()
ax[0].set_ylabel("Cumulative Fuel [GJ]")

ax[1].plot(
    df_no_derate["history.time_seconds"],
    df_no_derate["loco_con.history.pwr_out_max_watts"] / 1e6,
    label="no derating pwr max",
)
ax[1].plot(
    df_with_derate["history.time_seconds"],
    df_with_derate["loco_con.history.pwr_out_max_watts"] / 1e6,
    label="with derate pwr max",
)
ax[1].legend()
ax[1].set_ylabel("Engine Power [MW]")

ax[2].plot(
    df_no_derate["history.time_seconds"],
    df_no_derate["history.speed_meters_per_second"],
    label="no derating",
)
ax[2].plot(
    df_with_derate["history.time_seconds"],
    df_with_derate["history.speed_meters_per_second"],
    label="with derate",
)
ax[2].legend()
ax[2].set_ylabel("Speed [m/s]")
ax[2].set_xlabel("Times [s]")
plt.tight_layout()

fig0, ax0 = plot_train_level_powers(train_sim, "No Derating")
fig1, ax1 = plot_train_network_info(train_sim, "No Derating")
fig2, ax2 = plot_consist_pwr(train_sim, "No Derating")
fig3, ax3 = plot_hel_pwr_and_soc(train_sim, "No Derating")
# fig3.savefig("plots/hel with buffers.svg")
# fig4, ax4 = plot_bel_pwr_and_soc(train_sim, "No Derating")

fig0_sans_buffers, ax0_sans_buffers = plot_train_level_powers(train_sim_with_derating, "With Altitude and Temperature Derating")
fig1_sans_buffers, ax1_sans_buffers = plot_train_network_info(train_sim_with_derating, "With Altitude and Temperature Derating")
fig2_sans_buffers, ax2_sans_buffers = plot_consist_pwr(train_sim_with_derating, "With Altitude and Temperature Derating")
fig3_sans_buffers, ax3_sans_buffers = plot_hel_pwr_and_soc(train_sim_with_derating, "With Altitude and Temperature Derating")
# fig3_sans_buffers.savefig("plots/hel sans buffers.svg")
# fig4_sans_buffers, ax4_sans_buffers = plot_bel_pwr_and_soc(train_sim_sans_buffers, "With Altitude and Temperature Derating")

if SHOW_PLOTS:
    plt.tight_layout()
    plt.show()
# Impact of sweep of battery capacity TODO: make this happen

# whether to run assertions, enabled by default
ENABLE_ASSERTS = os.environ.get("ENABLE_ASSERTS", "true").lower() == "true"
# whether to override reference files used in assertions, disabled by default
ENABLE_REF_OVERRIDE = os.environ.get(
    "ENABLE_REF_OVERRIDE", "false").lower() == "true"
# directory for reference files for checking sim results against expected results
ref_dir = alt.resources_root() / "demo_data/speed_limit_train_sim_demo/"

if ENABLE_REF_OVERRIDE:
    ref_dir.mkdir(exist_ok=True, parents=True)
    df: pl.DataFrame = train_sim.to_dataframe().lazy().collect()[-1]
    df.write_csv(ref_dir / "to_dataframe_expected.csv")
if ENABLE_ASSERTS:
    print("Checking output of `to_dataframe`")
    to_dataframe_expected = pl.scan_csv(
        ref_dir / "to_dataframe_expected.csv").collect()[-1]
    assert to_dataframe_expected.equals(train_sim.to_dataframe()[-1]), \
        f"to_dataframe_expected: \n{to_dataframe_expected}\ntrain_sim.to_dataframe()[-1]: \n{train_sim.to_dataframe()[-1]}" + \
        "\ntry running with `ENABLE_REF_OVERRIDE=True`"
    print("Success!")

# %%
