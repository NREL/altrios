# %%
"""
SetSpeedTrainSim over a simple, hypothetical corridor
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Tuple

import altrios as alt

sns.set_theme()

SHOW_PLOTS = alt.utils.show_plots()
SAVE_INTERVAL = 1


def extract_bel_from_train_sim(ts: alt.SetSpeedTrainSim) -> list:
    ts_dict = ts.to_pydict()
    ts_list = ts_dict["loco_con"]["loco_vec"]
    loco_list = []
    for loco in ts_list:
        if "BatteryElectricLoco" in loco["loco_type"]:
            loco_list.append(loco)
    if not loco_list:
        print("NO BEL IS FOUND IN CONSIST")
        return False
    return loco_list


def extract_conv_from_train_sim(ts: alt.SetSpeedTrainSim) -> list:
    ts_dict = ts.to_pydict()
    ts_list = ts_dict["loco_con"]["loco_vec"]
    loco_list = []
    for loco in ts_list:
        if "ConventionalLoco" in loco["loco_type"]:
            loco_list.append(loco)
    if not loco_list:
        print("NO CONVENTIONAL LOCO IS FOUND IN CONSIST")
        return False
    return loco_list


def extract_hel_from_train_sim(ts: alt.SetSpeedTrainSim) -> list:
    ts_dict = ts.to_pydict()
    ts_list = ts_dict["loco_con"]["loco_vec"]
    loco_list = []
    for loco in ts_list:
        if "HybridLoco" in loco["loco_type"]:
            # Hybrid loco's loco type is somehow still BEL
            loco_list.append(loco)
    if not loco_list:
        print("NO HYBRID LOCO IS FOUND IN CONSIST")
        return False
    return loco_list


def plot_locos_from_ts(ts: alt.SetSpeedTrainSim, x: str, show_plots: bool = False):
    """
    Can take in either SetSpeedTrainSim or SpeedLimitTrainSim
    Extracts first instance of each loco_type and plots representative plots
    Offers two plotting options to put on x axis
    ts: train sim
    x: ["time","offset"]
    """
    ts_dict = ts.to_pydict()
    if isinstance(train_sim, alt.SpeedLimitTrainSim):
        plot_name = "Speed Limit Train Sim"
    if isinstance(train_sim, alt.SetSpeedTrainSim):
        plot_name = "Set Speed Train Sim"
    if x == "time" or x == "Time":
        x_axis = np.array(ts_dict["history"]["time_seconds"]) / 3_600
        x_label = "Time (hr)"
    if x == "distance" or x == "Distance":
        x_axis = np.array(ts_dict["history"]["offset_back_meters"]) / 1_000
        x_label = "Distance (km)"
    first_bel = []
    first_hel = []
    if extract_bel_from_train_sim(ts):
        first_bel = next(iter(extract_bel_from_train_sim(ts)))
        """
        first fig
        speed vs dist or time
        soc vs dist or time
        various powers along the powertrain vs dist or time
        various cumulative energies along the powertrain vs dist or time
        """
        fig, ax = plt.subplots(4, 1, sharex=True)
        ax[0].plot(
            x_axis,
            np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
            label="tract pwr",
        )
        ax[0].set_ylabel("Power [MW]")
        ax[0].legend()

        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_aero_newtons"]) / 1e3,
            label="aero",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_rolling_newtons"]) / 1e3,
            label="rolling",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_curve_newtons"]) / 1e3,
            label="curve",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_bearing_newtons"]) / 1e3,
            label="bearing",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_grade_newtons"]) / 1e3,
            label="grade",
        )
        ax[1].set_ylabel("Force [MN]")
        ax[1].legend()

        ax[2].plot(
            x_axis,
            np.array(
                first_bel["loco_type"]["BatteryElectricLoco"]["res"]["history"]["soc"]
            ),
        )
        ax[2].set_ylabel("SOC")

        ax[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
            label="achieved speed",
        )
        if isinstance(train_sim, alt.SpeedLimitTrainSim):
            ax[-1].plot(
                x_axis,
                ts_dict["history"]["speed_limit_meters_per_second"],
                label="limit",
            )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel("Speed [m/s]")
        ax[-1].legend()
        plt.suptitle(plot_name + " " + "Train Resistance, BEL SOC and Train Speed")

        fig1, ax1 = plt.subplots(3, 1, sharex=True)
        ax1[0].plot(
            x_axis,
            np.array(ts_dict["history"]["offset_in_link_meters"]) / 1_000,
            label="current link",
        )
        ax1[0].plot(
            x_axis,
            np.array(ts_dict["history"]["offset_meters"]) / 1_000,
            label="overall",
        )
        ax1[0].legend()
        ax1[0].set_ylabel("Net Dist. [km]")

        ax1[1].plot(
            x_axis,
            ts_dict["history"]["link_idx_front"],
            linestyle="",
            marker=".",
        )
        ax1[1].set_ylabel("Link Idx Front")

        ax1[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax1[-1].set_xlabel(x_label)
        ax1[-1].set_ylabel("Speed [m/s]")

        plt.suptitle(plot_name + " " + "Distance and Link Tracking")
        plt.tight_layout()

        fig2, ax2 = plt.subplots(3, 1, sharex=True)
        ax2[0].plot(
            x_axis,
            np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
            label="tract pwr",
        )
        ax2[0].set_ylabel("Power [MW]")
        ax2[0].legend()

        ax2[1].plot(
            x_axis,
            np.array(ts_dict["history"]["grade_front"]) * 100.0,
        )
        ax2[1].set_ylabel("Grade [%] at\nHead End")

        ax2[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax2[-1].set_xlabel(x_label)
        ax2[-1].set_ylabel("Speed [m/s]")

        plt.suptitle(plot_name + " " + "Power and Grade Profile")
        plt.tight_layout()
        if show_plots:
            plt.show()

    if extract_conv_from_train_sim(ts):
        # first_conv = next(iter(extract_conv_from_train_sim(ts)))
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(
            x_axis,
            np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
            label="tract pwr",
        )
        ax[0].set_ylabel("Power [MW]")
        ax[0].legend()

        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_aero_newtons"]) / 1e3,
            label="aero",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_rolling_newtons"]) / 1e3,
            label="rolling",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_curve_newtons"]) / 1e3,
            label="curve",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_bearing_newtons"]) / 1e3,
            label="bearing",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_grade_newtons"]) / 1e3,
            label="grade",
        )
        ax[1].set_ylabel("Force [MN]")
        ax[1].legend()
        ax[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
            label="achieved speed",
        )
        if isinstance(train_sim, alt.SpeedLimitTrainSim):
            ax[-1].plot(
                x_axis,
                ts_dict["history"]["speed_limit_meters_per_second"],
                label="limit",
            )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel("Speed [m/s]")
        ax[-1].legend()
        plt.suptitle(plot_name + " " + "Train Resistance, and Train Speed")

        fig1, ax1 = plt.subplots(3, 1, sharex=True)
        ax1[0].plot(
            x_axis,
            np.array(ts_dict["history"]["offset_in_link_meters"]) / 1_000,
            label="current link",
        )
        ax1[0].plot(
            x_axis,
            np.array(ts_dict["history"]["offset_meters"]) / 1_000,
            label="overall",
        )
        ax1[0].legend()
        ax1[0].set_ylabel("Net Dist. [km]")

        ax1[1].plot(
            x_axis,
            ts_dict["history"]["link_idx_front"],
            linestyle="",
            marker=".",
        )
        ax1[1].set_ylabel("Link Idx Front")

        ax1[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax1[-1].set_xlabel(x_label)
        ax1[-1].set_ylabel("Speed [m/s]")

        plt.suptitle(plot_name + " " + "Distance and Link Tracking")
        plt.tight_layout()

        fig2, ax2 = plt.subplots(3, 1, sharex=True)
        ax2[0].plot(
            x_axis,
            np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
            label="tract pwr",
        )
        ax2[0].set_ylabel("Power [MW]")
        ax2[0].legend()

        ax2[1].plot(
            x_axis,
            np.array(ts_dict["history"]["grade_front"]) * 100.0,
        )
        ax2[1].set_ylabel("Grade [%] at\nHead End")

        ax2[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax2[-1].set_xlabel(x_label)
        ax2[-1].set_ylabel("Speed [m/s]")

        plt.suptitle(plot_name + " " + "Power and Grade Profile")
        plt.tight_layout()
        if show_plots:
            plt.show()

    if extract_hel_from_train_sim(ts):
        first_hel = next(iter(extract_hel_from_train_sim(ts)))
        ts_dict = ts.to_pydict()
        fig, ax = plt.subplots(4, 1, sharex=True)
        ax[0].plot(
            x_axis,
            np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
            label="tract pwr",
        )
        ax[0].set_ylabel("Power [MW]")
        ax[0].legend()

        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_aero_newtons"]) / 1e3,
            label="aero",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_rolling_newtons"]) / 1e3,
            label="rolling",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_curve_newtons"]) / 1e3,
            label="curve",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_bearing_newtons"]) / 1e3,
            label="bearing",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_grade_newtons"]) / 1e3,
            label="grade",
        )
        ax[1].set_ylabel("Force [MN]")
        ax[1].legend()

        ax[2].plot(
            x_axis,
            np.array(first_hel["loco_type"]["HybridLoco"]["res"]["history"]["soc"]),
        )
        ax[2].set_ylabel("SOC")

        ax[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
            label="achieved speed",
        )
        if isinstance(train_sim, alt.SpeedLimitTrainSim):
            ax[-1].plot(
                x_axis,
                ts_dict["history"]["speed_limit_meters_per_second"],
                label="limit",
            )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel("Speed [m/s]")
        ax[-1].legend()
        plt.suptitle(plot_name + " " + "Train Resistance, BEL SOC and Train Speed")

        fig1, ax1 = plt.subplots(3, 1, sharex=True)
        ax1[0].plot(
            x_axis,
            np.array(ts_dict["history"]["offset_in_link_meters"]) / 1_000,
            label="current link",
        )
        ax1[0].plot(
            x_axis,
            np.array(ts_dict["history"]["offset_meters"]) / 1_000,
            label="overall",
        )
        ax1[0].legend()
        ax1[0].set_ylabel("Net Dist. [km]")

        ax1[1].plot(
            x_axis,
            ts_dict["history"]["link_idx_front"],
            linestyle="",
            marker=".",
        )
        ax1[1].set_ylabel("Link Idx Front")

        ax1[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax1[-1].set_xlabel(x_label)
        ax1[-1].set_ylabel("Speed [m/s]")

        plt.suptitle(plot_name + " " + "Distance and Link Tracking")
        plt.tight_layout()

        fig2, ax2 = plt.subplots(3, 1, sharex=True)
        ax2[0].plot(
            x_axis,
            np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
            label="tract pwr",
        )
        ax2[0].set_ylabel("Power [MW]")
        ax2[0].legend()

        ax2[1].plot(
            x_axis,
            np.array(ts_dict["history"]["grade_front"]) * 100.0,
        )
        ax2[1].set_ylabel("Grade [%] at\nHead End")

        ax2[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax2[-1].set_xlabel(x_label)
        ax2[-1].set_ylabel("Speed [m/s]")

        plt.suptitle(plot_name + " " + "Power and Grade Profile")
        plt.tight_layout()
        if show_plots:
            plt.show()

        fig3, ax3 = plt.subplots(3, 1, sharex=True)
        ax3[0].plot(
            x_axis,
            np.array(first_hel["history"]["pwr_out_watts"]) / 1e3,
            label="hybrid tract. pwr.",
        )
        ax3[0].plot(
            x_axis,
            np.array(
                first_hel["loco_type"]["HybridLoco"]["res"]["history"][
                    "pwr_out_electrical_watts"
                ]
            )
            / 1e3,
            # np.array(hybrid_loco['loco_type']['HybridLoco']['res']
            #        ['history']['pwr_out_electrical_watts']) / 1e3,
            label="hybrid batt. elec. pwr.",
        )
        ax3[0].set_ylabel("Power [kW]")
        ax3[0].legend()
        ax3[1].plot(
            x_axis,
            first_hel["loco_type"]["HybridLoco"]["res"]["history"]["soc"],
            label="soc",
        )
        ax3[1].plot(
            x_axis,
            first_hel["loco_type"]["HybridLoco"]["res"]["history"]["soc_chrg_buffer"],
            label="chrg buff",
        )
        ax3[1].plot(
            x_axis,
            first_hel["loco_type"]["HybridLoco"]["res"]["history"]["soc_disch_buffer"],
            label="disch buff",
        )
        ax3[1].set_ylabel("[-]")
        ax3[1].legend()
        ax3[2].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax3[2].set_ylabel("Speed [m/s]")
        ax3[2].set_xlabel("Times [s]")
        plt.suptitle(plot_name + " " + "Hybrid Loco Power and Buffer Profile")
        plt.tight_layout()
        if show_plots:
            plt.show()


# Build the train config
rail_vehicle_loaded = alt.RailVehicle.from_file(
    alt.resources_root() / "rolling_stock/Manifest_Loaded.yaml"
)
rail_vehicle_empty = alt.RailVehicle.from_file(
    alt.resources_root() / "rolling_stock/Manifest_Empty.yaml"
)

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
    alt.resources_root()
    / "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
)

edrv = alt.ElectricDrivetrain(
    pwr_out_frac_interp=[0.0, 1.0],
    eta_interp=[0.98, 0.98],
    pwr_out_max_watts=5e9,
    save_interval=SAVE_INTERVAL,
)

bel: alt.Locomotive = alt.Locomotive.from_pydict(
    {
        "loco_type": {
            "BatteryElectricLoco": {
                "res": res.to_pydict(),
                "edrv": edrv.to_pydict(),
            }
        },
        "pwr_aux_offset_watts": 8.55e3,
        "pwr_aux_traction_coeff": 540.0e-6,
        "force_max_newtons": 667.2e3,
        "mass_kilograms": alt.LocoParams.default().to_pydict()["mass_kilograms"],
        "save_interval": SAVE_INTERVAL,
    }
)

hel: alt.Locomotive = alt.Locomotive.default_hybrid_electric_loco()

# construct a vector of one BEL and several conventional locomotives
loco_vec = [bel.copy()] + [hel.copy()] + [alt.Locomotive.default()] * 7

# instantiate consist
loco_con = alt.Consist(loco_vec)

# Instantiate the intermediate `TrainSimBuilder`
tsb = alt.TrainSimBuilder(
    train_id="0",
    origin_id="A",
    destination_id="B",
    train_config=train_config,
    loco_con=loco_con,
)

# Load the network and construct the timed link path through the network.
network = alt.Network.from_file(
    alt.resources_root() / "networks/simple_corridor_network.yaml"
)

location_map = alt.import_locations(
    alt.resources_root() / "networks/simple_corridor_locations.csv"
)
train_sim: alt.SetSpeedTrainSim = tsb.make_speed_limit_train_sim(
    location_map=location_map,
    save_interval=1,
)
train_sim.set_save_interval(SAVE_INTERVAL)
est_time_net, _consist = alt.make_est_times(train_sim, network)

timed_link_path = alt.run_dispatch(
    network,
    alt.SpeedLimitTrainSimVec([train_sim]),
    [est_time_net],
    False,
    False,
)[0]

t0 = time.perf_counter()
train_sim.walk_timed_path(
    network=network,
    timed_path=timed_link_path,
)
t1 = time.perf_counter()
print(f"Time to simulate: {t1 - t0:.5g}")
ts_dict = train_sim.to_pydict()
assert len(ts_dict["history"]) > 1

# pull out solved locomotive for plotting convenience
loco0: alt.Locomotive = next(iter(ts_dict["loco_con"]["loco_vec"]))
loco0_type = next(iter(loco0["loco_type"].values()))

def plot_train_level_powers() -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(4, 1, sharex=True)
    plt.suptitle("Train Power")
    ax[0].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
        label="tract pwr",
    )
    ax[0].set_ylabel("Power [MW]")
    ax[0].legend()

    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["res_aero_newtons"]) / 1e3,
        label="aero",
    )
    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["res_rolling_newtons"]) / 1e3,
        label="rolling",
    )
    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["res_curve_newtons"]) / 1e3,
        label="curve",
    )
    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["res_bearing_newtons"]) / 1e3,
        label="bearing",
    )
    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["res_grade_newtons"]) / 1e3,
        label="grade",
    )
    ax[1].set_ylabel("Force [MN]")
    ax[1].legend()

    ax[2].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(loco0_type["res"]["history"]["soc"]),
    )
    ax[2].set_ylabel("SOC")

    ax[-1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        ts_dict["history"]["speed_meters_per_second"],
        label="achieved",
    )
    ax[-1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        ts_dict["history"]["speed_limit_meters_per_second"],
        label="limit",
    )
    ax[-1].set_xlabel("Time [hr]")
    ax[-1].set_ylabel("Speed [m/s]")
    ax[-1].legend()
    plt.suptitle("Speed Limit Train Sim Demo")

    return fig, ax

plot_locos_from_ts(train_sim, "Time", show_plots=SHOW_PLOTS)


def plot_train_network_info() -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Train Position in Network")
    ax[0].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["offset_in_link_meters"]) / 1_000,
        label="current link",
    )
    ax[0].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["offset_meters"]) / 1_000,
        label="overall",
    )
    ax[0].legend()
    ax[0].set_ylabel("Net Dist. [km]")

    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        ts_dict["history"]["link_idx_front"],
        linestyle="",
        marker=".",
    )
    ax[1].set_ylabel("Link Idx Front")

    ax[-1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        ts_dict["history"]["speed_meters_per_second"],
    )
    ax[-1].set_xlabel("Time [hr]")
    ax[-1].set_ylabel("Speed [m/s]")

    plt.suptitle("Speed Limit Train Sim Demo")
    plt.tight_layout()

    return fig, ax


def plot_consist_pwr() -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Loco. Consist")
    ax[0].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
        label="consist tract pwr",
    )
    ax[0].set_ylabel("Power [MW]")
    ax[0].legend()

    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["grade_front"]) * 100.0,
    )
    ax[1].set_ylabel("Grade [%] at\nHead End")

    ax[-1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        ts_dict["history"]["speed_meters_per_second"],
    )
    ax[-1].set_xlabel("Time [hr]")
    ax[-1].set_ylabel("Speed [m/s]")

    return fig, ax


ts_dict = train_sim.to_pydict()
hybrid_loco = ts_dict["loco_con"]["loco_vec"][1]
hel_type = "HybridLoco"


def plot_hel_pwr_and_soc() -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Hybrid Locomotive")

    ax_idx = 0
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(hybrid_loco["history"]["pwr_out_watts"]) / 1e3,
        label="tract. pwr.",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(
            hybrid_loco["loco_type"][hel_type]["res"]["history"]["pwr_disch_max_watts"]
        )
        / 1e3,
        label="batt. max disch. pwr",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(
            hybrid_loco["loco_type"][hel_type]["res"]["history"]["pwr_charge_max_watts"]
        )
        / 1e3,
        label="batt. max chrg. pwr",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(
            hybrid_loco["loco_type"][hel_type]["res"]["history"][
                "pwr_out_electrical_watts"
            ]
        )
        / 1e3,
        label="batt. elec. pwr.",
    )
    pwr_gen_elect_out = np.array(
        hybrid_loco["loco_type"][hel_type]["gen"]["history"]["pwr_elec_prop_out_watts"]
    ) + np.array(
        hybrid_loco["loco_type"][hel_type]["gen"]["history"]["pwr_elec_aux_watts"]
    )
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


batt_loco = ts_dict["loco_con"]["loco_vec"][0]


bel_type = "BatteryElectricLoco"


def plot_bel_pwr_and_soc() -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Battery Electric Locomotive")

    ax_idx = 0
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(batt_loco["history"]["pwr_out_watts"]) / 1e3,
        label="tract. pwr.",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(
            batt_loco["loco_type"][bel_type]["res"]["history"]["pwr_disch_max_watts"]
        )
        / 1e3,
        label="batt. max disch. pwr",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(
            batt_loco["loco_type"][bel_type]["res"]["history"]["pwr_charge_max_watts"]
        )
        / 1e3,
        label="batt. max chrg. pwr",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(
            batt_loco["loco_type"][bel_type]["res"]["history"][
                "pwr_out_electrical_watts"
            ]
        )
        / 1e3,
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
    ax[ax_idx].set_ylabel("[-]")
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        ts_dict["history"]["speed_meters_per_second"],
    )
    ax[ax_idx].set_ylabel("Speed [m/s]")
    ax[ax_idx].set_xlabel("Times [s]")
    plt.tight_layout()

    return fig, ax


fig0, ax0 = plot_train_level_powers()
fig1, ax1 = plot_train_network_info()
fig2, ax2 = plot_consist_pwr()
fig3, ax3 = plot_hel_pwr_and_soc()
fig4, ax4 = plot_bel_pwr_and_soc()

if SHOW_PLOTS:
    plt.show()
