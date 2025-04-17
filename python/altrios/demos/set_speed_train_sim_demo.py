# %%
import time
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import os

import altrios as alt 
sns.set_theme()
def extract_bel_from_train_sim(ts: alt.SetSpeedTrainSim) -> list:
    ts_list = ts.loco_con.loco_vec.tolist()
    loco_list = []
    for loco in ts_list:
        if "BatteryElectricLoco" in loco.loco_type():
            loco_list.append(loco)
    if not loco_list:
        print("NO BEL IS FOUND IN CONSIST")
        return False
    return loco_list

def extract_conv_from_train_sim(ts: alt.SetSpeedTrainSim) -> list:
    ts_list = ts.loco_con.loco_vec.tolist()
    loco_list = []
    for loco in ts_list:
        if "ConventionalLoco" in loco.loco_type():
            loco_list.append(loco)
    if not loco_list:
        print("NO CONVENTIONAL LOCO IS FOUND IN CONSIST")
        return False
    return loco_list

def extract_hel_from_train_sim(ts: alt.SetSpeedTrainSim) -> list:
    ts_list = ts.loco_con.loco_vec.tolist()
    loco_list = []
    for loco in ts_list:
        if "HybridLoco" in loco.loco_type():
            # Hybrid loco's loco type is somehow still BEL
            loco_list.append(loco)
    if not loco_list:
        print("NO HYBRID LOCO IS FOUND IN CONSIST")
        return False
    return loco_list

def plot_locos_from_ts(ts:alt.SetSpeedTrainSim,x:str):
    """
    Can take in either SetSpeedTrainSim or SpeedLimitTrainSim
    Extracts first instance of each loco_type and plots representative plots
    Offers two plotting options to put on x axis
    ts: train sim
    x: ["time","offset"]
    """
    if isinstance(train_sim,alt.SpeedLimitTrainSim):
        plot_name = "Speed Limit Train Sim"
    if isinstance(train_sim,alt.SetSpeedTrainSim):
        plot_name = "Set Speed Train Sim"
    if x == "time" or x =="Time":
        x_axis = np.array(ts.history.time_seconds) / 3_600
        x_label = "Time (hr)"
    if x == "distance" or x == "Distance":
        x_axis = np.array(ts.history.offset_back_meters) / 1_000
        x_label = "Distance (km)"
    first_bel = []
    first_conv = []
    first_hel = []
    if extract_bel_from_train_sim(ts) != False:
        first_bel = extract_bel_from_train_sim(ts)[0]
        '''
        first fig
        speed vs dist or time
        soc vs dist or time
        various powers along the powertrain vs dist or time
        various cumulative energies along the powertrain vs dist or time
        '''
        fig, ax = plt.subplots(4, 1, sharex=True)
        ax[0].plot(
            x_axis,
            np.array(ts.history.pwr_whl_out_watts) / 1e6,
            label="tract pwr",
        )
        ax[0].set_ylabel('Power [MW]')
        ax[0].legend()

        ax[1].plot(
            x_axis,
            np.array(ts.history.res_aero_newtons) / 1e3,
            label='aero',
        )
        ax[1].plot(
            x_axis,
            np.array(ts.history.res_rolling_newtons) / 1e3,
            label='rolling',
        )
        ax[1].plot(
            x_axis,
            np.array(ts.history.res_curve_newtons) / 1e3,
            label='curve',
        )
        ax[1].plot(
            x_axis,
            np.array(ts.history.res_bearing_newtons) / 1e3,
            label='bearing',
        )
        ax[1].plot(
            x_axis,
            np.array(ts.history.res_grade_newtons) / 1e3,
            label='grade',
        )
        ax[1].set_ylabel('Force [MN]')
        ax[1].legend()

        ax[2].plot(
            x_axis,
            np.array(first_bel.res.history.soc)
        )
        ax[2].set_ylabel('SOC')

        ax[-1].plot(
            x_axis,
            ts.history.speed_meters_per_second,
            label='achieved speed'
        )
        if isinstance(train_sim,alt.SpeedLimitTrainSim):
            ax[-1].plot(
                x_axis,
                ts.history.speed_limit_meters_per_second,
                label='limit'
            )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel('Speed [m/s]')
        ax[-1].legend()
        plt.suptitle(plot_name + " " + "Train Resistance, BEL SOC and Train Speed")

        fig1, ax1 = plt.subplots(3, 1, sharex=True)
        ax1[0].plot(
            x_axis,
            np.array(ts.history.offset_in_link_meters) / 1_000,
            label='current link',
        )
        ax1[0].plot(
            x_axis,
            np.array(ts.history.offset_meters) / 1_000,
            label='overall',
        )
        ax1[0].legend()
        ax1[0].set_ylabel('Net Dist. [km]')

        ax1[1].plot(
            x_axis,
            ts.history.link_idx_front,
            linestyle='',
            marker='.',
        )
        ax1[1].set_ylabel('Link Idx Front')

        ax1[-1].plot(
            x_axis,
            ts.history.speed_meters_per_second,
        )
        ax1[-1].set_xlabel(x_label)
        ax1[-1].set_ylabel('Speed [m/s]')

        plt.suptitle(plot_name + " " + "Distance and Link Tracking")
        plt.tight_layout()

        fig2, ax2 = plt.subplots(3, 1, sharex=True)
        ax2[0].plot(
            x_axis,
            np.array(ts.history.pwr_whl_out_watts) / 1e6,
            label="tract pwr",
        )
        ax2[0].set_ylabel('Power [MW]')
        ax2[0].legend()

        ax2[1].plot(
            x_axis,
            np.array(ts.history.grade_front) * 100.,
        )
        ax2[1].set_ylabel('Grade [%] at\nHead End')

        ax2[-1].plot(
            x_axis,
            ts.history.speed_meters_per_second,
        )
        ax2[-1].set_xlabel(x_label)
        ax2[-1].set_ylabel('Speed [m/s]')

        plt.suptitle(plot_name + " " + "Power and Grade Profile")
        plt.tight_layout()
        plt.show()

        # fig, ax = plt.subplots(3, 1, sharex=True)
        # ax[0].plot(
        #     x_axis,
        #     ts.history.speed_meters_per_second,
        #     label='achieved'
        # )
        # ax[0].plot(
        #     x_axis,
        #     ts.history.speed_limit_meters_per_second,
        #     label='limit'
        # )
        # ax[0].set_xlabel(x_label)
        # ax[0].set_ylabel('Speed [m/s]')
        # ax[0].legend()
        # ax[1].plot(
        #     x_axis, 
        #     np.array(first_bel.res.history.soc),
        #     label = "SOC"
        # )
        # ax[1].set_ylabel('SOC')
        # ax[1].legend()

        # ax[2].plot(
        #     x_axis,
        #     np.array(ts.history.res_aero_newtons) / 1e3,
        #     label='aero',
        # )
        # ax[2].plot(
        #     x_axis,
        #     np.array(ts.history.res_rolling_newtons) / 1e3,
        #     label='rolling',
        # )
        # ax[2].plot(
        #     x_axis,
        #     np.array(ts.history.res_curve_newtons) / 1e3,
        #     label='curve',
        # )
        # ax[2].plot(
        #     x_axis,
        #     np.array(ts.history.res_bearing_newtons) / 1e3,
        #     label='bearing',
        # )
        # ax[2].plot(
        #     x_axis,
        #     np.array(ts.history.res_grade_newtons) / 1e3,
        #     label='grade',
        # )
        # ax[2].set_ylabel('Force [MN]')
        # ax[2].legend()
        # plt.suptitle("BEL Speed Limit Train Sim Demo")
        # plt.tight_layout()
        # plt.show()
        # fig, ax = plt.subplots(3, 1, sharex=True)
        # ax[0].plot(
        #     x_axis,
        #     ts.history.speed_meters_per_second,
        #     label='achieved'
        # )
        # ax[0].plot(
        #     x_axis,
        #     ts.history.speed_limit_meters_per_second,
        #     label='limit'
        # )
        # ax[0].set_xlabel(x_label)
        # ax[0].set_ylabel('Speed [m/s]')
        # ax[0].legend()
        # ax[1].plot(
        #     x_axis, 
        #     np.array(first_bel.res.history.soc),
        #     label = "SOC"
        # )
        # ax[1].set_ylabel('SOC')
        # ax[1].legend()

        # cumulative_aero = np.cumsum(np.array(ts.history.res_aero_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # cumulative_rolling = np.cumsum(np.array(ts.history.res_rolling_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # cumulative_curve = np.cumsum(np.array(ts.history.res_curve_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # cumulative_bearing = np.cumsum(np.array(ts.history.res_bearing_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # cumulative_grade = np.cumsum(np.array(ts.history.res_grade_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_aero),
        #     label='aero',
        # )
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_rolling),
        #     label='rolling',
        # )
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_curve),
        #     label='curve',
        # )
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_bearing),
        #     label='bearing',
        # )
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_grade),
        #     label='grade',
        # )
        # ax[-1].set_xlabel(x_label)
        # ax[-1].set_ylabel('Cumulative Energy [J]')
        # ax[-1].legend()
        # plt.suptitle("BEL Speed Limit Train Sim Demo")
        # plt.tight_layout()
        # plt.show()
    if extract_conv_from_train_sim(ts) != False:
        first_conv = extract_conv_from_train_sim(ts)[0]
        fig, ax = plt.subplots(4, 1, sharex=True)
        #Need to find the current param for this:
        #np.array(first_conv.fc.state.pwr_out_frac_interp*pwr_out_max_watts/1e6)
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(
            x_axis,
            np.array(ts.history.pwr_whl_out_watts) / 1e6,
            label="tract pwr",
        )
        ax[0].set_ylabel('Power [MW]')
        ax[0].legend()

        ax[1].plot(
            x_axis,
            np.array(ts.history.res_aero_newtons) / 1e3,
            label='aero',
        )
        ax[1].plot(
            x_axis,
            np.array(ts.history.res_rolling_newtons) / 1e3,
            label='rolling',
        )
        ax[1].plot(
            x_axis,
            np.array(ts.history.res_curve_newtons) / 1e3,
            label='curve',
        )
        ax[1].plot(
            x_axis,
            np.array(ts.history.res_bearing_newtons) / 1e3,
            label='bearing',
        )
        ax[1].plot(
            x_axis,
            np.array(ts.history.res_grade_newtons) / 1e3,
            label='grade',
        )
        ax[1].set_ylabel('Force [MN]')
        ax[1].legend()

        ax[-1].plot(
            x_axis,
            ts.history.speed_meters_per_second,
            label='achieved speed'
        )
        if isinstance(train_sim,alt.SpeedLimitTrainSim):
            ax[-1].plot(
                x_axis,
                ts.history.speed_limit_meters_per_second,
                label='limit'
            )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel('Speed [m/s]')
        ax[-1].legend()
        plt.suptitle(plot_name + " " + "Train Resistance, and Train Speed")

        fig1, ax1 = plt.subplots(3, 1, sharex=True)
        ax1[0].plot(
            x_axis,
            np.array(ts.history.offset_in_link_meters) / 1_000,
            label='current link',
        )
        ax1[0].plot(
            x_axis,
            np.array(ts.history.offset_meters) / 1_000,
            label='overall',
        )
        ax1[0].legend()
        ax1[0].set_ylabel('Net Dist. [km]')

        ax1[1].plot(
            x_axis,
            ts.history.link_idx_front,
            linestyle='',
            marker='.',
        )
        ax1[1].set_ylabel('Link Idx Front')

        ax1[-1].plot(
            x_axis,
            ts.history.speed_meters_per_second,
        )
        ax1[-1].set_xlabel(x_label)
        ax1[-1].set_ylabel('Speed [m/s]')

        plt.suptitle(plot_name + " " + "Distance and Link Tracking")
        plt.tight_layout()

        fig2, ax2 = plt.subplots(3, 1, sharex=True)
        ax2[0].plot(
            x_axis,
            np.array(ts.history.pwr_whl_out_watts) / 1e6,
            label="tract pwr",
        )
        ax2[0].set_ylabel('Power [MW]')
        ax2[0].legend()

        ax2[1].plot(
            x_axis,
            np.array(ts.history.grade_front) * 100.,
        )
        ax2[1].set_ylabel('Grade [%] at\nHead End')

        ax2[-1].plot(
            x_axis,
            ts.history.speed_meters_per_second,
        )
        ax2[-1].set_xlabel(x_label)
        ax2[-1].set_ylabel('Speed [m/s]')

        plt.suptitle(plot_name + " " + "Power and Grade Profile")
        plt.tight_layout()
        plt.show()
        # fig, ax = plt.subplots(2, 1, sharex=True)
        # ax[0].plot(
        #     x_axis,
        #     ts.history.speed_meters_per_second,
        #     label='achieved'
        # )
        # ax[0].plot(
        #     x_axis,
        #     ts.history.speed_limit_meters_per_second,
        #     label='limit'
        # )
        # ax[0].set_xlabel(x_label)
        # ax[0].set_ylabel('Speed [m/s]')
        # ax[0].legend()

        # ax[1].plot(
        #     x_axis,
        #     np.array(ts.history.res_aero_newtons) / 1e3,
        #     label='aero',
        # )
        # ax[1].plot(
        #     x_axis,
        #     np.array(ts.history.res_rolling_newtons) / 1e3,
        #     label='rolling',
        # )
        # ax[1].plot(
        #     x_axis,
        #     np.array(ts.history.res_curve_newtons) / 1e3,
        #     label='curve',
        # )
        # ax[1].plot(
        #     x_axis,
        #     np.array(ts.history.res_bearing_newtons) / 1e3,
        #     label='bearing',
        # )
        # ax[1].plot(
        #     x_axis,
        #     np.array(ts.history.res_grade_newtons) / 1e3,
        #     label='grade',
        # )
        # ax[1].set_ylabel('Force [MN]')
        # ax[1].legend()
        # plt.suptitle("Conventional Loco Speed Limit Train Sim Demo")
        # plt.tight_layout()
        # plt.show()
        # fig, ax = plt.subplots(2, 1, sharex=True)
        # ax[0].plot(
        #     x_axis,
        #     ts.history.speed_meters_per_second,
        #     label='achieved'
        # )
        # ax[0].plot(
        #     x_axis,
        #     ts.history.speed_limit_meters_per_second,
        #     label='limit'
        # )
        # ax[0].set_xlabel(x_label)
        # ax[0].set_ylabel('Speed [m/s]')
        # ax[0].legend()

        # cumulative_aero = np.cumsum(np.array(ts.history.res_aero_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # cumulative_rolling = np.cumsum(np.array(ts.history.res_rolling_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # cumulative_curve = np.cumsum(np.array(ts.history.res_curve_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # cumulative_bearing = np.cumsum(np.array(ts.history.res_bearing_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # cumulative_grade = np.cumsum(np.array(ts.history.res_grade_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_aero),
        #     label='aero',
        # )
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_rolling),
        #     label='rolling',
        # )
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_curve),
        #     label='curve',
        # )
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_bearing),
        #     label='bearing',
        # )
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_grade),
        #     label='grade',
        # )
        # ax[-1].set_xlabel(x_label)
        # ax[-1].set_ylabel('Cumulative Energy [J]')
        # ax[-1].legend()
        # plt.suptitle("Conventional Loco Speed Limit Train Sim Demo")
        # plt.tight_layout()
        # plt.show()
    if extract_hel_from_train_sim(ts) != False:
        first_hel = extract_hel_from_train_sim(ts)[0]
        ts_dict = ts.to_pydict()
        # fig, ax = plt.subplots(3, 1, sharex=True)
        # ax[0].plot(
        #     x_axis,
        #     ts.history.speed_meters_per_second,
        #     label='achieved'
        # )
        # ax[0].plot(
        #     x_axis,
        #     ts.history.speed_limit_meters_per_second,
        #     label='limit'
        # )
        # ax[0].set_xlabel(x_label)
        # ax[0].set_ylabel('Speed [m/s]')
        # ax[0].legend()
        # ax[1].plot(
        #     x_axis, 
        #     np.array(first_bel.res.history.soc),
        #     label = "SOC"
        # )
        # ax[1].set_ylabel('SOC')
        # ax[1].legend()

        # ax[2].plot(
        #     x_axis,
        #     np.array(ts.history.res_aero_newtons) / 1e3,
        #     label='aero',
        # )
        # ax[2].plot(
        #     x_axis,
        #     np.array(ts.history.res_rolling_newtons) / 1e3,
        #     label='rolling',
        # )
        # ax[2].plot(
        #     x_axis,
        #     np.array(ts.history.res_curve_newtons) / 1e3,
        #     label='curve',
        # )
        # ax[2].plot(
        #     x_axis,
        #     np.array(ts.history.res_bearing_newtons) / 1e3,
        #     label='bearing',
        # )
        # ax[2].plot(
        #     x_axis,
        #     np.array(ts.history.res_grade_newtons) / 1e3,
        #     label='grade',
        # )
        # ax[2].set_ylabel('Force [MN]')
        # ax[2].legend()
        # plt.suptitle("Hybrid Loco Speed Limit Train Sim Demo")
        # plt.tight_layout()
        # plt.show()
        # fig, ax = plt.subplots(3, 1, sharex=True)
        # ax[0].plot(
        #     x_axis,
        #     ts.history.speed_meters_per_second,
        #     label='achieved'
        # )
        # ax[0].plot(
        #     x_axis,
        #     ts.history.speed_limit_meters_per_second,
        #     label='limit'
        # )
        # ax[0].set_xlabel(x_label)
        # ax[0].set_ylabel('Speed [m/s]')
        # ax[0].legend()
        # ax[1].plot(
        #     x_axis, 
        #     np.array(first_bel.res.history.soc),
        #     label = "SOC"
        # )
        # ax[1].set_ylabel('SOC')
        # ax[1].legend()
        # cumulative_aero = np.cumsum(np.array(ts.history.res_aero_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # cumulative_rolling = np.cumsum(np.array(ts.history.res_rolling_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # cumulative_curve = np.cumsum(np.array(ts.history.res_curve_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # cumulative_bearing = np.cumsum(np.array(ts.history.res_bearing_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # cumulative_grade = np.cumsum(np.array(ts.history.res_grade_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_aero),
        #     label='aero',
        # )
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_rolling),
        #     label='rolling',
        # )
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_curve),
        #     label='curve',
        # )
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_bearing),
        #     label='bearing',
        # )
        # ax[-1].plot(
        #     x_axis,
        #     np.array(cumulative_grade),
        #     label='grade',
        # )
        # ax[-1].set_xlabel(x_label)
        # ax[-1].set_ylabel('Cumulative Energy [J]')
        # ax[-1].legend()
        # plt.suptitle("Speed Limit Train Sim Demo")
        # plt.suptitle("Hybrid Loco Speed Limit Train Sim Demo")
        # plt.tight_layout()
        # plt.show()
        fig, ax = plt.subplots(4, 1, sharex=True)
        ax[0].plot(
            x_axis,
            np.array(ts.history.pwr_whl_out_watts) / 1e6,
            label="tract pwr",
        )
        ax[0].set_ylabel('Power [MW]')
        ax[0].legend()

        ax[1].plot(
            x_axis,
            np.array(ts.history.res_aero_newtons) / 1e3,
            label='aero',
        )
        ax[1].plot(
            x_axis,
            np.array(ts.history.res_rolling_newtons) / 1e3,
            label='rolling',
        )
        ax[1].plot(
            x_axis,
            np.array(ts.history.res_curve_newtons) / 1e3,
            label='curve',
        )
        ax[1].plot(
            x_axis,
            np.array(ts.history.res_bearing_newtons) / 1e3,
            label='bearing',
        )
        ax[1].plot(
            x_axis,
            np.array(ts.history.res_grade_newtons) / 1e3,
            label='grade',
        )
        ax[1].set_ylabel('Force [MN]')
        ax[1].legend()

        ax[2].plot(
            x_axis,
            np.array(first_hel.res.history.soc)
        )
        ax[2].set_ylabel('SOC')

        ax[-1].plot(
            x_axis,
            ts.history.speed_meters_per_second,
            label='achieved speed'
        )
        if isinstance(train_sim,alt.SpeedLimitTrainSim):
            ax[-1].plot(
                x_axis,
                ts.history.speed_limit_meters_per_second,
                label='limit'
            )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel('Speed [m/s]')
        ax[-1].legend()
        plt.suptitle(plot_name + " " + "Train Resistance, Battery SOC and Train Speed")

        fig1, ax1 = plt.subplots(3, 1, sharex=True)
        ax1[0].plot(
            x_axis,
            np.array(ts.history.offset_in_link_meters) / 1_000,
            label='current link',
        )
        ax1[0].plot(
            x_axis,
            np.array(ts.history.offset_meters) / 1_000,
            label='overall',
        )
        ax1[0].legend()
        ax1[0].set_ylabel('Net Dist. [km]')

        ax1[1].plot(
            x_axis,
            ts.history.link_idx_front,
            linestyle='',
            marker='.',
        )
        ax1[1].set_ylabel('Link Idx Front')

        ax1[-1].plot(
            x_axis,
            ts.history.speed_meters_per_second,
        )
        ax1[-1].set_xlabel(x_label)
        ax1[-1].set_ylabel('Speed [m/s]')

        plt.suptitle(plot_name + " " + "Distance and Link Tracking")
        plt.tight_layout()

        fig2, ax2 = plt.subplots(3, 1, sharex=True)
        ax2[0].plot(
            x_axis,
            np.array(ts.history.pwr_whl_out_watts) / 1e6,
            label="tract pwr",
        )
        ax2[0].set_ylabel('Power [MW]')
        ax2[0].legend()

        ax2[1].plot(
            x_axis,
            np.array(ts.history.grade_front) * 100.,
        )
        ax2[1].set_ylabel('Grade [%] at\nHead End')

        ax2[-1].plot(
            x_axis,
            ts.history.speed_meters_per_second,
        )
        ax2[-1].set_xlabel(x_label)
        ax2[-1].set_ylabel('Speed [m/s]')

        plt.suptitle(plot_name + " " + "Power and Grade Profile")
        plt.tight_layout()
        fig3, ax3 = plt.subplots(3, 1, sharex=True)
        ax3[0].plot(
            x_axis,
            np.array(first_hel.history.pwr_out_watts) / 1e3,
            label='hybrid tract. pwr.'
        )
        ax3[0].plot(
            x_axis,
            np.array(first_hel.res.history.pwr_out_electrical_watts) / 1e3,
            #np.array(hybrid_loco['loco_type']['HybridLoco']['res']
            #        ['history']['pwr_out_electrical_watts']) / 1e3,
            label='hybrid batt. elec. pwr.'
        )
        ax3[0].set_ylabel('Power [kW]')
        ax3[0].legend()
        ax3[1].plot(
            x_axis,
            first_hel.res.history.soc,
            #hybrid_loco['loco_type']['HybridLoco']['res']['history']['soc'],
            label='soc'
        )
        ax3[1].plot(
            x_axis,
            first_hel.res.history.soc_chrg_buffer,
            #hybrid_loco['loco_type']['HybridLoco']['res']['history']['soc_chrg_buffer'],
            label='chrg buff'
        )
        ax3[1].plot(
            x_axis,
            first_hel.res.history.soc_disch_buffer,
            #hybrid_loco['loco_type']['HybridLoco']['res']['history']['soc_disch_buffer'],
            label='disch buff'
        )
        ax3[1].set_ylabel('[-]')
        ax3[1].legend()
        ax3[2].plot(
            x_axis,
            ts.history.speed_meters_per_second,
            #ts_dict['history']['speed_meters_per_second'],
        )
        ax3[2].set_ylabel('Speed [m/s]')
        ax3[2].set_xlabel('Times [s]')
        plt.suptitle(plot_name + " " + "Hybrid Loco Power and Buffer Profile")
        plt.tight_layout()
        plt.show()
    
    return 
SHOW_PLOTS = alt.utils.show_plots()
PYTEST = os.environ.get("PYTEST", "false").lower() == "true"

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
# instantiate battery model
# https://docs.rs/altrios-core/latest/altrios_core/consist/locomotive/powertrain/reversible_energy_storage/struct.ReversibleEnergyStorage.html#
res = alt.ReversibleEnergyStorage.from_file(
    alt.resources_root() / "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
)
# instantiate electric drivetrain (motors and any gearboxes)
# https://docs.rs/altrios-core/latest/altrios_core/consist/locomotive/powertrain/electric_drivetrain/struct.ElectricDrivetrain.html
edrv = alt.ElectricDrivetrain(
    pwr_out_frac_interp=[0., 1.],
    eta_interp=[0.98, 0.98],
    pwr_out_max_watts=5e9,
    save_interval=SAVE_INTERVAL,
)

loco_type = alt.BatteryElectricLoco.from_pydict({
    "res": res.to_pydict(),
    "edrv": edrv.to_pydict(),
})

bel: alt.Locomotive = alt.Locomotive(
    loco_type=loco_type,
    loco_params=alt.LocoParams.from_dict(dict(
        pwr_aux_offset_watts=8.55e3,
        pwr_aux_traction_coeff_ratio=540.e-6,
        force_max_newtons=667.2e3,
)))
hel: alt.Locomotive = alt.Locomotive.default_hybrid_electric_loco()
# construct a vector of one BEL and several conventional locomotives
loco_vec = [bel] + [alt.Locomotive.default()] * 7 + [hel]
# instantiate consist
loco_con = alt.Consist(
    loco_vec,
    SAVE_INTERVAL,
)

# Instantiate the intermediate `TrainSimBuilder`
tsb = alt.TrainSimBuilder(
    train_id="0",
    train_config=train_config,
    loco_con=loco_con,
)

# Load the network and link path through the network.  
network = alt.Network.from_file(
    alt.resources_root() / "networks/Taconite-NoBalloon.yaml")
link_path = alt.LinkPath.from_csv_file(
    alt.resources_root() / "demo_data/link_path.csv"
)

# load the prescribed speed trace that the train will follow
speed_trace = alt.SpeedTrace.from_csv_file(
    alt.resources_root() / "demo_data/speed_trace.csv"
)

train_sim: alt.SetSpeedTrainSim = tsb.make_set_speed_train_sim(
    network=network,
    link_path=link_path,
    speed_trace=speed_trace,
    save_interval=SAVE_INTERVAL,
)

alt.utils.set_log_level("WARNING")

train_sim.set_save_interval(SAVE_INTERVAL)
t0 = time.perf_counter()
train_sim.walk()
t1 = time.perf_counter()
print(f'Time to simulate: {t1 - t0:.5g}')

df = train_sim.to_dataframe()

# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].plot(
#     np.array(train_sim.history.time_seconds) / 3_600,
#     np.array(train_sim.history.pwr_whl_out_watts) / 1e6,
#     label="tract pwr",
# )
# ax[0].set_ylabel('Power [MW]')
# ax[0].legend()

# ax[1].plot(
#     np.array(train_sim.history.time_seconds) / 3_600,
#     np.array(train_sim.history.res_aero_newtons) / 1e3,
#     label='aero',
# )
# ax[1].plot(
#     np.array(train_sim.history.time_seconds) / 3_600,
#     np.array(train_sim.history.res_rolling_newtons) / 1e3,
#     label='rolling',
# )
# ax[1].plot(
#     np.array(train_sim.history.time_seconds) / 3_600,
#     np.array(train_sim.history.res_curve_newtons) / 1e3,
#     label='curve',
# )
# ax[1].plot(
#     np.array(train_sim.history.time_seconds) / 3_600,
#     np.array(train_sim.history.res_bearing_newtons) / 1e3,
#     label='bearing',
# )
# ax[1].plot(
#     np.array(train_sim.history.time_seconds) / 3_600,
#     np.array(train_sim.history.res_grade_newtons) / 1e3,
#     label='grade',
# )
# ax[1].set_ylabel('Force [MN]')
# ax[1].legend()

# ax[-1].plot(
#     np.array(train_sim.history.time_seconds) / 3_600,
#     np.array(train_sim.speed_trace.speed_meters_per_second)[::SAVE_INTERVAL],
# )
# ax[-1].set_xlabel('Time [hr]')
# ax[-1].set_ylabel('Speed [m/s]')

# plt.suptitle("Set Speed Train Sim Demo")
plot_locos_from_ts(train_sim,"Distance")
# if SHOW_PLOTS:
#     plt.tight_layout()
#     plt.show()

# whether to run assertions, enabled by default
ENABLE_ASSERTS = os.environ.get("ENABLE_ASSERTS", "true").lower() == "true"
# whether to override reference files used in assertions, disabled by default
ENABLE_REF_OVERRIDE = os.environ.get("ENABLE_REF_OVERRIDE", "false").lower() == "true"
# directory for reference files for checking sim results against expected results
ref_dir = alt.resources_root() / "demo_data/set_speed_train_sim_demo/"

if ENABLE_REF_OVERRIDE:
    ref_dir.mkdir(exist_ok=True, parents=True)
    df:pl.DataFrame = train_sim.to_dataframe().lazy().collect()[-1]
    df.write_csv(ref_dir / "to_dataframe_expected.csv")
if ENABLE_ASSERTS:
    print("Checking output of `to_dataframe`")
    to_dataframe_expected = pl.scan_csv(ref_dir / "to_dataframe_expected.csv").collect()[-1]
    assert to_dataframe_expected.equals(train_sim.to_dataframe()[-1]), \
        f"to_dataframe_expected: \n{to_dataframe_expected}\ntrain_sim.to_dataframe()[-1]: \n{train_sim.to_dataframe()[-1]}" + \
            "\ntry running with `ENABLE_REF_OVERRIDE=True`"
    print("Success!")
