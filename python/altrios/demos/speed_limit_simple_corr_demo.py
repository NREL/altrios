# %%
"""
SetSpeedTrainSim over a simple, hypothetical corridor
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import altrios as alt 
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
        if "Hybrid" in loco.loco_type():
            loco_list.append(loco)
    if not loco_list:
        print("NO HYBRID LOCO IS FOUND IN CONSIST")
        return False
    return loco_list

def plot_locos_from_ts(ts:alt.SetSpeedTrainSim,x:str, y:str):
    """
    Extracts first instance of each loco_type and plots representative plots
    Offers several plotting options to put on x and y axis
    x: ["time","offset"]
    y: ["Force Requirement" ,"Consumption"]
    """
    if x == "time" or x =="Time":
        x_axis = np.array(ts.history.time_seconds) / 3_600
        x_label = "Time (hr)"
    if x == "distance" or x == "Distance":
        x_axis = np.array(ts.history.offset_back_meters)
        x_label = "Distance (m)"
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
            label='achieved'
        )
        ax[-1].plot(
            x_axis,
            ts.history.speed_limit_meters_per_second,
            label='limit'
        )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel('Speed [m/s]')
        ax[-1].legend()
        plt.suptitle("Speed Limit Train Sim Demo")

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

        plt.suptitle("Speed Limit Train Sim Demo")
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

        plt.suptitle("Speed Limit Train Sim Demo")
        plt.tight_layout()

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
            label='achieved'
        )
        ax[-1].plot(
            x_axis,
            ts.history.speed_limit_meters_per_second,
            label='limit'
        )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel('Speed [m/s]')
        ax[-1].legend()
        plt.suptitle("Speed Limit Train Sim Demo")

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

        plt.suptitle("Speed Limit Train Sim Demo")
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

        plt.suptitle("Speed Limit Train Sim Demo")
        plt.tight_layout()
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
            np.array(first_bel.res.history.soc)
        )
        ax[2].set_ylabel('SOC')

        ax[-1].plot(
            x_axis,
            ts.history.speed_meters_per_second,
            label='achieved'
        )
        ax[-1].plot(
            x_axis,
            ts.history.speed_limit_meters_per_second,
            label='limit'
        )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel('Speed [m/s]')
        ax[-1].legend()
        plt.suptitle("Speed Limit Train Sim Demo")

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

        plt.suptitle("Speed Limit Train Sim Demo")
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

        plt.suptitle("Speed Limit Train Sim Demo")
        plt.tight_layout()
        fig3, ax3 = plt.subplots(3, 1, sharex=True)
        ax3[0].plot(
            x_axis,
            np.array(hybrid_loco['history']['pwr_out_watts']) / 1e3,
            label='hybrid tract. pwr.'
        )
        ax3[0].plot(
            x_axis,
            np.array(first_hel.res.history.pwr_out_pwr_out_electrical_watts) / 1e3,
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
        plt.tight_layout()
    
    return 
sns.set_theme()

SHOW_PLOTS = alt.utils.show_plots()

SAVE_INTERVAL = 1
res = alt.ReversibleEnergyStorage.from_file(
    alt.resources_root() / "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
)
# https://docs.rs/altrios-core/latest/altrios_core/train/struct.TrainConfig.html
train_config = alt.TrainConfig(
    cars_empty=50,
    cars_loaded=50,
    rail_vehicle_type="Manifest",
    train_type=None, 
    train_length_meters=None,
    train_mass_kilograms=None,
)

edrv = alt.ElectricDrivetrain(
    pwr_out_frac_interp=[0., 1.],
    eta_interp=[0.98, 0.98],
    pwr_out_max_watts=5e9,
    save_interval=SAVE_INTERVAL,
)

bel: alt.Locomotive = alt.Locomotive.build_battery_electric_loco(
    reversible_energy_storage=res,
    drivetrain=edrv,
    loco_params=alt.LocoParams.from_dict(dict(
        pwr_aux_offset_watts=8.55e3,
        pwr_aux_traction_coeff_ratio=540.e-6,
        force_max_newtons=667.2e3,
)))

# construct a vector of one BEL and several conventional locomotives
loco_vec = [bel.clone()] + [alt.Locomotive.default()] * 7
# instantiate consist
loco_con = alt.Consist(
    loco_vec
)


tsb = alt.TrainSimBuilder(
    train_id="0",
    origin_id="A",
    destination_id="B",
    train_config=train_config,
    loco_con=loco_con,
)

rail_vehicle_file = "rolling_stock/rail_vehicles.csv"
rail_vehicle_map = alt.import_rail_vehicles(alt.resources_root() / rail_vehicle_file)
rail_vehicle = rail_vehicle_map[train_config.rail_vehicle_type]

network = alt.Network.from_file(
    alt.resources_root() / 'networks/simple_corridor_network.yaml')

location_map = alt.import_locations(alt.resources_root() / "networks/simple_corridor_locations.csv")
train_sim: alt.SetSpeedTrainSim = tsb.make_speed_limit_train_sim(
    rail_vehicle=rail_vehicle,
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
print(f'Time to simulate: {t1 - t0:.5g}')
assert len(train_sim.history) > 1

# pull out solved locomotive for plotting convenience
loco0:alt.Locomotive = train_sim.loco_con.loco_vec.tolist()[0]

# fig, ax = plt.subplots(4, 1, sharex=True)
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

# ax[2].plot(
#     np.array(train_sim.history.time_seconds) / 3_600, 
#     np.array(loco0.res.history.soc)
# )

# ax[2].set_ylabel('SOC')
# ax[2].legend()

# ax[-1].plot(
#     np.array(train_sim.history.time_seconds) / 3_600,
#     train_sim.history.speed_meters_per_second,
#     label='achieved'
# )
# ax[-1].plot(
#     np.array(train_sim.history.time_seconds) / 3_600,
#     train_sim.history.speed_limit_meters_per_second,
#     label='limit'
# )
# ax[-1].set_xlabel('Time [hr]')
# ax[-1].set_ylabel('Speed [m/s]')
# ax[-1].legend()
# plt.suptitle("Speed Limit Train Sim Demo")
# if SHOW_PLOTS:
#     plt.tight_layout()
#     plt.show()

plot_locos_from_ts(train_sim,"Time",0)