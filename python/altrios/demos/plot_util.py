import altrios as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
            # Hybrid loco's loco type is somehow still BEL
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
    if isinstance(ts,alt.SpeedLimitTrainSim):
        plot_name = "Speed Limit Train Sim"
    if isinstance(ts,alt.SetSpeedTrainSim):
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
        if isinstance(ts,alt.SpeedLimitTrainSim):
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

    if extract_conv_from_train_sim(ts) != False:
        first_conv = extract_conv_from_train_sim(ts)[0]
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
        if isinstance(ts,alt.SpeedLimitTrainSim):
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

    if extract_hel_from_train_sim(ts) != False:
        first_hel = extract_hel_from_train_sim(ts)[0]
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
        if isinstance(ts,alt.SpeedLimitTrainSim):
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
            np.array(first_hel['history']['pwr_out_watts']) / 1e3,
            label='hybrid tract. pwr.'
        )
        ax3[0].plot(
            x_axis,
            np.array(first_hel.res.history.pwr_out_pwr_out_electrical_watts) / 1e3,
            label='hybrid batt. elec. pwr.'
        )
        ax3[0].set_ylabel('Power [kW]')
        ax3[0].legend()
        ax3[1].plot(
            x_axis,
            first_hel.res.history.soc,
            label='soc'
        )
        ax3[1].plot(
            x_axis,
            first_hel.res.history.soc_chrg_buffer,
            label='chrg buff'
        )
        ax3[1].plot(
            x_axis,
            first_hel.res.history.soc_disch_buffer,
            label='disch buff'
        )
        ax3[1].set_ylabel('[-]')
        ax3[1].legend()
        ax3[2].plot(
            x_axis,
            ts.history.speed_meters_per_second,
        )
        ax3[2].set_ylabel('Speed [m/s]')
        ax3[2].set_xlabel('Times [s]')
        plt.suptitle(plot_name + " " + "Hybrid Loco Power and Buffer Profile")
        plt.tight_layout()
        plt.show()
    return 