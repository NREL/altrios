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
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(
            x_axis,
            ts.history.speed_meters_per_second,
            label='achieved'
        )
        ax[0].plot(
            x_axis,
            ts.history.speed_limit_meters_per_second,
            label='limit'
        )
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel('Speed [m/s]')
        ax[0].legend()
        ax[1].plot(
            x_axis, 
            np.array(first_bel.res.history.soc),
            label = "SOC"
        )
        ax[1].set_ylabel('SOC')
        ax[1].legend()

        ax[2].plot(
            x_axis,
            np.array(ts.history.res_aero_newtons) / 1e3,
            label='aero',
        )
        ax[2].plot(
            x_axis,
            np.array(ts.history.res_rolling_newtons) / 1e3,
            label='rolling',
        )
        ax[2].plot(
            x_axis,
            np.array(ts.history.res_curve_newtons) / 1e3,
            label='curve',
        )
        ax[2].plot(
            x_axis,
            np.array(ts.history.res_bearing_newtons) / 1e3,
            label='bearing',
        )
        ax[2].plot(
            x_axis,
            np.array(ts.history.res_grade_newtons) / 1e3,
            label='grade',
        )
        ax[2].set_ylabel('Force [MN]')
        ax[2].legend()
        plt.suptitle("BEL Speed Limit Train Sim Demo")
        plt.tight_layout()
        plt.show()
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(
            x_axis,
            ts.history.speed_meters_per_second,
            label='achieved'
        )
        ax[0].plot(
            x_axis,
            ts.history.speed_limit_meters_per_second,
            label='limit'
        )
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel('Speed [m/s]')
        ax[0].legend()
        ax[1].plot(
            x_axis, 
            np.array(first_bel.res.history.soc),
            label = "SOC"
        )
        ax[1].set_ylabel('SOC')
        ax[1].legend()

        cumulative_aero = np.cumsum(np.array(ts.history.res_aero_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        cumulative_rolling = np.cumsum(np.array(ts.history.res_rolling_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        cumulative_curve = np.cumsum(np.array(ts.history.res_curve_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        cumulative_bearing = np.cumsum(np.array(ts.history.res_bearing_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        cumulative_grade = np.cumsum(np.array(ts.history.res_grade_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        ax[-1].plot(
            x_axis,
            np.array(cumulative_aero),
            label='aero',
        )
        ax[-1].plot(
            x_axis,
            np.array(cumulative_rolling),
            label='rolling',
        )
        ax[-1].plot(
            x_axis,
            np.array(cumulative_curve),
            label='curve',
        )
        ax[-1].plot(
            x_axis,
            np.array(cumulative_bearing),
            label='bearing',
        )
        ax[-1].plot(
            x_axis,
            np.array(cumulative_grade),
            label='grade',
        )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel('Cumulative Energy [J]')
        ax[-1].legend()
        plt.suptitle("BEL Speed Limit Train Sim Demo")
        plt.tight_layout()
        plt.show()
    if extract_conv_from_train_sim(ts) != False:
        first_conv = extract_conv_from_train_sim(ts)[0]
        fig, ax = plt.subplots(4, 1, sharex=True)
        #Need to find the current param for this:
        #np.array(first_conv.fc.state.pwr_out_frac_interp*pwr_out_max_watts/1e6)
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(
            x_axis,
            ts.history.speed_meters_per_second,
            label='achieved'
        )
        ax[0].plot(
            x_axis,
            ts.history.speed_limit_meters_per_second,
            label='limit'
        )
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel('Speed [m/s]')
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
        plt.suptitle("Conventional Loco Speed Limit Train Sim Demo")
        plt.tight_layout()
        plt.show()
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(
            x_axis,
            ts.history.speed_meters_per_second,
            label='achieved'
        )
        ax[0].plot(
            x_axis,
            ts.history.speed_limit_meters_per_second,
            label='limit'
        )
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel('Speed [m/s]')
        ax[0].legend()

        cumulative_aero = np.cumsum(np.array(ts.history.res_aero_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        cumulative_rolling = np.cumsum(np.array(ts.history.res_rolling_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        cumulative_curve = np.cumsum(np.array(ts.history.res_curve_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        cumulative_bearing = np.cumsum(np.array(ts.history.res_bearing_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        cumulative_grade = np.cumsum(np.array(ts.history.res_grade_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        ax[-1].plot(
            x_axis,
            np.array(cumulative_aero),
            label='aero',
        )
        ax[-1].plot(
            x_axis,
            np.array(cumulative_rolling),
            label='rolling',
        )
        ax[-1].plot(
            x_axis,
            np.array(cumulative_curve),
            label='curve',
        )
        ax[-1].plot(
            x_axis,
            np.array(cumulative_bearing),
            label='bearing',
        )
        ax[-1].plot(
            x_axis,
            np.array(cumulative_grade),
            label='grade',
        )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel('Cumulative Energy [J]')
        ax[-1].legend()
        plt.suptitle("Conventional Loco Speed Limit Train Sim Demo")
        plt.tight_layout()
        plt.show()
    if extract_hel_from_train_sim(ts) != False:
        first_hel = extract_hel_from_train_sim(ts)[0]
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(
            x_axis,
            ts.history.speed_meters_per_second,
            label='achieved'
        )
        ax[0].plot(
            x_axis,
            ts.history.speed_limit_meters_per_second,
            label='limit'
        )
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel('Speed [m/s]')
        ax[0].legend()
        ax[1].plot(
            x_axis, 
            np.array(first_bel.res.history.soc),
            label = "SOC"
        )
        ax[1].set_ylabel('SOC')
        ax[1].legend()

        ax[2].plot(
            x_axis,
            np.array(ts.history.res_aero_newtons) / 1e3,
            label='aero',
        )
        ax[2].plot(
            x_axis,
            np.array(ts.history.res_rolling_newtons) / 1e3,
            label='rolling',
        )
        ax[2].plot(
            x_axis,
            np.array(ts.history.res_curve_newtons) / 1e3,
            label='curve',
        )
        ax[2].plot(
            x_axis,
            np.array(ts.history.res_bearing_newtons) / 1e3,
            label='bearing',
        )
        ax[2].plot(
            x_axis,
            np.array(ts.history.res_grade_newtons) / 1e3,
            label='grade',
        )
        ax[2].set_ylabel('Force [MN]')
        ax[2].legend()
        plt.suptitle("Hybrid Loco Speed Limit Train Sim Demo")
        plt.tight_layout()
        plt.show()
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(
            x_axis,
            ts.history.speed_meters_per_second,
            label='achieved'
        )
        ax[0].plot(
            x_axis,
            ts.history.speed_limit_meters_per_second,
            label='limit'
        )
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel('Speed [m/s]')
        ax[0].legend()
        ax[1].plot(
            x_axis, 
            np.array(first_bel.res.history.soc),
            label = "SOC"
        )
        ax[1].set_ylabel('SOC')
        ax[1].legend()
        cumulative_aero = np.cumsum(np.array(ts.history.res_aero_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        cumulative_rolling = np.cumsum(np.array(ts.history.res_rolling_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        cumulative_curve = np.cumsum(np.array(ts.history.res_curve_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        cumulative_bearing = np.cumsum(np.array(ts.history.res_bearing_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        cumulative_grade = np.cumsum(np.array(ts.history.res_grade_newtons) * train_sim.history.speed_meters_per_second * ts.get_save_interval())
        ax[-1].plot(
            x_axis,
            np.array(cumulative_aero),
            label='aero',
        )
        ax[-1].plot(
            x_axis,
            np.array(cumulative_rolling),
            label='rolling',
        )
        ax[-1].plot(
            x_axis,
            np.array(cumulative_curve),
            label='curve',
        )
        ax[-1].plot(
            x_axis,
            np.array(cumulative_bearing),
            label='bearing',
        )
        ax[-1].plot(
            x_axis,
            np.array(cumulative_grade),
            label='grade',
        )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel('Cumulative Energy [J]')
        ax[-1].legend()
        plt.suptitle("Speed Limit Train Sim Demo")
        plt.suptitle("Hybrid Loco Speed Limit Train Sim Demo")
        plt.tight_layout()
        plt.show()
    
    return 