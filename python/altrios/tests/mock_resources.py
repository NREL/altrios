from typing import Optional
import altrios as alt


def mock_fuel_converter(save_interval: Optional[int] = 1) -> alt.FuelConverter:
    fc = alt.FuelConverter.default()
    alt.set_param_from_path(fc, "save_interval", save_interval)
    return fc


def mock_generator(save_interval: Optional[int] = 1) -> alt.Generator:
    gen = alt.Generator.default()
    alt.set_param_from_path(gen, "save_interval", save_interval)
    return gen


def mock_reversible_energy_storage(save_interval: Optional[int] = 1) -> alt.ReversibleEnergyStorage:
    res = alt.ReversibleEnergyStorage.default()
    alt.set_param_from_path(res, "save_interval", save_interval)
    return res


def mock_electric_drivetrain(save_interval: Optional[int] = 1) -> alt.ElectricDrivetrain:
    edrv = alt.ElectricDrivetrain.default()
    alt.set_param_from_path(edrv, "save_interval", save_interval)
    return edrv


def mock_conventional_loco(
    fc: Optional[alt.FuelConverter] = None,
    gen: Optional[alt.Generator] = None,
    edrv: Optional[alt.ElectricDrivetrain] = None,
    save_interval: Optional[int] = 1,
    pwr_aux_offset_watts: float = 0.0,
    pwr_aux_traction_coeff: float = 0.0,
    force_max_newtons: float = 150000.0,
) -> alt.Locomotive:
    if not fc:
        fc = mock_fuel_converter(save_interval)
    if not gen:
        gen = mock_generator(save_interval)
    if not edrv:
        edrv = mock_electric_drivetrain(save_interval)

    loco_unit = alt.Locomotive.build_conventional_loco(
        fuel_converter=fc,
        generator=gen,
        drivetrain=edrv,
        save_interval=save_interval,
        loco_params=alt.LocoParams(
            pwr_aux_offset_watts=pwr_aux_offset_watts,
            pwr_aux_traction_coeff_ratio=pwr_aux_traction_coeff,
            force_max_newtons=force_max_newtons,
        )
    )
    return loco_unit


def mock_hybrid_loco(
    fc: Optional[alt.FuelConverter] = None,
    gen: Optional[alt.Generator] = None,
    res: Optional[alt.ReversibleEnergyStorage] = None,
    edrv: Optional[alt.ElectricDrivetrain] = None,
    save_interval: Optional[int] = 1,
    fuel_res_split: float = 0.5,
    fuel_res_ratio: float = 3.0,
    gss_interval: int = 60,
    pwr_aux_offset_watts: float = 0.0,
    pwr_aux_traction_coeff: float = 0.0,
    force_max_newtons: float = 150000.0,
) -> alt.Locomotive:
    if not fc:
        fc = mock_fuel_converter(save_interval)
    if not gen:
        gen = mock_generator(save_interval)
    if not edrv:
        edrv = mock_electric_drivetrain(save_interval)
    if not res:
        res = mock_reversible_energy_storage(save_interval)

    loco_unit = alt.Locomotive.build_hybrid_loco(
        fuel_converter=fc,
        generator=gen,
        reversible_energy_storage=res,
        drivetrain=edrv,
        save_interval=save_interval,
        fuel_res_split=fuel_res_split,
        fuel_res_ratio=fuel_res_ratio,
        gss_interval=gss_interval,
        loco_params=alt.LocoParams(
            pwr_aux_offset_watts=pwr_aux_offset_watts,
            pwr_aux_traction_coeff_ratio=pwr_aux_traction_coeff,
            force_max_newtons=force_max_newtons,
        )
    )
    return loco_unit


def mock_battery_electric_locomotive(
    res: Optional[alt.ReversibleEnergyStorage] = None,
    edrv: Optional[alt.ElectricDrivetrain] = None,
    save_interval: Optional[int] = 1,
    pwr_aux_offset_watts: float = 0.0,
    pwr_aux_traction_coeff: float = 0.0,
    force_max_newtons: float = 150000.0,
) -> alt.Locomotive:
    if not edrv:
        edrv = mock_electric_drivetrain(save_interval)
    if not res:
        res = mock_reversible_energy_storage(save_interval)
    loco_unit = alt.Locomotive.build_battery_electric_loco(
        reversible_energy_storage=res,
        drivetrain=edrv,
        save_interval=save_interval,
        loco_params=alt.LocoParams(
            pwr_aux_offset_watts=pwr_aux_offset_watts,
            pwr_aux_traction_coeff_ratio=pwr_aux_traction_coeff,
            force_max_newtons=force_max_newtons,
        )
    )
    return loco_unit


def mock_consist(save_interval: Optional[int] = 1) -> alt.Consist:
    consist = alt.Consist.default()
    if save_interval:
        consist.set_save_interval(save_interval)
    return consist


def mock_speed_trace() -> alt.SpeedTrace:
    st = alt.SpeedTrace.default()
    return st


def mock_power_trace() -> alt.PowerTrace:
    pt = alt.PowerTrace.default()
    return pt


def mock_locomotive_simulation(
    loco: Optional[alt.Locomotive] = None,
    pt: alt.PowerTrace = mock_power_trace(),
    save_interval: Optional[int] = 1,
) -> alt.LocomotiveSimulation:
    if not loco:
        loco = mock_conventional_loco(save_interval=save_interval)
    sim = alt.LocomotiveSimulation(loco, pt, save_interval)
    return sim


def mock_consist_simulation(
    consist: Optional[alt.Consist] = None,
    pt: alt.PowerTrace = mock_power_trace(),
    save_interval: Optional[int] = 1,
) -> alt.ConsistSimulation:
    if not consist:
        consist = mock_consist(save_interval=save_interval)
    sim = alt.ConsistSimulation(consist, pt, save_interval)
    return sim


def mock_train_state() -> alt.TrainState:
    train_length = 1_666
    axle_inertia_kg = 600.
    mass_static_kg = 8_000e3
    n_empty_cars = 40
    n_loaded_cars = 60
    n_railcars = n_empty_cars + n_loaded_cars

    return alt.TrainState(
        length_meters=train_length,
        mass_static_kilograms=mass_static_kg,
        mass_adj_kilograms=mass_static_kg + axle_inertia_kg * 4 * n_railcars,
        mass_freight_kilograms=mass_static_kg * 0.6,
        init_train_state=None,
    )


def mock_set_speed_train_simulation(
    consist: Optional[alt.Consist] = None,
    # path_tpc_file: Optional[int] = None,  # `None` triggers default
    st: alt.SpeedTrace = mock_speed_trace(),
    save_interval: Optional[int] = 1,
) -> alt.SetSpeedTrainSim:
    if not consist:
        consist = mock_consist(save_interval=save_interval)

    sim = alt.SetSpeedTrainSim(
        loco_con=consist,
        state=mock_train_state(),
        train_res_file=None,
        path_tpc_file=None,
        speed_trace=st,
        save_interval=save_interval,
    )

    if save_interval:
        sim.set_save_interval(save_interval)
    return sim

# def mock_speed_limit_train_simulation(
#     consist: Optional[alt.Consist] = None,
#     # path_tpc_file: Optional[int] = None,  # `None` triggers default
#     save_interval: Optional[int] = 1,
# ) -> alt.SpeedLimitTrainSim:
#     if not consist:
#         consist = mock_consist(save_interval=save_interval)

#     sim = alt.SpeedLimitTrainSim(
#         loco_con=consist,
#         state=mock_train_state(),
#         train_res_file=None,
#         path_tpc_file=None,
#         save_interval=save_interval,
#     )

#     if save_interval:
#         sim.set_save_interval(save_interval)
#     return sim


def mock_speed_limit_train_simulation_vector(
    scenario_year: Optional[int] = 2020,
    simulation_days: Optional[int] = 7
) -> alt.SpeedLimitTrainSimVec:
    mock_sim = alt.SpeedLimitTrainSim.default()
    alt.set_param_from_path(mock_sim, "simulation_days", simulation_days)
    alt.set_param_from_path(mock_sim, "scenario_year", scenario_year)
    sim_vec = alt.SpeedLimitTrainSimVec([mock_sim])
    return sim_vec


def mock_pymoo_conv_cal_df():
    import pandas as pd
    CUR_FUEL_LHV_J__KG = 43e6
    loco_sim = alt.LocomotiveSimulation(power_trace=alt.PowerTrace.default(),
                                         loco_unit=alt.Locomotive.default(),
                                         save_interval=1)
    loco_sim.walk()
    df = pd.DataFrame({'time [s]': loco_sim.power_trace.time_seconds.tolist(),
                      'Tractive Power [W]': loco_sim.power_trace.pwr_watts.tolist(),
                       'Fuel Power [W]': (loco_sim.loco_unit.fc.history.pwr_fuel_watts).tolist(),
                       })
    df['engine_on'] = df['Tractive Power [W]'] > 0
    return df
