from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import polars as pl

import altrios.altrios_pyo3 as altpy

# Add a custom list class with tolist method
class ListWithTolist(list):
    def tolist(self) -> list: ...

class SerdeAPI:
    @classmethod
    def from_bincode(cls, bincode: bytes, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_json(cls, json_str: str, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_yaml(cls, yaml_str: str, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_file(cls, filepath: str | Path, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_str(cls, contents: str, fmt: str, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_toml(cls, toml_str: str, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_msg_pack(cls, msg_pack: bytes, skip_init: bool = False) -> Self: ...
    @classmethod
    def from_resource(cls, filepath: str | Path, skip_init: bool = False) -> Self: ...
    def to_file(self, filepath: str | Path) -> None: ...
    def to_bincode(self) -> bytes: ...
    def to_json(self) -> str: ...
    def to_yaml(self) -> str: ...
    def to_toml(self) -> str: ...
    def to_msg_pack(self) -> bytes: ...
    def to_str(self, fmt: str) -> str: ...
    def to_pydict(self, data_fmt: str = "msg_pack", flatten: bool = False) -> dict: ...
    @classmethod
    def from_pydict(
        cls,
        pydict: dict,
        data_fmt: str = "msg_pack",
        skip_init: bool = False,
    ) -> Self: ...
    def clone(self) -> Self: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: dict) -> Self: ...

class Consist(SerdeAPI):
    def __init__(self, loco_vec: list[Locomotive], save_interval: int | None = None): ...
    assert_limits: bool
    history: ConsistStateHistoryVec
    loco_vec: list[Locomotive]
    save_interval: int
    state: ConsistState
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def get_save_interval(self) -> int: ...
    def set_pdct_prop(self) -> None: ...
    def set_pdct_resgreedy(self) -> None: ...
    def set_save_interval(self, save_interval: int) -> None: ...
    def __copy__(self) -> Self: ...

class ConsistSimulation(SerdeAPI):
    i: int
    loco_con: Consist
    power_trace: PowerTrace
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def get_save_interval(self) -> int: ...
    def set_save_interval(self, save_interval: int): ...
    def walk(self) -> None: ...
    def __copy__(self) -> Self: ...

class ConsistState(SerdeAPI):
    energy_fuel_joules: float
    energy_out_joules: float
    energy_res_joules: float
    i: int
    pwr_dyn_brake_max_watts: float
    pwr_fuel_watts: float
    pwr_out_max_non_reves_watts: float
    pwr_out_max_reves_watts: float
    pwr_out_max_watts: float
    pwr_out_req_watts: float
    pwr_out_unfulfilled_watts: float
    pwr_out_watts: float
    pwr_rate_out_max_watts_per_second: float
    pwr_regen_max_watts: float
    pwr_regen_unfulfilled_watts: float
    pwr_reves_watts: float
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class ConsistStateHistoryVec(SerdeAPI):
    energy_fuel_joules: list[float]
    energy_out_joules: list[float]
    energy_res_joules: list[float]
    i: list[int]
    pwr_dyn_brake_max_watts: list[float]
    pwr_fuel_watts: list[float]
    pwr_out_max_non_reves_watts: list[float]
    pwr_out_max_reves_watts: list[float]
    pwr_out_max_watts: list[float]
    pwr_out_req_watts: list[float]
    pwr_out_unfulfilled_watts: list[float]
    pwr_out_watts: list[float]
    pwr_rate_out_max_watts_per_second: list[float]
    pwr_regen_max_watts: list[float]
    pwr_regen_unfulfilled_watts: list[float]
    pwr_reves_watts: list[float]
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...

class ElectricDrivetrain(SerdeAPI):
    eta_interp: list[float]
    history: ElectricDrivetrainStateHistoryVec
    pwr_in_frac_interp: list[float]
    pwr_out_frac_interp: list[float]
    pwr_out_max_watts: float
    save_interval: int | None
    state: ElectricDrivetrainState

    @classmethod
    def __init__(
        cls,
        pwr_out_frac_interp: list[float] = None,
        eta_interp: list[float] = None,
        pwr_out_max_watts: float = None,
        save_interval: int | None = None,
    ) -> None: ...

    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class ElectricDrivetrainState(SerdeAPI):
    energy_elec_dyn_brake_joules: float
    energy_elec_prop_in_joules: float
    energy_loss_joules: float
    energy_mech_dyn_brake_joules: float
    energy_mech_prop_out_joules: float
    eta: float
    i: int
    pwr_elec_dyn_brake_watts: float
    pwr_elec_prop_in_watts: float
    pwr_loss_watts: float
    pwr_mech_dyn_brake_watts: float
    pwr_mech_out_max_watts: float
    pwr_mech_prop_out_watts: float
    pwr_mech_regen_max_watts: float
    pwr_rate_out_max_watts_per_second: float
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class ElectricDrivetrainStateHistoryVec(SerdeAPI):
    energy_elec_dyn_brake_joules: list[float]
    energy_elec_prop_in_joules: list[float]
    energy_loss_joules: list[float]
    energy_mech_dyn_brake_joules: list[float]
    energy_mech_prop_out_joules: list[float]
    eta: list[float]
    i: list[int]
    pwr_elec_dyn_brake_watts: list[float]
    pwr_elec_prop_in_watts: list[float]
    pwr_loss_watts: list[float]
    pwr_mech_dyn_brake_watts: list[float]
    pwr_mech_out_max_watts: list[float]
    pwr_mech_prop_out_watts: list[float]
    pwr_mech_regen_max_watts: list[float]
    pwr_rate_out_max_watts_per_second: list[float]
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...

class FuelConverter(SerdeAPI):
    eta_interp: list[float]
    eta_max: float
    eta_range: float
    history: FuelConverterStateHistoryVec
    pwr_idle_fuel_watts: float
    pwr_out_frac_interp: list[float]
    pwr_out_max_watts: float
    pwr_ramp_lag_seconds: float
    save_interval: int | None
    state: FuelConverterState
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class FuelConverterState(SerdeAPI):
    energy_brake_joules: float
    energy_fuel_joules: float
    energy_idle_fuel_joules: float
    energy_loss_joules: float
    engine_on: bool
    eta: float
    i: int
    pwr_fuel_watts: float
    pwr_brake_watts: float
    pwr_idle_fuel_watts: float
    pwr_loss_watts: float
    pwr_out_max_watts: float
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class FuelConverterStateHistoryVec(SerdeAPI):
    energy_brake_joules: list[float]
    energy_fuel_joules: list[float]
    energy_idle_fuel_joules: list[float]
    energy_loss_joules: list[float]
    engine_on: list[bool]
    eta: list[float]
    i: list[int]
    pwr_brake_watts: list[float]
    pwr_fuel_watts: list[float]
    pwr_idle_fuel_watts: list[float]
    pwr_loss_watts: list[float]
    pwr_out_max_watts: list[float]
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...

class Generator(SerdeAPI):
    eta_interp: list[float]
    history: GeneratorStateHistoryVec
    pwr_in_frac_interp: list[float]
    pwr_out_frac_interp: list[float]
    pwr_out_max_watts: float
    save_interval: int | None
    state: GeneratorState
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class GeneratorState(SerdeAPI):
    energy_elec_aux_joules: float
    energy_elec_prop_out_joules: float
    energy_loss_joules: float
    energy_mech_in_joules: float
    eta: float
    i: int
    pwr_elec_aux_watts: float
    pwr_elec_out_max_watts: float
    pwr_elec_prop_out_watts: float
    pwr_loss_watts: float
    pwr_mech_in_watts: float
    pwr_rate_out_max_watts_per_second: float
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class GeneratorStateHistoryVec(SerdeAPI):
    energy_elec_aux_joules: list[float]
    energy_elec_prop_out_joules: list[float]
    energy_loss_joules: list[float]
    energy_mech_in_joules: list[float]
    eta: list[float]
    i: list[int]
    pwr_elec_aux_watts: list[float]
    pwr_elec_out_max_watts: list[float]
    pwr_elec_prop_out_watts: list[float]
    pwr_loss_watts: list[float]
    pwr_mech_in_watts: list[float]
    pwr_rate_out_max_watts_per_second: list[float]
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...

@dataclass
class LocoParams(SerdeAPI):
    pwr_aux_offset_watts: float
    pwr_aux_traction_coeff_ratio: float
    force_max_newtons: float
    mass_kilograms: float | None = 0.0
    save_interval: int | None

    @classmethod
    def default(cls) -> Self: ...

@dataclass
class ConventionalLoco(SerdeAPI):
    fc: FuelConverter
    gen: Generator
    edrv: ElectricDrivetrain

@dataclass
class HybridLoco(SerdeAPI):
    fuel_converter: FuelConverter
    generator: Generator
    reversible_energy_storage: ReversibleEnergyStorage
    electric_drivetrain: ElectricDrivetrain

@dataclass
class BatteryElectricLoco(SerdeAPI):
    res: ReversibleEnergyStorage
    edrv: ElectricDrivetrain

@dataclass
class DummyLoco(SerdeAPI): ...

class Locomotive(SerdeAPI):
    assert_limits: bool
    edrv: ElectricDrivetrain
    fc: FuelConverter
    fuel_res_ratio: float
    fuel_res_split: float
    gen: Generator
    history: LocomotiveStateHistoryVec
    pwr_aux_watts: float
    res: ReversibleEnergyStorage
    save_interval: int
    state: LocomotiveState

    @classmethod
    def __new__(
        cls,
        loco_type: ConventionalLoco | HybridLoco | BatteryElectricLoco | DummyLoco,
        loco_params: LocoParams,
    ): ...
    @classmethod
    def default_battery_electric_loco(cls) -> Locomotive: ...
    @classmethod
    def build_conventional_loco(
        cls,
        fuel_converter: FuelConverter,
        generator: Generator,
        drivetrain: ElectricDrivetrain,
        loco_params: LocoParams,
        save_interval: int | None,
    ) -> Self: ...
    @classmethod
    def build_dummy_loco(cls) -> Self: ...
    @classmethod
    def default_hybrid_electric_loco(cls) -> Self: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def get_save_interval(self) -> Any: ...
    def set_save_interval(self, save_interval: int): ...
    def __copy__(self) -> Self: ...

class LocomotiveSimulation(SerdeAPI):
    i: int
    loco_unit: Locomotive
    power_trace: PowerTrace
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    def get_save_interval(self) -> int: ...
    def set_save_interval(self, save_interval: int): ...
    def walk(self) -> None: ...
    def __copy__(self) -> Self: ...

class LocomotiveState(SerdeAPI):
    energy_aux_joules: float
    energy_out_joules: float
    i: int
    pwr_aux_watts: float
    pwr_out_max_watts: float
    pwr_out_watts: float
    pwr_rate_out_max_watts_per_second: float
    pwr_regen_max_watts: float
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class LocomotiveStateHistoryVec(SerdeAPI):
    energy_aux_joules: list[float]
    energy_out_joules: list[float]
    i: list[int]
    pwr_aux_watts: list[float]
    pwr_out_max_watts: list[float]
    pwr_out_watts: list[float]
    pwr_rate_out_max_watts_per_second: list[float]
    pwr_regen_max_watts: list[float]
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...

# Add methods to properly support list attributes with tolist methods
# This is needed to handle usages like train_sim.history.time_seconds.tolist()
class ListWithTolist(list):
    def tolist(self) -> list: ...

# Update TypedDict fields to treat each list as having a tolist method
class TrainStateHistoryVec(SerdeAPI):
    time_seconds: ListWithTolist
    offset_meters: ListWithTolist
    offset_back_meters: ListWithTolist
    link_idx_front: ListWithTolist
    offset_in_link_meters: ListWithTolist
    grade_front: ListWithTolist
    speed_meters_per_second: ListWithTolist
    speed_limit_meters_per_second: ListWithTolist
    speed_target_meters_per_second: ListWithTolist
    dt_seconds: ListWithTolist
    length_meters: ListWithTolist
    mass_static_kilograms: ListWithTolist
    mass_adj_kilograms: ListWithTolist
    mass_freight_kilograms: ListWithTolist
    max_fric_braking: ListWithTolist
    weight_static_newtons: ListWithTolist
    res_rolling_newtons: ListWithTolist
    res_bearing_newtons: ListWithTolist
    res_davis_b_newtons: ListWithTolist
    res_aero_newtons: ListWithTolist
    res_grade_newtons: ListWithTolist
    res_curve_newtons: ListWithTolist
    pwr_whl_out_watts: ListWithTolist
    energy_whl_out_joules: ListWithTolist
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...
    def tolist(self) -> list[float]: ...

@dataclass
class PowerTrace(SerdeAPI):
    time_seconds: list[float]
    pwr_watts: list[float]
    engine_on: list[bool] | None
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...
    @classmethod
    def from_csv_file(cls, pathstr: str) -> Self: ...

class ReversibleEnergyStorage(SerdeAPI):
    energy_capacity_joules: float
    eta_interp_values: list[list[list[float]]]
    history: ReversibleEnergyStorageStateHistoryVec
    max_soc: float
    min_soc: float
    pwr_out_max_watts: float
    save_interval: int | None
    soc_hi_ramp_start: float | None
    soc_lo_ramp_start: float | None
    state: ReversibleEnergyStorageState
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class ReversibleEnergyStorageState(SerdeAPI):
    energy_aux_joules: float
    energy_loss_joules: float
    energy_out_chemical_joules: float
    energy_out_electrical_joules: float
    energy_out_propulsion_joules: float
    eta: float
    i: int
    max_soc: float
    min_soc: float
    pwr_aux_watts: float
    pwr_loss_watts: float
    pwr_out_chemical_watts: float
    pwr_out_electrical_watts: float
    pwr_out_max_watts: float
    pwr_out_propulsion_watts: float
    pwr_regen_max_watts: float
    soc: float
    soc_hi_ramp_start: float
    soc_lo_ramp_start: float
    soh: float
    temperature_celsius: float
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...

class ReversibleEnergyStorageStateHistoryVec(SerdeAPI):
    energy_aux_joules: list[float]
    energy_loss_joules: list[float]
    energy_out_chemical_joules: list[float]
    energy_out_electrical_joules: list[float]
    energy_out_propulsion_joules: list[float]
    eta: list[float]
    i: list[int]
    max_soc: list[float]
    min_soc: list[float]
    pwr_aux_watts: list[float]
    pwr_loss_watts: list[float]
    pwr_out_chemical_watts: list[float]
    pwr_out_electrical_watts: list[float]
    pwr_out_max_watts: list[float]
    pwr_out_propulsion_watts: list[float]
    pwr_regen_max_watts: list[float]
    soc: list[float]
    soc_hi_ramp_start: list[float]
    soc_lo_ramp_start: list[float]
    soh: list[float]
    temperature_celsius: list[float]
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...

@dataclass
class SpeedTrace(SerdeAPI):
    time_seconds: list[float]
    speed_meters_per_second: list[float]
    engine_on: list[bool] | None = None

    def __init__(
        self,
        time_seconds: list[float],
        speed_meters_per_second: list[float],
        engine_on: list[bool] | None = None,
    ) -> None: ...

    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...
    @classmethod
    def from_csv_file(cls, pathstr: str) -> Self: ...
    def to_csv_file(self, pathstr: str | Path): ...

class TemperatureTraceBuilder(SerdeAPI):
    time: list[float]
    temp_at_sea_level: list[float]
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...
    @classmethod
    def from_csv_file(cls, pathstr: str) -> Self: ...

class TemperatureTrace(SerdeAPI):
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def get_temp_at_time_and_elev(self, time: float, elev: float) -> float: ...

class TrainState:
    time_seconds: float
    i: int
    offset_meters: float
    offset_back_meters: float
    total_dist_meters: float
    link_idx_front: int
    offset_in_link_meters: float
    grade_front: float
    speed_meters_per_second: float
    speed_limit_meters_per_second: float
    speed_target_meters_per_second: float
    dt_seconds: float
    length_meters: float
    mass_static_kilograms: float
    mass_rot_kilograms: float
    mass_freight_kilograms: float
    weight_static_newtons: float
    res_rolling_newtons: float
    res_bearing_newtons: float
    res_davis_b_newtons: float
    res_aero_newtons: float
    res_grade_newtons: float
    res_curve_newtons: float
    elev_front_meters: float
    pwr_res_watts: float
    pwr_accel_watts: float
    pwr_whl_out_watts: float
    energy_whl_out_joules: float
    energy_whl_out_pos_joules: float
    energy_whl_out_neg_joules: float
    @classmethod
    def default(cls) -> TrainState: ...
    @classmethod
    def from_json(cls, json_str: str) -> TrainState: ...
    @classmethod
    def __new__(
        cls,
        length_meters: float,
        mass_static_kilograms: float,
        mass_adj_kilograms: float,
        mass_freight_kilograms: float,
        init_train_state: InitTrainState | None,
    ) -> Self: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_yaml(cls, yaml_str: str) -> TrainState: ...
    def to_yaml(self) -> str: ...
    def clone(self) -> TrainState: ...
    def reset_orphaned(self) -> None: ...

class TrainStateHistoryVec(SerdeAPI):
    time_seconds: ListWithTolist
    offset_meters: ListWithTolist
    offset_back_meters: ListWithTolist
    link_idx_front: ListWithTolist
    offset_in_link_meters: ListWithTolist
    grade_front: ListWithTolist
    speed_meters_per_second: ListWithTolist
    speed_limit_meters_per_second: ListWithTolist
    speed_target_meters_per_second: ListWithTolist
    dt_seconds: ListWithTolist
    length_meters: ListWithTolist
    mass_static_kilograms: ListWithTolist
    mass_adj_kilograms: ListWithTolist
    mass_freight_kilograms: ListWithTolist
    max_fric_braking: ListWithTolist
    weight_static_newtons: ListWithTolist
    res_rolling_newtons: ListWithTolist
    res_bearing_newtons: ListWithTolist
    res_davis_b_newtons: ListWithTolist
    res_aero_newtons: ListWithTolist
    res_grade_newtons: ListWithTolist
    res_curve_newtons: ListWithTolist
    pwr_whl_out_watts: ListWithTolist
    energy_whl_out_joules: ListWithTolist
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def __len__(self) -> int: ...
    def tolist(self) -> list[float]: ...

@dataclass
class SetSpeedTrainSim(SerdeAPI):
    loco_con: Consist
    state: TrainState
    speed_trace: SpeedTrace
    history: TrainStateHistoryVec
    i: int
    save_interval: int | None

    @classmethod
    def __init__(
        cls,
        loco_con: Consist,
        state: TrainState,
        train_res_file: str | None,
        path_tpc_file: str | None,
        speed_trace: SpeedTrace,
        save_interval: int | None,
        simulation_days: int | None,
    ) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def set_save_interval(self, save_interval: int): ...

class SpeedLimitTrainSim(SerdeAPI):
    train_id: str
    origs: list[Location]
    dests: list[Location]
    loco_con: Consist
    n_cars_by_type: dict[str, int]
    state: TrainState
    # train_res: TrainRes # not accessible in Python
    path_tpc: PathTpc
    braking_points: BrakingPoints
    fric_brake: FricBrake
    history: TrainStateHistoryVec
    save_interval: int | None
    simulation_days: int | None
    scenario_year: int | None

    @classmethod
    def __init__(
        cls,
        loco_con: Consist,
        n_cars_by_type: dict[str, int],
        state: TrainState,
        train_res_file: str | None = None,
        path_tpc_file: str | None = None,
        save_interval: int | None = None,
        simulation_days: int | None = None,
        scenario_year: int | None = None,
    ) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def __copy__(self) -> Self: ...
    def set_save_interval(self, save_interval: int): ...
    def walk(self): ...
    def walk_timed_path(self, network: Network, timed_path: list[LinkIdxTime] | TimedLinkPath): ...
    def get_energy_fuel_joules(self, include_idle: bool = True) -> float: ...
    def get_energy_fuel_soc_corrected_joules(self) -> float: ...
    def to_dataframe(self) -> pl.DataFrame: ...

@dataclass
class SpeedLimitTrainSimVec(SerdeAPI):
    @classmethod
    def __new__(cls, v: list[SpeedLimitTrainSim] | None = None) -> Self: ...
    @classmethod
    def __init__(cls, v: list[SpeedLimitTrainSim] | None = None) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def is_empty(self) -> bool: ...
    def tolist(self) -> list[SpeedLimitTrainSim]: ...
    def __copy__(self) -> Self: ...
    def __delitem__(self, other) -> None: ...
    def __getitem__(self, index) -> SpeedLimitTrainSim: ...
    def __len__(self) -> int: ...
    def __setitem__(self, index, value) -> None: ...
    def get_energy_fuel_joules(
        self, include_idle: bool = True, annualize: bool = False,
    ) -> float: ...
    def get_energy_fuel_soc_corrected_joules(
        self, annualize: bool = False,
    ) -> float: ...

@dataclass
class LinkIdx(SerdeAPI):
    idx: int
    @classmethod
    def default(cls) -> Self: ...

class LinkIdxTime(SerdeAPI):
    link_idx: LinkIdx
    time_seconds: float
    @classmethod
    def default(cls) -> Self: ...

class TrainSimBuilder(SerdeAPI):
    train_id: str
    origin_id: str
    destination_id: str
    train_config: TrainConfig
    loco_con: Consist
    init_train_state: InitTrainState | None
    drag_coeff_vec: list[float] | None
    @classmethod
    def default(cls) -> Self: ...
    def __init__(
        self,
        train_id: str,
        origin_id: str,
        destination_id: str,
        train_config: TrainConfig,
        loco_con: Consist,
        init_train_state: InitTrainState | None = None,
    ) -> None: ...
    def make_set_speed_train_sim(
        self,
        rail_vehicles: list[RailVehicle],
        network: list[Link],
        link_path: list[LinkIdx],
        speed_trace: SpeedTrace,
        save_interval: int | None,
        temp_trace: TemperatureTrace | None = None,
    ) -> SetSpeedTrainSim: ...
    def make_speed_limit_train_sim(
        self,
        rail_vehicles: list[RailVehicle] | None = None,
        location_map: dict[str, list[Location]] | None = None,
        save_interval: int | None = None,
        simulation_days: int | None = None,
        scenario_year: int | None = None,
        temp_trace: TemperatureTrace | None = None,
    ) -> SpeedLimitTrainSim: ...

@dataclass
class TrainConfig(SerdeAPI):
    n_cars_by_type: dict[str, int]
    rail_vehicle_type: str | None
    train_type: TrainType | None
    train_length_meters: float | None
    train_mass_kilograms: float | None
    cd_area_vec: list[float] | None
    rail_vehicles: list[RailVehicle] | None

    @classmethod
    def __init__(
        cls,
        rail_vehicles: list[RailVehicle] | None = None,
        n_cars_by_type: dict[str, int] | None = None,
        train_length_meters: float | None = None,
        train_mass_kilograms: float | None = None,
    ) -> None: ...

    @classmethod
    def default(cls) -> Self: ...

class RailVehicle(SerdeAPI):
    axle_count: int
    bearing_res_per_axle_newtons: float
    brake_count: int
    braking_ratio_empty: float
    braking_ratio_loaded: float
    car_type: str
    freight_type: str
    davis_b_seconds_per_meter: float
    cd_area_empty_square_meters: float
    cd_area_loaded_square_meters: float
    length_meters: float
    mass_rot_per_axle_kilograms: float
    mass_static_empty_kilograms: float
    mass_static_loaded_kilograms: float
    rolling_ratio: float
    speed_max_empty_meters_per_second: float
    speed_max_loaded_meters_per_second: float
    @classmethod
    def default(cls) -> Self: ...

class Location(SerdeAPI):
    location_id: str
    offset: float
    link_idx: LinkIdx
    is_front_end: bool
    grid_emissions_region: str
    electricity_price_region: str
    liquid_fuel_price_region: str
    @classmethod
    def default(cls) -> Self: ...

class EstTimeNet(SerdeAPI):
    @classmethod
    def default(cls) -> Self: ...
    def get_running_time_hours(self) -> float: ...

class Link(SerdeAPI):
    length_meters: float
    @classmethod
    def default(cls) -> Self: ...

class Elev(SerdeAPI):
    offset_meters: float
    elev_meters: float
    @classmethod
    def default(cls) -> Self: ...

class Heading(SerdeAPI):
    offset_meters: float
    heading: float
    lat: float | None
    lon: float | None
    @classmethod
    def default(cls) -> Self: ...

class SpeedSet(SerdeAPI):
    ...
    # TODO: finish fleshing this out

def import_locations(filename: str | Path) -> dict[str, list[Location]]: ...
def build_speed_limit_train_sims(
    train_sim_builders: list[TrainSimBuilder],
    rail_veh_map: dict[str, RailVehicle],
    location_map: dict[str, list[Location]],
    save_interval: int | None,
    simulation_days: int | None,
    scenario_year: int | None,
) -> SpeedLimitTrainSimVec: ...
def run_speed_limit_train_sims(
    speed_limit_train_sim_vec: SpeedLimitTrainSimVec,
    network: list[Link],
    train_consist_plan: pl.DataFrame,
    loco_pool: pl.DataFrame,
    refuel_facilities: pl.DataFrame,
    timed_paths: list[list[LinkIdxTime]],
) -> tuple[SpeedLimitTrainSimVec, pl.DataFrame]: ...
def run_dispatch(
    network: Network,
    speed_limit_train_sims: SpeedLimitTrainSimVec,
    est_time_nets: list[EstTimeNet],
    print_train_move: bool,
    print_train_exit: bool,
) -> list[TimedLinkPath]: ...
def make_est_times(
    speed_limit_train_sim: SpeedLimitTrainSim,
    network: Network,
    path_for_failed_sim: Path | None = None,
) -> tuple[EstTimeNet, Consist]: ...
@dataclass
class TimedLinkPath(SerdeAPI):
    @classmethod
    def __new__(cls, v: list[LinkIdxTime]) -> Self: ...
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def is_empty(self) -> bool: ...
    def tolist(self) -> list[LinkIdxTime]: ...
    def __copy__(self) -> Self: ...
    def __delitem__(self, other) -> None: ...
    def __getitem__(self, index) -> LinkIdxTime: ...
    def __len__(self) -> int: ...
    def __setitem__(self, index, value) -> None: ...

@dataclass
class Network(SerdeAPI):
    @classmethod
    def __new__(cls, v: list[Link]) -> Self: ...
    @classmethod
    def __init__(cls) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def is_empty(self) -> bool: ...
    def tolist(self) -> list[Link]: ...
    def __copy__(self) -> Self: ...
    def __delitem__(self, other) -> None: ...
    def __getitem__(self, index) -> Link: ...
    def __len__(self) -> int: ...
    def __setitem__(self, index, value) -> None: ...
    def set_speed_set_for_train_type(self, train_type: TrainType): ...

@dataclass
class LinkPath(SerdeAPI):
    @classmethod
    def __new__(cls, v: list[LinkIdx] | None = None) -> Self: ...
    @classmethod
    def __init__(cls, v: list[LinkIdx] | None = None) -> None: ...
    def clone(self) -> Self: ...
    @classmethod
    def default(cls) -> Self: ...
    def is_empty(self) -> bool: ...
    def tolist(self) -> list[LinkIdx]: ...
    def __copy__(self) -> Self: ...
    def __delitem__(self, other) -> None: ...
    def __getitem__(self, index) -> LinkIdx: ...
    def __len__(self) -> int: ...
    def __setitem__(self, index, value) -> None: ...
    def to_csv_file(self, pathstr: str | Path) -> None: ...

@dataclass
class InitTrainState(SerdeAPI):
    time_seconds: float
    offset_meters: float
    speed_meters_per_second: float
    dt_seconds: float
    @classmethod
    def default(cls) -> Self: ...

@dataclass
class TrainType(SerdeAPI):
    Freight = (altpy.TrainType.Freight,)  # type: ignore[has-type]
    Passenger = (altpy.TrainType.Passenger,)  # type: ignore[has-type]
    Intermodal = (altpy.TrainType.Intermodal,)  # type: ignore[has-type]
    HighSpeedPassenger = (altpy.TrainType.HighSpeedPassenger,)  # type: ignore[has-type]
    TiltTrain = (altpy.TrainType.TiltTrain,)  # type: ignore[has-type]
    Commuter = (altpy.TrainType.Commuter,)  # type: ignore[has-type]

class PathTpc(SerdeAPI):
    @classmethod
    def default(cls) -> Self: ...

class BrakingPoints(SerdeAPI):
    @classmethod
    def default(cls) -> Self: ...

class FricBrake(SerdeAPI):
    @classmethod
    def default(cls) -> Self: ...
