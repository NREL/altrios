//! Crate that wraps `altrios-core` and enables the `pyo3` feature to
//! expose most structs, methods, and functions to Python.

use altrios_core::prelude::*;
use altrios_core::track::SpeedSet;
pub use pyo3::exceptions::{
    PyAttributeError, PyFileNotFoundError, PyIndexError, PyNotImplementedError, PyRuntimeError,
};
pub use pyo3::prelude::*;
pub use pyo3::types::PyType;
pub use pyo3_polars::PyDataFrame;

#[pymodule]
fn altrios_pyo3(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<FuelConverter>()?;
    m.add_class::<FuelConverterState>()?;
    m.add_class::<FuelConverterStateHistoryVec>()?;
    m.add_class::<Generator>()?;
    m.add_class::<GeneratorState>()?;
    m.add_class::<GeneratorStateHistoryVec>()?;
    m.add_class::<ReversibleEnergyStorage>()?;
    m.add_class::<ReversibleEnergyStorageState>()?;
    m.add_class::<ReversibleEnergyStorageStateHistoryVec>()?;
    m.add_class::<ElectricDrivetrain>()?;
    m.add_class::<ElectricDrivetrainState>()?;
    m.add_class::<ElectricDrivetrainStateHistoryVec>()?;

    m.add_class::<Locomotive>()?;
    m.add_class::<LocoParams>()?;
    m.add_class::<ConventionalLoco>()?;
    m.add_class::<HybridLoco>()?;
    m.add_class::<BatteryElectricLoco>()?;
    m.add_class::<DummyLoco>()?;
    m.add_class::<LocomotiveState>()?;
    m.add_class::<LocomotiveStateHistoryVec>()?;
    m.add_class::<LocomotiveSimulation>()?;
    m.add_class::<PowerTrace>()?;

    m.add_class::<Consist>()?;
    m.add_class::<ConsistState>()?;
    m.add_class::<ConsistStateHistoryVec>()?;
    m.add_class::<ConsistSimulation>()?;

    m.add_class::<SpeedTrace>()?;
    m.add_class::<SetSpeedTrainSim>()?;
    m.add_class::<SpeedLimitTrainSim>()?;
    m.add_class::<LinkIdx>()?;
    m.add_class::<LinkIdxTime>()?;
    m.add_class::<TimedLinkPath>()?;
    m.add_class::<LinkPoint>()?;
    m.add_class::<Link>()?;
    m.add_class::<Elev>()?;
    m.add_class::<Heading>()?;
    m.add_class::<Location>()?;
    m.add_class::<Network>()?;
    m.add_class::<LinkPath>()?;
    m.add_class::<SpeedSet>()?;

    m.add_class::<InitTrainState>()?;
    m.add_class::<TrainState>()?;
    m.add_class::<TrainStateHistoryVec>()?;

    m.add_class::<TrainConfig>()?;
    m.add_class::<TrainType>()?;
    m.add_class::<TrainParams>()?;
    m.add_class::<RailVehicle>()?;
    m.add_class::<TrainSimBuilder>()?;
    m.add_class::<SpeedLimitTrainSimVec>()?;
    m.add_class::<EstTimeNet>()?;
    m.add_function(wrap_pyfunction!(import_rail_vehicles_py, m)?)?;
    m.add_function(wrap_pyfunction!(import_locations_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_est_times_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_dispatch_py, m)?)?;
    m.add_function(wrap_pyfunction!(check_od_pair_valid, m)?)?;
    m.add_function(wrap_pyfunction!(build_speed_limit_train_sims, m)?)?;
    m.add_function(wrap_pyfunction!(run_speed_limit_train_sims, m)?)?;

    m.add_class::<Pyo3VecWrapper>()?;
    m.add_class::<Pyo3Vec2Wrapper>()?;
    m.add_class::<Pyo3Vec3Wrapper>()?;
    m.add_class::<Pyo3VecBoolWrapper>()?;

    Ok(())
}
