use super::resistance::kind as res_kind;
use super::resistance::method as res_method;
use crate::consist::locomotive::locomotive_model::PowertrainType;
use crate::consist::Mass;

use super::{
    friction_brakes::*, rail_vehicle::RailVehicle, train_imports::*, InitTrainState, LinkIdxTime,
    SetSpeedTrainSim, SpeedLimitTrainSim, SpeedTrace, TrainState,
};
use crate::track::link::link_idx::LinkPath;
use crate::track::link::link_impl::Network;
use crate::track::LocationMap;

use polars::prelude::*;
use polars_lazy::dsl::max_horizontal;
#[allow(unused_imports)]
use polars_lazy::prelude::*;
use pyo3_polars::PyDataFrame;

#[altrios_api(
    #[new]
    fn __new__(
        n_cars_by_type: HashMap<String, u32>,
        rail_vehicle_type: Option<String>,
        train_type: Option<TrainType>,
        train_length_meters: Option<f64>,
        train_mass_kilograms: Option<f64>,
        drag_area_vec: Option<Vec<f64>>,
    ) -> anyhow::Result<Self> {
        Self::new(
            n_cars_by_type,
            rail_vehicle_type,
            train_type.unwrap_or_default(),
            train_length_meters.map(|v| v * uc::M),
            train_mass_kilograms.map(|v| v * uc::KG),
            drag_area_vec.map(|dcv| dcv.iter().map(|dc| *dc * uc::M2).collect())
        )
    }

    #[pyo3(name = "make_train_params")]
    /// - `rail_vehicles` - list of `RailVehicle` objects with 1 element for each _type_ of rail vehicle
    fn make_train_params_py(&self, rail_vehicles: Vec<RailVehicle>) -> anyhow::Result<TrainParams> {
        self.make_train_params(&rail_vehicles)
    }

    #[getter]
    fn get_train_length_meters(&self) -> Option<f64> {
        self.train_length.map(|l| l.get::<si::meter>())
    }

    #[setter]
    fn set_train_length_meters(&mut self, train_length: f64) -> anyhow::Result<()> {
        self.train_length = Some(train_length * uc::M);
        Ok(())
    }

    #[getter]
    fn get_train_mass_kilograms(&self) -> Option<f64> {
        self.train_mass.map(|l| l.get::<si::kilogram>())
    }

    #[setter]
    fn set_train_mass_kilograms(&mut self, train_mass: f64) -> anyhow::Result<()> {
        self.train_mass = Some(train_mass * uc::KG);
        Ok(())
    }

    #[getter]
    fn get_drag_coeff_vec_meters_squared(&self) -> Option<Vec<f64>> {
        self.drag_area_vec
            .as_ref()
                .map(
                    |dcv| dcv.iter().cloned().map(|x| x.get::<si::square_meter>()).collect()
                )
    }

    #[setter]
    fn set_drag_coeff_vec(&mut self, new_val: Vec<f64>) -> anyhow::Result<()> {
        self.drag_area_vec = Some(new_val.iter().map(|x| *x * uc::M2).collect());
        Ok(())
    }
)]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
/// User-defined train configuration used to generate
/// [crate::prelude::TrainParams]. Any optional fields will be populated later
/// in [TrainSimBuilder::make_train_sim_parts]
pub struct TrainConfig {
    /// Optional user-defined identifier for the car type on this train.
    pub rail_vehicle_type: Option<String>,
    /// Number of railcars by type on the train
    pub n_cars_by_type: HashMap<String, u32>,
    /// Train type matching one of the PTC types
    pub train_type: TrainType,
    /// Train length that overrides the railcar specific value
    #[api(skip_get, skip_set)]
    pub train_length: Option<si::Length>,
    /// Total train mass that overrides the railcar specific values
    #[api(skip_set, skip_get)]
    pub train_mass: Option<si::Mass>,
    #[api(skip_get, skip_set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    /// Optional vector of drag areas (i.e. drag coeff. times frontal area)
    /// for each car.  If provided, the total drag area (drag coefficient
    /// times frontal area) calculated from this vector is the sum of these
    /// coefficients.
    pub drag_area_vec: Option<Vec<si::Area>>,
}

impl SerdeAPI for TrainConfig {
    fn init(&mut self) -> anyhow::Result<()> {
        match &self.drag_area_vec {
            Some(dcv) => {
                ensure!(dcv.len() as u32 == self.cars_total());
            }
            None => {}
        };
        Ok(())
    }
}

impl TrainConfig {
    pub fn new(
        n_cars_by_type: HashMap<String, u32>,
        rail_vehicle_type: Option<String>,
        train_type: TrainType,
        train_length: Option<si::Length>,
        train_mass: Option<si::Mass>,
        drag_area_vec: Option<Vec<si::Area>>,
    ) -> anyhow::Result<Self> {
        let mut train_config = Self {
            n_cars_by_type,
            rail_vehicle_type,
            train_type,
            train_length,
            train_mass,
            drag_area_vec,
        };
        train_config.init()?;
        Ok(train_config)
    }

    pub fn cars_total(&self) -> u32 {
        self.n_cars_by_type.values().fold(0, |acc, n| *n + acc)
    }

    /// # Arguments
    /// - `rail_vehicles` - slice of `RailVehicle` objects with 1 element for each _type_ of rail vehicle
    pub fn make_train_params(&self, rail_vehicles: &[RailVehicle]) -> anyhow::Result<TrainParams> {
        // TODO: square this up with the mass_static calculation in
        let train_mass_static = self.train_mass.unwrap_or({
            rail_vehicles
                .iter()
                .try_fold(0. * uc::KG, |acc, rv| -> anyhow::Result<si::Mass> {
                    let mass = acc
                        + rv.mass_static_total()
                            * *self.n_cars_by_type.get(&rv.car_type).with_context(|| {
                                anyhow!(
                                    "{}\nExpected `self.n_cars_by_type` to contain '{}'",
                                    format_dbg!(),
                                    rv.car_type
                                )
                            })? as f64;
                    Ok(mass)
                })?
        });

        let length: si::Length = match self.train_length {
            Some(tl) => tl,
            None => rail_vehicles.iter().try_fold(
                0. * uc::M,
                |acc, rv| -> anyhow::Result<si::Length> {
                    Ok(acc
                        + rv.length
                            * *self.n_cars_by_type.get(&rv.car_type).with_context(|| {
                                anyhow!(
                                    "{}\nExpected `self.n_cars_by_type` to contain '{}'",
                                    format_dbg!(),
                                    rv.car_type
                                )
                            })? as f64)
                },
            )?,
        };

        let train_params = TrainParams {
            length,
            speed_max: rail_vehicles.iter().try_fold(
                f64::INFINITY * uc::MPS,
                |acc, rv| -> anyhow::Result<si::Velocity> {
                    Ok(
                        if *self.n_cars_by_type.get(&rv.car_type).with_context(|| {
                            anyhow!(
                                "{}\nExpected `self.n_cars_by_type` to contain '{}'",
                                format_dbg!(),
                                rv.car_type
                            )
                        })? > 0
                        {
                            acc.min(rv.speed_max)
                        } else {
                            acc
                        },
                    )
                },
            )?,
            mass_total: train_mass_static,
            // TODO: ask Tyler if `mass_per_brake` should include rotational mass
            mass_per_brake: train_mass_static
                / rail_vehicles
                    .iter()
                    .try_fold(0, |acc, rv| -> anyhow::Result<u32> {
                        let brake_count = acc
                            + rv.brake_count as u32
                                * *self.n_cars_by_type.get(&rv.car_type).with_context(|| {
                                    anyhow!(
                                        "{}\nExpected `self.n_cars_by_type` to contain '{}'",
                                        format_dbg!(),
                                        rv.car_type
                                    )
                                })?;
                        Ok(brake_count)
                    })? as f64,
            axle_count: rail_vehicles
                .iter()
                .try_fold(0, |acc, rv| -> anyhow::Result<u32> {
                    let axle_count = acc
                        + rv.axle_count as u32
                            * *self.n_cars_by_type.get(&rv.car_type).with_context(|| {
                                anyhow!(
                                    "{}\nExpected `self.n_cars_by_type` to contain '{}'",
                                    format_dbg!(),
                                    rv.car_type
                                )
                            })?;
                    Ok(axle_count)
                })?,
            train_type: self.train_type,
            // TODO: change it so that curve coefficient is specified at the train level, and replace `unwrap` function calls
            // with proper result handling, and relpace `first().unwrap()` with real code.
            curve_coeff_0: rail_vehicles.first().unwrap().curve_coeff_0,
            curve_coeff_1: rail_vehicles.first().unwrap().curve_coeff_1,
            curve_coeff_2: rail_vehicles.first().unwrap().curve_coeff_2,
        };
        Ok(train_params)
    }
}

impl Valid for TrainConfig {
    fn valid() -> Self {
        Self {
            rail_vehicle_type: Some("Bulk".to_string()),
            n_cars_by_type: HashMap::from([("loaded".into(), 100_u32)]),
            train_type: TrainType::Freight,
            train_length: None,
            train_mass: None,
            drag_area_vec: None,
        }
    }
}

#[altrios_api(
    #[new]
    fn __new__(
        train_id: String,
        train_config: TrainConfig,
        loco_con: Consist,
        origin_id: Option<String>,
        destination_id: Option<String>,
        init_train_state: Option<InitTrainState>,
    ) -> Self {
        Self::new(
            train_id,
            train_config,
            loco_con,
            origin_id,
            destination_id,
            init_train_state,
        )
    }

    #[pyo3(name = "make_set_speed_train_sim")]
    fn make_set_speed_train_sim_py(
        &self,
        rail_vehicles: Vec<RailVehicle>,
        network: &PyAny,
        link_path: &PyAny,
        speed_trace: SpeedTrace,
        save_interval: Option<usize>
    ) -> anyhow::Result<SetSpeedTrainSim> {
        let network = match network.extract::<Network>() {
            Ok(n) => n,
            Err(_) => {
                let n = network.extract::<Vec<Link>>().map_err(|_| anyhow!("{}", format_dbg!()))?;
                Network(n)
            }
        };

        let link_path = match link_path.extract::<LinkPath>() {
            Ok(lp) => lp,
            Err(_) => {
                let lp = link_path.extract::<Vec<LinkIdx>>().map_err(|_| anyhow!("{}", format_dbg!()))?;
                LinkPath(lp)
            }
        };

        self.make_set_speed_train_sim(
            &rail_vehicles,
            network,
            link_path,
            speed_trace,
            save_interval
        )
    }

    #[pyo3(name = "make_speed_limit_train_sim")]
    fn make_speed_limit_train_sim_py(
        &self,
        rail_vehicle: Vec<RailVehicle>,
        location_map: LocationMap,
        save_interval: Option<usize>,
        simulation_days: Option<i32>,
        scenario_year: Option<i32>,
    ) -> anyhow::Result<SpeedLimitTrainSim> {
        self.make_speed_limit_train_sim(
            &rail_vehicle,
            &location_map,
            save_interval,
            simulation_days,
            scenario_year,
        )
    }
)]
#[derive(Debug, Default, Clone, Deserialize, Serialize, PartialEq, SerdeAPI)]
pub struct TrainSimBuilder {
    /// Unique, user-defined identifier for the train
    pub train_id: String,
    pub train_config: TrainConfig,
    pub loco_con: Consist,
    /// Origin_ID from train planner to map to track network locations.  Only needed if
    /// [Self::make_speed_limit_train_sim] will be called.
    pub origin_id: Option<String>,
    /// Destination_ID from train planner to map to track network locations.  Only needed if
    /// [Self::make_speed_limit_train_sim] will be called.
    pub destination_id: Option<String>,
    #[api(skip_get, skip_set)]
    init_train_state: Option<InitTrainState>,
}

impl TrainSimBuilder {
    pub fn new(
        train_id: String,
        train_config: TrainConfig,
        loco_con: Consist,
        origin_id: Option<String>,
        destination_id: Option<String>,
        init_train_state: Option<InitTrainState>,
    ) -> Self {
        Self {
            train_id,
            train_config,
            loco_con,
            origin_id,
            destination_id,
            init_train_state,
        }
    }

    fn make_train_sim_parts(
        &self,
        rail_vehicles: &[RailVehicle],
        save_interval: Option<usize>,
    ) -> anyhow::Result<(TrainState, PathTpc, TrainRes, FricBrake)> {
        let rvs = rail_vehicles;
        let train_params = self.train_config.make_train_params(rail_vehicles)?;

        let length = train_params.length;
        // TODO: account for rotational mass of locomotive components (e.g. axles, gearboxes, motor shafts)
        let train_mass_static = train_params.mass_total
            + self
                .loco_con
                .mass()
                .with_context(|| format_dbg!())?
                .unwrap_or_else(|| {
                    log::warn!(
                        "Consist has no mass set so train dynamics don't include consist mass."
                    );
                    0. * uc::KG
                });
        let mass_adj = train_mass_static
            + rvs
                .iter()
                .try_fold(0. * uc::KG, |acc, rv| -> anyhow::Result<si::Mass> {
                    let mass = acc
                        + rv.mass_extra_per_axle
                            * *self
                                .train_config
                                .n_cars_by_type
                                .get(&rv.car_type)
                                .with_context(|| {
                                    anyhow!(
                                        "{}\nExpected `self.n_cars_by_type` to contain '{}'",
                                        format_dbg!(),
                                        rv.car_type
                                    )
                                })? as f64;
                    Ok(mass)
                })?
                * train_params.axle_count as f64;
        // TODO: this does not account for the unloaded mass of the railcar for loaded rail cars
        let mass_freight =
            rvs.iter()
                .try_fold(0. * uc::KG, |acc, rv| -> anyhow::Result<si::Mass> {
                    let mass = acc
                        + rv.mass_static_freight
                            * *self
                                .train_config
                                .n_cars_by_type
                                .get(&rv.car_type)
                                .with_context(|| {
                                    anyhow!(
                                        "{}\nExpected `self.n_cars_by_type` to contain '{}'",
                                        format_dbg!(),
                                        rv.car_type
                                    )
                                })? as f64;
                    Ok(mass)
                })?;
        let max_fric_braking = uc::ACC_GRAV
            * train_params.mass_total
            * rvs
                .iter()
                .try_fold(0. * uc::R, |acc, rv| -> anyhow::Result<si::Ratio> {
                    let train_braking_ratio = acc
                        + rv.braking_ratio
                            * *self
                                .train_config
                                .n_cars_by_type
                                .get(&rv.car_type)
                                .with_context(|| {
                                    anyhow!(
                                        "{}\nExpected `self.n_cars_by_type` to contain '{}'",
                                        format_dbg!(),
                                        rv.car_type
                                    )
                                })? as f64;
                    Ok(train_braking_ratio)
                })?
            / self.train_config.cars_total() as f64;

        let state = TrainState::new(
            length,
            train_mass_static,
            mass_adj,
            mass_freight,
            self.init_train_state,
        );

        let path_tpc = PathTpc::new(train_params);

        let train_res = {
            let res_bearing = res_kind::bearing::Basic::new(rvs.iter().try_fold(
                0. * uc::N,
                |acc, rv| -> anyhow::Result<si::Force> {
                    let train_bearing_res = acc
                        + rv.bearing_res_per_axle
                            * rv.axle_count as f64
                            * *self
                                .train_config
                                .n_cars_by_type
                                .get(&rv.car_type)
                                .with_context(|| {
                                    anyhow!(
                                        "{}\nExpected `self.n_cars_by_type` to contain '{}'",
                                        format_dbg!(),
                                        rv.car_type
                                    )
                                })? as f64;
                    Ok(train_bearing_res)
                },
            )?);
            // TODO: ask Tyler if total train rolling ratio can be calculated on a rail car mass-averaged basis
            let res_rolling = res_kind::rolling::Basic::new(rvs.iter().try_fold(
                0.0 * uc::R,
                |acc, rv| -> anyhow::Result<si::Ratio> {
                    let train_rolling_ratio = acc
                        + rv.rolling_ratio * rv.mass_static_total()
                            / train_mass_static
                            / *self
                                .train_config
                                .n_cars_by_type
                                .get(&rv.car_type)
                                .with_context(|| {
                                    anyhow!(
                                        "{}\nExpected `self.n_cars_by_type` to contain '{}'",
                                        format_dbg!(),
                                        rv.car_type
                                    )
                                })? as f64;
                    Ok(train_rolling_ratio)
                },
            )?);
            // TODO: ask Tyler if total train rolling ratio can be calculated on a rail car mass-averaged basis
            let davis_b = res_kind::davis_b::Basic::new(rvs.iter().try_fold(
                0.0 * uc::S / uc::M,
                |acc, rv| -> anyhow::Result<si::InverseVelocity> {
                    let train_rolling_ratio = acc
                        + rv.davis_b * rv.mass_static_total()
                            / train_mass_static
                            / *self
                                .train_config
                                .n_cars_by_type
                                .get(&rv.car_type)
                                .with_context(|| {
                                    anyhow!(
                                        "{}\nExpected `self.n_cars_by_type` to contain '{}'",
                                        format_dbg!(),
                                        rv.car_type
                                    )
                                })? as f64;
                    Ok(train_rolling_ratio)
                },
            )?);
            let res_aero =
                res_kind::aerodynamic::Basic::new(match &self.train_config.drag_area_vec {
                    Some(dave) => {
                        log::info!("Using `drag_coeff_vec` to calculate aero resistance.");
                        dave.iter().fold(0. * uc::M2, |acc, dc| *dc + acc)
                    }
                    None => rvs.iter().try_fold(
                        0.0 * uc::M2,
                        |acc, rv| -> anyhow::Result<si::Area> {
                            let train_drag = acc
                                + rv.drag_area
                                    * *self
                                        .train_config
                                        .n_cars_by_type
                                        .get(&rv.car_type)
                                        .with_context(|| {
                                            anyhow!(
                                            "{}\nExpected `self.n_cars_by_type` to contain '{}'",
                                            format_dbg!(),
                                            rv.car_type
                                        )
                                        })? as f64;
                            Ok(train_drag)
                        },
                    )?,
                });
            let res_grade = res_kind::path_res::Strap::new(path_tpc.grades(), &state)?;
            let res_curve = res_kind::path_res::Strap::new(path_tpc.curves(), &state)?;
            TrainRes::Strap(res_method::Strap::new(
                res_bearing,
                res_rolling,
                davis_b,
                res_aero,
                res_grade,
                res_curve,
            ))
        };

        // brake propagation rate is 800 ft/s (about speed of sound)
        // ramp up duration is ~30 s
        // TODO: make this not hard coded!
        // TODO: remove save_interval from new function!
        let ramp_up_time = 0.0 * uc::S;
        let ramp_up_coeff = 0.6 * uc::R;

        let fric_brake = FricBrake::new(
            max_fric_braking,
            ramp_up_time,
            ramp_up_coeff,
            None,
            save_interval,
        );

        Ok((state, path_tpc, train_res, fric_brake))
    }

    pub fn make_set_speed_train_sim<Q: AsRef<[Link]>, R: AsRef<[LinkIdx]>>(
        &self,
        rail_vehicles: &[RailVehicle],
        network: Q,
        link_path: R,
        speed_trace: SpeedTrace,
        save_interval: Option<usize>,
    ) -> anyhow::Result<SetSpeedTrainSim> {
        ensure!(
            self.origin_id.is_none() & self.destination_id.is_none(),
            "{}\n`origin_id` and `destination_id` must both be `None` when calling `make_set_speed_train_sim`.",
            format_dbg!()
        );

        let (state, mut path_tpc, train_res, _fric_brake) =
            self.make_train_sim_parts(rail_vehicles, save_interval)?;

        path_tpc.extend(network, link_path)?;
        Ok(SetSpeedTrainSim::new(
            self.loco_con.clone(),
            state,
            speed_trace,
            train_res,
            path_tpc,
            save_interval,
        ))
    }

    pub fn make_speed_limit_train_sim(
        &self,
        rail_vehicles: &[RailVehicle],
        location_map: &LocationMap,
        save_interval: Option<usize>,
        simulation_days: Option<i32>,
        scenario_year: Option<i32>,
    ) -> anyhow::Result<SpeedLimitTrainSim> {
        let (state, path_tpc, train_res, fric_brake) = self
            .make_train_sim_parts(rail_vehicles, save_interval)
            .with_context(|| format_dbg!())?;

        ensure!(
            self.origin_id.is_some() & self.destination_id.is_some(),
            "{}\nBoth `origin_id` and `destination_id` must be provided when initializing{} ",
            format_dbg!(),
            "`TrainSimBuilder` for `make_speed_limit_train_sim` to work."
        );

        Ok(SpeedLimitTrainSim::new(
            self.train_id.clone(),
            // `self.origin_id` verified to be `Some` earlier
            location_map
                .get(self.origin_id.as_ref().unwrap())
                .with_context(|| {
                    anyhow!(format!(
                        "{}\n`origin_id`: \"{}\" not found in `location_map` keys: {:?}",
                        format_dbg!(),
                        self.origin_id.as_ref().unwrap(),
                        location_map.keys(),
                    ))
                })?,
            // `self.destination_id` verified to be `Some` earlier
            location_map
                .get(self.destination_id.as_ref().unwrap())
                .with_context(|| {
                    anyhow!(format!(
                        "{}\n`destination_id`: \"{}\" not found in `location_map` keys: {:?}",
                        format_dbg!(),
                        self.destination_id.as_ref().unwrap(),
                        location_map.keys(),
                    ))
                })?,
            self.loco_con.clone(),
            state,
            train_res,
            path_tpc,
            fric_brake,
            save_interval,
            simulation_days,
            scenario_year,
        ))
    }
}

/// This may be deprecated soon! Slts building occurs in train planner.
#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn build_speed_limit_train_sims(
    train_sim_builders: Vec<TrainSimBuilder>,
    rail_vehicles: Vec<RailVehicle>,
    location_map: LocationMap,
    save_interval: Option<usize>,
    simulation_days: Option<i32>,
    scenario_year: Option<i32>,
) -> anyhow::Result<SpeedLimitTrainSimVec> {
    let mut speed_limit_train_sims = Vec::with_capacity(train_sim_builders.len());
    for tsb in train_sim_builders.iter() {
        speed_limit_train_sims.push(tsb.make_speed_limit_train_sim(
            &rail_vehicles,
            &location_map,
            save_interval,
            simulation_days,
            scenario_year,
        )?);
    }
    Ok(SpeedLimitTrainSimVec(speed_limit_train_sims))
}

#[allow(unused_variables)]
#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn run_speed_limit_train_sims(
    mut speed_limit_train_sims: SpeedLimitTrainSimVec,
    network: &PyAny,
    train_consist_plan_py: PyDataFrame,
    loco_pool_py: PyDataFrame,
    refuel_facilities_py: PyDataFrame,
    timed_paths: Vec<Vec<LinkIdxTime>>,
) -> anyhow::Result<(SpeedLimitTrainSimVec, PyDataFrame)> {
    let network = match network.extract::<Network>() {
        Ok(n) => n,
        Err(_) => {
            let n = network
                .extract::<Vec<Link>>()
                .map_err(|_| anyhow!("{}", format_dbg!()))?;
            Network(n)
        }
    };

    let train_consist_plan: DataFrame = train_consist_plan_py.into();
    let mut loco_pool: DataFrame = loco_pool_py.into();
    let refuel_facilities: DataFrame = refuel_facilities_py.into();

    loco_pool = loco_pool
        .lazy()
        .with_columns(vec![
            lit(f64::ZERO).alias("Trip_Energy_J").to_owned(),
            lit(f64::INFINITY).alias("Ready_Time_Min").to_owned(),
            lit(f64::INFINITY).alias("Ready_Time_Est").to_owned(),
            lit("Ready").alias("Status").to_owned(),
            col("SOC_Max_J").alias("SOC_J").to_owned(),
        ])
        .collect()
        .unwrap();

    let mut arrival_times = train_consist_plan
        .clone()
        .lazy()
        .select(vec![
            col("Arrival_Time_Actual_Hr"),
            col("Locomotive_ID"),
            col("Destination_ID"),
            col("TrainSimVec_Index"),
        ])
        .sort_by_exprs(
            vec![col("Arrival_Time_Actual_Hr"), col("Locomotive_ID")],
            vec![false, false],
            false,
            false,
        )
        .collect()
        .unwrap();

    let departure_times = train_consist_plan
        .clone()
        .lazy()
        .select(vec![col("Departure_Time_Actual_Hr"), col("Locomotive_ID")])
        .sort_by_exprs(
            vec![col("Locomotive_ID"), col("Departure_Time_Actual_Hr")],
            vec![false, false],
            false,
            false,
        )
        .collect()
        .unwrap();

    let mut refuel_sessions = DataFrame::default();

    let active_loco_statuses =
        Series::from_iter(vec!["Refueling".to_string(), "Dispatched".to_string()]);
    let mut current_time: f64 = (arrival_times)
        .column("Arrival_Time_Actual_Hr")?
        .min()
        .unwrap();

    let mut done = false;
    while !done {
        let arrivals_mask = arrival_times
            .column("Arrival_Time_Actual_Hr")?
            .equal(current_time)?;
        let arrivals = arrival_times.clone().filter(&arrivals_mask)?;
        let arrivals_merged =
            loco_pool
                .clone()
                .left_join(&arrivals, &["Locomotive_ID"], &["Locomotive_ID"])?;
        let arrival_locations = arrivals_merged.column("Destination_ID")?;
        if arrivals.height() > 0 {
            let arrival_ids = arrivals.column("Locomotive_ID")?;
            loco_pool = loco_pool
                .lazy()
                .with_columns(vec![
                    when(col("Locomotive_ID").is_in(lit(arrival_ids.clone())))
                        .then(lit("Queued"))
                        .otherwise(col("Status"))
                        .alias("Status"),
                    when(col("Locomotive_ID").is_in(lit(arrival_ids.clone())))
                        .then(lit(current_time))
                        .otherwise(col("Ready_Time_Est"))
                        .alias("Ready_Time_Est"),
                    when(col("Locomotive_ID").is_in(lit(arrival_ids.clone())))
                        .then(lit(arrival_locations.clone()))
                        .otherwise(col("Node"))
                        .alias("Node"),
                ])
                .drop_columns(vec![
                    "Refueler_J_Per_Hr",
                    "Refueler_Efficiency",
                    "Port_Count",
                    "Battery_Headroom_J",
                ])
                .join(
                    refuel_facilities.clone().lazy().select(&[
                        col("Node"),
                        col("Locomotive_Type"),
                        col("Fuel_Type"),
                        col("Refueler_J_Per_Hr"),
                        col("Refueler_Efficiency"),
                        col("Port_Count"),
                        col("Battery_Headroom_J"),
                    ]),
                    [col("Node"), col("Locomotive_Type"), col("Fuel_Type")],
                    [col("Node"), col("Locomotive_Type"), col("Fuel_Type")],
                    JoinArgs::new(JoinType::Left),
                )
                .with_columns(vec![col("Battery_Headroom_J").fill_null(0)])
                .with_columns(vec![max_horizontal([
                    col("SOC_Max_J") - col("Battery_Headroom_J"),
                    col("SOC_Min_J"),
                ])
                .alias("SOC_Target_J")])
                .sort("Locomotive_ID", SortOptions::default())
                .collect()
                .unwrap();

            let indices = arrivals.column("TrainSimVec_Index")?.u32()?.unique()?;
            for index in indices.into_iter() {
                let idx = index.unwrap() as usize;
                let departing_soc_pct = train_consist_plan
                    .clone()
                    .lazy()
                    // retain rows in which "TrainSimVec_Index" equals current `index`
                    .filter(col("TrainSimVec_Index").eq(index.unwrap()))
                    // Select "Locomotive_ID" column
                    .select(vec![col("Locomotive_ID")])
                    // find unique locomotive IDs
                    .unique(None, UniqueKeepStrategy::First)
                    .join(
                        loco_pool.clone().lazy(),
                        [col("Locomotive_ID")],
                        [col("Locomotive_ID")],
                        JoinArgs::new(JoinType::Left),
                    )
                    .sort("Locomotive_ID", SortOptions::default())
                    .with_columns(vec![(col("SOC_J") / col("Capacity_J")).alias("SOC_Pct")])
                    .collect()?
                    .column("SOC_Pct")?
                    .clone()
                    .into_series();

                let departing_soc_pct_vec: Vec<f64> =
                    departing_soc_pct.f64()?.into_no_null_iter().collect();
                let sim = &mut speed_limit_train_sims.0[idx];
                sim.loco_con
                    .loco_vec
                    .iter_mut()
                    .zip(departing_soc_pct_vec)
                    .for_each(
                        |(loco, soc)| match &mut loco.reversible_energy_storage_mut() {
                            Some(loco) => loco.state.soc = soc * uc::R,
                            None => {}
                        },
                    );
                let _ = sim
                    .walk_timed_path(&network, &timed_paths[idx])
                    .map_err(|err| err.context(format!("train sim idx: {}", idx)));

                let new_soc_vec: Vec<f64> = sim
                    .loco_con
                    .loco_vec
                    .iter()
                    .map(|loco| match loco.loco_type {
                        PowertrainType::BatteryElectricLoco(_) => {
                            (loco.reversible_energy_storage().unwrap().state.soc
                                * loco.reversible_energy_storage().unwrap().energy_capacity)
                                .get::<si::joule>()
                        }
                        _ => f64::ZERO,
                    })
                    .collect();
                let new_energy_j_vec: Vec<f64> = sim
                    .loco_con
                    .loco_vec
                    .iter()
                    .map(|loco| match loco.loco_type {
                        PowertrainType::BatteryElectricLoco(_) => (loco
                            .reversible_energy_storage()
                            .unwrap()
                            .state
                            .energy_out_chemical)
                            .get::<si::joule>(),
                        _ => f64::ZERO,
                    })
                    .collect();
                let mut all_current_socs: Vec<f64> = loco_pool
                    .column("SOC_J")?
                    .f64()?
                    .into_no_null_iter()
                    .collect();
                let mut all_energy_j: Vec<f64> = (loco_pool.column("SOC_J")?.f64()? * 0.0)
                    .into_no_null_iter()
                    .collect();
                let idx_mask = arrival_times
                    .column("TrainSimVec_Index")?
                    .equal(idx as u32)?;
                let arrival_locos = arrival_times.filter(&idx_mask)?;
                let arrival_loco_ids = arrival_locos.column("Locomotive_ID")?.u32()?;
                let arrival_loco_mask = loco_pool
                    .column("Locomotive_ID")?
                    .is_in(&(arrival_loco_ids.clone().into_series()))
                    .unwrap();
                // Get the indices of true values in the boolean ChunkedArray
                let arrival_loco_indices: Vec<usize> = arrival_loco_mask
                    .into_iter()
                    .enumerate()
                    .filter(|(_, val)| val.unwrap_or_default())
                    .map(|(i, _)| i)
                    .collect();

                // TODO: rewrite this a little so it doesn't depend on the previous sort
                for (index, value) in arrival_loco_indices.iter().zip(new_soc_vec) {
                    all_current_socs[*index] = value;
                }
                for (index, value) in arrival_loco_indices.iter().zip(new_energy_j_vec) {
                    all_energy_j[*index] = value;
                }
                loco_pool = loco_pool
                    .lazy()
                    .with_columns(vec![
                        when(lit(arrival_loco_mask.clone().into_series()))
                            .then(lit(Series::new("SOC_J", all_current_socs)))
                            .otherwise(col("SOC_J"))
                            .alias("SOC_J"),
                        when(lit(arrival_loco_mask.clone().into_series()))
                            .then(lit(Series::new("Trip_Energy_J", all_energy_j)))
                            .otherwise(col("Trip_Energy_J"))
                            .alias("Trip_Energy_J"),
                    ])
                    .collect()
                    .unwrap();
            }
            loco_pool = loco_pool
                .lazy()
                .sort("Ready_Time_Est", SortOptions::default())
                .collect()
                .unwrap();
        }

        let refueling_mask = (loco_pool).column("Status")?.equal("Refueling")?;
        let refueling_finished_mask =
            refueling_mask & (loco_pool).column("Ready_Time_Est")?.equal(current_time)?;
        let refueling_finished = loco_pool.clone().filter(&refueling_finished_mask)?;
        if refueling_finished_mask.sum().unwrap_or_default() > 0 {
            loco_pool = loco_pool
                .lazy()
                .with_columns(vec![when(lit(refueling_finished_mask.into_series()))
                    .then(lit("Ready"))
                    .otherwise(col("Status"))
                    .alias("Status")])
                .collect()
                .unwrap();
        }

        if (arrivals.height() > 0) || (refueling_finished.height() > 0) {
            // update queue
            let place_in_queue = loco_pool
                .clone()
                .lazy()
                .select(&[((col("Status").eq(lit("Refueling")).sum().over([
                    "Node",
                    "Locomotive_Type",
                    "Fuel_Type",
                ])) + (col("Status").eq(lit("Queued")).cumsum(false).over([
                    "Node",
                    "Locomotive_Type",
                    "Fuel_Type",
                ])))
                .alias("place_in_queue")])
                .collect()?
                .column("place_in_queue")?
                .clone()
                .into_series();
            let future_times_mask = departure_times
                .column("Departure_Time_Actual_Hr")?
                .f64()?
                .gt(current_time);

            let next_departure_time = departure_times
                .clone()
                .lazy()
                .filter(col("Departure_Time_Actual_Hr").gt(lit(current_time)))
                .groupby(["Locomotive_ID"])
                .agg([col("Departure_Time_Actual_Hr").min()])
                .collect()
                .unwrap();

            let departures_merged = loco_pool.clone().left_join(
                &next_departure_time,
                &["Locomotive_ID"],
                &["Locomotive_ID"],
            )?;
            let departure_times = departures_merged
                .column("Departure_Time_Actual_Hr")?
                .f64()?;

            let target_j = loco_pool
                .clone()
                .lazy()
                .select(&[(col("SOC_Max_J") - col("Battery_Headroom_J")).alias("Target_J")])
                .collect()?
                .column("Target_J")?
                .clone();
            let target_j_f64 = target_j.f64()?;
            let current_j = loco_pool.column("SOC_J")?.f64()?;

            let soc_target: Vec<f64> = target_j_f64
                .into_iter()
                .zip(current_j.into_iter())
                .map(|(b, v)| b.unwrap_or(f64::ZERO).max(v.unwrap_or(f64::ZERO)))
                .collect::<Vec<_>>();
            let soc_target_series = Series::new("soc_target", soc_target);

            let refuel_end_time_ideal = loco_pool
                .clone()
                .lazy()
                .select(&[(lit(current_time)
                    + (max_horizontal([col("SOC_J"), col("SOC_Target_J")]) - col("SOC_J"))
                        / col("Refueler_J_Per_Hr"))
                .alias("Refuel_End_Time")])
                .collect()?
                .column("Refuel_End_Time")?
                .clone()
                .into_series();

            let refuel_end_time: Vec<f64> = departure_times
                .into_iter()
                .zip(refuel_end_time_ideal.f64()?.into_iter())
                .map(|(b, v)| b.unwrap_or(f64::INFINITY).min(v.unwrap_or(f64::INFINITY)))
                .collect::<Vec<_>>();

            let mut refuel_duration: Vec<f64> = refuel_end_time.clone();
            for element in refuel_duration.iter_mut() {
                *element -= current_time;
            }

            let refuel_duration_series = Series::new("refuel_duration", refuel_duration);
            let refuel_end_series = Series::new("refuel_end_time", refuel_end_time);

            loco_pool = loco_pool
                .lazy()
                .with_columns(vec![
                    lit(place_in_queue),
                    lit(refuel_duration_series),
                    lit(refuel_end_series),
                ])
                .collect()
                .unwrap();

            // store the filter as an Expr
            let refuel_starting = loco_pool
                .clone()
                .lazy()
                .filter(
                    col("Status")
                        .eq(lit("Queued"))
                        .and(col("Port_Count").gt_eq(col("place_in_queue"))),
                )
                .collect()
                .unwrap();

            let these_refuel_sessions = df![
                "Node" => refuel_starting.column("Node").unwrap(),
                "Locomotive_Type" => refuel_starting.column("Locomotive_Type").unwrap(),
                "Fuel_Type" => refuel_starting.column("Fuel_Type").unwrap(),
                "Locomotive_ID" => refuel_starting.column("Locomotive_ID").unwrap(),
                "Refueler_J_Per_Hr" => refuel_starting.column("Refueler_J_Per_Hr").unwrap(),
                "Refueler_Efficiency" => refuel_starting.column("Refueler_Efficiency").unwrap(),
                "Trip_Energy_J" => refuel_starting.column("Trip_Energy_J").unwrap(),
                "SOC_J" => refuel_starting.column("SOC_J").unwrap(),
                "Refuel_Energy_J" => refuel_starting.clone().lazy().select(&[
                    (col("Refueler_J_Per_Hr")*col("refuel_duration")/col("Refueler_Efficiency")).alias("Refuel_Energy_J")
                    ]).collect()?.column("Refuel_Energy_J")?.clone().into_series(),
                "Refuel_Duration_Hr" => refuel_starting.column("refuel_duration").unwrap(),
                "Refuel_Start_Time_Hr" => refuel_starting.column("refuel_end_time").unwrap() -
                    refuel_starting.column("refuel_duration").unwrap(),
                "Refuel_End_Time_Hr" => refuel_starting.column("refuel_end_time").unwrap()
            ]?;
            refuel_sessions.vstack_mut(&these_refuel_sessions)?;
            // set finishedCharging times to min(max soc OR departure time)
            loco_pool = loco_pool
                .clone()
                .lazy()
                .with_columns(vec![
                    when(
                        col("Status")
                            .eq(lit("Queued"))
                            .and(col("Port_Count").gt_eq(col("place_in_queue"))),
                    )
                    .then(col("SOC_J") + col("refuel_duration") * col("Refueler_J_Per_Hr"))
                    .otherwise(col("SOC_J"))
                    .alias("SOC_J"),
                    when(
                        col("Status")
                            .eq(lit("Queued"))
                            .and(col("Port_Count").gt_eq(col("place_in_queue"))),
                    )
                    .then(col("refuel_end_time"))
                    .otherwise(col("Ready_Time_Est"))
                    .alias("Ready_Time_Est"),
                    when(
                        col("Status")
                            .eq(lit("Queued"))
                            .and(col("Port_Count").gt_eq(col("place_in_queue"))),
                    )
                    .then(lit("Refueling"))
                    .otherwise(col("Status"))
                    .alias("Status"),
                ])
                .collect()
                .unwrap();

            loco_pool = loco_pool.drop("place_in_queue")?;
            loco_pool = loco_pool.drop("refuel_duration")?;
            loco_pool = loco_pool.drop("refuel_end_time")?;
        }

        let active_loco_ready_times = loco_pool
            .clone()
            .lazy()
            .filter(col("Status").is_in(lit(active_loco_statuses.clone())))
            .select(vec![col("Ready_Time_Est")])
            .collect()?
            .column("Ready_Time_Est")?
            .clone()
            .into_series();
        arrival_times = arrival_times
            .lazy()
            .filter(col("Arrival_Time_Actual_Hr").gt(current_time))
            .collect()?;
        let arrival_times_remaining = arrival_times
            .clone()
            .lazy()
            .select(vec![
                col("Arrival_Time_Actual_Hr").alias("Arrival_Time_Actual_Hr")
            ])
            .collect()?
            .column("Arrival_Time_Actual_Hr")?
            .clone()
            .into_series();

        if (arrival_times_remaining.len() == 0) & (active_loco_ready_times.len() == 0) {
            done = true;
        } else {
            let min1 = active_loco_ready_times
                .f64()?
                .min()
                .unwrap_or(f64::INFINITY);
            let min2 = arrival_times_remaining
                .f64()?
                .min()
                .unwrap_or(f64::INFINITY);
            current_time = f64::min(min1, min2);
        }
    }

    Ok((speed_limit_train_sims, PyDataFrame(refuel_sessions)))
}

// This MUST remain a unit struct to trigger correct tolist() behavior
#[altrios_api(
    #[pyo3(name = "get_energy_fuel_joules")]
    pub fn get_energy_fuel_py(&self, annualize: bool) -> f64 {
        self.get_energy_fuel(annualize).get::<si::joule>()
    }

    #[pyo3(name = "get_net_energy_res_joules")]
    pub fn get_net_energy_res_py(&self, annualize: bool) -> f64 {
        self.get_net_energy_res(annualize).get::<si::joule>()
    }

    #[pyo3(name = "get_megagram_kilometers")]
    pub fn get_megagram_kilometers_py(&self, annualize: bool) -> f64 {
        self.get_megagram_kilometers(annualize)
    }

    #[pyo3(name = "get_kilometers")]
    pub fn get_kilometers_py(&self, annualize: bool) -> f64 {
        self.get_kilometers(annualize)
    }

    #[pyo3(name = "get_res_kilometers")]
    pub fn get_res_kilometers_py(&mut self, annualize: bool) -> f64 {
        self.get_res_kilometers(annualize)
    }

    #[pyo3(name = "get_non_res_kilometers")]
    pub fn get_non_res_kilometers_py(&mut self, annualize: bool) -> f64 {
        self.get_non_res_kilometers(annualize)
    }

    #[pyo3(name = "set_save_interval")]
    pub fn set_save_interval_py(&mut self, save_interval: Option<usize>) {
        self.set_save_interval(save_interval);
    }
)]
#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeedLimitTrainSimVec(pub Vec<SpeedLimitTrainSim>);

impl SpeedLimitTrainSimVec {
    pub fn get_energy_fuel(&self, annualize: bool) -> si::Energy {
        self.0
            .iter()
            .map(|sim| sim.get_energy_fuel(annualize))
            .sum()
    }

    pub fn get_net_energy_res(&self, annualize: bool) -> si::Energy {
        self.0
            .iter()
            .map(|sim| sim.get_net_energy_res(annualize))
            .sum()
    }

    pub fn get_megagram_kilometers(&self, annualize: bool) -> f64 {
        self.0
            .iter()
            .map(|sim| sim.get_megagram_kilometers(annualize))
            .sum()
    }

    pub fn get_kilometers(&self, annualize: bool) -> f64 {
        self.0.iter().map(|sim| sim.get_kilometers(annualize)).sum()
    }

    pub fn get_res_kilometers(&mut self, annualize: bool) -> f64 {
        self.0
            .iter_mut()
            .map(|sim| sim.get_res_kilometers(annualize))
            .sum()
    }

    pub fn get_non_res_kilometers(&mut self, annualize: bool) -> f64 {
        self.0
            .iter_mut()
            .map(|sim| sim.get_non_res_kilometers(annualize))
            .sum()
    }

    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.0
            .iter_mut()
            .for_each(|slts| slts.set_save_interval(save_interval));
    }
}

impl SerdeAPI for SpeedLimitTrainSimVec {
    fn init(&mut self) -> anyhow::Result<()> {
        self.0.iter_mut().try_for_each(|ts| ts.init())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::rail_vehicle::import_rail_vehicles;

    #[test]
    fn test_make_train_params() {
        let train_configs = vec![TrainConfig::valid()];
        let rail_vehicle_map =
            import_rail_vehicles(Path::new("./src/train/test_rail_vehicles.csv")).unwrap();
        for train_config in train_configs {
            let rail_vehicle =
                rail_vehicle_map[train_config.rail_vehicle_type.as_ref().unwrap()].clone();
            train_config.make_train_params(&[rail_vehicle]).unwrap();
        }
    }

    #[test]
    fn test_make_speed_limit_train_sims() {
        let train_configs = vec![TrainConfig::valid()];
        let mut rvm_file = project_root::get_project_root().unwrap();
        rvm_file.push("altrios-core/src/train/test_rail_vehicles.csv");
        let rail_vehicle_map = match import_rail_vehicles(&rvm_file) {
            Ok(rvm) => rvm,
            Err(_) => {
                import_rail_vehicles(Path::new("./src/train/test_rail_vehicles.csv")).unwrap()
            }
        };
        let mut location_map = LocationMap::new();
        location_map.insert("dummy".to_string(), vec![]);

        let consist = Consist::default();

        for train_config in train_configs {
            let tsb = TrainSimBuilder::new(
                "".to_string(),
                train_config.clone(),
                consist.clone(),
                Some("dummy".to_string()),
                Some("dummy".to_string()),
                None,
            );
            let rail_vehicle = rail_vehicle_map[&train_config.rail_vehicle_type.unwrap()].clone();
            tsb.make_speed_limit_train_sim(&[rail_vehicle], &location_map, None, None, None)
                .unwrap();
        }
    }
}
