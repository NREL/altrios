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
        cars_empty: u32,
        cars_loaded: u32,
        rail_vehicle_type: Option<String>,
        train_type: Option<TrainType>,
        train_length_meters: Option<f64>,
        train_mass_kilograms: Option<f64>,
        drag_coeff_vec: Option<Vec<f64>>,
    ) -> anyhow::Result<Self> {
        Self::new(
            cars_empty,
            cars_loaded,
            rail_vehicle_type,
            train_type.unwrap_or_default(),
            train_length_meters.map(|v| v * uc::M),
            train_mass_kilograms.map(|v| v * uc::KG),
            drag_coeff_vec.map(|dcv| dcv.iter().map(|dc| *dc * uc::R).collect())
        )
    }

    #[pyo3(name = "make_train_params")]
    fn make_train_params_py(&self, rail_vehicle: RailVehicle) -> anyhow::Result<TrainParams> {
        Ok(self.make_train_params(&rail_vehicle))
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
    fn get_drag_coeff_vec(&self) -> Option<Vec<f64>> {
        self.drag_coeff_vec
            .as_ref()
                .map(
                    |dcv| dcv.iter().cloned().map(|x| x.get::<si::ratio>()).collect()
                )
    }

    #[setter]
    fn set_drag_coeff_vec(&mut self, new_val: Vec<f64>) -> anyhow::Result<()> {
        self.drag_coeff_vec = Some(new_val.iter().map(|x| *x * uc::R).collect());
        Ok(())
    }
)]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
/// User-defined train configuration used to generate [crate::prelude::TrainParams].
/// Any optional fields will be populated later in [TrainSimBuilder::make_train_sim_parts]
pub struct TrainConfig {
    /// Optional user-defined identifier for the car type on this train.
    pub rail_vehicle_type: Option<String>,
    /// Number of empty railcars on the train
    pub cars_empty: u32,
    /// Number of loaded railcars on the train
    pub cars_loaded: u32,
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
    /// Optional vector of coefficients for each car.  If provided, the total drag area (drag
    /// coefficient times frontal area) calculated from this vector is the sum of these coefficients
    /// times the baseline, single-vehicle `drag_area_loaded`.
    pub drag_coeff_vec: Option<Vec<si::Ratio>>,
}

impl SerdeAPI for TrainConfig {
    fn init(&mut self) -> anyhow::Result<()> {
        match &self.drag_coeff_vec {
            Some(dcv) => {
                ensure!(dcv.len() as u32 == self.cars_loaded + self.cars_empty);
            }
            None => {}
        };
        Ok(())
    }
}

impl TrainConfig {
    pub fn new(
        cars_empty: u32,
        cars_loaded: u32,
        rail_vehicle_type: Option<String>,
        train_type: TrainType,
        train_length: Option<si::Length>,
        train_mass: Option<si::Mass>,
        drag_coeff_vec: Option<Vec<si::Ratio>>,
    ) -> anyhow::Result<Self> {
        let mut train_config = Self {
            cars_empty,
            cars_loaded,
            rail_vehicle_type,
            train_type,
            train_length,
            train_mass,
            drag_coeff_vec,
        };
        train_config.init()?;
        Ok(train_config)
    }

    pub fn cars_total(&self) -> u32 {
        self.cars_empty + self.cars_loaded
    }

    pub fn make_train_params(&self, rail_vehicle: &RailVehicle) -> TrainParams {
        let mass_static = self.train_mass.unwrap_or(
            rail_vehicle.mass_static_empty * self.cars_empty as f64
                + rail_vehicle.mass_static_loaded * self.cars_loaded as f64,
        );

        let length = self
            .train_length
            .unwrap_or(rail_vehicle.length * self.cars_total() as f64);

        TrainParams {
            length,
            speed_max: rail_vehicle.speed_max_loaded.max(if self.cars_empty > 0 {
                rail_vehicle.speed_max_empty
            } else {
                uc::MPS * f64::INFINITY
            }),
            mass_total: mass_static,
            mass_per_brake: mass_static
                / (self.cars_total() * rail_vehicle.brake_count as u32) as f64,
            axle_count: self.cars_total() * rail_vehicle.axle_count as u32,
            train_type: self.train_type,
            curve_coeff_0: rail_vehicle.curve_coeff_0,
            curve_coeff_1: rail_vehicle.curve_coeff_1,
            curve_coeff_2: rail_vehicle.curve_coeff_2,
        }
    }
}

impl Valid for TrainConfig {
    fn valid() -> Self {
        Self {
            rail_vehicle_type: Some("Bulk".to_string()),
            cars_empty: 0,
            cars_loaded: 100,
            train_type: TrainType::Freight,
            train_length: None,
            train_mass: None,
            drag_coeff_vec: None,
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
        rail_vehicle: RailVehicle,
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
            &rail_vehicle,
            network,
            link_path,
            speed_trace,
            save_interval
        )
    }

    #[pyo3(name = "make_speed_limit_train_sim")]
    fn make_speed_limit_train_sim_py(
        &self,
        rail_vehicle: RailVehicle,
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
        rail_vehicle: &RailVehicle,
        save_interval: Option<usize>,
    ) -> anyhow::Result<(TrainState, PathTpc, TrainRes, FricBrake)> {
        let veh = rail_vehicle;
        let train_params = self.train_config.make_train_params(rail_vehicle);

        let length = train_params.length;
        // TODO: figure out what to do about rotational mass of locomotive components (e.g. axles, gearboxes, motor shafts)
        let mass_static = train_params.mass_total + self.loco_con.mass()?.unwrap();
        let cars_total = self.train_config.cars_total() as f64;
        let mass_adj = mass_static + veh.mass_extra_per_axle * train_params.axle_count as f64;
        let mass_freight =
            si::Mass::ZERO.max(train_params.mass_total - veh.mass_static_empty * cars_total);
        let max_fric_braking = uc::ACC_GRAV
            * train_params.mass_total
            * (veh.braking_ratio_empty * self.train_config.cars_empty as f64
                + veh.braking_ratio_loaded * self.train_config.cars_loaded as f64)
            / cars_total;

        let state = TrainState::new(
            length,
            mass_static,
            mass_adj,
            mass_freight,
            self.init_train_state,
        );

        let path_tpc = PathTpc::new(train_params);

        let train_res = {
            let res_bearing = res_kind::bearing::Basic::new(
                veh.bearing_res_per_axle * train_params.axle_count as f64,
            );
            let res_rolling = res_kind::rolling::Basic::new(veh.rolling_ratio);
            let davis_b = res_kind::davis_b::Basic::new(veh.davis_b);
            let res_aero =
                res_kind::aerodynamic::Basic::new(match &self.train_config.drag_coeff_vec {
                    Some(dcv) => dcv
                        .iter()
                        .fold(0. * uc::M2, |acc, dc| *dc * veh.drag_area_loaded + acc),
                    None => {
                        veh.drag_area_empty * self.train_config.cars_empty as f64
                            + veh.drag_area_loaded * self.train_config.cars_loaded as f64
                    }
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
        rail_vehicle: &RailVehicle,
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
            self.make_train_sim_parts(rail_vehicle, save_interval)?;

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
        rail_vehicle: &RailVehicle,
        location_map: &LocationMap,
        save_interval: Option<usize>,
        simulation_days: Option<i32>,
        scenario_year: Option<i32>,
    ) -> anyhow::Result<SpeedLimitTrainSim> {
        let (state, path_tpc, train_res, fric_brake) = self
            .make_train_sim_parts(rail_vehicle, save_interval)
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
                .ok_or_else(|| {
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
                .ok_or_else(|| {
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
    rail_veh: RailVehicle,
    location_map: LocationMap,
    save_interval: Option<usize>,
    simulation_days: Option<i32>,
    scenario_year: Option<i32>,
) -> anyhow::Result<SpeedLimitTrainSimVec> {
    let mut speed_limit_train_sims = Vec::with_capacity(train_sim_builders.len());
    for tsb in train_sim_builders.iter() {
        speed_limit_train_sims.push(tsb.make_speed_limit_train_sim(
            &rail_veh,
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
                        col("Refueler_J_Per_Hr"),
                        col("Refueler_Efficiency"),
                        col("Port_Count"),
                        col("Battery_Headroom_J"),
                    ]),
                    [col("Node"), col("Locomotive_Type")],
                    [col("Node"), col("Locomotive_Type")],
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
                .select(&[((col("Status")
                    .eq(lit("Refueling"))
                    .sum()
                    .over(["Node", "Locomotive_Type"]))
                    + (col("Status")
                        .eq(lit("Queued"))
                        .cumsum(false)
                        .over(["Node", "Locomotive_Type"])))
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
#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
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
            let rail_vehicle = &rail_vehicle_map[train_config.rail_vehicle_type.as_ref().unwrap()];
            train_config.make_train_params(rail_vehicle);
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
            let rail_vehicle = &rail_vehicle_map[&train_config.rail_vehicle_type.unwrap()];
            tsb.make_speed_limit_train_sim(rail_vehicle, &location_map, None, None, None)
                .unwrap();
        }
    }
}
