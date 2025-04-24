use super::{braking_point::BrakingPoints, friction_brakes::*, train_imports::*};
use crate::imports::*;
use crate::track::link::network::Network;
use crate::track::{LinkPoint, Location};

#[altrios_api(
    #[new]
    fn __new__(
        link_idx: LinkIdx,
        time_seconds: f64,
    ) -> Self {
        Self::new(
            link_idx,
            time_seconds * uc::S
        )
    }
)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize, SerdeAPI)]
pub struct LinkIdxTime {
    pub link_idx: LinkIdx,
    pub time: si::Time,
}

impl LinkIdxTime {
    pub fn new(link_idx: LinkIdx, time: si::Time) -> Self {
        Self { link_idx, time }
    }
}

#[altrios_api]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize, SerdeAPI)]
/// Struct that contains a `Vec<LinkIdxTime>` for the purpose of providing `SerdeAPI` for
/// `Vec<LinkIdxTime>` in Python
pub struct TimedLinkPath(pub Vec<LinkIdxTime>);

impl TimedLinkPath {
    /// Implement the non-Python `new` method.
    pub fn new(value: Vec<LinkIdxTime>) -> Self {
        Self(value)
    }
}

impl AsRef<[LinkIdxTime]> for TimedLinkPath {
    fn as_ref(&self) -> &[LinkIdxTime] {
        &self.0
    }
}

impl From<&Vec<LinkIdxTime>> for TimedLinkPath {
    fn from(value: &Vec<LinkIdxTime>) -> Self {
        Self(value.to_vec())
    }
}

#[altrios_api(
    #[pyo3(name = "set_save_interval")]
    #[pyo3(signature = (save_interval=None))]
    /// Set save interval and cascade to nested components.
    fn set_save_interval_py(&mut self, save_interval: Option<usize>) {
        self.set_save_interval(save_interval);
    }

    #[pyo3(name = "get_save_interval")]
    fn get_save_interval_py(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.get_save_interval())
    }

    #[pyo3(name = "get_kilometers")]
    pub fn get_kilometers_py(&self, annualize: bool)  -> f64 {
        self.get_kilometers(annualize)
    }

    #[pyo3(name = "get_megagram_kilometers")]
    pub fn get_megagram_kilometers_py(&self, annualize: bool)  -> f64 {
        self.get_megagram_kilometers(annualize)
    }

    #[pyo3(name = "get_car_kilometers")]
    pub fn get_car_kilometers_py(&self, annualize: bool)  -> f64 {
        self.get_car_kilometers(annualize)
    }

    #[pyo3(name = "get_cars_moved")]
    pub fn get_cars_moved_py(&self, annualize: bool)  -> f64 {
        self.get_cars_moved(annualize)
    }

    #[pyo3(name = "get_res_kilometers")]
    pub fn get_res_kilometers_py(&mut self, annualize: bool)  -> f64 {
        self.get_res_kilometers(annualize)
    }

    #[pyo3(name = "get_non_res_kilometers")]
    pub fn get_non_res_kilometers_py(&mut self, annualize: bool)  -> f64 {
        self.get_non_res_kilometers(annualize)
    }

    #[pyo3(name = "get_net_energy_res_joules")]
    pub fn get_net_energy_res_py(&self, annualize: bool) -> f64 {
        self.get_net_energy_res(annualize).get::<si::joule>()
    }

    #[pyo3(name = "get_energy_fuel_joules")]
    pub fn get_energy_fuel_py(&self, annualize: bool) -> f64 {
        self.get_energy_fuel(annualize).get::<si::joule>()
    }

    #[pyo3(name = "get_energy_fuel_soc_corrected_joules")]
    pub fn get_energy_fuel_soc_corrected_py(&self) -> PyResult<f64> {
        Ok(self.get_energy_fuel_soc_corrected().map_err(|err| anyhow!("{:?}", err))?.get::<si::joule>())
    }

    #[pyo3(name = "walk")]
    fn walk_py(&mut self) -> anyhow::Result<()> {
        self.walk()
    }

    #[staticmethod]
    #[pyo3(name = "valid")]
    fn valid_py() -> Self {
        Self::valid()
    }

    #[pyo3(name = "extend_path")]
    pub fn extend_path_py(&mut self, network_file_path: String, link_path: Vec<LinkIdx>) -> anyhow::Result<()> {
        let network = Vec::<Link>::from_file(network_file_path, false).unwrap();

        self.extend_path(&network, &link_path)?;
        Ok(())
    }

    #[pyo3(name = "walk_timed_path")]
    pub fn walk_timed_path_py(
        &mut self,
        network: &Bound<PyAny>,
        timed_path: &Bound<PyAny>,
    ) -> anyhow::Result<()> {
        let network = match network.extract::<Network>() {
            Ok(n) => n,
            Err(_) => {
                let n = network.extract::<Vec<Link>>().map_err(|_| anyhow!("{}", format_dbg!()))?;
                Network( Default::default(), n)
            }
        };

        let timed_path = match timed_path.extract::<TimedLinkPath>() {
            Ok(tp) => tp,
            Err(_) => {
                let tp = timed_path.extract::<Vec<LinkIdxTime>>().map_err(|_| anyhow!("{}", format_dbg!()))?;
                TimedLinkPath(tp)
            }
        };
        self.walk_timed_path(&network, timed_path)
    }
)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
/// Train simulation in which speed is allowed to vary according to train capabilities and speed
/// limit.  Note that this is not guaranteed to produce identical results to [super::SetSpeedTrainSim]
/// because of differences in braking controls but should generally be very close (i.e. error in cumulative
/// fuel/battery energy should be less than 0.1%)
pub struct SpeedLimitTrainSim {
    #[api(skip_set)]
    pub train_id: String,
    pub origs: Vec<Location>,
    pub dests: Vec<Location>,
    pub loco_con: Consist,
    /// Number of railcars by type on the train
    pub n_cars_by_type: HashMap<String, u32>,
    #[serde(default)]
    #[serde(skip_serializing_if = "EqDefault::eq_default")]
    pub state: TrainState,
    #[api(skip_set, skip_get)]
    pub train_res: TrainRes,
    #[api(skip_set)]
    pub path_tpc: PathTpc,
    #[api(skip_set)]
    pub braking_points: BrakingPoints,
    pub fric_brake: FricBrake,
    /// Custom vector of [Self::state]
    #[serde(default, skip_serializing_if = "TrainStateHistoryVec::is_empty")]
    pub history: TrainStateHistoryVec,
    #[api(skip_set, skip_get)]
    save_interval: Option<usize>,
    simulation_days: Option<i32>,
    scenario_year: Option<i32>,
}

impl SpeedLimitTrainSim {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        train_id: String,
        origs: &[Location],
        dests: &[Location],
        loco_con: Consist,
        n_cars_by_type: HashMap<String, u32>,
        state: TrainState,
        train_res: TrainRes,
        path_tpc: PathTpc,
        fric_brake: FricBrake,
        save_interval: Option<usize>,
        simulation_days: Option<i32>,
        scenario_year: Option<i32>,
    ) -> Self {
        let mut train_sim = Self {
            train_id,
            origs: origs.to_vec(),
            dests: dests.to_vec(),
            loco_con,
            n_cars_by_type,
            state,
            train_res,
            path_tpc,
            braking_points: Default::default(),
            fric_brake,
            history: Default::default(),
            save_interval,
            simulation_days,
            scenario_year,
        };
        train_sim.set_save_interval(save_interval);

        train_sim
    }

    /// Returns the scaling factor to be used when converting partial-year
    /// simulations to a full year of output metrics.
    pub fn get_scaling_factor(&self, annualize: bool) -> f64 {
        if annualize {
            match self.simulation_days {
                Some(val) => 365.25 / val as f64,
                None => 365.25,
            }
        } else {
            1.0
        }
    }

    pub fn get_kilometers(&self, annualize: bool) -> f64 {
        self.state.total_dist.get::<si::kilometer>() * self.get_scaling_factor(annualize)
    }

    pub fn get_megagram_kilometers(&self, annualize: bool) -> f64 {
        self.state.mass_freight.get::<si::megagram>()
            * self.state.total_dist.get::<si::kilometer>()
            * self.get_scaling_factor(annualize)
    }

    pub fn get_car_kilometers(&self, annualize: bool) -> f64 {
        let n_cars = self.get_cars_moved(annualize);
        // Note: n_cars already includes an annualization scaling factor; no need to multiply twice.
        self.state.total_dist.get::<si::kilometer>() * n_cars
    }

    pub fn get_cars_moved(&self, annualize: bool) -> f64 {
        let n_cars: f64 = self.n_cars_by_type.values().fold(0, |acc, n| *n + acc) as f64;
        n_cars * self.get_scaling_factor(annualize)
    }

    pub fn get_res_kilometers(&mut self, annualize: bool) -> f64 {
        let n_res = self.loco_con.n_res_equipped() as f64;
        self.state.total_dist.get::<si::kilometer>() * n_res * self.get_scaling_factor(annualize)
    }

    pub fn get_non_res_kilometers(&mut self, annualize: bool) -> f64 {
        let n_res = self.loco_con.n_res_equipped() as usize;
        self.state.total_dist.get::<si::kilometer>()
            * ((self.loco_con.loco_vec.len() - n_res) as f64)
            * self.get_scaling_factor(annualize)
    }

    pub fn get_energy_fuel(&self, annualize: bool) -> si::Energy {
        self.loco_con.get_energy_fuel() * self.get_scaling_factor(annualize)
    }

    /// Returns total fuel and fuel-equivalent battery energy used for consist
    pub fn get_energy_fuel_soc_corrected(&self) -> Result<si::Energy, Error> {
        if self.save_interval != Some(1) && self.history.is_empty() {
            return Err(Error::Other(
                "Expected `save_interval = Some(1)` and non-empty history".into(),
            ));
        }
        let eta_eng_mean_consist: si::Ratio = self
            .loco_con
            .loco_vec
            .iter()
            .filter(|loco| loco.fuel_converter().is_some())
            .map(|loco| {
                let fc = loco.fuel_converter().unwrap();
                let eta_mean_for_loco = fc
                    .history
                    .eta
                    .iter()
                    .filter(|eta| eta > &&si::Ratio::ZERO)
                    .fold(si::Ratio::ZERO, |acc, eta_for_loco| acc + *eta_for_loco)
                    / (fc.history.len() as f64);
                eta_mean_for_loco
            })
            .fold(si::Ratio::ZERO, |acc, eta_for_loco_vec| {
                acc + eta_for_loco_vec
            })
            / (self.loco_con.loco_vec.len() as f64);
        let res_fuel_equiv = self
            .loco_con
            .loco_vec
            .iter()
            .map(|loco| {
                if let Some(res) = loco.reversible_energy_storage() {
                    let delta_soc: si::Ratio =
                        *res.history.soc.last().unwrap() - *res.history.soc.first().unwrap();
                    let loco_res_fuel_equiv: si::Energy = if delta_soc < si::Ratio::ZERO {
                        // net discharge
                        let eta_pos: si::Ratio = res
                            .history
                            .eta
                            .iter()
                            .zip(res.history.pwr_out_electrical.clone())
                            .filter(|(_, pwr_out)| pwr_out > &si::Power::ZERO)
                            .fold(si::Ratio::ZERO, |acc, (eta, _)| acc + *eta)
                            / (res.history.len() as f64);
                        // net discharge is equivalent to using fuel
                        -delta_soc * res.energy_capacity_usable() * eta_pos / eta_eng_mean_consist
                    } else if delta_soc > si::Ratio::ZERO {
                        // net charge
                        let eta_neg: si::Ratio = res
                            .history
                            .eta
                            .iter()
                            .zip(res.history.pwr_out_electrical.clone())
                            .filter(|(_, pwr_out)| pwr_out < &si::Power::ZERO)
                            .fold(si::Ratio::ZERO, |acc, (eta, _)| acc + *eta)
                            / (res.history.len() as f64);
                        // net charge is equivalent to making fuel fuel
                        -delta_soc * res.energy_capacity_usable() / eta_neg / eta_eng_mean_consist
                    } else {
                        si::Energy::ZERO
                    };
                    loco_res_fuel_equiv
                } else {
                    si::Energy::ZERO
                }
            })
            .sum::<si::Energy>();
        Ok(res_fuel_equiv + self.loco_con.get_energy_fuel())
    }

    pub fn get_net_energy_res(&self, annualize: bool) -> si::Energy {
        self.loco_con.get_net_energy_res() * self.get_scaling_factor(annualize)
    }

    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.save_interval = save_interval;
        self.loco_con.set_save_interval(save_interval);
        self.fric_brake.save_interval = save_interval;
    }
    pub fn get_save_interval(&self) -> Option<usize> {
        self.save_interval
    }

    pub fn extend_path(&mut self, network: &[Link], link_path: &[LinkIdx]) -> anyhow::Result<()> {
        self.path_tpc
            .extend(network, link_path)
            .with_context(|| format_dbg!())?;
        self.recalc_braking_points()
            .with_context(|| format_dbg!())?;
        Ok(())
    }
    pub fn clear_path(&mut self) {
        // let link_point_del = self.path_tpc.clear(self.state.offset_back);
        // self.train_res.fix_cache(&link_point_del);
    }

    pub fn finish(&mut self) {
        self.path_tpc.finish()
    }
    pub fn is_finished(&self) -> bool {
        self.path_tpc.is_finished()
    }
    pub fn offset_begin(&self) -> si::Length {
        self.path_tpc.offset_begin()
    }
    pub fn offset_end(&self) -> si::Length {
        self.path_tpc.offset_end()
    }
    pub fn link_idx_last(&self) -> Option<&LinkIdx> {
        self.path_tpc.link_idx_last()
    }
    pub fn link_points(&self) -> &[LinkPoint] {
        self.path_tpc.link_points()
    }

    pub fn step(&mut self) -> anyhow::Result<()> {
        self.solve_step()
            .map_err(|err| err.context(format!("time step: {}", self.state.i)))?;
        self.save_state();
        self.loco_con.step();
        self.fric_brake.step();
        self.state.i += 1;
        Ok(())
    }

    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        // set catenary power limit
        self.loco_con
            .set_cat_power_limit(&self.path_tpc, self.state.offset);
        // set aux power for the consist
        self.loco_con.set_pwr_aux(Some(true))?;
        // set the maximum power out based on dt.
        self.loco_con.set_curr_pwr_max_out(
            None,
            Some(self.state.mass_compound()?),
            Some(self.state.speed),
            self.state.dt,
        )?;
        // calculate new resistance
        self.train_res
            .update_res(&mut self.state, &self.path_tpc, &Dir::Fwd)?;
        set_link_and_offset(&mut self.state, &self.path_tpc)?;
        // solve the required power
        self.solve_required_pwr()?;

        self.loco_con.solve_energy_consumption(
            self.state.pwr_whl_out,
            Some(self.state.mass_compound()?),
            Some(self.state.speed),
            self.state.dt,
            Some(true),
        )?;
        Ok(())
    }

    fn save_state(&mut self) {
        if let Some(interval) = self.save_interval {
            if self.state.i % interval == 0 {
                self.history.push(self.state);
                self.loco_con.save_state();
                self.fric_brake.save_state();
            }
        }
    }

    /// Walks until getting to the end of the path
    fn walk_internal(&mut self) -> anyhow::Result<()> {
        while self.state.offset < self.path_tpc.offset_end() - 1000.0 * uc::FT
            || (self.state.offset < self.path_tpc.offset_end()
                && self.state.speed != si::Velocity::ZERO)
        {
            self.step()?;
        }
        Ok(())
    }

    /// Iterates `save_state` and `step` until offset >= final offset --
    /// i.e. moves train forward until it reaches destination.
    pub fn walk(&mut self) -> anyhow::Result<()> {
        self.save_state();
        self.walk_internal()
    }

    /// Iterates `save_state` and `step` until offset >= final offset --
    /// i.e. moves train forward and extends path TPC until it reaches destination.
    pub fn walk_timed_path<P: AsRef<[LinkIdxTime]>, Q: AsRef<[Link]>>(
        &mut self,
        network: Q,
        timed_path: P,
    ) -> anyhow::Result<()> {
        let network = network.as_ref();
        let timed_path = timed_path.as_ref();
        if timed_path.is_empty() {
            bail!("Timed path cannot be empty!");
        }

        self.save_state();
        let mut idx_prev = 0;
        while idx_prev != timed_path.len() - 1 {
            let mut idx_next = idx_prev + 1;
            while idx_next + 1 < timed_path.len() - 1 && timed_path[idx_next].time < self.state.time
            {
                idx_next += 1;
            }
            let time_extend = timed_path[idx_next - 1].time;
            self.extend_path(
                network,
                &timed_path[idx_prev..idx_next]
                    .iter()
                    .map(|x| x.link_idx)
                    .collect::<Vec<LinkIdx>>(),
            )?;
            idx_prev = idx_next;
            while self.state.time < time_extend {
                self.step()?;
            }
        }

        self.walk_internal()
    }

    /// Sets power requirements based on:
    /// - rolling resistance
    /// - drag
    /// - inertia
    /// - target acceleration
    pub fn solve_required_pwr(&mut self) -> anyhow::Result<()> {
        let res_net = self.state.res_net();

        // Verify that train can slow down -- if `self.state.res_net()`, which
        // includes grade, is negative (negative `res_net` means the downgrade
        // is steep enough that the train is overcoming bearing, rolling,
        // drag, ... all resistive forces to accelerate downhill in the absence
        // of any braking), and if `self.state.res_net()` is negative and has
        // a higher magnitude than `self.fric_brake.force_max`, then the train
        // cannot slow down.
        // TODO: dial this back to just show `self.state` via debug print
        ensure!(
            self.fric_brake.force_max + self.state.res_net() > si::Force::ZERO,
            format!(
                "Insufficient braking force.\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
                format_dbg!(self.fric_brake.force_max + self.state.res_net() > si::Force::ZERO),
                format_dbg!(self.fric_brake.force_max),
                format_dbg!(self.state.res_net()),
                format_dbg!(self.state.res_grade),
                format_dbg!(self.state.grade_front),
                format_dbg!(self.state.grade_back),
                format_dbg!(self.state.elev_front),
                format_dbg!(self.state.offset),
                format_dbg!(self.state.offset_back),
                format_dbg!(self.state.speed),
                format_dbg!(self.state.speed_limit),
                format_dbg!(self.state.speed_target),
                format_dbg!(self.state.time),
                format_dbg!(self.state.dt),
                format_dbg!(self.state.i),
                format_dbg!(self.state.total_dist),
                format_dbg!(self.state.link_idx_front),
                format_dbg!(self.state.offset_in_link)
            )
        );

        // TODO: Validate that this makes sense considering friction brakes
        // this figures out when to start braking in advance of a speed limit
        // drop.  Takes into account air brake dynamics. I have not reviewed
        // this code, but that is my understanding.
        let (speed_limit, speed_target) = self.braking_points.calc_speeds(
            self.state.offset,
            self.state.speed,
            self.fric_brake.ramp_up_time * self.fric_brake.ramp_up_coeff,
        );
        self.state.speed_limit = speed_limit;
        self.state.speed_target = speed_target;

        let f_applied_target = res_net
            + self.state.mass_compound().with_context(|| format_dbg!())?
                * (speed_target - self.state.speed)
                / self.state.dt;

        // calculate the max positive tractive effort.  this is the same as set_speed_train_sim
        let pwr_pos_max = self.loco_con.state.pwr_out_max.min(si::Power::ZERO.max(
            // NOTE: the effect of rate may already be accounted for in this snippet
            // from fuel_converter.rs:

            // ```
            // self.state.pwr_out_max = (self.state.pwr_brake
            //     + (self.pwr_out_max / self.pwr_ramp_lag) * dt)
            //     .min(self.pwr_out_max)
            //     .max(self.pwr_out_max_init);
            // ```
            self.state.pwr_whl_out + self.loco_con.state.pwr_rate_out_max * self.state.dt,
        ));

        // calculate the max braking that a consist can apply
        let pwr_neg_max = self.loco_con.state.pwr_dyn_brake_max.max(si::Power::ZERO);
        ensure!(
            pwr_pos_max >= si::Power::ZERO,
            format_dbg!(pwr_pos_max >= si::Power::ZERO)
        );
        let time_per_mass =
            self.state.dt / self.state.mass_compound().with_context(|| format_dbg!())?;

        // Concept: calculate the final speed such that the worst case
        // (i.e. maximum) acceleration force does not exceed `power_max`
        // Base equation: m * (v_max - v_curr) / dt = p_max / v_max – f_res
        // figuring out how fast we can be be going by next timestep.
        let v_max = 0.5
            * (self.state.speed - res_net * time_per_mass
                + ((self.state.speed - res_net * time_per_mass)
                    * (self.state.speed - res_net * time_per_mass)
                    + 4.0 * time_per_mass * pwr_pos_max)
                    .sqrt());

        // Final v_max value should also be bounded by speed_target
        // maximum achievable positive tractive force
        let f_pos_max = self
            .loco_con
            .force_max()?
            .min(pwr_pos_max / speed_target.min(v_max));
        // Verify that train has sufficient power to move
        if self.state.speed < uc::MPH * 0.1 && f_pos_max <= res_net {
            bail!(
                "{}\nTrain does not have sufficient power to move!\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}", // ,\nlink={:?}
                format_dbg!(),
                // force_max
                format!(
                    "force_max: {} N", 
                    self
                        .loco_con
                        .force_max()?
                        .get::<si::newton>()
                        .format_eng(Some(5))
                    ),
                    // force based on speed target
                    format!("pwr_pos_max / speed_target.min(v_max): {} N", (pwr_pos_max / speed_target.min(v_max)).get::<si::newton>().format_eng(Some(5))),
                    // pwr_pos_max
                    format!("pwr_pos_max: {} W", pwr_pos_max.get::<si::watt>().format_eng(Some(5)),
                ),
                // SOC across all RES-equipped locomotives
                format!(
                    "SOCs: {:?}", 
                    self
                        .loco_con
                        .loco_vec
                        .iter()
                        .map(|loco| {
                            loco.reversible_energy_storage()
                                .map(|res| res.state.soc.get::<si::ratio>().format_eng(Some(5)))
                                .unwrap_or_else(|| "N/A".into())})
                        .collect::<Vec<String>>()
                ),
                // minimum allowable SOC across all RES-equipped locomotives
                format!(
                    "Minimum allowed SOCs: {:?}", 
                    self
                        .loco_con
                        .loco_vec
                        .iter()
                        .map(|loco| {
                            loco.reversible_energy_storage()
                                .map(|res| res.min_soc.get::<si::ratio>().format_eng(Some(5)))
                                .unwrap_or_else(|| "N/A".into())
                        })
                        .collect::<Vec<String>>()
                ),
                // grade at front of train
                format!("grade_front: {}", self.state.grade_front.get::<si::ratio>().format_eng(Some(5))),
                // grade at rear of train
                format!("grade_back: {}", self.state.grade_back.get::<si::ratio>().format_eng(Some(5))),
                format!("f_pos_max: {} N", f_pos_max.get::<si::newton>().format_eng(Some(5))),
                format!("res_net: {} N", res_net.get::<si::newton>().format_eng(Some(5))),
                format_dbg!(self.fric_brake.force_max),
                format_dbg!(self.state.res_grade),
                format_dbg!(self.state.elev_front),
                format_dbg!(self.state.elev_back),
                format_dbg!(self.state.offset),
                format_dbg!(self.state.offset_back),
                format_dbg!(self.state.speed),
                format_dbg!(self.state.speed_limit),
                format_dbg!(self.state.speed_target),
                format_dbg!(self.state.time),
                format_dbg!(self.state.dt),
                format_dbg!(self.state.i),
                format_dbg!(self.state.total_dist),
                format_dbg!(self.state.link_idx_front),
                format_dbg!(self.state.offset_in_link)
            )
        }

        // set the maximum friction braking force that is possible.
        self.fric_brake.set_cur_force_max_out(self.state.dt)?;

        // Transition speed between force and power limited negative traction
        // figure out the velocity where power and force limits coincide
        let v_neg_trac_lim: si::Velocity =
            self.loco_con.state.pwr_dyn_brake_max / self.loco_con.force_max()?;

        // TODO: Make sure that train handling rules for consist dynamic braking force limit is respected!
        // figure out how much dynamic braking can be used as regenerative
        // braking to recharge the [ReversibleEnergyStorage]
        let f_max_consist_regen_dyn = if self.state.speed > v_neg_trac_lim {
            // If there is enough braking to slow down at v_max
            let f_max_dyn_fast = self.loco_con.state.pwr_dyn_brake_max / v_max;
            if res_net + self.fric_brake.state.force_max_curr + f_max_dyn_fast >= si::Force::ZERO {
                self.loco_con.state.pwr_dyn_brake_max / v_max // self.state.speed
            } else {
                f_max_dyn_fast
            }
        } else {
            self.loco_con.force_max()?
        };

        // total impetus force applied to control train speed
        // calculating the applied drawbar force based on targets and enfrocing limits.
        let f_applied = f_pos_max.min(
            f_applied_target.max(-self.fric_brake.state.force_max_curr - f_max_consist_regen_dyn),
        );

        // physics......
        let vel_change = time_per_mass * (f_applied - res_net);
        let vel_avg = self.state.speed + 0.5 * vel_change;

        // updating states of the train.
        self.state.pwr_res = res_net * vel_avg;
        self.state.pwr_accel = self.state.mass_compound().with_context(|| format_dbg!())?
            / (2.0 * self.state.dt)
            * ((self.state.speed + vel_change) * (self.state.speed + vel_change)
                - self.state.speed * self.state.speed);

        self.state.time += self.state.dt;
        self.state.offset += self.state.dt * vel_avg;
        self.state.total_dist += (self.state.dt * vel_avg).abs();
        self.state.speed += vel_change;
        if utils::almost_eq_uom(&self.state.speed, &speed_target, None) {
            self.state.speed = speed_target;
        }

        let f_consist = if f_applied >= si::Force::ZERO {
            self.fric_brake.state.force = si::Force::ZERO;
            f_applied
        } else {
            let f_consist = f_applied + self.fric_brake.state.force;
            // If the friction brakes should be released, don't add power
            if f_consist >= si::Force::ZERO {
                si::Force::ZERO
            }
            // If the current friction brakes and consist regen + dyn can handle
            // things, don't add friction braking
            else if f_consist + f_max_consist_regen_dyn >= si::Force::ZERO {
                f_consist
            }
            // If the friction braking must increase, max out the regen dyn first
            else {
                self.fric_brake.state.force = -(f_applied + f_max_consist_regen_dyn);
                ensure!(
                    utils::almost_le_uom(
                        &self.fric_brake.state.force,
                        &self.fric_brake.state.force_max_curr,
                        None
                    ),
                    "Too much force requested from friction brake! Req={:?}, max={:?}",
                    self.fric_brake.state.force,
                    self.fric_brake.state.force_max_curr,
                );
                -f_max_consist_regen_dyn
            }
        };

        self.state.pwr_whl_out = f_consist * self.state.speed;

        // this allows for float rounding error overshoot
        ensure!(
            utils::almost_le_uom(&self.state.pwr_whl_out, &pwr_pos_max, Some(1.0e-7)),
            format!("{}\nPower wheel out is larger than max positive power! pwr_whl_out={:?}, pwr_pos_max={:?}",
            format_dbg!(utils::almost_le_uom(&self.state.pwr_whl_out, &pwr_pos_max, Some(1.0e-7))),
            self.state.pwr_whl_out,
            pwr_pos_max)
        );
        ensure!(
            utils::almost_le_uom(&-self.state.pwr_whl_out, &pwr_neg_max, Some(1.0e-7)),
            format!("{}\nPower wheel out is larger than max negative power! pwr_whl_out={:?}, pwr_neg_max={:?}
            {:?}\n{:?}\n{:?}\n{:?}",
            format_dbg!(utils::almost_le_uom(&-self.state.pwr_whl_out, &pwr_neg_max, Some(1.0e-7))),
            -self.state.pwr_whl_out,
            pwr_neg_max,
            self.state.speed,
            self.fric_brake.state.force * self.state.speed,
            vel_change,
        res_net)
        );
        self.state.pwr_whl_out = self.state.pwr_whl_out.max(-pwr_neg_max).min(pwr_pos_max);

        self.state.energy_whl_out += self.state.pwr_whl_out * self.state.dt;
        if self.state.pwr_whl_out >= 0. * uc::W {
            self.state.energy_whl_out_pos += self.state.pwr_whl_out * self.state.dt;
        } else {
            self.state.energy_whl_out_neg -= self.state.pwr_whl_out * self.state.dt;
        }

        Ok(())
    }

    fn recalc_braking_points(&mut self) -> anyhow::Result<()> {
        self.braking_points.recalc(
            &self.state,
            &self.fric_brake,
            &self.train_res,
            &self.path_tpc,
        )
    }
}

impl Init for SpeedLimitTrainSim {
    fn init(&mut self) -> Result<(), Error> {
        self.origs.init()?;
        self.dests.init()?;
        self.loco_con.init()?;
        self.state.init()?;
        self.train_res.init()?;
        self.path_tpc.init()?;
        self.braking_points.init()?;
        self.fric_brake.init()?;
        self.history.init()?;
        Ok(())
    }
}
impl SerdeAPI for SpeedLimitTrainSim {}
impl Default for SpeedLimitTrainSim {
    fn default() -> Self {
        let mut slts = Self {
            train_id: Default::default(),
            origs: Default::default(),
            dests: Default::default(),
            loco_con: Default::default(),
            n_cars_by_type: Default::default(),
            state: TrainState::valid(),
            train_res: TrainRes::valid(),
            path_tpc: PathTpc::default(),
            braking_points: Default::default(),
            fric_brake: Default::default(),
            history: Default::default(),
            save_interval: None,
            simulation_days: None,
            scenario_year: None,
        };
        slts.set_save_interval(None);
        slts.init().unwrap();
        slts
    }
}

impl Valid for SpeedLimitTrainSim {
    fn valid() -> Self {
        let mut train_sim = Self {
            path_tpc: PathTpc::valid(),
            ..Default::default()
        };
        train_sim.recalc_braking_points().unwrap();
        train_sim
    }
}

pub fn speed_limit_train_sim_fwd() -> SpeedLimitTrainSim {
    let mut speed_limit_train_sim = SpeedLimitTrainSim::valid();
    speed_limit_train_sim.path_tpc = PathTpc::new(TrainParams::valid());
    speed_limit_train_sim.origs = vec![
        Location {
            location_id: "Barstow".into(),
            offset: si::Length::ZERO,
            link_idx: LinkIdx::new(96),
            is_front_end: Default::default(),
            grid_emissions_region: "CAMXc".into(),
            electricity_price_region: "CA".into(),
            liquid_fuel_price_region: "CA".into(),
        },
        Location {
            location_id: "Barstow".into(),
            offset: si::Length::ZERO,
            link_idx: LinkIdx::new(634),
            is_front_end: Default::default(),
            grid_emissions_region: "CAMXc".into(),
            electricity_price_region: "CA".into(),
            liquid_fuel_price_region: "CA".into(),
        },
    ];
    speed_limit_train_sim.dests = vec![
        Location {
            location_id: "Stockton".into(),
            offset: si::Length::ZERO,
            link_idx: LinkIdx::new(288),
            is_front_end: Default::default(),
            grid_emissions_region: "CAMXc".into(),
            electricity_price_region: "CA".into(),
            liquid_fuel_price_region: "CA".into(),
        },
        Location {
            location_id: "Stockton".into(),
            offset: si::Length::ZERO,
            link_idx: LinkIdx::new(826),
            is_front_end: Default::default(),
            grid_emissions_region: "CAMXc".into(),
            electricity_price_region: "CA".into(),
            liquid_fuel_price_region: "CA".into(),
        },
    ];
    speed_limit_train_sim
}

pub fn speed_limit_train_sim_rev() -> SpeedLimitTrainSim {
    let mut sltsr = speed_limit_train_sim_fwd();
    std::mem::swap(&mut sltsr.origs, &mut sltsr.dests);
    sltsr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;

    // TODO: Add more SpeedLimitTrainSim cases
    impl Cases for SpeedLimitTrainSim {}

    #[test]
    fn test_speed_limit_train_sim() {
        let mut train_sim = SpeedLimitTrainSim::valid();
        train_sim.walk().unwrap();
    }
}
