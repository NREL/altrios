use super::train_imports::*;

#[altrios_api(
    #[new]
    fn __new__(
        time_seconds: Vec<f64>,
        speed_meters_per_second: Vec<f64>,
        engine_on: Option<Vec<bool>>
    ) -> anyhow::Result<Self> {
        Ok(Self::new(time_seconds, speed_meters_per_second, engine_on))
    }

    #[staticmethod]
    #[pyo3(name = "from_csv_file")]
    fn from_csv_file_py(filepath: &PyAny) -> anyhow::Result<Self> {
        Self::from_csv_file(PathBuf::extract(filepath)?)
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    #[pyo3(name = "to_csv_file")]
    fn to_csv_file_py(&self, filepath: &PyAny) -> anyhow::Result<()> {
        self.to_csv_file(PathBuf::extract(filepath)?)
    }
)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, SerdeAPI)]
pub struct SpeedTrace {
    /// simulation time
    pub time: Vec<si::Time>,
    /// simulation speed
    pub speed: Vec<si::Velocity>,
    /// Whether engine is on
    pub engine_on: Option<Vec<bool>>,
}

impl SpeedTrace {
    pub fn new(time_s: Vec<f64>, speed_mps: Vec<f64>, engine_on: Option<Vec<bool>>) -> Self {
        SpeedTrace {
            time: time_s.iter().map(|x| uc::S * (*x)).collect(),
            speed: speed_mps.iter().map(|x| uc::MPS * (*x)).collect(),
            engine_on,
        }
    }

    pub fn trim(&mut self, start_idx: Option<usize>, end_idx: Option<usize>) -> anyhow::Result<()> {
        let start_idx = start_idx.unwrap_or(0);
        let end_idx = end_idx.unwrap_or(self.len());
        ensure!(end_idx <= self.len(), format_dbg!(end_idx <= self.len()));

        self.time = self.time[start_idx..end_idx].to_vec();
        self.speed = self.speed[start_idx..end_idx].to_vec();
        self.engine_on = self
            .engine_on
            .as_ref()
            .map(|eo| eo[start_idx..end_idx].to_vec());
        Ok(())
    }

    pub fn dt(&self, i: usize) -> si::Time {
        self.time[i] - self.time[i - 1]
    }

    pub fn mean(&self, i: usize) -> si::Velocity {
        0.5 * (self.speed[i] + self.speed[i - 1])
    }

    pub fn acc(&self, i: usize) -> si::Acceleration {
        (self.speed[i] - self.speed[i - 1]) / self.dt(i)
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    /// method to prevent rust-analyzer from complaining
    pub fn is_empty(&self) -> bool {
        self.time.is_empty() && self.speed.is_empty() && self.engine_on.is_none()
    }

    pub fn push(&mut self, speed_element: SpeedTraceElement) -> anyhow::Result<()> {
        self.time.push(speed_element.time);
        self.speed.push(speed_element.speed);
        self.engine_on
            .as_mut()
            .map(|eo| match speed_element.engine_on {
                Some(seeeo) => {
                    eo.push(seeeo);
                    Ok(())
                }
                None => bail!(
                    "`engine_one` in `SpeedTraceElement` and `SpeedTrace` must both have same option variant."),
            });
        Ok(())
    }

    pub fn empty() -> Self {
        Self {
            time: Vec::new(),
            speed: Vec::new(),
            engine_on: None,
        }
    }

    /// Load speed trace from csv file
    pub fn from_csv_file<P: AsRef<Path>>(filepath: P) -> anyhow::Result<Self> {
        let filepath = filepath.as_ref();

        // create empty SpeedTrace to be populated
        let mut st = Self::empty();

        let file = File::open(filepath)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        for result in rdr.deserialize() {
            let st_elem: SpeedTraceElement = result?;
            st.push(st_elem)?;
        }
        ensure!(
            !st.is_empty(),
            "Invalid SpeedTrace file {:?}; SpeedTrace is empty",
            filepath
        );
        Ok(st)
    }

    /// Save speed trace to csv file
    pub fn to_csv_file<P: AsRef<Path>>(&self, filepath: P) -> anyhow::Result<()> {
        let file = std::fs::OpenOptions::new().write(true).open(filepath)?;
        let mut wrtr = csv::WriterBuilder::new()
            .has_headers(true)
            .from_writer(file);
        let engine_on: Vec<Option<bool>> = match &self.engine_on {
            Some(eo_vec) => eo_vec
                .iter()
                .map(|eo| Some(*eo))
                .collect::<Vec<Option<bool>>>(),
            None => vec![None; self.len()],
        };
        for ((time, speed), engine_on) in self.time.iter().zip(&self.speed).zip(engine_on) {
            wrtr.serialize(SpeedTraceElement {
                time: *time,
                speed: *speed,
                engine_on,
            })?;
        }
        wrtr.flush()?;
        Ok(())
    }
}

impl Default for SpeedTrace {
    fn default() -> Self {
        let mut speed_mps: Vec<f64> = Vec::linspace(0.0, 20.0, 800);
        speed_mps.append(&mut [20.0; 100].to_vec());
        speed_mps.append(&mut Vec::linspace(20.0, 0.0, 200));
        speed_mps.push(0.0);
        let time_s: Vec<f64> = (0..speed_mps.len()).map(|x| x as f64).collect();
        Self::new(time_s, speed_mps, None)
    }
}

/// Element of [SpeedTrace].  Used for vec-like operations.
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub struct SpeedTraceElement {
    /// simulation time
    #[serde(alias = "time_seconds")]
    time: si::Time,
    /// prescribed speed
    #[serde(alias = "speed_meters_per_second")]
    speed: si::Velocity,
    /// whether engine is on
    engine_on: Option<bool>,
}

#[altrios_api(
    // TODO: consider whether this method should exist after verifying that it is not used anywhere
    // and should be superseded by `make_set_speed_train_sim`
    #[new]
    fn __new__(
        loco_con: Consist,
        state: TrainState,
        speed_trace: SpeedTrace,
        train_res_file: Option<String>,
        path_tpc_file: Option<String>,
        save_interval: Option<usize>,
    ) -> Self {
        let path_tpc = match path_tpc_file {
            Some(file) => PathTpc::from_file(file).unwrap(),
            None => PathTpc::valid()
        };
        let train_res = match train_res_file {
            Some(file) => TrainRes::from_file(file).unwrap(),
            None => TrainRes::valid()
        };

        Self::new(loco_con, state, speed_trace, train_res, path_tpc, save_interval)
    }

    #[setter]
    pub fn set_res_strap(&mut self, res_strap: method::Strap) -> anyhow::Result<()> {
        self.train_res = TrainRes::Strap(res_strap);
        Ok(())
    }

    #[setter]
    pub fn set_res_point(&mut self, res_point: method::Point) -> anyhow::Result<()> {
        self.train_res = TrainRes::Point(res_point);
        Ok(())
    }

    #[getter]
    pub fn get_res_strap(&self) -> anyhow::Result<Option<method::Strap>> {
        match &self.train_res {
            TrainRes::Strap(strap) => Ok(Some(strap.clone())),
            _ => Ok(None),
        }
    }

    #[getter]
    pub fn get_res_point(&self) -> anyhow::Result<Option<method::Point>> {
        match &self.train_res {
            TrainRes::Point(point) => Ok(Some(point.clone())),
            _ => Ok(None),
        }
    }

    #[pyo3(name = "walk")]
    /// Exposes `walk` to Python.
    fn walk_py(&mut self) -> anyhow::Result<()> {
        self.walk()
    }

    #[pyo3(name = "step")]
    fn step_py(&mut self) -> anyhow::Result<()> {
        self.step()
    }

    #[pyo3(name = "set_save_interval")]
    /// Set save interval and cascade to nested components.
    fn set_save_interval_py(&mut self, save_interval: Option<usize>) {
        self.set_save_interval(save_interval);
    }

    #[pyo3(name = "get_save_interval")]
    fn get_save_interval_py(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.get_save_interval())
    }

    #[pyo3(name = "trim_failed_steps")]
    fn trim_failed_steps_py(&mut self) -> anyhow::Result<()> {
        self.trim_failed_steps()?;
        Ok(())
    }
)]
#[derive(Clone, Debug, Serialize, Deserialize, SerdeAPI)]
/// Train simulation in which speed is prescribed
pub struct SetSpeedTrainSim {
    pub loco_con: Consist,
    pub state: TrainState,
    pub speed_trace: SpeedTrace,
    #[api(skip_get, skip_set)]
    /// train resistance calculation
    pub train_res: TrainRes,
    #[api(skip_get, skip_set)]
    path_tpc: PathTpc,
    #[serde(default)]
    /// Custom vector of [Self::state]
    pub history: TrainStateHistoryVec,
    #[api(skip_set, skip_get)]
    save_interval: Option<usize>,
}

impl SetSpeedTrainSim {
    pub fn new(
        loco_con: Consist,
        state: TrainState,
        speed_trace: SpeedTrace,
        train_res: TrainRes,
        path_tpc: PathTpc,
        save_interval: Option<usize>,
    ) -> Self {
        let mut train_sim = Self {
            loco_con,
            state,
            train_res,
            path_tpc,
            speed_trace,
            history: Default::default(),
            save_interval,
        };
        train_sim.set_save_interval(save_interval);

        train_sim
    }

    /// Trims off any portion of the trip that failed to run
    pub fn trim_failed_steps(&mut self) -> anyhow::Result<()> {
        if self.state.i <= 1 {
            bail!("`walk` method has not proceeded past first time step.")
        }
        self.speed_trace.trim(None, Some(self.state.i))?;

        Ok(())
    }

    /// Sets `save_interval` for self and nested `loco_con`.
    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.save_interval = save_interval;
        self.loco_con.set_save_interval(save_interval);
    }

    /// Returns `self.save_interval` and asserts that this is equal
    /// to `self.loco_con.get_save_interval()`.
    pub fn get_save_interval(&self) -> Option<usize> {
        // this ensures that save interval has been propagated
        assert_eq!(self.save_interval, self.loco_con.get_save_interval());
        self.save_interval
    }

    /// Solves step, saves state, steps nested `loco_con`, and increments `self.i`.
    pub fn step(&mut self) -> anyhow::Result<()> {
        self.solve_step()
            .map_err(|err| err.context(format!("time step: {}", self.state.i)))?;
        self.save_state();
        self.loco_con.step();
        self.state.i += 1;
        Ok(())
    }

    /// Solves time step.
    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        ensure!(
            self.speed_trace.speed[self.state.i] >= si::Velocity::ZERO,
            format_dbg!(self.speed_trace.speed[self.state.i] >= si::Velocity::ZERO)
        );
        self.loco_con
            .set_cat_power_limit(&self.path_tpc, self.state.offset);

        self.loco_con.set_pwr_aux(Some(true))?;
        self.loco_con
            .set_cur_pwr_max_out(None, self.speed_trace.dt(self.state.i))?;
        self.train_res
            .update_res(&mut self.state, &self.path_tpc, &Dir::Fwd)?;
        self.solve_required_pwr(self.speed_trace.dt(self.state.i));
        self.loco_con.solve_energy_consumption(
            self.state.pwr_whl_out,
            self.speed_trace.dt(self.state.i),
            Some(true),
        )?;

        self.state.time = self.speed_trace.time[self.state.i];
        self.state.speed = self.speed_trace.speed[self.state.i];
        self.state.offset += self.speed_trace.mean(self.state.i) * self.state.dt;
        set_link_and_offset(&mut self.state, &self.path_tpc)?;
        self.state.total_dist += (self.speed_trace.mean(self.state.i) * self.state.dt).abs();
        Ok(())
    }

    /// Saves current time step for self and nested `loco_con`.
    fn save_state(&mut self) {
        if let Some(interval) = self.save_interval {
            if self.state.i % interval == 0 || 1 == self.state.i {
                self.history.push(self.state);
                self.loco_con.save_state();
            }
        }
    }

    /// Iterates `save_state` and `step` through all time steps.
    pub fn walk(&mut self) -> anyhow::Result<()> {
        self.save_state();
        while self.state.i < self.speed_trace.len() {
            self.step()?;
        }
        Ok(())
    }

    /// Sets power requirements based on:
    /// - rolling resistance
    /// - drag
    /// - inertia
    /// - acceleration
    /// (some of these aren't implemented yet)
    pub fn solve_required_pwr(&mut self, dt: si::Time) {
        self.state.pwr_res = self.state.res_net() * self.speed_trace.mean(self.state.i);
        self.state.pwr_accel = self.state.mass_adj / (2.0 * self.speed_trace.dt(self.state.i))
            * (self.speed_trace.speed[self.state.i].powi(typenum::P2::new())
                - self.speed_trace.speed[self.state.i - 1].powi(typenum::P2::new()));
        self.state.dt = self.speed_trace.dt(self.state.i);

        self.state.pwr_whl_out = self.state.pwr_accel + self.state.pwr_res;
        self.state.energy_whl_out += self.state.pwr_whl_out * dt;
        if self.state.pwr_whl_out >= 0. * uc::W {
            self.state.energy_whl_out_pos += self.state.pwr_whl_out * dt;
        } else {
            self.state.energy_whl_out_neg -= self.state.pwr_whl_out * dt;
        }
    }
}

impl Default for SetSpeedTrainSim {
    fn default() -> Self {
        Self {
            loco_con: Consist::default(),
            state: TrainState::valid(),
            train_res: TrainRes::valid(),
            path_tpc: PathTpc::valid(),
            speed_trace: SpeedTrace::default(),
            history: TrainStateHistoryVec::default(),
            save_interval: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SetSpeedTrainSim;

    #[test]
    fn test_set_speed_train_sim() {
        let mut train_sim = SetSpeedTrainSim::default();
        train_sim.walk().unwrap();
        assert!(train_sim.loco_con.state.i > 1);
    }
}
