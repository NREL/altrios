use crate::imports::*;
use crate::track::PathTpc;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, HistoryVec)]
#[altrios_api(
    #[new]
    #[pyo3(signature = (
        time_seconds=None,
        offset_meters=None,
        speed_meters_per_second=None,
    ))]
    fn __new__(
        time_seconds: Option<f64>,
        offset_meters: Option<f64>,
        speed_meters_per_second: Option<f64>,
    ) -> Self {
        Self::new(
            time_seconds.map(|x| x * uc::S),
            offset_meters.map(|x| x * uc::M),
            speed_meters_per_second.map(|x| x * uc::MPS),
        )
    }
)]
/// For `SetSpeedTrainSim`, it is typically best to use the default for this.
pub struct InitTrainState {
    pub time: si::Time,
    pub offset: si::Length,
    pub speed: si::Velocity,
}

impl Init for InitTrainState {}
impl SerdeAPI for InitTrainState {}

impl Default for InitTrainState {
    fn default() -> Self {
        Self {
            time: si::Time::ZERO,
            offset: f64::NAN * uc::M,
            speed: si::Velocity::ZERO,
        }
    }
}

impl InitTrainState {
    pub fn new(
        time: Option<si::Time>,
        offset: Option<si::Length>,
        speed: Option<si::Velocity>,
    ) -> Self {
        let base = InitTrainState::default();
        Self {
            time: time.unwrap_or(base.time),
            offset: offset.unwrap_or(base.offset),
            speed: speed.unwrap_or(base.speed),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, HistoryVec, PartialEq)]
#[altrios_api(
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        length_meters,
        mass_static_kilograms,
        mass_rot_kilograms,
        mass_freight_kilograms,
        init_train_state=None,
    ))]
    fn __new__(
        length_meters: f64,
        mass_static_kilograms: f64,
        mass_rot_kilograms: f64,
        mass_freight_kilograms: f64,
        init_train_state: Option<InitTrainState>,
    ) -> Self {
        Self::new(
            length_meters * uc::M,
            mass_static_kilograms * uc::KG,
            mass_rot_kilograms * uc::KG,
            mass_freight_kilograms * uc::KG,
            init_train_state,
        )
    }

    #[getter("res_net")]
    fn res_net_py(&self) -> PyResult<f64> {
        Ok(self.res_net().get::<si::newton>())
    }
)]
pub struct TrainState {
    /// time since user-defined datum
    pub time: si::Time,
    /// index for time steps
    pub i: usize,
    /// Linear-along-track, directional distance of front of train from original
    /// starting position of back of train.
    ///
    /// If this is provided in [InitTrainState::new], it gets set as the train length or the value,
    /// whichever is larger, and if it is not provided, then it defaults to the train length.
    pub offset: si::Length,
    /// Linear-along-track, directional distance of back of train from original
    /// starting position of back of train.
    pub offset_back: si::Length,
    /// Linear-along-track, cumulative, absolute distance from initial starting position.
    pub total_dist: si::Length,
    /// Current link containing head end (i.e. pulling locomotives) of train
    pub link_idx_front: u32,
    /// Current link containing tail/back end of train
    pub link_idx_back: u32,
    /// Offset from start of current link
    pub offset_in_link: si::Length,
    /// Achieved speed based on consist capabilities and train resistance
    pub speed: si::Velocity,
    /// Speed limit
    pub speed_limit: si::Velocity,
    /// Speed target from meet-pass planner
    pub speed_target: si::Velocity,
    /// Time step size
    pub dt: si::Time,
    /// Train length
    pub length: si::Length,
    /// Static mass of train, including freight
    pub mass_static: si::Mass,
    /// Effective additional mass of train due to rotational inertia
    pub mass_rot: si::Mass,
    /// Mass of freight being hauled by the train (not including railcar empty weight)
    pub mass_freight: si::Mass,
    /// Static weight of train
    pub weight_static: si::Force,
    /// Rolling resistance force
    pub res_rolling: si::Force,
    /// Bearing resistance force
    pub res_bearing: si::Force,
    /// Davis B term resistance force
    pub res_davis_b: si::Force,
    /// Aerodynamic resistance force
    pub res_aero: si::Force,
    /// Grade resistance force
    pub res_grade: si::Force,
    /// Curvature resistance force
    pub res_curve: si::Force,

    /// Grade at front of train
    pub grade_front: si::Ratio,
    /// Grade at back of train of train if strap method is used
    pub grade_back: si::Ratio,
    /// Elevation at front of train
    pub elev_front: si::Length,
    /// Elevation at back of train
    pub elev_back: si::Length,

    /// Power to overcome train resistance forces
    pub pwr_res: si::Power,
    /// Power to overcome inertial forces
    pub pwr_accel: si::Power,
    /// Total tractive power exerted by locomotive consist
    pub pwr_whl_out: si::Power,
    /// Integral of [Self::pwr_whl_out]
    pub energy_whl_out: si::Energy,
    /// Energy out during positive or zero traction
    pub energy_whl_out_pos: si::Energy,
    /// Energy out during negative traction (positive value means negative traction)
    pub energy_whl_out_neg: si::Energy,
}

impl Init for TrainState {}
impl SerdeAPI for TrainState {}

impl Default for TrainState {
    fn default() -> Self {
        Self {
            time: Default::default(),
            i: 1,
            offset: Default::default(),
            offset_back: Default::default(),
            total_dist: si::Length::ZERO,
            link_idx_front: Default::default(),
            link_idx_back: Default::default(),
            offset_in_link: Default::default(),
            speed: Default::default(),
            speed_limit: Default::default(),
            dt: uc::S,
            length: Default::default(),
            mass_static: Default::default(),
            mass_rot: Default::default(),
            mass_freight: Default::default(),
            elev_front: Default::default(),
            elev_back: Default::default(),
            energy_whl_out: Default::default(),
            grade_front: Default::default(),
            grade_back: Default::default(),
            speed_target: Default::default(),
            weight_static: Default::default(),
            res_rolling: Default::default(),
            res_bearing: Default::default(),
            res_davis_b: Default::default(),
            res_aero: Default::default(),
            res_grade: Default::default(),
            res_curve: Default::default(),
            pwr_res: Default::default(),
            pwr_accel: Default::default(),
            pwr_whl_out: Default::default(),
            energy_whl_out_pos: Default::default(),
            energy_whl_out_neg: Default::default(),
        }
    }
}

impl Mass for TrainState {
    /// Static mass of train, not including effective rotational mass
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        self.derived_mass()
    }

    fn set_mass(
        &mut self,
        _new_mass: Option<si::Mass>,
        _side_effect: MassSideEffect,
    ) -> anyhow::Result<()> {
        bail!("`set_mass` is not enabled for `TrainState`")
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        Ok(Some(self.mass_static))
    }

    fn expunge_mass_fields(&mut self) {}
}

impl TrainState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        length: si::Length,
        mass_static: si::Mass,
        mass_rot: si::Mass,
        mass_freight: si::Mass,
        init_train_state: Option<InitTrainState>,
    ) -> Self {
        let init_train_state = init_train_state.unwrap_or_default();
        let offset = init_train_state.offset.max(length);
        Self {
            time: init_train_state.time,
            i: 1,
            offset,
            offset_back: offset - length,
            total_dist: si::Length::ZERO,
            speed: init_train_state.speed,
            // this needs to be set to something greater than or equal to actual speed and will be
            // updated after the first time step anyway
            speed_limit: init_train_state.speed,
            length,
            mass_static,
            mass_rot,
            mass_freight,
            ..Self::default()
        }
    }

    pub fn res_net(&self) -> si::Force {
        self.res_rolling
            + self.res_bearing
            + self.res_davis_b
            + self.res_aero
            + self.res_grade
            + self.res_curve
    }

    /// All base, freight, and rotational mass
    pub fn mass_compound(&self) -> anyhow::Result<si::Mass> {
        Ok(self
            .mass()
            .with_context(|| format_dbg!())?
            .with_context(|| format!("{}\nExpected `Some`", format_dbg!()))?
            + self.mass_rot)
    }
}

impl Valid for TrainState {
    fn valid() -> Self {
        Self {
            length: 2000.0 * uc::M,
            offset: 2000.0 * uc::M,
            offset_back: si::Length::ZERO,
            mass_static: 6000.0 * uc::TON,
            mass_rot: 200.0 * uc::TON,

            dt: uc::S,
            ..Self::default()
        }
    }
}

// TODO: Add new values!
impl ObjState for TrainState {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        si_chk_num_gtz_fin(&mut errors, &self.mass_static, "Mass static");
        si_chk_num_gtz_fin(&mut errors, &self.length, "Length");
        // si_chk_num_gtz_fin(&mut errors, &self.res_bearing, "Resistance bearing");
        // si_chk_num_fin(&mut errors, &self.res_davis_b, "Resistance Davis B");
        // si_chk_num_gtz_fin(&mut errors, &self.cd_area, "cd area");
        errors.make_err()
    }
}

/// Sets `link_idx_front` and `offset_in_link` based on `state` and `path_tpc`
///
/// Assumes that `offset` in `link_points()` is monotically increasing, which may not always be true.
pub fn set_link_and_offset(state: &mut TrainState, path_tpc: &PathTpc) -> anyhow::Result<()> {
    // index of current link within `path_tpc`
    // if the link_point.offset is greater than the train `state` offset, then
    // the train is in the previous link
    let idx_curr_link = path_tpc
        .link_points()
        .iter()
        .position(|&lp| lp.offset > state.offset)
        // if None, assume that it's the last element
        .unwrap_or_else(|| path_tpc.link_points().len())
        - 1;
    let link_point = path_tpc
        .link_points()
        .get(idx_curr_link)
        .with_context(|| format_dbg!())?;
    state.link_idx_front = link_point.link_idx.idx() as u32;
    state.offset_in_link = state.offset - link_point.offset;

    // link index of back of train at current time step
    let idx_back_link = path_tpc
        .link_points()
        .iter()
        .position(|&lp| lp.offset > state.offset_back)
        // if None, assume that it's the last element
        .unwrap_or_else(|| path_tpc.link_points().len())
        - 1;
    state.link_idx_back = path_tpc
        .link_points()
        .get(idx_back_link)
        .with_context(|| format_dbg!())?
        .link_idx
        .idx() as u32;

    Ok(())
}
