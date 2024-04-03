use crate::imports::*;
use crate::track::PathTpc;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, HistoryVec)]
#[altrios_api(
    #[new]
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
    fn __new__(
        length_meters: f64,
        mass_static_kilograms: f64,
        mass_adj_kilograms: f64,
        mass_freight_kilograms: f64,
        init_train_state: Option<InitTrainState>,
    ) -> Self {
        Self::new(
            length_meters * uc::M,
            mass_static_kilograms * uc::KG,
            mass_adj_kilograms * uc::KG,
            mass_freight_kilograms * uc::KG,
            init_train_state,
        )
    }
)]
pub struct TrainState {
    /// time since user-defined datum
    pub time: si::Time,
    /// index for time steps
    pub i: usize,
    /// Linear-along-track, directional distance from initial starting position.
    ///
    /// If this is provided in [InitTrainState::new], it gets set as the train length or the value,
    /// whichever is larger, and if it is not provided, then it defaults to the train length.
    pub offset: si::Length,
    pub offset_back: si::Length,
    /// Linear-along-track, cumulative, absolute distance from initial starting position.
    pub total_dist: si::Length,
    /// Current link containing head end (i.e. pulling locomotives) of train
    pub link_idx_front: u32,
    /// Offset from start of current link
    pub offset_in_link: si::Length,
    /// Achieved speed based on consist capabilities and train resistance
    pub speed: si::Velocity,
    /// Speed limit
    pub speed_limit: si::Velocity,
    /// Speed target from meet-pass planner
    pub speed_target: si::Velocity,
    pub dt: si::Time,
    pub length: si::Length,
    pub mass_static: si::Mass,
    pub mass_adj: si::Mass,
    pub mass_freight: si::Mass,
    pub weight_static: si::Force,
    pub res_rolling: si::Force,
    pub res_bearing: si::Force,
    pub res_davis_b: si::Force,
    pub res_aero: si::Force,
    pub res_grade: si::Force,
    pub res_curve: si::Force,

    /// Grade at front of train
    pub grade_front: si::Ratio,
    /// Elevation at front of train
    pub elev_front: si::Length,

    /// Power to overcome train resistance forces
    pub pwr_res: si::Power,
    /// Power to overcome inertial forces
    pub pwr_accel: si::Power,

    pub pwr_whl_out: si::Power,
    pub energy_whl_out: si::Energy,
    /// Energy out during positive or zero traction
    pub energy_whl_out_pos: si::Energy,
    /// Energy out during negative traction (positive value means negative traction)
    pub energy_whl_out_neg: si::Energy,
}

impl Default for TrainState {
    fn default() -> Self {
        Self {
            time: Default::default(),
            i: 1,
            offset: Default::default(),
            offset_back: Default::default(),
            total_dist: si::Length::ZERO,
            link_idx_front: Default::default(),
            offset_in_link: Default::default(),
            speed: Default::default(),
            speed_limit: Default::default(),
            dt: uc::S,
            length: Default::default(),
            mass_static: Default::default(),
            mass_adj: Default::default(),
            mass_freight: Default::default(),
            elev_front: Default::default(),
            energy_whl_out: Default::default(),
            grade_front: Default::default(),
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

impl TrainState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        length: si::Length,
        mass_static: si::Mass,
        mass_adj: si::Mass,
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
            mass_adj,
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
}

impl Valid for TrainState {
    fn valid() -> Self {
        Self {
            length: 2000.0 * uc::M,
            offset: 2000.0 * uc::M,
            offset_back: si::Length::ZERO,
            mass_static: 6000.0 * uc::TON,
            mass_adj: 6200.0 * uc::TON,

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
        // si_chk_num_gtz_fin(&mut errors, &self.drag_area, "Drag area");
        errors.make_err()
    }
}

/// Sets `link_idx_front` and `offset_in_link` based on `state` and `path_tpc`
///
/// Assumes that `offset` in `link_points()` is monotically increasing, which may not always be true.
pub fn set_link_and_offset(state: &mut TrainState, path_tpc: &PathTpc) -> anyhow::Result<()> {
    let idx_curr_link = path_tpc
        .link_points()
        .iter()
        .position(|&lp| lp.offset >= state.offset)
        // if None, assume that it's the last element
        .unwrap_or_else(|| path_tpc.link_points().len())
        - 1;
    state.link_idx_front = path_tpc
        .link_points()
        .get(idx_curr_link)
        .with_context(|| format_dbg!())?
        .link_idx
        .idx() as u32;
    state.offset_in_link = state.offset
        - path_tpc
            .link_points()
            .get(idx_curr_link)
            .with_context(|| format_dbg!())?
            .offset;

    Ok(())
}
