use crate::imports::*;
use crate::train::TrainState;

// TODO implement method for elevation-dependent air density
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, SerdeAPI)]
#[altrios_api]
pub struct Basic {
    drag_area: si::Area,
}
impl Basic {
    pub fn new(drag_area: si::Area) -> Self {
        Self { drag_area }
    }

    /// Return instance of [Self] with customized total drag area
    /// # Arguments
    /// - `drag_area_single`: drag area (incl. drag coefficient) of a single, standalone railcar
    ///   that is not part of a train  
    /// - `drag_area_ratios`: vector of ratios for calculating total train drag area as
    ///   ratio-weighted sum
    pub fn from_drag_area_vec(
        drag_area_single: si::Area,
        drag_area_ratios: Vec<si::Ratio>,
    ) -> Self {
        let drag_area = drag_area_ratios
            .iter()
            .fold(0. * uc::M2, |acc, x| acc + drag_area_single * *x);
        Self { drag_area }
    }

    pub fn calc_res(&mut self, state: &TrainState) -> si::Force {
        self.drag_area * uc::rho_air() * state.speed * state.speed
    }
}
