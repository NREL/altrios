use crate::imports::*;
use crate::train::TrainState;

#[serde_api]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct Basic {
    davis_b: si::InverseVelocity,
}

#[pyo3_api]
impl Basic {}

impl Basic {
    pub fn new(davis_b: si::InverseVelocity) -> Self {
        Self { davis_b }
    }
    pub fn calc_res(&mut self, state: &TrainState) -> si::Force {
        self.davis_b * state.speed * state.weight_static
    }
}
