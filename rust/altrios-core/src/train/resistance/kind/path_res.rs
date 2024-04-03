use crate::imports::*;
use crate::track::PathResCoeff;
use crate::train::TrainState;

fn calc_res_val(res_coeff: si::Ratio, state: &TrainState) -> si::Force {
    res_coeff * state.weight_static
}

#[altrios_api]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, SerdeAPI)]
pub struct Point {
    idx: usize,
}
impl Point {
    pub fn new(vals: &[PathResCoeff], state: &TrainState) -> anyhow::Result<Self> {
        Ok(Self {
            idx: vals.calc_idx(state.offset - state.length * 0.5, 0, &Dir::Fwd)?,
        })
    }

    pub fn calc_res(
        &mut self,
        vals: &[PathResCoeff],
        state: &TrainState,
        dir: &Dir,
    ) -> anyhow::Result<si::Force> {
        self.idx = vals.calc_idx(state.offset - state.length * 0.5, self.idx, dir)?;
        Ok(calc_res_val(vals[self.idx].res_coeff, state))
    }

    pub fn res_coeff_front(&self, vals: &[PathResCoeff]) -> si::Ratio {
        vals[self.idx].res_coeff
    }

    pub fn res_net_front(&self, vals: &[PathResCoeff], state: &TrainState) -> si::Length {
        vals[self.idx].calc_res_val(state.offset)
    }

    /// Returns index of current element containing front of train within `PathTPC`
    pub fn path_tpc_idx_front(&self) -> usize {
        self.idx
    }

    pub fn fix_cache(&mut self, idx_sub: usize) {
        self.idx -= idx_sub;
    }
}

#[ext(CalcResStrap)]
impl [PathResCoeff] {
    fn calc_res_strap(&self, idx_front: usize, idx_back: usize, state: &TrainState) -> si::Ratio {
        debug_assert!(state.length > si::Length::ZERO);
        (self[idx_front].calc_res_val(state.offset)
            - self[idx_back].calc_res_val(state.offset_back))
            / state.length
    }
}

#[altrios_api]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, SerdeAPI)]
pub struct Strap {
    idx_front: usize,
    idx_back: usize,
}

impl Strap {
    pub fn new(vals: &[PathResCoeff], state: &TrainState) -> anyhow::Result<Self> {
        if vals.len() <= 1 {
            Ok(Self {
                idx_front: 0,
                idx_back: 0,
            })
        } else {
            let idx_back = vals.calc_idx(state.offset - state.length, 0, &Dir::Fwd)?;
            Ok(Self {
                idx_back,
                idx_front: vals.calc_idx(state.offset, idx_back, &Dir::Fwd)?,
            })
        }
    }
    pub fn calc_res(
        &mut self,
        vals: &[PathResCoeff],
        state: &TrainState,
        dir: &Dir,
    ) -> anyhow::Result<si::Force> {
        match dir {
            Dir::Fwd => {
                self.idx_front = vals.calc_idx(state.offset, self.idx_front, dir)?;
            }
            Dir::Bwd => {
                self.idx_back = vals.calc_idx(state.offset_back, self.idx_back, dir)?;
            }
            Dir::Unk => {
                self.idx_front = vals.calc_idx(state.offset, self.idx_front, dir)?;
                self.idx_back = vals.calc_idx(state.offset_back, self.idx_back, dir)?;
            }
        }

        let res_coeff = if self.idx_front == self.idx_back {
            vals[self.idx_front].res_coeff
        } else {
            match dir {
                Dir::Fwd => {
                    self.idx_back = vals.calc_idx(state.offset_back, self.idx_back, dir)?;
                }
                Dir::Bwd => {
                    self.idx_front = vals.calc_idx(state.offset, self.idx_front, dir)?;
                }
                _ => {}
            }
            vals.calc_res_strap(self.idx_front, self.idx_back, state)
        };

        Ok(calc_res_val(res_coeff, state))
    }
    pub fn res_coeff_front(&self, vals: &[PathResCoeff]) -> si::Ratio {
        vals[self.idx_front].res_coeff
    }

    pub fn res_net_front(&self, vals: &[PathResCoeff], state: &TrainState) -> si::Length {
        vals[self.idx_front].calc_res_val(state.offset)
    }

    /// Returns index of current element containing front of train within `PathTPC`
    pub fn path_tpc_idx_front(&self) -> usize {
        self.idx_front
    }

    pub fn fix_cache(&mut self, idx_sub: usize) {
        self.idx_back -= idx_sub;
        self.idx_front -= idx_sub;
    }
}
