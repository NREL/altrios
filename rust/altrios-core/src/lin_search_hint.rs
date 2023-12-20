use crate::imports::*;
use crate::si;

// Per Geordie, we could probably get rid of this `DirT` code
pub type DirT = u8;

#[allow(non_snake_case)]
pub mod Dir {
    pub use super::*;
    #[allow(non_upper_case_globals)]
    pub const Unk: DirT = 0;
    #[allow(non_upper_case_globals)]
    pub const Fwd: DirT = 1;
    #[allow(non_upper_case_globals)]
    pub const Bwd: DirT = 2;
}

/// Has method that returns offset from start of PathTpc
pub trait GetOffset {
    /// Returns offset from start of PathTpc
    fn get_offset(&self) -> si::Length;
}

/// Contains method to calculate the index immediately before `offset` given the previous calculated
/// index, `idx`, and a direction `DirT`.
pub trait LinSearchHint {
    /// Calculate the index immediately before `offset` given the previous calculated index, `idx`,
    /// and a direction `DirT`.
    fn calc_idx<const DIR: DirT>(&self, offset: si::Length, idx: usize) -> anyhow::Result<usize>;
}

impl<T: GetOffset + core::fmt::Debug> LinSearchHint for &[T] {
    fn calc_idx<const DIR: DirT>(
        &self,
        offset: si::Length,
        mut idx: usize,
    ) -> anyhow::Result<usize> {
        if DIR != Dir::Bwd {
            ensure!(
                offset <= self.last().unwrap().get_offset(),
                "{}\nOffset larger than last slice offset!",
                format_dbg!()
            );
            while self[idx + 1].get_offset() < offset {
                idx += 1;
            }
        }
        if DIR != Dir::Fwd {
            ensure!(
                self.first().unwrap().get_offset() <= offset,
                "{}\nOffset smaller than first slice offset!",
                format_dbg!()
            );
            while offset < self[idx].get_offset() {
                idx -= 1;
            }
        }
        Ok(idx)
    }
}
