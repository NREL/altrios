use crate::imports::*;
use crate::si;

#[derive(PartialEq)]
pub enum Dir {
    Unk,
    Fwd,
    Bwd,
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
    fn calc_idx(&self, offset: si::Length, idx: usize, dir: &Dir) -> anyhow::Result<usize>;
}

impl<T: GetOffset + core::fmt::Debug> LinSearchHint for &[T] {
    fn calc_idx(&self, offset: si::Length, mut idx: usize, dir: &Dir) -> anyhow::Result<usize> {
        if dir != &Dir::Bwd {
            ensure!(
                offset <= self.last().unwrap().get_offset(),
                "{}\nOffset in forward direction larger than last slice offset at idx: {}!",
                format_dbg!(),
                idx
            );
            while self[idx + 1].get_offset() < offset {
                idx += 1;
            }
        } else if dir != &Dir::Fwd {
            ensure!(
                self.first().unwrap().get_offset() <= offset,
                "{}\nOffset in reverse direction smaller than first slice offset at idx: {}!",
                format_dbg!(),
                idx
            );
            while offset < self[idx].get_offset() {
                idx -= 1;
            }
        }
        Ok(idx)
    }
}
