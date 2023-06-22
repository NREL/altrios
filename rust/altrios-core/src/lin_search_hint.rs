use crate::si;

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

pub trait GetOffset {
    fn get_offset(&self) -> si::Length;
}

pub trait LinSearchHint {
    fn calc_idx<const DIR: DirT>(&self, offset: si::Length, idx: usize) -> usize;
}

impl<T: GetOffset + core::fmt::Debug> LinSearchHint for &[T] {
    fn calc_idx<const DIR: DirT>(&self, offset: si::Length, mut idx: usize) -> usize {
        if DIR != Dir::Bwd {
            assert!(
                offset <= self.last().unwrap().get_offset(),
                "Offset larger than last slice offset! {self:?}"
            );
            while self[idx + 1].get_offset() < offset {
                idx += 1;
            }
        }
        if DIR != Dir::Fwd {
            assert!(
                self.first().unwrap().get_offset() <= offset,
                "Offset smaller than first slice offset! {self:?}"
            );
            while offset < self[idx].get_offset() {
                idx -= 1;
            }
        }
        idx
    }
}
