/// TODO: Geordie, provide doc string for `disp_structs`, just a short sentence or two explaining
/// what this module comprises
use super::{disp_imports::*, est_times::EstTime};

// TODO: Geordie, what's goin on with these TODOs?
// TODO:  Could possibly implement this as Option<NonZeroU16>, and...
pub type EstIdx = u32;
// TODO:  This as None
pub const EST_IDX_NA: EstIdx = 0;

/// Type of estimated time node
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, SerdeAPI)]
#[repr(u8)] // TODO: Geordie, could you explain what this does by replacing this comment?
pub enum EstType {
    /// Train arrives at entry point to link
    Arrive,
    /// Train clears entry point to link
    Clear,
    /// Fake node to handle multiple alternates
    #[default]
    Fake,
}

impl std::hash::Hash for EstType {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        hasher.write_u8(*self as u8);
    }
}
impl nohash_hasher::IsEnabled for EstType {}

/// Link index plus estimated time type (arrive, clear, or fake)
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, SerdeAPI)]
pub struct LinkEvent {
    pub link_idx: LinkIdx,
    pub est_type: EstType,
}

impl std::hash::Hash for LinkEvent {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        hasher.write_u64(self.link_idx.idx() as u64 + ((self.est_type as u64) << 32));
    }
}
impl nohash_hasher::IsEnabled for LinkEvent {}

pub type LinkEventMap = IntMap<LinkEvent, IntSet<EstIdx>>;

/// Dispatch node index.
pub type DispNodeIdx = Option<NonZeroU16>;

/// Train index.
pub type TrainIdx = Option<NonZeroU16>;

/// Dispatch Authority index.
///
/// Always tied to the same type as [TrainIdx].
pub type DispAuthIdx = TrainIdx;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub struct DispAuth {
    /// Arrive Entry Time
    pub arrive_entry: si::Time,
    /// Arrive Exit Time
    pub arrive_exit: si::Time,
    /// Clear Entry Time, TODO: Geordie, what does "Clear" mean here?
    pub clear_entry: si::Time,
    /// Clear Exit Time, TODO: Geordie, what does "Clear" mean here?
    pub clear_exit: si::Time,

    /// Offset (distance) between start of track link and front of train
    /// TODO: Geordie, is the above accurate?
    pub offset_front: si::Length,
    /// Offset (distance) between start of track link and front of train
    /// TODO: Geordie, is the above accurate?
    pub offset_back: si::Length,

    // this is a locally defined package so it's probably ok without a field-level doc string
    pub train_idx: TrainIdx,
}

impl DispAuth {
    pub fn train_idx_curr(&self) -> TrainIdx {
        if self.offset_back == f64::INFINITY * uc::M {
            None
        } else {
            self.train_idx
        }
    }
}

impl Default for DispAuth {
    /// TODO: Geordie, explain why `f64::INFINITY` is being used below.  
    fn default() -> Self {
        Self {
            arrive_entry: f64::INFINITY * uc::S,
            arrive_exit: f64::INFINITY * uc::S,
            clear_entry: f64::INFINITY * uc::S,
            clear_exit: f64::INFINITY * uc::S,
            offset_front: Default::default(),
            offset_back: Default::default(),
            train_idx: Default::default(),
        }
    }
}

type TrainIdxsViewIdx = u32;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, SerdeAPI)]
/// TODO: Geordie, put a doc string here
pub struct TrainIdxsView {
    pub idx_begin: TrainIdxsViewIdx,
    pub idx_end: TrainIdxsViewIdx,
}

impl TrainIdxsView {
    pub fn new(idx_begin: TrainIdxsViewIdx, idx_end: TrainIdxsViewIdx) -> Self {
        Self { idx_begin, idx_end }
    }
    pub fn is_empty(&self) -> bool {
        self.idx_begin == self.idx_end
    }
    pub fn len(&self) -> usize {
        (self.idx_end - self.idx_begin) as usize
    }
    pub fn range(&self) -> std::ops::Range<usize> {
        self.idx_begin as usize..self.idx_end as usize
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, SerdeAPI)]
/// TODO: Geordie, put a doc string here
pub struct EstTimeStatus {
    pub train_idxs_view: TrainIdxsView,
    pub link_idx: LinkIdx,
    pub est_type: EstType,
    pub is_on_path: bool,
}

impl EstTimeStatus {
    pub fn new(est_time: &EstTime) -> Self {
        Self {
            link_idx: est_time.link_event.link_idx,
            est_type: est_time.link_event.est_type,
            ..Default::default()
        }
    }

    pub fn is_blocked(&self) -> bool {
        self.train_idxs_view.idx_end != 0
    }
    pub fn unblock(&mut self) {
        self.train_idxs_view = TrainIdxsView::new(0, 0);
    }
    pub fn block_empty(&mut self) {
        self.train_idxs_view = TrainIdxsView::new(1, 1);
    }
    pub fn link_event(&self) -> LinkEvent {
        LinkEvent {
            link_idx: self.link_idx,
            est_type: self.est_type,
        }
    }
}

// TODO: Geordie, switch to using single train_idx per node with duplicate disp_node_idx
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq, SerdeAPI)]
/// TODO: Geordie, put a doc string here and adddress the todo above
pub struct DivergeNode {
    pub train_idx: TrainIdx,        // default to: None, which is 0.
    pub disp_node_idx: DispNodeIdx, // default to: None, which is 0.
}

impl DivergeNode {
    pub fn new(train_idx: TrainIdx, disp_node_idx: DispNodeIdx) -> Self {
        Self {
            train_idx,
            disp_node_idx,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, SerdeAPI)]
/// TODO: Geordie, put a doc string here.  What is `Disp` short for in this context?
pub struct DispNode {
    /// TODO: Geordie, what does this offset represent?   distance from beginning of link to front
    /// of train?
    pub offset: si::Length,
    pub time_pass: si::Time, // = units::numPosInf * units::s

    pub link_event: LinkEvent,
    pub est_idx: EstIdx,                  // = EST_TIME_IDX_NA;
    pub disp_auth_idx_entry: DispAuthIdx, // = DISP_AUTH_IDX_NA;
}

impl Default for DispNode {
    fn default() -> Self {
        Self {
            offset: Default::default(),
            time_pass: f64::INFINITY * uc::S,
            link_event: Default::default(),
            est_idx: EST_IDX_NA,
            disp_auth_idx_entry: None, // Could also use Default::default()
        }
    }
}
