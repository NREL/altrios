use thiserror::Error;

#[derive(Debug, Error)]
pub enum AltriosError {
    #[error("SerdeAPI::Init failed: {0}")]
    Init(String),
    #[error("Simulation failed: {0}")]
    Simulation(String),
    #[error("{0}")]
    Other(String),
}

pub type AltriosResult<T> = Result<T, AltriosError>;
