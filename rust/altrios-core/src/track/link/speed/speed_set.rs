use super::speed_limit::*;
use super::speed_param::*;
use crate::imports::*;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, SerdeAPI, Hash)]
#[repr(u8)]
#[cfg_attr(feature = "pyo3", pyclass)]
/// Enum with variants representing train types
pub enum TrainType {
    #[default]
    None = 0,
    Freight = 1,
    Passenger = 2,
    Intermodal = 3,
    HighSpeedPassenger = 4,
    TiltTrain = 5,
    Commuter = 6,
}

impl Valid for TrainType {
    fn valid() -> Self {
        TrainType::Freight
    }
}

impl ObjState for TrainType {
    fn is_fake(&self) -> bool {
        *self == TrainType::None
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, SerdeAPI)]
#[altrios_api]
pub struct SpeedSet {
    pub speed_limits: Vec<SpeedLimit>,
    #[api(skip_get, skip_set)]
    pub speed_params: Vec<SpeedParam>,
    pub is_head_end: bool,
}


#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, SerdeAPI)]
#[altrios_api]
/// Helper struct to create [SpeedSet] from deprecated data format
pub struct OldSpeedSet {
    pub speed_limits: Vec<SpeedLimit>,
    #[api(skip_get, skip_set)]
    pub speed_params: Vec<SpeedParam>,
    pub is_head_end: bool,
}


impl Valid for SpeedSet {
    fn valid() -> Self {
        Self {
            speed_limits: Vec::<SpeedLimit>::valid(),
            speed_params: Vec::<SpeedParam>::valid(),
            is_head_end: false,
        }
    }
}

impl ObjState for &SpeedSet {
    fn is_fake(&self) -> bool {
        self.speed_limits.is_empty()
    }
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        if self.is_fake() {
            validate_field_fake(&mut errors, &self.speed_limits, "Speed limits");

            if !self.speed_params.is_empty() {
                errors.push(anyhow!("Speed params must be empty!"));
            }
            if self.is_head_end {
                errors.push(anyhow!("Is head end must be false!"));
            }
        } else {
            validate_field_real(&mut errors, &self.speed_limits, "Speed limits");
            validate_field_real(&mut errors, &self.speed_params, "Speed params");
        }

        errors.make_err()
    }
}

impl ObjState for SpeedSet {
    fn is_fake(&self) -> bool {
        self.speed_limits.is_empty()
    }
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        if self.is_fake() {
            validate_field_fake(&mut errors, &self.speed_limits, "Speed limits");

            if !self.speed_params.is_empty() {
                errors.push(anyhow!("Speed params must be empty!"));
            }
            if self.is_head_end {
                errors.push(anyhow!("Is head end must be false!"));
            }
        } else {
            validate_field_real(&mut errors, &self.speed_limits, "Speed limits");
            validate_field_real(&mut errors, &self.speed_params, "Speed params");
        }

        errors.make_err()
    }
}

impl Valid for HashMap<TrainType, SpeedSet> {
    fn valid() -> Self {
        HashMap::from([(TrainType::valid(), SpeedSet::valid())])
    }
}

impl ObjState for HashMap<TrainType, SpeedSet> {
    fn is_fake(&self) -> bool {
        self.is_empty()
    }
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        validate_slice_real(
            &mut errors,
            &self.values().collect::<Vec<&SpeedSet>>(),
            "Speed set",
        );
        errors.make_err()
    }
}

#[cfg(test)]
mod test_train_type {
    use super::*;
    use crate::testing::*;
    impl Cases for TrainType {
        fn fake_cases() -> Vec<Self> {
            vec![Self::default()]
        }
    }
    check_cases!(TrainType);
}

#[cfg(test)]
mod test_speed_set {
    use super::*;
    use crate::testing::*;

    impl Cases for SpeedSet {
        fn real_cases() -> Vec<Self> {
            vec![
                Self::valid(),
                Self {
                    speed_params: Vec::<SpeedParam>::valid(),
                    ..Self::valid()
                },
            ]
        }
    }
    check_cases!(SpeedParam);
}

#[cfg(test)]
mod test_speed_sets {
    use super::*;
    use crate::testing::*;

    impl Cases for HashMap<TrainType, SpeedSet> {
        fn fake_cases() -> Vec<Self> {
            vec![HashMap::new()]
        }
    }
    check_cases!(HashMap<TrainType, SpeedSet>);
}
