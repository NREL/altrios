use super::speed_limit::*;
use super::speed_param::*;
use crate::imports::*;
use std::collections::HashMap;

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

#[derive(Debug, Default, Clone, PartialEq, SerdeAPI)]
/// helper struct to allow for serde deserialize flexibility (i.e. backward compatibility) for
/// `Network` generation methods
pub(crate) struct SpeedSetWrapper(pub HashMap<TrainType, SpeedSet>);

impl<'de> Deserialize<'de> for SpeedSetWrapper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // let speed_set_wrapper = match HashMap::deserialize(deserializer) {
        //     Ok(hm) => SpeedSetWrapper(hm), // first method succeeded
        //     Err(err) => {
        //         // try next method
        //         let v = Vec::deserialize(deserializer)?;
        //         SpeedSetWrapper::from(OldSpeedSetWrapper(v))
        //     }
        // };

        // Ok(speed_set_wrapper)

        Ok(SpeedSetWrapper(HashMap::deserialize(deserializer)?))
    }
}

impl Serialize for SpeedSetWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        HashMap::serialize(&self.0, serializer)
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, SerdeAPI)]
/// helper struct to allow for serde deserialize flexibility (i.e. backward compatibility) for
/// `Network` generation methods
pub(crate) struct OldSpeedSetWrapper(pub Vec<OldSpeedSet>);

impl From<OldSpeedSetWrapper> for SpeedSetWrapper {
    fn from(value: OldSpeedSetWrapper) -> Self {
        let mut hm: HashMap<TrainType, SpeedSet> = HashMap::new();
        for x in value.0 {
            hm.insert(
                x.train_type,
                SpeedSet {
                    speed_limits: x.speed_limits,
                    speed_params: x.speed_params,
                    is_head_end: x.is_head_end,
                },
            );
        }
        SpeedSetWrapper(hm)
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, SerdeAPI)]
#[altrios_api]
/// Helper struct to create [SpeedSet] from deprecated data format
pub struct OldSpeedSet {
    pub speed_limits: Vec<SpeedLimit>,
    #[api(skip_get, skip_set)]
    pub speed_params: Vec<SpeedParam>,
    #[api(skip_get, skip_set)]
    pub train_type: TrainType,
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
