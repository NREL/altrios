use crate::imports::*;
use std::collections::HashMap;

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize, SerdeAPI)]
#[altrios_api]
pub struct RailVehicle {
    /// Unique user-defined identifier for the car type.  This should include
    /// meta information about the car type's weight, e.g. `loaded`, `empty`, `partial`.
    #[serde(alias = "Car Type")]
    pub car_type: String,

    /// Railcar length (between pulling-faces)
    #[serde(alias = "Length (m)")]
    pub length: si::Length,
    /// Railcar axle count (typically 4)
    #[serde(alias = "Axle Count")]
    pub axle_count: u8,
    /// Brake valve count (typically 1)
    #[serde(alias = "Brake Count")]
    pub brake_count: u8,

    /// Railcar mass, not including freight
    #[serde(alias = "Mass Static Base (kg)")]
    pub mass_static_base: si::Mass,
    /// Freight component of total static railcar mass
    #[serde(alias = "Mass Freight (kg)")]
    pub mass_freight: si::Mass,
    /// Railcar speed limit
    #[serde(alias = "Speed Max (m/s)")]
    pub speed_max: si::Velocity,
    /// Braking ratio -- braking force per rail vehicle weight
    #[serde(alias = "Braking Ratio")]
    pub braking_ratio: si::Ratio,

    /// Additional mass value to adjust for rotating mass in wheels and axles (typically 1,500 lbs)
    // TODO: maybe change this to `mass_rot_per_axle`
    #[serde(alias = "Mass Extra per Axle (kg)")]
    pub mass_extra_per_axle: si::Mass,
    /// Bearing resistance as force
    #[serde(alias = "Bearing Res per Axle (N)")]
    pub bearing_res_per_axle: si::Force,
    /// Rolling resistance ratio (lb/ton is customary, lb/lb internal to code).
    /// This is the rolling resistance force per weight for each axle.
    #[serde(alias = "Rolling Ratio")]
    pub rolling_ratio: si::Ratio,
    /// Davis B coefficient (typically very close to zero)
    #[serde(alias = "Davis B (s/m)")]
    pub davis_b: si::InverseVelocity,
    /// Drag area (Cd*A), where Cd is drag coefficient and A is front cross-sectional area
    #[serde(alias = "Cd*A (m^2)")]
    pub cd_area: si::Area,
    // TODO: move these curve coefficients to the train somewhere?
    /// Curve coefficient 0
    #[serde(alias = "Curve Coefficient 0")]
    pub curve_coeff_0: si::Ratio,
    /// Curve coefficient 1
    #[serde(alias = "Curve Coefficient 1")]
    pub curve_coeff_1: si::Ratio,
    /// Curve coefficient 2
    #[serde(alias = "Curve Coefficient 2")]
    pub curve_coeff_2: si::Ratio,
}

impl RailVehicle {
    /// Returns total non-rotational mass, sum of `mass_static_freight` and `mass_static`
    pub fn mass_static_total(&self) -> si::Mass {
        self.mass_static_base + self.mass_freight
    }
}

pub type RailVehicleMap = HashMap<String, RailVehicle>;

#[cfg(feature = "pyo3")]
#[cfg_attr(feature = "pyo3", pyfunction(name = "import_rail_vehicles"))]
pub fn import_rail_vehicles_py(filepath: &PyAny) -> anyhow::Result<RailVehicleMap> {
    import_rail_vehicles(PathBuf::extract(filepath)?)
}

// TODO: change this to a method called `from_csv_file`, check file type, and
// make sure to do this the same as elsewhere
pub fn import_rail_vehicles<P: AsRef<Path>>(filename: P) -> anyhow::Result<RailVehicleMap> {
    let file_read = File::open(filename.as_ref())?;
    let mut reader = csv::Reader::from_reader(file_read);
    let mut rail_vehicle_map = RailVehicleMap::default();
    for result in reader.deserialize() {
        let rail_vehicle: RailVehicle = result?;
        #[cfg(feature = "logging")]
        log::debug!("Loaded `rail_vehicle`: {}", rail_vehicle.car_type);
        rail_vehicle_map.insert(rail_vehicle.car_type.clone(), rail_vehicle);
    }
    Ok(rail_vehicle_map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vehicles_import() {
        import_rail_vehicles(Path::new("./src/train/test_rail_vehicles.csv")).unwrap();
    }
}
