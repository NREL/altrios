use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, From, IsVariant, TryInto)]
pub enum PowertrainType {
    ConventionalLoco(ConventionalLoco),
    HybridLoco(Box<HybridLoco>),
    BatteryElectricLoco(BatteryElectricLoco),
    DummyLoco(DummyLoco),
}

impl Init for PowertrainType {
    fn init(&mut self) -> Result<(), Error> {
        match self {
            Self::ConventionalLoco(l) => l.init()?,
            Self::HybridLoco(l) => l.init()?,
            Self::BatteryElectricLoco(l) => l.init()?,
            Self::DummyLoco(_) => {}
        };
        Ok(())
    }
}
impl SerdeAPI for PowertrainType {}

impl LocoTrait for PowertrainType {
    fn set_curr_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        train_mass_for_loco: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        match self {
            PowertrainType::ConventionalLoco(conv) => {
                conv.set_curr_pwr_max_out(pwr_aux, train_mass_for_loco, train_speed, dt)
            }
            PowertrainType::HybridLoco(hel) => {
                hel.set_curr_pwr_max_out(pwr_aux, train_mass_for_loco, train_speed, dt)
            }
            PowertrainType::BatteryElectricLoco(bel) => {
                bel.set_curr_pwr_max_out(pwr_aux, train_mass_for_loco, train_speed, dt)
            }
            PowertrainType::DummyLoco(dummy) => {
                dummy.set_curr_pwr_max_out(pwr_aux, train_mass_for_loco, train_speed, dt)
            }
        }
    }

    fn save_state(&mut self) {
        match self {
            PowertrainType::ConventionalLoco(conv) => conv.save_state(),
            PowertrainType::HybridLoco(hel) => hel.save_state(),
            PowertrainType::BatteryElectricLoco(bel) => bel.save_state(),
            PowertrainType::DummyLoco(dummy) => dummy.save_state(),
        }
    }

    fn step(&mut self) {
        match self {
            PowertrainType::ConventionalLoco(conv) => conv.step(),
            PowertrainType::HybridLoco(hel) => hel.step(),
            PowertrainType::BatteryElectricLoco(bel) => bel.step(),
            PowertrainType::DummyLoco(dummy) => dummy.step(),
        }
    }

    fn get_energy_loss(&self) -> si::Energy {
        match self {
            PowertrainType::ConventionalLoco(conv) => conv.get_energy_loss(),
            PowertrainType::HybridLoco(hel) => hel.get_energy_loss(),
            PowertrainType::BatteryElectricLoco(bel) => bel.get_energy_loss(),
            PowertrainType::DummyLoco(dummy) => dummy.get_energy_loss(),
        }
    }
}

impl From<HybridLoco> for PowertrainType {
    fn from(value: HybridLoco) -> Self {
        Self::HybridLoco(Box::new(value))
    }
}

// #[cfg(feature = "pyo3")]
// impl TryFrom<Bound<PyAny>> for PowertrainType {
//     type Error = PyErr;
//     /// This allows us to construct PowertrainType from any struct that can be converted into PowertrainType
//     fn try_from(value: &Bound<PyAny>) -> std::result::Result<Self, Self::Error> {
//         value
//             .extract::<ConventionalLoco>()
//             .map(PowertrainType::from)
//             .or_else(|_| {
//                 value
//                     .extract::<HybridLoco>()
//                     .map(PowertrainType::from)
//                     .or_else(|_| {
//                         value
//                             .extract::<BatteryElectricLoco>()
//                             .map(PowertrainType::from)
//                             .or_else(|_| value.extract::<DummyLoco>().map(PowertrainType::from))
//                     })
//             })
//             .map_err(|_| {
//                 pyo3::PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
//                     "{}\nMust provide ConventionalLoco, HybridLoco, BatteryElectricLoco, or DummyLoco",
//                     format_dbg!()
//                 ))
//             })
//     }
// }

impl Default for PowertrainType {
    fn default() -> Self {
        Self::ConventionalLoco(Default::default())
    }
}

#[allow(clippy::to_string_trait_impl)]
impl std::string::ToString for PowertrainType {
    fn to_string(&self) -> String {
        let s = match self {
            PowertrainType::ConventionalLoco(_) => stringify!(ConventionalLoco),
            PowertrainType::HybridLoco(_) => stringify!(HybridLoco),
            PowertrainType::BatteryElectricLoco(_) => stringify!(BatteryElectricLoco),
            PowertrainType::DummyLoco(_) => stringify!(DummyLoco),
        };
        s.into()
    }
}

#[altrios_api(
    #[new]
    #[pyo3(signature = (pwr_aux_offset_watts, pwr_aux_traction_coeff_ratio, force_max_newtons, mass_kilograms=None))]
    fn __new__(
        pwr_aux_offset_watts: f64,
        pwr_aux_traction_coeff_ratio: f64,
        force_max_newtons: f64,
        mass_kilograms: Option<f64>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            pwr_aux_offset: pwr_aux_offset_watts * uc::W,
            pwr_aux_traction_coeff: pwr_aux_traction_coeff_ratio * uc::R,
            force_max: force_max_newtons * uc::N,
            mass: mass_kilograms.map(|m| m * uc::KG)
        })
    }

    #[staticmethod]
    #[pyo3(name = "from_dict")]
    fn from_dict_py(param_dict: &Bound<PyDict>) -> anyhow::Result<Self> {
        Self::from_dict(param_dict)
    }
)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
/// Struct to facilitate passing several parameters to builder
pub struct LocoParams {
    /// [Locomotive::pwr_aux_offset]
    pub pwr_aux_offset: si::Power,
    /// [Locomotive::pwr_aux_traction_coeff]
    pub pwr_aux_traction_coeff: si::Ratio,
    /// [Locomotive::force_max]
    pub force_max: si::Force,
    #[api(skip_get, skip_set)]
    /// [Locomotive::mass]
    pub mass: Option<si::Mass>,
}

impl LocoParams {
    #[allow(unused)]
    fn from_hash(mut params: HashMap<&str, f64>) -> anyhow::Result<Self> {
        let pwr_aux_offset_watts = params
            .remove("pwr_aux_offset_watts")
            .with_context(|| anyhow!("Must provide 'pwr_aux_offset_watts'."))?;
        let pwr_aux_traction_coeff_ratio = params
            .remove("pwr_aux_traction_coeff_ratio")
            .with_context(|| anyhow!("Must provide 'pwr_aux_traction_coeff_ratio'."))?;
        let force_max_newtons = params
            .remove("force_max_newtons")
            .with_context(|| anyhow!("Must provide 'force_max_newtons'."))?;
        let mass_kg = params.remove("mass_kg");
        ensure!(
            params.is_empty(),
            "{}\nSuperfluous `params` keys: {:?}",
            format_dbg!(),
            params.keys()
        );
        Ok(Self {
            pwr_aux_offset: pwr_aux_offset_watts * uc::W,
            pwr_aux_traction_coeff: pwr_aux_traction_coeff_ratio * uc::R,
            force_max: force_max_newtons * uc::N,
            mass: mass_kg.map(|m| m * uc::KG),
        })
    }

    #[cfg(feature = "pyo3")]
    fn from_dict(param_dict: &Bound<PyDict>) -> anyhow::Result<Self> {
        let mut param_hm: HashMap<String, f64> = HashMap::new();
        for (key, value) in param_dict
            .keys()
            .into_iter()
            .zip(param_dict.values().into_iter())
        {
            let key_extracted = key
                .extract::<String>()
                .with_context(|| format!("{}\nFailed to extract key", format_dbg!()))?;
            let value_extracted = value
                .extract()
                .with_context(|| format!("{}\nFailed to extract value", format_dbg!()))?;
            param_hm.insert(key_extracted, value_extracted);
        }
        // type conversion on keys to satisfy function arg below
        let param_hm = HashMap::from_iter(param_hm.iter().map(|item| (item.0.as_str(), *item.1)));

        Self::from_hash(param_hm)
    }
}

impl Default for LocoParams {
    fn default() -> Self {
        Self {
            pwr_aux_offset: 8554.15 * uc::W,
            pwr_aux_traction_coeff: 0.000539638 * uc::R,
            force_max: 667.2e3 * uc::N,
            // https://www.wabteccorp.com/media/3641/download?inline
            // per above, 432,000 lbs = 195,000 kg
            mass: Some(195_000.0 * uc::KG),
        }
    }
}

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
#[altrios_api(
    #[staticmethod]
    fn __new__() -> Self {
        Default::default()
    }
)]
/// DummyLoco locomotive with infinite power and free energy, used for
/// working with train performance calculator with
/// [crate::train::SetSpeedTrainSim] with no effort to ensure loads
/// on locomotive are realistic.
pub struct DummyLoco {}

impl LocoTrait for DummyLoco {
    fn set_curr_pwr_max_out(
        &mut self,
        _pwr_aux: Option<si::Power>,
        _train_mass: Option<si::Mass>,
        _train_speed: Option<si::Velocity>,
        _dt: si::Time,
    ) -> anyhow::Result<()> {
        Ok(())
    }
    fn save_state(&mut self) {}
    fn step(&mut self) {}
    fn get_energy_loss(&self) -> si::Energy {
        si::Energy::ZERO
    }
}

#[altrios_api(
    #[new]
    #[pyo3(signature = (loco_type, loco_params, save_interval=None))]
    fn __new__(
        // needs to be variant in PowertrainType
        loco_type: &Bound<PyAny>,
        loco_params: LocoParams,
        save_interval: Option<usize>
    ) -> anyhow::Result<Self> {
        let loco_type = loco_type
            .extract::<ConventionalLoco>()
            .map(PowertrainType::from)
            .or_else(|_| {
                loco_type
                    .extract::<HybridLoco>()
                    .map(PowertrainType::from)
                    .or_else(|_| {
                        loco_type
                            .extract::<BatteryElectricLoco>()
                            .map(PowertrainType::from)
                            .or_else(|_| loco_type.extract::<DummyLoco>().map(PowertrainType::from))
                    })
            })
            .map_err(|_| {
                pyo3::PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "{}\nMust provide ConventionalLoco, HybridLoco, BatteryElectricLoco, or DummyLoco",
                    format_dbg!()
                ))
            })?;

        Ok(Self {
            loco_type,
            state: Default::default(),
            save_interval,
            assert_limits: true,
            pwr_aux_offset: loco_params.pwr_aux_offset,
            pwr_aux_traction_coeff: loco_params.pwr_aux_traction_coeff,
            force_max: loco_params.force_max,
            ..Default::default()
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (fuel_converter, generator, drivetrain, loco_params, save_interval=None))]
    #[staticmethod]
    fn build_conventional_loco(
        fuel_converter: FuelConverter,
        generator: Generator,
        drivetrain: ElectricDrivetrain,
        loco_params: LocoParams,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        let mut loco = Self {
            loco_type: PowertrainType::ConventionalLoco(ConventionalLoco::new(
                fuel_converter,
                generator,
                drivetrain,
            )),
            state: Default::default(),
            save_interval,
            history: LocomotiveStateHistoryVec::new(),
            assert_limits: true,
            pwr_aux_offset: loco_params.pwr_aux_offset,
            pwr_aux_traction_coeff: loco_params.pwr_aux_traction_coeff,
            force_max: loco_params.force_max,
            ..Default::default()
        };
        // make sure save_interval is propagated
        loco.set_save_interval(save_interval);
        Ok(loco)
    }

    #[staticmethod]
    #[pyo3(name = "default_battery_electric_loco")]
    fn default_battery_electric_loco_py () -> anyhow::Result<Self> {
        Ok(Self::default_battery_electric_loco())
    }

    #[staticmethod]
    #[pyo3(name = "default_hybrid_electric_loco")]
    fn default_hybrid_electric_loco_py () -> anyhow::Result<Self> {
        Ok(Self::default_hybrid_electric_loco())
    }


    #[staticmethod]
    fn build_dummy_loco() -> Self {
        let mut dummy  = Self {
            loco_type: PowertrainType::DummyLoco(DummyLoco::default()),
            state: LocomotiveState::default(),
            save_interval: None,
            history: LocomotiveStateHistoryVec::new(),
            assert_limits: true,
            pwr_aux_offset: 50e3 * uc::W,
            pwr_aux_traction_coeff: 0.01 * uc::R,
            force_max: 50e6 * uc::N,
            ..Default::default()
        };
        dummy.set_mass(None, MassSideEffect::None).unwrap();
        dummy
    }

    #[pyo3(name = "set_save_interval")]
    #[pyo3(signature = (save_interval=None))]
    /// Set save interval and cascade to nested components.
    fn set_save_interval_py(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.set_save_interval(save_interval);
        Ok(())
    }

    #[pyo3(name = "get_save_interval")]
    /// Set save interval and cascade to nested components.
    fn get_save_interval_py(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.get_save_interval())
    }

    #[getter]
    fn get_fc(&self) -> Option<FuelConverter> {
        self.fuel_converter().cloned()
    }
    #[setter]
    fn set_fc(&mut self, _fc: FuelConverter) -> anyhow::Result<()> {
        bail!(PyAttributeError::new_err(DIRECT_SET_ERR))
    }

    #[setter(__fc)]
    fn set_fc_hidden(&mut self, fc: FuelConverter) -> anyhow::Result<()> {
        Ok(self.set_fuel_converter(fc).map_err(|e| PyAttributeError::new_err(e.to_string()))?)
    }
    #[getter]
    fn get_gen(&self) -> Option<Generator> {
        self.generator().cloned()
    }

    #[setter]
    fn set_gen(&mut self, _gen: Generator) -> anyhow::Result<()> {
        bail!(PyAttributeError::new_err(DIRECT_SET_ERR))
    }
    #[setter(__gen)]
    fn set_gen_hidden(&mut self, gen: Generator) -> anyhow::Result<()> {
        Ok(self.set_generator(gen).map_err(|e| PyAttributeError::new_err(e.to_string()))?)
    }
    #[getter]
    fn get_res(&self) -> Option<ReversibleEnergyStorage> {
        self.reversible_energy_storage().cloned()
    }
    #[setter]
    fn set_res(&mut self, _res: ReversibleEnergyStorage) -> anyhow::Result<()> {
        bail!(PyAttributeError::new_err(DIRECT_SET_ERR))
    }

    #[setter(__res)]
    fn set_res_hidden(&mut self, res: ReversibleEnergyStorage) -> anyhow::Result<()> {
        Ok(self.set_reversible_energy_storage(res).map_err(|e| PyAttributeError::new_err(e.to_string()))?)
    }
    #[getter]
    fn get_edrv(&self) -> Option<ElectricDrivetrain> {
        self.electric_drivetrain()
    }
    #[setter]
    fn set_edrv(&mut self, _edrv: ElectricDrivetrain) -> anyhow::Result<()> {
        bail!(PyAttributeError::new_err(DIRECT_SET_ERR))
    }
    #[setter(__edrv)]
    fn set_edrv_hidden(&mut self, edrv: ElectricDrivetrain) -> anyhow::Result<()> {
        Ok(self.set_electric_drivetrain(edrv).map_err(|e| PyAttributeError::new_err(e.to_string()))?)
    }

    fn loco_type(&self) -> anyhow::Result<String> {
        Ok(self.loco_type.to_string())
    }

    #[getter]
    fn get_pwr_rated_kilowatts(&self) -> f64 {
        self.get_pwr_rated().get::<si::kilowatt>()
    }

    #[getter("force_max_lbs")]
    fn get_force_max_pounds_py(&self) -> anyhow::Result<f64> {
        Ok(self.force_max()?.get::<si::pound_force>())
    }

    #[getter("force_max_newtons")]
    fn get_force_max_newtons_py(&self) -> anyhow::Result<f64> {
        Ok(self.force_max()?.get::<si::newton>())
    }

    #[pyo3(name="set_force_max_newtons")]
    /// Sets max tractive force and applies `side_effect`.  Note that this should be 
    /// used only on a standalone `Locomotive` (i.e. not nested in another object).
    /// # Arguments
    /// - `force_max`: max tractive force
    /// - `side_effect`: string form of `ForceMaxSideEffect`
    fn set_force_max_newtons_py(
        &mut self,
        force_max: f64,
        side_effect: String
    ) -> anyhow::Result<()> {
        self.set_force_max(
            force_max * uc::N,
            side_effect.try_into()?
        )?;
        Ok(())
    }

    #[pyo3(name="set_force_max_pounds")]
    fn set_force_max_pounds_py(
        &mut self,
        force_max: f64,
        side_effect: String
    ) -> anyhow::Result<()> {
        self.set_force_max(
            force_max * uc::LBF,
            side_effect.try_into()?
        )?;
        Ok(())
    }

    #[getter]
    fn get_mass_kg(&self) -> anyhow::Result<Option<f64>> {
        Ok(self.mass()?.map(|m| m.get::<si::kilogram>()))
    }

    #[getter]
    fn get_ballast_mass_kg(&self) -> anyhow::Result<Option<f64>> {
        Ok(self.ballast_mass.map(|m| m.get::<si::kilogram>()))
    }

    #[getter]
    fn get_baseline_mass_kg(&self) -> anyhow::Result<Option<f64>> {
        Ok(self.baseline_mass.map(|m| m.get::<si::kilogram>()))
    }

    #[getter("mu")]
    fn get_mu_py(&self) -> anyhow::Result<Option<f64>> {
        Ok(self.mu()?.map(|mu| mu.get::<si::ratio>()))
    }

    /// Sets traction coefficient and applies `side_effect`.  Note that this should be 
    /// used only on a standalone `Locomotive` (i.e. not nested in another object).
    /// # Arguments
    /// - `mu`: tractive coefficient between wheel and rail
    /// - `side_effect`: string form of `MuSideEffect`
    #[pyo3(name="set_mu")]
    fn set_mu_py(
        &mut self,
        mu: f64,
        mu_side_effect: String
    ) -> anyhow::Result<()> {
        self.set_mu(
            mu * uc::R,
            mu_side_effect.try_into()?
        )?;
        Ok(())
    }
)]
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
/// Struct for simulating any type of locomotive
pub struct Locomotive {
    #[api(skip_get, skip_set)]
    /// type of locomotive including contained type-specific parameters
    /// and variables
    pub loco_type: PowertrainType,
    /// current state of locomotive
    #[serde(default)]
    #[serde(skip_serializing_if = "EqDefault::eq_default")]
    pub state: LocomotiveState,
    #[api(skip_get, skip_set)]
    #[serde(default)]
    /// Locomotive mass
    mass: Option<si::Mass>,
    /// Locomotive coefficient of friction between wheels and rail when
    /// stopped (i.e. traction coefficient)
    #[api(skip_get, skip_set)]
    mu: Option<si::Ratio>,
    /// Ballast mass, any mass that must be added to achieve nominal
    /// locomotive weight of 432,000 lb.
    #[api(skip_get, skip_set)]
    ballast_mass: Option<si::Mass>,
    /// Baseline mass, which comprises any non-differentiating
    /// components between technologies, e.g. chassis, motors, trucks,
    /// cabin
    #[api(skip_get, skip_set)]
    baseline_mass: Option<si::Mass>,
    /// time step interval between saves.  1 is a good option.  If None,
    /// no saving occurs.
    #[api(skip_set, skip_get)]
    save_interval: Option<usize>,
    /// Custom vector of [Self::state]
    #[serde(default, skip_serializing_if = "LocomotiveStateHistoryVec::is_empty")]
    pub history: LocomotiveStateHistoryVec,
    #[serde(default = "utils::return_true")]
    /// If true, requires power demand to not exceed consist
    /// capabilities.  May be deprecated soon.
    pub assert_limits: bool,
    /// constant aux load
    pub pwr_aux_offset: si::Power,
    /// gain for linear model on traction power used to compute traction-power-dependent component
    /// of aux load, in terms of ratio of aux power per tractive power
    pub pwr_aux_traction_coeff: si::Ratio,
    /// maximum tractive force
    #[api(skip_get, skip_set)]
    force_max: si::Force,
}

impl Default for Locomotive {
    /// Returns locomotive with defaults for Tier 4 [ConventionalLoco]
    fn default() -> Self {
        let loco_params = LocoParams::default();
        let mut loco = Self {
            loco_type: PowertrainType::ConventionalLoco(ConventionalLoco::default()),
            pwr_aux_offset: loco_params.pwr_aux_offset,
            pwr_aux_traction_coeff: loco_params.pwr_aux_traction_coeff,
            mass: loco_params.mass,
            force_max: loco_params.force_max,
            state: Default::default(),
            ballast_mass: Default::default(),
            baseline_mass: Default::default(),
            save_interval: Some(1),
            history: Default::default(),
            assert_limits: true,
            mu: Default::default(),
        };
        loco.init().unwrap();
        loco.set_save_interval(Some(1));
        loco
    }
}

impl Init for Locomotive {
    fn init(&mut self) -> Result<(), Error> {
        let _mass = self
            .mass()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.loco_type.init()?;
        Ok(())
    }
}
impl SerdeAPI for Locomotive {}

impl Mass for Locomotive {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let derived_mass = self.derived_mass().with_context(|| format_dbg!())?;
        match (derived_mass, self.mass) {
            (Some(derived_mass), Some(set_mass)) => {
                ensure!(
                    utils::almost_eq_uom(&set_mass, &derived_mass, None),
                    format!(
                        "{}",
                        format_dbg!(utils::almost_eq_uom(&set_mass, &derived_mass, None)),
                    )
                );
                Ok(Some(set_mass))
            }
            (None, None) => Ok(None),
            _ => Ok(self.mass.or(derived_mass)),
        }
    }

    fn set_mass(
        &mut self,
        new_mass: Option<si::Mass>,
        side_effect: MassSideEffect,
    ) -> anyhow::Result<()> {
        ensure!(
            side_effect == MassSideEffect::None,
            "At the locomotive level, only `MassSideEffect::None` is allowed"
        );

        let derived_mass = self.derived_mass().with_context(|| format_dbg!())?;
        self.mass = match new_mass {
            // Set using provided `new_mass`, setting constituent mass fields to `None` to match if inconsistent
            Some(new_mass) => {
                if let Some(dm) = derived_mass {
                    if dm != new_mass {
                        self.expunge_mass_fields();
                    }
                }
                Some(new_mass)
            }
            // Set using `derived_mass()`, failing if it returns `None`
            None => Some(derived_mass.with_context(|| {
                format!(
                    "Not all mass fields in `{}` are set and no mass was provided.",
                    stringify!(Locomotive)
                )
            })?),
        };
        self.force_max = self
            .mu()
            .with_context(|| format_dbg!())?
            .with_context(|| format!("{}\nExpected `mu` to be set", format_dbg!()))?
            * self
                .mass()?
                .with_context(|| format!("{}\nExpected `mass` to be set", format_dbg!()))?
            * uc::ACC_GRAV;
        Ok(())
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        match &self.loco_type {
            PowertrainType::ConventionalLoco(conv) => conv.mass(),
            PowertrainType::HybridLoco(hev) => hev.mass(),
            PowertrainType::BatteryElectricLoco(bev) => bev.mass(),
            PowertrainType::DummyLoco(_) => Ok(None),
        }
    }

    fn expunge_mass_fields(&mut self) {
        match &mut self.loco_type {
            PowertrainType::ConventionalLoco(conv) => conv.expunge_mass_fields(),
            PowertrainType::HybridLoco(hev) => hev.expunge_mass_fields(),
            PowertrainType::BatteryElectricLoco(bev) => bev.expunge_mass_fields(),
            PowertrainType::DummyLoco(_) => {}
        };
    }
}

impl Locomotive {
    /// Sets force max based on provided value or previously set
    /// `self.mu`.
    ///
    /// Arugments:
    /// * `force_max` - option for setting `self.force_max` directly
    /// * `side_effect` - which dependent parameter to correspondingly update
    pub fn set_force_max(
        &mut self,
        force_max: si::Force,
        side_effect: ForceMaxSideEffect,
    ) -> anyhow::Result<()> {
        self.force_max = force_max;
        match side_effect {
            ForceMaxSideEffect::Mass => self
                .set_mass(
                    Some(
                        force_max
                            / (self.mu().with_context(|| format_dbg!())?.with_context(|| {
                                format_dbg!("Expected traction coefficient to be set.")
                            })? * uc::ACC_GRAV),
                    ),
                    MassSideEffect::None,
                )
                .with_context(|| format_dbg!())?,
            ForceMaxSideEffect::UpdateMu => {
                self.mu = self.mass.map(|mass| force_max / (mass * uc::ACC_GRAV))
            }
            ForceMaxSideEffect::SetMuToNone => {
                self.mu = None;
            }
            ForceMaxSideEffect::SetMassToNone => {
                self.mass = None;
            }
            ForceMaxSideEffect::SetMassAndMuToNone => {
                self.mu = None;
                self.mass = None;
            }
        }
        Ok(())
    }

    pub fn force_max(&self) -> anyhow::Result<si::Force> {
        self.check_force_max()
            .with_context(|| anyhow!(format_dbg!()))?;
        Ok(self.force_max)
    }

    pub fn check_force_max(&self) -> anyhow::Result<()> {
        if let (Some(mu), Some(mass)) = (self.mu, self.mass) {
            ensure!(utils::almost_eq_uom(
                    &self.force_max,
                    &(mu * mass * uc::ACC_GRAV),
                    None
                ),
                format!(
                    "`force_max` is not almost equal to calculation from `mu` and `mass`.\n{}\n`force_max`: {:?}\ncalculated `force_max`: {:?}\n`mu`: {:?}\n`mass`: {:?}",
                    format_dbg!(),
                    self.force_max,
                    mu * mass * uc::ACC_GRAV,
                    mu,
                    mass
                )
            );
        }
        Ok(())
    }

    pub fn default_battery_electric_loco() -> Self {
        let mut loco = Locomotive {
            loco_type: PowertrainType::BatteryElectricLoco(Default::default()),
            mass: Some(194.6e3 * uc::KG),
            ballast_mass: None,
            baseline_mass: None,
            force_max: 667.2e3 * uc::N,
            pwr_aux_offset: 8.55e3 * uc::W,
            pwr_aux_traction_coeff: 540e-6 * uc::R,
            mu: None,
            state: Default::default(),
            history: Default::default(),
            save_interval: Some(1),
            assert_limits: true,
        };
        loco.init().unwrap();
        loco.set_save_interval(Some(1));
        loco
    }

    pub fn default_hybrid_electric_loco() -> Self {
        // TODO: add `pwr_aux_offset` and `pwr_aux_traction_coeff` based on calibration
        let hel_type = PowertrainType::HybridLoco(Box::default());
        let mut loco = Locomotive {
            loco_type: hel_type,
            ..Default::default()
        };
        loco.init().unwrap();
        loco.set_save_interval(Some(1));
        loco
    }

    pub fn get_pwr_rated(&self) -> si::Power {
        if self.fuel_converter().is_some() && self.reversible_energy_storage().is_some() {
            self.fuel_converter().unwrap().pwr_out_max
                + self.reversible_energy_storage().unwrap().pwr_out_max
        } else if self.fuel_converter().is_some() {
            self.fuel_converter().unwrap().pwr_out_max
        } else {
            self.reversible_energy_storage().unwrap().pwr_out_max
        }
    }

    pub fn get_save_interval(&self) -> Option<usize> {
        self.save_interval
    }

    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.save_interval = save_interval;
        match &mut self.loco_type {
            PowertrainType::ConventionalLoco(loco) => {
                loco.fc.save_interval = save_interval;
                loco.gen.save_interval = save_interval;
                loco.edrv.save_interval = save_interval;
            }
            PowertrainType::HybridLoco(loco) => {
                loco.fc.save_interval = save_interval;
                loco.gen.save_interval = save_interval;
                loco.res.save_interval = save_interval;
                loco.edrv.save_interval = save_interval;
            }
            PowertrainType::BatteryElectricLoco(loco) => {
                loco.res.save_interval = save_interval;
                loco.edrv.save_interval = save_interval;
            }
            PowertrainType::DummyLoco(_) => { /* maybe return an error for this in the future */ }
        }
    }

    pub fn fuel_converter(&self) -> Option<&FuelConverter> {
        match &self.loco_type {
            PowertrainType::ConventionalLoco(loco) => Some(&loco.fc),
            PowertrainType::HybridLoco(loco) => Some(&loco.fc),
            PowertrainType::BatteryElectricLoco(_) => None,
            PowertrainType::DummyLoco(_) => None,
        }
    }

    pub fn fuel_converter_mut(&mut self) -> Option<&mut FuelConverter> {
        match &mut self.loco_type {
            PowertrainType::ConventionalLoco(loco) => Some(&mut loco.fc),
            PowertrainType::HybridLoco(loco) => Some(&mut loco.fc),
            PowertrainType::BatteryElectricLoco(_) => None,
            PowertrainType::DummyLoco(_) => None,
        }
    }

    pub fn set_fuel_converter(&mut self, fc: FuelConverter) -> Result<()> {
        match &mut self.loco_type {
            PowertrainType::ConventionalLoco(loco) => {
                loco.fc = fc;
                Ok(())
            }
            PowertrainType::HybridLoco(loco) => {
                loco.fc = fc;
                Ok(())
            }
            PowertrainType::BatteryElectricLoco(_) => bail!("BEL has no FuelConverter."),
            PowertrainType::DummyLoco(_) => bail!("DummyLoco locomotive has no FuelConverter."),
        }
    }

    pub fn generator(&self) -> Option<&Generator> {
        match &self.loco_type {
            PowertrainType::ConventionalLoco(loco) => Some(&loco.gen),
            PowertrainType::HybridLoco(loco) => Some(&loco.gen),
            PowertrainType::BatteryElectricLoco(_) => None,
            PowertrainType::DummyLoco(_) => None,
        }
    }

    pub fn generator_mut(&mut self) -> Option<&mut Generator> {
        match &mut self.loco_type {
            PowertrainType::ConventionalLoco(loco) => Some(&mut loco.gen),
            PowertrainType::HybridLoco(loco) => Some(&mut loco.gen),
            PowertrainType::BatteryElectricLoco(_) => None,
            PowertrainType::DummyLoco(_) => None,
        }
    }

    pub fn set_generator(&mut self, gen: Generator) -> Result<()> {
        match &mut self.loco_type {
            PowertrainType::ConventionalLoco(loco) => {
                loco.gen = gen;
                Ok(())
            }
            PowertrainType::HybridLoco(loco) => {
                loco.gen = gen;
                Ok(())
            }
            PowertrainType::BatteryElectricLoco(_) => bail!("BEL has no Generator."),
            PowertrainType::DummyLoco(_) => bail!("DummyLoco locomotive has no Generator."),
        }
    }

    pub fn reversible_energy_storage(&self) -> Option<&ReversibleEnergyStorage> {
        match &self.loco_type {
            PowertrainType::ConventionalLoco(_) => None,
            PowertrainType::HybridLoco(loco) => Some(&loco.res),
            PowertrainType::BatteryElectricLoco(loco) => Some(&loco.res),
            PowertrainType::DummyLoco(_) => None,
        }
    }

    pub fn reversible_energy_storage_mut(&mut self) -> Option<&mut ReversibleEnergyStorage> {
        match &mut self.loco_type {
            PowertrainType::ConventionalLoco(_) => None,
            PowertrainType::HybridLoco(loco) => Some(&mut loco.res),
            PowertrainType::BatteryElectricLoco(loco) => Some(&mut loco.res),
            PowertrainType::DummyLoco(_) => None,
        }
    }

    pub fn set_reversible_energy_storage(&mut self, res: ReversibleEnergyStorage) -> Result<()> {
        match &mut self.loco_type {
            PowertrainType::ConventionalLoco(_) => {
                bail!("Conventional has no ReversibleEnergyStorage.")
            }
            PowertrainType::HybridLoco(loco) => {
                loco.res = res;
                Ok(())
            }
            PowertrainType::BatteryElectricLoco(loco) => {
                loco.res = res;
                Ok(())
            }
            PowertrainType::DummyLoco(_) => bail!("DummyLoco locomotive has no RES."),
        }
    }

    pub fn electric_drivetrain(&self) -> Option<ElectricDrivetrain> {
        match &self.loco_type {
            PowertrainType::ConventionalLoco(loco) => {
                let edrv = loco.edrv.clone();
                Some(edrv)
            }
            PowertrainType::HybridLoco(loco) => {
                let edrv = loco.edrv.clone();
                Some(edrv)
            }
            PowertrainType::BatteryElectricLoco(loco) => {
                let edrv = loco.edrv.clone();
                Some(edrv)
            }
            PowertrainType::DummyLoco(_) => None,
        }
    }

    pub fn set_electric_drivetrain(&mut self, edrv: ElectricDrivetrain) -> Result<()> {
        match &mut self.loco_type {
            PowertrainType::ConventionalLoco(loco) => {
                loco.edrv = edrv;
                Ok(())
            }
            PowertrainType::HybridLoco(loco) => {
                loco.edrv = edrv;
                Ok(())
            }
            PowertrainType::BatteryElectricLoco(loco) => {
                loco.edrv = edrv;
                Ok(())
            }
            PowertrainType::DummyLoco(_) => {
                bail!("DummyLoco locomotive has no ElectricDrivetrain.")
            }
        }
    }

    /// Calculate mass from components.
    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        if let (Some(baseline), Some(ballast)) = (self.baseline_mass, self.ballast_mass) {
            match self.loco_type {
                PowertrainType::ConventionalLoco(_) => {
                    if let (Some(fc), Some(gen)) = (
                        self.fuel_converter().unwrap().mass()?,
                        self.generator().unwrap().mass()?,
                    ) {
                        Ok(Some(fc + gen + baseline + ballast))
                    } else {
                        bail!(
                            "Locomotive fields baseline and ballast masses are both specified\n{}\n{}",
                            "so `fc` and `gen` masses must also be specified.",
                            format_dbg!()
                        )
                    }
                }
                PowertrainType::HybridLoco(_) => {
                    if let (Some(fc), Some(gen), Some(res)) = (
                        self.fuel_converter().unwrap().mass()?,
                        self.generator().unwrap().mass()?,
                        self.reversible_energy_storage().unwrap().mass()?,
                    ) {
                        Ok(Some(fc + gen + res + baseline + ballast))
                    } else {
                        bail!(
                            "Locomotive fields baseline and ballast masses are both specified\n{}\n{}",
                            "so `fc`, `gen`, and `res` masses must also be specified.",
                            format_dbg!()
                        )
                    }
                }
                PowertrainType::BatteryElectricLoco(_) => {
                    if let Some(res) = self.reversible_energy_storage().unwrap().mass()? {
                        Ok(Some(res + baseline + ballast))
                    } else {
                        bail!(
                            "Locomotive fields baseline and ballast masses are both specified\n{}\n{}",
                            "so `res` mass must also be specified.",
                            format_dbg!()
                        )
                    }
                }
                PowertrainType::DummyLoco(_) => {
                    bail!(
                        "`baseline` and `ballast` mass must be `None` with DummyLoco locomotive.\n{}",
                        format_dbg!()
                    )
                }
            }
        } else if self.baseline_mass.is_none() && self.ballast_mass.is_none() {
            match self.loco_type {
                PowertrainType::ConventionalLoco(_) => {
                    if self.fuel_converter().unwrap().mass()?.is_none()
                        && self.generator().unwrap().mass()?.is_none()
                    {
                        Ok(None)
                    } else {
                        bail!(
                            "Locomotive fields baseline and ballast masses are both `None`\n{}\n{}",
                            "so `fc` and `gen` masses must also be `None`.",
                            format_dbg!()
                        )
                    }
                }
                PowertrainType::HybridLoco(_) => {
                    if self.fuel_converter().unwrap().mass()?.is_none()
                        && self.generator().unwrap().mass()?.is_none()
                        && self.reversible_energy_storage().unwrap().mass()?.is_none()
                    {
                        Ok(None)
                    } else {
                        bail!(
                            "Locomotive fields baseline and ballast masses are both `None`\n{}\n{}",
                            "so `fc`, `gen`, and `res` masses must also be `None`.",
                            format_dbg!()
                        )
                    }
                }
                PowertrainType::BatteryElectricLoco(_) => {
                    if self.reversible_energy_storage().unwrap().mass()?.is_none() {
                        Ok(None)
                    } else {
                        bail!(
                            "Locomotive fields baseline and ballast masses are both `None`\n{}\n{}",
                            "so `res` mass must also also be `None`.",
                            format_dbg!()
                        )
                    }
                }
                PowertrainType::DummyLoco(_) => Ok(Some(0.0 * uc::KG)),
            }
        } else {
            bail!(
                "Both `baseline` and `ballast` masses must be either `Some` or `None`\n{}",
                format_dbg!()
            )
        }
    }

    /// Given required power output and time step, solves for energy
    /// consumption Arguments:
    /// ----------
    /// - `pwr_out_req:` float, output brake power required from fuel
    ///   converter
    /// - `train_speed`: current train speed
    /// - `dt:` current time step size engine_on whether or not
    ///   locomotive is active
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        engine_on: Option<bool>,
        train_mass: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
    ) -> anyhow::Result<()> {
        // maybe put logic for toggling `engine_on` here

        self.state.pwr_out = pwr_out_req;
        match &mut self.loco_type {
            PowertrainType::ConventionalLoco(loco) => {
                loco.solve_energy_consumption(
                    pwr_out_req,
                    dt,
                    engine_on.unwrap_or(true),
                    self.state.pwr_aux,
                    self.assert_limits,
                )
                .with_context(|| format_dbg!("ConventionalLoco"))?;
                self.state.pwr_out =
                    loco.edrv.state.pwr_mech_prop_out - loco.edrv.state.pwr_mech_dyn_brake;
            }
            PowertrainType::HybridLoco(loco) => {
                loco.solve_energy_consumption(
                    pwr_out_req,
                    train_mass.with_context(|| format!(
                        "{}\n`train_mass` must be provided in `SpeedTrace` or `PowerTrace` if simulating at the consist level or below"
                        , format_dbg!()
                    ))?,
                    train_speed.with_context(|| format!(
                        "{}\n`train_speed` must be provided in `SpeedTrace` or `PowerTrace` if simulating at the consist level or below"
                        , format_dbg!()
                    ))?,
                    dt,
                    self.state.pwr_aux,
                    self.assert_limits,
                ).with_context(|| format_dbg!("HybridLoco"))?;
                // TODO: add `engine_on` and `pwr_aux` here as inputs
                self.state.pwr_out =
                    loco.edrv.state.pwr_mech_prop_out - loco.edrv.state.pwr_mech_dyn_brake;
            }
            PowertrainType::BatteryElectricLoco(loco) => {
                // todo: put something in here for deep sleep that is the
                // equivalent of engine_on in conventional loco
                loco.solve_energy_consumption(pwr_out_req, dt, self.state.pwr_aux)
                    .with_context(|| format_dbg!("BatteryElectricLoco"))?;
                self.state.pwr_out =
                    loco.edrv.state.pwr_mech_prop_out - loco.edrv.state.pwr_mech_dyn_brake;
            }
            PowertrainType::DummyLoco(_) => { /* maybe put an error error in the future */ }
        }
        self.state.energy_out += self.state.pwr_out * dt;
        self.state.energy_aux += self.state.pwr_aux * dt;

        Ok(())
    }

    /// Sets aux power for locomotive
    /// # Arguments
    /// - `loco_on` whether this locomotive is active or just dead weight
    pub fn set_pwr_aux(&mut self, loco_on: Option<bool>) {
        self.state.pwr_aux = if loco_on.unwrap_or(true) {
            // TODO: make this optionally asymmetrical to allow for locomotives that
            // do not have an aux penalty related to dynamic braking
            self.pwr_aux_offset + self.pwr_aux_traction_coeff * self.state.pwr_out.abs()
        } else {
            si::Power::ZERO
        };
    }

    pub fn mu(&self) -> anyhow::Result<Option<si::Ratio>> {
        self.check_force_max().with_context(|| format_dbg!())?;
        Ok(self.mu)
    }

    pub fn set_mu(&mut self, mu: si::Ratio, mu_side_effect: MuSideEffect) -> anyhow::Result<()> {
        self.mu = Some(mu);
        match mu_side_effect {
            MuSideEffect::Mass => self.set_mass(
                Some(self.force_max / (mu * uc::ACC_GRAV)),
                MassSideEffect::None,
            ),
            MuSideEffect::ForceMax => {
                self.force_max = mu
                    * uc::ACC_GRAV
                    * self
                        .mass()?
                        .with_context(|| format_dbg!("Expected `mass` to be Some."))?;
                Ok(())
            }
            MuSideEffect::SetMassToNone => {
                self.mass = None;
                Ok(())
            }
        }
    }
}

fn set_pwr_lims(state: &mut LocomotiveState, edrv: &ElectricDrivetrain) {
    state.pwr_out_max = edrv.state.pwr_mech_out_max;
    state.pwr_rate_out_max = edrv.state.pwr_rate_out_max;
    state.pwr_regen_max = edrv.state.pwr_mech_regen_max;
}

impl LocoTrait for Locomotive {
    fn step(&mut self) {
        self.loco_type.step();
        self.state.i += 1;
    }

    fn save_state(&mut self) {
        self.loco_type.save_state();
        if let Some(interval) = self.save_interval {
            if self.state.i % interval == 0 {
                self.history.push(self.state);
            }
        }
    }

    fn get_energy_loss(&self) -> si::Energy {
        self.loco_type.get_energy_loss()
    }

    fn set_curr_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        train_mass_for_loco: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        ensure!(
            pwr_aux.is_none(),
            format!(
                "{}\ntime step: {}",
                format_dbg!(pwr_aux.is_none()),
                self.state.i
            )
        );

        self.loco_type.set_curr_pwr_max_out(
            Some(self.state.pwr_aux),
            train_mass_for_loco,
            train_speed,
            dt,
        )?;
        match &self.loco_type {
            PowertrainType::ConventionalLoco(loco) => {
                set_pwr_lims(&mut self.state, &loco.edrv);
                assert_eq!(self.state.pwr_regen_max, si::Power::ZERO);
            }
            PowertrainType::HybridLoco(loco) => {
                set_pwr_lims(&mut self.state, &loco.edrv);
            }
            PowertrainType::BatteryElectricLoco(loco) => {
                set_pwr_lims(&mut self.state, &loco.edrv);
            }
            PowertrainType::DummyLoco(_) => {
                // this locomotive has the power of 1,000 suns and more
                // power absorption ability than really big numbers that
                // are not inf to avoid null in json
                self.state.pwr_out_max = uc::W * 1e15;
                self.state.pwr_rate_out_max = uc::WPS * 1e15;
                self.state.pwr_regen_max = uc::W * 1e15;
            }
        }
        Ok(())
    }
}

/// Locomotive state for current time step
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec)]
#[altrios_api]
pub struct LocomotiveState {
    pub i: usize,
    /// maximum forward propulsive power locomotive can produce
    pub pwr_out_max: si::Power,
    /// maximum rate of increase of forward propulsive power locomotive
    /// can produce
    pub pwr_rate_out_max: si::PowerRate,
    /// maximum regen power locomotive can absorb at the wheel
    pub pwr_regen_max: si::Power,
    /// actual wheel power achieved
    pub pwr_out: si::Power,
    /// time varying aux load
    pub pwr_aux: si::Power,
    // todo: add variable for statemachine pwr_out_prev,
    // time_at_or_below_idle, time_in_engine_state
    /// integral of [Self::pwr_out]
    pub energy_out: si::Energy,
    /// integral of [Self::pwr_aux]
    pub energy_aux: si::Energy,
    // pub force_max: si::Mass,
}

impl Init for LocomotiveState {}
impl SerdeAPI for LocomotiveState {}
impl Default for LocomotiveState {
    fn default() -> Self {
        Self {
            i: 1,
            pwr_out_max: Default::default(),
            pwr_rate_out_max: Default::default(),
            pwr_out: Default::default(),
            pwr_regen_max: Default::default(),
            energy_out: Default::default(),
            pwr_aux: Default::default(),
            energy_aux: Default::default(),
        }
    }
}

pub enum MuSideEffect {
    /// Update `mass`
    Mass,
    /// Update `force_max`
    ForceMax,
    /// Set `mass` to `None`
    SetMassToNone,
}

impl TryFrom<String> for MuSideEffect {
    type Error = anyhow::Error;
    fn try_from(value: String) -> anyhow::Result<Self> {
        let mass_side_effect = match value.as_str() {
            "Mass" => Self::Mass,
            "ForceMax" => Self::ForceMax,
            "SetMassToNone" => Self::SetMassToNone,
            _ => {
                bail!(format!(
                    "`MuSideEffect` must be 'Mass', 'ForceMax', or `SetMassToNone`."
                ))
            }
        };
        Ok(mass_side_effect)
    }
}

pub enum ForceMaxSideEffect {
    /// Update mass, leaving traction coefficient unchanged
    Mass,
    /// Update traction coefficient, leaving mass unchanged
    UpdateMu,
    /// Set traction coefficient to be `None`
    SetMuToNone,
    /// Set mass to be `None`
    SetMassToNone,
    /// Set both traction coefficient and mass to be `None`, e.g. if only `force_max`
    /// is needed
    SetMassAndMuToNone,
}

impl TryFrom<String> for ForceMaxSideEffect {
    type Error = anyhow::Error;
    fn try_from(value: String) -> anyhow::Result<Self> {
        let mass_side_effect = match value.as_str() {
            "Mass" => Self::Mass,
            "UpdateMu" => Self::UpdateMu,
            "SetMuToNone" => Self::SetMuToNone,
            "SetMassAndMuToNone" => Self::SetMassAndMuToNone,
            _ => {
                bail!(format!("`ForceMaxSideEffect` must be 'Mass', `UpdateMu`, `SetMuToNone`, or 'SetMassAndMuToNone'."))
            }
        };
        Ok(mass_side_effect)
    }
}
