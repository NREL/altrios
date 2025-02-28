use super::powertrain::electric_drivetrain::{ElectricDrivetrain, ElectricDrivetrainState};
use super::powertrain::fuel_converter::{FuelConverter, FuelConverterState};
use super::powertrain::generator::Generator;
use super::powertrain::reversible_energy_storage::ReversibleEnergyStorage;
use super::powertrain::ElectricMachine;
use super::{LocoTrait, Mass, MassSideEffect};
use crate::imports::*;
use crate::train::TrainState;

#[derive(Clone, Default, Debug, PartialEq, Deserialize, Serialize, HistoryMethods)]
#[altrios_api]
/// Hybrid locomotive with both engine and reversible energy storage (aka battery)  
/// This type of locomotive is not likely to be widely prevalent due to modularity of consists.  
pub struct HybridLoco {
    #[has_state]
    pub fc: FuelConverter,
    #[has_state]
    pub gen: Generator,
    #[has_state]
    pub res: ReversibleEnergyStorage,
    #[has_state]
    pub edrv: ElectricDrivetrain,
    /// control strategy for distributing power demand between `fc` and `res`
    #[api(skip_get, skip_set)]
    #[serde(default)]
    pub pt_cntrl: HybridPowertrainControls,
    /// field for tracking current state
    #[serde(default)]
    pub state: HELState,
    /// vector of [Self::state]
    #[serde(default, skip_serializing_if = "HELStateHistoryVec::is_empty")]
    pub history: HELStateHistoryVec,
}

impl Init for HybridLoco {
    fn init(&mut self) -> anyhow::Result<()> {
        self.fc.init()?;
        self.gen.init()?;
        self.res.init()?;
        self.edrv.init()?;
        Ok(())
    }
}
impl SerdeAPI for HybridLoco {}

impl Mass for HybridLoco {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        self.derived_mass().with_context(|| format_dbg!())
    }

    fn set_mass(
        &mut self,
        _new_mass: Option<si::Mass>,
        _side_effect: MassSideEffect,
    ) -> anyhow::Result<()> {
        Err(anyhow!(
            "`set_mass` not enabled for {}",
            stringify!(HybridLoco)
        ))
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        self.fc.mass().with_context(|| format_dbg!())
    }

    fn expunge_mass_fields(&mut self) {
        self.fc.expunge_mass_fields();
        self.gen.expunge_mass_fields();
        self.res.expunge_mass_fields();
    }
}

impl LocoTrait for Box<HybridLoco> {
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        train_state: TrainState,
        // amount of total train mass for this locomotive
        mass_for_loco: si::Mass,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        self.state.fc_on_causes.clear();
        match &self.pt_cntrl {
            HybridPowertrainControls::RGWDB(rgwb) => {
                if self.fc.state.engine_on && self.fc.state.time_on
                    < rgwb.fc_min_time_on.with_context(|| {
                    anyhow!(
                        "{}\n Expected `ResGreedyWithBuffers::init` to have been called beforehand.",
                        format_dbg!()
                    )
                })? {
                    self.state.fc_on_causes.push(FCOnCause::OnTimeTooShort)
                }
            }
            HybridPowertrainControls::Placeholder => {
                todo!()
            }
        };

        self.fc.set_cur_pwr_out_max(dt)?;
        let disch_buffer: si::Energy = match &self.pt_cntrl {
            HybridPowertrainControls::RGWDB(rgwb) => {
                (0.5 * mass_for_loco
                    * (rgwb
                        .speed_soc_disch_buffer
                        .with_context(|| format_dbg!())?
                        .powi(typenum::P2::new())
                        - train_state.speed.powi(typenum::P2::new())))
                .max(si::Energy::ZERO)
                    * rgwb
                        .speed_soc_disch_buffer_coeff
                        .with_context(|| format_dbg!())?
            }
            HybridPowertrainControls::Placeholder => {
                todo!()
            }
        };
        let chrg_buffer: si::Energy = match &self.pt_cntrl {
            HybridPowertrainControls::RGWDB(rgwb) => {
                (0.5 * train_state.mass_compound().with_context(|| format_dbg!())?
                    * (train_state.speed.powi(typenum::P2::new())
                        - rgwb
                            .speed_soc_regen_buffer
                            .with_context(|| format_dbg!())?
                            .powi(typenum::P2::new())))
                .max(si::Energy::ZERO)
                    * rgwb
                        .speed_soc_regen_buffer_coeff
                        .with_context(|| format_dbg!())?
            }
            HybridPowertrainControls::Placeholder => {
                todo!()
            }
        };

        self.res.set_cur_pwr_out_max(
            pwr_aux.with_context(|| anyhow!(format_dbg!("`pwr_aux` not provided")))?,
            None,
            None,
        )?;

        self.gen
            .set_cur_pwr_max_out(self.fc.state.pwr_out_max, pwr_aux)?;

        self.edrv.set_cur_pwr_max_out(
            self.gen.state.pwr_elec_prop_out_max + self.res.state.pwr_prop_out_max,
            None,
        )?;

        self.edrv
            .set_cur_pwr_regen_max(self.res.state.pwr_regen_out_max)?;

        self.gen
            .set_pwr_rate_out_max(self.fc.pwr_out_max / self.fc.pwr_ramp_lag);
        self.edrv
            .set_pwr_rate_out_max(self.gen.state.pwr_rate_out_max);
        Ok(())
    }

    fn save_state(&mut self) {
        self.deref_mut().save_state();
    }

    fn step(&mut self) {
        self.deref_mut().step()
    }

    fn get_energy_loss(&self) -> si::Energy {
        self.fc.state.energy_loss
            + self.gen.state.energy_loss
            + self.res.state.energy_loss
            + self.edrv.state.energy_loss
    }
}

impl HybridLoco {
    /// Solve fc and res energy consumption
    /// Arguments:
    /// - pwr_out_req: tractive power require
    /// - dt: time step size
    pub fn solve_energy_consumption(
        &mut self,
        _pwr_out_req: si::Power,
        dt: si::Time,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        let engine_on: bool = !self.state.fc_on_causes.is_empty();
        let fc_pwr_out_req = todo!("steal the logic for this from fastsim-3");
        self.fc
            .solve_energy_consumption(fc_pwr_out_req, dt, engine_on, assert_limits)?;

        Ok(())
    }
}

#[altrios_api]
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct FCOnCauses(Vec<FCOnCause>);
impl Init for FCOnCauses {}
impl SerdeAPI for FCOnCauses {}
impl FCOnCauses {
    fn clear(&mut self) {
        self.0.clear();
    }

    #[allow(dead_code)]
    fn pop(&mut self) -> Option<FCOnCause> {
        self.0.pop()
    }

    fn push(&mut self, new: FCOnCause) {
        self.0.push(new)
    }
}

#[altrios_api]
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, HistoryVec)]
#[non_exhaustive]
#[serde(default)]
pub struct HELState {
    /// time step index
    pub i: usize,
    /// Vector of posssible reasons the fc is forced on
    pub fc_on_causes: FCOnCauses,
    /// Number of `walk` iterations required to achieve SOC balance (i.e. SOC
    /// ends at same starting value, ensuring no net [ReversibleEnergyStorage] usage)
    pub soc_bal_iters: u32,
}

impl Init for HELState {}
impl SerdeAPI for HELState {}

// Custom serialization
impl Serialize for FCOnCauses {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let joined = self
            .0
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>()
            .join(", ");
        serializer.serialize_str(&format!("\"[{}]\"", joined))
    }
}

use serde::de::{self, Visitor};
struct FCOnCausesVisitor;
impl Visitor<'_> for FCOnCausesVisitor {
    type Value = FCOnCauses;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(
            "String form of `FCOnCauses`, e.g. `\"[VehicleSpeedTooHigh, FCTemperatureTooLow]\"`",
        )
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Self::visit_str(self, &v)
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let inner: String = v
            .replace("\"", "") // this solves a problem in interactive mode
            .strip_prefix("[")
            .ok_or("Missing leading `[`")
            .map_err(|err| de::Error::custom(err))?
            .strip_suffix("]")
            .ok_or("Missing trailing`]`")
            .map_err(|err| de::Error::custom(err))?
            .to_string();
        let fc_on_causes_str = inner.split(",").map(|x| x.trim()).collect::<Vec<&str>>();
        let fc_on_causes_unchecked = fc_on_causes_str
            .iter()
            .map(|x| {
                if x.is_empty() {
                    None
                } else {
                    Some(FromStr::from_str(x))
                }
            })
            .collect::<Vec<Option<Result<FCOnCause, derive_more::FromStrError>>>>();
        let mut fc_on_causes: FCOnCauses = FCOnCauses(vec![]);
        for (fc_on_cause_unchecked, fc_on_cause_str) in
            fc_on_causes_unchecked.into_iter().zip(fc_on_causes_str)
        {
            if let Some(fc_on_cause_unchecked) = fc_on_cause_unchecked {
                fc_on_causes.0.push(fc_on_cause_unchecked.map_err(|err| {
                    de::Error::custom(format!(
                        "{}\nfc_on_cause_unchecked: {:?}\nfc_on_cause_str: {}",
                        err, fc_on_cause_unchecked, fc_on_cause_str
                    ))
                })?)
            }
        }
        Ok(fc_on_causes)
    }
}

// Custom deserialization
impl<'de> Deserialize<'de> for FCOnCauses {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_string(FCOnCausesVisitor)
    }
}

impl std::fmt::Display for FCOnCauses {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
        // or, alternatively:
        // fmt::Debug::fmt(self, f)
    }
}

#[altrios_enum_api]
#[derive(
    Clone, Copy, Debug, Deserialize, Serialize, PartialEq, IsVariant, From, TryInto, FromStr,
)]
pub enum FCOnCause {
    /// Engine must be on to self heat if thermal model is enabled
    FCTemperatureTooLow,
    /// Engine must be on for high vehicle speed to ensure powertrain can meet
    /// any spikes in power demand
    VehicleSpeedTooHigh,
    /// Engine has not been on long enough (usually 30 s)
    OnTimeTooShort,
    /// Powertrain power demand exceeds motor and/or battery capabilities
    PropulsionPowerDemand,
    /// Powertrain power demand exceeds optimal motor and/or battery output
    PropulsionPowerDemandSoft,
    /// Aux power demand exceeds battery capability
    AuxPowerDemand,
    /// SOC is below min buffer so FC is charging RES
    ChargingForLowSOC,
}
impl SerdeAPI for FCOnCause {}
impl Init for FCOnCause {}
impl fmt::Display for FCOnCause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
        // or, alternatively:
        // fmt::Debug::fmt(self, f)
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, Default, IsVariant, From, TryInto)]
pub enum HEVAuxControls {
    /// If feasible, use [ReversibleEnergyStorage] to handle aux power demand
    #[default]
    AuxOnResPriority,
    /// If feasible, use [FuelConverter] to handle aux power demand
    AuxOnFcPriority,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, IsVariant, From, TryInto)]
pub enum HybridPowertrainControls {
    /// Greedily uses [ReversibleEnergyStorage] with buffers that derate charge
    /// and discharge power inside of static min and max SOC range.  Also, includes
    /// buffer for forcing [FuelConverter] to be active/on.
    RGWDB(Box<RESGreedyWithDynamicBuffers>),
    /// place holder for future variants
    Placeholder,
}

impl Default for HybridPowertrainControls {
    fn default() -> Self {
        Self::RGWDB(Default::default())
    }
}

impl Init for HybridPowertrainControls {
    fn init(&mut self) -> anyhow::Result<()> {
        match self {
            Self::RGWDB(rgwb) => rgwb.init()?,
            Self::Placeholder => {
                todo!()
            }
        }
        Ok(())
    }
}

/// Detemrines whether engine must be on to charge battery
fn handle_fc_on_causes_for_low_soc(
    res: &ReversibleEnergyStorage,
    rgwdb: &mut Box<RESGreedyWithDynamicBuffers>,
    hev_state: &mut HELState,
    train_state: TrainState,
) -> anyhow::Result<()> {
    rgwdb.state.soc_fc_on_buffer = {
        let energy_delta_to_buffer_speed: si::Energy = 0.5
            * train_state.mass()?.with_context(|| format_dbg!())?
            * (rgwdb
                .speed_soc_fc_on_buffer
                .with_context(|| format_dbg!())?
                .powi(typenum::P2::new())
                - train_state.speed.powi(typenum::P2::new()));
        energy_delta_to_buffer_speed.max(si::Energy::ZERO)
            * rgwdb
                .speed_soc_fc_on_buffer_coeff
                .with_context(|| format_dbg!())?
    } / res.energy_capacity_usable()
        + res.min_soc;
    if res.state.soc < rgwdb.state.soc_fc_on_buffer {
        hev_state.fc_on_causes.push(FCOnCause::ChargingForLowSOC)
    }
    Ok(())
}

/// Determines whether power demand requires engine to be on.  Not needed during
/// negative traction.
fn handle_fc_on_causes_for_pwr_demand(
    rgwdb: &mut Box<RESGreedyWithDynamicBuffers>,
    pwr_out_req: si::Power,
    em_state: &ElectricDrivetrainState,
    fc_state: &FuelConverterState,
    hev_state: &mut HELState,
) -> Result<(), anyhow::Error> {
    let frac_pwr_demand_fc_forced_on: si::Ratio = rgwdb
        .frac_pwr_demand_fc_forced_on
        .with_context(|| format_dbg!())?;
    if pwr_out_req
        > frac_pwr_demand_fc_forced_on * (em_state.pwr_mech_out_max + fc_state.pwr_out_max)
    {
        hev_state
            .fc_on_causes
            .push(FCOnCause::PropulsionPowerDemandSoft);
    }
    if pwr_out_req - em_state.pwr_mech_out_max >= si::Power::ZERO {
        hev_state
            .fc_on_causes
            .push(FCOnCause::PropulsionPowerDemand);
    }
    Ok(())
}

/// Determines whether enigne must be on for high speed
fn handle_fc_on_causes_for_speed(
    train_state: TrainState,
    rgwdb: &mut Box<RESGreedyWithDynamicBuffers>,
    hev_state: &mut HELState,
) -> anyhow::Result<()> {
    if train_state.speed > rgwdb.speed_fc_forced_on.with_context(|| format_dbg!())? {
        hev_state.fc_on_causes.push(FCOnCause::VehicleSpeedTooHigh);
    }
    Ok(())
}

impl HybridPowertrainControls {
    /// Determines power split between engine and electric machine
    ///
    /// # Arguments
    /// - `pwr_prop_req`: tractive power required
    /// - `veh_state`: vehicle state
    /// - `hev_state`: HEV powertrain state
    /// - `fc`: fuel converter
    /// - `em_state`: electric machine state
    /// - `res`: reversible energy storage (e.g. high voltage battery)
    fn get_pwr_fc_and_em(
        &mut self,
        pwr_prop_req: si::Power,
        // TODO: replace this with TrainState and then make it so that this
        // locomotive is reponsible for the total train weight (and maybe factor in
        // elevation delta somehow) per count of locomotives
        train_state: TrainState,
        hev_state: &mut HELState,
        fc: &FuelConverter,
        em_state: &ElectricDrivetrainState,
        res: &ReversibleEnergyStorage,
    ) -> anyhow::Result<(si::Power, si::Power)> {
        let fc_state = &fc.state;
        ensure!(
            // `almost` is in case of negligible numerical precision discrepancies
            almost_le_uom(
                &pwr_prop_req,
                &(em_state.pwr_mech_out_max + fc_state.pwr_out_max),
                None
            ),
            "{}
`pwr_out_req`: {} kW
`em_state.pwr_mech_fwd_out_max`: {} kW
`fc_state.pwr_prop_max`: {} kW
`res.state.soc`: {}",
            format_dbg!(),
            pwr_prop_req.get::<si::kilowatt>(),
            em_state.pwr_mech_out_max.get::<si::kilowatt>(),
            fc_state.pwr_out_max.get::<si::kilowatt>(),
            res.state.soc.get::<si::ratio>()
        );

        // # Brain dump for thermal stuff
        // TODO: engine on/off w.r.t. thermal stuff should not come into play
        // if there is no component (e.g. cabin) demanding heat from the engine.  My 2019
        // Hyundai Ioniq will turn the engine off if there is no heat demand regardless of
        // the coolant temperature
        // TODO: make sure idle fuel gets converted to heat correctly
        let (fc_pwr, em_pwr) = match self {
            Self::RGWDB(ref mut rgwdb) => {
                // handle_fc_on_causes_for_temp(fc, rgwdb, hev_state)?;
                handle_fc_on_causes_for_speed(train_state, rgwdb, hev_state)?;
                handle_fc_on_causes_for_low_soc(res, rgwdb, hev_state, train_state)?;
                // `handle_fc_*` below here are asymmetrical for positive tractive power only
                handle_fc_on_causes_for_pwr_demand(
                    rgwdb,
                    pwr_prop_req,
                    em_state,
                    fc_state,
                    hev_state,
                )?;

                // Tractive power `em` must provide before deciding power
                // split, cannot exceed ElectricMachine max output power.
                // Excess demand will be handled by `fc`.  Favors drawing
                // power from `em` before engine
                let em_pwr = pwr_prop_req
                    .min(em_state.pwr_mech_out_max)
                    .max(-em_state.pwr_mech_regen_max);
                // tractive power handled by fc
                if hev_state.fc_on_causes.is_empty() {
                    // engine is off, and `em_pwr` has already been limited within bounds
                    (si::Power::ZERO, em_pwr)
                } else {
                    // engine has been forced on
                    let frac_of_pwr_for_peak_eff: si::Ratio = rgwdb
                        .frac_of_most_eff_pwr_to_run_fc
                        .with_context(|| format_dbg!())?;
                    let fc_pwr = if pwr_prop_req < si::Power::ZERO {
                        // negative tractive power
                        // max power system can receive from engine during negative traction
                        (em_state.pwr_mech_regen_max + pwr_prop_req)
                            // or peak efficiency power if it's lower than above
                            .min(fc.pwr_for_peak_eff * frac_of_pwr_for_peak_eff)
                            // but not negative
                            .max(si::Power::ZERO)
                    } else {
                        // positive tractive power
                        if pwr_prop_req - em_pwr > fc.pwr_for_peak_eff * frac_of_pwr_for_peak_eff {
                            // engine needs to run higher than peak efficiency point
                            pwr_prop_req - em_pwr
                        } else {
                            // engine does not need to run higher than peak
                            // efficiency point to make tractive demand

                            // fc handles all power not covered by em
                            (pwr_prop_req - em_pwr)
                                // and if that's less than the
                                // efficiency-focused value, then operate at
                                // that value
                                .max(fc.pwr_for_peak_eff * frac_of_pwr_for_peak_eff)
                                // but don't exceed what what the battery can
                                // absorb + tractive demand
                                .min(pwr_prop_req + em_state.pwr_mech_regen_max)
                        }
                    }
                    // and don't exceed what the fc can do
                    .min(fc_state.pwr_out_max);

                    // recalculate `em_pwr` based on `fc_pwr`
                    let em_pwr_corrected =
                        (pwr_prop_req - fc_pwr).max(-em_state.pwr_mech_regen_max);
                    (fc_pwr, em_pwr_corrected)
                }
            }
            Self::Placeholder => todo!(),
        };

        Ok((fc_pwr, em_pwr))
    }
}

/// Greedily uses [ReversibleEnergyStorage] with buffers that derate charge
/// and discharge power inside of static min and max SOC range.  Also, includes
/// buffer for forcing [FuelConverter] to be active/on. See [Self::init] for
/// default values.
#[altrios_api]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, Default)]
#[non_exhaustive]
pub struct RESGreedyWithDynamicBuffers {
    /// RES energy delta from minimum SOC corresponding to kinetic energy of
    /// vehicle at this speed that triggers ramp down in RES discharge.
    #[api(skip_get, skip_set)]
    pub speed_soc_disch_buffer: Option<si::Velocity>,
    /// Coefficient for modifying amount of accel buffer
    #[api(skip_get, skip_set)]
    pub speed_soc_disch_buffer_coeff: Option<si::Ratio>,
    /// RES energy delta from minimum SOC corresponding to kinetic energy of
    /// vehicle at this speed that triggers FC to be forced on.
    #[api(skip_get, skip_set)]
    pub speed_soc_fc_on_buffer: Option<si::Velocity>,
    /// Coefficient for modifying amount of [Self::speed_soc_fc_on_buffer]
    #[api(skip_get, skip_set)]
    pub speed_soc_fc_on_buffer_coeff: Option<si::Ratio>,
    /// RES energy delta from maximum SOC corresponding to kinetic energy of
    /// vehicle at current speed minus kinetic energy of vehicle at this speed
    /// triggers ramp down in RES discharge
    #[api(skip_get, skip_set)]
    pub speed_soc_regen_buffer: Option<si::Velocity>,
    /// Coefficient for modifying amount of regen buffer
    #[api(skip_get, skip_set)]
    pub speed_soc_regen_buffer_coeff: Option<si::Ratio>,
    /// Minimum time engine must remain on if it was on during the previous
    /// simulation time step.
    #[api(skip_get, skip_set)]
    pub fc_min_time_on: Option<si::Time>,
    /// Speed at which [FuelConverter] is forced on.
    #[api(skip_get, skip_set)]
    pub speed_fc_forced_on: Option<si::Velocity>,
    /// Fraction of total aux and powertrain rated power at which
    /// [FuelConverter] is forced on.
    #[api(skip_get, skip_set)]
    pub frac_pwr_demand_fc_forced_on: Option<si::Ratio>,
    /// Force engine, if on, to run at this fraction of power at which peak
    /// efficiency occurs or the required power, whichever is greater. If SOC is
    /// below min buffer or engine is otherwise forced on and battery has room
    /// to receive charge, engine will run at this level and charge.
    #[api(skip_get, skip_set)]
    pub frac_of_most_eff_pwr_to_run_fc: Option<si::Ratio>,
    // /// temperature at which engine is forced on to warm up
    // #[serde(default)]
    // pub temp_fc_forced_on: Option<si::Temperature>,
    // /// temperature at which engine is allowed to turn off due to being sufficiently warm
    // #[serde(default)]
    // pub temp_fc_allowed_off: Option<si::Temperature>,
    /// current state of control variables
    #[serde(default)]
    pub state: RGWDBState,
    #[serde(default, skip_serializing_if = "RGWDBStateHistoryVec::is_empty")]
    /// history of current state
    pub history: RGWDBStateHistoryVec,
}

impl Init for RESGreedyWithDynamicBuffers {
    fn init(&mut self) -> anyhow::Result<()> {
        // TODO: make sure these values propagate to the documented defaults above
        self.speed_soc_disch_buffer = self.speed_soc_disch_buffer.or(Some(40.0 * uc::MPH));
        self.speed_soc_disch_buffer_coeff = self.speed_soc_disch_buffer_coeff.or(Some(1.0 * uc::R));
        self.speed_soc_fc_on_buffer = self
            .speed_soc_fc_on_buffer
            .or(Some(self.speed_soc_disch_buffer.unwrap() * 1.1));
        self.speed_soc_fc_on_buffer_coeff = self.speed_soc_fc_on_buffer_coeff.or(Some(1.0 * uc::R));
        self.speed_soc_regen_buffer = self.speed_soc_regen_buffer.or(Some(30. * uc::MPH));
        self.speed_soc_regen_buffer_coeff = self.speed_soc_regen_buffer_coeff.or(Some(1.0 * uc::R));
        self.fc_min_time_on = self.fc_min_time_on.or(Some(uc::S * 5.0));
        self.speed_fc_forced_on = self.speed_fc_forced_on.or(Some(uc::MPH * 75.));
        self.frac_pwr_demand_fc_forced_on =
            self.frac_pwr_demand_fc_forced_on.or(Some(uc::R * 0.75));
        self.frac_of_most_eff_pwr_to_run_fc =
            self.frac_of_most_eff_pwr_to_run_fc.or(Some(1.0 * uc::R));
        Ok(())
    }
}
impl SerdeAPI for RESGreedyWithDynamicBuffers {}

#[altrios_api]
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, HistoryVec)]
#[serde(default)]
/// State for [RESGreedyWithDynamicBuffers ]
pub struct RGWDBState {
    /// time step index
    pub i: usize,
    /// Vector of posssible reasons the fc is forced on
    pub fc_on_causes: FCOnCauses,
    /// Number of `walk` iterations required to achieve SOC balance (i.e. SOC
    /// ends at same starting value, ensuring no net [ReversibleEnergyStorage] usage)
    pub soc_bal_iters: u32,
    /// buffer at which FC is forced on
    pub soc_fc_on_buffer: si::Ratio,
}

impl Init for RGWDBState {}
impl SerdeAPI for RGWDBState {}
