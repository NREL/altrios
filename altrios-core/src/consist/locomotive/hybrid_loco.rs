use super::powertrain::electric_drivetrain::{ElectricDrivetrain, ElectricDrivetrainState};
use super::powertrain::fuel_converter::FuelConverter;
use super::powertrain::generator::{Generator, GeneratorState};
use super::powertrain::reversible_energy_storage::{
    ReversibleEnergyStorage, ReversibleEnergyStorageState,
};
use super::powertrain::ElectricMachine;
use super::{LocoTrait, Mass, MassSideEffect};
use crate::imports::*;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, HistoryMethods)]
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

impl Default for HybridLoco {
    fn default() -> Self {
        Self {
            fc: {
                let mut fc: FuelConverter = Default::default();
                // dial it back for Hybrid
                fc.pwr_out_max /= 2.0;
                fc
            },
            gen: Default::default(),
            res: {
                let mut res: ReversibleEnergyStorage = Default::default();
                // dial it back for Hybrid
                res.pwr_out_max /= 3.0;
                res.energy_capacity /= 8.0;
                res
            },
            edrv: Default::default(),
            pt_cntrl: Default::default(),
            state: Default::default(),
            history: Default::default(),
        }
    }
}

impl Init for HybridLoco {
    fn init(&mut self) -> Result<(), Error> {
        self.fc.init()?;
        self.gen.init()?;
        self.res.init()?;
        self.edrv.init()?;
        self.pt_cntrl.init()?;
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
    fn set_curr_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        // amount of assigned train mass for this locomotive
        train_mass_for_loco: Option<si::Mass>,
        train_speed: Option<si::Velocity>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let mass_for_loco: si::Mass = train_mass_for_loco.with_context(|| {
            format!(
                "{}\n`train_mass_for_loco` must be provided for `HybridLoco` ",
                format_dbg!()
            )
        })?;
        let train_speed: si::Velocity = train_speed.with_context(|| {
            format!(
                "{}\n`train_speed` must be provided for `HybridLoco` ",
                format_dbg!()
            )
        })?;
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
                todo!("placeholder")
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
                        - train_speed.powi(typenum::P2::new())))
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
                (0.5 * mass_for_loco
                    * (train_speed.powi(typenum::P2::new())
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

        self.res.set_curr_pwr_out_max(
            dt,
            pwr_aux.with_context(|| format!("{}\nExpected `pwr_aux` to be Some", format_dbg!()))?,
            disch_buffer,
            chrg_buffer,
        )?;

        self.gen
            .set_cur_pwr_max_out(self.fc.state.pwr_out_max, Some(si::Power::ZERO))?;

        self.edrv.set_cur_pwr_max_out(
            self.gen.state.pwr_elec_prop_out_max + self.res.state.pwr_prop_max,
            None,
        )?;

        self.edrv
            .set_cur_pwr_regen_max(self.res.state.pwr_charge_max)?;

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
    /// # Arguments:
    /// - `pwr_out_req`: tractive power require
    /// - `train_speed`: current train speed
    /// - `dt`: time step size
    ///
    /// Controls decide the split between the alternator and the battery
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        train_mass: si::Mass,
        train_speed: si::Velocity,
        dt: si::Time,
        pwr_aux: si::Power,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        self.edrv.set_pwr_in_req(pwr_out_req, dt)?;
        let (gen_pwr_out_req, res_pwr_out_req) = self
            .pt_cntrl
            .get_pwr_gen_and_res(
                self.edrv.state.pwr_elec_prop_in,
                train_mass,
                train_speed,
                &mut self.state,
                &self.fc,
                &self.gen,
                &self.edrv.state,
                &self.res,
            )
            .with_context(|| format_dbg!())?;

        let fc_on: bool = !self.state.fc_on_causes.is_empty();
        self.gen
            .set_pwr_in_req(
                // TODO: maybe this should be zero if not loco_on
                gen_pwr_out_req,
                si::Power::ZERO,
                // if fc_on { pwr_aux } else { si::Power::ZERO },
                fc_on,
                dt,
            )
            .with_context(|| {
                format!(
                    "{}\n
{}",
                    format_dbg!(self.state.fc_on_causes),
                    format_dbg!(fc_on)
                )
            })?;
        let fc_pwr_mech_out = self.gen.state.pwr_mech_in;

        self.fc
            .solve_energy_consumption(fc_pwr_mech_out, dt, fc_on, assert_limits)
            .with_context(|| {
                format!(
                    "{}

{} kW

{} kW

{} kW
{} kW
{} kW
{} kW

{} kW

{} kW
{} kW
{} kW
{} kW",
                    format_dbg!(fc_on).replace("\"", ""),
                    format_dbg!(pwr_out_req.get::<si::kilowatt>()).replace("\"", ""),
                    format_dbg!(self.edrv.state.pwr_elec_prop_in.get::<si::kilowatt>())
                        .replace("\"", ""),
                    format_dbg!(gen_pwr_out_req.get::<si::kilowatt>()).replace("\"", ""),
                    format_dbg!(self.gen.state.pwr_elec_aux.get::<si::kilowatt>())
                        .replace("\"", ""),
                    format_dbg!(self.gen.state.pwr_elec_out_max.get::<si::kilowatt>())
                        .replace("\"", ""),
                    format_dbg!(self.gen.state.pwr_mech_in.get::<si::kilowatt>()).replace("\"", ""),
                    format_dbg!(res_pwr_out_req.get::<si::kilowatt>()).replace("\"", ""),
                    format_dbg!(self.res.state.pwr_prop_max.get::<si::kilowatt>())
                        .replace("\"", ""),
                    format_dbg!(self.res.state.pwr_disch_max.get::<si::kilowatt>())
                        .replace("\"", ""),
                    format_dbg!(self.res.state.pwr_regen_max.get::<si::kilowatt>())
                        .replace("\"", ""),
                    format_dbg!(self.res.state.pwr_charge_max.get::<si::kilowatt>())
                        .replace("\"", ""),
                )
            })?;

        self.res
            .solve_energy_consumption(
                res_pwr_out_req,
                pwr_aux,
                // if !fc_on { pwr_aux } else { si::Power::ZERO },
                dt,
            )
            .with_context(|| format_dbg!())?;

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
    pub fn pop(&mut self) -> Option<FCOnCause> {
        self.0.pop()
    }

    pub fn push(&mut self, new: FCOnCause) {
        self.0.push(new)
    }

    pub fn is_empty(&mut self) -> bool {
        self.0.is_empty()
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
    fn init(&mut self) -> Result<(), Error> {
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
    mass: si::Mass,
    train_speed: si::Velocity,
) -> anyhow::Result<()> {
    rgwdb.set_soc_fc_on_buffer(res, mass, train_speed)?;
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
    gen_state: &GeneratorState,
    res_state: &ReversibleEnergyStorageState,
    hev_state: &mut HELState,
) -> Result<(), anyhow::Error> {
    let frac_pwr_demand_fc_forced_on: si::Ratio = rgwdb
        .frac_pwr_demand_fc_forced_on
        .with_context(|| format_dbg!())?;
    if pwr_out_req > frac_pwr_demand_fc_forced_on * res_state.pwr_disch_max {
        hev_state
            .fc_on_causes
            .push(FCOnCause::PropulsionPowerDemandSoft);
    }
    if pwr_out_req - gen_state.pwr_elec_out_max >= si::Power::ZERO {
        hev_state
            .fc_on_causes
            .push(FCOnCause::PropulsionPowerDemand);
    }
    Ok(())
}

/// Determines whether enigne must be on for high speed
fn handle_fc_on_causes_for_speed(
    train_speed: si::Velocity,
    rgwdb: &mut Box<RESGreedyWithDynamicBuffers>,
    hev_state: &mut HELState,
) -> anyhow::Result<()> {
    if train_speed > rgwdb.speed_fc_forced_on.with_context(|| format_dbg!())? {
        hev_state.fc_on_causes.push(FCOnCause::VehicleSpeedTooHigh);
    }
    Ok(())
}

impl HybridPowertrainControls {
    /// Determines power split between engine/generator and battery
    ///
    /// # Arguments
    /// - `pwr_res_and_gen_to_edrv`: tractive power required as input to [ElectricDrivetrain]
    /// - `veh_state`: vehicle state
    /// - `hev_state`: HEV powertrain state
    /// - `fc`: fuel converter
    /// - `em_state`: electric machine state
    /// - `res`: reversible energy storage (e.g. high voltage battery)
    #[allow(clippy::too_many_arguments)]
    fn get_pwr_gen_and_res(
        &mut self,
        pwr_res_and_gen_to_edrv: si::Power,
        // TODO: make a `TrainMomentum` object to pass both mass and speed
        train_mass: si::Mass,
        train_speed: si::Velocity,
        hel_state: &mut HELState,
        fc: &FuelConverter,
        gen: &Generator,
        edrv_state: &ElectricDrivetrainState,
        res: &ReversibleEnergyStorage,
    ) -> anyhow::Result<(si::Power, si::Power)> {
        let fc_state = &fc.state;
        ensure!(
            // `almost` is in case of negligible numerical precision discrepancies
            almost_le_uom(
                &pwr_res_and_gen_to_edrv,
                &(gen.state.pwr_elec_prop_out_max + res.state.pwr_prop_max),
                None
            ),
            "{}
`pwr_out_req`: {} kW
`em_state.pwr_mech_fwd_out_max`: {} kW
`fc_state.pwr_prop_max`: {} kW
`res.state.soc`: {}",
            format_dbg!(),
            pwr_res_and_gen_to_edrv.get::<si::kilowatt>(),
            edrv_state.pwr_mech_out_max.get::<si::kilowatt>(),
            fc_state.pwr_out_max.get::<si::kilowatt>(),
            res.state.soc.get::<si::ratio>()
        );

        let (gen_prop_pwr, res_prop_pwr) = match self {
            Self::RGWDB(ref mut rgwdb) => {
                // handle_fc_on_causes_for_temp(fc, rgwdb, hev_state)?;
                handle_fc_on_causes_for_speed(train_speed, rgwdb, hel_state)?;
                handle_fc_on_causes_for_low_soc(res, rgwdb, hel_state, train_mass, train_speed)?;
                // `handle_fc_*` below here are asymmetrical for positive tractive power only
                handle_fc_on_causes_for_pwr_demand(
                    rgwdb,
                    pwr_res_and_gen_to_edrv,
                    &gen.state,
                    &res.state,
                    hel_state,
                )?;

                let res_prop_pwr = pwr_res_and_gen_to_edrv
                    .min(res.state.pwr_prop_max)
                    .max(-res.state.pwr_regen_max);

                if hel_state.fc_on_causes.is_empty() {
                    // engine is off, and `em_pwr` has already been limited within bounds
                    ensure!(
                        res_prop_pwr == pwr_res_and_gen_to_edrv,
                        format!(
                            "{}\n{}\n`res_prop_pwr` must be able to handle everything when engine is off",
                            format_dbg!(res_prop_pwr.get::<si::kilowatt>()),
                            format_dbg!(pwr_res_and_gen_to_edrv.get::<si::kilowatt>())
                        )
                    );
                    (si::Power::ZERO, pwr_res_and_gen_to_edrv)
                } else {
                    // engine has been forced on
                    let pwr_gen_elec_out_for_eff_fc =
                        get_pwr_gen_elec_out_for_eff_fc(fc, gen, rgwdb)?;
                    let gen_pwr = if pwr_res_and_gen_to_edrv < si::Power::ZERO {
                        // negative tractive power
                        // max power system can receive from engine during negative traction
                        (res.state.pwr_regen_max + pwr_res_and_gen_to_edrv)
                            // or peak efficiency power if it's lower than above
                            .min(pwr_gen_elec_out_for_eff_fc)
                            // but not negative
                            .max(si::Power::ZERO)
                    } else {
                        // positive tractive power
                        if pwr_res_and_gen_to_edrv - res_prop_pwr > pwr_gen_elec_out_for_eff_fc {
                            // engine needs to run higher than peak efficiency point
                            pwr_res_and_gen_to_edrv - res_prop_pwr
                        } else {
                            // engine does not need to run higher than peak
                            // efficiency point to make tractive demand

                            // gen handles all power not covered by em
                            (pwr_res_and_gen_to_edrv - res_prop_pwr)
                                // and if that's less than the
                                // efficiency-focused value, then operate at
                                // that value
                                .max(pwr_gen_elec_out_for_eff_fc)
                                // but don't exceed what what the battery can
                                // absorb + tractive demand
                                .min(pwr_res_and_gen_to_edrv + res.state.pwr_regen_max)
                        }
                    }
                    // and don't exceed what the fc -> gen can do
                    .min(gen.state.pwr_elec_prop_out_max);

                    // recalculate `em_pwr` based on `fc_pwr`
                    let res_pwr_corrected =
                        (pwr_res_and_gen_to_edrv - gen_pwr).max(-res.state.pwr_regen_max);
                    (gen_pwr, res_pwr_corrected)
                }
            }
            Self::Placeholder => todo!(),
        };

        ensure!(
            almost_le_uom(&res_prop_pwr, &res.state.pwr_prop_max, None),
            format!(
                "{}\n{}",
                format_dbg!(res_prop_pwr),
                format_dbg!(res.state.pwr_prop_max)
            )
        );

        Ok((gen_prop_pwr, res_prop_pwr))
    }
}

fn get_pwr_gen_elec_out_for_eff_fc(
    fc: &FuelConverter,
    gen: &Generator,
    rgwdb: &mut Box<RESGreedyWithDynamicBuffers>,
) -> anyhow::Result<si::Power> {
    let pwr_gen_elec_out_for_eff_fc: si::Power =
        if let Some(pwr_gen_elec_out_for_eff_fc) = rgwdb.pwr_gen_elec_out_for_eff_fc {
            pwr_gen_elec_out_for_eff_fc
        } else {
            let frac_of_pwr_for_peak_eff: si::Ratio = rgwdb
                .frac_of_max_pwr_to_run_fc
                .with_context(|| format_dbg!())?;
            let mut gen = gen.clone();
            // this assumes that the generator has a fairly flat efficiency curve
            gen.set_cur_pwr_max_out(
                frac_of_pwr_for_peak_eff * fc.pwr_out_max,
                Some(si::Power::ZERO),
            )
            .with_context(|| format_dbg!())?;
            rgwdb.pwr_gen_elec_out_for_eff_fc = Some(gen.state.pwr_elec_out_max);
            gen.state.pwr_elec_out_max
        };

    Ok(pwr_gen_elec_out_for_eff_fc)
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
    // TODO: make it so that the engine never goes off at all
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
    /// Force engine, if on, to run at this fraction of peak power or the
    /// required power, whichever is greater. If SOC is below min buffer or
    /// engine is otherwise forced on and battery has room to receive charge,
    /// engine will run at this level and charge.
    #[api(skip_get, skip_set)]
    pub frac_of_max_pwr_to_run_fc: Option<si::Ratio>,
    /// Force generator, if engine is on, to run at this power to help run engine efficiently
    #[api(skip_get, skip_set)]
    pub pwr_gen_elec_out_for_eff_fc: Option<si::Power>,
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

impl RESGreedyWithDynamicBuffers {
    fn set_soc_fc_on_buffer(
        &mut self,
        res: &ReversibleEnergyStorage,
        mass: si::Mass,
        train_speed: si::Velocity,
    ) -> anyhow::Result<()> {
        self.state.soc_fc_on_buffer = {
            let energy_delta_to_buffer_speed: si::Energy = 0.5
                * mass
                * (self
                    .speed_soc_fc_on_buffer
                    .with_context(|| format_dbg!())?
                    .powi(typenum::P2::new())
                    - train_speed.powi(typenum::P2::new()));
            energy_delta_to_buffer_speed.max(si::Energy::ZERO)
                * self
                    .speed_soc_fc_on_buffer_coeff
                    .with_context(|| format_dbg!())?
        } / res.energy_capacity_usable()
            + res.min_soc;
        Ok(())
    }
}

impl Init for RESGreedyWithDynamicBuffers {
    fn init(&mut self) -> Result<(), Error> {
        // TODO: tunnel buffer!
        // TODO: make sure these values propagate to the documented defaults above
        init_opt_default!(self, speed_soc_disch_buffer, 40.0 * uc::MPH);
        init_opt_default!(self, speed_soc_disch_buffer_coeff, 1.0 * uc::R);
        init_opt_default!(self, speed_soc_fc_on_buffer, 100.0 * uc::MPH);
        init_opt_default!(self, speed_soc_fc_on_buffer_coeff, 1.0 * uc::R);
        init_opt_default!(self, speed_soc_regen_buffer, 10. * uc::MPH);
        init_opt_default!(self, speed_soc_regen_buffer_coeff, 1.0 * uc::R);
        init_opt_default!(self, fc_min_time_on, uc::S * 5.0);
        // Force FC to be on all the time by default
        init_opt_default!(self, speed_fc_forced_on, uc::MPH * 0.0);
        init_opt_default!(self, frac_pwr_demand_fc_forced_on, uc::R * 0.75);
        // 20% of peak power gets most of peak efficiency
        init_opt_default!(self, frac_of_max_pwr_to_run_fc, 0.2 * uc::R);
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
