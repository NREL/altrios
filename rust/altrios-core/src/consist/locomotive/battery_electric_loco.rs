use super::powertrain::electric_drivetrain::ElectricDrivetrain;
use super::powertrain::reversible_energy_storage::ReversibleEnergyStorage;
use super::powertrain::ElectricMachine;
use super::*;
use super::{LocoTrait, Mass, MassSideEffect};
use crate::imports::*;

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize, HistoryMethods)]
#[altrios_api]
/// Battery electric locomotive
pub struct BatteryElectricLoco {
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
    pub state: BELState,
    /// vector of [Self::state]
    #[serde(default, skip_serializing_if = "BELStateHistoryVec::is_empty")]
    pub history: BELStateHistoryVec,
}

impl BatteryElectricLoco {
    /// Solve energy consumption for the current power output required
    /// Arguments:
    /// - pwr_out_req: tractive power required
    /// - dt: time step size
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        pwr_aux: si::Power,
    ) -> anyhow::Result<()> {
        self.edrv.set_pwr_in_req(pwr_out_req, dt)?;
        if self.edrv.state.pwr_elec_prop_in > si::Power::ZERO {
            // positive traction
            self.res
                .solve_energy_consumption(self.edrv.state.pwr_elec_prop_in, pwr_aux, dt)?;
        } else {
            // negative traction
            self.res.solve_energy_consumption(
                self.edrv.state.pwr_elec_prop_in,
                // limit aux power to whatever is actually available
                pwr_aux
                    // whatever power is available from regen plus normal
                    .min(self.res.state.pwr_prop_max - self.edrv.state.pwr_elec_prop_in)
                    .max(si::Power::ZERO),
                dt,
            )?;
        }
        Ok(())
    }
}

impl Mass for BatteryElectricLoco {
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
            stringify!(BatteryElectricLoco)
        ))
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        self.res.mass().with_context(|| format_dbg!())
    }

    fn expunge_mass_fields(&mut self) {
        self.res.expunge_mass_fields();
    }
}

impl Init for BatteryElectricLoco {
    fn init(&mut self) -> anyhow::Result<()> {
        self.res.init()?;
        self.edrv.init()?;
        self.pt_cntrl.init()?;
        Ok(())
    }
}
impl SerdeAPI for BatteryElectricLoco {}

impl LocoTrait for BatteryElectricLoco {
    fn set_curr_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        _train_mass: Option<si::Mass>,
        _train_speed: Option<si::Velocity>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let disch_buffer: si::Energy = todo!(
            "Calculate buffers similarly to the `HybridPowertrainControls`, but
make a `BatteryPowertrainControls` for the BEL and/or a `HybridConsistControls`
thing at the consist level"
        );
        let chrg_buffer: si::Energy = todo!();

        self.res.set_curr_pwr_out_max(
            dt,
            pwr_aux.with_context(|| anyhow!(format_dbg!("`pwr_aux` not provided")))?,
            /// TODO: calculate buffers similarly to the `HybridPowertrainControls`, but make a `HybridConsistControls` things at the consist level
            disch_buffer,
            chrg_buffer,
        )?;
        self.edrv
            .set_cur_pwr_max_out(self.res.state.pwr_prop_max, None)?;
        self.edrv
            .set_cur_pwr_regen_max(self.res.state.pwr_charge_max)?;

        // power rate is never limiting in BEL, but assuming dt will be same
        // in next time step, we can synthesize a rate
        self.edrv.set_pwr_rate_out_max(
            (self.edrv.state.pwr_mech_out_max - self.edrv.state.pwr_mech_prop_out) / dt,
        );
        Ok(())
    }

    fn save_state(&mut self) {
        self.save_state();
    }

    fn step(&mut self) {
        self.step()
    }

    fn get_energy_loss(&self) -> si::Energy {
        self.res.state.energy_loss + self.edrv.state.energy_loss
    }
}

#[altrios_api]
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, HistoryVec)]
#[non_exhaustive]
#[serde(default)]
pub struct BELState {
    /// time step index
    pub i: usize,
}

impl Init for BELState {}
impl SerdeAPI for BELState {}
