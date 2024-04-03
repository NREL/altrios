use super::powertrain::electric_drivetrain::ElectricDrivetrain;
use super::powertrain::fuel_converter::FuelConverter;
use super::powertrain::generator::Generator;
use super::powertrain::ElectricMachine;
use super::LocoTrait;
use crate::imports::*;

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize, HistoryMethods, SerdeAPI)]
#[altrios_api(
    #[new]
    pub fn __new__(
        fuel_converter: FuelConverter,
        generator: Generator,
        electric_drivetrain: ElectricDrivetrain,
    ) -> Self {
        Self {
            fc: fuel_converter,
            gen: generator,
            edrv: electric_drivetrain,
        }
    }
)]
/// Conventional locomotive
pub struct ConventionalLoco {
    // The fields in this struct are all locally defined structs and are therefore not documented in
    // this context
    #[has_state]
    pub fc: FuelConverter,
    #[has_state]
    pub gen: Generator,
    #[has_state]
    pub edrv: ElectricDrivetrain,
}

impl ConventionalLoco {
    pub fn new(
        fuel_converter: FuelConverter,
        generator: Generator,
        electric_drivetrain: ElectricDrivetrain,
    ) -> Self {
        Self {
            fc: fuel_converter,
            gen: generator,
            edrv: electric_drivetrain,
        }
    }

    /// # Arguments
    /// - `pwr_out_req`: power required at the wheel/rail interface
    /// - `dt`: time step size
    /// - `engine_on`: whether engine is on (i.e. rotating and consuming at least idle fuel)
    /// - `pwr_aux`: power demand for auxilliary systems
    /// - `assert_limits`: whether to fail when powertrain capabilities are exceeded
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        engine_on: bool,
        pwr_aux: si::Power,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        self.edrv.set_pwr_in_req(pwr_out_req, dt)?;

        self.gen.set_pwr_in_req(
            self.edrv.state.pwr_elec_prop_in,
            if engine_on { pwr_aux } else { si::Power::ZERO },
            dt,
        )?;

        ensure!(
            self.gen.state.pwr_mech_in >= si::Power::ZERO,
            format!(
                "{}\nfc can only produce positive power",
                format_dbg!(self.gen.state.pwr_mech_in >= si::Power::ZERO)
            ),
        );
        self.fc.solve_energy_consumption(
            self.gen.state.pwr_mech_in,
            dt,
            engine_on,
            assert_limits,
        )?;
        Ok(())
    }
}

impl LocoTrait for ConventionalLoco {
    /// returns current max power, current max power rate, and current max regen
    /// power that can be absorbed by the RES/battery
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        self.fc.set_cur_pwr_out_max(dt)?;
        self.gen.set_cur_pwr_max_out(
            self.fc.state.pwr_out_max,
            Some(pwr_aux.with_context(|| format_dbg!("`pwr_aux` not provided"))?),
        )?;
        self.edrv
            .set_cur_pwr_max_out(self.gen.state.pwr_elec_prop_out_max, None)?;
        self.gen
            .set_pwr_rate_out_max(self.fc.pwr_out_max / self.fc.pwr_ramp_lag);
        self.edrv
            .set_pwr_rate_out_max(self.gen.state.pwr_rate_out_max);
        Ok(())
    }

    fn save_state(&mut self) {
        self.save_state();
    }

    fn step(&mut self) {
        self.step()
    }

    fn get_energy_loss(&self) -> si::Energy {
        self.fc.state.energy_loss + self.gen.state.energy_loss + self.edrv.state.energy_loss
    }
}
