import unittest
import altrios as alt
from .mock_resources import *

class TestParamPath(unittest.TestCase):
    def test_variable_path_list(self):
        from altrios import LocomotiveSimulation, Locomotive, PowerTrace
        mock_df = mock_pymoo_conv_cal_df()
        pt = PowerTrace(
            mock_df['time [s]'],
            mock_df['Tractive Power [W]'],
            engine_on=mock_df['engine_on'],
        )
        loco_sim = LocomotiveSimulation(
            loco_unit=Locomotive.default(),
            power_trace=pt,
            save_interval=1)

        baseline_variable_paths = ['power_trace.time_seconds', 'power_trace.pwr_watts', 'power_trace.time_hours', 'power_trace.engine_on', 'loco_unit.edrv.save_interval', 
                                   'loco_unit.edrv.pwr_out_max_watts', 'loco_unit.edrv.eta_max', 'loco_unit.edrv.eta_min', 'loco_unit.edrv.state.eta', 
                                   'loco_unit.edrv.state.pwr_elec_prop_in_watts', 'loco_unit.edrv.state.pwr_rate_out_max_watts_per_second', 'loco_unit.edrv.state.energy_loss_joules', 
                                   'loco_unit.edrv.state.energy_mech_prop_out_joules', 'loco_unit.edrv.state.pwr_mech_out_max_watts', 'loco_unit.edrv.state.energy_elec_dyn_brake_joules', 
                                   'loco_unit.edrv.state.energy_elec_prop_in_joules', 'loco_unit.edrv.state.pwr_out_req_watts', 'loco_unit.edrv.state.pwr_mech_dyn_brake_watts', 
                                   'loco_unit.edrv.state.i', 'loco_unit.edrv.state.energy_mech_dyn_brake_joules', 'loco_unit.edrv.state.pwr_mech_regen_max_watts', 
                                   'loco_unit.edrv.state.pwr_mech_prop_out_watts', 'loco_unit.edrv.state.pwr_elec_dyn_brake_watts', 'loco_unit.edrv.state.pwr_loss_watts', 
                                   'loco_unit.edrv.eta_range', 'loco_unit.edrv.pwr_out_frac_interp', 'loco_unit.edrv.pwr_in_frac_interp', 'loco_unit.edrv.history.pwr_elec_dyn_brake_watts', 
                                   'loco_unit.edrv.history.pwr_elec_prop_in_watts', 'loco_unit.edrv.history.pwr_mech_out_max_watts', 'loco_unit.edrv.history.energy_loss_joules', 
                                   'loco_unit.edrv.history.energy_elec_dyn_brake_joules', 'loco_unit.edrv.history.pwr_mech_regen_max_watts', 
                                   'loco_unit.edrv.history.energy_mech_dyn_brake_joules', 'loco_unit.edrv.history.eta', 'loco_unit.edrv.history.pwr_loss_watts', 
                                   'loco_unit.edrv.history.pwr_rate_out_max_watts_per_second', 'loco_unit.edrv.history.pwr_mech_dyn_brake_watts', 'loco_unit.edrv.history.pwr_out_req_watts', 
                                   'loco_unit.edrv.history.pwr_mech_prop_out_watts', 'loco_unit.edrv.history.energy_mech_prop_out_joules', 'loco_unit.edrv.history.energy_elec_prop_in_joules', 
                                   'loco_unit.edrv.history.i', 'loco_unit.edrv.eta_interp', 'loco_unit.assert_limits', 'loco_unit.mass_kg', 'loco_unit.baseline_mass_kg', 
                                   'loco_unit.force_max_pounds', 'loco_unit.gen.eta_max', 'loco_unit.gen.eta_min', 'loco_unit.gen.state.pwr_elec_prop_out_max_watts', 
                                   'loco_unit.gen.state.pwr_elec_aux_watts', 'loco_unit.gen.state.pwr_elec_out_max_watts', 'loco_unit.gen.state.energy_elec_aux_joules', 
                                   'loco_unit.gen.state.energy_mech_in_joules', 'loco_unit.gen.state.energy_loss_joules', 'loco_unit.gen.state.pwr_loss_watts', 
                                   'loco_unit.gen.state.energy_elec_prop_out_joules', 'loco_unit.gen.state.eta', 'loco_unit.gen.state.pwr_rate_out_max_watts_per_second', 
                                   'loco_unit.gen.state.pwr_mech_in_watts', 'loco_unit.gen.state.pwr_elec_prop_out_watts', 'loco_unit.gen.state.i', 'loco_unit.gen.pwr_out_max_watts', 
                                   'loco_unit.gen.eta_range', 'loco_unit.gen.history.energy_loss_joules', 'loco_unit.gen.history.pwr_mech_in_watts', 
                                   'loco_unit.gen.history.pwr_rate_out_max_watts_per_second', 'loco_unit.gen.history.i', 'loco_unit.gen.history.pwr_elec_prop_out_max_watts', 
                                   'loco_unit.gen.history.pwr_elec_out_max_watts', 'loco_unit.gen.history.pwr_elec_prop_out_watts', 'loco_unit.gen.history.energy_mech_in_joules', 
                                   'loco_unit.gen.history.energy_elec_prop_out_joules', 'loco_unit.gen.history.energy_elec_aux_joules', 'loco_unit.gen.history.pwr_elec_aux_watts', 
                                   'loco_unit.gen.history.eta', 'loco_unit.gen.history.pwr_loss_watts', 'loco_unit.gen.pwr_out_frac_interp', 'loco_unit.gen.mass_kg', 
                                   'loco_unit.gen.pwr_in_frac_interp', 'loco_unit.gen.save_interval', 'loco_unit.gen.eta_interp', 'loco_unit.gen.specific_pwr_kw_per_kg', 
                                   'loco_unit.history.pwr_out_watts', 'loco_unit.history.pwr_out_max_watts', 'loco_unit.history.pwr_regen_max_watts', 'loco_unit.history.energy_out_joules', 
                                   'loco_unit.history.pwr_rate_out_max_watts_per_second', 'loco_unit.history.i', 'loco_unit.history.pwr_aux_watts', 'loco_unit.history.energy_aux_joules', 
                                   'loco_unit.fuel_res_ratio', 'loco_unit.pwr_aux_offset_watts', 'loco_unit.res', 'loco_unit.ballast_mass_kg', 'loco_unit.state.pwr_out_max_watts', 
                                   'loco_unit.state.pwr_regen_max_watts', 'loco_unit.state.pwr_out_watts', 'loco_unit.state.pwr_aux_watts', 'loco_unit.state.i', 
                                   'loco_unit.state.pwr_rate_out_max_watts_per_second', 'loco_unit.state.energy_aux_joules', 'loco_unit.state.energy_out_joules', 
                                   'loco_unit.fuel_res_split', 'loco_unit.pwr_aux_traction_coeff', 'loco_unit.force_max_newtons', 'loco_unit.fc.state.eta', 
                                   'loco_unit.fc.state.pwr_idle_fuel_watts', 'loco_unit.fc.state.pwr_brake_watts', 'loco_unit.fc.state.pwr_out_max_watts', 'loco_unit.fc.state.pwr_loss_watts', 
                                   'loco_unit.fc.state.energy_brake_joules', 'loco_unit.fc.state.energy_fuel_joules', 'loco_unit.fc.state.i', 'loco_unit.fc.state.pwr_fuel_watts', 
                                   'loco_unit.fc.state.engine_on', 'loco_unit.fc.state.energy_loss_joules', 'loco_unit.fc.state.energy_idle_fuel_joules', 'loco_unit.fc.eta_interp', 
                                   'loco_unit.fc.pwr_out_max_watts', 'loco_unit.fc.eta_max', 'loco_unit.fc.eta_min', 'loco_unit.fc.specific_pwr_kw_per_kg', 'loco_unit.fc.pwr_ramp_lag_hours', 
                                   'loco_unit.fc.pwr_idle_fuel_watts', 'loco_unit.fc.save_interval', 'loco_unit.fc.history.energy_brake_joules', 'loco_unit.fc.history.energy_fuel_joules', 
                                   'loco_unit.fc.history.energy_idle_fuel_joules', 'loco_unit.fc.history.pwr_fuel_watts', 'loco_unit.fc.history.i', 'loco_unit.fc.history.pwr_out_max_watts', 
                                   'loco_unit.fc.history.engine_on', 'loco_unit.fc.history.pwr_loss_watts', 'loco_unit.fc.history.pwr_idle_fuel_watts', 
                                   'loco_unit.fc.history.energy_loss_joules', 'loco_unit.fc.history.pwr_brake_watts', 'loco_unit.fc.history.eta', 'loco_unit.fc.eta_range', 
                                   'loco_unit.fc.pwr_out_max_init_watts', 'loco_unit.fc.pwr_out_frac_interp', 'loco_unit.fc.mass_kg', 'loco_unit.fc.pwr_ramp_lag_seconds', 
                                   'loco_unit.pwr_rated_kilowatts', 'i']
        
        baseline_history_variable_paths = ['loco_unit.edrv.history.pwr_elec_dyn_brake_watts', 'loco_unit.edrv.history.pwr_elec_prop_in_watts', 
                                           'loco_unit.edrv.history.pwr_mech_out_max_watts', 'loco_unit.edrv.history.energy_loss_joules', 
                                           'loco_unit.edrv.history.energy_elec_dyn_brake_joules', 'loco_unit.edrv.history.pwr_mech_regen_max_watts', 
                                           'loco_unit.edrv.history.energy_mech_dyn_brake_joules', 'loco_unit.edrv.history.eta', 'loco_unit.edrv.history.pwr_loss_watts', 
                                           'loco_unit.edrv.history.pwr_rate_out_max_watts_per_second', 'loco_unit.edrv.history.pwr_mech_dyn_brake_watts', 
                                           'loco_unit.edrv.history.pwr_out_req_watts', 'loco_unit.edrv.history.pwr_mech_prop_out_watts', 
                                           'loco_unit.edrv.history.energy_mech_prop_out_joules', 'loco_unit.edrv.history.energy_elec_prop_in_joules', 'loco_unit.edrv.history.i', 
                                           'loco_unit.gen.history.energy_loss_joules', 'loco_unit.gen.history.pwr_mech_in_watts', 'loco_unit.gen.history.pwr_rate_out_max_watts_per_second', 
                                           'loco_unit.gen.history.i', 'loco_unit.gen.history.pwr_elec_prop_out_max_watts', 'loco_unit.gen.history.pwr_elec_out_max_watts', 
                                           'loco_unit.gen.history.pwr_elec_prop_out_watts', 'loco_unit.gen.history.energy_mech_in_joules', 
                                           'loco_unit.gen.history.energy_elec_prop_out_joules', 'loco_unit.gen.history.energy_elec_aux_joules', 
                                           'loco_unit.gen.history.pwr_elec_aux_watts', 'loco_unit.gen.history.eta', 'loco_unit.gen.history.pwr_loss_watts', 
                                           'loco_unit.history.pwr_out_watts', 'loco_unit.history.pwr_out_max_watts', 'loco_unit.history.pwr_regen_max_watts', 
                                           'loco_unit.history.energy_out_joules', 'loco_unit.history.pwr_rate_out_max_watts_per_second', 'loco_unit.history.i', 
                                           'loco_unit.history.pwr_aux_watts', 'loco_unit.history.energy_aux_joules', 'loco_unit.fc.history.energy_brake_joules', 
                                           'loco_unit.fc.history.energy_fuel_joules', 'loco_unit.fc.history.energy_idle_fuel_joules', 'loco_unit.fc.history.pwr_fuel_watts', 
                                           'loco_unit.fc.history.i', 'loco_unit.fc.history.pwr_out_max_watts', 'loco_unit.fc.history.engine_on', 'loco_unit.fc.history.pwr_loss_watts', 
                                           'loco_unit.fc.history.pwr_idle_fuel_watts', 'loco_unit.fc.history.energy_loss_joules', 'loco_unit.fc.history.pwr_brake_watts', 
                                           'loco_unit.fc.history.eta', 'loco_unit.edrv.history.pwr_elec_dyn_brake_watts', 'loco_unit.edrv.history.pwr_elec_prop_in_watts', 
                                           'loco_unit.edrv.history.pwr_mech_out_max_watts', 'loco_unit.edrv.history.energy_loss_joules', 
                                           'loco_unit.edrv.history.energy_elec_dyn_brake_joules', 'loco_unit.edrv.history.pwr_mech_regen_max_watts', 
                                           'loco_unit.edrv.history.energy_mech_dyn_brake_joules', 'loco_unit.edrv.history.eta', 'loco_unit.edrv.history.pwr_loss_watts', 
                                           'loco_unit.edrv.history.pwr_rate_out_max_watts_per_second', 'loco_unit.edrv.history.pwr_mech_dyn_brake_watts', 
                                           'loco_unit.edrv.history.pwr_out_req_watts', 'loco_unit.edrv.history.pwr_mech_prop_out_watts', 'loco_unit.edrv.history.energy_mech_prop_out_joules', 
                                           'loco_unit.edrv.history.energy_elec_prop_in_joules', 'loco_unit.edrv.history.i', 'loco_unit.gen.history.energy_loss_joules', 
                                           'loco_unit.gen.history.pwr_mech_in_watts', 'loco_unit.gen.history.pwr_rate_out_max_watts_per_second', 'loco_unit.gen.history.i', 
                                           'loco_unit.gen.history.pwr_elec_prop_out_max_watts', 'loco_unit.gen.history.pwr_elec_out_max_watts', 'loco_unit.gen.history.pwr_elec_prop_out_watts', 
                                           'loco_unit.gen.history.energy_mech_in_joules', 'loco_unit.gen.history.energy_elec_prop_out_joules', 'loco_unit.gen.history.energy_elec_aux_joules', 
                                           'loco_unit.gen.history.pwr_elec_aux_watts', 'loco_unit.gen.history.eta', 'loco_unit.gen.history.pwr_loss_watts', 'loco_unit.history.pwr_out_watts', 
                                           'loco_unit.history.pwr_out_max_watts', 'loco_unit.history.pwr_regen_max_watts', 'loco_unit.history.energy_out_joules', 
                                           'loco_unit.history.pwr_rate_out_max_watts_per_second', 'loco_unit.history.i', 'loco_unit.history.pwr_aux_watts', 
                                           'loco_unit.history.energy_aux_joules', 'loco_unit.fc.history.energy_brake_joules', 'loco_unit.fc.history.energy_fuel_joules', 
                                           'loco_unit.fc.history.energy_idle_fuel_joules', 'loco_unit.fc.history.pwr_fuel_watts', 'loco_unit.fc.history.i', 
                                           'loco_unit.fc.history.pwr_out_max_watts', 'loco_unit.fc.history.engine_on', 'loco_unit.fc.history.pwr_loss_watts', 
                                           'loco_unit.fc.history.pwr_idle_fuel_watts', 'loco_unit.fc.history.energy_loss_joules', 'loco_unit.fc.history.pwr_brake_watts', 
                                           'loco_unit.fc.history.eta']
        # compare relative variable paths within fuel converter to baseline
        assert(baseline_variable_paths.sort()==loco_sim.variable_path_list().sort())
        # self.assertAlmostEqual(baseline_variable_paths, loco_sim.variable_path_list())
        # compare relative history variable paths within fuel converter to baseline
        assert(baseline_history_variable_paths.sort()==loco_sim.history_path_list().sort())
        # self.assertAlmostEqual(baseline_history_variable_paths, loco_sim.history_path_list())


if __name__ == "__main__":
    a = TestParamPath()
    a.test_variable_path_list()