import unittest


import altrios as alt
from .mock_resources import *


class TestElectricDrivetrain(unittest.TestCase):
    def test_to_from_json(self):
        gen1 = mock_electric_drivetrain()

        j = gen1.to_json()
        gen2 = alt.ElectricDrivetrain.from_json(j)

        self.assertEqual(gen1.pwr_out_max_watts, gen2.pwr_out_max_watts)
        self.assertEqual(gen1.eta_interp.tolist(), gen2.eta_interp.tolist())

    def test_set_nested_state_error(self):
        edrv = mock_electric_drivetrain()

        with self.assertRaises(AttributeError): 
            # not allowed to set value on nested state
            edrv.state.pwr_loss_watts = 0.5

    def test_set_nested_state_proper(self):
        edrv = mock_electric_drivetrain()

        alt.set_param_from_path(edrv, "state.pwr_loss_watts", 1.0)

        self.assertEqual(edrv.state.pwr_loss_watts, 1.0)
