import unittest

from altrios import package_root

import altrios as alt
from .mock_resources import *


class TestFuelConverter(unittest.TestCase):
    def test_load_from_yaml_absolute_path(self):
        test_file = (
            package_root()
            / "resources"
            / "powertrains"
            / "fuel_converters"
            / "wabtec_tier4.yaml"
        )
        fc = alt.FuelConverter.from_file(str(test_file))

        self.assertEqual(fc.pwr_out_max_watts, 3.255e6)
        self.assertEqual(fc.pwr_idle_fuel_watts, 1.97032784e+04)

    def test_to_from_json(self):
        gen1 = mock_fuel_converter()

        j = gen1.to_json()
        gen2 = alt.FuelConverter.from_json(j)

        self.assertEqual(gen1.pwr_out_max_watts, gen2.pwr_out_max_watts)
        self.assertEqual(gen1.eta_interp.tolist(), gen2.eta_interp.tolist())

    def test_set_nested_state_error(self):
        fc = mock_fuel_converter()

        with self.assertRaises(AttributeError):
            # not allowed to set value on nested state
            fc.state.pwr_loss_watts = 0.5

    def test_set_nested_state_proper(self):
        fc = mock_fuel_converter()

        alt.set_param_from_path(fc, "state.pwr_loss_watts", 1.0)

        self.assertEqual(fc.state.pwr_loss_watts, 1.0)
