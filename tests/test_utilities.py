import unittest

from .mock_resources import *

from altrios.utilities import set_param_from_path


class TestUtilities(unittest.TestCase):
    def test_set_param(self):
        c = mock_consist()

        c = set_param_from_path(c, "loco_vec[0].state.i", 10)

        self.assertEqual(c.loco_vec.tolist()[0].state.i, 10)

        c = set_param_from_path(c, "state.pwr_fuel_watts", -100)

        self.assertEqual(c.state.pwr_fuel_watts, -100)
