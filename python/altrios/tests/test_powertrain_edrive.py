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
