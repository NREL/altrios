import unittest

import altrios as alt
from .mock_resources import *


class TestGenerator(unittest.TestCase):

    def test_to_from_json(self):
        gen1 = mock_generator()

        j = gen1.to_json()
        gen2 = alt.Generator.from_json(j)

        self.assertEqual(gen1.eta_interp.tolist(), gen2.eta_interp.tolist())

    def test_set_nested_state_error(self):
        gen = mock_generator()

        with self.assertRaises(AttributeError):
            # not allowed to set value on nested state
            gen.state.pwr_loss_watts = 0.5

    def test_set_nested_state_proper(self):
        gen = mock_generator()

        alt.set_param_from_path(gen, "state.pwr_loss_watts", 1.0)

        self.assertEqual(gen.state.pwr_loss_watts, 1.0)
