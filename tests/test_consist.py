import unittest

import altrios as alt
from .mock_resources import *


class TestConsist(unittest.TestCase):
    def test_consist_to_from_json(self):
        consist1 = mock_consist()

        j = consist1.to_json()
        consist2 = alt.Consist.from_json(j)
        self.assertEqual(
            consist1.state.pwr_out_max_watts, consist2.state.pwr_out_max_watts
        )

    def test_conv_nested_state_error(self):
        consist = mock_consist()

        with self.assertRaises(AttributeError):
            # not allowed to set value on nested state
            consist.state.pwr_out_watts = 0.5
