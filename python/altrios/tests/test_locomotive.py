import unittest

from .mock_resources import *
import altrios as alt


class TestLocomotive(unittest.TestCase):
    def test_conventional_loco_to_from_json(self):
        loco1 = mock_conventional_loco()

        j = loco1.to_json()
        loco2 = alt.Locomotive.from_json(j)
        self.assertEqual(loco1.state.pwr_out_max_watts,
                         loco2.state.pwr_out_max_watts)

    def test_hybrid_loco_to_from_json(self):
        loco1 = mock_hybrid_loco()

        j = loco1.to_json()
        loco2 = alt.Locomotive.from_json(j)
        self.assertEqual(loco1.state.pwr_out_max_watts,
                         loco2.state.pwr_out_max_watts)

    def test_conv_nested_state_error(self):
        loco = mock_conventional_loco()

        with self.assertRaises(AttributeError):
            # not allowed to set value on nested state
            loco.state.pwr_out_watts = 0.5

    def test_conv_set_nested_state_proper(self):
        loco = mock_hybrid_loco()

        alt.set_param_from_path(loco, "state.pwr_out_watts", 1.0)

        self.assertEqual(loco.state.pwr_out_watts, 1.0)

    def test_hybrid_nested_state_error(self):
        loco = mock_hybrid_loco()

        with self.assertRaises(AttributeError):
            # not allowed to set value on nested state
            loco.state.pwr_out_watts = 0.5

    def test_hybrid_set_nested_state_proper(self):
        loco = mock_hybrid_loco()

        alt.set_param_from_path(loco, "state.pwr_out_watts", 1.0)

        self.assertEqual(loco.state.pwr_out_watts, 1.0)
