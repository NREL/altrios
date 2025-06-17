import unittest

from .mock_resources import *
import altrios as alt


class TestLocomotive(unittest.TestCase):
    def test_conventional_loco_to_from_json(self):
        loco1 = mock_conventional_loco()

        j = loco1.to_json()
        loco2 = alt.Locomotive.from_json(j)
        self.assertEqual(loco1.state.pwr_out_max_watts, loco2.state.pwr_out_max_watts)

    def test_hybrid_loco_to_from_json(self):
        loco1 = mock_hybrid_loco()

        j = loco1.to_json()
        loco2 = alt.Locomotive.from_json(j)
        self.assertEqual(loco1.state.pwr_out_max_watts, loco2.state.pwr_out_max_watts)
