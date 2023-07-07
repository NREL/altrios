import unittest


import altrios as alt
from .mock_resources import *


class TestConsistSimulation(unittest.TestCase):
    def test_walk_no_save(self):
        mock_sim = mock_consist_simulation(save_interval=None)

        mock_sim.walk()

    def test_walk_w_save(self):
        mock_sim = mock_consist_simulation(save_interval=1)

        mock_sim.walk()

    def test_to_from_json(self):
        mock_sim = mock_consist_simulation()

        j = mock_sim.to_json()
        mock_sim2 = alt.ConsistSimulation.from_json(j)

        self.assertEqual(
            mock_sim.power_trace.time_seconds.tolist(),
            mock_sim2.power_trace.time_seconds.tolist(),
        )
