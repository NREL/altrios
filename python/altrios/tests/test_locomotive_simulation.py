import unittest

import altrios as alt
from .mock_resources import *


class TestLocomotiveSimulation(unittest.TestCase):
    def test_walk_no_save(self):
        mock_sim = mock_locomotive_simulation(save_interval=None)

        mock_sim.walk()

    def test_walk_w_save(self):
        mock_sim = mock_locomotive_simulation(save_interval=1)

        mock_sim.walk()

    def test_to_from_json(self):
        mock_sim = mock_locomotive_simulation()

        j = mock_sim.to_json()
        mock_sim2 = alt.LocomotiveSimulation.from_json(j)

        self.assertEqual(
            mock_sim.power_trace.time_seconds.tolist(),
            mock_sim2.power_trace.time_seconds.tolist(),
        )
