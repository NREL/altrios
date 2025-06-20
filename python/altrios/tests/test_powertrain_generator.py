import unittest

import altrios as alt
from .mock_resources import *


class TestGenerator(unittest.TestCase):
    def test_to_from_json(self):
        gen1 = mock_generator()

        j = gen1.to_json()
        gen2 = alt.Generator.from_json(j)

        self.assertEqual(gen1.eta_interp.tolist(), gen2.eta_interp.tolist())
