import tempfile
import unittest
from pathlib import Path

from .mock_resources import *

from altrios.utilities import set_param_from_path
import altrios as alt


class TestUtilities(unittest.TestCase):
    def test_set_param(self):
        c = mock_consist()

        c = set_param_from_path(c, "loco_vec[0].state.i", 10)

        self.assertEqual(c.loco_vec.tolist()[0].state.i, 10)

        c = set_param_from_path(c, "state.pwr_fuel_watts", -100)

        self.assertEqual(c.state.pwr_fuel_watts, -100)

    def test_copy_demo_files(self):
        v = f"v{alt.__version__}"
        prepend_str = f"# %% Copied from ALTRIOS version '{v}'. Guaranteed compatibility with this version only.\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            tf_path = Path(tmpdir)
            alt.copy_demo_files(tf_path)
            with open(next(tf_path.glob("*demo*.py")), 'r') as file:
                lines = file.readlines()
                assert prepend_str in lines[0]
                assert len(lines) > 3
