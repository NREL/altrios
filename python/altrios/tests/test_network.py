import unittest
import altrios as alt
from altrios import resources_root
from altrios import utilities as utils


class TestNetwork(unittest.TestCase):
    def test_load_network(self):
        pass # TODO: When Garrett provides a publicly releaseable network file, fix this
        # this test and uncomment the next two lines
        # filename = resources_root() / "networks/bar_stock_simple_network.yaml"
        # _network = alt.import_network(str(filename))
        # TODO: Make another test for speed_limit_train_sims
        # sims = alt.build_and_run_speed_limit_train_sims(
        #     tsb, str(filename), rv_map)
        # sims.walk() # TODO: make and expose this method
