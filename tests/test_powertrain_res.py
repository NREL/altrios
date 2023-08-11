import unittest

from altrios import package_root

import altrios as alt
from .mock_resources import *


class TestRES(unittest.TestCase):
    def test_load_from_excel(self):
        test_file = (
            package_root()
            / "resources"
            / "powertrains"
            / "reversible_energy_storages"
            / "Kokam_NMC_75Ah.xlsx"
        )
        res = alt.ReversibleEnergyStorage.from_excel(test_file)

        self.assertEqual(res.min_soc, 0.1)
        self.assertEqual(res.max_soc, 0.9)

    def test_load_from_yaml_absolute_path(self):
        test_file = (
            package_root()
            / "resources"
            / "powertrains"
            / "reversible_energy_storages"
            / "Kokam_NMC_75Ah_flx_drive.yaml"
        )
        res = alt.ReversibleEnergyStorage.from_file(str(test_file))

        self.assertEqual(res.min_soc, 0.05)
        self.assertEqual(res.max_soc, 0.95)

    def test_to_from_json(self):
        res1 = mock_reversible_energy_storage()

        j = res1.to_json()
        res2 = alt.ReversibleEnergyStorage.from_json(j)

        self.assertEqual(res1.min_soc, res2.min_soc)

    def test_set_nested_state_error(self):
        res = mock_reversible_energy_storage()

        with self.assertRaises(AttributeError):
            # not allowed to set value on nested state
            res.state.min_soc = 0.5

    def test_set_nested_state_proper(self):
        res = mock_reversible_energy_storage()

        alt.set_param_from_path(res, "state.soc", 1.0)

        self.assertEqual(res.state.soc, 1.0)

    def test_get_set_eta_max(self):
        res = mock_reversible_energy_storage()

        alt.set_param_from_path(res, "eta_max", 0.8)

        self.assertEqual(res.eta_max, 0.8)
