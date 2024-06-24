import unittest
import altrios as alt

class TestParamPath(unittest.TestCase):
    def test_variable_path_list(self):
        gen = alt.Generator.default()

        with open(alt.resources_root() / "benchmark_variable_paths/generator_variable_paths.txt") as file:
            baseline_variable_paths = file.readlines()
        with open(alt.resources_root() / "benchmark_variable_paths/generator_history_paths.txt") as file:
            baseline_history_variable_paths = file.readlines()

        # compare relative variable paths within Generator to baseline
        assert(baseline_variable_paths.sort()==gen.variable_path_list().sort())

        # compare relative history variable paths within Generator to baseline
        assert(baseline_history_variable_paths.sort()==gen.history_path_list().sort())


if __name__ == "__main__":
    a = TestParamPath()
    a.test_variable_path_list()