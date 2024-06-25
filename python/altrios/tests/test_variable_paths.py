import unittest
import altrios as alt

class TestParamPath(unittest.TestCase):
    def test_variable_path_list(self):
        gen = alt.Generator.default()

        with open(alt.resources_root() / "benchmark_variable_paths/generator_variable_paths.txt") as file:
            baseline_variable_paths = [line.strip() for line in file.readlines()]
        with open(alt.resources_root() / "benchmark_variable_paths/generator_history_paths.txt") as file:
            baseline_history_variable_paths = [line.strip() for line in file.readlines()]

        # compare relative variable paths within Generator to baseline
        assert(sorted(baseline_variable_paths)==sorted(gen.variable_path_list()))

        # compare relative history variable paths within Generator to baseline
        assert(sorted(baseline_history_variable_paths)==sorted(gen.history_path_list()))


if __name__ == "__main__":
    a = TestParamPath()
    a.test_variable_path_list()