import unittest


from .mock_resources import mock_set_speed_train_simulation


class TestSetSpeedTrainSim(unittest.TestCase):
    def test_walk_no_save(self):
        mock_sim = mock_set_speed_train_simulation(save_interval=None)  

        mock_sim.walk()

    def test_walk_w_save(self):
        mock_sim = mock_set_speed_train_simulation(save_interval=1)  

        mock_sim.walk()

    # TODO: Convert this test to yaml? The train simulation uses infinity, which breaks json
    # def test_to_from_json(self):
    #     mock_sim = mock_train_simulation()

    #     j = mock_sim.to_json()
    #     mock_sim2 = alt.SetSpeedTrainSim.from_json(j)

    #     self.assertEqual(
    #         mock_sim.speed_trace.time_seconds.tolist(),
    #         mock_sim2.speed_trace.time_seconds.tolist(),
    #     )
