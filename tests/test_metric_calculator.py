import unittest
import pandas as pd
from .mock_resources import *
import altrios as alt
from altrios import metric_calculator

class TestMetricCalculator(unittest.TestCase):
    def test_dummy_sim(self):        
        scenario_infos = []
        years = [2020, 2021]
        for year in years:
            scenario_infos.append(metric_calculator.ScenarioInfo(
                mock_speed_limit_train_simulation_vector(scenario_year=year), 
                year, 
                pd.DataFrame()))
            
        for info in scenario_infos:
            lcotkm = metric_calculator.calculate_lcotkm_singleyear(info,'usd_per_million_tonne_km')
            energy_cost = metric_calculator.calculate_energy_cost(info,'USD')
            ghg = metric_calculator.calculate_ghg(info,'tonne_co2eq')
            self.assertEqual(lcotkm.loc[lcotkm.Metric=='LCOTKM','Units'].to_numpy()[0], 'usd_per_million_tonne_km')
            self.assertEqual(ghg.loc[ghg.Metric=='GHG_Energy','Value'].to_numpy()[0], 0.0)
            self.assertEqual(lcotkm.loc[lcotkm.Metric=='Cost_Energy','Value'].to_numpy()[0], 
                energy_cost.loc[energy_cost.Metric=='Cost_Energy','Value'].to_numpy()[0])
            self.assertEqual(ghg.loc[ghg.Metric=='Diesel_Usage','Value'].to_numpy()[0], 
                energy_cost.loc[energy_cost.Metric=='Diesel_Usage','Value'].to_numpy()[0])

