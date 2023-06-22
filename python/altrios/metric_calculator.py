import pandas as pd
import math
import numpy as np
from typing import List, Optional, Dict, Tuple

import altrios as alt
from altrios import sim_manager
from altrios import utilities
discount_rate = 0.07
charger_efficiency = 0.85
diesel_price = 4.164  # dollar per gallon, 2021 average CA 2021 on-road diesel retail price incl taxes from EIA https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_sca_a.htm
# dollars per kWh, CA 2021 average from EIA, transportation sector retail customers
electricity_price = 0.1179
# gCO2eq/MJ, converted to gCO2eq/MWh, CA-GREET 3.0, 2020 avg emissions factors
ca_grid_avg_factor = 83.85 / utilities.MWH_PER_MJ
new_non_bel_cost = 2950000  # Zenith
new_bel_cost = new_non_bel_cost + 1105200  # NREL ATB https://atb.nrel.gov/electricity/2022/commercial_battery_storage
bel_battery_size_kWh = 2400
charger_speed_kW = 1000 # Charger speed in kW (assuming a constant charge rate, but adding some heuristic-y buffer time)
charger_servicing_time_hours = 0.5 #Time to plug and unplug EVSE, navigate locomotive to it, etc.
charger_cost = 1500000 # NREL Cost of Charging (Borlaug) showing ~linear trend on kW; #ICCT report showing little change through 2030
evse_lifespan = 15
locomotive_lifespan = 20
annual_fleet_turnover = 1/locomotive_lifespan
charge_queue_acceptable_length = 2
emissionsFactors = pd.read_csv(
    alt.resources_root() / "emissions_factors.csv")

class ScenarioInfo:
    def __init__(self, sims: alt.SpeedLimitTrainSimVec, scenario_year: int, consist_plan: pd.DataFrame):
        self.sims = sims
        self.year = scenario_year
        self.consist_plan = consist_plan


def main(
        scenario_infos: List[ScenarioInfo],
        metricsToCalc: Dict = {
            'Metric': ['LCOTKM', 'GHG', 'BEL_Pct', 'Count_EVSE_Ports'],
            'Units': ['usd_per_million_tonne_km', 'tonne_co2eq', 'pct', 'ports']
        }) -> pd.DataFrame:
    """
    Given a set of simulation results and the associated consist plans, computes economic and environmental metrics.
    Arguments:
    ----------
    sims: List (with one entry per scenario year) of SpeedLimitTrainSimVec (with one entry per simulated train run)
    consist_plans: List (with one entry per scenario year) of consist plans from the train planner
    metricsToCalc: Dict of output metrics to compute and units for each one
    Outputs:
    ----------
    values: DataFrame of output and intermediate metrics (metric name, units, value, and scenario year)
    """
    metrics = pd.DataFrame(metricsToCalc)
    all_year_metrics = []
    for scenario_info in scenario_infos:
        annual_metrics = metrics.apply(
            calculate_metric, 'columns', info=scenario_info)
        values = pd.concat(annual_metrics.tolist()).reset_index(drop=True)
        values['Year'] = scenario_info.year
        all_year_metrics.append(values)

    values = pd.concat(all_year_metrics).reset_index(drop=True)

    values = calculate_fleet_makeup_multiyear(values)
    multiyear = calculate_lcotkm_multiyear(values)
    values = pd.concat([values, pd.DataFrame(data=multiyear)],axis=0, join='outer',ignore_index=True)
    print(values)
    return values


def calculate_metric(
        thisRow: pd.DataFrame,
        info: ScenarioInfo) -> pd.DataFrame:
    """
    Given a years' worth of simulation results and the associated consist plan, computes the requested metric.
    Arguments:
    ----------
    thisRow: DataFrame containing the requested metric and requested units
    year: Year being modeled by a scenario
    sim: A scenario year's SpeedLimitTrainSimVec (with one entry per simulated locomotive run)
    consist_plan: The scenario year's associated consist plan from the train planner
    Outputs:
    ----------
    values: DataFrame of requested output metric + any intermediate metrics (metric name, units, value, and scenario year)
    """
    if thisRow.Metric == 'diesel_usage':
        values = calculate_diesel_use(info, thisRow.Units)
    elif thisRow.Metric == 'LCOTKM':
        values = calculate_lcotkm_singleyear(info, thisRow.Units)
    elif thisRow.Metric == 'GHG':
        values = calculate_ghg(info, thisRow.Units)
    elif thisRow.Metric == 'BEL_Pct':
        values = calculate_fleet_makeup_singleyear(info)
    elif thisRow.Metric == 'Count_EVSE_Ports':
        values = calculate_evse_needs_heuristic(info)
    else:
        print(f"Metric calculation not implemented: {thisRow.Metric}.")
        values = thisRow.copy()
        values.value = math.nan

    return values


def calculate_lcotkm_singleyear(
        info: ScenarioInfo,
        units: str) -> pd.DataFrame:
    """
    Given a years' worth of simulation results, computes a single year's levelized cost per gross tonne-km of freight delivered.
    Arguments:
    ----------
    thisRow: DataFrame containing the requested metric and requested units
    sim: A scenario year's SpeedLimitTrainSimVec (with one entry per simulated locomotive run)
    consist_plan: The scenario year's associated consist plan from the train planner
    Outputs:
    ----------
    values: DataFrame of LCOTKM + intermediate metrics (metric name, units, value, and scenario year)
    """
    energy_cost = calculate_energy_cost(info, units="USD")
    energy_cost_val = energy_cost.loc[energy_cost.Metric == 'Cost_Energy', 'Value'].to_numpy()[
        0]
    tkm = calculate_tkm(info, units="Mt-km")
    tkm_val = tkm.loc[tkm.Metric == 'Mt-km', 'Value'].to_numpy()[0]
    capital_cost_val = 0.0
    cost_total_val = energy_cost_val + capital_cost_val
    lcotkm_val = cost_total_val/tkm_val if tkm_val > 0 else math.nan
    cost_total = pd.DataFrame(
        [{'Metric': 'Cost_Total', 'Units': "USD", 'Value': cost_total_val}])
    lcotkm = pd.DataFrame(
        [{'Metric': 'LCOTKM', 'Units': units, 'Value': lcotkm_val}])
    return pd.concat([energy_cost, cost_total, tkm, lcotkm]).reset_index(drop=True)


def calculate_lcotkm_multiyear(values: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of each year's costs and gross freight deliveries, computes the multi-year levelized cost per gross tonne-km of freight delivered.
    Arguments:
    ----------
    values: DataFrame containing total costs and gross freight deliveries for each modeled scenario year
    Outputs:
    ----------
    DataFrame of LCOTKM result (metric name, units, value, and scenario year)
    """
    cost_timeseries = values[values.Metric == 'Cost_Total'].copy()
    tkm_timeseries = values[values.Metric == 'Mt-km'].copy()
    cost_timeseries.loc[:,'Year_Offset'] = cost_timeseries.loc[:,'Year'] - \
        min(cost_timeseries.loc[:,'Year']) + 1
    tkm_timeseries.loc[:,'Year_Offset'] = tkm_timeseries.loc[:,'Year'] - \
        min(tkm_timeseries.loc[:,'Year']) + 1
    cost_timeseries.loc[:,'Value_Discounted'] = cost_timeseries.loc[:,'Value'] / \
        ((1+discount_rate)**cost_timeseries.loc[:,'Year_Offset'])
    tkm_timeseries.loc[:,'Value_Discounted'] = tkm_timeseries.loc[:,'Value'] / \
        ((1+discount_rate)**tkm_timeseries.loc[:,'Year_Offset'] )
    cost_total = sum(cost_timeseries.Value_Discounted)
    tkm_total = sum(tkm_timeseries.Value_Discounted)
    lcotkm_total = cost_total/tkm_total
    return pd.DataFrame([{'Metric': 'LCOTKM', 'Units': 'usd_per_million_tonne_km', 'Value': lcotkm_total, 'Year': 'All'}])


def calculate_energy_cost(info: ScenarioInfo,
        units: str) -> pd.DataFrame:
    """
    Given a years' worth of simulation results, computes a single year's energy costs.
    Arguments:
    ----------
    sim: A scenario year's SpeedLimitTrainSimVec (with one entry per simulated locomotive run)
    units: Requested units
    Outputs:
    ----------
    DataFrame of energy costs + intermediate metrics (metric name, units, value, and scenario year)
    """
    diesel_used = calculate_diesel_use(info, units="gallons")
    electricity_used = calculate_electricity_use(info, units="kWh")
    # fuel_price.loc[fuel_price.Metric=='fuel_price','Value'].to_numpy()[0]
    diesel_cost_value = diesel_used.loc[diesel_used.Metric == 'Diesel_Usage', 'Value'].to_numpy()[
        0] * diesel_price
    diesel_cost = pd.DataFrame(
        [{'Metric': 'Cost_Diesel', 'Units': "USD", 'Value': diesel_cost_value}])
    electricity_cost_value = electricity_used.loc[electricity_used.Metric == 'Electricity_Usage', 'Value'].to_numpy()[
        0] * electricity_price
    electricity_cost = pd.DataFrame(
        [{'Metric': 'Cost_Electricity', 'Units': "USD", 'Value': electricity_cost_value}])
    energy_cost = pd.DataFrame(
        [{'Metric': 'Cost_Energy', 'Units': units, 'Value': diesel_cost_value + electricity_cost_value}])
    return pd.concat([diesel_used, diesel_cost, electricity_used, electricity_cost, energy_cost]).reset_index(drop=True)


def calculate_diesel_use(
        info: ScenarioInfo,
        units: str):
    """
    Given a years' worth of simulation results, computes a single year's diesel fuel use.
    Arguments:
    ----------
    sim: A scenario year's SpeedLimitTrainSimVec (with one entry per simulated locomotive run)
    lhv_kj_per_kg: Low Heating Value (KJ/kg) of the modeled diesel fuel
    Outputs:
    ----------
    DataFrame of diesel use (metric name, units, value, and scenario year)
    """

    # TODO: handle arbitrary units for LHV and density
    lhv_kj_per_kg = emissionsFactors[emissionsFactors['Name'] ==
                                     'ultra low sulfur diesel']['LowHeatingValue'].to_numpy()[0]*1e3
    # Rho read in as g/gal
    rho_fuel_g_per_gallon = (
        emissionsFactors[emissionsFactors['Name'] == 'ultra low sulfur diesel']['Density'].to_numpy()[0])
    if units == 'gallons':
        val = (info.sims.get_energy_fuel_joules(annualize=True) /
               1e3 / lhv_kj_per_kg) * 1e3 / rho_fuel_g_per_gallon
    elif units == 'MJ':
        val = info.sims.get_energy_fuel_joules(annualize=True) / 1e6
    else:
        print(f"Units of {units} not supported for fuel usage calculation.")
        val = math.nan

    return pd.DataFrame([{'Metric': 'Diesel_Usage', 'Units': units, 'Value': val}])


def calculate_electricity_use(
        info: ScenarioInfo,
        units: str) -> pd.DataFrame:
    """
    Given a years' worth of simulation results, computes a single year's grid electricity use.
    Arguments:
    ----------
    sim: A scenario year's SpeedLimitTrainSimVec (with one entry per simulated locomotive run)
    units: Requested units
    Outputs:
    ----------
    DataFrame of grid electricity use (metric name, units, value, and scenario year)
    """
    if units == 'kWh':
        val = (info.sims.get_net_energy_res_joules(annualize=True) / 1e6) * \
            utilities.KWH_PER_MJ / charger_efficiency
    elif units == 'MJ':
        val = info.sims.get_net_energy_res_joules(annualize=True) / 1e6
        val = val / charger_efficiency
    else:
        print(f"Units of {units} not supported for fuel usage calculation.")
        val = math.nan

    return pd.DataFrame([{'Metric': 'Electricity_Usage', 'Units': units, 'Value': val}])


def calculate_tkm(
        info: ScenarioInfo,
        units: str) -> pd.DataFrame:
    """
    Given a years' worth of simulation results, computes a single year's gross tonne-km of freight delivered
    Arguments:
    ----------
    sim: A scenario year's SpeedLimitTrainSimVec (with one entry per simulated locomotive run)
    units: Requested units
    Outputs:
    ----------
    DataFrame of gross tonne-km of freight (metric name, units, value, and scenario year)
    """
    return pd.DataFrame([{'Metric': 'Mt-km', 'Units': units, 'Value': info.sims.get_megagram_kilometers(annualize=True)/1.0e6}])


def calculate_ghg(
        info: ScenarioInfo,
        units: str) -> pd.DataFrame:
    """
    Given a years' worth of simulation results, computes a single year's GHG emissions from energy use
    Arguments:
    ----------
    sim: A scenario year's SpeedLimitTrainSimVec (with one entry per simulated locomotive run)
    units: Requested units
    Outputs:
    ----------
    DataFrame of GHG emissions from energy use (metric name, units, value, and scenario year)
    """
    if units != 'tonne_co2eq':
        return pd.DataFrame([{'Metric': 'GHGs', 'Units': units, 'Value': math.nan}])

    sim = info.sims
    year = info.year
    diesel_ghg_factor = emissionsFactors.loc[
        (emissionsFactors.Name == 'ultra low sulfur diesel') &
        (emissionsFactors.Year == 2020) &
        (emissionsFactors.Units == 'gCO2 eq/MJ'), "Value"].to_numpy()[0]

    diesel_MJ = calculate_diesel_use(info, "MJ")
    diesel_ghg_val = diesel_MJ.loc[diesel_MJ.Metric == 'Diesel_Usage', 'Value'].to_numpy()[
        0]*diesel_ghg_factor/utilities.G_PER_TONNE

    electricity_MJ = calculate_electricity_use(info, "MJ")
    electricity_kWh = calculate_electricity_use(info, "kWh")
    electricity_ghg_val = (electricity_kWh.loc[electricity_kWh.Metric == 'Electricity_Usage', 'Value'].to_numpy()[
                           0] / 1.0e3) * ca_grid_avg_factor/utilities.G_PER_TONNE

    diesel_ghg = pd.DataFrame(
        [{'Metric': 'GHG_Diesel', 'Units': units, 'Value': diesel_ghg_val}])
    electricity_ghg = pd.DataFrame(
        [{'Metric': 'GHG_Electricity', 'Units': units, 'Value': electricity_ghg_val}])
    total_energy_ghg = pd.DataFrame(
        [{'Metric': 'GHG_Energy', 'Units': units, 'Value': diesel_ghg_val+electricity_ghg_val}])

    # TODO: other fuel types. also, are embodied emissions in scope?
    return pd.concat([diesel_MJ, electricity_MJ, diesel_ghg, electricity_ghg, total_energy_ghg]).reset_index(drop=True)


def calculate_fleet_makeup_singleyear(
        info: ScenarioInfo
        ) -> pd.DataFrame:
    """
    Given a single scenario year's locomotive consist plan, computes the year's locomotive fleet composition
    Arguments:
    ----------
    sim: A scenario year's locomotive consist plan from the train planner
    Outputs:
    ----------
    DataFrame of locomotive fleet composition metrics (metric name, units, value, and scenario year)
    """
    unique_locomotives = info.consist_plan.groupby(
        ['Locomotive Type', 'Locomotive ID']).size().reset_index()
    bel_pct_avg_consist_val = (
        info.consist_plan['Locomotive Type'] == 'BEL').sum()/len(info.consist_plan)
    bel_pct_of_locomotives_val = (
        unique_locomotives['Locomotive Type'] == 'BEL').sum()/len(unique_locomotives)
    num_bel_val = (unique_locomotives['Locomotive Type'] == 'BEL').sum()
    num_non_bel_val = (unique_locomotives['Locomotive Type'] != 'BEL').sum()

    bel_pct_consist = pd.DataFrame(
        [{'Metric': 'BEL_Pct_Consist_Members', 'Units': 'Percent', 'Value': bel_pct_avg_consist_val}])
    bel_pct_of_locomotives = pd.DataFrame(
        [{'Metric': 'BEL_Pct_Locomotives', 'Units': 'Percent', 'Value': bel_pct_of_locomotives_val}])
    num_bel = pd.DataFrame(
        [{'Metric': 'Count_BEL', 'Units': 'Count', 'Value': num_bel_val}])
    num_non_bel = pd.DataFrame(
        [{'Metric': 'Count_Non_BEL', 'Units': 'Count', 'Value': num_non_bel_val}])
    num_total = pd.DataFrame(
        [{'Metric': 'Count_Total', 'Units': 'Count', 'Value': num_bel_val + num_non_bel_val}])
    return pd.concat([bel_pct_consist, bel_pct_of_locomotives, num_bel, num_non_bel, num_total]).reset_index(drop=True)


def calculate_fleet_makeup_multiyear(values: pd.DataFrame) -> pd.DataFrame:
    """
    Given multiple scenario years' locomotive fleet compositions, computes additional across-year metrics
    Arguments:
    ----------
    values: DataFrame with multiple scenario years' locomotive fleet composition metrics
    Outputs:
    ----------
    DataFrame of across-year locomotive fleet composition metrics (metric name, units, value, and scenario year)
    """
    values.sort_values(by=['Units', 'Metric', 'Year'])
    bels = values[values['Metric'] == 'Count_BEL'].copy()
    bels.loc[:,'Change_BELs'] = bels['Value'] - bels['Value'].shift(1)
    bels = bels[["Year", "Change_BELs"]]
    non_bels = values[values['Metric'] == 'Count_Non_BEL'].copy()
    non_bels.loc[:,'Change_Non_BELs'] = non_bels['Value'] - \
        non_bels['Value'].shift(1)
    non_bels = non_bels[["Year", "Change_Non_BELs"]]

    evse = values[values['Metric'] == 'Count_EVSE_Ports'].copy()
    evse.loc[:,'Count_EVSE_Ports_New'] = evse['Value'] - evse['Value'].shift(1)
    evse = evse[["Year", "Count_EVSE_Ports_New"]]
    
    last_year_size = values[values['Metric'] == 'Count_Total'].copy()
    last_year_size.loc[:,'Last_Year_Count_Total'] = last_year_size['Value'].shift(1)
    last_year_size.loc[:,'Count_Change'] = last_year_size['Value'] - \
        last_year_size['Last_Year_Count_Total']
    last_year_size = last_year_size[[
        "Year", "Count_Change", "Last_Year_Count_Total"]]
    all = pd.merge(bels, non_bels, on="Year")
    all = pd.merge(all, evse, on="Year")
    all = pd.merge(all, last_year_size, on="Year")
    all.loc[:,'Count_New_Needed'] = np.floor(
        all['Last_Year_Count_Total']*annual_fleet_turnover) + all['Count_Change']
    # TODO this logic isn't quite right. Works for base case but there are lots of edge cases.
    # TODO handle mid-simulation retirements (not just end-of-sim retirements)
    all.loc[:,'Count_BEL_New'] = np.maximum(0, all['Change_BELs'])
    all.loc[:,'Count_Non_BEL_New'] = np.maximum(
        0, all['Count_New_Needed'] - all['Count_BEL_New'])
    all.loc[:,'Cost_BEL_New'] = all['Count_BEL_New'] * new_bel_cost
    all.loc[:,'Cost_Non_BEL_New'] = all['Count_Non_BEL_New'] * new_non_bel_cost
    all.loc[:,'Cost_EVSE_Ports_New'] = all['Count_EVSE_Ports_New'] * charger_cost

    num_non_bel_new = all[['Year']].copy()
    num_non_bel_new.loc[:,'Metric'] = 'Count_Non_BEL_New'
    num_non_bel_new.loc[:,'Units'] = 'Count'
    num_non_bel_new.loc[:,'Value'] = all['Count_Non_BEL_New'].fillna(0)
    num_non_bel_new = num_non_bel_new[['Metric', 'Units', 'Value', 'Year']]

    num_bel_new = all[['Year']].copy()
    num_bel_new.loc[:,'Metric'] = 'Count_BEL_New'
    num_bel_new.loc[:,'Units'] = 'Count'
    num_bel_new.loc[:,'Value'] = all['Count_BEL_New'].fillna(0)
    num_bel_new = num_bel_new[['Metric', 'Units', 'Value', 'Year']]
    num_evse_new = all[['Year']].copy()
    num_evse_new.loc[:,'Metric'] = 'Cost_EVSE_Ports_New'
    num_evse_new.loc[:,'Units'] = 'Count'
    num_evse_new.loc[:,'Value'] = all['Cost_EVSE_Ports_New'].fillna(0)
    num_evse_new = num_evse_new[['Metric', 'Units', 'Value', 'Year']]


    cost_non_bel_new = all[['Year']].copy()
    cost_non_bel_new.loc[:,'Metric'] = 'Cost_Non_Bel_New'
    cost_non_bel_new.loc[:,'Units'] = 'USD'
    cost_non_bel_new.loc[:,'Value'] = all['Cost_Non_BEL_New'].fillna(0.0)
    cost_non_bel_new = cost_non_bel_new[['Metric', 'Units', 'Value', 'Year']]
    cost_bel_new = all[['Year']].copy()
    cost_bel_new.loc[:,'Metric'] = 'Cost_BEL_New'
    cost_bel_new.loc[:,'Units'] = 'USD'
    cost_bel_new.loc[:,'Value'] = all['Cost_BEL_New'].fillna(0.0)
    cost_bel_new = cost_bel_new[['Metric', 'Units', 'Value', 'Year']]
    cost_evse_new = all[['Year']].copy()
    cost_evse_new.loc[:,'Metric'] = 'Cost_EVSE_Ports_New'
    cost_evse_new.loc[:,'Units'] = 'USD'
    cost_evse_new.loc[:,'Value'] = all['Cost_EVSE_Ports_New'].fillna(0.0)
    cost_evse_new = cost_evse_new[['Metric', 'Units', 'Value', 'Year']]

    retirement_year = np.max(cost_bel_new['Year'])+1
    bel_retirement_value = 0.0
    for idx, row in cost_bel_new.iterrows():
        pct_health = 1-(retirement_year - row['Year'])/locomotive_lifespan
        bel_retirement_value += row['Value']*pct_health

    non_bel_retirement_value = 0.0
    for idx, row in cost_non_bel_new.iterrows():
        pct_health = 1-(retirement_year - row['Year'])/locomotive_lifespan
        non_bel_retirement_value += row['Value']*pct_health

    evse_retirement_value = 0.0
    for idx, row in cost_evse_new.iterrows():
        pct_health = 1-(retirement_year - row['Year'])/evse_lifespan
        evse_retirement_value += row['Value']*pct_health

    total_retirement_value = bel_retirement_value + non_bel_retirement_value + evse_retirement_value

    total_retirement = pd.DataFrame(
        [{'Metric': 'Cost_Total', 'Units': 'USD', 'Value': -total_retirement_value, 'Year': retirement_year}])

    for year in cost_non_bel_new['Year']:
        idx = values[(values['Metric'] == 'Cost_Total') & (values['Year'] == year)].index
        values.loc[idx,'Value'] = values.loc[idx,'Value'] + \
            np.sum(cost_non_bel_new[cost_non_bel_new['Year'] == year]['Value']) + \
            np.sum(cost_bel_new[cost_bel_new['Year'] == year]['Value']) + \
            np.sum(cost_evse_new[cost_evse_new['Year'] == year]['Value'])

    return pd.concat([values, total_retirement, num_bel_new, num_non_bel_new, cost_bel_new, cost_non_bel_new, cost_evse_new]).reset_index(drop=True)

def calculate_evse_needs_heuristic(
        info: ScenarioInfo) -> pd.DataFrame:
    """
    Given a consist plan, computes a conservative estimate of necessary port counts using some simplifying assumptions.
    Arguments:
    ----------
    consist_plan: The scenario year's associated consist plan from the train planner
    Outputs:
    ----------
    DataFrame of single-year port count required to meet charging demand witholut altering operations (metric name, units, value, and scenario year)
    """
    charge_time_needed = charge_queue_acceptable_length*(bel_battery_size_kWh/charger_speed_kW + charger_servicing_time_hours)
    bel_consist = info.consist_plan[info.consist_plan['Locomotive Type']=='BEL']
    #Assumes only two ODs for now, so we can assume that for a given destination, trains leaving at a specific time will each arrive at the same time.
    bel_consist = bel_consist.rename(columns={"Origin Departure Time(hr)": "Arrival Time"})
    bel_arrivals = bel_consist.groupby(['Destination ID','Arrival Time']).agg({'Locomotive ID': ['count']}).reset_index() 
    bel_arrivals.columns = ['Destination ID','Arrival Time','Locomotive_Count']
    bel_arrivals.loc[:,'Charge_Finish_Time'] = bel_arrivals['Arrival Time'] + charge_time_needed
    bel_arrivals.loc[:,'Total_Ports_Needed'] = bel_arrivals['Locomotive_Count']
    for idx, row in bel_arrivals.iterrows():
        bel_arrivals.loc[idx,'Total_Ports_Needed'] = math.ceil(sum(bel_arrivals[
            (bel_arrivals['Destination ID'] == row['Destination ID']) & \
            (bel_arrivals['Arrival Time'] <= row['Arrival Time']) & \
            (bel_arrivals['Charge_Finish_Time'] >= row['Arrival Time'])]['Locomotive_Count'])/charge_queue_acceptable_length)

    port_count_per_destination = bel_arrivals.groupby(['Destination ID']).agg({'Total_Ports_Needed': ['max']}).reset_index() 
    port_count_per_destination.columns = ['Destination ID','Total_Ports_Needed']
    port_count_total = sum(port_count_per_destination['Total_Ports_Needed'])


    return pd.DataFrame([{'Metric': 'Count_EVSE_Ports', 'Units': 'Ports', 'Value': port_count_total}])
