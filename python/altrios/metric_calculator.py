import pandas as pd
import polars as pl
import polars.selectors as cs
import math
import numpy as np
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import altrios as alt
from altrios import utilities, defaults

MetricType = pl.DataFrame

emissions_factors_greet = pl.read_csv(defaults.FUEL_EMISSIONS_FILE,
                                      null_values="NA")
electricity_prices_eia = (pl.read_csv(source = defaults.ELECTRICITY_PRICE_FILE, skip_rows = 5)
    .select(["Year", "Industrial"])
    .rename({"Industrial": "Price"}))
liquid_fuel_prices_eia = (pl.read_csv(source = defaults.LIQUID_FUEL_PRICE_FILE, skip_rows = 5)
    .select(["Year", "   Diesel Fuel (distillate fuel oil) 6/"])
    .rename({"   Diesel Fuel (distillate fuel oil) 6/": "Price"}))
battery_prices_nrel_atb = (pl.read_csv(source = defaults.BATTERY_PRICE_FILE)).filter(pl.col("Scenario")=="Moderate")

metric_columns = ["Subset","Value","Metric","Units","Year"]

class ScenarioInfo:
    def __init__(self, 
                 sims: alt.SpeedLimitTrainSimVec, 
                 simulation_days: int,
                 scenario_year: int, 
                 loco_pool: pl.DataFrame,
                 consist_plan: pl.DataFrame,
                 refuel_facilities: pl.DataFrame,
                 refuel_sessions: pl.DataFrame,
                 emissions_factors: pl.DataFrame,
                 nodal_energy_prices: pl.DataFrame,
                 count_unused_locomotives: bool):
        self.sims = sims
        self.simulation_days = simulation_days
        self.year = scenario_year
        self.loco_pool = loco_pool
        self.consist_plan = consist_plan
        self.refuel_facilities = refuel_facilities
        self.refuel_sessions = refuel_sessions
        self.emissions_factors = emissions_factors
        self.nodal_energy_prices = nodal_energy_prices
        self.count_unused_locomotives = count_unused_locomotives

def metric(
        name: str,
        units: str,
        value: float,
        subset: str = "All",
        year: str = "All") -> MetricType:
    return MetricType({
        "Metric" : [name],
        "Units" : [units],
        "Value" : [float(value)],
        "Subset": [subset],
        "Year": [year]
        })

def metrics_from_list(metrics: list[MetricType]) -> MetricType:
    return pl.concat(metrics, how="diagonal")

def value_from_metrics(metrics: MetricType, 
                      name: str = None,
                      units: str = None,
                      subset: str = None) -> float:
    
    out = metrics
    if name is not None:
        out = out.filter(pl.col("Metric") == pl.lit(name))

    if units is not None:
        out = out.filter(pl.col("Units") == pl.lit(units))

    if subset is not None: 
        out = out.filter(pl.col("Subset") == subset)
        
    if out.height == 1:
        return out.get_column("Value")[0]
    else:
        return math.nan

def main(
        scenario_infos: List[ScenarioInfo],
        annual_metrics: dict = {
            'Metric': ['Mt-km', 'GHG', 'Count_Locomotives', 'Count_Refuelers', 'Energy_Costs'],
            'Units': ['million tonne-km', 'tonne CO2-eq', 'assets', 'assets', 'USD']}
) -> pl.DataFrame:
    """
    Given a set of simulation results and the associated consist plans, computes economic and environmental metrics.
    Arguments:
    ----------
    scenario_infos: List (with one entry per scenario year) of Scenario Info objects
    metricsToCalc: Dictionary of metrics to calculate
    Outputs:
    ----------
    values: DataFrame of output and intermediate metrics (metric name, units, value, and scenario year)
    """
    annual_metrics = pl.DataFrame(annual_metrics)
    values = pl.DataFrame()
    for row in annual_metrics.iter_rows(named = True):
        for info in scenario_infos:
            annual_value = (calculate_annual_metric(row['Metric'], row['Units'], info)
                .with_columns(pl.lit(info.year).alias("Year")))
            values = pl.concat([values, annual_value], how="diagonal")

    values = values.unique()
    values = calculate_rollout_investments(values)
    values = calculate_rollout_total_costs(values)
    values = calculate_rollout_lcotkm(values)
    values = values.sort(["Year","Subset"], descending = [False, True])
    return values


def calculate_annual_metric(
        metric: str,
        units: str,
        info: ScenarioInfo) -> MetricType:
    """
    Given a years' worth of simulation results and the associated consist plan, computes the requested metric.
    Arguments:
    ----------
    thisRow: DataFrame containing the requested metric and requested units
    info: A scenario information object representing parameters and results for a single year
    Outputs:
    ----------
    values: DataFrame of requested output metric + any intermediate metrics (metric name, units, value, and scenario year)
    """
    def return_metric_error(info, units) -> MetricType:
        print(f"Metric calculation not implemented: {metric}.")
        return value_from_metrics(metric(metric, units, math.nan))
    
    function_for_metric = function_mappings.get(metric, return_metric_error)
    return function_for_metric(info, units)

def calculate_rollout_lcotkm(values: MetricType) -> MetricType:
    """
    Given a DataFrame of each year's costs and gross freight deliveries, computes the multi-year levelized cost per gross tonne-km of freight delivered.
    Arguments:
    ----------
    values: DataFrame containing total costs and gross freight deliveries for each modeled scenario year
    Outputs:
    ----------
    DataFrame of LCOTKM result (metric name, units, value, and scenario year)
    """

    cost_timeseries = (values
                       .filter((pl.col("Metric")==pl.lit("Cost_Total")) & (pl.col("Subset")=="All"))
                       .select(["Year","Value"])
                       .rename({"Value": "Cost_Total"}))
    tkm_timeseries = (values
                      .filter(pl.col("Metric")==pl.lit("Mt-km"))
                      .select(["Year", "Value"])
                      .rename({"Value": "Mt-km"}))
    timeseries = cost_timeseries.join(tkm_timeseries, on="Year", how="outer")
    timeseries = (timeseries
                  .with_columns((pl.col("Year") - pl.col("Year").min()).alias("Year_Offset"))
                  .with_columns(((1+defaults.DISCOUNT_RATE)**pl.col("Year_Offset")).alias("Discounting_Factor"))
                  .with_columns(
                    (pl.col("Cost_Total") / pl.col("Discounting_Factor")).alias("Cost_Total_Discounted"),
                    (pl.col("Mt-km") / pl.col("Discounting_Factor")).alias("Mt-km_Discounted"))
                  .with_columns(pl.col("Year").cast(pl.Utf8)))           
    cost_total = timeseries.get_column("Cost_Total_Discounted").sum()
    starting_residual_value_to_subtract = (values
        .filter(pl.col("Metric")==pl.lit("Asset_Value_Initial"))
        .get_column("Value")[0]
    )
    cost_minus_residual_baseline = cost_total + starting_residual_value_to_subtract / timeseries.get_column("Discounting_Factor").max()
    tkm_total = timeseries.get_column("Mt-km_Discounted").sum()
    lcotkm_all = cost_minus_residual_baseline/tkm_total if tkm_total > 0 else math.nan

    cost_discounted = timeseries.select(
            pl.col("Year"),
            Value = pl.col("Cost_Total_Discounted"),
            Metric = pl.lit("Cost_Total_Discounted"),
            Subset = pl.lit("All"),
            Units = pl.lit("USD (discounted)")
    )
    tkm_discounted = timeseries.select(
            pl.col("Year"),
            Value = pl.col("Mt-km_Discounted"),
            Metric = pl.lit("Mt-km_Discounted"),
            Subset = pl.lit("All"),
            Units = pl.lit("million tonne-km (discounted)")
    )
    cotkm_annual = timeseries.select(
            pl.col("Year"),
            Value = pl.col("Cost_Total")/pl.col("Mt-km"),
            Metric = pl.lit("Cost_Per_Mt-km"),
            Subset = pl.lit("All"),
            Units = pl.lit("USD per million tonne-km")
    )
    discount = timeseries.select(
            pl.col("Year"),
            Value = pl.col("Discounting_Factor"),
            Metric = pl.lit("Discounting_Factor"),
            Subset = pl.lit("All"),
            Units = pl.lit("fraction (0-1)")
    )
    return metrics_from_list([
        values.with_columns(pl.col("Year").cast(pl.Utf8)),
        cost_discounted,
        tkm_discounted,
        cotkm_annual,
        discount,
        metric("LCOTKM", "USD per million tonne-km (levelized)", lcotkm_all)
    ])

def calculate_energy_cost(info: ScenarioInfo,
        units: str) -> MetricType:
    """
    Given a years' worth of simulation results, computes a single year energy costs.
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    units: Requested units
    Outputs:
    ----------
    DataFrame of energy costs + intermediate metrics (metric name, units, value, and scenario year)
    """
    diesel_used = calculate_diesel_use(info, units="gallons")
    electricity_used = (calculate_electricity_use(info, units="kWh")
                        )
    electricity_costs_disagg = (electricity_used
        .filter(pl.col("Subset") != "All")
        .join(info.nodal_energy_prices.filter(pl.col("Fuel")=="Electricity"), left_on="Subset", right_on="Node", how="left")
        .with_columns((pl.col("Value") * pl.col("Price") / 100).alias("Value"),
                      pl.lit("Cost_Electricity").alias("Metric"),
                      pl.lit("USD").alias("Units"))
        .select(metric_columns))
    electricity_costs_agg = (electricity_costs_disagg
        .groupby(["Metric","Units","Year"])
        .agg(pl.col("Value").sum())
        .with_columns(pl.lit("All").alias("Subset")))
    electricity_cost_value = 0
    if electricity_costs_agg.height > 0:
        electricity_cost_value = electricity_costs_agg.get_column("Value")[0]
    # Diesel refueling is not yet tracked spatiotemporally; just use average price across the network.
    diesel_price = info.nodal_energy_prices.filter(pl.col("Fuel")=="Diesel").get_column("Price").mean()
    diesel_cost_value = value_from_metrics(diesel_used,"Diesel_Usage") * diesel_price
    return metrics_from_list([diesel_used, 
                      metric("Cost_Diesel", "USD", diesel_cost_value), 
                      electricity_costs_disagg, 
                      electricity_costs_agg, 
                      metric("Cost_Energy", units, diesel_cost_value + electricity_cost_value)])

def calculate_diesel_use(
        info: ScenarioInfo,
        units: str):
    """
    Given a years' worth of simulation results, computes a single year diesel fuel use.
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    units: Requested units.
    Outputs:
    ----------
    DataFrame of diesel use (metric name, units, value, and scenario year)
    """
    diesel_emissions_factors = emissions_factors_greet.filter(pl.col("Name") == pl.lit("ultra low sulfur diesel"))
    lhv_kj_per_kg = diesel_emissions_factors.get_column("LowHeatingValue")[0] * 1e3
    rho_fuel_g_per_gallon = diesel_emissions_factors.get_column("Density")[0]
    if units == 'gallons':
        val = (info.sims.get_energy_fuel_joules(annualize=True) /
               1e3 / lhv_kj_per_kg) * 1e3 / rho_fuel_g_per_gallon
    elif units == 'MJ':
        val = info.sims.get_energy_fuel_joules(annualize=True) / 1e6
    else:
        print(f"Units of {units} not supported for fuel usage calculation.")
        val = math.nan

    return metric("Diesel_Usage", units, val)
  
def calculate_electricity_use(
        info: ScenarioInfo,
        units: str) -> MetricType:
   """
    Given a years' worth of simulation results, computes a single year grid electricity use.
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    units: Requested units
    Outputs:
    ----------
    DataFrame of grid electricity use (metric name, units, value, and scenario year)
    """    
   
   disagg_energy = (info.refuel_sessions
                    .filter(pl.col("Locomotive_Type")==pl.lit("BEL"))
                    .groupby(["Node"])
                    .agg((pl.col('Refuel_Energy_J') * pl.lit(info.simulation_days)).sum())
                    )
   
   if units == "MJ":
       disagg_energy = disagg_energy.with_columns(
           (pl.col("Refuel_Energy_J") / 1e6).alias("MJ"))
   elif units == "kWh":
       disagg_energy = disagg_energy.with_columns(
           ( pl.col("Refuel_Energy_J") * utilities.KWH_PER_MJ / 1e6).alias("kWh"))       
   elif units == "MWh":
       disagg_energy = disagg_energy.with_columns(
           (pl.col("Refuel_Energy_J") * utilities.KWH_PER_MJ / 1e9).alias("MWh"))  
   else:
        print(f"Units of {units} not supported for electricity use calculation.")
        return metric("Electricity_Usage", units, math.nan)
   
   disagg_energy = (disagg_energy
        .drop("Refuel_Energy_J")
        .with_columns(pl.lit("Electricity_Usage").alias("Metric"))
        .melt(
            id_vars=["Metric","Node"],
            value_vars=units,
            variable_name="Units",
            value_name="Value")
        .rename({"Node": "Subset"})
        .with_columns(pl.lit(str(info.year)).alias("Year"))
    )
   agg_energy = (disagg_energy
                 .groupby(["Metric","Units","Year"])
                 .agg(pl.col("Value").sum())
                 .with_columns(
                    pl.lit("All").alias("Subset"))
                )
   return metrics_from_list([
       agg_energy,
       disagg_energy])


def calculate_freight(
        info: ScenarioInfo,
        units: str) -> MetricType:
    """
    Given a years' worth of simulation results, computes a single year gross million tonne-km of freight delivered
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    units: Requested units
    Outputs:
    ----------
    DataFrame of gross million tonne-km of freight (metric name, units, value, and scenario year)
    """
    return metric("Mt-km", units, info.sims.get_megagram_kilometers(annualize=True)/1.0e6)

def calculate_ghg(
        info: ScenarioInfo,
        units: str) -> MetricType:
    """
    Given a years' worth of simulation results, computes a single year GHG emissions from energy use
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    units: Requested units
    Outputs:
    ----------
    DataFrame of GHG emissions from energy use (metric name, units, value, and scenario year)
    """
    if units != 'tonne CO2-eq' :
        return metric("GHG_Energy", units, math.nan)

    if info.emissions_factors.height == 0:
        return metric("GHG_Energy", units, 0.0)
    
    #GREET table only has a value for 2020; apply that to all years
    diesel_ghg_factor = (emissions_factors_greet
                         .filter((pl.col("Name") == "ultra low sulfur diesel") &
                                 (pl.col("Units") == "gCO2 eq/MJ") &
                                 (pl.col("Year") == 2020))
                        .get_column("Value")
    )[0]
                            
    diesel_MJ = calculate_diesel_use(info, "MJ")
    electricity_MJ = calculate_electricity_use(info,"MJ")
    electricity_MWh = calculate_electricity_use(info,"MWh")

    diesel_ghg_val = value_from_metrics(diesel_MJ,"Diesel_Usage") * \
        diesel_ghg_factor / utilities.G_PER_TONNE
    electricity_ghg_val = (electricity_MWh
        .filter(pl.col("Subset") != pl.lit("All"))
        .join(info.emissions_factors, 
              left_on="Subset",
              right_on="Node",
              how="inner")
        .select(pl.col("CO2eq_kg_per_MWh") * pl.col("Value") / 1000.0)
        .sum().to_series()[0]
    )

    return metrics_from_list([
        diesel_MJ, 
        electricity_MJ, 
        electricity_MWh,
        metric("GHG_Diesel", units, diesel_ghg_val), 
        metric("GHG_Electricity", units, electricity_ghg_val), 
        metric("GHG_Energy", units, diesel_ghg_val+electricity_ghg_val)])


def calculate_locomotive_counts(
        info: ScenarioInfo,
        _
        ) -> MetricType:
    """
    Given a single scenario year's locomotive consist plan, computes the year's locomotive fleet composition
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    Outputs:
    ----------
    DataFrame of locomotive fleet composition metrics (metric name, units, value, and scenario year)
    """
    def get_locomotive_counts(df: pl.DataFrame) -> pl.DataFrame:
        out = (df
                .groupby(["Locomotive_Type"])
                .agg((pl.col("Locomotive_ID").n_unique()).alias("Count_Locomotives"),
                     pl.col("Cost_USD").mean().alias("Unit_Cost"),
                     pl.col("Lifespan_Years").mean().alias("Lifespan"))
                .with_columns((pl.col("Count_Locomotives") / pl.col("Count_Locomotives").sum()).alias("Pct_Locomotives"))
                .melt(
                    id_vars="Locomotive_Type",
                    variable_name="Metric",
                    value_name="Value")
                .with_columns(
                    pl.when(pl.col("Metric") == "Unit_Cost")
                        .then("USD")
                        .when(pl.col("Metric") == "Lifespan")
                        .then("years")
                        .when(pl.col("Metric")=="Pct_Locomotives")
                        .then("fraction (0-1)")
                        .otherwise("assets")
                    .alias("Units"),
                    pl.lit("All").alias("Year"))
                .rename({"Locomotive_Type": "Subset"})
        )
        out_agg = (out
                    .filter(pl.col("Metric") == ("Count_Locomotives"))
                    .groupby(["Metric","Units","Year"])
                    .agg(pl.col("Value").sum())
                    .with_columns(pl.lit("All").alias("Subset"))
        )        
        return metrics_from_list([out, out_agg])

    locos_to_use = None
    if info.count_unused_locomotives:
        locos_to_use = info.loco_pool
    else:
        locos_to_use = (info.consist_plan
                        .join(
                            info.loco_pool.select(["Locomotive_ID","Cost_USD","Lifespan_Years"]),
                            on="Locomotive_ID",
                            how="left"))
        
    return get_locomotive_counts(locos_to_use)

def calculate_refueler_counts(
        info: ScenarioInfo,
        _
        ) -> MetricType:
    """
    Given a single scenario year's results, counts how many refuelers were included in the simulation.
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    Outputs:
    ----------
    DataFrame of locomotive fleet composition metrics (metric name, units, value, and scenario year)
    """
    num_ports = (info.refuel_facilities
                 .groupby(["Refueler_Type"])
                 .agg(
                    pl.col("Cost_USD").mean().alias("Unit_Cost"),
                    pl.col("Port_Count").sum().alias("Count_Refuelers"),
                    pl.col("Lifespan_Years").mean().alias("Lifespan"))
                 .melt(
                    id_vars="Refueler_Type",
                    variable_name="Metric",
                    value_name="Value")
                 .with_columns(
                    pl.when(pl.col("Metric") == "Unit_Cost")
                        .then("USD")
                        .when(pl.col("Metric") == "Lifespan")
                        .then("years")
                        .otherwise("assets")
                    .alias("Units"),
                    pl.lit("All").alias("Year"))
                 .rename({"Refueler_Type": "Subset"})
    )

    return num_ports

def calculate_rollout_investments(values: MetricType) -> MetricType:
    """
    Given multiple scenario years' locomotive fleet compositions, computes additional across-year metrics
    Arguments:
    ----------
    values: DataFrame with multiple scenario years' locomotive fleet composition metrics
    Outputs:
    ----------
    DataFrame of across-year locomotive fleet composition metrics (metric name, units, value, and scenario year)
    """
    item_id_cols = ["Subset"]
    loco_types = (values
        .filter((pl.col("Units")=="assets") & (pl.col("Subset") != "All"))
        .get_column("Subset")
        .unique())
    costs = (values
        .filter(pl.col("Metric") == "Unit_Cost")
        .rename({"Value": "Unit_Cost"})
        .drop(["Metric","Units"]))
    lifespans = (values
        .filter(pl.col("Metric") == "Lifespan")
        .with_columns(pl.col("Value").cast(pl.Int32).alias("Lifespan"))
        .drop(["Metric","Value","Units"]))
    stock_values = (values
        .filter((pl.col("Metric").str.contains("Count_")) & (pl.col("Subset") != "All"))
        .drop(["Metric","Units"])
        .rename({"Value": "Count"}))
    
    #TODO: add a check for whether the # of unique combos of subset-lifespan differs from the combos of subset.
    # Calculations are not yet implemented for that case.
    years = values.get_column("Year").unique().sort()
    years = years.extend_constant(years.max()+1, 1)
    age_values = pl.DataFrame({"Age": range(1, 1 + lifespans.get_column("Lifespan").max())}, schema=[("Age", pl.Int32)])
    vintage_values = pl.DataFrame({"Age": range(years.min() - age_values.get_column("Age").max(), years.max())}, schema=[("Vintage", pl.Int32)])

    year_zero_fleet_size = (values
        .filter((pl.col("Metric")=="Count_Locomotives") & 
                (pl.col("Subset").is_in(loco_types)) &
                (pl.col("Year") == years.min()))
        .get_column("Value")
        .sum()
    )
    counts = (stock_values
        .select(item_id_cols).unique()
        .join(stock_values.select("Year").unique(), how="cross")
        .join(stock_values, on = item_id_cols + ["Year"], how="left")
        .with_columns(cs.numeric().fill_null(0))
        .sort(item_id_cols + ["Year"])
    )
    item_list = counts.select(item_id_cols).unique()
    incumbents = (counts
        .filter((pl.col("Subset").str.contains("Diesel")) & (pl.col("Year") == years.min()))
        .select(item_id_cols + ["Count"])
        .with_columns(pl.when(pl.col("Subset") == "Diesel_Large")
            .then(pl.lit(year_zero_fleet_size))
            .otherwise(pl.col("Count"))
            .alias("Count"))
    )
    
    #Initialize to only include incumbent asset counts; all else = 0
    # Start at year -1: all-incumbent fleet,
    # then immediately increment ages to get year 0 incumbent fleet
    age_tracker = (item_list.select(item_id_cols).unique()
        .join(vintage_values, how="cross")
        .join(costs, how="left", left_on=item_id_cols + ["Vintage"], right_on = item_id_cols + ["Year"])
        .join(lifespans, how="left", left_on=item_id_cols + ["Vintage"], right_on = item_id_cols + ["Year"])
        .with_columns(
            pl.col("*").backward_fill().over(item_id_cols),
            (pl.lit(years.min())-pl.col("Vintage")).alias("Age"))
        .join(incumbents, how="left", on=item_id_cols)
        .with_columns(
            (pl.col("Count") * (pl.col("Age").truediv(pl.col("Lifespan")))).round().alias("Count"),
            (pl.col("Vintage") + pl.col("Lifespan") - 1).alias("Scheduled_Final_Year"))
        .sort(item_id_cols + ["Age"])
        .with_columns(pl.when(pl.col("Age")==1)
                      .then(pl.col("Count"))
                      .when(pl.col("Age") <= 0)
                      .then(pl.lit(0.0))
                      .otherwise(pl.col("Count") - (pl.col("Count").shift().over(item_id_cols))).fill_null(0))
    )
        
    portfolio_value_initial = (age_tracker
        .select((pl.col("Count") * 1.0 * (1.0 - pl.col("Age")*1.0 / pl.col("Lifespan")*1.0) * pl.col("Unit_Cost")).sum().alias("Value"))
        .with_columns(pl.lit("Asset_Value_Initial").alias("Metric"),
                        pl.lit("USD").alias("Units"),
                        pl.lit("All").alias("Subset"),
                        pl.lit(years.min()-1).alias("Year"))
        .select(metric_columns)
    )

    #Increment ages to get year 0 incumbent fleet
    age_tracker = (age_tracker
        .with_columns(pl.col("Age") + 1)
        .filter((pl.col("Vintage") >= pl.lit(years.min())) | (pl.col("Count") > 0.0))
        .sort(item_id_cols + ["Scheduled_Final_Year"])
    )

    scheduled_retirements_year_zero = (age_tracker
        .filter(pl.col("Scheduled_Final_Year") == years.min() - 1)
        .groupby(item_id_cols)
        .agg(pl.col("Count").sum().alias("Value"))
        .with_columns(pl.lit("Retirements_Scheduled").alias("Metric"),
                        pl.lit("assets").alias("Units"),
                        pl.lit(years.min()).alias("Year"))
        .select(metric_columns)
    )

    age_tracker = (age_tracker
        .with_columns(
            pl.when(pl.col("Scheduled_Final_Year") < years.min())
                .then(0.0)
                .otherwise(pl.col("Count"))
                .alias("Count")
        )
    )

    purchase_metrics = [portfolio_value_initial, scheduled_retirements_year_zero]
    for year in years: 
        #If year 1, this includes incumbents only (prior-year retirements already removed)
        #So, we should do the same for subsequent years
        counts_prior_year_end = age_tracker.groupby(item_id_cols).agg(pl.col("Count").sum())
        counts_current_year_start = counts.filter(pl.col("Year") == year).drop("Year")
        changes = (item_list
            .join(counts_prior_year_end, on=item_id_cols, how="left")
            .join(counts_current_year_start, on=item_id_cols, how="left", suffix="_Current")
            .with_columns(cs.numeric().fill_null(0))
            .with_columns((pl.col("Count_Current") - pl.col("Count")).alias("Change"))
            .drop(["Count","Count_Current"]))
        purchases = changes.filter(pl.col("Change") > 0).with_columns(pl.lit(1).alias("Age"))
        early_retirements = changes.filter(pl.col("Change") < 0).with_columns(pl.col("Change")*-1)
        age_tracker = (age_tracker
            .join(purchases, on=item_id_cols + ["Age"], how="left")
            .with_columns(pl.when((pl.col("Change") > 0))
                            .then(pl.col("Count") + pl.col("Change"))
                            .otherwise(pl.col("Count"))
                            .alias("Count"))
            .drop("Change")
            .join(early_retirements, on=item_id_cols, how="left")
            .with_columns(pl.col("Count").cumsum().over(item_id_cols).alias("Retirements_Early_Cumsum"))
            .with_columns(pl.when(pl.col("Change") > 0)
                            .then(pl.when(pl.col("Retirements_Early_Cumsum") <= pl.col("Change"))
                                    .then(pl.col("Count"))
                                    .otherwise(pl.max_horizontal([0,
                                                       pl.min_horizontal([
                                                           pl.col("Count"), 
                                                           pl.col("Change") - (pl.col("Retirements_Early_Cumsum")-pl.col("Count"))])])))
                            .otherwise(pl.lit(0))
                            .alias("Retirements_Early"))
            .with_columns(pl.when(pl.col("Retirements_Early")>0)
                .then(pl.col("Count") - pl.col("Retirements_Early"))
                .otherwise(pl.col("Count"))
                .alias("Count"))
            .drop(["Change","Retirements_Early_Cumsum"])
        )

        recovery_share = defaults.STRANDED_ASSET_RESALE_PCT
        retirement_label = "Retirements_Early"
        if year == years[len(years)-1]:
            recovery_share = 1.0
            retirement_label = "Retirements_End_Of_Rollout"

        early_retirements = (age_tracker
            .filter((pl.col("Retirements_Early") > 0) & (pl.col("Scheduled_Final_Year") >= year))
            .groupby(item_id_cols)
            .agg(pl.col("Retirements_Early").sum().alias("Value"))
            .with_columns(pl.lit(retirement_label).alias("Metric"),
                          pl.lit("assets").alias("Units"),
                          pl.lit(year).alias("Year"))
            .select(metric_columns))
        
        scheduled_retirements = (age_tracker
            .filter((pl.col("Scheduled_Final_Year") == year) & (pl.col("Count") > 0))
            .groupby(item_id_cols)
            .agg(pl.col("Count").sum().alias("Value"))
            .with_columns(pl.lit("Retirements_Scheduled").alias("Metric"),
                          pl.lit("assets").alias("Units"),
                          pl.lit(year+1).alias("Year"))
            .select(metric_columns))
            
        early_retirement_costs = (age_tracker
            .filter(pl.col("Retirements_Early") > 0)
            .with_columns(
                pl.lit("Cost_Retired_Assets").alias("Metric"),
                pl.lit("USD").alias("Units"),
                pl.lit(year).alias("Year"),
                pl.when((pl.col("Age") <= pl.lit(year - years.min() + 1)) | (defaults.INCLUDE_EXISTING_ASSETS))
                    .then(pl.col("Retirements_Early") * -1.0 * (1.0 - pl.col("Age")*1.0 / pl.col("Lifespan")*1.0) * pl.col("Unit_Cost") * pl.lit(recovery_share))
                    .otherwise(0.0)
                    .alias("Value"))
            .groupby("Metric","Units","Year","Subset")
            .agg(pl.col("Value").sum())
            .select(metric_columns))                     
        purchases = (purchases
            .with_columns(pl.lit("Purchases").alias("Metric"),
                          pl.lit("assets").alias("Units"),
                          pl.lit(year).alias("Year"),
                          pl.col("Change").alias("Value"))
            .select(metric_columns))
        purchase_costs = (purchases
            .join(values.filter(pl.col("Metric") == "Unit_Cost"), on= item_id_cols + ["Year"], how="left")
            .with_columns((pl.col("Value") * pl.col("Value_right")).alias("Value"),
                          pl.lit("Cost_Purchases").alias("Metric"),
                          pl.lit("USD").alias("Units"))
            .select(metric_columns))
        # Remove temporary columns and increment item ages to prep for the next year
        age_tracker = (age_tracker
            .with_columns(
                pl.col("Age") + 1,
                pl.when(pl.col("Scheduled_Final_Year") <= year)
                    .then(0.0)
                    .otherwise(pl.col("Count"))
                    .alias("Count")
            )
            .drop(["Retirements_Early"])
        )
        # Add metrics for early retirements, purchases, and incurred costs
        purchase_metrics.extend([
            scheduled_retirements,
            early_retirements,
            early_retirement_costs,
            purchases,
            purchase_costs
        ])

    values = pl.concat([values] + purchase_metrics, how="diagonal")

    total_costs = (values
        .filter(pl.col("Metric").is_in(["Cost_Retired_Assets","Cost_Purchases"]))
        .groupby(["Metric","Units","Year"])
        .agg(pl.col("Value").sum())
        .with_columns(pl.lit("All").alias("Subset"))
    )
    return metrics_from_list([values, total_costs])

def calculate_rollout_total_costs(values: MetricType) -> MetricType:
    """
    Given multiple scenario years' locomotive fleet compositions, computes total per-year costs
    Arguments:
    ----------
    values: DataFrame with annual cost metrics
    Outputs:
    ----------
    DataFrame of across-year locomotive fleet composition metrics (metric name, units, value, and scenario year)
    """

    total_costs = (values
        .filter((pl.col("Metric").is_in(["Cost_Energy", "Cost_Purchases","Cost_Retired_Assets"])) &
                (pl.col("Subset") == "All"))
        .groupby(pl.col("Units","Subset","Year"))
        .agg(pl.col("Value").sum())
        .with_columns(pl.lit("Cost_Total").alias("Metric")))
    

    return metrics_from_list([values, total_costs])

def import_emissions_factors_cambium(
        location_map: Dict[str, List[alt.Location]],
        scenario_year: int,
        file_path: Path = defaults.GRID_EMISSIONS_FILE) -> pl.DataFrame:
    
    location_ids = [loc.location_id for loc_list in location_map.values() for loc in loc_list]
    grid_emissions_regions = [loc.grid_emissions_region for loc_list in location_map.values() for loc in loc_list]
    region_mappings = pl.DataFrame({
        "Node": location_ids,
        "Region": grid_emissions_regions
    }).unique()

    emissions_factors = (
        pl.scan_csv(source = file_path, skip_rows = 5)
            .filter(pl.col("gea").is_in(
                region_mappings.get_column("Region")))
            .select(pl.col("gea").alias("Region"),
                    pl.col("t").alias("Year"),
                    (pl.col("lrmer_co2e_c") + pl.col("lrmer_co2e_p")).alias("CO2eq_kg_per_MWh"),
                    ((pl.col("t") - pl.lit(scenario_year)).alias("Year_Diff")))
            .join(region_mappings.lazy(), on="Region", how="inner")
            .drop("Region")
            .collect()
    )
    if emissions_factors.get_column("Year_Diff").abs().min() == 0:
        emissions_factors = (emissions_factors
                                .filter(pl.col("Year_Diff") == 0)
                                .drop("Year_Diff")
        )
    else:
        earlier_year = (emissions_factors
                            .filter(pl.col("Year_Diff") < 0)
                            .filter(pl.col("Year_Diff") == pl.col("Year_Diff").max())
        )
        later_year = (emissions_factors
                            .filter(pl.col("Year_Diff") > 0)
                            .filter(pl.col("Year_Diff") == pl.col("Year_Diff").min())
        )
        if later_year.height == 0:
            return earlier_year
        elif earlier_year.height == 0:
            return later_year
        else:
            emissions_factors = earlier_year.join(later_year,
                                                on=["Node"],how="left")
            total_diff = emissions_factors.get_column("Year_Diff").abs() + \
                emissions_factors.get_column("Year_Diff_right").abs()
            emissions_factors = emissions_factors.with_columns(
                (pl.col("CO2eq_kg_per_MWh") * (total_diff - pl.col("Year_Diff").abs()) / total_diff +
                 pl.col("CO2eq_kg_per_MWh_right") * (total_diff - pl.col("Year_Diff_right").abs()) / total_diff
                 ).alias("CO2eq_kg_per_MWh"),
                (pl.col("Year") * (total_diff - pl.col("Year_Diff").abs()) / total_diff +
                 pl.col("Year_right") * (total_diff - pl.col("Year_Diff_right").abs()) / total_diff
                 ).cast(pl.Int64, strict=False).alias("Year")
            ).drop(["Year_right","CO2eq_kg_per_MWh_right","Year_Diff_right","Year_Diff"])

    return emissions_factors

def import_energy_prices_eia(
        location_map: Dict[str, List[alt.Location]],
        scenario_year: int,
        reference_year: int = 2022,
        reference_year_values: Dict[str, List] = {'Fuel': ["Diesel", "Electricity"],
                                                'Region': ["MN", "MN"],
                                                'Price': [3.616, 9.19]}
) -> pl.DataFrame:
    reference_year_values = pl.DataFrame(reference_year_values)
    location_ids = [loc.location_id for loc_list in location_map.values() for loc in loc_list]
    electricity_price_regions = [loc.electricity_price_region for loc_list in location_map.values() for loc in loc_list]
    liquid_fuel_price_regions = [loc.liquid_fuel_price_region for loc_list in location_map.values() for loc in loc_list]
    nodal_reference_prices = (pl.concat([
        pl.DataFrame({
            "Node": location_ids,
            "Region": electricity_price_regions
            }).unique().with_columns(pl.lit("Electricity").alias("Fuel")),
        pl.DataFrame({
            "Node": location_ids,
            "Region": liquid_fuel_price_regions
            }).unique().with_columns(pl.lit("Diesel").alias("Fuel"))
        ], how="diagonal")
        .join(reference_year_values, on=["Region", "Fuel"]))

    eia_reference_electric_price = electricity_prices_eia.filter(pl.col("Year") == reference_year).get_column("Price")[0]
    electric = (electricity_prices_eia
        .with_columns((pl.col("Price")/eia_reference_electric_price).alias("Price"),
                      (pl.col("Year") - pl.lit(scenario_year)).alias("Year_Diff")))

    eia_reference_liquid_fuel_price = liquid_fuel_prices_eia.filter(pl.col("Year") == reference_year).get_column("Price")[0]
    liquid = (liquid_fuel_prices_eia
        .with_columns((pl.col("Price")/eia_reference_liquid_fuel_price).alias("Price"),
                      (pl.col("Year") - pl.lit(scenario_year)).alias("Year_Diff")))

    if electric.get_column("Year_Diff").abs().min() == 0:
        electric = (electric
            .filter(pl.col("Year_Diff") == 0)
            .drop("Year_Diff"))
    else:
        earlier_year = (electric
            .filter(pl.col("Year_Diff") < 0)
            .filter(pl.col("Year_Diff") == pl.col("Year_Diff").max()))
        later_year = (electric
            .filter(pl.col("Year_Diff") > 0)
            .filter(pl.col("Year_Diff") == pl.col("Year_Diff").min()))
        if later_year.height == 0:
            electric = earlier_year
        elif earlier_year.height == 0:
            electric = later_year
        else:
            electric = earlier_year.join(later_year, on=["Node"],how="left")
            total_diff = electric.get_column("Year_Diff").abs() + \
                electric.get_column("Year_Diff_right").abs()
            electric = electric.with_columns(
                (pl.col("Price") * (total_diff - pl.col("Year_Diff").abs()) / total_diff +
                 pl.col("Price_right") * (total_diff - pl.col("Year_Diff_right").abs()) / total_diff
                 ).alias("Price"),
                (pl.col("Year") * (total_diff - pl.col("Year_Diff").abs()) / total_diff +
                 pl.col("Year_right") * (total_diff - pl.col("Year_Diff_right").abs()) / total_diff
                 ).cast(pl.Int64, strict=False).alias("Year")
            ).drop(["Year_right","Price_right","Year_Diff_right","Year_Diff"])

    if liquid.get_column("Year_Diff").abs().min() == 0:
        liquid = (liquid
            .filter(pl.col("Year_Diff") == 0)
            .drop("Year_Diff"))
    else:
        earlier_year = (liquid
            .filter(pl.col("Year_Diff") < 0)
            .filter(pl.col("Year_Diff") == pl.col("Year_Diff").max()))
        later_year = (liquid
            .filter(pl.col("Year_Diff") > 0)
            .filter(pl.col("Year_Diff") == pl.col("Year_Diff").min()))
        if later_year.height == 0:
            liquid = earlier_year
        elif earlier_year.height == 0:
            liquid = later_year
        else:
            liquid = earlier_year.join(later_year, on=["Node"],how="left")
            total_diff = liquid.get_column("Year_Diff").abs() + \
                liquid.get_column("Year_Diff_right").abs()
            liquid = liquid.with_columns(
                (pl.col("Price") * (total_diff - pl.col("Year_Diff").abs()) / total_diff +
                 pl.col("Price_right") * (total_diff - pl.col("Year_Diff_right").abs()) / total_diff
                 ).alias("Price"),
                (pl.col("Year") * (total_diff - pl.col("Year_Diff").abs()) / total_diff +
                 pl.col("Year_right") * (total_diff - pl.col("Year_Diff_right").abs()) / total_diff
                 ).cast(pl.Int64, strict=False).alias("Year")
            ).drop(["Year_right","Price_right","Year_Diff_right","Year_Diff"])

    liquid_multiplier = liquid.get_column("Price")[0]
    electric_multiplier = electric.get_column("Price")[0]

    nodal_energy_prices = (nodal_reference_prices
        .with_columns(pl.when(pl.col("Fuel") == pl.lit("Electricity"))
                      .then(pl.col("Price") * electric_multiplier)
                      .otherwise(pl.col("Price") * liquid_multiplier)
                      .alias("Price"))
        .select(["Node","Fuel","Price"]))
    return nodal_energy_prices

def add_battery_costs(loco_info: pd.DataFrame, year: int) -> pd.DataFrame:
    prices_to_use = (battery_prices_nrel_atb
        .filter((pl.col("Year")-pl.lit(year)).abs().min() == (pl.col("Year")-pl.lit(year)).abs()))
    usd_per_kWh = prices_to_use.filter(pl.col("Metric")=="USD_Per_kWh").get_column("Value")[0]
    usd_per_kW = prices_to_use.filter(pl.col("Metric")=="USD_Per_kW").get_column("Value")[0]
    usd_constant = prices_to_use.filter(pl.col("Metric")=="USD_Constant").get_column("Value")[0]

    for idx, row in loco_info.iterrows():
        if (hasattr(row['Rust_Loco'], "res")) and row['Rust_Loco'].res is not None:
            kW = row['Rust_Loco'].res.pwr_out_max_watts / 1000
            kWh = row['Rust_Loco'].res.energy_capacity_joules * 1e-6 * utilities.KWH_PER_MJ 
            loco_info.at[idx,'Cost_USD'] = (
                defaults.BEL_MINUS_BATTERY_COST_USD + 
                usd_constant + 
                kW * usd_per_kW + 
                kWh * usd_per_kWh) * defaults.RETAIL_PRICE_EQUIVALENT_MULTIPLIER
    return loco_info


function_mappings = {'Energy_Costs': calculate_energy_cost,
    'Mt-km': calculate_freight,
    'GHG': calculate_ghg,
    'Count_Locomotives': calculate_locomotive_counts,
    'Count_Refuelers': calculate_refueler_counts
}