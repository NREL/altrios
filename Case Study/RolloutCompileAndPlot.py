# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:37:23 2023

@author: ganderson
"""

import pandas as pd
import json
import glob
import os
import SwRIPlot
import plotly.express as px



DemandFolder = 'D:/Projects/ALTRIOS/TaconiteHacking/Case Study/Conventional Freight Rollout Results'
RolloutFolder = 'D:/Projects/ALTRIOS/TaconiteHacking/Case Study/Rollout Results'
DemandFiles = glob.glob(DemandFolder + '/Taconite Base Demand*.csv')
RolloutMetricFiles = glob.glob(RolloutFolder + '/Metrics*.xlsx')

# DemandFolder = 'D:/Projects/ALTRIOS/TaconiteHacking/Case Study/Convention Freight Rollout Results Minneapolis'
# RolloutFolder = 'D:/Projects/ALTRIOS/TaconiteHacking/Case Study/Rollout Results Minneapolis'
# DemandFiles = glob.glob(DemandFolder + '/Minn* Base Demand*.csv')
# RolloutMetricFiles = glob.glob(RolloutFolder + '/Metrics*.xlsx')





TrainCountDict = {"Taconite Base Demand2095.csv" : "3 Trains Per Day",
    "Taconite Base Demand2067.csv" : "1 Train Per Day",
    "Taconite Base Demand2080.csv" : "2 Trains Per Day",
    "Minneapolis Base Demand2068.csv" : "3 Trains Per Day",
    "Minneapolis Base Demand2045.csv" : "1 Train Per Day",
    "Minneapolis Base Demand2055.csv" : "2 Trains Per Day"}



def TotalDemand(DemandData, DemandFile):
    SubSet = DemandData[DemandData.File == DemandFile]
    return SubSet['Number_of_Cars'].sum()
    


TempList = []
for File in DemandFiles:
    Temp = pd.read_csv(File)
    Temp['File'] = os.path.basename(File)
    TempList.append(Temp.copy())
    
DemandData = pd.concat(TempList)
DemandData = DemandData.drop('Unnamed: 0', axis=1)

TempList = []
for File in RolloutMetricFiles:
    Temp = pd.read_excel(File)
    Temp['File'] = os.path.basename(File).split('_')[2].replace('.xlsx','.csv')
    # TempSeries = Temp.iloc[0,:]
    # TempSeries['Metric'] = os.path.basename(File).split('_')[1].replace('.xlsx','.csv')
    # TempSeries.Units = ''
    TempList.append(Temp.copy())

RolloutMetricData = pd.concat(TempList)
RolloutMetricData['Column Name'] = RolloutMetricData['Metric'] + '[' + RolloutMetricData['Units'] + ']'
RolloutMetricData = RolloutMetricData.drop('Unnamed: 0', axis=1)


RolloutMetricData=RolloutMetricData.pivot(index=['Year', 'File'], columns='Column Name', values = 'Value')

# Years = []
# DemandFiles = []
# for index in RolloutMetricData.index:
#     Years.append(index[0])
#     DemandFiles.append(index[1].replace('DemandFile ', ''))

# RolloutMetricData['Year'] = Years 

RolloutMetricData.reset_index(inplace=True)
RolloutMetricData['File'] = RolloutMetricData['File'].str.replace('DemandFile ', '')
RolloutMetricData['Total Demand'] = RolloutMetricData['File'].map(lambda x: TotalDemand(DemandData, x))
RolloutMetricData=RolloutMetricData[RolloutMetricData.Year != 2051]
RolloutMetricData=RolloutMetricData[RolloutMetricData.Year != 'All']
RolloutMetricData['Legend Label'] = RolloutMetricData['File'].map(TrainCountDict)

#%%
# DFiles = RolloutMetricData['File'].unique()

# for DFile in DFiles:
#     DemandInterest = RolloutMetricData[RolloutMetricData['File'] == DFile]
    
#     SwRIPlot.MakeStackLineSub('Year', 
#                       [['GHG_Diesel[tonne_co2eq]','GHG_Electricity[tonne_co2eq]']], 
#                       'GHG Emissions' + DFile, 
#                       'GHG [Tons]', 
#                       'Year', 
#                       DemandInterest,
#                       auto_open=True)
    
    
#     SwRIPlot.MakeStackLineSub('Year', 
#                       [['Count_BEL[Count]','Count_Non_BEL[Count]']], 
#                       'Loco Fraction' + DFile, 
#                       ['Count'], 
#                       'Year', 
#                       DemandInterest,
#                       auto_open=True)
    
#     SwRIPlot.MakeLine('Year', 
#                       ['BEL_Pct_Locomotives[Percent]'], 
#                       'BEL Fraction' + DFile, 
#                       'BEL Fraction', 
#                       'Year', 
#                       DemandInterest,
#                       auto_open=True)
    
#     SwRIPlot.MakeStackLineSub('Year', 
#                       [['Cost_Diesel[USD]', 'Cost_Electricity[USD]']], 
#                       'Energy Cost' + DFile, 
#                       ['Cost [$]'], 
#                       'Year', 
#                       DemandInterest,
#                       auto_open=True)
    
#     SwRIPlot.MakeLine('Year', 
#                       ['LCOTKM[usd_per_million_tonne_km]'], 
#                       'Levelized Cost' + DFile, 
#                       'Cost [$]', 
#                       'Year', 
#                       DemandInterest,
#                       auto_open=True)
    
#     # fig = px.scatter_3d(RolloutMetricData, x='Year', y='Total Demand', z='Cost_Diesel[USD]')
#     # fig.write_html('test.html', auto_open=True)
RolloutFolder = RolloutFolder + '/'

SwRIPlot.MakeStackLineSubFancy('Year', 
                  [['GHG_Diesel[tonne_co2eq]','GHG_Electricity[tonne_co2eq]']], 
                  RolloutFolder + 'GHG Emissions', 
                  ['GHG Emissions [Tons]'], 
                  'Year', 
                  RolloutMetricData,
                  auto_open=True,
                  Color=['Legend Label'])


SwRIPlot.MakeStackLineSubFancy('Year', 
                [['Count_BEL[Count]','Count_Non_BEL[Count]']], 
                RolloutFolder + 'Loco Fraction', 
                ['Count'], 
                'Year', 
                RolloutMetricData,
                auto_open=True,
                Color=['Legend Label'])
    
SwRIPlot.MakeStackLineSubFancy('Year', 
                [['BEL_Pct_Locomotives[Percent]']], 
                RolloutFolder + 'BEL Fraction', 
                'BEL Fraction', 
                'Year', 
                RolloutMetricData,
                auto_open=True,
                Color=['Legend Label'])
    
SwRIPlot.MakeStackLineSubFancy('Year', 
                [['Cost_Diesel[USD]', 'Cost_Electricity[USD]']], 
                RolloutFolder + 'Energy Cost', 
                ['Cost [$]'], 
                'Year', 
                RolloutMetricData,
                auto_open=True,
                Color=['Legend Label'])
    
SwRIPlot.MakeStackLineSubFancy('Year', 
                [['LCOTKM[usd_per_million_tonne_km]']], 
                RolloutFolder + 'Levelized Cost', 
                ['Levelized Cost [$/million-tonne-km]'], 
                'Year', 
                RolloutMetricData,
                auto_open=True,
                Color=['Legend Label'])

SwRIPlot.MakeStackLineSubFancy('Year', 
                [['Cost_Total[USD]', 'Cost_BEL_New[USD]', 'Cost_Non_Bel_New[USD]']], 
                RolloutFolder + 'Total Costs', 
                ['Cost [$]'], 
                'Year', 
                RolloutMetricData,
                auto_open=True,
                Color=['Legend Label'])

print(RolloutFolder + 'Levelized Cost')
print(RolloutMetricData.columns.values)
