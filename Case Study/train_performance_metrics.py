import json
import pandas as pd
import glob
import os
from scipy import integrate
import plotly.express as px

Cases =[]

Case = {'Folder' : 'D:/Projects/ALTRIOS/TaconiteHacking/Case Study/Conventional Freight Rollout Results/*.json',
        'Route' : 'Taconite Freight Sweep'}
Cases.append(Case)

Case = {'Folder' : 'D:/Projects/ALTRIOS/TaconiteHacking/Case Study/Convention Freight Rollout Results Minneapolis/*.json',
        'Route' : 'Minneapolis Freight Sweep'}
Cases.append(Case)

for Case in Cases:
    Files = glob.glob(Case['Folder'])
    
    TrainList = []
    for File in Files:
        print(File)
        with open(File) as json_file:
            data = json.load(json_file)
        
        BELCount = 0
        ConvCount = 0   
        df_denormalized_loco = pd.DataFrame()
        TrainID = 0
        for train in data:
            train_perf_dict ={}
            train_perf_dict['Train ID'] = train['train_id'] #TrainID
            train_perf_dict['destination'] = train['dests'][0]['Location ID'] 
            train_perf_dict['origin'] = train['origs'][0]['Location ID'] 
            train_perf_dict['File'] = os.path.basename(File)
            train_perf_dict['Loco Count'] = 0
            train_perf_dict['BEL Count'] = 0
            LocoPosition = 0
            train_history =  pd.DataFrame.from_dict(train['history'])
            
            newcols=[]
            for col in train_history.columns:
                newcols.append(col + "_train")
            train_history.columns = newcols
            
            df_locos = pd.DataFrame()
            BEL = 0
            for loco in train['loco_con']['loco_vec']:
                loco_history = pd.DataFrame.from_dict(loco['history'])
                
                train_perf_dict['Loco Count'] = train_perf_dict['Loco Count'] + 1
                
                try:
                    fc_history =  pd.DataFrame.from_dict(loco['loco_type']['ConventionalLoco']['fc']['history']).add_suffix('_fc')
                    temp = pd.concat([loco_history, fc_history], axis=1)
                    # print('Conventional')
                    ConvCount = ConvCount + 1
                except:
                    res_history =  pd.DataFrame.from_dict(loco['loco_type']['BatteryElectricLoco']['res']['history'])
                    res_history = res_history.add_suffix('_res')
                    temp = pd.concat([loco_history, res_history], axis=1)
                    BELCount = BELCount + 1
                    train_perf_dict['BEL Count'] = train_perf_dict['BEL Count'] + 1
                    BEL = 1
                    
                # temp = temp.reset_index(drop=True)
                
                # df_locos.reset_index(drop=True, inplace=True)
                df_locos = pd.concat([df_locos, temp], axis=0, ignore_index=True)
            
            df_locos2 = df_locos.groupby('i').sum()
            df_locos2.reset_index(inplace=True, drop=True)
            
            if BEL == 1:
                df_locos = df_locos[['soc_res', 'i']].groupby('i').mean().add_suffix('_avg')
                df_locos.reset_index(inplace=True, drop=True)
                df_locos = pd.concat([train_history, df_locos2, df_locos], axis=1)
                # print(df_locos.columns)
            else:
                df_locos = pd.concat([train_history, df_locos2], axis=1)
                
            df_locos['Train ID'] = train['train_id'] #TrainID
                # temp['Loco Position'] = LocoPosition
                # LocoPosition = LocoPosition + 1
                
        
            df_denormalized_loco = pd.concat([df_denormalized_loco, df_locos], axis = 0, ignore_index=True)
            train_perf_dict['Mean Speed [m/s]'] = train_history['velocity_train'].mean()
            train_perf_dict['train length [m]'] = train['state']['length']
            train_perf_dict['train length [ft]'] = train_perf_dict['train length [m]'] * 3.2808399
            train_perf_dict['Train Mass [kg]'] = train['state']['mass_static']
            train_perf_dict['Train Mass [tons]'] = train_perf_dict['Train Mass [kg]'] / 907.18474
            
            train_history['Moving'] = train_history['velocity_train'] > 1
            train_history['Moving'] = train_history.Moving * 1
            train_perf_dict['Stops'] = train_history.Moving.diff().clip(lower=0,upper=1).sum()
            train_perf_dict['Miles Traveled'] = train_history['offset_train'].values[-1]*0.00062137
            train_perf_dict['Stops Per Mile'] = train_perf_dict['Stops'] / train_perf_dict['Miles Traveled']
            train_perf_dict['Number of Trains'] = len(data)
            TrainList.append(train_perf_dict)
            
            # fig = px.line(train_history, x='time_train', y=['velocity_train', 'speed_limit_train','speed_target_train'])
            # fig.write_html(File.replace('.json', ' ' + str(train['train_id']) + '.html'))
            TrainID = TrainID + 1
                
        # df_denormalized_loco['UID'] = df_denormalized_loco['Train ID'].astype(str)  + '_' + df_denormalized_loco['Loco Position'].astype(str) 
        if BELCount > 0:
            df_denormalized_loco['soc_res_avg'] = df_denormalized_loco['soc_res_avg'].fillna(0)
        
        
    Data = pd.DataFrame(TrainList)
    Data.to_excel('TrainStats - {}.xlsx'.format(Case['Route']))
    MeanData=Data.groupby(['File', 'destination', 'origin']).mean()
    MeanData.to_csv('MeanResults - {}.csv'.format(Case['Route']))
