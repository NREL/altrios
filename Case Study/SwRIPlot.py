# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 18:44:20 2023

@author: ganderson
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
from scipy import integrate
# import inspect
# import sys

# print(sys.modules['__main__'])
# x=sys.modules['__main__']
# print(x.__file__)

def StatCalc(Data, Target, Feedback):
    StatSeries = pd.Series()
    Data.dropna(subset=[Target, Feedback], inplace=True)
    
    IntegratedTarget = integrate.cumtrapz(Data[Target], Data['TimStamp'])
    IntegratedFeedback = integrate.cumtrapz(Data[Feedback], Data['TimStamp'])
    
    StatSeries['Integrated Error ' + Feedback]  = IntegratedFeedback[-1] - IntegratedTarget[-1]
    
    StatSeries['Maximum Error ' + Feedback] = np.max(np.abs(Data[Feedback] - Data[Target]))
    
    StatSeries['RMSE ' + Feedback] = (np.sum((Data[Feedback] - Data[Target])**2)/Data.shape[0])**.5
    
    return StatSeries
    

#%%
def MakeLineSubFixLegend(XColumn, YColumns, PlotName, Yaxis, Xaxis, Data, auto_open=False, EnableFilter=False, MeetingDate=''):
    """XColumn - is the column of the dataframe that you want on the X Axis.  
    Specify this as a string
    
    YColumns - This will be a list of column names.  Even one column should
    still be a list
    
    PlotName - This will be a string that will be the filename for the plot
    
    YAxis / XAxis - This is a string that will be the axis label
    
    Data - This will be the dataframe with the data
    
    auto_open - This will open the plot when created if set to true.  Its
    an optional argument and is false by default."""
    
    
    #change this line to scatter or bar if you want different plot types.
    #might be worth making multiple functions if you like.
    #check out their documentation here:
    #https://plotly.com/python/plotly-express/
    

    
    i=0
    for collist in YColumns:
        print('-------------------------------')
        print(collist)
        fig = px.line(Data, x=XColumn, y=collist)

        fig.update_xaxes(tickfont=dict(family='Arial', size=30))
        fig.update_yaxes(tickfont=dict(family='Arial', size=30))
        fig.update_yaxes(title_font=dict(size=38, family='Arial'))
    
        fig.update_xaxes(title_text=Xaxis, title_font=dict(size=38, family='Arial'))
        fig.update_yaxes(title_text=Yaxis[i], title_font=dict(size=38, family='Arial'))
        fig.update_layout(
            legend_title=" ",
                font=dict(
                    size=28,
                        )
                )
            
        if i == 0:
            FileOption = 'w'
        else:
            FileOption = 'a'
    
        with open(PlotName + ' ' + MeetingDate + '.html', FileOption) as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        
        i=i+1


def MakeLineSub(XColumn, YColumns, PlotName, Yaxis, Xaxis, Data, auto_open=False, EnableFilter=False, MeetingDate=''):
    """XColumn - is the column of the dataframe that you want on the X Axis.  
    Specify this as a string
    
    YColumns - This will be a list of column names.  Even one column should
    still be a list
    
    PlotName - This will be a string that will be the filename for the plot
    
    YAxis / XAxis - This is a string that will be the axis label
    
    Data - This will be the dataframe with the data
    
    auto_open - This will open the plot when created if set to true.  Its
    an optional argument and is false by default."""
    
    
    #change this line to scatter or bar if you want different plot types.
    #might be worth making multiple functions if you like.
    #check out their documentation here:
    #https://plotly.com/python/plotly-express/
    
    fig = make_subplots(rows = len(YColumns), cols=1, shared_xaxes=True, vertical_spacing=0.01)
    
    i=1
    for collist in YColumns:
        for col in collist:
            if EnableFilter:
                sos = signal.butter(2, 5, btype ='lowpass', fs=100, output='sos')
                filtered = signal.sosfilt(sos, Data[col])
            else:
                filtered = Data[col]
            fig.add_trace(go.Scatter(x=Data[XColumn], y=filtered,
                                mode='lines',
                                name=col), row=i, col=1)
            fig.update_yaxes(title_text=Yaxis[i-1], title_font=dict(size=20, family='Arial'), row=i, col=1)
        i=i+1
        

    fig.update_xaxes(tickfont=dict(family='Arial', size=20))
    fig.update_yaxes(tickfont=dict(family='Arial', size=20))
    fig.update_yaxes(title_font=dict(size=38, family='Arial'))

    fig.update_xaxes(title_text=Xaxis, title_font=dict(size=20, family='Arial'), row=len(YColumns), col=1)
    # fig.update_yaxes(title_text=Yaxis, title_font=dict(size=20, family='Arial'))
    fig.update_layout(
        legend_title=" ",
            font=dict(
                size=20,
    )
)
    fig['layout'].update(height=len(YColumns)*300)
    pio.write_html(fig, file=PlotName + ' ' + MeetingDate + '.html', auto_open=auto_open)
    
def MakeLine(XColumn, YColumns, PlotName, Yaxis, Xaxis, Data, auto_open=False, MeetingDate=''):
    """XColumn - is the column of the dataframe that you want on the X Axis.  
    Specify this as a string
    
    YColumns - This will be a list of column names.  Even one column should
    still be a list
    
    PlotName - This will be a string that will be the filename for the plot
    
    YAxis / XAxis - This is a string that will be the axis label
    
    Data - This will be the dataframe with the data
    
    auto_open - This will open the plot when created if set to true.  Its
    an optional argument and is false by default."""
    
    
    #change this line to scatter or bar if you want different plot types.
    #might be worth making multiple functions if you like.
    #check out their documentation here:
    #https://plotly.com/python/plotly-express/
    fig = px.line(Data, x=XColumn, y=YColumns)

    fig.update_xaxes(tickfont=dict(family='Arial', size=30))
    fig.update_yaxes(tickfont=dict(family='Arial', size=30))
    fig.update_yaxes(title_font=dict(size=38, family='Arial'))

    fig.update_xaxes(title_text=Xaxis, title_font=dict(size=38, family='Arial'))
    fig.update_yaxes(title_text=Yaxis, title_font=dict(size=38, family='Arial'))
    fig.update_layout(
        legend_title=" ",
            font=dict(
                size=28,
    )
)
    pio.write_html(fig, file=PlotName + ' ' + MeetingDate + '.html', auto_open=auto_open)
 
    
def MakeScatter(XColumn, YColumns, PlotName, Yaxis, Xaxis, Data, color='', auto_open=False, MeetingDate='', Size=15):
    """XColumn - is the column of the dataframe that you want on the X Axis.  
    Specify this as a string
    
    YColumns - This will be a list of column names.  Even one column should
    still be a list
    
    PlotName - This will be a string that will be the filename for the plot
    
    YAxis / XAxis - This is a string that will be the axis label
    
    Data - This will be the dataframe with the data
    
    auto_open - This will open the plot when created if set to true.  Its
    an optional argument and is false by default."""
    
    Data['Size'] = Size
    #change this line to scatter or bar if you want different plot types.
    #might be worth making multiple functions if you like.
    #check out their documentation here:
    #https://plotly.com/python/plotly-express/
    if color== '':
        fig = px.scatter(Data, x=XColumn, y=YColumns, size='Size')
    else:
        fig = px.scatter(Data, x=XColumn, y=YColumns, color=color, size='Size')
        
    fig.update_xaxes(tickfont=dict(family='Arial', size=30))
    fig.update_yaxes(tickfont=dict(family='Arial', size=30))
    fig.update_yaxes(title_font=dict(size=38, family='Arial'))

    fig.update_xaxes(title_text=Xaxis, title_font=dict(size=38, family='Arial'))
    fig.update_yaxes(title_text=Yaxis, title_font=dict(size=38, family='Arial'))
    fig.update_layout(
        legend_title=" ",
            font=dict(
                size=28,
    )
)
    pio.write_html(fig, file=PlotName + ' ' + MeetingDate + '.html', auto_open=auto_open) 
    
def MapColors(ColorMap, index):
    # used to map the color to an index  
    return ColorMap[index % len(ColorMap)]
        
def MakeStackLineSub(XColumn, YColumns, PlotName, Yaxis, Xaxis, Data, auto_open=False, EnableFilter=False, MeetingDate=''):
    """XColumn - is the column of the dataframe that you want on the X Axis.  
    Specify this as a string
    
    YColumns - This will be a list of column names.  Even one column should
    still be a list
    
    PlotName - This will be a string that will be the filename for the plot
    
    YAxis / XAxis - This is a string that will be the axis label
    
    Data - This will be the dataframe with the data
    
    auto_open - This will open the plot when created if set to true.  Its
    an optional argument and is false by default.
    
    
    """
 
    
    #change this line to scatter or bar if you want different plot types.
    #might be worth making multiple functions if you like.
    #check out their documentation here:
    #https://plotly.com/python/plotly-express/
    
    TotalHeight = len(YColumns)*800
    RowHeights = (np.ones(len(YColumns))/TotalHeight).tolist()
    fig = make_subplots(rows = len(YColumns), cols=1, shared_xaxes=True, row_heights=RowHeights)#, vertical_spacing=0.01)
    
    i=1
    for collist in YColumns:
        for col in collist:
            if EnableFilter:
                sos = signal.butter(2, 5, btype ='lowpass', fs=100, output='sos')
                filtered = signal.sosfilt(sos, Data[col])
            else:
                filtered = Data[col]
            fig.add_trace(go.Scatter(x=Data[XColumn], y=filtered,
                                mode='lines',
                                name=col,
                                stackgroup='Data'), row=i, col=1)
            fig.update_yaxes(title_text=Yaxis[i-1], title_font=dict(size=20, family='Arial'), row=i, col=1)
        i=i+1
        

    fig.update_xaxes(tickfont=dict(family='Arial', size=20))
    fig.update_yaxes(tickfont=dict(family='Arial', size=20))
    fig.update_yaxes(title_font=dict(size=38, family='Arial'))

    fig.update_xaxes(title_text=Xaxis, title_font=dict(size=20, family='Arial'), row=len(YColumns), col=1)
    # fig.update_yaxes(title_text=Yaxis, title_font=dict(size=20, family='Arial'))
    fig.update_layout(
        height=TotalHeight,
        legend_title=" ",
            font=dict(
                size=20,
    )
)
    fig['layout'].update(height=len(YColumns)*800)
    pio.write_html(fig, file=PlotName + ' ' + MeetingDate + '.html', auto_open=auto_open)

    
def MakeStackLineSubFancy(XColumn, YColumns, PlotName, Yaxis, Xaxis, Data, auto_open=False, EnableFilter=False, MeetingDate='', Color=[], LineStyle=[]):
    """XColumn - is the column of the dataframe that you want on the X Axis.  
    Specify this as a string
    
    YColumns - This will be a list of column names.  Even one column should
    still be a list
    
    PlotName - This will be a string that will be the filename for the plot
    
    YAxis / XAxis - This is a string that will be the axis label
    
    Data - This will be the dataframe with the data
    
    auto_open - This will open the plot when created if set to true.  Its
    an optional argument and is false by default.
    
    Color - List of columns used to set line color
    
    """
    if len(Color) > 0:
        # https://stackoverflow.com/questions/35268817/unique-combinations-of-values-in-selected-columns-in-pandas-data-frame-and-count
        UniqueRows=Data[Color].groupby(Color).size().reset_index()
        
        UniqueRows['Color'] = UniqueRows.index.map(lambda x: MapColors(pc.qualitative.Plotly, x))
        
        Data = pd.merge(Data, UniqueRows, how='left', left_on=Color, right_on=Color)
    
    PossibleLines = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']

    
    #change this line to scatter or bar if you want different plot types.
    #might be worth making multiple functions if you like.
    #check out their documentation here:
    #https://plotly.com/python/plotly-express/
    
    TotalHeight = len(YColumns)*800
    RowHeights = (np.ones(len(YColumns))/TotalHeight).tolist()
    fig = make_subplots(rows = len(YColumns), cols=1, shared_xaxes=True, row_heights=RowHeights)#, vertical_spacing=0.01)
    
    i=1
    
    for collist in YColumns:
        PossibleLines2 = PossibleLines[0:len(collist)]
        LineDict = dict(zip(collist, PossibleLines2))
        for col in collist:
            for idx, row in UniqueRows.iterrows():
                print('------------------')
                for ColorCol in Color:
                    DataSubSet = Data[Data[ColorCol] == row[ColorCol]]

                print("{}, {}".format(col, row[Color[0]]))
                print(row['Color'])
                print(LineDict[col])
                if EnableFilter:
                    sos = signal.butter(2, 5, btype ='lowpass', fs=100, output='sos')
                    filtered = signal.sosfilt(sos, DataSubSet[col])
                else:
                    filtered = DataSubSet[col]
                print(DataSubSet.shape)
                    
                fig.add_trace(go.Scatter(x=DataSubSet[XColumn], 
                                        y=filtered,
                                        mode='lines',
                                        name="{}, {}".format(col, row[Color[0]]),
                                        # stackgroup=row[Color[0]],
                                        line=dict(color=row['Color'],
                                                dash=LineDict[col]
                                            ))
                            , row=i, col=1)
        fig.update_yaxes(title_text=Yaxis[i-1], title_font=dict(size=20, family='Arial'), row=i, col=1)
        i=i+1
        

    fig.update_xaxes(tickfont=dict(family='Arial', size=20))
    fig.update_yaxes(tickfont=dict(family='Arial', size=20))
    fig.update_yaxes(title_font=dict(size=38, family='Arial'))

    fig.update_xaxes(title_text=Xaxis, title_font=dict(size=20, family='Arial'), row=len(YColumns), col=1)
    # fig.update_yaxes(title_text=Yaxis, title_font=dict(size=20, family='Arial'))
    fig.update_layout(
        height=TotalHeight,
        legend_title=" ",
            font=dict(
                size=20,
    )
)
    fig['layout'].update(height=len(YColumns)*800)
    pio.write_html(fig, file=PlotName + ' ' + MeetingDate + '.html', auto_open=auto_open)
    