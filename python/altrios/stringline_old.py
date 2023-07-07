# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:59:15 2021

@author: groscoe2
"""
import matplotlib.pyplot as plt
import json

with open("dispOccupancy.json") as f:
    dispResult = json.load(f)['dispResult']

for linkIdx in range(len(dispResult)):
    if dispResult[linkIdx]['trackType'] == 0:
        for entry in dispResult[linkIdx]['trainTimes']:
            plt.plot([entry['timeArrStart'],entry['timeArrEnd']],[dispResult[linkIdx]['offsetStart'],dispResult[linkIdx]['offsetEnd']],color='blue')
    else:
        for entry in dispResult[linkIdx]['trainTimes']:
            plt.plot([entry['timeArrStart'],entry['timeArrEnd']],[dispResult[linkIdx]['offsetStart'],dispResult[linkIdx]['offsetEnd']],color='orange')
plt.savefig("stringline.svg", format="svg")
