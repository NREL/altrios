# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 08:57:22 2022

@author: ganderson
"""

import pandas as pd
import requests
import os

File = 'data.txt'


FolderForRawData = 'GeoTiffs/'

URLData = pd.read_csv(File, names=['Download URL'])
# URLData = URLData.head()

for id, row in URLData.iterrows():
    filename = FolderForRawData + row['Download URL'].split('/')[-1]
    print(filename)
    NeedToDownload = not(os.path.exists(filename))

    if NeedToDownload:
        GeoTiff = requests.get(row['Download URL'])
        #retrieving data from the URL using get method
    
        print('file downloaded')
        
        with open(filename, 'wb') as f:
        #giving a name and saving it in any required format
        #opening the file in write mode
    
            f.write(GeoTiff.content) 
            print('file written to disk')
        #writes the URL contents from the server
    else:
        print('Already Have it')
    