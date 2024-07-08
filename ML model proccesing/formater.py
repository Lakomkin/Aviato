import os
import pandas as pd 

import tensorflow as tf
import numpy as np

def Find_Circle(data):
    
    cols = flight_file.columns
    cols = cols.values
    labels = dict()
    for i in range(len(cols)):
        labels[cols[i]] = i+1

    circles = []
    start = -1 
    end = -1 
    circle_leg_time = 0
    for moment in data:
        if start == -1 and (moment[labels['AltB']] > 700 and moment[labels['VSpd']] >0):
            start = moment[0]

        
        if moment[labels['TRK']] < 213 + 5 and moment[labels['TRK']] > 213 - 5: circle_leg_time+=1
        if moment[labels['TRK']] < 303 + 5 and moment[labels['TRK']] > 213 - 5: circle_leg_time+=1
        if moment[labels['TRK']] < 33 + 5 and moment[labels['TRK']] > 33 - 5  : circle_leg_time+=1
        if moment[labels['TRK']] < 123 + 5 and moment[labels['TRK']] > 123 - 5: circle_leg_time+=1   

        if moment[labels['AltB']] < 700 and start != -1:
            end = moment[0]
            if circle_leg_time > 300 and circle_leg_time < 600: circles.append((start, end))
            circle_leg_time = 0
            start = -1
            end = -1 
    return circles 


#______________________________________D_A_T_A______________________________________#
logs = os.listdir('./01723/')
print(logs)

train_routes = []
train_labels = []
test_routes = []
test_labels = []

#__________________________TRAIN_ROUTES_/_TRAIN_LABELS____________________________#
for log in logs:
    if log == '.DS_Store': continue

    flight_file = pd.read_csv('01723/'+log, encoding='cp1251', header=2)

    flight_file.columns = flight_file.columns.str.strip().str.replace(' ', '_')

    cols = flight_file.select_dtypes(exclude=['float']).columns
    flight_file[cols] = flight_file[cols].apply(pd.to_numeric, errors='coerce')

    flight_file = flight_file[['AltB', 'Latitude', 'Longitude', 'TRK', 'VSpd']]

    flight = list(flight_file.itertuples())


    #____________________DATA_PROCCESING_______________________#

    data_sample = [ [0 for _ in range(256)] for _ in range(256)]
    prev_lon = 1
    prev_lat = 1

    for i in range(flight_file.shape[0]):
        row = flight_file.iloc[i]

        if (row['Latitude'] != None and row['Longitude'] != None and row['AltB'] != None) and (row['Latitude'] != prev_lat and row['Longitude'] != prev_lon):
            if (row['Longitude'] > 52.0000 and row['Longitude'] < 52.2500) and (row['Latitude'] > 55.4500 and row['Latitude'] < 55.7000) :

                lon = int((round(row['Longitude'], 4) - 52) * 1000)
                lat = int((round(row['Latitude'], 4) - 55) * 1000) - 450 
                height = round(row['AltB'])
                #print(lon, lat, row['Longitude'], row['Latitude'] ,' | ' ,prev_lon, prev_lat)
                data_sample[lat][lon] = height / 6500 


            prev_lon = row['Longitude']
            prev_lat = row['Latitude']

    if len(Find_Circle(flight)) > 0: 
        train_labels.append(1)
    else:
        train_labels.append(0)

    train_routes.append(data_sample)
    
    print(log)
    #____________________DATA_PROCCESING_______________________#


train_routes = np.array( train_routes )
train_labels = np.array( train_labels )
test_routes = np.array( train_routes )
test_labels = np.array( train_labels )

np.save('datasets/train_routes', train_routes)
np.save('datasets/train_labels', train_labels)
np.save('datasets/test_routes', test_routes)
np.save('datasets/test_labels', test_labels)



