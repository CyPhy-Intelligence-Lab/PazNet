import pickle
import numpy as np
import pandas as pd
import os

path = "/Volumes/Samsung_T5/inagt_response_dataset/"
df = []

person = np.zeros(2734)
bicyle = np.zeros(2734)
car = np.zeros(2734)
motorcycle = np.zeros(2734)
bus = np.zeros(2734)
truck = np.zeros(2734)
traffic_light = np.zeros(2734)
stop_sign = np.zeros(2734)

i = 0

path_list = os.listdir(path)
path_list.sort()
for p in path_list:
    event = os.listdir(path + p)
    event.sort()
    for e in event:
        maskrcnn = pickle.load(open(path + p + '/' + e + '/maskrcnn.pkl', "rb"), encoding='utf-8')
        objects = []
        for frame in maskrcnn:
            objects.append(frame['class_ids'])
        #if len(objects) == 0:
            #i += 1
        for obj in objects[60]:
            if obj == 1:
                person[i] += 1
            elif obj == 2:
                bicyle[i] += 1
            elif obj == 3:
                car[i] += 1
            elif obj == 4:
                motorcycle[i] += 1
            elif obj == 6:
                bus[i] += 1
            elif obj == 8:
                truck[i] += 1
            elif obj == 10:
                traffic_light[i] += 1
            elif obj == 12:
                stop_sign[i] += 1
            else:
                continue
        i += 1
data = {'person': pd.Series(person),
        'bicyle': pd.Series(bicyle),
        'car': pd.Series(car),
        'motorcycle': pd.Series(motorcycle),
        'bus': pd.Series(bus),
        'truck': pd.Series(truck),
        'traffic light': pd.Series(traffic_light),
        'stop sign': pd.Series(stop_sign)}

dataframe = pd.DataFrame(data)
print(dataframe.head())
dataframe.to_csv('objects')
