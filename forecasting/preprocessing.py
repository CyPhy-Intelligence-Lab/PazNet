import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os



def load_data(path='/Volumes/Samsung_T5/INAGT_clean/m028/m028_final.csv'):
    data = pd.read_csv(path)
    df = pd.DataFrame(data, columns=["timestamp", "accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ",
                                     "emp_accelX", "emp_accelY", "emp_accelZ", "BVP",
                                     "HR", "EDA", "TEMP", "IBI", "Vehicle speed", "Accelerator pedal angle",
                                     "Parked condition",
                                     "Brake oil pressure", "Steering signal"])

    df = df.interpolate()
    df = df.fillna(method="ffill")
    df = df.fillna(method="bfill")

    df.drop('timestamp', axis=1, inplace=True)

    # code for resample
    ratio = int(504/10)
    print(df.columns)
    for column in df.columns:
        df[column] = pd.Series(signal.resample(df[column], int(len(df[column])/ratio)))
    #ts = pd.Series(df['timestamp'][::ratio]).reset_index()
    #df['timestamp'] = ts['timestamp']
    df = df.dropna(how='all')
    print(len(df))

    return df