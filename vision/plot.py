import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import signal


def load_data(path='/Volumes/Samsung_T5/INAGT_clean/m028/m028_final.csv'):
    data = pd.read_csv(path)
    df1 = pd.DataFrame(data, columns=["lat", "long", "accelX", "accelY", "accelZ"])

    df2 = pd.DataFrame(data, columns=["accelX", "accelY", "accelZ"])
    df3 = pd.DataFrame(data, columns=["gyroX", "gyroY", "gyroZ"])
    df4 = pd.DataFrame(data, columns=["emp_accelX", "emp_accelY", "emp_accelZ"])
    df5 = pd.DataFrame(data, columns=["BVP","HR", "EDA", "TEMP", "IBI", ])
    df6 = pd.DataFrame(data, columns=["Vehicle speed", "Accelerator pedal angle",
    "Parked condition","Driver window position",  "Turn signal operation",
    "Brake oil pressure", "Steering signal"])


    df = df1.interpolate()
    #plt.scatter(df['timestamp'], df['lat'])

    df.plot(subplots=True, legend=True)
    plt.show()

load_data()