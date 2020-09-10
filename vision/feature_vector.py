import pandas as pd
import numpy as np
from scipy import signal
from datetime import datetime

df = pd.read_csv('../data/objects.csv')
label = pd.read_csv('/Volumes/Samsung_T5/inagt_response_labels.csv', usecols=['clipname', 'goodtime'])
df['label'] = label['goodtime']
df['clipname'] = label['clipname']
dt_list = []
for time in df['clipname']:
    dt = datetime.utcfromtimestamp(float(time[5:19])).strftime('%Y-%m-%d %H:%M:%S.%f')
    dt_list.append(dt)
df = df.assign(timestamp=dt_list)

samples = np.load('../data/X_10hz_6_0_ts.npy')
sample_time_list = []
for sample in samples:
    sample_time = sample[59, 0]
    sample_time_list.append(sample_time)

idx = []
jdx = []
for i in range(len(sample_time_list)):
    for j in range(len(dt_list)):
        if sample_time_list[i][:18] == dt_list[j][:18] or (sample_time_list[i][:17] == dt_list[j][:17] and abs(abs(int(sample_time_list[i][17:19])-60)-abs(int(dt_list[j][17:19])-60))<5):
            idx.append(i)
            jdx.append(j)
            break
print()
samples = samples[idx]
samples_new = []
sample_4 = []
for sample in samples:
    samples_new.append(sample[:, 1:])
    sample_4.append(sample[19:59, 1:])
df = df.iloc[jdx, :]
df.drop('Unnamed: 0',axis=1, inplace=True)
np.save('../data/used_samples_jdx.npy', np.array(jdx))

np.save('../data/concat_X_10hz_6_0.npy', np.array(samples_new).astype(float))
np.save('../data/concat_X_10hz_4_0.npy', np.array(sample_4).astype(float))
df.to_csv('../data/concat_objects.csv')
label = np.load('/Users/wuxiaodong/PycharmProjects/isgoodtime/data/y_10hz.npy')
label = label[idx]
np.save('../data/concat_label.npy', label)
print()