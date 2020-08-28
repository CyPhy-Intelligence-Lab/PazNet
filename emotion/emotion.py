import pandas as pd
import numpy as np

df = pd.read_csv('/Users/wuxiaodong/Downloads/emotion.csv', usecols=['path', 'most frequent', 'second most',
                                                                     'longest consecutive'])
df.sort_values('path', inplace=True, ignore_index=True)

# match the label
path = '/Volumes/Samsung_T5/inagt_response_labels.csv'
label = pd.read_csv(path, usecols=['clipname', 'goodtime'])
label_list = []
for i in range(len(df)):
    txt = df.iloc[i, 0]
    clip = txt[:-13]
    flag = 0
    for j in range(len(label)):
        if label.iloc[j, 0] == clip:
            print(1)
            label_list.append(label.iloc[j, 1])
            flag = 1
            break
    if flag == 0:
        label_list.append(np.nan)
df = df.assign(label=label_list)

df = df.dropna(how='any')
df.to_csv('emotion.csv')

print()
