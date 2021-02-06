import pandas as pd
import numpy as np

samples = pd.read_csv("/Volumes/Samsung_T5/inagt_response_labels.csv", usecols=['clipname'])
used_sample_index = np.load("../data/used_samples_jdx.npy")
used_samples = samples.iloc[used_sample_index]
used_samples = used_samples.reset_index(drop=True)

list = []
annotators = pd.read_csv("../data/annotator_labels_new.csv", usecols=['video', 'driver_ans', 'safe_binary'])
for i in range(len(annotators)):
    for j in range(len(used_samples)):
        if annotators.iloc[i, 0][:-4] == used_samples.iloc[j, 0][:-4]:
            list.append(i)
            break
annotated_samples = annotators.iloc[list]
annotated_samples = annotated_samples.reset_index(drop=True)
annotated_samples.to_csv("../data/annotated_samples.csv")
print()
