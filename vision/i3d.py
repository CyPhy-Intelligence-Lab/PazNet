import numpy as np
import pandas as pd
import os

i3d_path = "../data/i3d_inceptionv1/"
path_list = os.listdir(i3d_path)
path_list.sort()
feature_matrix = []
for clip in path_list:
    clip_path = i3d_path + clip
    feature_vector = np.load(clip_path)
    feature_matrix.append(feature_vector)
feature_matrix = np.array(feature_matrix)
feature_matrix = feature_matrix.reshape((-1, 1024))
index = np.load('../data/used_samples_jdx.npy')
np.save("../data/i3d_inceptionv1_features.npy", feature_matrix[index])
assert not np.any(np.isnan(feature_matrix))
print(np.max(feature_matrix))
print(np.min(feature_matrix))
