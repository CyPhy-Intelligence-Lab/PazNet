import numpy as np
import pandas as pd
from tensorflow import set_random_seed
import data_preprocessing
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot

TRAIN = False

ts = np.load('../data/concat_X_10hz_4_0.npy')
ts_tsfresh = pd.read_csv('../data/tsfresh_features_4_0.csv')
obj = pd.read_csv('../data/concat_objects.csv',
                  usecols=['person', 'bicyle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light',
                           'stop sign'])
label = np.load('../data/concat_label.npy')

# calculate feature importance
#model = LogisticRegression()
model = DecisionTreeClassifier()
model.fit(obj, label)
#importance = model.coef_[0]
importance = model.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.xticks([0,1,2,3,4,5,6,7], ['person', 'bicyle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light',
                           'stop sign'], rotation=45)
pyplot.ylabel('Score')
pyplot.title('Feature importance score by CART')
pyplot.show()