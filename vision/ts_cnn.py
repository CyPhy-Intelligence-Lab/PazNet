import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import keras
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.utils import shuffle
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.metrics import categorical_accuracy
import data_preprocessing
import keras.backend as K


def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


learning_rate = 0.005
batch_size = 32

time_series = np.load("../data/concat_X_10hz_4_0.npy")
time_series = time_series[:, :, [0, 1, 2, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18]]
label = np.load("../data/concat_label.npy")

# shuffle
ts_data, label = shuffle(time_series, label, random_state=0)

# train-test split
sep = int(len(ts_data) * 0.7)
ts_train, y_train = ts_data[:sep], label[:sep]
ts_test, y_test = ts_data[sep:], label[sep:]

y_train = pd.get_dummies(y_train, columns=['l1', 'l2'])
y_test = pd.get_dummies(y_test, columns=['l1', 'l2'])

assert not np.any(np.isnan(ts_train))
assert not np.any(np.isnan(ts_test))
print(ts_train.shape)
print(ts_test.shape)
print(y_train.shape)
print(y_test.shape)

# CAN and Physiological channel
input1 = Input(shape=(40, 15, 1))
bn1 = BatchNormalization()(input1)
conv11 = Conv2D(16, kernel_size=3, activation='relu')(bn1)
pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
bn2 = BatchNormalization()(pool11)
conv12 = Conv2D(8, kernel_size=3, activation='relu')(bn2)
pool12 = MaxPooling2D(pool_size=(2, 2))(bn2)
flat1 = Flatten()(pool12)

# Mask-rcnn and I3D



bn3 = BatchNormalization()(flat1)
hidden1 = Dense(32, activation='relu')(bn3)
dropout1 = Dropout(0.5)(hidden1)
bn4 = BatchNormalization()(dropout1)
output = Dense(2, activation='softmax')(bn4)
model = Model(input1, output)
model.summary()


model.compile(optimizer=Adam(learning_rate), loss=categorical_crossentropy, metrics=[get_f1])

model.fit(x=ts_train, y=y_train, epochs=1000,
          batch_size=batch_size, validation_data=(ts_test, y_test))
