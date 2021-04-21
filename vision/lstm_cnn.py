import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import keras
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn import preprocessing
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
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.metrics import categorical_accuracy
import data_preprocessing
import keras.backend as K
from keras import initializers
from imblearn.over_sampling import SMOTE
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# param
learning_rate = float(sys.argv[1])
batch_size = int(sys.argv[2])
decay_rate = float(sys.argv[3])
l2_value = float(sys.argv[4])
epoch = int(sys.argv[5])


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def shuffle_train_test_split(size, ratio):
    index = np.arange(size)
    shuffle(index, random_state=0)
    sep = int(size * ratio)
    return index[:sep], index[sep:]


# read data
time_series = np.load("../data/concat_X_10hz_6_0.npy")
# time_series = time_series[:, :, [0, 1, 2, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18]]
open_pose = np.load("../data/op.npy")
i3d_inception_features = np.load("../data/i3d_inceptionv1_features.npy")

# drivers answer
#label = np.load("../data/concat_label.npy")

# annotated label
annotator_label = pd.read_csv("../data/annotated_samples.csv", usecols=['safe_binary'])
label = np.array(annotator_label).T.flatten()


# normalization
ts_min_max_scaler = preprocessing.MinMaxScaler()
op_min_max_scaler = preprocessing.MinMaxScaler()

time_series = ts_min_max_scaler.fit_transform(time_series.reshape(-1, 19))
time_series = time_series.reshape(-1, 60, 19)
open_pose = op_min_max_scaler.fit_transform(open_pose.reshape(-1, 252))
open_pose = open_pose.reshape(-1, 60, 252)

'''
# shuffle
# ts_data, label = shuffle(time_series, open_pose, i3d_inception_features, label, random_state=0)
# train-test split
train_index, test_index = shuffle_train_test_split(len(label), 0.9)
ts_train, op_train, i3d_train, y_train = time_series[train_index], open_pose[train_index], \
                                         i3d_inception_features[train_index], label[train_index]
ts_test, op_test, i3d_test, y_test = time_series[test_index], open_pose[test_index], \
                                     i3d_inception_features[test_index], label[test_index]

# over sampling on training set
sm = SMOTE(random_state=0)
ts_train = np.array(ts_train)
ts_train = ts_train.reshape(-1, 60 * 19)
op_train = op_train.reshape(-1, 60 * 252)
X_train = np.concatenate((ts_train, op_train, i3d_train), axis=1)
oversampled_X_train, oversampled_y_train = sm.fit_resample(X_train, np.array(y_train))
oversampled_ts_train = oversampled_X_train[:, :60 * 19].reshape(-1, 60, 19)
oversampled_op_train = oversampled_X_train[:, 60 * 19:60 * (252 + 19)].reshape(-1, 60, 252)
oversampled_i3d_train = oversampled_X_train[:, 60 * (252 + 19):]

oversampled_y_train = pd.get_dummies(oversampled_y_train, columns=['l1', 'l2'])
y_test = pd.get_dummies(y_test, columns=['l1', 'l2'])
'''
time_series = time_series.reshape(-1, 60*19)
open_pose = open_pose.reshape(-1, 60*252)
data_concat = np.concatenate((time_series, open_pose, i3d_inception_features), axis=1)
skf = StratifiedKFold(n_splits=10)
total_score = []
for train_index, test_index in skf.split(data_concat, label):
    X_train = data_concat[train_index]
    y_train = label[train_index]
    X_test = data_concat[test_index]
    y_test = label[test_index]

    score = []

    # over sampling on training set
    sm = SMOTE(random_state=0)
    oversampled_X_train, oversampled_y_train = sm.fit_resample(X_train, y_train)

    oversampled_ts_train = oversampled_X_train[:, :60 * 19].reshape(-1, 60, 19)
    oversampled_op_train = oversampled_X_train[:, 60 * 19:60 * (252 + 19)].reshape(-1, 60, 252)
    oversampled_i3d_train = oversampled_X_train[:, 60 * (252 + 19):]

    ts_test = X_test[:, :60 * 19].reshape(-1, 60, 19)
    op_test = X_test[:, 60 * 19:60 * (252 + 19)].reshape(-1, 60, 252)
    i3d_test = X_test[:, 60 * (252 + 19):]

    oversampled_y_train = pd.get_dummies(oversampled_y_train, columns=['l1', 'l2'])
    y_test = pd.get_dummies(y_test, columns=['l1', 'l2'])

    assert not np.any(np.isnan(oversampled_X_train))
    assert not np.any(np.isnan(oversampled_y_train))

    print("Training")
    print(oversampled_ts_train.shape)
    print(oversampled_op_train.shape)
    print(oversampled_i3d_train.shape)
    print(oversampled_y_train.shape)
    print("Test")
    print(ts_test.shape)
    print(op_test.shape)
    print(i3d_test.shape)
    print(y_test.shape)

    CAN = [0, 1, 2, 3, 4, 5, 14, 15, 16, 17, 18]
    physiological = [6, 7, 8, 9, 10, 11, 12, 13]

    # CAN channel
    input1 = Input(shape=(60, 11, 1))
    bn11 = BatchNormalization()(input1)
    conv11 = Conv2D(c11, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_value),
                    bias_regularizer=l2(l2_value))(bn11)
    pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
    bn12 = BatchNormalization()(pool11)
    conv12 = Conv2D(c12, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_value),
                    bias_regularizer=l2(l2_value))(bn12)
    pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
    reshape1 = Reshape((13, 16))(pool12)
    lstm1 = LSTM(64, activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(reshape1)
    flat1 = Flatten()(lstm1)

    # Physiological channel
    input2 = Input(shape=(60, 8, 1))
    bn21 = BatchNormalization()(input2)
    conv21 = Conv2D(c21, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_value),
                    bias_regularizer=l2(l2_value))(bn21)
    pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
    bn22 = BatchNormalization()(pool21)
    conv22 = Conv2D(c22, kernel_size=2, activation='relu', kernel_regularizer=l2(l2_value),
                    bias_regularizer=l2(l2_value))(bn22)
    pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
    reshape2 = Reshape((14, 16))(pool22)
    lstm2 = LSTM(64, activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(reshape2)
    flat2 = Flatten()(lstm2)

    # open pose channel
    n_body_pose = 14 * 2
    n_hands_pose = 21 * 4
    n_face_pose = 70 * 2

    input3 = Input(shape=(60, n_body_pose + n_hands_pose + n_face_pose, 1))
    bn31 = BatchNormalization()(input3)
    conv31 = Conv2D(c31, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_value),
                    bias_regularizer=l2(l2_value))(bn31)
    pool31 = MaxPooling2D(pool_size=(2, 2))(conv31)
    bn32 = BatchNormalization()(pool31)
    conv32 = Conv2D(c32, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_value),
                    bias_regularizer=l2(l2_value))(bn32)
    pool32 = MaxPooling2D(pool_size=(2, 2))(conv32)
    reshape3 = Reshape((13 * 61, 32))(pool32)
    lstm3 = LSTM(64, activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(reshape3)
    flat3 = Flatten()(lstm3)

    # I3D
    i3d_resnet_dimension = 2048
    i3d_inception_dimension = 1024

    input4 = Input(shape=(i3d_inception_dimension,))
    bn41 = BatchNormalization()(input4)
    # flat4 = Flatten()(bn41)

    # concatenate
    merge = concatenate([flat1, flat2, flat3, bn41])

    bn3 = BatchNormalization()(merge)
    hidden1 = Dense(8, activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(bn3)
    dropout1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(4, activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(dropout1)
    dropout2 = Dropout(0.5)(hidden2)
    bn4 = BatchNormalization()(dropout2)
    output = Dense(2, activation='softmax')(bn4)
    model = Model([input1, input2, input3, input4], output)
    model.summary()

    # learning rate decay
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=decay_rate)

    # early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

    # save the best model by measuring F1-score
    mc = ModelCheckpoint("checkpoints/best_4channel_CNN_LSTM_64_OS_"
                         + str(learning_rate) + "_" + str(decay_rate) + "_" + str(l2_value)+ '_'+str(batch_size) + ".h5",
                         monitor='val_get_f1', mode='max', verbose=2, save_best_only=True)

    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=categorical_crossentropy, metrics=[get_f1, categorical_accuracy])

    history = model.fit(x=[oversampled_ts_train[:, :, CAN], oversampled_ts_train[:, :, physiological],
                 oversampled_op_train, oversampled_i3d_train], y=oversampled_y_train, epochs=epoch,
              batch_size=batch_size, validation_data=([ts_test[:, :, CAN], ts_test[:, :, physiological],
                                                       op_test, i3d_test], y_test), callbacks=[es, mc])
    total_score.append(mc.best)
print(np.mean(total_score))
