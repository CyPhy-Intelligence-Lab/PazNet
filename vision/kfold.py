import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from sklearn.utils import shuffle
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dropout
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
import data_preprocessing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold

n_tsfresh = 472
n_body_pose = 14 * 2
n_hands_pose = 21 * 4
n_face_pose = 70 * 2

# param
learning_rate = float(sys.argv[1])
batch_size = int(sys.argv[2])


def load_data():
    ts = np.load('../data/concat_X_10hz_4_0.npy')
    ts_tsfresh = pd.read_csv('../data/tsfresh_features_4_0.csv')
    obj = pd.read_csv('../data/concat_objects.csv',
                      usecols=['person', 'bicyle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light',
                               'stop sign'])
    label = np.load('../data/concat_label.npy')

    concat_matrix = pd.concat([ts_tsfresh, obj], axis=1)
    concat_matrix = concat_matrix.drop('id', axis=1)
    concat_matrix['goodtime'] = pd.Series(label)

    return concat_matrix


def load_op_data():
    body_op_x = np.load('../data/body_op_x.npy')
    body_op_y = np.load('../data/body_op_y.npy')
    body_op_c = np.load('../data/body_op_c.npy')
    lhand_op_x = np.load('../data/lhand_op_x.npy')
    lhand_op_y = np.load('../data/lhand_op_y.npy')
    lhand_op_c = np.load('../data/lhand_op_c.npy')
    rhand_op_x = np.load('../data/rhand_op_x.npy')
    rhand_op_y = np.load('../data/rhand_op_y.npy')
    rhand_op_c = np.load('../data/rhand_op_c.npy')
    face_op_x = np.load('../data/face_op_x.npy')
    face_op_y = np.load('../data/face_op_y.npy')
    face_op_c = np.load('../data/face_op_c.npy')

    # remove keypoints on legs and feet
    body_op_x = np.delete(body_op_x, [10, 11, 13, 14, 18, 19, 20, 21, 22, 23, 24], 2)
    body_op_y = np.delete(body_op_y, [10, 11, 13, 14, 18, 19, 20, 21, 22, 23, 24], 2)
    body_op_c = np.delete(body_op_c, [10, 11, 13, 14, 18, 19, 20, 21, 22, 23, 24], 2)

    '''
    return np.nan_to_num(body_op_x), np.nan_to_num(body_op_y), np.nan_to_num(body_op_c), \
           np.nan_to_num(lhand_op_x), np.nan_to_num(lhand_op_y), np.nan_to_num(lhand_op_c), \
           np.nan_to_num(rhand_op_x), np.nan_to_num(rhand_op_y), np.nan_to_num(rhand_op_c), \
           np.nan_to_num(face_op_x), np.nan_to_num(face_op_y), np.nan_to_num(face_op_c)
    '''
    concat = np.concatenate((body_op_x, body_op_y, lhand_op_x, lhand_op_y,
                                    rhand_op_x, rhand_op_y, face_op_x, face_op_y), axis = 2)

    index = np.load('../data/used_samples_jdx.npy')
    return np.nan_to_num(concat[index]), np.load('../data/concat_label.npy')


def multi_conv():
    # first input model
    visible1 = Input(shape=(n_tsfresh, ))  # (None, n_tsfresh, 1)
    #flat1 = Flatten()(visible1)

    # second input model: all open pose (60, 252)
    visible2 = Input(shape=(60, n_body_pose + n_hands_pose + n_face_pose, 1))
    conv21 = Conv2D(32, kernel_size=3, activation='relu')(visible2)
    pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
    conv22 = Conv2D(16, kernel_size=3, activation='relu')(pool21)
    pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
    flat2 = Flatten()(pool22)

    # merge input models
    merge = concatenate([visible1, flat2])
    hidden1 = Dense(128, activation='relu')(merge)
    dropout1 = Dropout(0.3)(hidden1)
    hidden2 = Dense(128, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(hidden2)
    output = Dense(2, activation='softmax')(dropout2)
    model = Model(inputs=[visible1, visible2], output=output)

    print(model.summary())

    return model


def mlp():
    visible1 = Input(shape=(n_tsfresh,))
    hidden1 = Dense(128, activation='relu')(visible1)
    hidden2 = Dense(128, activation='relu')(hidden1)
    output = Dense(2, activation='softmax')(hidden2)
    model = Model(inputs=visible1, output=output)

    return model

op_data, label = load_op_data()
ts_data = load_data()
ts_data = ts_data.drop('goodtime',axis=1)

op_data = op_data.reshape(-1, 60*252)
data_concat = np.concatenate([ts_data, op_data], axis=1)

skf = StratifiedKFold(n_splits=5)
total_accuracy = []
for train_index, test_index in skf.split(data_concat, label):
    x_train = data_concat[train_index]
    y_train = label[train_index]
    x_test = data_concat[test_index]
    y_test = label[test_index]

    sm = SMOTE(random_state=0)
    oversampled_x_train, oversampled_y_train = sm.fit_resample(x_train, y_train)
    accuracy = []
    # 5 rounds of under sampling
    for i in range(5):
        us = RandomUnderSampler(random_state=0)
        undersampled_x_test, undersampled_y_test = us.fit_resample(x_test, y_test)
        #undersampled_x_test, undersampled_y_test = x_test, y_test

        x_ts = oversampled_x_train[:, :472]
        x_test_ts = undersampled_x_test[:, :472]
        x_op = oversampled_x_train[:, 480:]
        x_test_op = undersampled_x_test[:, 480:]

        x_op = x_op.reshape(-1, 60, 252)
        x_test_op = x_test_op.reshape(-1, 60, 252)

        # normalize:
        ts_scaled = data_preprocessing.norm(x_ts)
        x_test_ts_scaled = data_preprocessing.norm(x_test_ts)
        op_scaled = data_preprocessing.norm_op(x_op)
        x_test_op_scaled = data_preprocessing.norm_op(x_test_op)

        onehot_train = pd.get_dummies(oversampled_y_train, columns=['l1', 'l2'])
        onehot_test = pd.get_dummies(undersampled_y_test, columns=['l1', 'l2'])

        assert not np.any(np.isnan(ts_scaled))
        assert not np.any(np.isnan(op_scaled))
        assert not np.any(np.isnan(onehot_train))
        assert not np.any(np.isnan(onehot_test))

        # reshape input2
        op_scaled = np.expand_dims(op_scaled, axis=-1)
        x_test_op_scaled = np.expand_dims(x_test_op_scaled, axis=-1)

        '''
        model = multi_conv()
        model.compile(optimizer=Adam(learning_rate), loss=categorical_crossentropy,
                      metrics=[categorical_accuracy, ])

        model.fit(x=[ts_scaled, op_scaled], y=onehot_train, epochs=100,
                  batch_size=batch_size, validation_data=([x_test_ts_scaled, x_test_op_scaled], onehot_test))
        '''
        model = mlp()
        model.compile(optimizer=SGD(learning_rate), loss=categorical_crossentropy,
                      metrics=[categorical_accuracy, ])
        model.fit(x=ts_scaled, y=onehot_train, epochs=500, batch_size=batch_size,
                  validation_data=(x_test_ts_scaled, onehot_test), shuffle=True)

        loss, acc = model.evaluate(x_test_ts_scaled, onehot_test)
        print(acc)
        accuracy.append(acc)

        no_indices = [i for i in range(len(onehot_test)) if onehot_test.iloc[i, -1] == 0]
        no_loss, no_acc = model.evaluate(x_test_ts_scaled.iloc[no_indices, :], onehot_test.iloc[no_indices, :])
        yes_indices = [i for i in range(len(onehot_test)) if onehot_test.iloc[i, -1] == 1]
        yes_loss, yes_acc = model.evaluate(x_test_ts_scaled.iloc[yes_indices, :], onehot_test.iloc[yes_indices, :])
        print("accuracy on no: " + str(no_acc))
        print("accuracy on yes: " + str(yes_acc))

    total_accuracy.append(np.mean(accuracy))

print(total_accuracy)
print(np.mean(total_accuracy))


