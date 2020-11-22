import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import keras
from sklearn.utils import shuffle
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
from keras.losses import binary_crossentropy
import data_preprocessing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import LeaveOneGroupOut

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
    output = Dense(1, activation='softmax')(dropout2)
    model = Model(inputs=[visible1, visible2], output=output)

    print(model.summary())

    return model


op_data, label = load_op_data()
yes = [i for i in label if i==1]
no = [i for i in label if i==0]
print(len(yes))
print(len(no))
ts_data = load_data()
ts_data = ts_data.drop('goodtime',axis=1)

op_data = op_data.reshape(-1, 60*252)
data_concat = np.concatenate([ts_data, op_data], axis=1)

concat_objects = pd.read_csv('../data/concat_objects.csv')
subject_id = concat_objects['clipname'].str.slice(stop=4)
group_list = []
prev = 'm026'
i = 0
n = 0
for item in subject_id:
    if item == prev:
        group_list.append(i)
        prev = item
        n += 1
    else:
        group_list.append(i)
        prev = item
        i += 1
        n += 1

groups = np.array(group_list)
loso = LeaveOneGroupOut()

total_accuracy = []
total_recall = []
total_precision = []
for train_index, test_index in loso.split(data_concat, label, groups):
    x_train = data_concat[train_index]
    y_train = label[train_index]
    x_test = data_concat[test_index]
    y_test = label[test_index]

    sm = SMOTE(random_state=0)
    oversampled_x_train, oversampled_y_train = sm.fit_resample(x_train, y_train)
    accuracy = []
    precision = []
    recalls = []
    # 5 rounds of under sampling
    for i in range(1):
        us = RandomUnderSampler(random_state=0)
        undersampled_x_test, undersampled_y_test = us.fit_resample(x_test, y_test)
        # undersampled_x_test, undersampled_y_test = x_test, y_test

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

        assert not np.any(np.isnan(ts_scaled))
        assert not np.any(np.isnan(op_scaled))
        assert not np.any(np.isnan(oversampled_y_train))
        assert not np.any(np.isnan(undersampled_y_test))

        # reshape input2
        op_scaled = np.expand_dims(op_scaled, axis=-1)
        x_test_op_scaled = np.expand_dims(x_test_op_scaled, axis=-1)

        no_indices = [i for i in range(len(undersampled_y_test)) if undersampled_y_test[i] == 0]
        yes_indices = [i for i in range(len(undersampled_y_test)) if undersampled_y_test[i] == 1]

        metrics = [keras.metrics.TruePositives(name='tp'),
                   keras.metrics.FalsePositives(name='fp'),
                   keras.metrics.TrueNegatives(name='tn'),
                   keras.metrics.FalseNegatives(name='fn'),
                   keras.metrics.BinaryAccuracy(name='accuracy'),
                   keras.metrics.Precision(name='precision'),
                   keras.metrics.Recall(name='recall'),
                   keras.metrics.AUC(name='auc')]

        model = multi_conv()
        model.compile(optimizer=Adam(learning_rate), loss=binary_crossentropy,
                      metrics=metrics)

        model.fit(x=[ts_scaled, op_scaled], y=oversampled_y_train, epochs=50,
                  batch_size=batch_size, validation_data=([x_test_ts_scaled, x_test_op_scaled], undersampled_y_test))

        # loss, acc = model.evaluate([x_test_ts_scaled, x_test_op_scaled], undersampled_y_test)
        # loss, no_acc = model.evaluate(x=[x_test_ts_scaled.iloc[no_indices, :],
        #                              x_test_op_scaled[no_indices]], y=undersampled_y_test[no_indices])
        # loss, yes_acc = model.evaluate(x=[x_test_ts_scaled.iloc[yes_indices, :],
        #                              x_test_op_scaled[yes_indices]], y=undersampled_y_test[yes_indices])
        '''
        model = mlp()
        model.compile(optimizer=SGD(learning_rate), loss=binary_crossentropy,
                      metrics=metrics)
        model.fit(x=ts_scaled, y=oversampled_y_train, epochs=2000, batch_size=batch_size,
                  validation_data=(x_test_ts_scaled, y_test), shuffle=True)
        '''
        loss, tp, fp, tn, fn, acc, pre, recall, ruc \
            = model.evaluate([x_test_ts_scaled, x_test_op_scaled], undersampled_y_test)
        accuracy.append(acc)
        precision.append(pre)
        recalls.append(recall)
        # no_loss, no_acc = model.evaluate(x_test_ts_scaled.iloc[no_indices, :], y_test[no_indices])
        # yes_loss, yes_acc = model.evaluate(x_test_ts_scaled.iloc[yes_indices, :], y_test[yes_indices])

        # print(acc)
        # print("accuracy on no: " + str(no_acc))
        # print("accuracy on yes: " + str(yes_acc))
    total_precision.append(np.mean(precision))
    total_recall.append(np.mean(recalls))
    total_accuracy.append(np.mean(accuracy))

print(total_accuracy)
print(np.mean(total_accuracy))
print(np.mean(total_precision))
print(np.mean(total_recall))



