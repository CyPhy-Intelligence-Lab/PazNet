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

n_tsfresh = 480
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


def mlp():
    visible1 = Input(shape=(n_tsfresh,))
    hidden1 = Dense(128, activation='relu')(visible1)
    hidden2 = Dense(128, activation='relu')(hidden1)
    output = Dense(2, activation='softmax')(hidden2)
    model = Model(inputs=visible1, output=output)
    print(model.summary())

    return model


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


def conv():
    visible1 = Input(shape=(60, n_body_pose + n_hands_pose + n_face_pose, 1))
    conv1 = Conv2D(32, kernel_size=3, activation='relu')(visible1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(16, kernel_size=3, activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flat2 = Flatten()(pool2)

    hidden1 = Dense(128, activation='relu')(flat2)
    dropout1 = Dropout(0.3)(hidden1)
    output = Dense(2, activation='softmax')(dropout1)
    model = Model(inputs=visible1, output=output)

    print(model.summary())

    return model

op_data, label = load_op_data()
ts_data = load_data()

# shuffle
ts_data, op_data = shuffle(ts_data, op_data, random_state=0)

# normalize:
# for open pose data, there are 3 options: a)sample-wise normalization; b)feature-wise; c) batch norm
ts_scaled = data_preprocessing.norm(ts_data)
op_scaled = data_preprocessing.norm_op(op_data)

# train-test split
sep = int(len(ts_scaled) * 0.7)
ts_train, op_train = ts_scaled[:sep], op_scaled[:sep]
ts_test, op_test = ts_scaled[sep:], op_scaled[sep:]

# over-sampling
balanced_ts, balanced_op = data_preprocessing.over_sampling_op(ts_train, op_train)


# one hot encoding
onehot_train = pd.get_dummies(balanced_ts.iloc[:, -1], columns=['l1', 'l2'])
onehot_test = pd.get_dummies(ts_test.iloc[:, -1], columns=['l1', 'l2'])
balanced_ts = balanced_ts.iloc[:, :-1]
ts_test = ts_test.iloc[:, :-1]


assert not np.any(np.isnan(balanced_ts))
assert not np.any(np.isnan(balanced_op))
assert not np.any(np.isnan(ts_test))
assert not np.any(np.isnan(op_test))

# reshape input2
balanced_op = np.expand_dims(balanced_op, axis=-1)
op_test = np.expand_dims(op_test, axis=-1)

model = multi_conv()
model.compile(optimizer=Adam(learning_rate), loss=categorical_crossentropy,
              metrics=[categorical_accuracy, ])

model.fit(x=[balanced_ts, balanced_op], y=onehot_train, epochs=15,
          batch_size=batch_size, validation_data=([ts_test, op_test], onehot_test))

no_indices = [i for i in range(len(onehot_test)) if onehot_test.iloc[i, -1]==0]
yes_indices = [i for i in range(len(onehot_test)) if onehot_test.iloc[i, -1]==1]
loss, accuracy = model.evaluate(x=[ts_test.iloc[no_indices, :], op_test[no_indices]], y=onehot_test.iloc[no_indices, :])
print("Performance on no")
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
loss, accuracy = model.evaluate(x=[ts_test.iloc[yes_indices, :], op_test[yes_indices]], y=onehot_test.iloc[yes_indices, :])
print("Performance on yes")
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
print(model.summary())


def multi_conv_sep():
    # first input model
    visible1 = Input(shape=(n_tsfresh, 1))  # (None, n_tsfresh, 1)

    # second input model: body pose
    visible2 = Input(shape=(n_body_pose, 60, 1))

    # third input model: hand pose
    visible3 = Input(shape=(n_hands_pose, 60, 1))

    # forth input model:
    visible4 = Input(shape=(n_face_pose, 60, 1))
