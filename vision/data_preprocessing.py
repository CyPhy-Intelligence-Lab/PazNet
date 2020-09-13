import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import select_features
from time import mktime
from datetime import datetime
from scipy import signal

import math


def load_data(path='/Volumes/Samsung_T5/INAGT_clean/m028/m028_final.csv'):
    data = pd.read_csv(path)
    df = pd.DataFrame(data, columns=["timestamp", "accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ",
                                     "emp_accelX", "emp_accelY", "emp_accelZ", "BVP",
                                     "HR", "EDA", "TEMP", "IBI", "Vehicle speed", "Accelerator pedal angle",
                                     "Parked condition",
                                     "Brake oil pressure", "Steering signal"])

    df = df.interpolate()
    df = df.fillna(method="ffill")
    df = df.fillna(method="bfill")

    #df.drop('timestamp', axis=1, inplace=True)

    # code for resample
    ratio = int(504/10)
    print(df.columns)
    for column in df.columns[1:]:
        df[column] = pd.Series(signal.resample(df[column], int(len(df[column])/ratio)))
    ts = pd.Series(df['timestamp'][::ratio]).reset_index()
    df['timestamp'] = ts['timestamp']
    df = df.dropna(how='all')
    print(len(df))

    return df


def get_response(path_list):
    data_list = []
    for path in path_list:
        data = pd.read_csv(path)
        data_list.append(data['response'])
    all_data = pd.concat(data_list, ignore_index=True)
    return all_data


def generate_df(path_list):
    data_list = []
    for path in path_list:
        tmp = load_data(path=path)
        data_list.append(tmp)
    all_data = pd.concat(data_list, ignore_index=True)
    return all_data


def load_all_data(num):
    csv_path_list = []
    path = '/Volumes/Samsung_T5/INAGT_clean/'
    #path = '/home/tongwu/data/isnowgood/INAGT_clean/'
    for root, subdirs, files in os.walk(path):
        for subdir in subdirs:
            csv_path_list.append(path + subdir + '/' + subdir + '_final.csv')
    return csv_path_list[:num]


def find_response(data):
    return [i for i, e in enumerate(data) if np.isnan(e) == False]


def sample_for_diff_win_size(num, win_size_before, win_size_after):
    path_list = load_all_data(num)
    df = generate_df(path_list)
    response = get_response(path_list)
    response_location = find_response(response)

    ratio = int(504 / 10)
    response_location = [int(i / ratio) for i in response_location]
    pre_index = [i - 10 * win_size_before for i in response_location]
    post_index = [i + 10 * win_size_after for i in response_location]
    # pick the first response and generate the initial DataFrame based on mean value
    sample_list = []
    df_list = []
    for i in range(0, len(response_location)):

        # new form of samples
        col_list = []
        for j in range(0, len(df.columns)):
            list = df.iloc[pre_index[i]:post_index[i], j].values
            col_list.append(list)
        sample_list.append(col_list)

    samples = np.array(sample_list)


    # get label list
    X_labeled = pd.read_csv('/Users/wuxiaodong/PycharmProjects/isgoodtime/data/X_labeled.csv')
    label_list = X_labeled['goodtime'].values

    # assert len(label_list) == len(samples)

    label_list[label_list == 'other'] = np.nan
    ids = np.argwhere(pd.isnull(label_list))

    # find the indices of samples with NaN values
    idxs = np.argwhere(pd.isnull(samples))[:, 0]
    idxs = np.unique(idxs)

    ids = np.unique(np.append(ids, idxs))

    label_array = np.array(label_list)
    label_array = np.delete(label_array, ids, axis=0)

    samples = np.delete(samples, ids, axis=0)
    label_array = np.where(label_array == 'yes', 1, 0)

    return samples, label_array


def add_labels(X):
    path = '/Volumes/Samsung_T5/inagt_response_labels.csv'
    df = pd.read_csv(path, usecols=['clipname', 'goodtime', 'unixtime'])
    dt_list = []
    for time in df['clipname']:
        dt = datetime.utcfromtimestamp(float(time[5:19])).strftime('%Y-%m-%d %H:%M:%S.%f')
        dt_list.append(dt)
    df = df.assign(timestamp=dt_list)

    label_list = []
    for i in range(0, len(X)):
        print(i)
        flag = 0
        for j in range(0, len(df)):
            if X['timestamp'][i][:20] == df['timestamp'][j][:20]:
                flag = 1
                break
        if flag == 1:
            label_list.append(df['goodtime'][j])
        else:
            label_list.append(np.nan)

    X = X.assign(goodtime=label_list)
    return X, label_list


def read_labeled_data():
    X_labeled = pd.read_csv('data/X_labeled.csv')
    X_labeled.drop('Unnamed: 0', axis=1, inplace=True)
    X_labeled['goodtime'].replace('other', np.nan, inplace=True)
    # drop nan instances
    X_labeled = X_labeled.dropna()

    # consider no answer as no
    X_labeled['goodtime'].replace('no answer', 'no', inplace=True)
    X_labeled['goodtime'] = X_labeled['goodtime'].map(dict(yes=1, no=0))
    x_input = X_labeled.drop('timestamp', axis=1).drop('response', axis=1)
    return x_input.reset_index(drop=True)


def read_labeled_data_with_timestamp():
    X_labeled = pd.read_csv('X_labeled.csv')
    X_labeled.drop('Unnamed: 0', axis=1, inplace=True)
    X_labeled['goodtime'].replace('other', np.nan, inplace=True)
    # drop nan instances
    X_labeled = X_labeled.dropna()

    # consider no answer as no
    X_labeled['goodtime'].replace('no answer', 'no', inplace=True)
    X_labeled['goodtime'] = X_labeled['goodtime'].map(dict(yes=1, no=0))
    x_input = X_labeled.drop('response', axis=1)

    return x_input.reset_index(drop=True)


def under_sampling(data):
    positive_indices = data[data.goodtime == 1].index
    negative_indices = data[data.goodtime == 0].index
    n = len(negative_indices)
    random_indices = np.random.choice(positive_indices, n, replace=False)
    positive_data = data.loc[random_indices]
    new_data = pd.concat([positive_data, data.loc[negative_indices]], ignore_index=True)
    return new_data


def under_sampling_two(samples, labels):
    positive_indices = [i for i, e in enumerate(labels) if e == 1]
    negative_indices = [i for i, e in enumerate(labels) if e == 0]
    n = len(negative_indices)
    random_indices = np.random.choice(positive_indices, n, replace=False)
    positive_samples = samples[random_indices]
    negative_samples = samples[negative_indices]
    positive_labels = labels[random_indices]
    negative_labels = labels[negative_indices]

    return np.append(positive_samples, negative_samples, axis=0), np.append(positive_labels, negative_labels)


def over_sampling(data):
    ros = RandomOverSampler(random_state=0)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_resampled, y_resampled = ros.fit_resample(X, y)

    return pd.concat([X_resampled, y_resampled], axis=1)


def over_sampling_op(ts_data, op_data):
    ros = RandomOverSampler(random_state=0)
    ts_X = ts_data.iloc[:, :-1]
    ts_y = ts_data.iloc[:, -1]
    ts_X_resampled, ts_y_resampled = ros.fit_resample(ts_X, ts_y)
    indices = ros.sample_indices_
    ts_resampled = pd.concat([ts_X_resampled, ts_y_resampled], axis=1)

    added_op = op_data[indices[len(ts_X):]]
    op_resampled = np.concatenate((op_data, added_op), axis=0)
    return ts_resampled, op_resampled


def over_sampling_op_smote(ts_data, op_data):
    sm = SMOTE(random_state=0)
    ts_X = ts_data.iloc[:, :-1]
    ts_y = ts_data.iloc[:, -1]
    ts_X_resampled, ts_y_resampled = sm.fit_resample(ts_X, ts_y)
    ts_resampled = pd.concat([ts_X_resampled, ts_y_resampled], axis=1)

    op_data = op_data.reshape((len(op_data), -1))
    op_resampled, op_y_resampled = sm.fit_resample(op_data, ts_y)
    op_resampled = op_resampled.reshape((-1, 60, 252))
    return ts_resampled, op_resampled


def samplinghalfhalf(ts_data, op_data, ratio):
    rus = RandomUnderSampler(random_state=0)
    ts_leave = ts_data[int(len(ts_data)*ratio):]
    op_leave = op_data[int(len(op_data)*ratio):]
    ts_X = ts_leave.iloc[:, :-1]
    ts_y = ts_leave.iloc[:, -1]
    ts_X_resampled, ts_y_resampled = rus.fit_resample(ts_X, ts_y)
    indices = rus.sample_indices_
    ts_resampled = pd.concat([ts_X_resampled, ts_y_resampled], axis=1)

    op_resampled = op_leave[indices]
    return ts_resampled, op_resampled


def norm(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data.values)
    data_scaled = pd.DataFrame(data_scaled)
    return data_scaled


def norm_op(data):
    op_list = []
    for d in data:
        min_max_scaler = preprocessing.MinMaxScaler()
        d_scaled = min_max_scaler.fit_transform(d)
        op_list.append(d_scaled)
    return np.array(op_list)


def clustering_Kmeans(X):
    y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
    X['clusters'] = y_pred
    return X


def SGD_classifier(data):
    # train/test separation
    sep = int(len(data) * 0.7)
    data_train = data[:sep]
    data_test = data[sep:]

    x_train = data_train[:, :19]
    y_train = data_train[:, -1]
    x_test = data_test[:, :19]
    y_test = data_test[:, -1]

    sgdc = SGDClassifier(loss='log')
    sgdc.fit(x_train, y_train)
    sgdc_pre = sgdc.predict(x_test)
    print(sgdc.score(x_test, y_test))
    print(classification_report(y_test, sgdc_pre))


def pca(x_scaled, n_components):
    # PCA
    reduced_data = PCA(n_components=n_components).fit_transform(x_scaled)
    results = pd.DataFrame(reduced_data)
    return results


def add_prepost_clip(data_array):
    path_list = load_all_data(64)
    df = generate_df(path_list)
    n = 0
    for i in range(len(data_array)):
        flag = 0
        for j in range(len(df)):
            if data_array[i][19] == df['timestamp'][j]:
                flag = 1
                for k in range(len(data_array[i]) - 1):
                    list = df.iloc[j - 500 * 6:j + 500 * 9, k + 1].values

                    data_array[i][k] = list
                print(n)
                n = n + 1
                break
    return data_array


def lda(data, n_components):
    X = data.iloc[:, :19]
    y = data.iloc[:, 19]
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    new = lda.fit(X, y).transform(X)
    return new


def get_tsfresh_data():
    samples = np.load('../data/concat_X_10hz_4_0.npy')
    labels = np.load('../data/concat_label.npy')
    df_list = []
    id = 0
    for sample in samples:
        df = pd.DataFrame(sample)
        df['id'] = pd.Series(np.ones(len(df)) * id)
        df_list.append(df)
        id += 1
    df = pd.concat(df_list, ignore_index=True)

    extracted_features = extract_features(df, column_id="id")
    impute(extracted_features)
    features_filtered = select_features(extracted_features, labels)
    print(features_filtered.head())
    features_filtered.to_csv('../data/tsfresh_features_4_0.csv')


def choose_features(feature_index_list):
    #samples = np.load('/home/tongwu/data/isnowgood/X_10hz.npy')
    samples = np.load('data/X_10hz.npy')
    new_samples = []
    for sample in samples:
        new_samples.append(sample[:, feature_index_list])
    new_samples = np.array(new_samples)
    np.save('data/X_10hz'+str(feature_index_list)+'.npy', new_samples)


if __name__ == '__main__':
    print()