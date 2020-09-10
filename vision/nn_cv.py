import numpy as np
import pandas as pd
from tensorflow import set_random_seed
import data_preprocessing
import tensorflow as tf
from sklearn.model_selection import KFold

TRAIN = True

ts = np.load('../data/concat_X_10hz_4_0.npy')
ts_tsfresh = pd.read_csv('../data/tsfresh_features_4_0.csv')
obj = pd.read_csv('../data/concat_objects.csv',
                  usecols=['person', 'bicyle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light',
                           'stop sign'])
label = np.load('../data/concat_label.npy')

concat_matrix = pd.concat([ts_tsfresh, obj], axis=1)


set_random_seed(2)

samples = concat_matrix
labels = label

samples['goodtime'] = pd.Series(labels)
samples = samples.drop('id', axis=1)

data = data_preprocessing.over_sampling(samples)

onehot = pd.get_dummies(data['goodtime'], columns=['l1', 'l2'])
data = data.drop('goodtime', axis=1)

# normalize
data_scaled = data_preprocessing.norm(data)

# data_scaled = data_scaled.join(onehot, lsuffix='_left', rsuffix='_right')
data_scaled = pd.concat([data_scaled, onehot], axis=1)
data_scaled = data_scaled.values.astype(np.float32)

np.random.shuffle(data_scaled)

assert not np.any(np.isnan(data_scaled))


# Define per-fold score containers
acc_per_fold = []
prec_per_fold = []
recall_per_fold = []
loss_per_fold = []

kfold = KFold(n_splits=10)
nfold = 0
for train_index, test_index in kfold.split(data_scaled):
    data_train = data_scaled[train_index]
    data_test = data_scaled[test_index]

    num_cols = len(data_scaled[0])

    # NN

    tf.reset_default_graph()

    tf_input = tf.placeholder(tf.float32, [None, num_cols], "input")

    tfx = tf_input[:, :num_cols - 2]
    tfy = tf_input[:, num_cols - 2:]

    l1 = tf.layers.dense(tfx, 128, tf.nn.relu, "l1")
    l2 = tf.layers.dense(l1, 128, tf.nn.relu, "l2")
    out = tf.layers.dense(l2, 2, tf.nn.softmax, "output")

    prediction = tf.nn.softmax(out, name="pred")

    loss = tf.losses.softmax_cross_entropy(tfy, out)
    accuracy = tf.metrics.accuracy(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
    precision = tf.metrics.precision(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
    recall = tf.metrics.recall(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
    fn = tf.metrics.false_negatives(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
    fp = tf.metrics.false_positives(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
    tp = tf.metrics.true_positives(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
    tn = tf.metrics.true_negatives(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
    auc = tf.metrics.auc(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_op = opt.minimize(loss)

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    for t in range(500000):
        batch_index = np.random.randint(len(data_train), size=32)
        sess.run(train_op, {tf_input: data_train[batch_index]})

        if t % 500 == 0:
            # testing
            acc_, prec_, pred_, recall_, tp_, tn_, fp_, fn_, auc_, loss_ = \
                sess.run([accuracy, precision, prediction, recall, tp, tn, fp, fn, auc, loss],
                                                          {tf_input: data_test})
            print("| Accuracy: %.4f" % acc_, "| Precision: %.4f" % prec_,
                  "|Recall: %.4f" % recall_, "|TP: %.4f" % tp_, "|TN: %.4f" % tn_,
                  "|FP: %.4f" % fp_, "|FN: %.4f" % fn_, "|AUC: %.4f" % auc_, "| Loss: %.2f" % loss_, )
        if t == 499999:
            acc_per_fold.append(acc_)
            prec_per_fold.append(prec_)
            recall_per_fold.append(recall_)
            loss_per_fold.append(loss_)
            print('---------------------------------')

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Accuracy: {acc_per_fold[i]} - Precision: {acc_per_fold[i]} - Recall: {recall_per_fold[i]} %')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Precision: {np.mean(prec_per_fold)} (+- {np.std(prec_per_fold)})')
print(f'> Recall: {np.mean(recall_per_fold)} (+- {np.std(recall_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')