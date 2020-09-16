import numpy as np
import pandas as pd
from tensorflow import set_random_seed
import data_preprocessing
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot

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
#data = samples

onehot = pd.get_dummies(data['goodtime'], columns=['l1', 'l2'])
data = data.drop('goodtime', axis=1)

# normalize
data_scaled = data_preprocessing.norm(data)


# perform PCA
#data_scaled = data_preprocessing.pca(data_scaled, 0.9)

# data_scaled = data_scaled.join(onehot, lsuffix='_left', rsuffix='_right')
data_scaled = pd.concat([data_scaled, onehot], axis=1)
data_scaled = data_scaled.values.astype(np.float32)

np.random.shuffle(data_scaled)

assert not np.any(np.isnan(data_scaled))

# train/test separation
sep = int(len(data_scaled) * 0.7)
data_train = data_scaled[:sep]
data_test = data_scaled[sep:]

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
#loss = tf.losses.softmax_cross_entropy(tfy, out)
accuracy = tf.metrics.accuracy(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
precision = tf.metrics.precision(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
recall = tf.metrics.recall(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
fn = tf.metrics.false_negatives(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
fp = tf.metrics.false_positives(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
tp = tf.metrics.true_positives(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
tn = tf.metrics.true_negatives(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]
auc = tf.metrics.auc(labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(prediction, axis=1))[1]

opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# opt = tf.train.AdamOptimizer(learning_rate=0.1)
train_op = opt.minimize(loss)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

if TRAIN is True:

    fig_loss = []
    fig_accuracy = []

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
            fig_loss.append(loss_)
            fig_accuracy.append(acc_)

        if t == 499999:
            saver.save(sess, 'checkpoint_dir/US_4_0_video')
            print('Saving the checkpoints...')
    #loss_accuracy_plot(fig_loss, fig_accuracy)
else:
    #saver = tf.train.import_meta_graph('checkpoint_dir/6_9.meta')
    saver.restore(sess, tf.train.latest_checkpoint('checkpoint_dir'))
    print('Loading the checkpoints...')

    acc_, prec_, pred_, recall_, tp_, tn_, fp_, fn_, auc_,loss_= \
        sess.run([accuracy, precision, prediction, recall, tp, tn, fp, fn, auc, loss],
                 {tf_input: data_test})

    print("| Accuracy: %.4f" % acc_, "| Precision: %.4f" % prec_,
          "|Recall: %.4f" % recall_, "|TP: %.4f" % tp_, "|TN: %.4f" % tn_,
          "|FP: %.4f" % fp_, "|FN: %.4f" % fn_, "|AUC: %.4f" % auc_, "| Loss: %.2f" % loss_, )
