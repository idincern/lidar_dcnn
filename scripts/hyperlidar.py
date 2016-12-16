import tensorflow as tf
import numpy as np
import math

import random # for random.sample()

# Python 2 or Python 3?
import sys
python_version = sys.version_info.major

# There are 181 measurements in a scan
NUM_RANGES = 181
# There are 26 classes of object
NUM_CLASSES = 26
BATCH_SIZE = 50;

# k_i = height of 1d convolution kernel in layer i
# s_i = stride length in layer i
# n_i = number of feature detectors (kernels) in layer i
# h_i = height of feature map in layer i
# h_0 = height of original data
# h_(i+1) = floor((h_i-k_i)/s_i + 1)
# c_i = number of channels in feature map in layer i
# c_(i+1) = n_i

h = [NUM_RANGES, ];
c = (  1, 48, 128, 196, 196, 128)
k = (  6,  4,   3,   3,   3,   3)
s = (  2,  2,   2,   1,   1,   1)
for i in range(len(c)-1):
    h.append( math.floor( ( h[i] - k[i] ) / s[i] + 1 ) )
fc = (1024, 1024, NUM_CLASSES)

with tf.name_scope('conv_layer_0'):
    f0 = tf.placeholder(tf.float32, [BATCH_SIZE, h[0], c[0]])
    k0 = tf.Variable(tf.truncated_normal([k[0], c[0], c[1]],
                                          stddev = 1.0 /
                                          math.sqrt(float(k[0]*c[1]))),
                     name="kernel0")
    b0 = tf.Variable(tf.zeros([1, h[1], c[1]]), name='biases0')
    # add Leaky ReLU and Batch Normalization
    f1 = tf.nn.relu(tf.nn.conv1d(f0,k0,stride=s[0],padding='VALID') + b0)

with tf.name_scope('conv_layer_1'):
    k1 = tf.Variable(tf.truncated_normal([k[1], c[1], c[2]],
                                          stddev = 1.0 /
                                          math.sqrt(float(k[1]*c[2]))),
                     name="kernel1")
    b1 = tf.Variable(tf.zeros([1, h[2], c[2]]), name='biases1')
    # add Leaky ReLU and Batch Normalization
    f2 = tf.nn.relu(tf.nn.conv1d(f1,k1,stride=s[1],padding='VALID') + b1)

with tf.name_scope('conv_layer_2'):
    k2 = tf.Variable(tf.truncated_normal([k[2], c[2], c[3]],
                                          stddev = 1.0 /
                                          math.sqrt(float(k[2]*c[3]))),
                     name="kernel2")
    b2 = tf.Variable(tf.zeros([1, h[3], c[3]]), name='biases2')
    # add Leaky ReLU and Batch Normalization
    f3 = tf.nn.relu(tf.nn.conv1d(f2,k2,stride=s[2],padding='VALID') + b2)

with tf.name_scope('conv_layer_3'):
    k3 = tf.Variable(tf.truncated_normal([k[3], c[3], c[4]],
                                          stddev = 1.0 /
                                          math.sqrt(float(k[3]*c[4]))),
                     name="kernel3")
    b3 = tf.Variable(tf.zeros([1, h[4], c[4]]), name='biases3')
    # add Leaky ReLU and Batch Normalization
    f4 = tf.nn.relu(tf.nn.conv1d(f3,k3,stride=s[3],padding='VALID') + b3)

with tf.name_scope('conv_layer_4'):
    k4 = tf.Variable(tf.truncated_normal([k[4], c[4], c[5]],
                                          stddev = 1.0 /
                                          math.sqrt(float(k[4]*c[5]))),
                     name="kernel4")
    b4 = tf.Variable(tf.zeros([1, h[5], c[5]]), name='biases4')
    # add Leaky ReLU and Batch Normalization
    f5 = tf.nn.relu(tf.nn.conv1d(f4,k4,stride=s[4],padding='VALID') + b4)

with tf.name_scope('fc_layer_5'):
    reshape = tf.reshape(f5, [BATCH_SIZE, -1])
    w5 = tf.Variable(tf.truncated_normal([c[5]*h[5], fc[0]],
                                         stddev = 1.0 /
                                         math.sqrt(float(c[5]*h[5]))),
                     name='weights5')
    b5 = tf.Variable(tf.zeros([1, 1024]), name='biases5')
    # add Leaky ReLU and Batch Normalization
    f6 = tf.nn.relu(tf.matmul(reshape,w5) + b5)

with tf.name_scope('fc_layer_6'):
    w6 = tf.Variable(tf.truncated_normal([fc[0], fc[1]],
                                         stddev = 1.0 /
                                         math.sqrt(float(fc[0]))),
                     name='weights6')
    b6 = tf.Variable(tf.zeros([1, 1024]), name='biases6')
    # add Leaky ReLU and Batch Normalization
    f7 = tf.nn.relu(tf.matmul(f6,w6) + b6)

with tf.name_scope('fc_layer_7'):
    w7 = tf.Variable(tf.truncated_normal([fc[1],fc[2]],
                                         stddev = 1.0 /
                                         math.sqrt(float(fc[1]))),
                     name='weights7')
    b7 = tf.Variable(tf.zeros([1, 26]), name='biases7')
    # add Leaky ReLU and Batch Normalization
    f8 = tf.matmul(f7,w7) + b7

targets = tf.placeholder(tf.float32, [BATCH_SIZE, 26])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(f8,targets))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Load the shapes as onehot vectors (shape n -> [0, 0, ..., 1, ..., 0], where
# the 1 is at the nth index in the vector (starting with 0).
path_to_targs = "../world/targs_struct_s5-10_r15-35.npy"
if python_version == 3:
    shape_onehot = np.eye(26)[list(map(int,np.load(path_to_targs)[:,0].tolist()))]
else:
    shape_onehot = np.eye(26)[map(int,np.load(path_to_targs)[:,0].tolist())]

# Load the scans
path_to_scans = "../world/scans_struct_s5-10_r15-35.npy"
scans = np.load(path_to_scans)
scans = np.expand_dims(scans, axis=2)

# separate training data and validation data
val_size = 10000 # reserve this many data points for validation
val_ind = random.sample(range(0,scans.shape[0]),val_size)
shape_onehot_val = shape_onehot[val_ind,:]
scans_val = scans[val_ind,:]
shape_onehot_train = np.delete(shape_onehot, val_ind, 0)
scans_train = np.delete(scans, val_ind, 0)

sess = tf.Session()
with sess.as_default():
    tf.global_variables_initializer().run()

    for i in range(10000):
        # Select new random indicies
        batch_ind = random.sample(range(0,scans_train.shape[0]),BATCH_SIZE)

        output = sess.run(train_step, feed_dict={f0: scans[batch_ind,:], targets: shape_onehot[batch_ind,:]})
    print(output)

    correct = tf.equal(tf.argmax(tf.nn.softmax(f8),1), tf.argmax(targets,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    print(sess.run(correct, feed_dict={f0: scans_val[0:BATCH_SIZE,:], targets: shape_onehot_val[0:BATCH_SIZE,:]}))

    print(sess.run(accuracy, feed_dict={f0: scans_val[0:BATCH_SIZE,:], targets: shape_onehot_val[0:BATCH_SIZE,:]}))
