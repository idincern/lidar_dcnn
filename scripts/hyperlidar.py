from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
# Python 2 or Python 3?
import sys
python_version = sys.version_info.major
import os
import tensorflow as tf
import numpy as np
import math
import random # for random.sample()

FLAGS = None

def train():
    # Import data
    # First load all the targets (shape class, size, distance, angle, rotation,
    # and noise level).
    path_to_targs = FLAGS.data_dir + "/targs_struct_s5-10_r15-35.npy"
    targs_raw = np.load(path_to_targs)
    # Load the classes as onehot vectors (shape n -> [0, 0, ..., 1, ..., 0],
    # where the 1 is at the nth index in the vector (starting with 0).
    if python_version == 3:
        yClass = np.eye(26)[list(map(int,targs_raw[:,0].tolist()))]
    else:
        yClass = np.eye(26)[map(int,targs_raw[:,0].tolist())]
    # Then load the size, distance, angle, and rotation. They may need to be
    # scaled.  But, I think this can be done by adjusting their loss's weights
    # in the total loss function.
    ySize = targs_raw[:,1]
    yDist = targs_raw[:,2]
    yAng = targs_raw[:,3]
    yRot = targs_raw[:,4]
    yNoise = targs_raw[:,5]

    # Load the scans
    path_to_scans = FLAGS.data_dir + "/scans_struct_s5-10_r15-35.npy"
    x = np.load(path_to_scans)
    x = np.expand_dims(x, axis=2)
    x = (x-np.mean(x))/np.std(x)

    # separate training data and test data
    test_size = 10000 # reserve this many data points for testing
    # pick test_size indicies and put those into y_test and x_test
    test_ind = random.sample(range(0,x.shape[0]),test_size)
    yClass_test = yClass[test_ind,:]
    ySize_test = ySize[test_ind]
    yDist_test = yDist[test_ind]
    yAng_test = yAng[test_ind]
    yRot_test = yRot[test_ind]
    yNoise_test = yNoise[test_ind]
    x_test = x[test_ind,:]
    # Then remove the test data from the set to make the training data
    yClass_train = np.delete(yClass, test_ind, 0)
    ySize_train = np.delete(ySize, test_ind, 0)
    yDist_train = np.delete(yDist, test_ind, 0)
    yAng_train = np.delete(yAng, test_ind, 0)
    yRot_train = np.delete(yRot, test_ind, 0)
    yNoise_train = np.delete(yNoise, test_ind, 0)
    x_train = np.delete(x, test_ind, 0)

    print('Data loaded and divided into training and testing sets.')

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard vis).
        from https://www.tensorflow.org/how_tos/summaries_and_tensorboard/
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def conv1d_layer(input_tensor, kernel_height, input_height, input_channels,
                     output_height, output_channels, stride, layer_name,
                     act=tf.nn.relu):
        """Reusable code for a 1D convolution layer.
        """
        # Add a name scope to group layers
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the kernel weights
            with tf.name_scope('weights'):
                weights = tf.Variable(
                    tf.truncated_normal(
                        [kernel_height,input_channels,output_channels],
                        stddev=1.0/math.sqrt(float(
                            kernel_height*input_channels))))
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros([1, output_height,
                                               output_channels]))
                variable_summaries(biases)
            with tf.name_scope('preactivations'):
                preactivations = tf.nn.conv1d(
                    input_tensor,weights,stride=stride,padding='VALID')+biases
                tf.summary.histogram('preactivations', preactivations)
            with tf.name_scope('activations'):
                activations = act(preactivations)
                tf.summary.histogram('activations', activations)
            return activations

    def fc_layer(input_tensor, input_dim, output_dim, layer_name,
                 act=tf.nn.relu):
        """Reusable code for a fully connected layer
        """
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = tf.Variable(tf.truncated_normal(
                    [input_dim, output_dim],
                    stddev = 1.0 / math.sqrt(float(input_dim))))
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros([1, output_dim]))
                variable_summaries(biases)
            with tf.name_scope('preactivations'):
                preactivations = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('preactivations',preactivations)
            with tf.name_scope('activations'):
                activations = act(preactivations)
                tf.summary.histogram('activations',activations)
            return activations

    def conv2fc(input_tensor, layer_name):
        """Reusable code to reshape a conv layer to be connected to an fc layer
           First dimension of reshape must change dynamically, since the
           dimension is different during training and testing.
        """
        with tf.name_scope(layer_name):
            reshaped_tensor = tf.reshape(input_tensor,
                                         [tf.shape(input_tensor)[0],-1],
                                         name='reshape')
            return reshaped_tensor

    '''Set hyperparameters of the network
    k_i = height of 1d convolution kernel in layer i
    s_i = stride length in layer i
    n_i = number of feature detectors (kernels) in layer i
    h_i = height of feature map in layer i
    h_0 = height of original data
    h_(i+1) = floor((h_i-k_i)/s_i + 1)
    c_i = number of channels in feature map in layer i
    c_(i+1) = n_i
    '''
    # There are 181 measurements in a scan
    NUM_RANGES = 181
    # There are 26 classes of object (The alphabet)
    NUM_CLASSES = 26
    BATCH_SIZE = 50;

    h = [NUM_RANGES, ];               # conv1d feature map height
    c = (  1, 48, 128, 196, 196, 128) # channels
    k = (  6,  4,   3,   3,   3,   3) # kernel height
    s = (  2,  2,   2,   1,   1,   1) # stride
    for i in range(len(c)-1):         # equation to determine fm heights
        h.append( math.floor( ( h[i] - k[i] ) / s[i] + 1 ) )
    fc = (1024, 1024, NUM_CLASSES)    # fully connected feature map height

    ''' Define input placeholders
    x0 is the input data (i.e. a lidar scan).
    yClass_ is the target (shape class, size, distance, angle, and rotation)
    '''
    with tf.name_scope('input'):
        x0 = tf.placeholder(tf.float32, [None, h[0], c[0]], name='x-input')
        yClass_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='y-class-input')

    ''' Define the inference model
    5 1d convolutional layers and 3 fully connected layers.
    hyperparameters are listed above (h,c,k,s,fc)
    '''
    x1 = conv1d_layer(x0,k[0],h[0],c[0],h[1],c[1],s[0],'conv_layer_0')
    x2 = conv1d_layer(x1,k[1],h[1],c[1],h[2],c[2],s[1],'conv_layer_1')
    x3 = conv1d_layer(x2,k[2],h[2],c[2],h[3],c[3],s[2],'conv_layer_2')
    x4 = conv1d_layer(x3,k[3],h[3],c[3],h[4],c[4],s[3],'conv_layer_3')
    x5 = conv1d_layer(x4,k[4],h[4],c[4],h[5],c[5],s[4],'conv_layer_4')
    x5_reshape = conv2fc(x5, 'reshape_5')
    x6 = fc_layer(x5_reshape,h[5]*c[5],fc[0],'fc_layer_5')
    x7 = fc_layer(x6,fc[0],fc[1],'fc_layer_6')
    yClass = fc_layer(x7,fc[1],fc[2],'fc_layer_7',act=tf.identity)

    ''' Define the loss functions
    For HyperScan, the loss function will be minimized by correctly
    guessing the shape class, size, distance, angle, and rotation of the shape.
    '''
    # Loss for shape class
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(yClass, yClass_)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy',cross_entropy)

    ''' Define the training ops
    Train using the AdamOptimizer, which includes momentum
    https://www.tensorflow.org/api_docs/python/train/optimizers#AdamOptimizer
    '''
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(yClass,1),
                                              tf.argmax(yClass_,1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                  tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    sess = tf.Session()

    # Merger all the summaries and write them out to a file
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

    # Add op to save and restore all the variables.
    saver = tf.train.Saver()

    def feed_dict(train):
        if train:
            # Select new random indicies
            batch_ind = random.sample(range(0,x_train.shape[0]),BATCH_SIZE)
            xs = x_train[batch_ind,:]
            ysClass = yClass_train[batch_ind,:]
            k = FLAGS.dropout
        else:
            xs = x_test
            ysClass = yClass_test
            k = 1.0
        return {x0: xs, yClass_: ysClass}

    # Set the file where the variables will be stored
    checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')

    with sess.as_default():
        tf.global_variables_initializer().run()
        # saver.restore(sess, checkpoint_file)
        for i in range(FLAGS.max_steps):
            if i%10 == 0: # Record summaries adn test-set accuracy
                summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
                test_writer.add_summary(summary,i)
                print('Accuracy at step %s: %s' % (i, acc))
            else: # Record train set summaries, and train
                if i % 100 == 99: # Record execution stats
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict=feed_dict(True),
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for ', i)
                    # Save a checkpoint
                    save_path = saver.save(sess, checkpoint_file,
                                           global_step=i)
                    print("Model saved in file: %s" % save_path)
                else: # Record a summary
                    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                    train_writer.add_summary(summary, i)
        train_writer.close()
        test_writer.close()

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='../logs',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
