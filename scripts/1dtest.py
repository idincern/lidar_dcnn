import tensorflow as tf
from ops import *

# hyperparameters - general
batch_size = 10
stddev = 0.2
bias_start = 0.0

# random input with size (batch_size, z0_h)
z0_h = 100
z0 = tf.constant(1.0, dtype=tf.float32, shape=[batch_size,z0_h], name='z0')

"""
Generator
"""

# z0 expands linearly to z1 = W*z0+b and is then rehaped to form z2 with
# size (batch_size, z2_h, 1, z2_c). Therefore, the size of z1 is
# (batch_size, z2_h*z2_c).
# Thus, we must determine z2_h and z2_c before defining z1
z2_h = 12
z2_c = 2048
with tf.variable_scope('z1'):
    W1 = tf.get_variable('W', [z0_h, z2_h*z2_c], tf.float32,
                         tf.random_normal_initializer(stddev=stddev))
    b1 = tf.get_variable("b", [z2_h*z2_c],
                         initializer=tf.constant_initializer(bias_start))
    z1 = tf.matmul(z0, W1) + b1

# z2 is our first feature map layer. Its size is (batch_size, z2_h, 1, z2_c).
# The only operation that occurs is a reshaping and a batch norm and relu.
with tf.variable_scope('z2'):
    a2 = tf.reshape(z1,[-1, z2_h, 1, z2_c])
    bn2 = batch_norm(name='bn2')
    z2 = tf.nn.relu(bn2(a2))

# z3 is our first convolution transpose (deconvolution is bad terminology).
z3_h = 23
z3_c = z2_c//2 # 2048//2 = 1024
with tf.variable_scope('z3'):
    W3 = tf.get_variable('W', [5, 1, z3_c, z2_c],
                         initializer=tf.random_normal_initializer(stddev=
                                                                  stddev))
    b3 = tf.get_variable('b',[z3_c], initializer=tf.constant_initializer(0.0))
    a3 = tf.nn.conv2d_transpose(z2, W3,
                                output_shape=[batch_size, z3_h, 1, z3_c],
                                strides=[1, 2, 1, 1]) + b3
    bn3 = batch_norm(name='bn3')
    z3 = lrelu(bn3(a3))

# z4
z4_h = 46
z4_c = z2_c//4 # 2048//4 = 512
with tf.variable_scope('z4'):
    W4 = tf.get_variable('W', [5, 1, z4_c, z3_c],
                         initializer=tf.random_normal_initializer(stddev=
                                                                  stddev))
    b4 = tf.get_variable('b',[z4_c], initializer=tf.constant_initializer(0.0))
    a4 = tf.nn.conv2d_transpose(z3, W4,
                                output_shape=[batch_size, z4_h, 1, z4_c],
                                strides=[1, 2, 1, 1]) + b4
    bn4 = batch_norm(name='bn4')
    z4 = lrelu(bn4(a4))

# z5
z5_h = 91
z5_c = z2_c//8 # 2048//8 = 256
with tf.variable_scope('z5'):
    W5 = tf.get_variable('W', [5, 1, z5_c, z4_c],
                         initializer=tf.random_normal_initializer(stddev=
                                                                  stddev))
    b5 = tf.get_variable('b',[z5_c], initializer=tf.constant_initializer(0.0))
    a5 = tf.nn.conv2d_transpose(z4, W5,
                                output_shape=[batch_size, z5_h, 1, z5_c],
                                strides=[1, 2, 1, 1]) + b5
    bn5 = batch_norm(name='bn5')
    z5 = lrelu(bn5(a5))

# z6 - This is the result of the generator and should be the same shape as an
# incoming batch of scans. No batch_norm on this layer.
z6_h = 181
z6_c = 1
with tf.variable_scope('z6'):
    W6 = tf.get_variable('W', [5, 1, z6_c, z5_c],
                         initializer=tf.random_normal_initializer(stddev=
                                                                  stddev))
    b6 = tf.get_variable('b',[z6_c], initializer=tf.constant_initializer(0.0))
    a6 = tf.nn.conv2d_transpose(z5, W6,
                                output_shape=[batch_size, z6_h, 1, z6_c],
                                strides=[1, 2, 1, 1]) + b6
    z6 = tf.nn.tanh(a6)

"""
Discriminator
"""

# z7 - no batch_norm at this layer.
z7_h = 91
z7_c = z2_c//8 # 2048//8 = 256
with tf.variable_scope('z7'):
    W7 = tf.get_variable('W', [5, 1, z6_c, z7_c],
                         initializer=tf.random_normal_initializer(stddev=
                                                                  stddev))
    b7 = tf.get_variable('b',[z7_c], initializer=tf.constant_initializer(0.0))
    a7 = tf.nn.conv2d(z6, W7, strides=[1, 2, 1, 1], padding='SAME') + b7
    z7 = lrelu(a7)

# z8
z8_h = 46
z8_c = z2_c//4 # 2048//4 = 512
with tf.variable_scope('z8'):
    W8 = tf.get_variable('W', [5, 1, z7_c, z8_c],
                         initializer=tf.random_normal_initializer(stddev=
                                                                  stddev))
    b8 = tf.get_variable('b',[z8_c], initializer=tf.constant_initializer(0.0))
    a8 = tf.nn.conv2d(z7, W8, strides=[1, 2, 1, 1], padding='SAME') + b8
    bn8 = batch_norm(name='bn8')
    z8 = lrelu(bn8(a8))

# z9
z9_h = 23
z9_c = z2_c//2 # 2048//2 = 1024
with tf.variable_scope('z9'):
    W9 = tf.get_variable('W', [5, 1, z8_c, z9_c],
                         initializer=tf.random_normal_initializer(stddev=
                                                                  stddev))
    b9 = tf.get_variable('b',[z9_c], initializer=tf.constant_initializer(0.0))
    a9 = tf.nn.conv2d(z8, W9, strides=[1, 2, 1, 1], padding='SAME') + b9
    bn9 = batch_norm(name='bn9')
    z9 = lrelu(bn9(a9))

# z10
z10_h = 12
z10_c = z2_c # 2048
with tf.variable_scope('z10'):
    W10 = tf.get_variable('W', [5, 1, z9_c, z10_c],
                         initializer=tf.random_normal_initializer(stddev=
                                                                  stddev))
    b10 = tf.get_variable('b',[z10_c],
                          initializer=tf.constant_initializer(0.0))
    a10 = tf.nn.conv2d(z9, W10, strides=[1, 2, 1, 1], padding='SAME') + b10
    bn10 = batch_norm(name='bn10')
    z10 = lrelu(bn10(a10))

# z11 - reshape back up to a big linear
with tf.variable_scope('z11'):
    z11 = tf.reshape(z10, [-1, z10_h*z10_c])

# z12 - reduce to one dimension with linear combination (sigmoid over lrelu and
# no batch norm)
with tf.variable_scope('z12'):
    W12 = tf.get_variable('W', [z10_h*z10_c, 1], tf.float32,
                         tf.random_normal_initializer(stddev=stddev))
    b12 = tf.get_variable("b", [1],
                         initializer=tf.constant_initializer(bias_start))
    a12 = tf.matmul(z11, W12) + b12
    z12 = tf.nn.sigmoid(a12)


sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

DEBUG = True
if DEBUG:
    print('z12')
    print(sess.run(z12))
    print('z12.shape')
    print(z12.shape)
