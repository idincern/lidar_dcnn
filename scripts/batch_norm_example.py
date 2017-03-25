from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))

    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

# Size of each batch
batch_size = 64

def generator(z):
    # feature map heigh and width (i.e. size)
    h0_s = 4
    h1_s = 7
    h2_s = 14
    h3_s = 28
    # channels (number of feature maps) for discriminator
    h0_c = 512
    h1_c = 256
    h2_c = 128
    h3_c = 1
    with tf.variable_scope('generator'):
        # Create batch norm objects, used below
        g_bn0 = batch_norm(name='g_bn0')
        g_bn1 = batch_norm(name='g_bn1')
        g_bn2 = batch_norm(name='g_bn2')
        # project
        z_, h0_w, h0_b = linear(z,h0_c*h0_s*h0_s, 'g_h0_lin', with_w=True)
        # reshape
        h0 = tf.nn.relu(g_bn0(tf.reshape(z_, [-1, h0_s, h0_s, h0_c])))
        # conv transpose 1
        w1 = weight_variable([5,5,h1_c,h0_c])
        h1 = lrelu(g_bn1(tf.nn.conv2d_transpose(
            h0,w1,output_shape=[batch_size,h1_s,h1_s,h1_c],strides=[1,2,2,1])))
        # conv transpose 2
        w2 = weight_variable([5,5,h2_c,h1_c])
        h2 = lrelu(g_bn2(tf.nn.conv2d_transpose(
            h1,w2,output_shape=[batch_size,h2_s,h2_s,h2_c],strides=[1,2,2,1])))
        # conv transpose 3 (no batch norm on last layer)
        w3 = weight_variable([5,5,h3_c,h2_c])
        h3 = lrelu(tf.nn.conv2d_transpose(
            h2,w3,output_shape=[batch_size,h3_s,h3_s,h3_c],strides=[1,2,2,1]))
        return tf.nn.tanh(h3)

def discriminator(image, reuse=False):
    # channels (number of feature maps) for discriminator
    h0_c = 128
    h1_c = 256
    h2_c = 512
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        # Create batch norm objects, used below
        d_bn1 = batch_norm(name='d_bn1')
        d_bn2 = batch_norm(name='d_bn2')
        d_bn3 = batch_norm(name='d_bn3')
        h0 = lrelu(conv2d(image, h0_c, name='d_h0_conv'))
        h1 = lrelu(d_bn1(conv2d(h0, h1_c, name='d_h1_conv')))
        h2 = lrelu(d_bn2(conv2d(h1, h2_c, name='d_h2_conv')))
        h3 = linear(tf.reshape(h2, [batch_size, -1]), 1, 'd_h3_lin')
        return tf.nn.sigmoid(h3), h3

# Image dimensions (height, width, channels)
image_dims = [28, 28, 1]
# Create a placeholder for the real images
inputs = tf.placeholder(tf.float32, [batch_size] + image_dims,
                        name='real_images')

# Create a placeholder for the random input to the generator
z = tf.placeholder(tf.float32, [batch_size, 100], name='z')

# Create generator op
G = generator(z)
# And create discriminator ops that include real images...
D, D_logits = discriminator(inputs)
# ... and fake (generated) images. reuse=True reuses variables.
D_, D_logits_ = discriminator(G, reuse=True)

# Define loss operations
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(D), logits=D_logits))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(D_), logits=D_logits_))
d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(D_), logits=D_logits_))

t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

epoch = 25
learning_rate = 0.0002
beta1 = 0.5

d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                  .minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                  .minimize(g_loss, var_list=g_vars)

"""
# Create a placeholder for the real images
x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])


for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
"""
