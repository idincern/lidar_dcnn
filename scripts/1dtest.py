import tensorflow as tf

# hyperparameters - general
batch_size = 10
stddev = 0.2
bias_start = 0.0

# random input with size (batch_size, z0_w)
z0_w = 100
z0 = tf.constant(1.0, dtype=tf.float32, shape=[1,z0_w], name='z0')

# hyperparameters - generator
# z0 expands linearly to z1 = W*z0+b and is then rehaped to form z2 with
# size (batch_size, z2_w, 1, z2_c). Therefore, the size of z1 is
# (batch_size, z2_w*z2_c).
# Thus, we must determine z2_w and z2_c before defining z1
z2_w = 4
z2_c = 2048
with tf.variable_scope('z1'):
    W1 = tf.get_variable('W', [z0_w, z2_w*z2_c], tf.float32,
                         tf.random_normal_initializer(stddev=stddev))
    b1 = tf.get_variable("b", [z2_w*z2_c],
                         initializer=tf.constant_initializer(bias_start))
    a1 = tf.matmul(z0, W1) + b1
    z1 = tf.nn.relu(a1)

#with tf.variable_scope('z2'):

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
print(sess.run(z1))
