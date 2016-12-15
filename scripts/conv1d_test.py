import tensorflow as tf
import math

BATCH_SIZE = 2

INPUT_CHANNELS = 3
INPUT_WIDTH = 1
INPUT_HEIGHT = 10

KERNEL_HEIGHT = 5
KERNEL_WIDTH = 1
STRIDE = 2
NUM_FILTERS = 4

OUTPUT_CHANNELS = NUM_FILTERS
OUTPUT_WIDTH = 1
OUTPUT_HEIGHT = math.floor( ( INPUT_HEIGHT - KERNEL_HEIGHT ) / STRIDE + 1 )


scan = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_HEIGHT, INPUT_CHANNELS])

kernel = tf.truncated_normal([KERNEL_HEIGHT, INPUT_CHANNELS, NUM_FILTERS],
                             stddev=1.0 / math.sqrt(float(KERNEL_HEIGHT*NUM_FILTERS)))

output = tf.nn.conv1d(scan, kernel, stride=STRIDE, padding='VALID')

batch_scans = tf.random_uniform([BATCH_SIZE, INPUT_HEIGHT, INPUT_CHANNELS],
                                maxval=50)

sess = tf.Session()
print(sess.run(output, feed_dict={scan: batch_scans.eval(session=sess)}))
