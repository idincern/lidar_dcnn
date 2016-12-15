import tensorflow as tf

# There are 26 classes of object
NUM_CLASSES = 26

# k_i = length of 1d convolution kernel in layer i
# s_i = stride length in layer i
# n_i = number of feature detectors (kernels) in layer i
# h_i = height of feature map in layer i
# h_0 = height of original data
# h_(i+1) = (h_i-k_i)/s_i + 1
# d_i = depth of feature map in layer i
# d_(i+1) = n_i
def inference(scans, k,s,n):
