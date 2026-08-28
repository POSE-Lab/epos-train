import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

x = tf.constant(np.random.rand(1,224,224,3), tf.float32)
w = tf.constant(np.random.rand(3,3,3,64), tf.float32)

y = tf.nn.conv2d(x, w, [1,1,1,1], "SAME")

with tf.Session(config=config) as sess:
    print(sess.run(y).shape)