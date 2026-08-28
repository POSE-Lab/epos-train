import tensorflow as tf
import numpy as np

print(tf.__version__)

x = tf.constant(np.random.rand(1,224,224,3), dtype=tf.float32)
w = tf.constant(np.random.rand(3,3,3,64), dtype=tf.float32)

y = tf.nn.conv2d(x, w, [1,1,1,1], "SAME")

with tf.Session() as sess:
    print(sess.run(y).shape)