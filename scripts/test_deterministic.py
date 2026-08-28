import tensorflow as tf
import numpy as np

a_np = np.ones((1000, 1000), dtype=np.float32)
b_np = np.ones((1000, 1000), dtype=np.float32)

with tf.Session() as sess:
    x = sess.run(tf.matmul(tf.constant(a_np), tf.constant(b_np)))

print("min:", x.min())
print("max:", x.max())
print("mean:", x.mean())
print("unique:", np.unique(x)[:20])