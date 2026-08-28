import tensorflow as tf
import numpy as np

print("TF:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())

a_np = np.random.rand(1000, 1000).astype(np.float32)
b_np = np.random.rand(1000, 1000).astype(np.float32)

with tf.Session() as sess:
    a = tf.constant(a_np)
    b = tf.constant(b_np)
    c = tf.matmul(a, b)
    x = sess.run(c)

print("shape:", x.shape)
print("min:", x.min())
print("max:", x.max())
print("mean:", x.mean())
print("finite:", np.isfinite(x).all())
print("nonzero:", np.count_nonzero(x), "/", x.size)