import tensorflow as tf
import numpy as np

print("TensorFlow:", tf.__version__)
print("GPU:", tf.test.gpu_device_name())
print("build:", tf.sysconfig.get_build_info() if hasattr(tf.sysconfig, "get_build_info") else "N/A")

a_np = np.ones((1000, 1000), dtype=np.float32)
b_np = np.ones((1000, 1000), dtype=np.float32)

with tf.Session() as sess:
    a = tf.constant(a_np)
    b = tf.constant(b_np)
    c = tf.matmul(a, b)
    x = sess.run(c)

print("matmul:")
print("  shape:", x.shape)
print("  min:", x.min())
print("  max:", x.max())
print("  mean:", x.mean())
print("  finite:", np.isfinite(x).all())