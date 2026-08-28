import tensorflow as tf
import numpy as np

np.random.seed(123)

x_np = np.random.rand(1, 224, 224, 3).astype(np.float32)
w_np = np.random.rand(3, 3, 3, 64).astype(np.float32)

with tf.Session() as sess:
    x = tf.constant(x_np)
    w = tf.constant(w_np)
    y = tf.nn.conv2d(
        x, w,
        strides=[1, 1, 1, 1],
        padding="SAME"
    )
    z = sess.run(y)
print("shape:", x_np.shape)
print("min:", x_np.min())
print("max:", x_np.max())
print("mean:", x_np.mean())
print("std:", x_np.std())
print("finite:", np.isfinite(x_np).all())
print("nonzero:", np.count_nonzero(x_np), "/", x_np.size)

print("shape:", w_np.shape)
print("min:", w_np.min())
print("max:", w_np.max())
print("mean:", w_np.mean())
print("std:", w_np.std())
print("finite:", np.isfinite(w_np).all())
print("nonzero:", np.count_nonzero(w_np), "/", w_np.size)

print("shape:", z.shape)
print("min:", z.min())
print("max:", z.max())
print("mean:", z.mean())
print("std:", z.std())
print("finite:", np.isfinite(z).all())
print("nonzero:", np.count_nonzero(z), "/", z.size)
