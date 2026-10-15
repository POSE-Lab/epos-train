import tensorflow as tf

tfrecord = "/home/store/tf_data/essential2_train.tfrecord"

dataset = tf.data.TFRecordDataset([tfrecord])

iterator = dataset.make_one_shot_iterator()
next_record = iterator.get_next()

with tf.Session() as sess:
    for i in range(5):
        serialized = sess.run(next_record)

        example = tf.train.Example()
        example.ParseFromString(serialized)

        f = example.features.feature

        path = f["image/path"].bytes_list.value[0].decode()
        width = f["image/width"].int64_list.value[0]
        height = f["image/height"].int64_list.value[0]

        print(f"{i}: {path}")
        print(f"    TFRecord size: {width} x {height}")
