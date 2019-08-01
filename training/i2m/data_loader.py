import numpy
import tensorflow as tf

# Create a description of the features.  
feature_description = {
    'image': tf.FixedLenFeature([], tf.string, default_value=''),
    'kpts_3d': tf.FixedLenFeature([], tf.string, default_value=''),
    'kpts_2d': tf.FixedLenFeature([], tf.string, default_value=''),
    'points': tf.FixedLenFeature([], tf.string, default_value=''),

    'is_left': tf.FixedLenFeature([], tf.int64, default_value=-1),
    'sample_index': tf.FixedLenFeature([], tf.int64, default_value=-1),

    'height': tf.FixedLenFeature([], tf.int64, default_value=224),
    'width': tf.FixedLenFeature([], tf.int64, default_value=224),
    'points_len': tf.FixedLenFeature([], tf.int64, default_value=7907),
    'joints_len': tf.FixedLenFeature([], tf.int64, default_value=21)
}

def _parse_function(example_proto, mean_points, std, return_init_img=False, lms_only=False):
    # Parse the input tf.Example proto using the dictionary above.
    parsed_record = tf.parse_single_example(example_proto, feature_description)
    # Image.
    init_img = tf.reshape(tf.decode_raw(parsed_record['image'], tf.uint8), (224, 224, 3))
    init_img = tf.cast(init_img, dtype=tf.float32)
    img = tf.image.per_image_standardization(init_img)
    # Mesh vertices.
    points = tf.reshape(tf.decode_raw(parsed_record['points'], tf.float32), (7907, 3))
    points = (points - mean_points) / std
    # Keypoints aligned with images.
    kpts = tf.reshape(tf.decode_raw(parsed_record['kpts_2d'], tf.float64), (21, 3))
    kpts_2d = tf.cast(kpts[:, :2], dtype=tf.float32)
    # Keypoints in the camera frame.
    kpts_3d = tf.cast(kpts, dtype=tf.float32)
    if return_init_img: return init_img, points, kpts_2d, kpts_3d
    return img, points, kpts_2d, kpts_3d

def create_dataset(filenames, n_epochs, batch_size, mean_points, std, num_parallel_calls, return_init_img=False, shuffle=True, seed=None, reshuffle=False):
    dataset = tf.data.TFRecordDataset(filenames)

    # Load anad preprocess the data.
    if shuffle:
        dataset = dataset.shuffle(buffer_size=40000, reshuffle_each_iteration=reshuffle)
    dataset = dataset.repeat(n_epochs)
    dataset = dataset.apply(tf.data.experimental.map_and_batch((lambda x : _parse_function(x, mean_points, std, return_init_img=return_init_img)), 
                                                                            batch_size, 
                                                                            num_parallel_calls=num_parallel_calls))
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def create_feedable_iterator(dataset):
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, 
                                dataset.output_types, dataset.output_shapes)
    return handle, iterator