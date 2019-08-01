from os.path import join
import numpy as np
import pickle

import tensorflow as tf

def load_spectral_operators(mesh_data_path):
    """Load the spectral operator data."""

    # Load the pooling data.
    pooling_data = pickle.load(open(join(mesh_data_path, 'operators', 'transforms.p'), "rb" ), encoding='latin1')
    D = pooling_data['down_transforms']
    U = pooling_data['up_transforms']

    # Load the spectral operator data.
    spectral_data = pickle.load(open(join(mesh_data_path, 'operators', 'spectral_LA.p'), "rb" ), encoding='latin1')
    L = spectral_data['laplacian']
    A = spectral_data['adjacency']

    # Cast the data to the correct format.
    L = list(map(lambda x: x.astype('float32'), L))
    A = list(map(lambda x: x.astype('float32'), A))
    D = list(map(lambda x: x.astype('float32'), D))
    U = list(map(lambda x: x.astype('float32'), U))
    p = list(map(lambda x: x.shape[0], A))
    return L, A, D, U, p

def load_training_data(dataset_path):
    """Load training and validation data."""

    # Load the data.
    data_splits = pickle.load(open(dataset_path, "rb"), encoding="latin1")
    train_db_np, val_db_np = np.array(data_splits['train'], dtype=np.float32), np.array(data_splits['val'], dtype=np.float32)

    mean_points = np.mean(train_db_np, axis=0)
    std_points = np.std(train_db_np, axis=0)

    # Normalize vertices.
    train_db_np = (train_db_np - mean_points) / std_points
    val_db_np = (val_db_np - mean_points) / std_points

    return train_db_np, val_db_np

def load_test_data(dataset_path):
    """Load test data and return statistics."""

    # Load the data.
    data_splits = pickle.load(open(dataset_path, "rb"), encoding="latin1")
    train_db_np, test_db_np = np.array(data_splits['train'], dtype=np.float32), np.array(data_splits['test'], dtype=np.float32)

    mean_points = np.mean(train_db_np, axis=0)
    std_points = np.std(train_db_np, axis=0)

    # Normalize vertices.
    test_db_np = (test_db_np - mean_points) / std_points

    return test_db_np, mean_points, std_points

def create_dataset(data, n_epochs, batch_size, reshuffle=False):
    features_placeholder = tf.placeholder(data.dtype, data.shape)
    labels_placeholder = tf.placeholder(data.dtype, data.shape)

    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    dataset = dataset.shuffle(buffer_size=512, reshuffle_each_iteration=reshuffle)
    dataset = dataset.repeat(n_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset, features_placeholder, labels_placeholder

def create_feedable_iterator(dataset):
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, 
                                dataset.output_types, dataset.output_shapes)
    return handle, iterator