import tensorflow as tf

import os
from os.path import join
import time
import numpy as np
import scipy.sparse

import sys
sys.path.append('../../../')

from networks.i2m import network_i2m_cam_densenet as i2m_net
from networks.spectral import spectral_ae

from training.i2m import data_loader as i2m_dl
from training.i2m.log import *
from training.i2m.loss import loss_function
from training.spectral import data_loader as spectral_dl
from training.model_config import AttrDict

# Define the training parameters.
def define_config():
    config = AttrDict()

    config.n_epochs = 300
    config.batch_size = 32
    config.lr = 0.00001
    config.l2_reg = 0.00001
    config.z_l2_penalty = 0.00005
    config.cam_latent_variable = 32
    config.optimizer = 'AdamW'
    
    config.info = ""
    
    # Generator specification (fixed).
    config.mesh_latent_variable = 64
    config.filters = [16, 32, 32, 48]
    config.sampling_steps = len(config.filters)
    config.poly_order = [3] * config.sampling_steps
    config.type = 'sampling_{}'.format(config.sampling_steps)
    
    config.num_parallel_calls = 8

    return config

config = define_config()
num_parallel_calls = config.num_parallel_calls
mesh_latent_variable, cam_latent_variable, n_epochs = config.mesh_latent_variable, config.cam_latent_variable, config.n_epochs
batch_size, lr, l2_reg, z_l2_penalty = config.batch_size, config.lr, config.l2_reg, config.z_l2_penalty

# Specify the id of the trained model or set to None if new training.
model_id = None
init_epoch = 0
training_id = model_id if model_id else time.time()

# The id of the pre-trianed mesh decoder.
spectral_model_id = 'z64_d4_1550943246.5350132'

# Define file paths.
ROOT = '../../../'
DATA_DIR = 'data'
DATA_PATH = join(ROOT, DATA_DIR)
DATASET_PATH = join(ROOT, DATA_DIR, 'datasets/hand-panoptic')
# Spectral operators.
TENPLATE_DATA_PATH = join(ROOT, DATA_DIR, 'template')
GRAPH_STRUCTURE = join(TENPLATE_DATA_PATH, config.type)
MEAN_PATH = join(TENPLATE_DATA_PATH, 'mean.obj')
# Saved decoder.
GENERATOR_OUTPUT_PATH = join(ROOT, DATA_DIR, 'models/spectral-ae', spectral_model_id)
GENERATOR_CHECKPOINT_PATH = join(GENERATOR_OUTPUT_PATH, 'models')
# Training logs.
OUTPUT_PATH = join(ROOT, DATA_DIR, 'models/i2m', str(training_id))
CHECKPOINT_PATH = join(OUTPUT_PATH, 'models')

logdir, ckpts_path = create_logdir(OUTPUT_PATH, '{}'.format(training_id))

# Load spectral operators.
L, A, D, U, p = spectral_dl.load_spectral_operators(GRAPH_STRUCTURE)
# Load the joint regressor.
J_regressor = np.array(np.load(join(TENPLATE_DATA_PATH, 'J_regressor.npy')), dtype=np.float32)

mean_points = np.load(join(TENPLATE_DATA_PATH, 'mean_points.npy'))
std = np.load(join(TENPLATE_DATA_PATH, 'std_points.npy'))

tf.reset_default_graph()

# Create dataset iterators.
TRAIN_SAMPLES, VALIDATION_SAMPLES = 2 * 20180, 2 * 1500

train_filenames = [join(DATASET_PATH, 'hand_domedb_train_r.tfrecord'),
                   join(DATASET_PATH, 'hand_domedb_train_lm.tfrecord')]
train_db = i2m_dl.create_dataset(train_filenames, n_epochs=n_epochs, batch_size=batch_size,
                                    mean_points=mean_points, std=std, num_parallel_calls=num_parallel_calls, reshuffle=True)

val_filenames = [join(DATASET_PATH, 'hand_domedb_val_r.tfrecord'),
                 join(DATASET_PATH, 'hand_domedb_val_lm.tfrecord')]
val_db = i2m_dl.create_dataset(val_filenames, n_epochs=n_epochs, batch_size=batch_size,
                                mean_points=mean_points, std=std, num_parallel_calls=num_parallel_calls, reshuffle=False)

handle, iterator = i2m_dl.create_feedable_iterator(train_db)
next_X, next_Y_mesh, next_Y_lms, next_Y_3d_kpts = iterator.get_next()

train_db_it = train_db.make_initializable_iterator()
val_db_it = val_db.make_initializable_iterator()

J_regressor = tf.tile(tf.constant(np.expand_dims(J_regressor, 0)), [batch_size, 1, 1])
is_train = tf.placeholder(tf.bool, name="is_train")

mean_points = tf.expand_dims(mean_points, axis=0)
std = tf.expand_dims(std, axis=0)

output_mesh, mesh_embedding, camera_embedding, scale, trans, rot, restore_saver = i2m_net.build_network(
                                                                next_X, L, A, U,
                                                                is_train, mesh_latent_variable, cam_latent_variable)

loss = loss_function(output_mesh, next_Y_mesh, 
                     next_Y_lms, next_Y_3d_kpts, 
                     scale, trans, rot,
                     mean_points, std, J_regressor,
                     l2_reg, mesh_embedding, z_l2_penalty, summary=True)

train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='image_encoder')
all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

# Merge all the summaries.
merged = tf.summary.merge_all()

global_step = tf.Variable(0, name='global_step', trainable=False)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    init_opt = tf.contrib.opt.AdamWOptimizer(learning_rate=lr, weight_decay=0.000001)
    # Freeze the generator.
    opt = init_opt.minimize(loss, global_step=global_step, var_list=train_vars)

sess_config = tf.ConfigProto()
sess_config.gpu_options.per_process_gpu_memory_fraction = 1.

init = tf.global_variables_initializer()

with tf.Session(config=sess_config) as sess:
    
    training_handle = sess.run(train_db_it.string_handle())
    validation_handle = sess.run(val_db_it.string_handle())

    sess.run(init, feed_dict={handle: training_handle})
    
    # gen_ckpt = tf.train.get_checkpoint_state(os.path.dirname(GENERATOR_CHECKPOINT_PATH + '/'))
    # restore_saver.restore(sess, gen_ckpt.model_checkpoint_path)
    restore_saver.restore(sess, join(GENERATOR_CHECKPOINT_PATH, 'mesh_ae.1550965497.0982995-599'))

    sess.run(train_db_it.initializer)
    sess.run(val_db_it.initializer)
    
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
    
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpts_path + '/'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    
    train_writer = tf.summary.FileWriter(join(logdir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(join(logdir, 'test'))
    train_writer.add_summary(sess.run(model_summary(config)), init_epoch)
    
    for epoch in range(init_epoch, n_epochs):

        for batch in range(TRAIN_SAMPLES // batch_size):
            _, train_loss, summary = sess.run([opt, loss, merged], feed_dict={handle: training_handle, is_train: True})
             
        train_writer.add_summary(summary, epoch)
        saver.save(sess, join(ckpts_path, 'i2m.panoptic.{}'.format(time.time())), global_step=epoch)
        
        if epoch % 5 == 0:
            for batch in range(VALIDATION_SAMPLES // batch_size):
                val_loss, summary = sess.run([loss, merged], feed_dict={handle: validation_handle, is_train: False})

            test_writer.add_summary(summary, epoch)
            print("Epoch: {:d}, Step: {:8d}, Train loss: {:.5f}, Val loss: {:.5f}".format(epoch, global_step.eval(), train_loss, val_loss))
