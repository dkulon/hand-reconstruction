import numpy as np
import tensorflow as tf

import os
from os.path import join
import pickle
import time

import sys
sys.path.append('../../../')

from networks.spectral import spectral_ae
from training.model_config import AttrDict
from training.spectral import data_loader
from training.spectral.log import *
from training.spectral.loss import loss_function

# Define the training parameters.
def define_config():
    config = AttrDict()
    
    config.latent_variable = 64
    config.filters = [16, 32, 32, 48]
    config.sampling_steps = len(config.filters)
    config.poly_order = [3] * config.sampling_steps

    config.n_epochs = 300
    config.batch_size = 64
    config.lr = 0.001
    config.l2_reg = 0.00005
    config.z_l2_penalty = 0.0000005

    config.batch_norm = False
    config.residual = False

    config.type = 'sampling_{}'.format(config.sampling_steps)
    config.optimizer = 'AdamW'

    config.info = "data: synth. sampler"
    return config

config = define_config()
latent_variable, sampling_steps, filters, n_epochs = config.latent_variable, config.sampling_steps, config.filters, config.n_epochs
poly_order, batch_size, lr, l2_reg, type, batch_norm = config.poly_order, config.batch_size, config.lr, config.l2_reg, config.type, config.batch_norm
z_l2_penalty = config.z_l2_penalty

# Specify the id of the trained model or set to None if new training.
model_id = None
training_id = model_id if model_id else time.time()

# Define file paths.
ROOT = '../../../'
DATA_DIR = 'data'
DATASET_PATH = join(ROOT, DATA_DIR, 'datasets/mesh-samples', 'data_splits_sampler.pkl')

TENPLATE_DATA_PATH = join(ROOT, DATA_DIR, 'template')
GRAPH_STRUCTURE = join(TENPLATE_DATA_PATH, config.type)
MEAN_PATH = join(TENPLATE_DATA_PATH, 'mean.obj')
OUTPUT_PATH = join(ROOT, DATA_DIR, 'models/spectral-ae')

logdir, ckpts_path = create_logdir(OUTPUT_PATH, training_id, latent_variable, sampling_steps)

# Load spectral operators.
L, A, D, U, p = data_loader.load_spectral_operators(GRAPH_STRUCTURE)

# Load the training data.
train_db_np, val_db_np = data_loader.load_training_data(DATASET_PATH)
TRAIN_SAMPLES, VALIDATION_SAMPLES = len(train_db_np), len(val_db_np)

# Create dataset iterators.
train_db, fp_train, lp_train = data_loader.create_dataset(train_db_np, n_epochs, batch_size, reshuffle=True)
val_db, fp_val, lp_val = data_loader.create_dataset(val_db_np, n_epochs, batch_size, reshuffle=False)

handle, iterator = data_loader.create_feedable_iterator(train_db)
next_X, next_Y = iterator.get_next()

train_db_it = train_db.make_initializable_iterator()
val_db_it = val_db.make_initializable_iterator()

# Build the spectral autoencoder.
is_train = tf.placeholder(tf.bool, name="is_train")
net, mesh_embedding = spectral_ae.build_model(next_X, L, D, U, A, filters, latent_variable, poly_order, lr, is_train, batch_norm=batch_norm)
loss = loss_function(net, next_Y, l2_reg, z_l2_penalty, mesh_embedding)

variable_summaries(mesh_embedding, name='latent_vector_summary')

# Merge all the summaries.
merged = tf.summary.merge_all()

global_step = tf.Variable(0, name='global_step', trainable=False)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    opt = tf.contrib.opt.AdamWOptimizer(learning_rate=lr, weight_decay=0.000001).minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    training_handle = sess.run(train_db_it.string_handle())
    validation_handle = sess.run(val_db_it.string_handle())

    sess.run(init, feed_dict={handle: training_handle})

    sess.run(train_db_it.initializer, feed_dict={fp_train: train_db_np, 
                                                 lp_train: train_db_np})
    sess.run(val_db_it.initializer, feed_dict={fp_val: val_db_np, 
                                               lp_val: val_db_np})
    
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
    
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpts_path + '/'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    train_writer = tf.summary.FileWriter(join(logdir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(join(logdir, 'test'))
    train_writer.add_summary(sess.run(model_summary(config)), 0)

    for epoch in range(n_epochs):

        for batch in range(TRAIN_SAMPLES // batch_size):
            _, train_loss, summary = sess.run([opt, loss, merged], feed_dict={handle: training_handle, is_train: True})
        
        train_writer.add_summary(summary, epoch)
        saver.save(sess, join(ckpts_path, 'mesh_ae.{}'.format(time.time())), global_step=epoch)

        if epoch % 10 == 0:
            for batch in range(max(VALIDATION_SAMPLES // batch_size, 1)):
                val_loss, summary = sess.run([loss, merged], feed_dict={handle: validation_handle, is_train: False})

            test_writer.add_summary(summary, epoch)
            print("Epoch: {:d}, Step: {:8d}, Train loss: {:.5f}, Val loss: {:.5f}".format(epoch, global_step.eval(), train_loss, val_loss))
