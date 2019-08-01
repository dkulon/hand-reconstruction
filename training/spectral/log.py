import numpy as np
import tensorflow as tf

import pickle
from os.path import join
import time

def create_logdir(base_path, training_id, latent_vector, n_sampling):
    logbase = join(base_path, 'z{}_d{}_{}'.format(latent_vector, n_sampling, training_id))
    logdir = join(logbase, 'summaries')
    ckpts_path = join(logbase, 'models')
    
    if not tf.gfile.Exists(logbase):
        tf.gfile.MakeDirs(logdir)
        tf.gfile.MkDir(ckpts_path)
        tf.gfile.MkDir(join(logdir, 'train'))
        tf.gfile.MkDir(join(logdir, 'test'))

    return logdir, ckpts_path

def model_summary(config, return_array=False):
    hyperparams = np.array([
        "embedding={}".format(config.latent_variable),
        "sampling_steps={}".format(config.sampling_steps),
        "filters={}".format(config.filters),
        "poly_order={}".format(config.poly_order),

        "batch_size={}".format(config.batch_size),
        "learning_rate={}".format(config.lr),
        "l2_regularization={}".format(config.l2_reg),
        "l2_latent_vector={}".format(config.z_l2_penalty),
        "optimizer={}".format(config.optimizer),

        "residual={}".format(config.residual),
        "batch_norm={}".format(config.batch_norm),
        "mesh_type={}".format(config.type),
        config.info])

    if return_array: return hyperparams

    return tf.summary.text("hyperparameters info", tf.constant(hyperparams))

def vae_model_summary(config):
    hyperparams = model_summary(config, return_array=True)
    beta_info = "beta={}".format(config.beta)
    hyperparams = np.insert(hyperparams, 8, beta_info)
    return tf.summary.text("hyperparameters info", tf.constant(hyperparams))

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
