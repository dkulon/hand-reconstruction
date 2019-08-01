import numpy as np
import tensorflow as tf

def loss_function(net, next_Y, l2_reg, z_l2_penalty, mesh_embedding, summary=True):
    l1_loss = tf.losses.absolute_difference(predictions=net, labels=next_Y, reduction=tf.losses.Reduction.MEAN)
    loss = l1_loss

    if l2_reg:
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg, scope=None)
        l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, tf.trainable_variables())
        loss += l2_regularization_penalty

    if z_l2_penalty:
        loss += z_l2_penalty * tf.nn.l2_loss(mesh_embedding)

    if summary: 
        tf.summary.scalar('l1_loss', l1_loss)
        tf.summary.scalar('loss', loss)
    return loss

def vae_loss_function(net, next_Y, l2_reg, mesh_embedding, mu, sigma, beta=1, summary=True):
    l1_loss = tf.losses.absolute_difference(predictions=net, labels=next_Y, reduction=tf.losses.Reduction.MEAN)
    latent_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + sigma - tf.square(mu) - tf.exp(sigma), 1))
    loss = l1_loss + beta * latent_loss

    if l2_reg:
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg, scope=None)
        l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, tf.trainable_variables())
        loss += l2_regularization_penalty

    if summary: 
        tf.summary.scalar('l1_loss', l1_loss)
        tf.summary.scalar('latent_loss', latent_loss)
        tf.summary.scalar('loss', loss)
    return loss