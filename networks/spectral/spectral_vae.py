import numpy as np
import tensorflow as tf

import spectral_layers
import mesh_activations

def MeshEncoder(inputs, embeding, graph_laplacians, downsampling_matrices, polynomial_order, 
                filter_list, is_train, name='mesh_encoder', batch_norm=False, reuse=False, **kwargs):

    with tf.variable_scope(name, reuse=reuse):
        net = inputs
        for nf, poly, nl, nd in zip(filter_list, polynomial_order, graph_laplacians, downsampling_matrices):
            
            net = spectral_layers.MeshConv(nl, nf=nf, polynomial_order=poly, **kwargs)(net)
            if batch_norm:
                net = tf.keras.layers.BatchNormalization()(net, training=is_train)
            net = mesh_activations.MeshLeakyReLU1B()(net)
            
            net = spectral_layers.MeshSampling(nd)(net)

        # Fully connected hidden layers.
        net = tf.layers.flatten(net)
        
        mu = tf.layers.dense(net, embeding)
        sigma = tf.layers.dense(net, embeding)
        epsilon = tf.random_normal(tf.shape(sigma), dtype=tf.float32) 
        net = mu + tf.multiply(epsilon, 0.5*tf.exp(sigma))

    return net, mu, sigma

def MeshDecoder(inputs, out_channel, graph_laplacians, adj_matrices, upsampling_matrices, polynomial_order, 
                filter_list, is_train, name='mesh_decoder', batch_norm=False, reuse=False, **kwargs):

    with tf.variable_scope(name, reuse=reuse):
        pool_size = list(map(lambda x: x.shape[0], adj_matrices))
        net = inputs
        net = tf.layers.Dense(pool_size[-1] * filter_list[-1])(net)
        net = tf.reshape(net, [-1, pool_size[-1], filter_list[-1]])

        for nf, poly, nl, nu in zip(filter_list[::-1][:-1], polynomial_order[::-1][:-1], 
                                    graph_laplacians[-2::-1], upsampling_matrices[::-1]):

            net = spectral_layers.MeshSampling(nu)(net)
            net = spectral_layers.MeshConv(nl, nf=nf, polynomial_order=poly, **kwargs)(net)
            if batch_norm:
                net = tf.keras.layers.BatchNormalization()(net, training=is_train)
            net = mesh_activations.MeshLeakyReLU1B()(net)

        net = spectral_layers.MeshSampling(upsampling_matrices[0])(net)
        net = spectral_layers.MeshConv(graph_laplacians[0], nf=out_channel,
                       polynomial_order=polynomial_order[0], **kwargs)(net)

    return net

def build_model(input_mesh, graph_laplacians, downsampling_matrices, upsampling_matrices, 
                adj_matrices, filters, latent_variable, polynomial_order, lr, is_train, input_dim=3, batch_norm=False):

    mesh_embedding, mu, sigma = MeshEncoder(
        input_mesh, latent_variable, graph_laplacians, 
        downsampling_matrices, polynomial_order, filters, is_train, batch_norm=batch_norm)
    
    output_mesh = MeshDecoder(
        mesh_embedding, input_dim, graph_laplacians, adj_matrices, 
        upsampling_matrices, polynomial_order, filters, is_train, batch_norm=batch_norm)

    return output_mesh, mesh_embedding, mu, sigma