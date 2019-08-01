import numpy as np
import tensorflow as tf

from networks import mesh_activations
from networks.spectral import spectral_layers

def MeshEncoder(inputs, embeding, graph_laplacians, downsampling_matrices, polynomial_order, 
                filter_list, is_train, name='mesh_encoder', batch_norm=False, reuse=False, layer_norm=False, activation='lrelu', **kwargs):

    with tf.variable_scope(name, reuse=reuse):
        net = inputs
        for nf, poly, nl, nd in zip(filter_list, polynomial_order, graph_laplacians, downsampling_matrices):
            
            net = spectral_layers.MeshConv(nl, nf=nf, polynomial_order=poly, **kwargs)(net)
            
            if batch_norm: net = tf.keras.layers.BatchNormalization()(net, training=is_train)
            if layer_norm: net = tf.contrib.layers.layer_norm(net)
            if activation == 'elu': net = mesh_activations.MeshELU1B()(net)
            else: net = mesh_activations.MeshLeakyReLU1B()(net)
            
            net = spectral_layers.MeshSampling(nd)(net)

        # Fully connected hidden layers.
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, embeding)

    return net

def MeshDecoder(inputs, out_channel, graph_laplacians, adj_matrices, upsampling_matrices, polynomial_order, 
                filter_list, is_train, name='mesh_decoder', batch_norm=False, reuse=False, layer_norm=False, activation='lrelu', **kwargs):

    with tf.variable_scope(name, reuse=reuse):
        pool_size = list(map(lambda x: x.shape[0], adj_matrices))
        net = inputs
        net = tf.layers.Dense(pool_size[-1] * filter_list[-1])(net)
        net = tf.reshape(net, [-1, pool_size[-1], filter_list[-1]])

        for nf, poly, nl, nu in zip(filter_list[::-1][:-1], polynomial_order[::-1][:-1], 
                                    graph_laplacians[-2::-1], upsampling_matrices[::-1]):

            net = spectral_layers.MeshSampling(nu)(net)
            net = spectral_layers.MeshConv(nl, nf=nf, polynomial_order=poly, **kwargs)(net)
            if batch_norm: net = tf.keras.layers.BatchNormalization()(net, training=is_train)
            if activation == 'elu': net = mesh_activations.MeshELU1B()(net)
            else: net = mesh_activations.MeshLeakyReLU1B()(net)

        net = spectral_layers.MeshSampling(upsampling_matrices[0])(net)
        net = spectral_layers.MeshConv(graph_laplacians[0], nf=out_channel,
                       polynomial_order=polynomial_order[0], **kwargs)(net)

    return net

def build_model(input_mesh, graph_laplacians, downsampling_matrices, upsampling_matrices, 
                adj_matrices, filters, latent_variable, polynomial_order, lr, is_train, input_dim=3, batch_norm=False, layer_norm=False, activation='lrelu'):

    mesh_embedding = MeshEncoder(
        input_mesh, latent_variable, graph_laplacians, 
        downsampling_matrices, polynomial_order, filters, is_train, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation)
    
    output_mesh = MeshDecoder(
        mesh_embedding, input_dim, graph_laplacians, adj_matrices, 
        upsampling_matrices, polynomial_order, filters, is_train, batch_norm=batch_norm, layer_norm=layer_norm, activation=activation)

    return output_mesh, mesh_embedding
