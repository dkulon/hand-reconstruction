import numpy as np
import tensorflow as tf

import spectral_layers
import mesh_activations

def MeshEncoder(inputs, embeding, graph_laplacians, downsampling_matrices, polynomial_order, 
                filter_list, is_train, batch_norm=False, name='mesh_encoder', reuse=False, **kwargs):

    with tf.variable_scope(name, reuse=reuse):
        net = inputs

        for ii in range(0, len(filter_list), 2):
            nf, poly, nl, nd = filter_list[ii], polynomial_order[ii], graph_laplacians[ii], downsampling_matrices[ii]
            nf2, poly2, nl2, nd2 = filter_list[ii + 1], polynomial_order[ii + 1], graph_laplacians[ii + 1], downsampling_matrices[ii + 1]
            net = ResidualModule(net, nl, nf, poly, nd, nl2, nf2, poly2, nd2, is_train, batch_norm, **kwargs)

    # Fully connected hidden layers.
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, embeding)

    return net

def MeshDecoder(inputs, out_channel, graph_laplacians, adj_matrices, upsampling_matrices, polynomial_order, 
                filter_list, is_train, batch_norm=False, name='mesh_decoder', reuse=False, **kwargs):

    with tf.variable_scope(name, reuse=reuse):
        pool_size = list(map(lambda x: x.shape[0], adj_matrices))
        net = inputs
        net = tf.layers.Dense(pool_size[-1] * filter_list[-1])(net)
        net = tf.reshape(net, [-1, pool_size[-1], filter_list[-1]])

        for ii in range(len(filter_list) - 1, 0, -2):
            nf, poly, nl, nd = filter_list[ii], polynomial_order[ii], graph_laplacians[ii], upsampling_matrices[ii]
            nf2, poly2, nl2, nd2 = filter_list[ii - 1], polynomial_order[ii - 1], graph_laplacians[ii - 1], upsampling_matrices[ii - 1]
            net = ResidualModuleTrans(net, nl, nf, poly, nd, nl2, nf2, poly2, nd2, is_train, batch_norm, **kwargs)

        net = spectral_layers.MeshConv(graph_laplacians[0], nf=out_channel,
           polynomial_order=polynomial_order[0], **kwargs)(net)

    return net

def ResidualModule(net, nl, nf, poly, nd, nl2, nf2, poly2, nd2, is_train, batch_norm, **kwargs):
    out = spectral_layers.MeshConv(nl, nf=nf, polynomial_order=poly, **kwargs)(net)
    if batch_norm: out = tf.layers.BatchNormalization()(out, training=is_train)
    out = mesh_activations.MeshLeakyReLU1B()(out)
    out = spectral_layers.MeshSampling(nd)(out)

    out = spectral_layers.MeshConv(nl2, nf=nf2, polynomial_order=poly2, **kwargs)(out)
    if batch_norm: out = tf.layers.BatchNormalization()(out, training=is_train)
    out = mesh_activations.MeshLeakyReLU1B()(out)
    out = spectral_layers.MeshSampling(nd2)(out)

    residual = spectral_layers.MeshSampling(nd)(net)
    residual = spectral_layers.MeshConv(nl2, nf=nf2, polynomial_order=poly2, **kwargs)(residual)
    if batch_norm: residual = tf.layers.BatchNormalization()(residual, training=is_train)
    residual = spectral_layers.MeshSampling(nd2)(residual)

    net = tf.keras.layers.Add()([residual, out])
    net = mesh_activations.MeshLeakyReLU1B()(net)
    return net

def ResidualModuleTrans(net, nl, nf, poly, nd, nl2, nf2, poly2, nd2, is_train, batch_norm, **kwargs):
    out = spectral_layers.MeshSampling(nd)(net)
    out = spectral_layers.MeshConv(nl, nf=nf, polynomial_order=poly, **kwargs)(out)
    if batch_norm: out = tf.layers.BatchNormalization()(out, training=is_train)
    out = mesh_activations.MeshLeakyReLU1B()(out)

    out = spectral_layers.MeshSampling(nd2)(out)
    out = spectral_layers.MeshConv(nl2, nf=nf2, polynomial_order=poly2, **kwargs)(out)
    if batch_norm: out = tf.layers.BatchNormalization()(out, training=is_train)
    out = mesh_activations.MeshLeakyReLU1B()(out)

    residual = spectral_layers.MeshSampling(nd)(net)
    residual = spectral_layers.MeshSampling(nd2)(residual)
    residual = spectral_layers.MeshConv(nl2, nf=nf2, polynomial_order=poly2, **kwargs)(residual)
    if batch_norm: residual = tf.layers.BatchNormalization()(residual, training=is_train)

    net = tf.keras.layers.Add()([residual, out])
    net = mesh_activations.MeshLeakyReLU1B()(net)
    return net

def build_model(input_mesh, graph_laplacians, downsampling_matrices, upsampling_matrices, 
                adj_matrices, filters, latent_variable, polynomial_order, lr, is_train, input_dim=3, batch_norm=False):

    mesh_embedding = MeshEncoder(
        input_mesh, latent_variable, graph_laplacians, downsampling_matrices, 
        polynomial_order, filters, is_train, batch_norm=batch_norm)
    
    output_mesh = MeshDecoder(
        mesh_embedding, input_dim, graph_laplacians, adj_matrices, upsampling_matrices, 
        polynomial_order, filters, is_train, batch_norm=batch_norm)

    return output_mesh, mesh_embedding
