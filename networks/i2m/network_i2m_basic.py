import tensorflow as tf

import sys
sys.path.append('../spectral')
import spectral_ae

from tensorflow.keras.applications.densenet import DenseNet121

# A map from the variables in the new network to the variables in the mesh decoder.
reuse_vars_map = {
    'mesh_decoder/mesh_leaky_re_l_u1b/kernel' : 'mesh_decoder/mesh_leaky_re_l_u1b_4/kernel',
    'mesh_decoder/mesh_leaky_re_l_u1b_2/kernel' : 'mesh_decoder/mesh_leaky_re_l_u1b_6/kernel',
    'mesh_decoder/dense/bias' : 'mesh_decoder/dense/bias',
    'mesh_decoder/mesh_leaky_re_l_u1b_1/kernel' : 'mesh_decoder/mesh_leaky_re_l_u1b_5/kernel',
    'mesh_decoder/mesh_conv_2/kernel' : 'mesh_decoder/mesh_conv_6/kernel',
    'mesh_decoder/mesh_conv/kernel' : 'mesh_decoder/mesh_conv_4/kernel',
    'mesh_decoder/dense/kernel' : 'mesh_decoder/dense/kernel',
    'mesh_decoder/mesh_conv_1/kernel' : 'mesh_decoder/mesh_conv_5/kernel',
    'mesh_decoder/mesh_conv_3/kernel' : 'mesh_decoder/mesh_conv_7/kernel'
}

def build_network(next_X, L, A, U, is_train, mesh_latent_variable):
    """Build the image-to-mesh network."""
    with tf.variable_scope('image_encoder'):
        # Build the image encoder to the mesh embedding.
        mesh_embedding = import_densenet(next_X, embedding_size=mesh_latent_variable)
    
    output_mesh, restore_saver = import_mesh_decoder(mesh_embedding, L, A, U, is_train)

    return output_mesh, mesh_embedding, restore_saver

def import_densenet(next_X, embedding_size):
    features = DenseNet121(weights='imagenet', include_top=False)(next_X)
    features = tf.layers.flatten(features)
    image_embedding = tf.layers.dense(features, embedding_size)
    return image_embedding

def import_mesh_decoder(mesh_embedding, L, A, U, is_train, filters=[16, 32, 32, 48], poly_order=[3, 3, 3, 3], output_dim=3, batch_norm=False):
    """Load the generator."""
    output_mesh = spectral_ae.MeshDecoder(
        mesh_embedding, output_dim, L, A, U, poly_order, filters, is_train, batch_norm=batch_norm)

    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mesh_decoder')
    reuse_vars_dict = dict([(reuse_vars_map[var.op.name], var) for var in reuse_vars])
    restore_saver = tf.train.Saver(reuse_vars_dict)
    return output_mesh, restore_saver


