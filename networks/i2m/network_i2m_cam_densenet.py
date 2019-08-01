import tensorflow as tf

from networks.spectral import spectral_ae

from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D

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

def build_network(next_X, L, A, U, is_train, mesh_embedding_size, cam_embedding_size, 
    J_regressor=None, should_regress_joints=False, std=None, mean_points=None):
    """Build the image-to-mesh network."""
    
    with tf.variable_scope('image_encoder'):
        # Build the image encoder to the mesh and camera embeddings.
        mesh_embedding, camera_embedding = import_image_encoder(next_X, mesh_embedding_size, cam_embedding_size)
        scale, trans, rot = camera_regressor(mesh_embedding, camera_embedding)
         
    output_mesh, restore_saver = import_mesh_decoder(mesh_embedding, L, A, U, is_train)        

    if should_regress_joints:
        keypoints = regress_joints(output_mesh, J_regressor, std, mean_points)
        return output_mesh, mesh_embedding, camera_embedding, scale, trans, rot, restore_saver, keypoints

    return output_mesh, mesh_embedding, camera_embedding, scale, trans, rot, restore_saver

def import_image_encoder(next_X, mesh_embedding_size, cam_embedding_size, name=None):    
    features = DenseNet121(weights='imagenet', include_top=False)(next_X)
    features = GlobalAveragePooling2D()(features)

    features = tf.layers.flatten(features)
    embedding = tf.layers.dense(features, mesh_embedding_size + cam_embedding_size, name=name)
    mesh_embedding, camera_embedding = embedding[:, :mesh_embedding_size], embedding[:, mesh_embedding_size:]
    return mesh_embedding, camera_embedding

def import_mesh_decoder(mesh_embedding, L, A, U, is_train, filters=[16, 32, 32, 48], poly_order=[3, 3, 3, 3], output_dim=3, batch_norm=False):
    """Load the generator."""
    output_mesh = spectral_ae.MeshDecoder(
        mesh_embedding, output_dim, L, A, U, poly_order, filters, is_train, batch_norm=batch_norm)

    reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mesh_decoder')
    reuse_vars_dict = dict([(reuse_vars_map[var.op.name], var) for var in reuse_vars])
    restore_saver = tf.train.Saver(reuse_vars_dict)
    return output_mesh, restore_saver

def camera_regressor(mesh_embedding, camera_embedding):  
    with tf.variable_scope("camera_params"):
        cam_net = tf.nn.relu(tf.layers.dense(camera_embedding, 32))
        cam_net = tf.nn.relu(tf.layers.dense(cam_net, 32))
        cam_net = tf.layers.dense(cam_net, 7)
        
        scale = tf.layers.dense(cam_net, 1, bias_initializer=tf.constant_initializer(90))
        scale = tf.nn.relu(scale)
        trans = tf.layers.dense(cam_net, 3, bias_initializer=tf.constant_initializer(100))
        rot = tf.layers.dense(cam_net, 3)
    return scale, trans, rot

def regress_joints(mesh, J_regressor, std, mean_points):
    with tf.variable_scope("regress_joints"):
        # De-normalize meshes to get the keypoints.
        stopgrad_output_mesh = tf.stop_gradient(mesh)
        output_mesh_rec = stopgrad_output_mesh * std + mean_points

        FINGERTIP_IDXS = tf.constant([2964, 2350, 5170, 4773, 4719])
        keypoints = tf.matmul(J_regressor, output_mesh_rec)
        fingertip_kpts = tf.gather(output_mesh_rec, FINGERTIP_IDXS, axis=1)
        keypoints = tf.concat([keypoints, fingertip_kpts], axis=1)

        keypoints = tf.layers.flatten(keypoints)
        keypoints = tf.layers.dense(keypoints, 63)
        keypoints = tf.reshape(keypoints, [-1, 21, 3])
    return keypoints


