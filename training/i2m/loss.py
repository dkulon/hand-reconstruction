import numpy as np
import tensorflow as tf

def loss_function(output_mesh, next_Y, 
                  gt_keypoints, gt_3d_keypoints,
                  scale, trans, rot, 
                  mean_points, std, J_regressor, 
                  l2_reg, mesh_embedding, z_l2_penalty, summary=True, return_kpts=False):

    # L1 loss on meshes.
    l1_loss_mesh = tf.losses.absolute_difference(predictions=output_mesh, labels=next_Y, reduction=tf.losses.Reduction.MEAN)

    # De-normalize meshes to get the keypoints.
    stopgrad_output_mesh = tf.stop_gradient(output_mesh)
    output_mesh_rec = stopgrad_output_mesh * std + mean_points
    joints = regress_joints(output_mesh_rec, J_regressor)
    output_keypoints, output_3d_keypoints = camera_projection(scale, trans, rot, joints)

    # L1 loss on 2D landmarks.
    l1_loss_2d_kpts = tf.losses.absolute_difference(predictions=output_keypoints, labels=gt_keypoints, reduction=tf.losses.Reduction.MEAN)

    # L2 loss on 3D keypoints.
    l1_loss_3d_kpts = tf.losses.absolute_difference(predictions=output_3d_keypoints, labels=gt_3d_keypoints, reduction=tf.losses.Reduction.MEAN)

    # Compute the total loss.
    loss = l1_loss_mesh + 0.01 * l1_loss_3d_kpts

    if l2_reg:
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg, scope=None)
        l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, tf.trainable_variables())
        loss += l2_regularization_penalty

    if z_l2_penalty:
        loss += z_l2_penalty * tf.nn.l2_loss(mesh_embedding)

    if summary: 
        tf.summary.scalar('l1_loss_mesh', l1_loss_mesh)
        tf.summary.scalar('l1_landmarks', l1_loss_2d_kpts)
        tf.summary.scalar('l1_keypoints', l1_loss_3d_kpts)
        tf.summary.scalar('loss', loss)

    if return_kpts: return loss, output_keypoints, output_3d_keypoints
    return loss

def camera_projection(scale, trans, rot, joints):
    with tf.variable_scope("camera_projection"):
        rot_joints = rotate(joints, rot)
        shape = tf.shape(rot_joints)
        output_3d_keypoints = tf.reshape(scale * tf.reshape(rot_joints, [shape[0], -1]), shape) + tf.reshape(trans, [-1, 1, 3])
        output_keypoints = output_3d_keypoints[:, :, :2]
    return output_keypoints, output_3d_keypoints

def regress_joints(mesh, J_regressor):
    with tf.variable_scope("j_regressor"):
        FINGERTIP_IDXS = tf.constant([2964, 2350, 5170, 4773, 4719])
        keypoints = tf.matmul(J_regressor, mesh)
        fingertip_kpts = tf.gather(mesh, FINGERTIP_IDXS, axis=1)
        keypoints = tf.concat([keypoints, fingertip_kpts], axis=1)
    return keypoints

def rotate(joints, theta):
    R = batch_rodrigues(theta, name=None)
    joints_t = tf.linalg.transpose(joints)
    rot_joints = tf.matmul(R, joints_t)
    rot_joints = tf.linalg.transpose(rot_joints)
    return rot_joints

def batch_skew(vec, batch_size=32):
    """
    vec is N x 3, batch_size is int
    returns N x 3 x 3. Skew_sym version of each matrix.
    """
    with tf.name_scope("batch_skew"):
        col_inds = tf.constant([1, 2, 3, 5, 6, 7])
        indices = tf.reshape(
            tf.reshape(tf.range(0, batch_size) * 9, [-1, 1]) + col_inds,
            [-1, 1])
        updates = tf.reshape(
            tf.stack(
                [
                    -vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1],
                    vec[:, 0]
                ],
                axis=1), [-1])
        out_shape = [batch_size * 9]
        res = tf.scatter_nd(indices, updates, out_shape)
        res = tf.reshape(res, [batch_size, 3, 3])

    return res


def batch_rodrigues(theta, name=None, batch_size=32):
    """
    Theta is N x 3
    """
    with tf.name_scope(name, "batch_rodrigues"):
        angle = tf.expand_dims(tf.norm(theta + 1e-8, axis=1), -1)
        r = tf.expand_dims(tf.div(theta, angle), -1)

        angle = tf.expand_dims(angle, -1)
        cos = tf.cos(angle)
        sin = tf.sin(angle)

        outer = tf.matmul(r, r, transpose_b=True, name="outer")

        eyes = tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])
        R = cos * eyes + (1 - cos) * outer + sin * batch_skew(
            r, batch_size=batch_size)
    return R
