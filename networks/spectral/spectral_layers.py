import scipy
import numpy as np
import io
from scipy import sparse

import tensorflow as tf

"""
The code is taken from:
https://github.com/anuragranj/coma
"""

def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L

class MeshSampling(tf.keras.layers.Layer):

    def poolwT(self, x):
        L = self._gl
        Mp = L.shape[0]
        _, M, Fin = x.get_shape().as_list()
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)

        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, -1])  # M x Fin*N
        x = tf.sparse_tensor_dense_matmul(L, x)  # Mp x Fin*N
        x = tf.reshape(x, [Mp, Fin, -1])  # Mp x Fin x N
        x = tf.transpose(x, perm=[2, 0, 1])  # N x Mp x Fin

        return x

    def __init__(self, graph_laplacians, **kwargs):
        self._gl = graph_laplacians.astype(np.float32)
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.poolwT(x)

class MeshConv(tf.keras.layers.Layer):

    def chebyshev5(self, x, L, Fout, nK):
        L = L.astype(np.float32)
        _, M, Fin = x.get_shape().as_list()
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if nK > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, nK):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [nK, M, Fin, -1])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin*nK])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable
        x = tf.matmul(x, W)  # N*M x Fout
        out = tf.reshape(x, [-1, M, Fout])  # N x M x Fout
        return out

    def __init__(self, graph_laplacians, polynomial_order=6, nf=16, **kwargs):
        self._gl = graph_laplacians
        self._nf = nf
        self._po = polynomial_order
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""

        _, _, n_channels = input_shape.as_list()
        # Create a trainable weight variable for this layer.
        self._weight_variable = self.add_variable(
            name='kernel',
            shape=[n_channels * self._po, self._nf]
        )

    def call(self, x):
        return self.chebyshev5(x, self._gl, self._nf, self._po)