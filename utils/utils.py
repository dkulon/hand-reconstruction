import numpy as np

def export_obj(verts, trilist, filepath):
    with open(filepath, 'w') as fp:
        for v_i in verts:
            fp.write('v %f %f %f\n' % (v_i[0], v_i[1], v_i[2]))

        for f in trilist:
            fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12.

    https://github.com/mattloper/opendr/blob/master/topology.py
    """
    import scipy.sparse as sp

    vpv = sp.csc_matrix((len(mesh_v), len(mesh_v)))

    def row(A):
        return A.reshape((1, -1))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv

def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix.

    Source: COMA by Ranjan et al.
    """

    import scipy.sparse

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L
