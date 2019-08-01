import cPickle as pickle
from scipy.spatial import KDTree
import menpo3d.io as m3io
from menpo.shape import TriMesh, PointCloud

import pyface.qt
import matplotlib.pyplot as plt
import numpy as np

import math
import os
import scipy.sparse as sp
from psbody.mesh import Mesh, MeshViewers
from opendr.topology import get_vert_connectivity, get_vertices_per_edge

import sys
sys.path.append('./coma_lib')
import mesh_sampling

def compute_downsampling_transforms(mesh, sampling_factors):
    # A list of downampled meshes.
    down_meshes = [mesh]

    # A list of adjacency matrices for each downsampled mesh.
    adj_matrices = [get_vert_connectivity(mesh.v, mesh.f)]

    # Downsampling and upsampling transforms for each sampling layer.
    down_transforms = []
    up_transforms = []

    for factor in sampling_factors:
        # Compute a downsampled mesh.
        down_faces, down_transform = mesh_sampling.qslim_decimator_transformer(down_meshes[-1], factor=factor)
        down_mesh_v = down_transform.dot(down_meshes[-1].v)
        down_mesh = Mesh(down_mesh_v, down_faces)

        # Append a mesh and its adjacency matrix.
        down_meshes.append(down_mesh)
        adj_matrices.append(get_vert_connectivity(down_mesh_v, down_faces))

        # Append pooling transforms.
        down_transforms.append(down_transform)
        up_transforms.append(mesh_sampling.setup_deformation_transfer(down_meshes[-1], down_meshes[-2]))

    pooling_data = {
        'factors'         :    sampling_factors,
        'down_meshes'     :    down_meshes[1:],
        'adj_matrices'    :    adj_matrices,
        'down_transforms' :    down_transforms,
        'up_transforms'   :    up_transforms
    }

    return pooling_data

def compute_transforms_from_downsampled_meshes(down_meshes, template, sampling_factors=None):
	# A list of adjacency matrices for each downsampled mesh.
	adj_matrices = [get_vert_connectivity(template.v, template.f)]

	# Downsampling and upsampling transforms for each sampling layer.
	down_transforms = []
	up_transforms = []

	for ii in range(1, len(down_meshes)):
	    down_mesh = down_meshes[ii]
	    
	    # Get an adjacency matrix.
	    adj_matrix = get_vert_connectivity(down_mesh.v, down_mesh.f)
	    adj_matrices.append(adj_matrix)
	    
	    # Find correspondence between downsampled meshes.
	    kdtree = KDTree(down_meshes[ii - 1].v)
	    idxs = kdtree.query(down_mesh.v, k=1)[1]

	    # Create a downsampling matrix.
	    down_transform = np.zeros((down_mesh.v.shape[0], down_meshes[ii - 1].v.shape[0]))
	    down_transform[[range(down_mesh.v.shape[0])], idxs.tolist()] = 1
	    down_transform = sp.csr_matrix(down_transform)
	    down_transforms.append(down_transform)
	    
	    # Create an upsampling matrix.
	    upsampling = mesh_sampling.setup_deformation_transfer(down_meshes[ii], down_meshes[ii - 1])
	    up_transforms.append(upsampling)
	    
	pooling_data = {
	    'factors'         :    sampling_factors,
	    'down_meshes'     :    down_meshes[1:],
	    'adj_matrices'    :    adj_matrices,
	    'down_transforms' :    down_transforms,
	    'up_transforms'   :    up_transforms
	}

	return pooling_data

def compute_laplacian(adj_matrices):
	from utils import laplacian
	L = [laplacian(a, normalized=True) for a in adj_matrices]
	return L
