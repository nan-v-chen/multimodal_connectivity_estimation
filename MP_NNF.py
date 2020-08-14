#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:Axe Chen
import logging
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

# use logging module for easy debug
logging.basicConfig(format='%(asctime)s %(levelname)8s: %(message)s', datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


def threshold(matrix, cost, binary=False, check_tri=True, interpolation='midpoint', normalize=False, mst=False, test_matrix=True):
    """
    Threshold a numpy matrix to obtain a certain "cost".
    Parameters
    ----------
        matrix: a numpy matrix
        cost: the proportion of edges. e.g., a cost of 0.1 has 10 percent
        of all possible edges in the graph
        binary: False, convert weighted values to 1
        check_tri: True, ensure that the matrix contains upper and low triangles.
        if it does not, the cost calculation changes.
        interpolation: midpoint, the interpolation method to pass to np.percentile
        normalize: False, make all edges sum to 1. Convienient for comparisons across subjects,
        as this ensures the same sum of weights and number of edges are equal across subjects
        mst: False, calculate the maximum spanning tree, which is the strongest set of edges that
        keep the graph connected. This is convienient for ensuring no nodes become disconnected.
    Returns
    -------
        matrix: thresholded matrix
    """
    matrix[np.isnan(matrix)] = 0.0
    matrix[matrix < 0.0] = 0.0
    np.fill_diagonal(matrix, 0.0)
    c_cost_int = 100 - (cost * 100)
    if check_tri:
        if np.sum(np.triu(matrix)) == 0.0 or np.sum(np.tril(matrix)) == 0.0:
            c_cost_int = 100. - ((cost / 2.) * 100.)
    if c_cost_int > 0:
        if not mst:
            matrix[matrix < np.percentile(matrix, c_cost_int, interpolation=interpolation)] = 0.
        else:
            if test_matrix:
                t_m = matrix.copy()
            assert (np.tril(matrix, -1) == np.triu(matrix, 1).transpose()).all()
            matrix = np.tril(matrix, -1)
            mst = minimum_spanning_tree(matrix * -1) * -1
            mst = mst.toarray()
            mst = mst.transpose() + mst
            matrix = matrix.transpose() + matrix
            if test_matrix:
                assert (matrix == t_m).all() == True
            matrix[(matrix < np.percentile(matrix, c_cost_int, interpolation=interpolation)) & (mst == 0.0)] = 0.
    if binary:
        matrix[matrix > 0] = 1
    if normalize:
        matrix = matrix / np.sum(matrix)
    return matrix


def MP_NNF(A, k=10, c=0.7, max_iter=200):
    """
    Fuse multi-modal networks.
    @author Axe Nan Chen
    Parameters
    ----------
        A: initial adjacent matrices, numpy array shaped like (number of modalities, number of regions, number of regions)
        k: the number of neighbors of local structure
        c: sparsity
        max_iter: the max number of iterations
    Returns
    -------
        D*: a symmetric matrix, the unified multi-modal network
    """
    S = np.ones((A.shape[0], A.shape[1], A.shape[2]))
    D = np.ones((A.shape[0], A.shape[1], A.shape[2]))
    for v in range(A.shape[0]):
        A[v] = threshold(A[v], cost=c)
        A[v] = (A[v] - np.min(A[v])) / (np.max(A[v]) - np.min(A[v]))
        D[v] = A[v].copy()
        for i in range(A.shape[1]):
            Ni = np.argsort(-A[v, i])[:k]
            for j in range(A.shape[1]):
                if j in Ni:
                    S[v, i, j] = A[v, i, j] / np.sum(A[v, i, Ni])
                else:
                    S[v, i, j] = 0
    iter_count = 0
    D_sum = np.sum(D, axis=0)
    D_new = np.ones((A.shape[0], A.shape[1], A.shape[2]))
    Dc = np.zeros((A.shape[1], A.shape[2]))
    Dc_old = np.zeros((A.shape[1], A.shape[2]))
    while iter_count < max_iter:
        for v in range(A.shape[0]):
            D_new[v] = S[v] @ ((D_sum-D[v])/(A.shape[0]-1)) @ S[v].T
        for v in range(A.shape[0]):
            D_new[v] = threshold(D_new[v], cost=c)
            D[v] = (D_new[v] - np.min(D_new[v])) / (np.max(D_new[v]) - np.min(D_new[v]))
        D_sum = np.sum(D, axis=0)
        Dc = D_sum / A.shape[0]
        if np.max(np.abs(Dc-Dc_old)) < 1e-4:
            # print("converge in ", iter_count)
            return (Dc+Dc.T)/2
        Dc_old = Dc.copy()
        iter_count += 1
    return (Dc+Dc.T)/2