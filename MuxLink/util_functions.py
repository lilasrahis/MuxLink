from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/../pytorch_DGCNN' % cur_dir)
from util import GNNGraph
import multiprocessing as mp
from itertools import islice

def sample_neg(net, test_ratio=0.1, train_pos=None, test_neg=None,test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train
    row, col, _ = ssp.find(net_triu)
    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num = len(train_pos[0]) if train_pos else 0
    test_num = len(test_pos[0]) if test_pos else 0
    neg = ([], [])
    n = net.shape[0]
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0:
            if test_neg is not None:
                if i in test_neg[0] or i in test_pos[0]:
                    continue

                elif j in test_neg[1] or j in test_pos[1]:
                    continue

                else:

                    print (str(i)+" "+str(j))
                    neg[0].append(i)
                    neg[1].append(j)
            else:
                neg[0].append(i)
                neg[1].append(j)
        else:
            continue
    train_neg  =(neg[0][:train_num], neg[1][:train_num])
    return train_pos, train_neg


def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=1, node_information=None, no_parallel=False):

    min_n_label = {'value': 1000}
    max_n_label = {'value': 0}
    def helper(A, links, g_label):
        g_list = []
        if no_parallel:
            for i, j in tqdm(zip(links[0], links[1])):
                g, n_labels, n_features = subgraph_extraction_labeling(
                    (i, j), A, h, node_information)
                min_n_label['value'] = min(min(n_labels), min_n_label['value'])#EDIT: NODE
                max_n_label['value'] = max(max(n_labels), max_n_label['value'])
                g_list.append(GNNGraph(g, g_label, n_labels, n_features))
            return g_list
        else:
            # the parallel extraction code
            start = time.time()
            pool = mp.Pool(mp.cpu_count())
            results = pool.map_async(
                parallel_worker,
                [((i, j), A, h, node_information) for i, j in zip(links[0], links[1])]
            )
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            g_list = [GNNGraph(g, g_label, n_labels, n_features) for g, n_labels, n_features in results]

            #EDIT: NODE
            min_n_label['value'] = min(
                min([min(n_labels) for _, n_labels, _ in results]), min_n_label['value']
            )
            max_n_label['value'] = max(
                max([max(n_labels) for _, n_labels, _ in results]), max_n_label['value']
            )
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
            return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs, test_graphs = None, None
    if train_pos and train_neg:
        train_graphs = helper(A, train_pos, 1) + helper(A, train_neg, 0)
    if test_pos and test_neg:
        test_graphs = helper(A, test_pos, 1) + helper(A, test_neg, 0)
    elif test_pos:
        test_graphs = helper(A, test_pos, 1)
    return train_graphs, test_graphs, max_n_label['value'],min_n_label['value'] #EDIT: NODE

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)

def subgraph_extraction_labeling(ind, A, h=1,node_information=None ):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :][:, nodes]
    labels = node_label(subgraph)
    # get node features
    features = node_information[nodes]
    # construct nx graph

    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    return g, labels.tolist(), features


def neighbors(fringe, A):
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def node_label(subgraph):
    # The original double-radius node labeling (DRNL), taken from SEAL platform.
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels
