import numpy as np
import scipy.sparse as sp
import networkx as nx
from collections import defaultdict
from itertools import chain


def generate_corrupted(num_nodes, num_features, num_clusters, alpha=2.0, ratio=0.9,
                       p_adj=0.1, p_att=0.1, p_both=0.01,
                       rel_max_deg_corrupted=0.1, max_tries=10, seed=0):
    """
    Generate an attributed graph with corruptions:
        - Generate the graph structure according to the Configuration model
        - Generate Attributes according to a Binomial Mixutre model

    :param num_nodes: Number of nodes to generate
    :param num_features: Number of features
    :param num_clusters: Number of clusters
    :param alpha: Parameter realted to the power-law exponent of the degree sequence
    :param ratio: Ratio of intra-cluster edges

    :param p_adj: Probability to generate a corrupted node in the graph space
    :param p_att: Probability to generate a corrupted node in the attribute space
    :param p_both: Probability to generate a corrupted node in the both spaces

    :param rel_max_deg_corrupted: Maximum degree for corrputed nodes relative to total number of nodes
    :param max_tries: Maximum number of tries to generated a valid degree sequence
    :param seed: Random seed

    :return:
        :adj: Sparse adjacency matrix
        :att: Sparse attribute matrix
        :cluster_ind: A vector indicating the cluster for each node
        :corrupt_adj: A vector indicating whether the node is corrupted in the graph space
        :corrupt_att: A vector indicating whether the node is corrupted in the attribute space
    """
    np.random.seed(seed)
    random_ind = np.random.permutation(num_nodes)

    num_adj, num_att, num_both = int(p_adj * num_nodes), int(p_att * num_nodes), int(p_both * num_nodes)
    ind_adj = random_ind[:num_adj]
    ind_att = random_ind[num_adj:num_adj + num_att]
    ind_both = random_ind[num_adj + num_att:num_adj + num_att + num_both]

    # indicator vector for nodes that have a corrupted adjacency matrix
    corrupt_adj = np.zeros(num_nodes, dtype=np.int)
    corrupt_adj[ind_adj] = 1
    corrupt_adj[ind_both] = 1

    # indicator vector for nodes that have a corrupted attribute matrix
    corrupt_att = np.zeros(num_nodes, dtype=np.int)
    corrupt_att[ind_att] = 1
    corrupt_att[ind_both] = 1

    # generated the cluster indicator where the cluster sizes are drawn from a dirichlet prior
    pi = np.random.dirichlet(alpha=10 * np.ones(num_clusters))
    cluster_ind = np.argmax(np.random.multinomial(1, pi, size=num_nodes), axis=1)

    # generate a degree sequence deg with a given power-law degree
    # if alpha=None the degrees are constant (0.01*num_nodes)
    if alpha is None:
        deg = int(0.01 * num_nodes)*np.ones(num_nodes, np.int32)
    else:
        for _ in range(max_tries):
            deg = np.ceil(np.random.pareto(alpha - 1, num_nodes)).astype('int')
            if nx.is_valid_degree_sequence_erdos_gallai(deg):
                break

    assert nx.is_valid_degree_sequence_erdos_gallai(deg)

    edges = defaultdict(list)
    # add the good nodes randomly proportional to ratio
    for k in range(num_clusters):
        q_idx = np.where(np.logical_and(corrupt_adj == 0, cluster_ind == k))[0]
        # repeat the node_id so that it matches the prescribed degree
        q_repeats = np.repeat(q_idx, deg[q_idx])
        np.random.shuffle(q_repeats)

        # split the edges into intra-cluster and inter-cluster edges
        split_point = int(np.ceil((len(q_repeats) * ratio) / 2.) * 2)
        intra, inter = q_repeats[:split_point], q_repeats[split_point:]

        # split every edge into two end-points (left and right)
        left, right = np.array_split(intra, 2)
        edges[(k, k, 'left')] += left.tolist()
        edges[(k, k, 'right')] += right.tolist()

        inter = np.array_split(inter, num_clusters-1)
        for split, l in zip(inter, (l for l in range(num_clusters) if l != k)):
            if k < l:
                edges[(k, l, 'left')] += split.tolist()
            else:
                edges[(l, k, 'right')] += split.tolist()

    edge_list = (zip(edges[(k, l, 'left')], edges[(k, l, 'right')])
                 for k in range(num_clusters+1)
                 for l in range(num_clusters+1) if k <= l)

    graph = nx.from_edgelist(chain.from_iterable(edge_list))
    graph.add_edges_from((ca_idx, other_idx)
                         for ca_idx in np.where(corrupt_adj)[0]
                         for other_idx in np.random.permutation(num_nodes)[:np.random.randint(
                             1, num_nodes*rel_max_deg_corrupted)])
    # remove self-loops
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # select the larget connected component
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = nx.subgraph(graph, largest_cc)
    largest_cc = graph.nodes()
    cluster_ind = cluster_ind[largest_cc]
    corrupt_adj = corrupt_adj[largest_cc]
    corrupt_att = corrupt_att[largest_cc]

    # generate the adjacency matrix
    adj = nx.to_scipy_sparse_matrix(graph)

    # generate the topics and the attribute matrix
    topics = np.random.beta(0.1, 0.1, (num_features, num_clusters))
    att = np.random.binomial(1, topics[:, cluster_ind].T)

    # corrupt the attributes (assume 0 or 1 is equally likely so prob=0.5)
    att[corrupt_att == 1] = np.random.binomial(1, 0.5, (corrupt_att.sum(), num_features))
    att = sp.csr_matrix(att)

    return adj, att, cluster_ind, corrupt_adj, corrupt_att
