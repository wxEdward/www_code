import numpy as np
import networkx as nx

def _one_hot(list_scalars, one_hot_dim=1):
    features = np.array(list_scalars,dtype=np.int16)
    features = features - min(features)
    features = np.minimum(features,one_hot_dim-1)
    features = np.maximum(features,0)
    one_hot = np.eye(one_hot_dim)[features]
    return one_hot

def degree_fun(g,feature_dim):
    degrees = _one_hot(
        [d for _, d in g.degree()],
        one_hot_dim = feature_dim)
    features = {i:degrees[i] for i in range(len(g.nodes()))}
    nx.set_node_attributes(g,features,'degree')
    return g

def centrality_fun(g,feature_dim):
    features = nx.betweenness_centrality(g)
    nx.set_node_attributes(g,features,'betweenness_centrality')
    return g

def path_len_fun(g,feature_dim):
    path_len = _one_hot(
        [np.mean(list(nx.shortest_path_length(g,
                source=x).values())) for x in list(g.nodes)],
        one_hot_dim=feature_dim)
    features = {i:path_len[i] for i in range(len(g.nodes()))}
    nx.set_node_attributes(g,features,'path_len')
    return g

def pagerank_fun(g,feature_dim):
    features = nx.pagerank(g)
    nx.set_node_attributes(g,features,'pagerank')
    return g

def clustering_coefficient_fun(g,feature_dim):
    features = nx.clustering(g)
    nx.set_node_attributes(g,features,'node_clustering_coefficient')
    return g
    
def identity_fun(g,feature_dim):
    nl = nx.normalized_laplacian_matrix(g)
    i = np.identity(len(list(g.nodes)))
    adj = i - nl#adj = I - nl = D^(-0.5)@A@D^(-0.5)
    diag_all = [np.diag(adj)]
    adj_power = adj
    for i in range(1,feature_dim):
        adj_power = adj_power @ adj
        diag_all.append(np.diag(adj_power))
    diag_all = np.stack(diag_all,axis=1)
    identity = diag_all
    features = {i:identity[i] for i in range(len(g.nodes()))}
    nx.set_node_attributes(g,features,'identity')
    return g

def get_features(g):
    g = degree_fun(g,100)
    g = centrality_fun(g,1)
    g = path_len_fun(g,100)
    g = pagerank_fun(g,1)
    g = clustering_coefficient_fun(g,1)
    g = identity_fun(g,10)
    return g


ba_ori = nx.read_gpickle('ba_ori.gpickle')
ws_ori = nx.read_gpickle('ws_ori.gpickle')

ba_ori_attr = get_features(ba_ori)
ws_ori_attr = get_features(ws_ori)

nx.write_gpickle(ba_ori_attr,'ba_ori_attr')
nx.write_gpickle(ws_ori_attr,'ws_ori_attr')


