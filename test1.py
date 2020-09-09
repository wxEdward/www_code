import networkx as nx
import numpy as np
import random
import scipy
import collections
from scipy import stats
from scipy.stats import entropy


N = 2000
G_1 = nx.generators.random_graphs.barabasi_albert_graph(N,5,100)
G_2 = nx.generators.random_graphs.barabasi_albert_graph(N,5,100)


def get_distribution(G):
    nodes_degree = list(dict(G.degree(G.nodes)).values())
    degree_distribution = nodes_degree/np.sum(nodes_degree)
    return degree_distribution

def preferential_add(G):
    nodes_list = list(G.nodes)
    num_nodes = len(nodes_list)
    distribution = get_distribution(G)
    source = np.random.choice(nodes_list)
    rand = random.random()
    target = 0
    for i in range(num_nodes):
        if np.sum(distribution[:i]) < rand and rand <= np.sum(distribution[:i+1]) and i != source:
            target = i
            break
    if (source,target) in G.edges:
        preferential_add(G)
    G.add_edge(source,target)
    return G

def small_world_add(G,k=3):
    nodes_list = list(G.nodes)
    
    if k%2 == 0:
        k = k/2
    else: k = (k-1)/2
        
    left_dst=0
    right_dst=nodes_list[-1]
    
    source = np.random.choice(nodes_list)
    if source>=k:
        left_dst = source-k
    if source<=nodes_list[-1]-k:
        right_dst = source+k
        
    choices = list(np.arange(left_dst,right_dst+1))
    choices.remove(source)
    target = np.random.choice(choices)
    
    while (source,target) in G.edges:
        left_dst=0
        right_dst=nodes_list[-1]
        source = np.random.choice(nodes_list)
        if source>=k:
            left_dst = source-k
        if source<=nodes_list[-1]-k:
            right_dst = source+k
        
        choices = list(np.arange(left_dst,right_dst+1))
        choices.remove(source)
        target = np.random.choice(choices)
        
    G.add_edge(source,target)
    
    return G
    
def matrix_sign(old_l, new_l, N, p = 0.0001):

    I = np.identity(N)
    C = old_l-new_l
    A = old_l+p*I
    H = np.zeros((2*N,2*N))
    
    H[:N,:N] = A
    H[:N,N:] = C
    H[N:,N:] = -A
    
    sign_H = H @ (scipy.linalg.sqrtm(H @ H) @ np.linalg.inv(H))
    
    return sign_H

def compute_X(sign_of_H):
    tmp = 1/2 * (sign_of_H + np.identity(2*N))
    X = tmp[:N,N:]
    return X

def compute_E(old_l,new_l,N,p=0.0001):
    sign_H = matrix_sign(old_l,new_l,N,p=0.0001)
    X = compute_X(sign_H)
    return X

def norm_X(X):
    return np.sum(np.power(np.absolute(X),2))

def deg_distribution(deg,cnt,size):
    deg = np.flip(list(deg))
    cnt = np.flip(list(cnt))
    cnt_dis = cnt/np.sum(cnt)
    deg_dis = np.zeros(size,dtype=float)
    for i in range(len(deg)):
        deg_dis[deg[i]]= cnt_dis[i]
    return deg_dis

def KL_div(deg1,cnt1,deg2,cnt2):
    size = max(max(deg1),max(deg2))+1
    old_dis = deg_distribution(deg1,cnt1,size)
    new_dis = deg_distribution(deg2,cnt2,size)
    return entropy(old_dis,new_dis)


laplacian_mat_original = nx.laplacian_matrix(G_1)
laplacian_mat_original = laplacian_mat_original.todense()

adj_original = nx.adjacency_matrix(G_1)
adj_original = adj_original.todense()


num_edges = 1000

for i in range(num_edges):
    G1 = preferential_add(G_1)
    G2 = small_world_add(G_2,3)
    
laplacian_mat_1 = nx.laplacian_matrix(G_1)
laplacian_mat_1 = laplacian_mat_1.todense()
laplacian_mat_2 = nx.laplacian_matrix(G_2)
laplacian_mat_2 = laplacian_mat_2.todense()

adj_mat_1 = nx.adjacency_matrix(G_1)
adj_mat_1 = adj_mat_1.todense()
adj_mat_2 = nx.adjacency_matrix(G_2)
adj_mat_2 = adj_mat_2.todense()


E_1 = compute_E(laplacian_mat_original,laplacian_mat_1,N)
E_2 = compute_E(laplacian_mat_original,laplacian_mat_2,N)


print(norm_X(E_1),norm_X(E_2))

np.sum(np.power(laplacian_mat_1-laplacian_mat_original,2))

np.sum(np.power(laplacian_mat_2-laplacian_mat_original,2))

np.sum(np.power(adj_mat_1-adj_original,2))

np.sum(np.power(adj_mat_2-adj_original,2))




