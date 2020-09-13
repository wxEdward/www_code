import networkx as nx
import numpy as np
import random
import scipy
import collections
from scipy import stats
from scipy.stats import entropy

def get_distribution(G):
    nodes_degree = list(dict(G.degree(G.nodes)).values())
    degree_distribution = nodes_degree/np.sum(nodes_degree)
    return degree_distribution

def preferential_add(G,t):
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
        preferential_add(G,t)
    G.add_edge(source,target,timestamp=t)
    return G

def small_world_add(G,t,k=6):
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
        
    G.add_edge(source,target,timestamp=t)
    
    return G

def mat_norm(X):
    return np.sum(np.power(np.absolute(X),2))
    
def deg_count(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    return deg, cnt

def deg_distribution(deg,cnt,size):
    deg = np.flip(list(deg))
    cnt = np.flip(list(cnt))
    cnt_dis = cnt/np.sum(cnt)
    deg_dis = np.zeros(size,dtype=float)
    for i in range(len(deg)):
        deg_dis[deg[i]]= cnt_dis[i]
    deg_dis = deg_dis + np.finfo(np.float32).eps
    return deg_dis

def KL_div(deg1,cnt1,deg2,cnt2):
    size = max(max(deg1),max(deg2))+1
    old_dis = deg_distribution(deg1,cnt1,size)
    new_dis = deg_distribution(deg2,cnt2,size)
    return entropy(old_dis,new_dis)

N = 5000

num_batch = 8
edges_per_batch = 600
p = 0.0001

'''If original graph is BA graph'''
G_1 = nx.generators.random_graphs.barabasi_albert_graph(N,2,100)
G_2 = nx.generators.random_graphs.barabasi_albert_graph(N,2,100)

'''If original graph is small world graph'''
#G_1 = nx.generators.random_graphs.watts_strogatz_graph(N,4,0.3,100)
#G_2 = nx.generators.random_graphs.watts_strogatz_graph(N,4,0.3,100)
nx.set_edge_attributes(G_1,0,'timestamp')
nx.set_edge_attributes(G_2,0,'timestamp')

nx.write_gpickle(G_1,'ba_ori.gpickle')

laplacian_mat_original = nx.laplacian_matrix(G_1)
laplacian_mat_original = laplacian_mat_original.todense()
l_ori = laplacian_mat_original + np.identity(N)*p

adj_original = nx.adjacency_matrix(G_1)
adj_original = adj_original.todense()

norm_laplacian_original = nx.normalized_laplacian_matrix(G_1)
norm_laplacian_original = norm_laplacian_original.todense()
nl_ori = norm_laplacian_original + np.identity(N)*p

deg,cnt = deg_count(G_1)

for batch in range(num_batch):

    for i in range(edges_per_batch):
        G1 = preferential_add(G_1,batch+1)
        G2 = small_world_add(G_2,batch+1)
        #if i%200 == 0 and i!=0:
         #   print('%d edges added'%i)
    edges_added = (batch+1)*edges_per_batch
    print('%d edges added'%edges_added)
    laplacian_mat_1 = nx.laplacian_matrix(G_1)
    laplacian_mat_1 = laplacian_mat_1.todense()
    laplacian_mat_2 = nx.laplacian_matrix(G_2)
    laplacian_mat_2 = laplacian_mat_2.todense()

    adj_mat_1 = nx.adjacency_matrix(G_1)
    adj_mat_1 = adj_mat_1.todense()
    adj_mat_2 = nx.adjacency_matrix(G_2)
    adj_mat_2 = adj_mat_2.todense()

    norm_laplacian_1 = nx.normalized_laplacian_matrix(G_1)
    norm_laplacian_1 = norm_laplacian_1.todense()
    norm_laplacian_2 = nx.normalized_laplacian_matrix(G_2)
    norm_laplacian_2 = norm_laplacian_2.todense()

    '''If S is laplacian matrix'''
    print(' If S is laplacian matrix:')
    
    E_1 = scipy.linalg.solve_sylvester(l_ori,l_ori,laplacian_mat_1-laplacian_mat_original)
    print('Norm(E) & Preferential attachment:', mat_norm(E_1))
   
    E_2 = scipy.linalg.solve_sylvester(l_ori,l_ori,laplacian_mat_2-laplacian_mat_original)
    print('Norm(E) & Small world:',mat_norm(E_2))
    
    print('E_1 over E_2:',mat_norm(E_1)/mat_norm(E_2))

    '''If S is normalized laplacian matrix''' 
    print(' If S is normalized laplacian matrix:')
    
    E_1 = scipy.linalg.solve_sylvester(nl_ori,nl_ori,norm_laplacian_1-norm_laplacian_original)
    print('Norm(E) & Preferential attachment:', mat_norm(E_1))
    
    E_2 = scipy.linalg.solve_sylvester(nl_ori,nl_ori,norm_laplacian_2-norm_laplacian_original)
    print('Norm(E) & Small world:',mat_norm(E_2))
    
    print('E_1 over E_2:',mat_norm(E_1)/mat_norm(E_2))

    deg1, cnt1 = deg_count(G_1)
    deg2, cnt2 = deg_count(G_2)
    print('KL-div under preferential attachment', KL_div(deg,cnt,deg1,cnt1))
    print('KL-div under small world', KL_div(deg,cnt,deg2,cnt2))

nx.write_gpickle(G_1,'ba_ba.gpickle')
nx.write_gpickle(G_2,'ba_ws.gpickle')

