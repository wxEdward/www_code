#!/usr/bin/env python
# coding: utf-8

# In[2]:


import networkx as nx
import numpy as np
import scipy
from numpy import linalg as LA


# In[20]:


G_ori_1 = nx.read_gpickle('ws_ori.gpickle')
G_ori_2 = nx.read_gpickle('ws_ori.gpickle')
G_ba = nx.read_gpickle('ws_ba.gpickle')
G_ws = nx.read_gpickle('ws_ws.gpickle')

p = 0.0001
N = len(G_ori_1.nodes)
I = np.identity(N)*p

l_ori_1 = nx.laplacian_matrix(G_ori_1) + I
nl_ori_1 = nx.normalized_laplacian_matrix(G_ori_1) + I

l_ori_2 = nx.laplacian_matrix(G_ori_2) + I
nl_ori_2 = nx.normalized_laplacian_matrix(G_ori_2) + I


# In[21]:


ba_added = [i for i in G_ba.edges if i not in G_ori_1.edges]
ws_added = [i for i in G_ws.edges if i not in G_ori_2.edges]

ba_attr = nx.get_edge_attributes(G_ba,'timestamp')
ws_attr = nx.get_edge_attributes(G_ws,'timestamp')
batches = list(set(ba_attr.values()))


# In[22]:


l_ba_Es = []
nl_ba_Es = []
l_ws_Es = []
nl_ws_Es = []


# In[ ]:


for batch in batches[1:]:
    ba_edges = [edge for edge in ba_added if ba_attr[edge] == batch]
    ws_edges = [edge for edge in ws_added if ws_attr[edge] == batch]
    
    '''if added like ba'''
    G_ori_1.add_edges_from(ba_edges)
    l_ba = nx.laplacian_matrix(G_ori_1) + I
    nl_ba = nx.laplacian_matrix(G_ori_1) + I
    
    E_l_ba = scipy.linalg.solve_sylvester(l_ori_1,l_ori_1,l_ba-l_ori_1)
    E_nl_ba = scipy.linalg.solve_sylvester(nl_ori_1,nl_ori_1,nl_ba-nl_ori_1)
    
    l_ba_Es.append(E_l_ba)
    nl_ba_Es.append(E_nl_ba)
    
    '''if added like ws'''
    G_ori_2.add_edges_from(ws_edges)
    l_ws = nx.laplacian_matrix(G_ori_2) + I
    nl_ws = nx.laplacian_matrix(G_ori_2) + I
    
    E_l_ws = scipy.linalg.solve_sylvester(l_ori_2,l_ori_2,l_ws-l_ori_2)
    E_nl_ws = scipy.linalg.solve_sylvester(nl_ori_2,nl_ori_2,nl_ws-nl_ori_2)
    
    l_ws_Es.append(E_l_ws)
    nl_ws_Es.append(E_nl_ws)


# In[ ]:


np.save('ws_E/l_ba_Es',l_ba_Es)
np.save('ws_E/nl_ba_Es',nl_ba_Es)
np.save('ws_E/l_ws_Es',l_ws_Es)
np.save('ws_E/nl_ws_Es',nl_ws_Es)


# In[4]:


nE_ba_l = [min(LA.norm(E/LA.norm(E)-np.identity(N)),LA.norm(E/LA.norm(E)+np.identity(N))) for E in l_ba_Es]
nE_ba_nl = [min(LA.norm(E/LA.norm(E)-np.identity(N)),LA.norm(E/LA.norm(E)+np.identity(N))) for E in nl_ba_Es]

nE_ws_l = [min(LA.norm(E/LA.norm(E)-np.identity(N)),LA.norm(E/LA.norm(E)+np.identity(N))) for E in l_ws_Es]
nE_ws_nl = [min(LA.norm(E/LA.norm(E)-np.identity(N)),LA.norm(E/LA.norm(E)+np.identity(N))) for E in nl_ws_Es]


# In[ ]:


print('BA + L: ',nE_ba_l)
print('BA + NL: ',nE_ba_nl)
print('WS + L: ',nE_ws_l)
print('WS + NL: ',nE_ws_nl)


# In[ ]:


np.save('ws_E/nE_ba_l',nE_ba_l)
np.save('ws_E/nE_ba_nl',nE_ba_nl)
np.save('ws_E/nE_ws_l',nE_ws_l)
np.save('ws_E/nE_ws_nl',nE_ws_nl)


# In[ ]:




