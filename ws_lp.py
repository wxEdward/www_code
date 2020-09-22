import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
import networkx as nx
# Load Pytorch as backend
dgl.load_backend('pytorch')
import numpy as np
from dgl.nn.pytorch import conv as dgl_conv

node_attr = ['degree','betweenness_centrality','path_len','pagerank','node_clustering_coefficient','identity']
edge_attr = ['timestamp']

def load_ws():
    g = nx.read_gpickle('ws_ori_attr.gpickle')
    g = dgl.from_networkx(g1,node_attr)
    features = g.ndata['degree']
    for i in node_attr:
        if i != 'degree':
            features = th.cat((features,g.ndata[i].view(5000,-1)),1)
    g.ndata['features'] = features
    return g

def load_ba():
    g = nx.read_gpickle('ba_ori_attr.gpickle')
    g = dgl.from_networkx(g1,node_attr)
    features = g.ndata['degree']
    for i in node_attr:
        if i != 'degree':
            features = th.cat((features,g.ndata[i].view(5000,-1)),1)
    g.ndata['features'] = features
    return g

def ws_ba_add(g,t):
    G_ba = nx.read_gpickle('ws_ba.gpickle')
    ba_attr = nx.get_edge_attributes(G_ba,'timestamp')
    ba_edges = [edge for edge in G_ba.edges if ba_attr[edge] == t]
    u = [x for (x,y) in ba_edges]
    v = [y for (x,y) in ba_edges]
    g.add_edges(u,v)
    return g

def ws_ws_add(g,t):
    G_ws = nx.read_gpickle('ws_ws.gpickle')
    ws_attr = nx.get_edge_attributes(G_ws,'timestamp')
    ws_edges = [edge for edge in G_ws.edges if ws_attr[edge] == t]
    u = [x for (x,y) in ws_edges]
    v = [y for (x,y) in ws_edges]
    g.add_edges(u,v)
    return g
    
def ba_ws_add(g,t):
    G_ws = nx.read_gpickle('ba_ws.gpickle')
    ws_attr = nx.get_edge_attributes(G_ws,'timestamp')
    ws_edges = [edge for edge in G_ws.edges if ws_attr[edge] == t]
    u = [x for (x,y) in ws_edges]
    v = [y for (x,y) in ws_edges]
    g.add_edges(u,v)
    return g
    
def ba_ba_add(g,t):
    G_ba = nx.read_gpickle('ba_ba.gpickle')
    ba_attr = nx.get_edge_attributes(G_ba,'timestamp')
    ba_edges = [edge for edge in G_ba.edges if ba_attr[edge] == t]
    u = [x for (x,y) in ba_edges]
    v = [y for (x,y) in ba_edges]
    g.add_edges(u,v)
    return g

    
class GraphSAGEModel(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGEModel, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(dgl_conv.SAGEConv(in_feats, n_hidden, aggregator_type,
                                         feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(dgl_conv.SAGEConv(n_hidden, n_hidden, aggregator_type,
                                             feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(dgl_conv.SAGEConv(n_hidden, out_dim, aggregator_type,
                                         feat_drop=dropout, activation=None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h
# NCE loss
def NCE_loss(pos_score, neg_score, neg_sample_size):
    pos_score = F.logsigmoid(pos_score)
    neg_score = F.logsigmoid(-neg_score).reshape(-1, neg_sample_size)
    return -pos_score - torch.sum(neg_score, dim=1)

class LinkPrediction(nn.Module):
    def __init__(self, gconv_model):
        super(LinkPrediction, self).__init__()
        self.gconv_model = gconv_model

    def forward(self, g, features, neg_sample_size):
        emb = self.gconv_model(g, features)
        pos_g, neg_g = edge_sampler(g, neg_sample_size, return_false_neg=False)
        pos_score = score_func(pos_g, emb)
        neg_score = score_func(neg_g, emb)
        return torch.mean(NCE_loss(pos_score, neg_score, neg_sample_size))

def edge_sampler(g, neg_sample_size, edges=None, return_false_neg=True):
    sampler = dgl.contrib.sampling.EdgeSampler(g, batch_size=int(g.number_of_edges()/10),
                                               seed_edges=edges,
                                               neg_sample_size=neg_sample_size,
                                               negative_mode='tail',
                                               shuffle=True,
                                               return_false_neg=return_false_neg)
    sampler = iter(sampler)
    return next(sampler)

def score_func(g, emb):
    src_nid, dst_nid = g.all_edges(order='eid')
    # Get the node Ids in the parent graph.
    src_nid = g.parent_nid[src_nid]
    dst_nid = g.parent_nid[dst_nid]
    # Read the node embeddings of the source nodes and destination nodes.
    pos_heads = emb[src_nid]
    pos_tails = emb[dst_nid]
    return torch.sum(pos_heads * pos_tails, dim=1)
    
def LPEvaluate(gconv_model, g, features, eval_eids, neg_sample_size):
    gconv_model.eval()
    with torch.no_grad():
        emb = gconv_model(g, features)
        
        pos_g, neg_g = edge_sampler(g, neg_sample_size, eval_eids, return_false_neg=True)
        pos_score = score_func(pos_g, emb)
        neg_score = score_func(neg_g, emb).reshape(-1, neg_sample_size)
        filter_bias = neg_g.edata['false_neg'].reshape(-1, neg_sample_size)

        pos_score = F.logsigmoid(pos_score)
        neg_score = F.logsigmoid(neg_score)
        neg_score -= filter_bias.float()
        pos_score = pos_score.unsqueeze(1)
        rankings = torch.sum(neg_score >= pos_score, dim=1) + 1
        return np.mean(1.0/rankings.cpu().numpy())


g = load_ws()
dgl.save_graphs('./ws.bin',g)
features = g.ndata['features']
in_feats = g.ndata['features'].shape[1]

#Model hyperparameters
n_hidden = in_feats
n_layers = 1
dropout = 0.5
aggregator_type = 'gcn'

# create GraphSAGE model
gconv_model = GraphSAGEModel(in_feats,
                             n_hidden,
                             n_hidden,
                             n_layers,
                             F.relu,
                             dropout,
                             aggregator_type)

eids = np.random.permutation(g.number_of_edges())
train_eids = eids[:int(len(eids) * 0.8)]
valid_eids = eids[int(len(eids) * 0.8):int(len(eids) * 0.9)]
test_eids = eids[int(len(eids) * 0.9):]
train_g = g.edge_subgraph(train_eids, preserve_nodes=True)

# Model for link prediction
model = LinkPrediction(gconv_model)

# Training hyperparameters
weight_decay = 5e-4
n_epochs = 30
lr = 2e-3
neg_sample_size = 100

# use optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# initialize graph
dur = []
for epoch in range(n_epochs):
    model.train()
    loss = model(train_g, features, neg_sample_size)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = LPEvaluate(gconv_model, g, features, valid_eids, neg_sample_size)
    print("Epoch {:05d} | Loss {:.4f} | MRR {:.4f}".format(epoch, loss.item(), acc))

print()
# Let's save the trained node embeddings.
acc = LPEvaluate(gconv_model, g, features, test_eids, neg_sample_size)
print("Test MRR {:.4f}".format(acc))

torch.save(gconv_model,'gconv.pt')
torch.save(model,'model.pt')


num_new_batch_nodes = 8
for t in range(1,num_new_batch_nodes):
    g = ws_ba_add(g,t)
    eids = np.random.permutation(g.number_of_edges())
    valid_eids = eids[int(len(eids) * 0.8):int(len(eids) * 0.9)]
    acc = LPEvaluate(gconv_model, g, features, valid_eids, neg_sample_size)
    print("New edges batch{:05d) | MRR {:.4f}".format(t,acc))

g1 = dgl.load('./ws.bin')

for t in range(1,num_new_batch_nodes):
    g1 = ws_ws_add(g1,t)
    eids = np.random.permutation(g1.number_of_edges())
    valid_eids = eids[int(len(eids) * 0.8):int(len(eids) * 0.9)]
    acc = LPEvaluate(gconv_model, g1, features, valid_eids, neg_sample_size)
    print("New edges batch{:05d) | MRR {:.4f}".format(t,acc))









