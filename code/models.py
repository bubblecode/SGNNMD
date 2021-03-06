import random
import numpy as np
import scipy.sparse as sp
import time
import multiprocessing as mp
from tqdm import tqdm
import networkx as nx

class GNNGraph(object):
    def __init__(self, g, label, node_labels=None, node_feat=None, pos_edge=None,neg_edge=None,i=None,j=None):
        self.num_nodes = node_labels.shape[0] #! node nums
        self.label = label # graph label
        self.node_labels = node_labels  # numpy array (node_num * feature_dim) #?None
        self.node_feat = node_feat
        self.location = (i, j) 
        self.degs = list(dict(g.degree).values()) #! len(degree) == num_nodes
        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)    ## num of edges
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y #[[x1,y1],[x2,y2],...]
            self.edge_pairs = self.edge_pairs.flatten()
        else: #! False
            self.num_edges = 0
            self.edge_pairs = np.array([])
        self.pos_edge = pos_edge
        self.neg_edge = neg_edge

max_n = 0
train_max_shape = (0,0)
val_max_shape=(0,0)
test_max_shape=(0,0)
def constructNet(miRNA_disease=None, miRNA_similarity=None, disease_similarity=None):
    if miRNA_disease is None:
        # Special purpose: unless you know what you are doing, don't let this parameter be None.
        miRNA_disease = np.zeros((miRNA_similarity.shape[0], disease_similarity.shape[0]))
    if miRNA_similarity is None:
        miRNA_similarity = np.array(np.eye(miRNA_disease.shape[0]), dtype=np.int8)
    if disease_similarity is None:
        disease_similarity = np.array(np.eye(miRNA_disease.shape[1]), dtype=np.int8)
    m1 = np.hstack((miRNA_similarity, miRNA_disease))
    m2 = np.hstack((miRNA_disease.T, disease_similarity))
    return np.vstack((m1, m2))

def getNeighborsFromM(fr0, fr1, fr2, Mtx) -> set:
    pure_up = set()
    pure_dn = set()
    mixed = set()
    # direct
    if type(Mtx) is not np.matrix:
        Mtx = np.matrix(Mtx)
    if fr1 is None and fr2 is None:
        for n in fr0:
            pure_up_ctx = np.where(Mtx[n,:] == 1)[1]
            pure_dn_ctx = np.where(Mtx[n,:] ==-1)[1]
            pure_up_ctx = set(pure_up_ctx)
            pure_dn_ctx = set(pure_dn_ctx)
            pure_up = pure_up.union(pure_up_ctx)
            pure_dn = pure_dn.union(pure_dn_ctx)
            return pure_up, pure_dn, mixed
    # search in all '+' link
    for n in fr0:
        pure_up_ctx = np.where(Mtx[n,:] == 1)[1]
        mixed_ctx = np.where(Mtx[n,:] == -1)[1]
        pure_up = pure_up.union(pure_up_ctx)
        mixed = mixed.union(mixed_ctx)
    # search in all '-' link
    for n in fr1:
        pure_dn_ctx = np.where(Mtx[n,:] == -1)[1]
        mixed_ctx = np.where(Mtx[n,:] == 1)[1]
        pure_dn = pure_dn.union(pure_dn_ctx)
        mixed = mixed.union(mixed_ctx)
    # mixed
    for n in fr2:
        mixed_ctx = np.where(Mtx[n,:] != 0)[1]
        mixed = mixed.union(mixed_ctx)
    # pure link first
    pure_dn = pure_dn - (pure_up&pure_dn)
    mixed = mixed - (mixed&pure_up)
    mixed = mixed - (mixed&pure_dn)

    pure_up  = pure_up - mixed
    pure_dn  = pure_dn - mixed
    return pure_up, pure_dn, mixed

def getNeighborsFromD(fr0, fr1, fr2, Mtx) -> set:
    pure_up = set()
    pure_dn = set()
    mixed = set()
    ff = lambda item: item in pure_up

    if type(Mtx) is not np.matrix:
        Mtx = np.matrix(Mtx)
    if fr1 is None and fr2 is None:
        for n in fr0:
            pure_up_ctx = np.where(Mtx[:,n] == 1)[0]
            pure_dn_ctx = np.where(Mtx[:,n] ==-1)[0]
            pure_up = pure_up.union(pure_up_ctx)
            pure_dn = pure_dn.union(pure_dn_ctx)
            return pure_up, pure_dn, mixed
    for n in fr0:
        pure_up_ctx = np.where(Mtx[:,n] == 1)[0]
        mixed_ctx = np.where(Mtx[:,n] == -1)[0]
        pure_up = pure_up.union(pure_up_ctx)
        mixed = mixed.union(mixed_ctx)
    for n in fr1:
        pure_dn_ctx = np.where(Mtx[:,n] == -1)[0]
        mixed_ctx = np.where(Mtx[:,n] == 1)[0]
        pure_dn = pure_dn.union(pure_dn_ctx)
        mixed = mixed.union(mixed_ctx)
    for n in fr2:
        mixed_ctx = np.where(Mtx[:,n] != 0)[0]
        mixed = mixed.union(mixed_ctx)

    pure_dn = pure_dn - (pure_up&pure_dn)
    mixed = mixed - (mixed&pure_up)
    mixed = mixed - (mixed&pure_dn)

    pure_up  = pure_up - mixed
    pure_dn  = pure_dn - mixed
    return pure_up, pure_dn, mixed

def subgraph_extraction_labeling(ind, Mtx:np.matrix, h=1, max_nodes_per_hop=None, featM=None, featD=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    m_nodes, d_nodes = [], []
    m_0_visited, m_1_visited, m_2_visited = set([]), set([]), set([])
    d_0_visited, d_1_visited, d_2_visited = set([]), set([]), set([])
    
    m_0_fringe, m_1_fringe, m_2_fringe = set([]), set([]), set([])
    d_0_fringe, d_1_fringe, d_2_fringe = set([]), set([]), set([])

    m_0_mark, m_1_mark, m_2_mark = set([]), set([]), set([])
    d_0_mark, d_1_mark, d_2_mark = set([]), set([]), set([])
    node_label_m, node_label_d = [], []  ## [0],  [1]
    
    # node labeling
    for hop in range(1, h+1):
        if hop == 1:
            d_0_fringe, d_1_fringe, d_2_fringe = getNeighborsFromM(set([ind[0]]),None,None, Mtx)
            m_0_fringe, m_1_fringe, m_2_fringe = getNeighborsFromD(set([ind[1]]),None,None, Mtx)
        else:
            d_0_fringe, d_1_fringe, d_2_fringe = getNeighborsFromM(m_0_mark, m_1_mark, m_2_mark, Mtx)
            m_0_fringe, m_1_fringe, m_2_fringe = getNeighborsFromD(d_0_mark, d_1_mark, d_2_mark, Mtx)
        assert not bool(d_0_fringe&d_1_fringe or d_0_fringe&d_2_fringe or d_1_fringe&d_2_fringe), "ERROR: found repeat number in hop {}".format(hop)
        assert not bool(m_0_fringe&m_1_fringe or m_0_fringe&m_2_fringe or m_1_fringe&m_2_fringe), "ERROR: found repeat number in hop {}".format(hop)
        m_0_fringe = m_0_fringe - m_0_visited - m_1_visited - m_2_visited - set(m_nodes)
        m_1_fringe = m_1_fringe - m_0_visited - m_1_visited - m_2_visited - set(m_nodes)
        m_2_fringe = m_2_fringe - m_0_visited - m_1_visited - m_2_visited - set(m_nodes)
        d_0_fringe = d_0_fringe - d_0_visited - d_1_visited - d_2_visited - set(d_nodes)
        d_1_fringe = d_1_fringe - d_0_visited - d_1_visited - d_2_visited - set(d_nodes)
        d_2_fringe = d_2_fringe - d_0_visited - d_1_visited - d_2_visited - set(d_nodes)
        m_0_mark = m_0_fringe  # pred result
        m_1_mark = m_1_fringe  # pred result
        m_2_mark = m_2_fringe  # pred result
        d_0_mark = d_0_fringe  # pred result
        d_1_mark = d_1_fringe  # pred result
        d_2_mark = d_2_fringe  # pred result
        # update visit
        m_0_visited = m_0_visited.union(m_0_fringe)
        m_1_visited = m_1_visited.union(m_1_fringe)
        m_2_visited = m_2_visited.union(m_2_fringe)
        d_0_visited = d_0_visited.union(d_0_fringe)
        d_1_visited = d_1_visited.union(d_1_fringe)
        d_2_visited = d_2_visited.union(d_2_fringe)
        # ignore none link
        if (len(m_0_fringe) == 0 and len(m_1_fringe) == 0 and len(m_2_fringe) == 0 
            and len(d_0_fringe) == 0 and len(d_1_fringe) == 0 and len(d_2_fringe) == 0):
            break
        m_nodes = m_nodes + list(m_0_fringe) + list(m_1_fringe) + list(m_2_fringe)
        d_nodes = d_nodes + list(d_0_fringe) + list(d_1_fringe) + list(d_2_fringe)
        assert len(m_nodes) == len(set(m_nodes)), "Error: Duplicate node found in hop {}.".format(hop)
        assert len(d_nodes) == len(set(d_nodes)), "Error: Duplicate node found in hop {}.".format(hop)
        node_label_m = node_label_m + [6*hop-4] * len(m_0_fringe) + [6*hop-2] * len(m_1_fringe) + [6*hop]   * len(m_2_fringe)
        node_label_d = node_label_d + [6*hop-3] * len(d_0_fringe) + [6*hop-1] * len(d_1_fringe) + [6*hop+1] * len(d_2_fringe)
        print('',end='')
    rand_state_m = np.random.get_state()
    np.random.set_state(rand_state_m); np.random.shuffle(m_nodes)
    np.random.set_state(rand_state_m); np.random.shuffle(node_label_m)
    rand_state_d = np.random.get_state()
    np.random.set_state(rand_state_d); np.random.shuffle(d_nodes)
    np.random.set_state(rand_state_d); np.random.shuffle(node_label_d)
    m_nodes = [ind[0]] + m_nodes
    d_nodes = [ind[1]] + d_nodes
    node_label_m = [0] + node_label_m
    node_label_d = [1] + node_label_d

    subgraph = np.array(Mtx[m_nodes, :][:, d_nodes])
    subgraph[0, 0] = 0
    node_labels = np.array(node_label_m + node_label_d)
    node_features = sp.coo_matrix(np.vstack((featM[m_nodes,:],featD[d_nodes,:])))
    ################################################
    # generate graph
    subgraphAdj = constructNet(subgraph)
    pos_edge = np.where(subgraphAdj == 1)
    neg_edge = np.where(subgraphAdj == -1)
    subgraphAdj = sp.coo_matrix(subgraphAdj)
    print('\rSubgraph extracting: ({},{}) '.format(subgraphAdj.shape[0], subgraphAdj.shape[1]), end='')
    global train_max_shape
    global val_max_shape
    if subgraphAdj.shape[0] > val_max_shape[0]:
        val_max_shape = subgraphAdj.shape
    g = nx.from_scipy_sparse_matrix(subgraphAdj)
    return g, node_labels, node_features ,pos_edge, neg_edge, max(node_labels), ind
def links2subgraphs(Mtx, Mtxori, train_pos,train_neg, test_pos,test_neg, h=1, max_nodes_per_hop=None, featM=None,featD=None) -> list:
    # extract enclosing subgraphs
    ###############################################################
    global train_max_shape
    global val_max_shape
    def helper(M, links, g_label):
        global max_n
        g_list = []
        # the parallel extraction code
        start = time.time()
        results = [subgraph_extraction_labeling(*((i, j), M, h, max_nodes_per_hop, featM, featD)) for i, j in zip(links[0], links[1])]
        g_list = [GNNGraph(g, g_label, n_label, n_feat, pos_edge, neg_edge) for g,n_label,n_feat,pos_edge,neg_edge,_,_ in results]
        if len(results) != 0:
            max_n = max([i[5] for i in results])
        end = time.time()
        print(" \rSubgraph extracting ... ok (Time: {:.2f}s)".format(end-start), flush=True)
        return g_list
    ###########################################################
    train_graphs = helper(Mtx, train_pos, 1) + helper(Mtx, train_neg, -1)
    train_max_shape = val_max_shape
    val_max_shape = (0,0)
    test_graphs = helper(Mtxori, test_pos, 1) + helper(Mtxori, test_neg, -1)
    print('max_n:{}'.format(max_n))
    print('subgraph train shape:({},{})  subgraph val shape:({},{})'.format(
            train_max_shape[0], train_max_shape[1],
            val_max_shape[0], val_max_shape[1]
        )
    )
    train_max_shape = (0,0)
    val_max_shape = (0,0)
    return train_graphs, test_graphs, max_n
def constructSubgraph4pred(Mtx, edge_pos, edge_neg, edge_zero, h=1, featM=None,featD=None):
    global max_n
    def helper(M, links, g_label):
        global max_n
        g_list = []
        start = time.time()
        results = [subgraph_extraction_labeling(*((i, j), M, h, None, featM, featD)) for i, j in links]
        g_list = [GNNGraph(g, g_label, n_label, n_feat, pos_edge, neg_edge,ind[0],ind[1]) for g,n_label,n_feat,pos_edge,neg_edge,_,ind in results]
        if len(results) != 0:
            max_n = max([i[5] for i in results])
        end = time.time()
        print(" \rSubgraph extracting ... ok (Time: {:.2f}s)".format(end-start), flush=True)
        return g_list
    ###########################################################
    test_graphs = helper(Mtx, edge_pos, 1) + helper(Mtx, edge_neg, -1) + helper(Mtx, edge_zero, 0)
    max_n = 5 if max_n < 5 else max_n
    print('pred_max_n:{}'.format(max_n))
    return test_graphs, max_n

#!############################################################################
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)

def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)

def weights_init(m): 
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch import optim
import ctypes
import numpy as np
import os
import sys
import torch
import pdb
from torch_geometric.nn import DenseGraphConv,DenseGCNConv
def one_hot(labels, max_n):
    labels = torch.LongTensor(labels)
    out_tensor = torch.nn.functional.one_hot(labels, int(max_n)+1)
    return out_tensor.type(torch.FloatTensor)

def MyPrepareSparseMatrices(graph_list: GNNGraph, graph_sizes: list):
    """graph_list:
       (return): torch.sparse.FloatTensor
    """
    account = 0
    all_node_num = sum([i.num_nodes for i in graph_list])
    n2n = np.zeros((all_node_num, all_node_num))
    for i in range(len(graph_list)):
        if graph_list[i].num_edges == 0:
            pass
        else:
            ep_pos = (graph_list[i].pos_edge[0]+account, 
                                    graph_list[i].pos_edge[1]+account)
            ep_neg = (graph_list[i].neg_edge[0]+account, 
                                    graph_list[i].neg_edge[1]+account)
            n2n[ep_pos] = 1;  n2n[(ep_pos[1],ep_pos[0])] = 1
            n2n[ep_neg] = -1; n2n[(ep_neg[1],ep_neg[0])] = -1
        account += graph_sizes[i]
    pos_edges = np.where(n2n == 1)
    neg_edges = np.where(n2n == -1)
    n2n_sp = sp.coo_matrix(n2n)
    values = n2n_sp.data
    indices = np.vstack((n2n_sp.row, n2n_sp.col))
    values = torch.FloatTensor(values)
    indices = torch.LongTensor(indices)
    shape = n2n_sp.shape

    return torch.sparse.FloatTensor(indices, values, shape), pos_edges, neg_edges

class GCN(nn.Module):
    def __init__(self, num_node_feats, hidden_dim=[32, 32], k=25, conv1d_channels=[16, 32], conv1d_kws=[0, 5], bio_feat=128, conv1d_activation='ReLU'):
        super(GCN, self).__init__()
        self.predict = False
        self.hidden_dim = hidden_dim
        self.k = k
        self.total_hidden_dim = sum(hidden_dim) ## 97
        conv1d_kws[0] = self.total_hidden_dim
        self.gcn_topo1 = DenseGCNConv(num_node_feats, hidden_dim[0])
        self.gcn_topo2 = DenseGCNConv(hidden_dim[0], hidden_dim[1])
        self.locations = None
        self.__hid_type = 'c'  #! 't'=just topo  's'=just bio  'c'=combine topo and bio
        weights_init(self)

    def forward(self, graph_list):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)
        n2n_sp, pos_edge, neg_edge= MyPrepareSparseMatrices(graph_list, graph_sizes) #?Tensor

        node_feat = np.concatenate([i.node_labels for i in graph_list])
        node_feat = one_hot(node_feat, max_n) #?Tensor 

        bio_feat = None
        for g in graph_list:
            bio_feat = sp.vstack((bio_feat, g.node_feat))
        bio_feat = Variable(torch.FloatTensor(bio_feat.todense()))
        bio_feat = Variable(bio_feat)
        n2n_sp = Variable(n2n_sp.to_dense())
        node_degs = Variable(node_degs)
        #! GCNembedding ????????????
        h = self.pooling(node_feat, bio_feat, n2n_sp, graph_sizes, node_degs)

        if self.predict:
            self.locations = [graph_list[i].location for i in range(len(graph_list))]
        return h

    def pooling(self, node_feat, concat_feat, n2n_sp, graph_sizes, node_degs):
        hid_topo = self.gcn_topo1(node_feat, n2n_sp)[0]
        hid_topo = F.elu(hid_topo)
        hid_topo = self.gcn_topo2(hid_topo, n2n_sp)[0]
        #############################
        # cur_message_layer = hid_topo
        # cur_message_layer = concat_feat
        cur_message_layer = torch.cat((hid_topo, concat_feat), 1)
        self.total_hidden_dim = cur_message_layer.shape[1]
        ''' pooling layer ''' #! DGCNN????????????
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.total_hidden_dim) ## 50???(27,xxx)?????????
        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)  ## tensor(50,27,xxx)
        accum_count = 0
        for i in range(len(graph_sizes)):  ## ??????50???,????????????batch_size
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]  ## k?????????self.k???graph_size???????????????
            topk_indices = torch.LongTensor(list(range(k)))
            topk_indices += accum_count  ## ??????????????????+accum_count
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices)  ## ???????????????????????????????????????
            if k < self.k:  ## ?????????????????????
                to_pad = torch.zeros(self.k-k, self.total_hidden_dim) ## (27-k,xxx)
                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0) ##?????????np.vstack (27,xxx)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i] ## ??????graph_sizes[i]???????????????

        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_hidden_dim))
        return to_conv1d.squeeze(1)

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.linear_w1 = nn.Linear(input_dim, hidden_dim)
        self.linear_w2 = nn.Linear(hidden_dim, output_dim)
        # self.training = training

        for p in self.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    if isinstance(pp, Parameter):
                        self.reset_weight(pp.data)
                    elif isinstance(pp, nn.Linear):
                        pp.bias.data.zero_()
                        self.reset_weight(pp.weight.data)
            else:
                if isinstance(p, Parameter):
                    self.reset_weight(p.data)
                elif isinstance(p, nn.Linear):
                    p.bias.data.zero_()
                    self.reset_weight(p.weight.data)
        for name,p in self.named_parameters():
            if not '.' in name:
                if isinstance(p, Parameter):
                    self.reset_weight(p.data)
                elif isinstance(p, nn.Linear):
                    p.bias.data.zero_()
                    self.reset_weight(p.weight.data)

    def reset_weight(self, t):
        if len(t.size()) == 2:
            fan_in, fan_out = t.size()
        elif len(t.size()) == 3:
            fan_in = t.size()[1] * t.size()[2]
            fan_out = t.size()[0] * t.size()[2]
        else:
            fan_in = np.prod(t.size())
            fan_out = np.prod(t.size())

        limit = np.sqrt(6.0 / (fan_in + fan_out))
        t.uniform_(-limit, limit)

    def forward(self, x):
        hid1 = self.linear_w1(x)
        hid1 = F.relu(hid1)
        out = self.linear_w2(hid1)
        # return out
        return torch.sigmoid(out)
class Mymodel(nn.Module):
    def __init__(self, topo_feat=5, bio_feat=128):
        super(Mymodel, self).__init__()
        k = 25    # [EXP] [16, 32, 64, 128, 256, 512, 1024]
        self.gcn = GCN(num_node_feats=max_n+1, k=k,bio_feat=bio_feat, conv1d_activation='ReLU')
        # self.clsf   = Classifier(input_dim=1792, hidden_dim=32, output_dim=2)
        self.clsf   = Classifier(input_dim=k*32*5, hidden_dim=32, output_dim=2)
        self.feat_dim = topo_feat ## ??????????????????+1
        self.predict = False
        self.locations = None
        self.last_embedding = None
    def forward(self, batch_input):
        embedding = self.gcn(batch_input)   #(50,) (xxxx,4)
        output = self.clsf(embedding) ## ??????:(50,288)?????????:(50,3)
        if self.predict:
            self.locations = self.gcn.locations
            self.last_embedding = embedding
        return output
    def mytrain(self):
        self.predict = False
        self.gcn.predict = False
    def pred(self):
        self.predict = True
        self.gcn.predict = True
