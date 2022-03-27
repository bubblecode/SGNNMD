from models import Mymodel, links2subgraphs_pred, links2subgraphs
from utils import getTopMetrics
from tqdm import tqdm
from torch.autograd import Variable
from torch import optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics
import torch.nn as nn
import torch
import scipy.sparse as sp
import numpy as np
from functools import reduce
import argparse
import os
import random
import time


class ProgressBar():
    def __init__(self, total_num, epoch, moniter=[]):
        self.__inner_width = 20
        self.__total = total_num
        self.__moniter = dict(zip(moniter, ['' for _ in moniter]))
        self.epoch = epoch

    def update(self, i, **moniter):
        if i > self.__total:
            i = self.__total
        progress = round((i/self.__total)*self.__inner_width)
        str_progress = '\rBatch:{:02d} ['.format(self.epoch)+'='*progress+'-'*(
            self.__inner_width-progress)+'] {}/{} '.format(i, self.__total)
        ext_info = ''
        for i in moniter.keys():
            ext_info += '{}:{:.4f} '.format(i, moniter[i])
        print(str_progress+ext_info, end='', flush=True)

    def __len__(self):
        return self.__total + 1


def constructNet(miRNA_disease, miRNA_similarity=None, disease_similarity=None):
    if miRNA_similarity is None:
        miRNA_similarity = np.array(
            np.eye(miRNA_disease.shape[0]), dtype=np.int8)
    if disease_similarity is None:
        disease_similarity = np.array(
            np.eye(miRNA_disease.shape[1]), dtype=np.int8)
    m1 = np.hstack((miRNA_similarity, miRNA_disease))
    m2 = np.hstack((miRNA_disease.T, disease_similarity))
    return np.vstack((m1, m2))


def train(model, g_list, sample_idxes, optimizer=None, criterion=None, bsize=50, epoch=-1, type='train',cv=-1):
    allnum = 0
    count = 0
    np.random.shuffle(g_list)
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1)
                   * (optimizer is None)) // bsize

    pbar = ProgressBar(total_iters, epoch=epoch+1, moniter=['loss', 'acc'])
    all_targets = []
    all_scores = []
    all_loc = []
    all_output = np.array([])
    n_samples = 0

    for pos in range(1, len(pbar)):
        selected_idx = sample_idxes[(pos-1) * bsize: pos * bsize]
        graph_list = [g_list[idx] for idx in selected_idx]
        g_loc = [g.location for g in graph_list]

        output = model(graph_list)
        targets = np.array([g_list[idx].label for idx in selected_idx])
        all_output = output.detach().numpy() if all_output.shape[0] == 0 else np.vstack(
            (all_output, output.detach().numpy()))
        targets[np.where(targets == 1)] = 0
        targets[np.where(targets == -1)] = 1
        targets = torch.tensor(targets.tolist())  # ?Tensor
        loss = criterion(output, targets)

        all_targets += targets.tolist()
        all_scores += output.argmax(axis=1).tolist()
        all_loc += g_loc
        # accuracy = metrics.accuracy_score(all_targets, all_scores)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = loss.data.detach().numpy()

        pbar.update(pos, loss=loss)
        total_loss.append(np.array([loss]) * len(selected_idx))
        n_samples += len(selected_idx)

    if optimizer is None:
        assert n_samples == len(sample_idxes)

    return getMetrics(np.array(all_scores), all_targets)


def predict(md, featM, featD, test_pos, test_neg):
    bsize = 50
    md[test_pos] = 0
    md[test_neg] = 0
    test_graphs, max_n = links2subgraphs_pred(md, 
                                             test_pos,
                                             test_neg,
                                             featM=featM,
                                             featD=featD)
    sample_idxes=list(range(len(test_graphs)))
    model = Mymodel(max_n, bio_feat=128)
    model_dict = torch.load('./model_dict_SGNNMD-1.pkl')
    model.load_state_dict(model_dict['param'])
    model.eval()
    model.pred()
    total_iters = (len(sample_idxes) + (bsize - 1)) // bsize

    pbar = ProgressBar(total_iters, epoch=1)
    all_targets = []
    init_targets = []
    all_scores = []
    all_output = []
    n_samples = 0
    for pos in range(1, len(pbar)):
        selected_idx = sample_idxes[(pos-1) * bsize : pos * bsize] 
        graph_list = [test_graphs[idx] for idx in selected_idx]
        
        ###########################################################################
        output = model(graph_list)
        init_targets = init_targets + [test_graphs[idx].label for idx in selected_idx]
        targets = np.array([test_graphs[idx].label for idx in selected_idx])
        targets[np.where(targets == 1)] = 0
        targets[np.where(targets == -1)] = 1
        ###########################################################################
        all_targets += targets.tolist()
        all_scores += output.argmax(axis=1).tolist()
        all_output += output[:,1].tolist()

        pbar.update(pos)
        n_samples += len(selected_idx)

    assert n_samples == len(sample_idxes)
    all_targets = np.array(all_targets)
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_output)
    auc = metrics.auc(fpr, tpr)
    pr, re, _ = metrics.precision_recall_curve(all_targets, all_output)
    aupr = metrics.auc(re, pr)
    precision = metrics.precision_score(all_targets, all_scores)
    recall  = metrics.recall_score(all_targets, all_scores)
    f1 = metrics.f1_score(all_targets, all_scores)
    results = np.array([auc, aupr, precision, recall, f1])
    return results, all_scores, all_targets, init_targets, all_output
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

def mask_array(md: np.array, mask_rate=0.5):
    shape = md.shape
    md_arr = np.copy(md.reshape(-1))
    nzero_idx = np.where(md_arr != 0)[0]
    masked_idx = np.random.choice(nzero_idx, int(
        len(nzero_idx)*mask_rate), replace=False)
    md_arr[masked_idx] = 0
    return md_arr.reshape(shape)


def cross_validation(k, md, featM, featD, train_pos, train_neg, test_pos, test_neg, bio_dim) -> list:
    print('----------CV {}----------'.format(k + 1))
    epochs = int(args.epoch)
    md = mask_array(md, mask_rate=0.10)
    MD = np.copy(md)

    MD[test_pos] = 0
    MD[test_neg] = 0

    MDT = np.copy(md)
    MDT[train_pos] = 0
    MDT[train_neg] = 0

    train_graphs, test_graphs, max_n = links2subgraphs(MD, MDT,
                                                       train_positive_edges, train_negative_edges,
                                                       test_positive_edges,  test_negative_edges,
                                                       h=hop, max_nodes_per_hop=None,
                                                       featM=featM, featD=featD)
    model = Mymodel(max_n, args.trunc, bio_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-4,
                           weight_decay=5e-4)  # [EXP]
    criterion = nn.CrossEntropyLoss()

    no_train = False
    if no_train:
        val_loss = np.array([0, 0, 0, 0, 0])
    else:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            train_loss = train(g_list=train_graphs,
                               model=model,
                               sample_idxes=list(range(len(train_graphs))),
                               optimizer=optimizer,
                               criterion=criterion,
                               epoch=epoch,
                               type='train',
                               cv=k)

            print('\r'+str(train_loss))
            model.eval()
            val_loss = train(g_list=test_graphs,
                             model=model,
                             sample_idxes=list(range(len(test_graphs))),
                             criterion=criterion,
                             optimizer=None,
                             epoch=epoch,
                             type='val',
                             cv=k)
            print('\r'+str(val_loss))

    return val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Up & Down regulation prediction.')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-m', '--misim-file', default=None, required=True)
    parser.add_argument('-d', '--disim-file', default=None, required=True)
    parser.add_argument('-b', '--bio-dim', default=128)
    parser.add_argument('-o', '--output', default='result.txt')
    parser.add_argument('--hop', default=1, type=int)
    parser.add_argument('-s', '--seed', default=44)
    parser.add_argument('-e', '--epoch', default=20)
    parser.add_argument('-k', '--kfolds', default=5, type=int)
    parser.add_argument('-t', '--trunc', default=25, type=int)
    parser.add_argument('-mask', '--mask', default=0, type=float)
    args = parser.parse_args()
    assert args.hop == 1, '[ValueWaring]: recommand hop == 1'

    bio_dim = int(args.bio_dim)
    seed = int(args.seed)  # np.random.randint(0, 100)
    print('seed={}'.format(seed))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    hop = args.hop  # 1
    if args.input[-4:] == '.npz':
        md = np.load(args.input)['data']
    elif args.input[-4:] == '.csv':
        md = np.loadtxt(args.input, delimiter=',')
    else:
        raise 'Error file extension name!'

    if args.misim_file is None:
        mm = np.eye(md.shape[0])
    elif args.misim_file[-4:] == '.csv':
        mm = np.loadtxt(args.misim_file, delimiter=',')
    elif args.misim_file[-4:] == '.npz':
        mm = np.load(args.misim_file)['data']
    else:
        raise 'Error file extension name!'

    if args.disim_file is None:
        dd = np.eye(md.shape[1])
    elif args.disim_file[-4:] == '.csv':
        dd = np.loadtxt(args.disim_file, delimiter=',')
    elif args.disim_file[-4:] == '.npz':
        dd = np.load(args.disim_file)['data']
    else:
        raise 'Error file extension name!'

    pca = PCA(n_components=bio_dim)
    featM = pca.fit_transform(mm)
    featD = pca.fit_transform(dd)
    md_sp = sp.coo_matrix(md)
    row, col, data = sp.find(md_sp)

    pos_idx = np.where(data == 1)
    neg_idx = np.where(data == -1)
    all_positive_edges = np.array((row[pos_idx], col[pos_idx])).T
    all_negative_edges = np.array((row[neg_idx], col[neg_idx])).T

    # dataset split
    load_split_data = False
    if load_split_data:
        data = np.load("split_data_{}.npz".format(args.mask), allow_pickle=True)
        kfolds = data['kfolds']
        kf_pos = data['kf_pos']
        kf_neg = data['kf_neg']
    else:
        kf = KFold(n_splits=int(args.kfolds), shuffle=True)
        kfolds = kf.get_n_splits()
        kf_pos = [train_test for train_test in kf.split(all_positive_edges)]
        kf_neg = [train_test for train_test in kf.split(all_negative_edges)]
        np.savez('split_data_{}1.npz'.format(args.mask),
                 kfolds=kfolds,
                 kf_pos=kf_pos,
                 kf_neg=kf_neg,
                 allpos=all_positive_edges,
                 allneg=all_negative_edges)

    kf_metrics = []
    for k in range(kfolds):
        train_positive_edges = all_positive_edges[kf_pos[k][0]]
        train_negative_edges = all_negative_edges[kf_neg[k][0]]
        test_positive_edges = all_positive_edges[kf_pos[k][1]]
        test_negative_edges = all_negative_edges[kf_neg[k][1]]

        train_positive_edges = (
            train_positive_edges.T[0], train_positive_edges.T[1])
        train_negative_edges = (
            train_negative_edges.T[0], train_negative_edges.T[1])
        test_positive_edges = (
            test_positive_edges.T[0],  test_positive_edges.T[1])
        test_negative_edges = (
            test_negative_edges.T[0],  test_negative_edges.T[1])
        print('train_pos:{}  train_neg:{}  test_pos:{}  test_neg:{}'.format(
            train_positive_edges[0].shape[0],
            train_negative_edges[0].shape[0],
            test_positive_edges[0].shape[0],
            test_negative_edges[0].shape[0]
        )
        )
        res = cross_validation(k, md, featM, featD, train_positive_edges, train_negative_edges,
                               test_positive_edges, test_negative_edges, bio_dim)
        kf_metrics.append(res)
    print(reduce(lambda x, y: x+y, np.array(kf_metrics))/int(args.kfolds))
    with open(args.output, 'w') as f:
        f.write('[aupr auc pr re f1]\n')
        f.write(str(reduce(lambda x, y: x+y, np.array(kf_metrics))/int(args.kfolds)))