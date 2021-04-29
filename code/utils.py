from sklearn import metrics
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def draw_acc_loss(train_acc, train_loss, val_acc, val_loss, path=None, title=None):
    epoch = list(range(len(train_acc)))
    if path is None:
        path = 'train_loss__{}.png'.format(str(time.time()))
    if title is None:
        title = 'undefined'
    plt.cla()
    plt.plot(epoch, train_loss, 'b-', label='train_loss')
    plt.plot(epoch, val_loss, 'k-', label='val_loss')
    plt.plot(epoch, train_acc, 'r-', label='train_acc')
    plt.plot(epoch, val_acc, 'g-', label='val_acc')
    plt.title(title)
    plt.legend()
    plt.savefig(path)


def getTopMetrics(preds, targets):
    ## TOP: 取值高的那一个进行指标计算
    """
    请确保传入的数据已做过如下操作：output.argmax(axis=1).tolist()
    preds:   [0.333, 0.444, 0.555, ...]
    targets：[1, 0, 1, 0, 1, ...]
    """
    preds = np.array(preds)
    targets = np.array(targets)
    all_targets = np.array(targets)
    fpr, tpr, _ = metrics.roc_curve(targets, preds)
    auc = metrics.auc(fpr, tpr)
    pr, re, _ = metrics.precision_recall_curve(targets, preds)
    aupr = metrics.auc(re, pr)
    # preds[np.where(preds >= 0.5)] = 1
    # preds[np.where(preds < 0.5)] = 0
    precision = metrics.precision_score(targets, preds)
    recall = metrics.recall_score(targets, preds)
    f1 = metrics.f1_score(targets, preds)
    return [aupr, auc, precision, recall, f1]


def getMacroMetrics(predict_score, real_score):
    """
    output: np.matrix([[0.11,0.22,...]])
    target: np.matrix([[1,0,1,0,1,...]])
    """
    l = np.array(real_score)
    p = 1-(1-predict_score[:, 0])*(1-predict_score[:, 1])  ## 此处存在链接的概率
    l[np.where(l != 0)] = 1
    p[np.where(p > 0.5)]= 1; p = p.astype(np.int)
    fpr, tpr, _ = metrics.roc_curve(l, p)
    auc = metrics.auc(fpr, tpr)
    pr, re, _ = metrics.precision_recall_curve(l, p)
    aupr = metrics.auc(re, pr)
    precision = metrics.precision_score(l, p)
    recall = metrics.recall_score(l, p)
    f1 = metrics.f1_score(l, p)
    return [aupr, auc, f1, precision, recall]
