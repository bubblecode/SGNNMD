from sklearn import metrics
import time
import numpy as np

def getMetrics(preds, targets):
    preds = np.array(preds).argmax(axis=1)
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
    l = np.array([[0,1] if i == 1 else [1,0] for i in real_score]).flatten()
    p = predict_score.flatten()
    # p = 1-(1-predict_score[:, 0])*(1-predict_score[:, 1])  ## 此处存在链接的概率
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
