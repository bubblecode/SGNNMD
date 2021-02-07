import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
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