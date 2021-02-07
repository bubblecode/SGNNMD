import numpy as np
from os import system
from datavision import iseedeadpeople
class FilePath():
    def __init__(self, m_d,m_m,d_d,m_name,d_name,d_id):
        self.m_d = m_d
        self.m_m = m_m
        self.d_d = d_d
        self.m_name = m_name
        self.d_name = d_name
        self.d_id = d_id

def save_adj_edge_list(matrix: np.array):
    row_num = matrix.shape[0]
    col_num = matrix.shape[1]
    mm = np.zeros((row_num, row_num))
    dd = np.zeros((col_num, col_num))
    m1 = np.hstack((mm, matrix))
    m2 = np.hstack((matrix.T, dd))
    adj = np.vstack((m1, m2))
    neg_x = np.where(adj == -1)[0]
    neg_y = np.where(adj == -1)[1]
    neg_lab = np.repeat(-1, neg_x.shape[0])

    pos_x = np.where(adj == 1)[0]
    pos_y = np.where(adj == 1)[1]
    pos_lab = np.repeat(1, pos_x.shape[0])

    label = np.hstack((neg_lab, pos_lab))
    x = np.hstack((neg_x, pos_x))
    y = np.hstack((neg_y, pos_y))
    res = np.array((x, y, label)).T
    np.savetxt('xxx_md_edge_list.txt', res, delimiter=' ', fmt='%d')
    print('{} edge list file saved ...'.format(adj.shape))


def data_clearning(fp: FilePath, threshold_up: int, threshold_down: int, flag='&', savefile=False) -> np.array:
    m_d = np.loadtxt(fp.m_d, delimiter=',')

    up_regulated_idx_m = np.zeros((m_d.shape[0],))
    down_regulated_idx_m = np.zeros((m_d.shape[0],))
    row_idx = np.where(m_d == 1)[0]
    for i in row_idx:
        up_regulated_idx_m[i] += 1
    row_idx = np.where(m_d == -1)[0]
    for i in row_idx:
        down_regulated_idx_m[i] += 1


    res_row_idx1 = np.where(up_regulated_idx_m >= threshold_up)[0]
    res_row_idx2 = np.where(down_regulated_idx_m >= threshold_down)[0]
    if flag == '&':
        res_row_idx = list(set(res_row_idx1) & set(res_row_idx2))
    elif flag == '|':
        tmp = np.where(up_regulated_idx_m >= threshold_up+threshold_down)[0]
        res_row_idx = list((set(res_row_idx1) & set(res_row_idx2)) | set(tmp))
    else:
        assert 'ERROR'
        return
    filtered_m_d = m_d[res_row_idx,]
    
    m_dT = filtered_m_d.T
    up_regulated_idx_d   = np.zeros((m_dT.shape[0],))
    down_regulated_idx_d = np.zeros((m_dT.shape[0],))
    col_idx = np.where(m_dT == 1)[0]
    for i in col_idx:
        up_regulated_idx_d[i] += 1
    col_idx = np.where(m_dT == -1)[0]
    for i in col_idx:
        down_regulated_idx_d[i] += 1


    res_col_idx1 = np.where(up_regulated_idx_d >= threshold_up)[0]
    res_col_idx2 = np.where(down_regulated_idx_d >= threshold_down)[0]
    if flag == '&':
        res_col_idx = list(set(res_col_idx1) & set(res_col_idx2))
    elif flag == '|':
        tmp = np.where(up_regulated_idx_d >= threshold_up+threshold_down)[0]
        res_col_idx = list((set(res_col_idx1) & set(res_col_idx2)) | set(tmp))
    else:
        assert 'ERROR'
        return

    filtered_m_d = filtered_m_d[:,res_col_idx]
    print("filiterd：{}".format(filtered_m_d.shape))
    print('up-regulation：{}'.format(np.where(filtered_m_d == 1)[0].shape[0]))
    print('down-regulation：{}'.format(np.where(filtered_m_d == -1)[0].shape[0]))
    print('all link nums：{}'.format(np.where(filtered_m_d != 0)[0].shape[0]))

    m_m = np.loadtxt(fp.m_m, delimiter=',')
    d_d = np.loadtxt(fp.d_d, delimiter=',')
    m_name = np.loadtxt(fp.m_name, dtype=str)
    d_name = np.loadtxt(fp.d_name, dtype=str, delimiter='\t')
    d_id   = np.loadtxt(fp.d_id, dtype=str)

    filtered_m_m = m_m[res_row_idx,]
    filtered_m_m = filtered_m_m.T[res_row_idx,].T
    filtered_d_d = d_d[res_col_idx,]
    filtered_d_d = filtered_d_d.T[res_col_idx,].T
    filtered_m_name = m_name[res_row_idx]
    filtered_d_name = d_name[res_col_idx]
    filtered_d_id   = d_id[res_col_idx]
    print()

    iseedeadpeople(filtered_m_d, showDifference=True, name='m_d2')
    if savefile == False:
        return filtered_m_d, \
            filtered_m_m, \
            filtered_d_d, \
            filtered_m_name, \
            filtered_d_name, \
            filtered_d_id
    else:
        system('mkdir -p {}'.format(savefile))
        save_adj_edge_list(filtered_m_d)
        np.savetxt('{}/m_d.csv'.format(savefile),filtered_m_d,delimiter=',', fmt='%d')
        np.savetxt('{}/m_m.csv'.format(savefile),filtered_m_m,delimiter=',', fmt='%lf')
        np.savetxt('{}/d_d.csv'.format(savefile),filtered_d_d,delimiter=',', fmt='%lf')
        np.savetxt('{}/m_name.txt'.format(savefile),filtered_m_name,delimiter=',', fmt='%s')
        np.savetxt('{}/d_name.txt'.format(savefile),filtered_d_name,delimiter=',', fmt='%s')
        np.savetxt('{}/d_id.txt'.format(savefile),filtered_d_id,delimiter=',', fmt='%s')
        print('saved ...')



if __name__ == '__main__':
    fp = FilePath(
        'data/miRNA-disease.csv',
        "data/miRNA-similarity.csv",
        "data/disease-similarity.csv",
        "data/miRNA_name.txt",
        "data/dis_name.txt",
        "data/dis_id.txt"
    )
    data_clearning(fp, threshold_up=1, threshold_down=1, flag='&', savefile='dataset')
    print()

