import numpy as np
import pandas as pd
import time

cache = {}

def load_data(mirNamePath):
    stime = time.time()
    print("from database HumanNet-XN load gene-gene interaction ... ", end='', flush=True)
    GGI = pd.read_csv("./miRSIm/HumanNet-XN.tsv", delimiter='\t')
    lls = GGI[['LLS']].values.T[0]
    LLSN = (lls-lls.min()) / (lls.max()-lls.min())
    GGI_dict = dict(zip(zip(GGI[['#EntrezGeneID1']].values.T[0], GGI[['EntrezGeneID2']].values.T[0]), LLSN))
    del lls
    del LLSN
    del GGI
    print('用时{:.2f}s'.format(time.time()-stime))

    stime = time.time()
    print("from database miRTarBase_MTI load miR target ... ", end='', flush=True)
    mirTar = pd.read_excel("./miRSIm/miRTarBase_MTI.xlsx", sheet_name="miRTarBase")
    mirTar = mirTar[['miRNA', 'Target Gene (Entrez ID)']]
    mirTar['miRNA'] = mirTar['miRNA'].str.upper()
    mirTar_dict = dict(zip(
        mirTar[['miRNA']].values.T[0], 
        [set({}) for i in range(len(mirTar[['miRNA']].values.T[0]))]
    ))
    for mname, gid in mirTar.values:
        mirTar_dict[mname].add(gid)
    del mirTar
    print('用时{:.2f}min'.format((time.time()-stime)/60))

    stime = time.time()
    print("reading miRNA name seq {} ... ".format(mirNamePath), end='', flush=True)
    m_name = np.loadtxt(mirNamePath, delimiter='\n', dtype=object)
    print('用时{:.2f}s'.format(time.time()-stime))

    return GGI_dict, mirTar_dict, m_name

def formula4(gi, gj, HumanNet):
    '''gene-gene interaction'''
    if gi == gj:
        return 1
    elif tuple((gi, gj)) in HumanNet:
        return HumanNet[(gi, gj)]
    elif tuple((gj, gi)) in HumanNet:
        return HumanNet[(gj, gi)]
    else:
        return 0

def formula5(g, G, _HumanNet):
    max_sim = 0.0
    for i in G:
        sim = formula4(g, i, _HumanNet)
        max_sim = sim if sim > max_sim else max_sim
    return max_sim

def formula6(mi:str, mj:str, mirTar_dict, _HumanNet):
    '''two node's of similarity'''    # hsa-let-7
    mi = mi.upper()
    mj = mj.upper()
    G_mi = mirTar_dict.get(mi)
    if G_mi is None:
        if cache.get(mi) is None:
            G_mi = set({})
            for mname in mirTar_dict:
                if mi in mname and mname[len(mi)] not in '0123456789':
                    G_mi |= mirTar_dict[mname]
            cache[mi] = G_mi
        else:
            G_mi = cache[mi]
    
    G_mj = mirTar_dict.get(mj)
    if G_mj is None:
        if cache.get(mj) is None:
            G_mj = set({})
            for mname in mirTar_dict:
                if mj in mname and mname[len(mj)] not in '0123456789':
                    G_mj |= mirTar_dict[mname]
            cache[mj] = G_mj
        else:
            G_mj = cache[mj]
    if G_mi is None or G_mj is None:
        print("NotFoundError: {} or {} is/are not in miRTarBase.".format(mi, mj))
        exit(0)
    
    sums = 0
    for g in G_mi:
        sums += formula5(g, G_mj, _HumanNet)
    for g in G_mj:
        sums += formula5(g, G_mi, _HumanNet)
    if len(G_mi) == 0 or len(G_mj) == 0:
        raise 'ERROR: len(G_mi) {} len(G_mj) {} sum {}'.format(len(G_mi),len(G_mj),sums)
    sim = sums / (len(G_mi) + len(G_mj))
    return sim

if __name__ == "__main__":
    GGI_dict, mirTar_dict, m_name = load_data()
    m_num = m_name.shape[0]
    Mtx = np.zeros((m_num, m_num))
    print("calculating ...")
    for i in range(m_num):
        Mtx[i,i] = 1
        for j in range(i+1, m_num):
            print("calculating '{}' and '{}' ...".format(m_name[i], m_name[j]))
            Mtx[i,j] = formula6(m_name[i], m_name[j], mirTar_dict, GGI_dict)
            Mtx[j,i] = Mtx[i,j]
    np.savetxt("sim.csv",Mtx, fmt='%.5f',delimiter=',')
