import os,sys
from   pdb import *
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import chainer
from chainer import Variable

# loadin from .npy
def load_npz(gdir):
    #print("loading",gdir)
    assert os.path.exists(gdir), "Not found `{}` directory".format(gdir)
    cwd = os.getcwd()
    os.chdir(gdir)
    a = np.load('lx1_Linear_out.npy') 
    a = np.reshape(a, (-1, 50, 1))
    b = np.load('ly2_Linear_in.npy')
    b = np.reshape(b, (-1, 50, 1))
    c = np.load('ly2_Linear_out.npy')
    c = np.reshape(c, (-1, 50, 1))
    d = np.load('lz3_Linear_in.npy')
    d = np.reshape(d, (-1, 50, 1))
    os.chdir(cwd)
    return np.concatenate((a, b, c, d), axis=2)

# loading normal system result
def load_normal(gdir):
    x= load_npz(gdir)
    return x

# loading faults system result
def load_faults(faultNo_list, Nimage):
    hm  = np.zeros((max(faultNo_list)+1, Nimage, 50, 4 )) # hm.shape = (faultNo, imageNo, 50, 4 )
    for faultNo in faultNo_list:
        fault_dir = "list%d_no0-%d"%(faultNo, Nimage-1)
        h = load_npz(fault_dir)
        hm[faultNo] = h
    return hm

# spec to load
faultNo_list = list(set([10,0,1]))              # ex. Fault-wise
faultNo_list = range(784)                       # ex. About all faults
data_P = 20
normal  = 'original_data2'

# load normal and fault systems
nmap = load_normal(normal)                      # nmap.shape = (imageNo, 50, 4)
fmap = load_faults(faultNo_list, data_P)        # fmap.shape = (faultNo, imageNo, 50, 4)

#jは推論する画像の番号、iは壊す範囲の左上座標のリストでの番号

hm_sum= np.zeros((50,4),dtype=np.float64)

for j in range(0,data_P): #オリジナルから行列を作成  # j:number of image
    # Normal System
    hm_a = nmap[j]

    for i in faultNo_list:
        hm_b = fmap[i][j]
        assert hm_b.shape == hm_a.shape,"Internal Error"

        hm = hm_a-hm_b
        zettaichi = np.abs(hm)
        hm_sum = zettaichi + hm_sum
print("Check sum-1:",np.sum(nmap[:data_P]),np.sum(fmap[:784][:data_P]))
print("CHeck sum-2:",np.sum(hm_sum))
#sys.exit(-1)

#型変換等
mylist=[0]
del mylist[0]
for k in range(50):
    for l in range(4):
        a='variable' in str(hm_sum[k,l])
        if a==True:
            s = str(hm_sum[k,l]).split('(')
            s = s[1].split(')')
            s = s[0]
            #print(s)
            mylist.append(s)
        else:
            mylist.append(str(hm_sum[k,l]))

mylist_f = [float(s) for s in mylist]
mylist = np.array(mylist_f)
mylist = np.reshape(mylist, (50, 4))


#ヒートマップをつくる、表示する
heatmap = sns.heatmap(mylist)
plt.show(heatmap)


    

    
