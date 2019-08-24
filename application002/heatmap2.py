import os,sys,argparse
from   pdb import *
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import chainer
from chainer import Variable

args = argparse.ArgumentParser()
args.add_argument('-i','--images',type=int,default=20)
args.add_argument('-f','--faults',type=int,default=784, dest='faults')
args.add_argument('-F','--Faults',type=int,nargs='+')
args = args.parse_args()

# spec to load
#faultNo_list = range(args.faults)
faultNo_list = args.Faults if args.Faults is not None else range(args.faults)
data_P       = args.images
normal_dir   = 'original_data2'
print("Run on Faults as",faultNo_list)

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
    return load_npz(gdir)

# loading faults system result
def load_faults(faultNo_list, Nimage):
    fm  = np.zeros((max(faultNo_list)+1, Nimage, 50, 4 )) # fm.shape = (faultNo, imageNo, 50, 4 )
    for faultNo in faultNo_list:
        fault_dir = "list%d_no0-%d"%(faultNo, Nimage-1)
        assert os.path.exists(fault_dir), 'Image dir such as {} not found'.format(fault_dir)
        h = load_npz(fault_dir)
        fm[faultNo] = h
    return fm

# load normal and fault systems
nmap = load_normal(normal_dir)                  # nmap.shape = (imageNo, 50, 4)
fmap = load_faults(faultNo_list, data_P)        # fmap.shape = (faultNo, imageNo, 50, 4)
print('normal system image = %d and fault system image = %d'%(nmap.shape[0],fmap.shape[1]))
assert nmap.shape[0] >= fmap.shape[1], 'Unsufficiant normal system results'

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
print("Check sum-1:normal-falut systems:",np.sum(nmap[:data_P]),np.sum(fmap[:784][:data_P]))
print("CHeck sum-2:hm_sum              :",np.sum(hm_sum))

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


    

    
