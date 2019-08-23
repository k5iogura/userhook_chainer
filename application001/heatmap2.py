import os,sys
from   pdb import *
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import chainer
from chainer import Variable

def load_npz(gdir):
    print("loading",gdir)
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

def load_normal(gdir):
    return load_npz(gdir)

def load_faults(faultNo_list, Nimage):
    hm  = np.zeros((max(faultNo_list)+1, Nimage, 50, 4 )) # hm.shape = (faultNo, imageNo, 50, 4 )
    for faultNo in faultNo_list:
        fault_dir = "list%d_no0-%d"%(faultNo, Nimage-1)
        h = load_npz(fault_dir)
        hm[faultNo] = h
    return hm

faultNo_list = list(set([10,0,1]))              # ex. Fault-wise
faultNo_list = range(784)                       # ex. About all faults
data_P = 20
normal  = 'original_data2'
nmap = load_normal(normal)                      # nmap.shape = (imageNo, 50, 4)
fmap = load_faults(faultNo_list, data_P)        # fmap.shape = (faultNo, imageNo, 50, 4)

#jは推論する画像の番号、iは壊す範囲の左上座標のリストでの番号

hm_sum= np.zeros((50,4))

for j in range(0,data_P): #オリジナルから行列を作成  # j:number of image
    # Normal System
#    filename='C:\\Users\\neutron\\Desktop\\Python_program_forHST\\origin_data\\No_' + str(j)
#    os.chdir(filename)
#    a = np.load('lx1_Linear_out.npy') 
#    a = np.reshape(a, (50, 1))
#    b = np.load('ly2_Linear_in.npy')
#    b = np.reshape(b, (50, 1))
#    c = np.load('ly2_Linear_out.npy')
#    c = np.reshape(c, (50, 1))
#    d = np.load('lz3_Linear_in.npy')
#    d = np.reshape(d, (50, 1))
#    hm_a = np.concatenate((a, b, c, d), axis=1)
    hm_a = nmap[j]

    for i in faultNo_list:
#    for i in range(3):#オリジナルとの差分を一つの行列に足しこむ # i:fault pattern No.
        # Fault System a fault pattern
#        path = 'C:\\Users\\neutron\\Desktop\\Python_program_forHST\\list' + str(i) +'_no' +str(j)
#        os.chdir(path)
#        e = np.load('lx1_Linear_out.npy') 
#        e = np.reshape(e, (50, 1))
#        f = np.load('ly2_Linear_in.npy')
#        f = np.reshape(f, (50, 1))
#        g = np.load('ly2_Linear_out.npy')
#        g = np.reshape(g, (50, 1))
#        h = np.load('lz3_Linear_in.npy')
#        h = np.reshape(h, (50, 1))
#        hm_b = np.concatenate((e, f, g, h), axis=1)
        hm_b = fmap[i][j]
        assert hm_b.shape == hm_a.shape,"Internal Error"

        hm = hm_a-hm_b
        zettaichi = np.abs(hm)
        hm_sum = zettaichi + hm_sum


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


    

    
