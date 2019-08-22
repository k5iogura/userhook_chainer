import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import chainer
from chainer import Variable

#jは推論する画像の番号、iは壊す範囲の左上座標のリストでの番号

hm_sum= np.zeros((50,4))

for j in range(0,2): #オリジナルから行列を作成
    filename='C:\\Users\\neutron\\Desktop\\Python_program_forHST\\origin_data\\No_' + str(j)
    os.chdir(filename)
    
    a = np.load('lx1_Linear_out.npy') 
    a = np.reshape(a, (50, 1))

    b = np.load('ly2_Linear_in.npy')
    b = np.reshape(b, (50, 1))
    c = np.load('ly2_Linear_out.npy')
    c = np.reshape(c, (50, 1))
    d = np.load('lz3_Linear_in.npy')
    d = np.reshape(d, (50, 1))
    hm_a = np.concatenate((a, b, c, d), axis=1)



    for i in range(3):#オリジナルとの差分を一つの行列に足しこむ

        path = 'C:\\Users\\neutron\\Desktop\\Python_program_forHST\\list' + str(i) +'_no' +str(j)

        os.chdir(path)

        e = np.load('lx1_Linear_out.npy') 
        e = np.reshape(e, (50, 1))
        f = np.load('ly2_Linear_in.npy')
        f = np.reshape(f, (50, 1))
        g = np.load('ly2_Linear_out.npy')
        g = np.reshape(g, (50, 1))
        h = np.load('lz3_Linear_in.npy')
        h = np.reshape(h, (50, 1))

        hm_b = np.concatenate((e, f, g, h), axis=1)
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


    

    
