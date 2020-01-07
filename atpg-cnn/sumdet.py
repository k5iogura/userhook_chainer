#! /usr/bin/env python3
import os,sys,argparse
import numpy as np
from   pdb import set_trace

def countup(npy):
    layer_idx, node_idx, bit_idx, sa_idx = (0,1,2,3)

  #  layer_stat = [0]*4
    layer_stat = [0]*(np.max(npy[:,layer_idx])+1)
    for layer in npy[:,layer_idx]:
        layer_stat[layer]+=1

    sa_stat = [0]*2
    for sa in npy[:,sa_idx]:
        sa_stat[sa]+=1

    bit_stat = [0]*32
    for bit in npy[:,bit_idx]:
        bit_stat[bit]+=1

#    for layer,node in zip(npy[:,0], npy[:,1]):
#        layer_node_stat[layer,node]+=1

    print(layer_stat)
    print(bit_stat)
    print(sa_stat)

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('file',type=str)
    args = args.parse_args()

    formatA = 0
    print('Loading',args.file)
    npy = np.load(args.file)
    faultN, items = npy.shape
    if items == 3: formatA = 1
    if formatA == 1:
        npy = npy[:,0]

    status = countup(npy)
