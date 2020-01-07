#! /usr/bin/env python3
import os,sys,argparse
import numpy as np
from   pdb import set_trace

layer_idx, node_idx, bit_idx, sa_idx = (0,1,2,3)
def countup(npy):

    layer_stat = [0]*(np.max(npy[:,layer_idx])+1)
    for layer in npy[:,layer_idx]:
        layer_stat[layer]+=1

    #sa_stat = [0]*2
    sa_stat = [0]*(np.max(npy[:,sa_idx])+1)
    for sa in npy[:,sa_idx]:
        sa_stat[sa]+=1

    #bit_stat = [0]*32
    bit_stat = [0]*(np.max(npy[:,bit_idx])+1)
    for bit in npy[:,bit_idx]:
        bit_stat[bit]+=1

    print('layer:',layer_stat)
    print('bit  :',bit_stat)
    print('sa01 :',sa_stat)

editmode=False
def ARG(x):
    global editmode
    editmode=True
    assert x.isnumeric(), '{} {}'.format(type(x),x)
    return int(x)

def search_index(np_src, np0):
    _idx = -1
    try: _idx = np_src.tolist().index(np0.tolist())
    except: pass
    return _idx

def yes_or_else():
    while True:
        choice = input("Please respond with 'yes' or 'no' [y/N]: ").lower()
        if choice in ['y', 'ye', 'yes']: return True
        else: return False

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('file',type=str)

    args.add_argument('-l','--layerNo',   type=ARG,  default=None)
    args.add_argument('-n','--nodeNo',    type=ARG,  default=None)
    args.add_argument('-b','--bitNo',     type=ARG,  default=None)
    args.add_argument('-s','--sa',        type=ARG,  default=None)
    args.add_argument('-v','--verbose',   action='store_true')

    group = args.add_mutually_exclusive_group()
    group.add_argument('-d','--delete',        action='store_true')
    group.add_argument('-di','--delete_index', type=int, default=None)

    args = args.parse_args()

    if args.delete:
        if editmode is False:
            assert False,'Specify fault point spec with -l/-n/-b/-s option at delete mode'

    updated_npy = False
    if editmode:
        npy0 = np.zeros((4), dtype=np.int32)
        npy0[layer_idx] = args.layerNo if args.layerNo is not None else 0
        npy0[node_idx]  = args.nodeNo  if args.nodeNo  is not None else 0
        npy0[bit_idx]   = args.bitNo   if args.bitNo   is not None else 0
        npy0[sa_idx]    = args.sa      if args.sa      is not None else 0
        print('Specified fault point:',npy0)
    else:
        npy0 = np.full((4),-1, dtype=np.int32)

    if args.delete_index is not None:
        editmode = args.delete = True

    if os.path.exists(args.file):               # loading exist npy file
        print('Loading',args.file)
        npy = np.load(args.file)
        assert len(npy.shape)==2, 'Unsupported file format {}'.format(npy.shape)
        faultN, items = npy.shape

    if os.path.exists(args.file) and editmode:
        _idx = search_index(npy, npy0)
        if args.delete is False and _idx<0:     # add specified item at the end
            npy1 = np.zeros((faultN+1, items), dtype=np.int32)
            for i in range(faultN): npy1[i] = npy[i]
            npy1[-1] = npy0
            npy = npy1
            updated_npy = True

        elif args.delete is False and _idx>=0:  # already existed
            print(npy0,'already exist in file',args.file,'index',_idx,',nothing to do')

        elif args.delete is True:               # delete specified item
            if args.delete_index is not None: _idx = args.delete_index
            if _idx >= faultN: print('specified index {} is out of file size {}'.format(_idx, faultN))
            if _idx >= 0 and _idx < faultN:
                npy0 = npy[_idx]
                print('delete index',_idx,npy[_idx],'of',args.file,'?')
                if yes_or_else():
                    lst1 = npy.tolist()
                    lst1.remove(npy0.tolist())
                    npy  = np.asarray(lst1)
                    updated_npy = True
                else:
                    print('Nothing to delete!', _idx)

    elif editmode and args.delete is False:
        npy = npy0.reshape(1,-1)    # create new table
        updated_npy = True

    if updated_npy:
        if os.path.exists(args.file):
            print('Update file', args.file, '?')
            if yes_or_else():
                print('Update file',args.file)   # create new file
                np.save(args.file, npy) # update existed file
            else: print('Nothing to update!')
        else:
            print('Creating new file as',args.file)   # create new file
            np.save(args.file, npy)

    if args.verbose:
        print('* verbose',args.file,'all items *')
        for i,item in enumerate(npy):
            print('{:4d} : {}'.format(i,item))
    print('* summary',args.file,'*')
    status = countup(npy)

