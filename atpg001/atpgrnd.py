import numpy as np
from distutils.dir_util import copy_tree
from shutil import rmtree
import os,sys,argparse
from pdb import set_trace

import sample2
from userfunc_var import VAR
from random import seed, random, randrange

var = VAR()

args = argparse.ArgumentParser()
args.add_argument('-i','--images',type=int,default=20)
args.add_argument('-f','--faults',type=int,default=784, dest='faults')
args.add_argument('-F','--Faults',type=int,nargs='+')
args = args.parse_args()

# random number generators for int32 and float32
# python random function generates 53bit float random number by Mersenne twister
seed(2222222222)
def GenRndPatFloat32(batch, img_hw=28, img_ch=1):
    maxf32 = np.finfo(np.float32).max
    minf32 = np.finfo(np.float32).min
    randpat = []
    for b in range(batch):
        randpat.append([np.clip(random(),minf32,maxf32) for i in range(pow(img_hw,2)*img_ch)])
    return np.asarray(randpat, dtype=np.float32).reshape(-1, img_hw, img_hw, img_ch)

def GenRndPatInt32(batch, img_hw=28, img_ch=1):
    maxint32 = np.iinfo(np.int32).max
    minint32 = np.iinfo(np.int32).min
    randpat = []
    for b in range(batch):
        randpat.append([randrange(minint32, maxint32) for i in range(pow(img_hw,2)*img_ch)])
    return np.asarray(randpat, dtype=np.int32).reshape(-1,img_hw,img_hw,img_ch)

faultNo_list = args.Faults if args.Faults is not None else range(args.faults)
data_P       = args.images
print("Run on Faults as",faultNo_list)

#from mylist_block import mylist

#d=mylist()
for k, spec in enumerate(var.faultpat):

    # spec: [0]detect_flag [1]layer [2]node [3]bit [4]sa01
    if spec[0]: continue

    var.n = k

    print("{:8d} faultpattern={}".format(k, spec))
    break
    sample2.infer(data_P)
#    print('sample done')
#    rename =  "list%d"%(k)
#    rmtree(rename, ignore_errors=True)
#    copy_tree("dnn_params", rename)
