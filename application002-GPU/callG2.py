import cupy # For GPU
import chainer
from distutils.dir_util import copy_tree
from shutil import rmtree
import os,sys,argparse

import sampleG2 as sample2
from userfunc_var import VAR
var = VAR()

args = argparse.ArgumentParser()
args.add_argument('-i','--images',type=int,default=20)
args.add_argument('-g','--gpu',   type=int,default=0)
args.add_argument('-f','--faults',type=int,default=784, dest='faults')
args.add_argument('-F','--Faults',type=int,nargs='+')
args = args.parse_args()

# For GPU
device = chainer.get_device(args.gpu)
assert '@cupy' in str(device),"Supports Only GPU"
print('GPU device is ',device)
device.use()
sample2.model.to_device(device) # load model to GPU
txs = cupy.asarray(sample2.txs) # load data  to GPU

faultNo_list = args.Faults if args.Faults is not None else range(args.faults)
data_P       = args.images
print("Run on Faults as",faultNo_list)

from mylist_block import mylist

d=mylist()
for k in faultNo_list:

    var.n = k

    print("mylist=",d[k],k)   
    sample2.infer(data_P, sample2.model, txs, sample2.tts)
    print('sample done')
    rename =  "list%d"%(k)
    rmtree(rename, ignore_errors=True)
    copy_tree("dnn_params", rename)

