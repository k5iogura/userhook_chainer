# For GPU
try:
    import cupy
    print("call3:import cupy:OK")
except:pass
from distutils.dir_util import copy_tree
from shutil import rmtree
import os,sys,argparse

from train import *
import sample3 as sample
from userfunc_var import VAR
var = VAR()

args = argparse.ArgumentParser()
args.add_argument('-i','--images',type=int,default=20)
args.add_argument('-g','--gpu',type=int,default=-1)
args.add_argument('-f','--faults',type=int,default=784, dest='faults')
args.add_argument('-F','--Faults',type=int,nargs='+')
args = args.parse_args()

#var.device=args.gpu
#import sample3 as sample

# load model and dataset
model=NeuralNet(50,10)
serializers.load_npz('mnist.npz',model)
txs, tts = sample.load_ds()

# Select GPU/CPU
device = chainer.get_device(args.gpu)
if '@cupy' in str(device):
    var.device=args.gpu
    # Send model and data to GPU Memory
    print('GPU device is ',device)
    model.to_device(device)
    device.use()
    txs = cupy.asarray(txs)
else:
    print('Device is CPU',device)

# Setup option
faultNo_list = args.Faults if args.Faults is not None else range(args.faults)
data_P       = args.images
print("Run on Faults as",faultNo_list)

from mylist_block import mylist

d=mylist()
for k in faultNo_list:

    # Send n to userfunc
    var.n = k

    print("mylist=",d[k],k)   
    sample.infer(data_P, model, txs, tts)
    print('sample done')
    rename =  "list%d"%(k)
    rmtree(rename, ignore_errors=True)
    copy_tree("dnn_params", rename)

