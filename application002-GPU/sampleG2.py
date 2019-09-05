import cupy # For GPU
import chainer
import numpy as np
import chainer.computational_graph as c
import sys,os,argparse
from chainer import serializers
from userhook import UserHook
import chainer.functions as F

from train import *
from distutils.dir_util import copy_tree
from pdb import *

from userfunc_var import VAR
var=VAR()

model=NeuralNet(50,10)
serializers.load_npz('mnist.npz',model)

_, test = chainer.datasets.get_mnist()
txs, tts = test._datasets

def infer(inp, model, txs, tts):
    print("*** infer ***")
#    inp=0
    a = txs[:inp]
    print(":inp=:",inp)
    print("gt:input number  =" ,tts[:inp])

    x = txs[:inp].reshape((-1,28,28,1))

    hook = UserHook()
    with chainer.using_config('train',False):
        with hook:
            p = model(x)
    softmax_a = F.softmax(p)

    ans = np.argmax(softmax_a.data, axis=1)
    #idx = int(ans.data)
    print("pr:answer number =",ans)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-i','--images',type=int,default=20)
    args.add_argument('-g','--gpu',type=int,default=0)
    args = args.parse_args()

    # For GPU
    device = chainer.get_device(args.gpu)
    assert '@cupy' in str(device),"Supports Only GPU"
    print('GPU device is ',device)
    device.use()
    model.to_device(device) # load model to GPU
    txs = cupy.asarray(txs) # load data  to GPU

    var.n = -1 # No fault injection for Normal System case
    infer(args.images, model, txs, tts)
    copy_tree('dnn_params', 'original_data2')
