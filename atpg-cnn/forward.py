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

model=CNN()
serializers.load_npz('mnist.npz',model)

def infer(txs):

    x = txs.transpose((0, 3, 1, 2)) # BHWC -> BCHW

    hook = UserHook()
    with chainer.using_config('train',False):
        with hook:
            before_softmax = model(x)
    softmax_a = F.softmax(before_softmax)

    after_softmax = np.argmax(softmax_a.data, axis=1)
    return before_softmax, after_softmax

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-i','--images',type=int,default=10)
    args = args.parse_args()

    # loading dataset
    _, test = chainer.datasets.get_mnist()
    txs, tts = test._datasets

    var.n = -1 # No fault injection for Normal System case
    txs_tmp = txs[:args.images].reshape(-1,28,28,1)
    before_softmax, after_softmax = infer(txs=txs_tmp)

    print('GroundTruth:',tts[:args.images])
    print('Inferenced :',after_softmax)
