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

#_, test = chainer.datasets.get_mnist()
#txs, tts = test._datasets

def infer(txs):
    print("*** infer ***")
#    a = txs[:inp]
#    print(":inp=:",inp)
#    print("gt:input number  =" ,tts[:inp])

    x = txs.reshape((-1,28,28,1))

    hook = UserHook()
    with chainer.using_config('train',False):
        with hook:
            before_softmax = model(x)
    softmax_a = F.softmax(before_softmax)

    after_softmax = np.argmax(softmax_a.data, axis=1)
#    print("pr:answer number =",ans)
    return before_softmax, after_softmax

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-i','--images',type=int,default=len(tts))
    args = args.parse_args()
    var.n = -1 # No fault injection for Normal System case
    infer(inp=args.images)
    copy_tree('dnn_params', 'original_data2')
