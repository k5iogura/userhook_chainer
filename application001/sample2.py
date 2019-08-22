import chainer
import numpy as np
import chainer.computational_graph as c
import csv
from chainer import serializers
from userhook import UserHook
import chainer.functions as F

from train import *
from pdb import *

from userfunc_var import VAR
var=VAR()

model=NeuralNet(50,10)
serializers.load_npz('mnist.npz',model)

_, test = chainer.datasets.get_mnist()
txs, tts = test._datasets

def infer(inp):
    print("*** infer ***")
#    inp=0
    a = txs[:inp]
    print("inp=",inp)
    print("gt:input number = " ,tts[:inp])

    x = txs[:inp].reshape((-1,28,28,1))

    hook = UserHook()
    with chainer.using_config('train',False):
        with hook:
            p = model(x)
    softmax_a = F.softmax(p)

    ans = np.argmax(softmax_a.data, axis=1)
    #idx = int(ans.data)
    print("pr:answer =",ans)

if __name__ == '__main__':
    var.n = -1 # No fault injection for Normal System case
    infer(inp=len(tts))
