import chainer
import numpy as np
import chainer.computational_graph as c
import csv
from chainer import serializers
from userhook import UserHook
import chainer.functions as F

from train import *
from pdb import *

model=NeuralNet(50,10)
serializers.load_npz('mnist.npz',model)

_, test = chainer.datasets.get_mnist()
txs, tts = test._datasets

inp=0
a = txs[inp]
print("inp=",inp)
print("input number = " ,tts[inp])

x = txs[inp].reshape((1,28,28,1))

hook = UserHook()
with chainer.using_config('train',False):
    with hook:
        p = model(x)
softmax_a = F.softmax(p)

ans = F.argmax(softmax_a)
idx = int(ans.data)
print("answer =",idx)
