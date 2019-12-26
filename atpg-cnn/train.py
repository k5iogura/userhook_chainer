import numpy as np
import chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from random import random, sample, randint
from pdb import *

class NeuralNet(chainer.Chain):
    def __init__(self, n_units, n_out):
        super().__init__(
            lx1=L.Linear(None, n_units),
            ly2=L.Linear(n_units, n_units),
            lz3=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.lx1(x))
        h2 = F.relu(self.ly2(h1))
        h3 = self.lz3(h2)
        return h3

class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 32, 5)  # 28x28x1 ,s1,k5 -> 28x28x32
            self.conv2 = L.Convolution2D(32, 64, 5) # 14x14x32,s1,k5 -> 14x14x64

            self.bn1 = L.BatchNormalization(32)
            self.bn2 = L.BatchNormalization(64)

            self.l1 = L.Linear(None, 300)           # 7x7x64 -> 300
            self.l2 = L.Linear(None, 10)            # 300 -> 100

    def __call__(self, x):
        y1 = F.max_pooling_2d(self.bn1(F.relu(self.conv1(x))), 2)
        y2 = F.max_pooling_2d(self.bn2(F.relu(self.conv2(y1))), 2)
        y3 = F.relu(self.l1(y2))
        y4 = self.l2(y3)
        return y4

def check_accuracy(model, xs, ts):
    ys = model(xs)
    loss = F.softmax_cross_entropy(ys, ts)
    ys = np.argmax(ys.data, axis=1)
    cors = (ys == ts)
    num_cors = sum(cors)
    accuracy = num_cors / ts.shape[0]
    set_trace()
    return accuracy, loss

def main():
    model = CNN()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist()
    xs, ts   = train._datasets
    txs, tts = test._datasets
    xs  = xs.reshape(-1,1,28,28) # BCHW
    txs = txs.reshape(-1,1,28,28) # BCHW

    bm = 100
#    bm = 10

    #for i in range(100):
    for i in range(20):

        #for j in range(600):
        for j in range(60):
            model.zerograds()
            x = xs[(j * bm):((j + 1) * bm)]
            t = ts[(j * bm):((j + 1) * bm)]
            t = Variable(np.array(t, "i"))
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            loss.backward()
            optimizer.update()

        i_xs  = sample(range(xs.shape[0] ),1000)
        i_txs = sample(range(txs.shape[0]),1000)
        accuracy_train, loss_train = check_accuracy(model, xs[i_xs],   ts[i_xs]  )
        accuracy_test, _           = check_accuracy(model, txs[i_txs], tts[i_txs])

        print(
            "Epoch {:4d} loss(train) = {:8.4f}, accuracy(train) = {:8.4f}, accuracy(test) = {:8.4f}".format(
                i + 1, loss_train.data, accuracy_train, accuracy_test)
        )
    serializers.save_npz('mnist.npz',model)

if __name__ == '__main__':
    main()

