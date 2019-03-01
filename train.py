import numpy as np
import chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from pdb import *

from runtimeviz import runtimeviz
rtv = runtimeviz()

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
#        rtv.regist_var('h1', h1)
#        rtv.regist_var('h2', h2)
#        rtv.regist_var('h3', h3)
#        rtv.list()
#        rtv.regist_end()
        return h3

def check_accuracy(model, xs, ts):
    ys = model(xs)
    loss = F.softmax_cross_entropy(ys, ts)
    ys = np.argmax(ys.data, axis=1)
    cors = (ys == ts)
    num_cors = sum(cors)
    accuracy = num_cors / ts.shape[0]
    return accuracy, loss

def main():
    model = NeuralNet(50, 10)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist()
    xs, ts = train._datasets
    txs, tts = test._datasets

    bm = 100
    bm = 10

    #for i in range(100):
    for i in range(10):

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

        accuracy_train, loss_train = check_accuracy(model, xs, ts)
        accuracy_test, _           = check_accuracy(model, txs, tts)

        print("Epoch %d loss(train) = %f, accuracy(train) = %f, accuracy(test) = %f" % (i + 1, loss_train.data, accuracy_train, accuracy_test))
    serializers.save_npz('mnist.npz',model)

if __name__ == '__main__':
    main()

