import argparse
import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import training
from chainer.training import extensions, triggers
from chainer.dataset import convert

# Define mnist-cnn network
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
        return self.l2(y3)

def main():
    # コマンドライン引数の読み込み
    parser = argparse.ArgumentParser(description='Chainer MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=20, help='Batch size')
    parser.add_argument('--epoch'    , '-e', type=int, default=20, help='Epoch')
    parser.add_argument('--gpu'      , '-g', type=int, default=-1, help='GPU ID')
    parser.add_argument('--out'      , '-o', default='result', help='output directory')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # reading model
    model = L.Classifier(CNN(), lossfun=F.softmax_cross_entropy)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # adam optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # loading MNIST dataset
    train, test = chainer.datasets.get_mnist(ndim=3)

    # Iterator of dataset with Batchsize
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize) # for training
    test_iter  = chainer.iterators.SerialIterator(test,  args.batchsize, repeat=False, shuffle=False)

    # updater/trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # setup evaluator
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # plotting mnist-cnn network
    trainer.extend(extensions.dump_graph('main/loss'))

    # Reporting
    # setup log
    trainer.extend(extensions.LogReport())

    # progress plot
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png')
        )
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png')
        )

    # progress console
    trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time'])
        )

    # Saving at updated test-accuracy
    trigger = triggers.MaxValueTrigger('validation/main/accuracy', trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, filename='mnist-cnn-best'), trigger=trigger)

    # progress bar
    trainer.extend(extensions.ProgressBar())

    # Training
    trainer.run()

    # Saving model final
    serializers.save_npz('mnist-cnn.npz', model)

if __name__ == '__main__':
    main()
