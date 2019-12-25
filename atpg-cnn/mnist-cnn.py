# コマンドライン引数を解析するライブラリを読み込む
import argparse

# pythonの数値計算用ライブラリを読み込む
import numpy as np

# chainerを読み込む
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.dataset import convert

#　ニューラルネットワークの定義
class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 32, 5)   # 入力次元1,  出力次元32, 畳み込みカーネルサイズ5
            self.conv2 = L.Convolution2D(32, 64, 5) # 入力次元32, 出力次元64, 畳み込みカーネルサイズ5

            self.bn1 = L.BatchNormalization(32)
            self.bn2 = L.BatchNormalization(64)

            self.l1 = L.Linear(None, 300)
            self.l2 = L.Linear(None, 10)

    def __call__(self, x):
        y1 = F.max_pooling_2d(self.bn1(F.relu(self.conv1(x))), 2)
        y2 = F.max_pooling_2d(self.bn2(F.relu(self.conv2(y1))), 2)
        y3 = F.relu(self.l1(y2))
        return self.l2(y3)

def main():
    # コマンドライン引数の読み込み
    parser = argparse.ArgumentParser(description='Chainer MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=20, help='バッチサイズ(デフォルトは20)')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='エポック数(デフォルトは20)')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID')
    parser.add_argument('--out', '-o', default='result', help='出力フォルダ名')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # modelを読み込む
    model = L.Classifier(CNN(), lossfun=F.softmax_cross_entropy)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # モデルをGPU用にする

    # Optimizerの設定
    # Adamを用いる
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # MNISTのデータセットを読み込む
    train, test = chainer.datasets.get_mnist(ndim=3)

    # iteratorの設定。batchsize等を設定する
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False) # こちらはテスト用なのでリピートやシャッフルは必要ない

    # updater/trainerの設定
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # テストデータを用いて評価する
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # ネットワークモデルを図として出力する。出力形式はdotファイル
    trainer.extend(extensions.dump_graph('main/loss'))

    # Logの出力
    trainer.extend(extensions.LogReport())

    # 学習グラフの保存
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))

    # 学習結果をコンソールに出力する
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # プログレスバーの表示
    trainer.extend(extensions.ProgressBar())

    # 学習を始める
    trainer.run()

if __name__ == '__main__':
    main()
