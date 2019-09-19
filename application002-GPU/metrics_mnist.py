from distutils.dir_util import copy_tree
from shutil import rmtree
import os,sys,argparse
import numpy as np
import chainer.functions as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

import sample2
tsx, tts = sample2.txs, sample2.tts
data_N   = len(tts)
from userfunc_var import VAR
var = VAR()

args = argparse.ArgumentParser()
args.add_argument('-i','--images',type=int,default=20)
args.add_argument('-f','--faults',type=int,default=784, dest='faults')
args.add_argument('-F','--Faults',type=int,nargs='+')
args = args.parse_args()

faultNo_list = args.Faults if args.Faults is not None else range(args.faults)
data_P       = args.images
print("* Run on Faults as",faultNo_list)

# loadin from .npy
def load_npz(gdir, filename='lz3_Linear_out.npy', n_units=10):
    assert os.path.exists(gdir), "Not found `{}` directory".format(gdir)
    cwd = os.getcwd()
    os.chdir(gdir)
    d = np.load(filename)
    d = np.reshape(d, (-1, n_units))
    os.chdir(cwd)
    return d

# normal := ( imageNo, prediction )
# faults := ( faultNo, imageNo, prediction )
normal = load_npz('original_data2')
buffers= []
for k in faultNo_list:
    list_dir  = "list%d"%(k)
    list_file = "lz3_Linear_out.npy"
    buffers.append( load_npz(list_dir, list_file) )
faults = np.asarray(buffers)
Nfaults, Nimages, Nclass = faults.shape
normal = normal[:Nimages]
print("* faults patterns = %d Images = %d class = %d"%(Nfaults, Nimages, Nclass))

# pred_normal := ( imageNo )
# pred_faults := ( faultNo, imageNo )
pred_normal = np.argmax(F.softmax(normal,axis=1).data,axis=1)
pred_faults = np.argmax(F.softmax(faults,axis=2).data,axis=2)

# truth := ( imageNo )
# diffs := ( imageNo )
truth = tts[:Nimages]
diffs = truth - pred_normal

# estimation of normal system
conf_normal = confusion_matrix(truth, pred_normal, labels=[i for i in range(Nclass)])
assert np.sum(conf_normal)==truth.shape[0]
print("normal system confusion matrix")
print(conf_normal)
print("normal system accuracy = %.9f"%(accuracy_score(truth, pred_normal)))

# estimation of fault system
faultX = pred_faults.reshape(-1)
truthX = np.asarray(list(truth)*pred_faults.shape[0])

conf_fault = confusion_matrix(truthX, faultX, labels=[i for i in range(Nclass)])
assert np.sum(conf_fault)==truthX.shape[0]
print("fault system confusion matrix")
print(conf_fault)
print("fault system accuracy = %.9f"%(accuracy_score(truthX, faultX)))


