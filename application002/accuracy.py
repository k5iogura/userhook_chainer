from distutils.dir_util import copy_tree
from shutil import rmtree
import os,sys,argparse
import numpy as np
import chainer.functions as F

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
    list_dir  = "list%d_no%d-%d"%(k, 0, data_P-1)
    list_file = "lz3_Linear_out.npy"
    buffers.append( load_npz(list_dir, list_file) )
faults = np.asarray(buffers)
Nfaults, Nimages, Npred = faults.shape
normal = normal[:Nimages]
print("* faults patterns = %d Images = %d Predictions = %d"%(Nfaults, Nimages, Npred))

# pred_normal := ( imageNo )
# pred_faults := ( faultNo, imageNo )
pred_normal = np.argmax(F.softmax(normal,axis=1).data,axis=1)
pred_faults = np.argmax(F.softmax(faults,axis=2).data,axis=2)

# truth := ( imageNo )
# diffs := ( imageNo )
truth = tts[:Nimages]
diffs = truth - pred_normal
diffs[diffs!=0] = 1
error = 1.0*np.sum(diffs)/np.prod(pred_normal.shape)
print("normal system acc=",1.0-error)

diffs = np.zeros(Nimages,dtype=np.int)
for f in range(Nfaults):
    diff = truth - pred_faults[f]
    diffs[np.where(diff!=0)[0]] += 1
error = 1.0*np.sum(diffs)/np.prod(pred_faults.shape)
print("faults system acc=",1.0-error)

