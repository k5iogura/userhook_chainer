from pdb import set_trace
import os,sys,argparse
assert sys.version_info.major >= 3, 'Use over python3 version but now in {}'.format(sys.version_info)

import numpy as np

import forward
from   userfunc_var import VAR
from   random import seed, random, randrange

var = VAR()

args = argparse.ArgumentParser()
args.add_argument('-i','--images',type=int,default=20)
args.add_argument('-f','--faults',type=int,default=784, dest='faults')
args.add_argument('-F','--Faults',type=int,nargs='+')
args = args.parse_args()

# random number generators for int32 and float32
# python random function generates 53bit float random number by Mersenne twister
seed(2222222222)
def GenRndPatFloat32(batch, img_hw=28, img_ch=1):
    maxf32 = np.finfo(np.float32).max
    minf32 = np.finfo(np.float32).min
    randpat = []
    for b in range(batch):
        #rdata = random() * 255.
        rdata = random() * 1.
        randpat.append([np.clip(rdata,minf32,maxf32) for i in range(pow(img_hw,2)*img_ch)])
    return np.asarray(randpat, dtype=np.float32).reshape(-1, img_hw, img_hw, img_ch)

def GenRndPatInt32(batch, img_hw=28, img_ch=1):
    maxint32 = np.iinfo(np.int32).max
    minint32 = np.iinfo(np.int32).min
    randpat = []
    for b in range(batch):
        randpat.append([randrange(minint32, maxint32) for i in range(pow(img_hw,2)*img_ch)])
    return np.asarray(randpat, dtype=np.int32).reshape(-1,img_hw,img_hw,img_ch)

#faultNo_list = args.Faults if args.Faults is not None else range(args.faults)
#data_P       = args.images
#print("Run on Faults as",faultNo_list)

# fault diff function
# Notice: Can not use xor operator for float32 type
def faultDiff(A,B):
    diff = [ I==J for I,J in zip(A.reshape(-1),B.reshape(-1)) ]
    return np.asarray(diff).reshape(A.shape)

# Generating float32 patterns at random
# Inference result of Before or After of SoftMax
# Notice!:
#   B(b)eforeSMax type is chainer.variable.Variable
#   A(a)fterSMax  type is numpy.ndarray
var.batch = 1024
print('* Generating Test Pattern with batch ',var.batch)
Test_Patterns = GenRndPatFloat32(var.batch)
print('* Generating Expected value of normal system')
var.n = -1  # For normal system inference
BeforeSMax, AfterSMax = forward.infer(Test_Patterns)

print('* Fault Point insertion and varify')
fault_injection_table = []
while True:
    detects = 0
    for k, spec in enumerate(var.faultpat):

        # spec: [0]detect_flag [1]layer [2]node [3]bit [4]sa01
        (detect_flag_idx, layer_idx, node_idx, bit_idx, sa01_idx) = (0, 1, 2, 3, 4)
        if spec[0]: continue

        var.n = k
#        var.n = -1 # for debugging

        #print("{:8d} faultpattern={}".format(k, spec))
        beforeSMax, afterSMax = forward.infer(Test_Patterns)
        diffA = faultDiff(AfterSMax,  afterSMax)
        diffB = faultDiff(BeforeSMax.data, beforeSMax.data)
        diff  = diffB
        #if diff.all():      # not detected
        if not diff.all():  # detected
            print('* detect fault', spec[1:])
            var.faultpat[k][detect_flag_idx]=True
            detPtNo = np.where(diff)[0][0]
            fault_injection_table.append( [ spec, Test_Patterns[detPtNo], BeforeSMax[detPtNo] ] )
            detects += 1
            break
        else: # discard patterns
            print('* Matched fault insertion run and normal system run, Discard')

        if k==7000: break   # for debugging
    break   # for debugging

