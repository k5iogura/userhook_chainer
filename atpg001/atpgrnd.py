from pdb import set_trace
import os,sys,argparse
assert sys.version_info.major >= 3, 'Use over python3 version but now in {}'.format(sys.version_info)

import numpy as np

import forward
from   userfunc_var import *
from   random import seed, random, randrange

var = VAR()

args = argparse.ArgumentParser()
args.add_argument('-l','--layerNo',  type=int,default=None)
args.add_argument('-L','--layerList',type=int,nargs='+')
args.add_argument('-n','--nodeNo',   type=int,default=None)
args.add_argument('-N','--nodeList', type=int,nargs='+')
args.add_argument('-b','--bitNo',    type=int,default=None)
args.add_argument('-B','--bitList',  type=int,nargs='+')
args.add_argument('-s','--sa',       type=int,default=None)
args.add_argument('-S','--saList',   type=int,nargs='+')
args = args.parse_args()

# Generate fault list
seed(2222222222)
net_spec=(28*28, 50, 50, 10)
if args.layerNo is not None:
    if   args.layerNo == 0: net_spec=(28*28, 0, 0, 0)
    elif args.layerNo == 1: net_spec=(0, 50, 0, 0)
    elif args.layerNo == 2: net_spec=(0, 0, 50, 0)
    elif args.layerNo == 3: net_spec=(0, 0, 0, 10)
#var.init(Batch=1024, Net_spec=net_spec)
var.init(Batch=1024, Net_spec=(1,1,1,1))

# random number generators for int32 and float32
# python random function generates 53bit float random number by Mersenne twister
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

# fault diff function
# Notice: Can not use xor operator for float32 type
def faultDiff(A,B):
    assert len(A.reshape(-1))==len(B.reshape(-1)),'Mismatch length btn A and B'
    diff = [ I==J for I,J in zip(A.reshape(-1),B.reshape(-1)) ]
    return np.asarray(diff).reshape(A.shape)

# Generating float32 patterns at random
# Calculate inference result of Before or After of SoftMax
# Notice!:
#   B(b)eforeSMax type is chainer.variable.Variable
#   A(a)fterSMax  type is numpy.ndarray
print('* Generating Test Pattern with batch ',var.batch)
Test_Patterns = GenRndPatFloat32(var.batch)
print('* Generating Expected value of normal system')
var.n = -1  # For normal system inference
BeforeSMax, AfterSMax = forward.infer(Test_Patterns)

print('* Fault Point insertion and varify')
fault_injection_table = []
all_detects = 0
while True:
    detects = 0
    for var.n, spec in enumerate(var.faultpat):

        # spec: [0]detect_flag [1]layer [2]node [3]bit [4]sa01
        (detect_flag_idx, layer_idx, node_idx, bit_idx, sa01_idx) = (0, 1, 2, 3, 4)
        if spec[0]: continue

        # For fault system inference
        beforeSMax, afterSMax = forward.infer(Test_Patterns)

        # Calculate fault differencial function
        diffA = faultDiff(AfterSMax,  afterSMax)
        diffB = faultDiff(BeforeSMax.data, beforeSMax.data)
        diff  = ~diffB  # True : propagated fault / False : disappearance fault
                        # diff.shape : ( batch, output_nodes )

        # Choice test pattern to detect fault point
        if diff.any():  # detected
            var.faultpat[var.n][detect_flag_idx]=True
            detPtNo = np.where(diff)[0][0]
            fault_injection_table.append( [ spec, Test_Patterns[detPtNo], BeforeSMax[detPtNo] ] )
            detects += 1
            print('* detect fault faultNo={:6d} detPtNo={:6d} detects={:6d} spec={}'.format(
                var.n, detPtNo, detects, spec[1:]))
        elif 0: # inserted fault disappeared, discard patterns
            print('* Matched fault insertion run and normal system run, Discard')

    if detects>0: # Create new random patterns
        all_detects += detects
        Test_Patterns = GenRndPatFloat32(var.batch)
        print('* Created New Test pattern')
        print('* Detected fault points {}/{}/{}'.format(detects, all_detects, var.faultN))
    else:
        break

#    break   # for debugging
print('* Summary for Detected fault points {}/{}'.format(all_detects, var.faultN))
print('* End of Flow')

