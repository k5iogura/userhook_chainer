import warnings
warnings.simplefilter("ignore")
from pdb import set_trace
import os,sys,argparse
assert sys.version_info.major >= 3, 'Use over python3 version but now in {}'.format(sys.version_info)

import numpy as np

import forward
from   userfunc_var import *
from   userfunc import __f2i_union
from   random import seed, random, randint

var = VAR()

args = argparse.ArgumentParser()
args.add_argument('-l','--layerNo',   type=int,  default=None)
args.add_argument('-D','--debug',     action='store_true')

args.add_argument('-L','--layerList', type=int,  nargs='+')
args.add_argument('-n','--nodeNo',    type=int,  default=None)
args.add_argument('-b','--bitNo',     type=int,  default=None)
args.add_argument('-s','--sa',        type=int,  default=None)

args.add_argument('-t','--targetFile',type=str,  default=None, help='fault simulation target file')
args.add_argument('-u','--ud_list',   type=str,  default='ud_list')
args.add_argument('-d','--dt_list',   type=str,  default='dt_list')
args.add_argument('--batch',          type=int,  default=1024)
args.add_argument('--seed',           type=int,  default=2222222222)
args.add_argument('--upper8bit',      type=int,  default=1,   help='specify as %')
args.add_argument('--onehot',         type=int,  default=784, help='onehot patterns')
args.add_argument('--pos_only',       action='store_true')
args.add_argument('-r','--randmax',   type=float,default=255.0)
args = args.parse_args()

# add heuristic patterns
args.batch += args.onehot

# << Generate fault list >>
seed(args.seed)
net_spec=(28*28, 50, 50, 10)
if args.layerNo is not None:
    if   args.layerNo == 0: net_spec=(28*28, 0, 0, 0)
    elif args.layerNo == 1: net_spec=(0, 50, 0, 0)
    elif args.layerNo == 2: net_spec=(0, 0, 50, 0)
    elif args.layerNo == 3: net_spec=(0, 0, 0, 10)
var.init(Batch=args.batch, Net_spec=net_spec, target=args.targetFile)
#var.init(Batch=1024, Net_spec=(5,5,2,2))

# << random number generators for int32 and float32 >>
# python random function generates 53bit float random number by Mersenne twister
def RxX(X, pos_only, u8b):

    rndV   = X*random()     # generate a value at random

    rndV = __f2i_union(rndV)
    if randint(1,100)<=u8b: # make upper 8bit at random
        #rndV.uint = rndV.uint | np.uint32(np.uint8(randint(0,255))<<24)
        rndV.uint = rndV.uint | np.uint32(randint(0,0x0f)<<27)
        #rndV.uint = rndV.uint | np.uint32(randint(0,0xff)<<24)
        #rndV = rndV.uint

    if pos_only:       # enforce data to positive
        if rndV.float>=0.: return rndV.float
        else       : return -rndV.float
    else:                   # generate nega/posi value
        negpos = 1 if randint(0,1)==1 else -1
        return np.float32(negpos * rndV.float)

def GenRndPatFloat32(batch, img_hw=28, img_ch=1, X=1., u8b=1, onehot=784, pos_only=False):
    maxf32 = np.finfo(np.float32).max
    minf32 = np.finfo(np.float32).min
    randpat = []
    for b in range(batch):
        randpat.append([np.clip(RxX(X, pos_only, u8b),minf32,maxf32) for i in range(pow(img_hw,2)*img_ch)])
    # Update patterns with OneHot
    for oh in range(onehot):
        randpat[oh] = [0.0]*pow(img_hw,2)
        #oneHot      = RxX(X, pos_only,  u8b=100)
        oneHot      = 1.111111111   # 0b111111100011100011100011100100
        randpat[oh][ randint(0,pow(img_hw,2)*img_ch-1) ] = oneHot
        #randpat[oh][ randint(0,pow(img_hw,2)*img_ch-1) ] = 1.111111111
    return np.asarray(randpat, dtype=np.float32).reshape(-1, img_hw, img_hw, img_ch)

# << Calculator fault difference function >>
# Notice: Can not use xor operator for float32 type
#
def faultDiff(A,B):
    assert len(A.reshape(-1))==len(B.reshape(-1)),'Mismatch length btn A and B'
    data_correction = False
    viewA = A.reshape(-1).copy()
    viewB = B.reshape(-1)
    # To avoid miss judgement about numpy.nan
    if np.isnan(viewA).any() or np.isnan(viewB).any():
        data_correction = True
        for idx,(I,J) in enumerate(zip(viewA,viewB)):
            if np.isnan(I+J):       # operation with np.nan become np.nan
                viewA[idx] = viewB[idx] = 0.0
       # << If allow detect by nan and float then use below code, >>
       # for idx,(I,J) in enumerate(zip(viewA,viewB)):
       #     if   np.isnan(I) and np.isnan(J):
       #         print('{} A {} => B {}'.format(idx,viewA[idx],viewB[idx]))
       #         viewA[idx] = viewB[idx] = 0.0
       #     elif np.isnan(I) :
       #         print('{} A {} => B {}'.format(idx,viewA[idx],viewB[idx]))
       #         viewA[idx] = J
       #     elif np.isnan(J) :
       #         print('{} A {} => B {}'.format(idx,viewA[idx],viewB[idx]))
       #         viewB[idx] = I
    # Create differences table
    diff = [ __f2i_union(I).uint==__f2i_union(J).uint for I,J in zip(viewA,viewB) ]
    return np.asarray(diff).reshape(A.shape), data_correction
#    assert len(A.reshape(-1))==len(B.reshape(-1)),'Mismatch length btn A and B'
#    diff = [ __f2i_union(I).uint==__f2i_union(J).uint for I,J in zip(A.reshape(-1),B.reshape(-1)) ]
#    return np.asarray(diff).reshape(A.shape)

# << Generating float32 patterns at random >>
# Calculate inference result of Before or After of SoftMax
# Notice!:
#   B(b)eforeSMax type is chainer.variable.Variable
#   A(a)fterSMax  type is numpy.ndarray
print('* Generating Test Pattern with batch ',var.batch)
Test_Patterns = GenRndPatFloat32(
    var.batch, X=args.randmax, pos_only=args.pos_only, u8b=args.upper8bit, onehot=args.onehot
)
print('* Generating Expected value of normal system')
var.n = -1  # For normal system inference
BeforeSMax, AfterSMax = forward.infer(Test_Patterns)

print('* Fault Point insertion and varify')
fault_injection_table = []
patSerrialNos         = set()
subsum = 0
RetryNo     = 0
overflows   = 0
while True:
    print('* << Try {:06d} >> fault simulation started'.format(RetryNo))
    detects = 0
    for var.n, spec in enumerate(var.faultpat):

        # spec: [0]detect_flag [1]layer [2]node [3]bit [4]sa01
        (detect_flag_idx, layer_idx, node_idx, bit_idx, sa01_idx) = (0, 1, 2, 3, 4)
        if spec[0]: continue

        # For fault system inference
        beforeSMax, afterSMax = forward.infer(Test_Patterns)

        # Calculate fault differencial function
##      diffA, data_correctionA = faultDiff(AfterSMax,  afterSMax)
        diffB, data_correctionB = faultDiff(BeforeSMax.data, beforeSMax.data)
        diff  = ~diffB  # True : propagated fault / False : disappearance fault
                        # diff.shape : ( batch, output_nodes )
        ovflag= ''
        if args.debug and data_correctionB:
            overflows += 1
            ovflag= 'OVF-{:06d}'.format(overflows)

        # Choice test pattern to detect fault point
        if diff.any():  # case detected
                                        # <diff>    dim-0:pattern          / dim01:fault point
            detInfo = np.where(diff)    # <detInfo> dim-0:differencial row / dim-1:differencial column
            detPtNo = detInfo[0][0]
            detColm = detInfo[1][0]
            if Test_Patterns[detPtNo][detColm] is np.inf or BeforeSMax.data[detPtNo][detColm] is np.inf:
                # Discard infinite calculation result
                print('\***** Warning np.inf FaultSim:{} <-> NormalSim:{}'.format(
                    beforeSMax[detPtNo][detColm],BeforeSMax.data[detPtNo][detColm]))
            else:
                var.faultpat[var.n][detect_flag_idx]=True
                fault_injection_table.append([ spec, Test_Patterns[detPtNo], BeforeSMax.data[detPtNo] ])
                detects += 1
                SerrialNo = detPtNo + RetryNo * var.batch
                patSerrialNos.add(SerrialNo)
                print('> detect try={:3d} faultNo={:6d} detPtNo={:6d} detects={:6d} spec={} {}'.format(
                    RetryNo, var.n, SerrialNo, detects, spec[1:], ovflag))
        elif 0: # case not detected, inserted faults disappeared, discard the patterns
            print('* Matched fault insertion run and normal system run, Discard')

    if detects>0: # Create new random patterns
        subsum += detects
        RetryNo+=1
        print('* Detected fault points det/subsum/all/% = {}/{}/{}/{:.4f}%'.format(
            detects, subsum, var.faultN, 100.*subsum/var.faultN))
        print('* Saving detected fault points, pattern and expected into',args.dt_list+'.npy')
        np.save(args.dt_list, fault_injection_table)
        print('* Saving undetected fault points list into',args.ud_list+'.npy')
        ud_table = np.asarray([i[layer_idx:] for i in var.faultpat if i[0] is False])
        np.save(args.ud_list, ud_table)
        print('* Creating New {} Test pattern'.format(var.batch))
        Test_Patterns = GenRndPatFloat32(
            var.batch, X=args.randmax, pos_only=args.pos_only, u8b=args.upper8bit, onehot=args.onehot
        )
        print('* Generating Expected value of normal system')
        var.n = -1  # For normal system inference
        BeforeSMax, AfterSMax = forward.infer(Test_Patterns)
        print('* Unique {} Random Patterns to Detect'.format(len(patSerrialNos)))
    else:
        break

#    break   # for debugging
if var.faultN>0:
    print('* Summary for Detected fault points det/all/%={}/{}/{:.3f}%'.format(
        subsum,var.faultN,100.*subsum/var.faultN)
    )
print('* End of Flow')

