from pdb import set_trace
import os,sys,argparse
assert sys.version_info.major >= 3, 'Use over python3 version but now in {}'.format(sys.version_info)

import numpy as np

import forward
from   userfunc_var import *
from   userfunc import __f2i_union
from   random import seed, random, randint
from   rnd_generator import GenRndPatFloat32

# for sharing Class variables
var = VAR()

# PI setup
try:
    from pi_generator import pi_generator
    if var.pi is not None: print('* Link pi_generator into atpg with PI {}'.format(var.pi))
except: pass

args = argparse.ArgumentParser()
args.add_argument('-N','--normal_only',  action='store_true')

args.add_argument('-l','--layerNo',   type=int,  nargs='+', default=None)
#args.add_argument('-n','--nodeNo',    type=int,  nargs='+', default=None)
#args.add_argument('-b','--bitNo',     type=int,  nargs='+', default=None)
#args.add_argument('-s','--sa',        type=int,  nargs='+', default=None)

grp1 = args.add_mutually_exclusive_group()
grp1.add_argument('-i','--inputName', type=str,  default=None)
grp1.add_argument('-pi','--pi_generator',action='store_true')

args.add_argument('-t','--targetFile',type=str,  default=None, help='fault simulation target file')
args.add_argument('-u','--ud_list',   type=str,  default='ud_list')
args.add_argument('-d','--dt_list',   type=str,  default='dt_list')
args.add_argument('-sd','--save_dt',  action='store_true', help='saving detect pat and expected pat')
args.add_argument('-tB','--tableB',   type=str,  default='dt_tableB.npy')
args.add_argument('-pB','--patternB', type=str,  default='dt_patternB.npy')

args.add_argument('--batch',          type=int,  default=1024)
args.add_argument('--bmax',           type=int,  default=1024*4)
args.add_argument('--seed',           type=int,  default=2222222222)
args.add_argument('--upper8bit',      type=int,  default=0, help='specify as %')
args.add_argument('--onehot',         type=int,  default=0, help='onehot patterns')
args.add_argument('--retrymax',       type=int,  default=10000)
args.add_argument('--pos_only',       action='store_true')
args.add_argument('-r','--randmax',   type=float,default=255.0)
args = args.parse_args()

# add heuristic patterns
args.batch += args.onehot

print(args)
# Warn no saving result
if args.save_dt is False: print('* No saving detect pattern table')

# << Generate fault list >>
seed(args.seed)
net_spec=(28*28, 12*12*32, 4*4*64, 300, 10)
layerSet = set()
if args.layerNo is not None:
    _ = [layerSet.add(i) for i in args.layerNo]
    net_specList = [0]*len(net_spec)
    for i in layerSet: net_specList[i] = net_spec[i]
    net_spec = tuple(net_specList)
var.init(Batch=args.batch, Net_spec=net_spec, target=args.targetFile)

# << Calculator fault difference function >>
# Notice: Can not use xor operator for float32 type
#
def faultDiff(A,B):
    assert len(A.reshape(-1))==len(B.reshape(-1)),'Mismatch length btn A and B'
    diff = [ __f2i_union(I).uint==__f2i_union(J).uint for I,J in zip(A.reshape(-1),B.reshape(-1)) ]
    return np.asarray(diff).reshape(A.shape)

# << increase batch >>
# max : bmax min : batch
def update_batch(Try, batch, bmax, Try2max=10):
    alpha = (bmax-batch)/Try2max
    x = int(batch + alpha * Try)
    x = x if x <= bmax else bmax
    return x

# << Switch Generator which User or Embedded Random >>
var.enable_user_generator = args.pi_generator
if var.enable_user_generator is False: var.pi = None
if args.inputName         is not None: var.pi = args.inputName

# << Generating float32 patterns at random >>
print('* Generating Test Pattern with batch ',var.batch)
# Setup spec. of pattern generator infrom userhook.py
rnd_generator_option = {
    'batch':var.batch, 'img_hw':28, 'img_ch':1,
    'X':args.randmax, 'pos_only':args.pos_only, 'u8b':args.upper8bit, 'onehot':args.onehot
}
Test_Patterns = GenRndPatFloat32(   # NHWC
    var.batch, X=args.randmax, pos_only=args.pos_only, u8b=args.upper8bit, onehot=args.onehot
)
# Calculate inference result of Before or After of SoftMax
# Notice!:
#   B(b)eforeSMax type is chainer.variable.Variable
#   A(a)fterSMax  type is numpy.ndarray
print('* Generating Expected value of normal system')
var.n = -1  # For normal system inference
BeforeSMax, AfterSMax = forward.infer(Test_Patterns)

print('* Fault Point insertion and varify')
fault_injection_table = []
fault_injection_tableB = []
fault_injection_tableI = 0
fault_injection_tableP = []
subsum = 0
RetryNo     = 0
patSerrialNos = set()
while True:
    if args.normal_only:break   # skip fault simulation
    print('* << Try {:06d} >> fault simulation started'.format(RetryNo))
    detects = 0
    for var.n, spec in enumerate(var.faultpat):

        # spec: [0]detect_flag [1]layer [2]node [3]bit [4]sa01
        (detect_flag_idx, layer_idx, node_idx, bit_idx, sa01_idx) = (0, 1, 2, 3, 4)
        if spec[detect_flag_idx]: continue    # skip already detected fault list

        # For fault system inference
        beforeSMax, afterSMax = forward.infer(Test_Patterns)

        # Calculate fault differencial function
        diffA = faultDiff(AfterSMax,  afterSMax)
        diffB = faultDiff(BeforeSMax.data, beforeSMax.data)
        diff  = ~diffB  # True : propagated fault / False : disappearance fault
                        # diff.shape : ( batch, output_nodes )

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
                detects += 1
                var.faultpat[var.n][detect_flag_idx]=True
                fault_injection_table.append ([ spec, Test_Patterns[detPtNo], BeforeSMax.data[detPtNo] ])
                SerrialNo = detPtNo + RetryNo * var.batch
                if not SerrialNo in patSerrialNos:
                    fault_injection_tableP.append( [ SerrialNo, Test_Patterns[detPtNo] ] )
                fault_injection_tableI = [ i for i,p in enumerate(fault_injection_tableP) if p[0] == SerrialNo ][0]
                fault_injection_tableB.append([ spec[layer_idx:], fault_injection_tableI, BeforeSMax.data[detPtNo] ])
                patSerrialNos.add(SerrialNo)
                print('> detect try={:3d} faultNo={:6d} detPtNo={:6d} detects={:6d} spec={}'.format(
                    RetryNo, var.n, SerrialNo, detects, spec[1:]))

    if detects>0: # Create new random patterns
        # << write TableB out >>
        print('* Saving detected fault points, pattern and expected into as tableB', args.tableB, args.patternB)
        np.save(args.tableB,   np.asarray(fault_injection_tableB))
        np.save(args.patternB, np.asarray([ i[-1] for i in fault_injection_tableP ]))

        # << increase batch size for next simulation and reporting >>
        subsum += detects
        RetryNo+= 1
        if var.batch < args.bmax: var.batch = update_batch( Try=RetryNo, batch=var.batch, bmax=args.bmax, Try2max=10)
        print('* Detected fault points det/subsum/all/% = {}/{}/{}/{:.4f}%'.format(
            detects, subsum, var.faultN, 100.*subsum/var.faultN))

        # << write ud_list.npy out >>
        if args.save_dt:
            print('* Saving detected fault points, pattern and expected into',args.dt_list+'.npy')
            np.save(args.dt_list, fault_injection_table)
        ud_table = np.asarray([i[layer_idx:] for i in var.faultpat if i[detect_flag_idx] == 0])   # Use 0 instead of False when numpy
        print('* Saving undetected fault points list into',args.ud_list+'.npy','size',len(ud_table),ud_table.shape)
        if len(ud_table) == 0: print('* undetected fault points is Zero! ** Congratulations **')
        np.save(args.ud_list, ud_table)

        # << creating new additional tenst patterns and re-run normal system >>
        print('* Creating New {} Test pattern'.format(var.batch))
        Test_Patterns = GenRndPatFloat32(
            var.batch, X=args.randmax, pos_only=args.pos_only, u8b=args.upper8bit, onehot=args.onehot
        )
        print('* Generating Expected value of normal system')
        var.n = -1  # For normal system inference
        BeforeSMax, AfterSMax = forward.infer(Test_Patterns)
        print('* Unique {} Random Patterns to Detect'.format(len(patSerrialNos)))
    else: break

    if RetryNo > args.retrymax:
        print('Stop simulation {}th at over --retrymax {}'.format(RetryNo, args.retrymax))
        break

if var.faultN>0:
    print('* Summary for Detected fault points det/all/%={}/{}/{:.3f}%'.format(
        subsum,var.faultN,100.*subsum/var.faultN)
    )
print('* End of Flow')

