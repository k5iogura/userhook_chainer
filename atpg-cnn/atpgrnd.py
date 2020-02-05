import warnings
warnings.simplefilter("ignore")
from pdb import set_trace
import os,sys,argparse
from copy import copy
assert sys.version_info.major >= 3, 'Use over python3 version but now in {}'.format(sys.version_info)

import numpy as np

import forward
from   userfunc_var import *
from   userfunc import __f2i_union
from   random import seed, random, randint, choice
from   rnd_generator import GenRndPatFloat32
from   utils import PatNames, timestamp, printargs

# for sharing Class variables
var = VAR()

# PI setup
try:
    from pi_generator import pi_generator
    if var.pi is not None: print('* Link pi_generator into atpg with PI {}'.format(var.pi))
except: pass

# << Arguments >>
args = argparse.ArgumentParser()
args.add_argument('-N','--normal_only',  action='store_true')
args.add_argument('-D','--debug',        action='store_true')
args.add_argument('-g','--gpu',          type=int,default=-1)

args.add_argument('-l','--layerNo',   type=int,  nargs='+', default=None)
#args.add_argument('-n','--nodeNo',    type=int,  nargs='+', default=None)
#args.add_argument('-b','--bitNo',     type=int,  nargs='+', default=None)
#args.add_argument('-s','--sa',        type=int,  nargs='+', default=None)

grp1 = args.add_mutually_exclusive_group()
grp1.add_argument('-i','--inputName', type=str,  default=None)
grp1.add_argument('-pi','--pi_generator',action='store_true')

grp2 = args.add_mutually_exclusive_group()
grp2.add_argument('-t','--targetFile',type=str,  default=None, help='fault simulation target file')
grp2.add_argument('-rp','--reproduct',action='store_false', dest='faultsim_mode')

args.add_argument('-u','--ud_list',   type=str,  default='ud_list')
args.add_argument('-d','--dt_list',   type=str,  default='dt_list')
args.add_argument('-sd','--save_dt',  action='store_true', help='saving detect pat and expected pat')
args.add_argument('-tB','--tableB',   type=str,  default='dt_tableB.npy')
args.add_argument('-pB','--patternB', type=str,  default='dt_patternB.npy')
args.add_argument('-sX','--skip_tX',  action='store_true')
args.add_argument('-tX','--tableX',   type=str,  default='dt_tableX.npy')
args.add_argument('-pX','--patternX', type=str,  default='dt_patternX.npy')

args.add_argument('--batch',          type=int,  default=1024)
args.add_argument('--bmax',           type=int,  default=1024*4)
args.add_argument('--bmax2',          type=int,  default=10)
args.add_argument('--seed',           type=int,  default=2222222222)
args.add_argument('--upper8bit',      type=int,  default=0, help='specify as %')
args.add_argument('--onehot',         type=int,  default=0, help='onehot patterns')
args.add_argument('--retrymax',       type=int,  default=10000)
args.add_argument('--pos_only',       action='store_true')
args.add_argument('-r','--randmax',   type=float,default=255.0)
args.add_argument('-px','--prefix',   type=str,  default='')

args = args.parse_args()

# For GPU
if args.gpu >= 0:
    try:
        import cupy, chainer
        device = chainer.get_device(args.gpu)
        assert '@cupy' in str(device),"Supports Only CPU"
        print('GPU device is ',device)
        device.use()
        forward.model.to_device(device) # load model to GPU
    except:
        print('--gpu',args.gpu,'but CuPy not found out or GPU Full, goto CPU mode')
        args.gpu = -1 # only CPU supported

# add heuristic patterns
args.batch += args.onehot
var.batch   = args.batch

# << marking files a job up >>
if args.prefix != '':
    args.prefix+= '_'
    args.ud_list = args.prefix + args.ud_list
    args.dt_list = args.prefix + args.dt_list
    if args.faultsim_mode:
        args.tableB   = args.prefix + args.tableB
        args.patternB = args.prefix + args.patternB
        args.tableX   = args.prefix + args.tableX
        args.patternX = args.prefix + args.patternX

printargs(args)
# Warn no saving result
if args.save_dt is False: print('* No saving detect pattern table')
if args.skip_tX is True : print('* No saving detect pattern tableX')

# << Generate fault list >>
seed(args.seed)
net_spec=(28*28, 12*12*32, 4*4*64, 300, 10)
layerSet = set()
if args.layerNo is not None:
    _ = [layerSet.add(i) for i in args.layerNo]
    net_specList = [0]*len(net_spec)
    for i in layerSet: net_specList[i] = net_spec[i]
    net_spec = tuple(net_specList)
repro_tableB   = None if args.faultsim_mode else args.tableB
repro_patternB = None if args.faultsim_mode else args.patternB
var.init(Batch=args.batch, Net_spec=net_spec, target=args.targetFile, repro=repro_tableB)
if args.targetFile is not None and args.layerNo is not None:
    print('* Ignored --layerNo option at --targetFile mode')
if not args.faultsim_mode and args.layerNo is not None:
    print('* Ignored --layerNo option at --reproduct mode')

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
    assert not np.isnan(viewA).any() and not np.isnan(viewB).any()
    diff = [ __f2i_union(I).uint==__f2i_union(J).uint for I,J in zip(viewA,viewB) ]
    return np.asarray(diff).reshape(A.shape), data_correction

# << increase batch >>
# max : bmax min : batch
def update_batch(Try, batch, bmax, Try2max=args.bmax2):
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
var.rnd_generator_option = {
    'batch':var.batch, 'img_hw':28, 'img_ch':1,
    'X':args.randmax, 'pos_only':args.pos_only, 'u8b':args.upper8bit, 'onehot':args.onehot
}
if args.faultsim_mode:
    print('* Generate Test Pattern with GenRndPatFloat32')
    Test_Patterns = GenRndPatFloat32(   # NHWC
        var.batch, X=args.randmax, pos_only=args.pos_only, u8b=args.upper8bit, onehot=args.onehot
    )
else:
    Test_Patterns = np.load(args.patternB, allow_pickle=True)
    print('* Reproduction : Load Test Pattern from', args.patternB,end=' ')
    print('* Updated Batch size from {} to {} *'.format(var.batch, len(Test_Patterns)))
    var.batch = len(Test_Patterns)

if args.gpu >= 0: Test_Patterns = cupy.asarray(Test_Patterns) # load data  to GPU

# Calculate inference result of Before or After of SoftMax
# Notice!:
#   B(b)eforeSMax type is chainer.variable.Variable
#   A(a)fterSMax  type is numpy.ndarray
print('* Generating Expected value of normal system')
var.n = -1  # For normal system inference
BeforeSMax, AfterSMax = forward.infer(Test_Patterns)

# << Unique TestPattern Selector >>
# fdiff.shape       : ( batch, output_nodes )
# det_history.shape : ( batch )
# pickup : select method such as min, choice
def uniquetest(fdiff, det_history, pickup=min):
    assert pickup in [min, choice]
    assert type(fdiff) == np.ndarray and type(fdiff) == type(det_history) and fdiff.any()
    if det_history.all():
        # already all patterns are true in det_history table
        return ( 0, 0, det_history )
    # make diffsmmry table
    batch, nodes = fdiff.shape
    diffsmmry = np.asarray( [0] * batch )      # initialize as all False
    for b in range(batch): diffsmmry[b] = 1 if any(fdiff[b]) else 0
    # make common detected table
    cmmn_det = diffsmmry * det_history
    if cmmn_det.any():
        # detectable by other patterns
        cmmn_pat = np.where(cmmn_det)[0][0]
        return ( np.where(cmmn_det)[0][0], np.where(fdiff[cmmn_pat])[0][0], det_history )
    # detectable by new pattern
    new_smmry = ( (diffsmmry ^ det_history) * diffsmmry )
    detptnNos = np.where(new_smmry==1)[0]
    new_ptnNo = pickup(detptnNos) if len(detptnNos)>0 else -1
    new_clmNo = np.where(fdiff[new_ptnNo])[0][0] if new_ptnNo>=0 else -1
    # update det_history table
    if new_ptnNo >= 0: det_history[new_ptnNo] = True
    return ( new_ptnNo, new_clmNo, det_history )

print('* Fault Point insertion and varify')
fault_injection_table  = []
fault_injection_tableB = []
fault_injection_tableI = 0
fault_injection_tableP = []
fault_injection_tableX = []
fault_injection_tablePX= []
patNames = PatNames(var.batch)

subsum        = 0
RetryNo       = 0
patSerrialNos = set()
DetHistory    = np.asarray([0]*var.batch)
overflows     = 0

Tstamp = timestamp('start')
while True:
    if args.normal_only:break   # skip fault simulation
    print('* << Try {:06d} >> fault simulation started'.format(RetryNo))
    detects = 0
    fault_injection_tableXS= set()
    for var.n, spec in enumerate(var.faultpat):

        # spec: [0]detect_flag [1]layer [2]node [3]bit [4]sa01
        (detect_flag_idx, layer_idx, node_idx, bit_idx, sa01_idx) = (0, 1, 2, 3, 4)
        if spec[detect_flag_idx]: continue    # skip already detected fault list

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

            # prepare tableX
            # <diff>    dim-0:pattern No        / dim01:detect Location
            # tableX [ [[spec],[index pat]], ... ]
            xDetInfo = np.where(diff)
            xDetPtNo = xDetInfo[0]
            xDetPtNoSet = set()
            for i in xDetPtNo: xDetPtNoSet.add(i)           # Unique sort by set class
            xDetPtNo = np.asarray(list(xDetPtNoSet))
            xDetPtSn = xDetPtNo + patNames.offset           # To serrial No.
            fault_injection_tableX.append([ spec[layer_idx:], [patNames.index2name(i) for i in xDetPtSn] ])

            detPtNo, detColm, DetHistory = uniquetest(diff,DetHistory)
            if Test_Patterns[detPtNo][detColm] is np.inf or BeforeSMax.data[detPtNo][detColm] is np.inf:
                # Discard infinite calculation result
                print('\***** Warning np.inf FaultSim:{} <-> NormalSim:{}'.format(
                    beforeSMax[detPtNo][detColm],BeforeSMax.data[detPtNo][detColm]))
            else:
                detects += 1
                var.faultpat[var.n][detect_flag_idx]=True
                fault_injection_table.append ([ spec, Test_Patterns[detPtNo], BeforeSMax.data[detPtNo] ])
                SerrialNo = detPtNo + RetryNo * var.batch
                new_flg = '*' if not SerrialNo in patSerrialNos else ' '
                if not SerrialNo in patSerrialNos:
                    fault_injection_tableP.append( [ SerrialNo, Test_Patterns[detPtNo] ] )
                fault_injection_tableI = [ i for i,p in enumerate(fault_injection_tableP) if p[0] == SerrialNo ][0]
                fault_injection_tableB.append([ spec[layer_idx:], fault_injection_tableI, BeforeSMax.data[detPtNo] ])
                patSerrialNos.add(SerrialNo)
                print('> detect try={:3d} faultNo={:6d} detPtNo={:6d}{} detects={:6d} spec={} {}'.format(
                    RetryNo, var.n, SerrialNo, new_flg, detects, spec[1:], ovflag))

    Tstamp.click('Until Retry {} elapsed time'.format(RetryNo))
    if detects>0: # Create new random patterns
        printargs(args)

        # << update tablePX and write TableX out >>
        if args.faultsim_mode:
            extendN = patNames.count - len(fault_injection_tablePX)
            fault_injection_tablePX.extend( [None] * extendN )
            for ptn_idx, ptn_name in patNames:
                if ptn_idx < patNames.offset: continue
                ptnIndexLocal = ptn_idx - patNames.offset
                fault_injection_tablePX[ptn_name] = copy( Test_Patterns[ ptnIndexLocal ] )
            print('* Saving detected fault points, pattern and expected into as tableX', args.tableX, args.patternX)
            print('* tableX size={} patternX size={}'.format(len(fault_injection_tableX), len(fault_injection_tablePX)))
            np.save(args.tableX,   np.asarray( fault_injection_tableX ))
            np.save(args.patternX, np.asarray( [ i for i in fault_injection_tablePX ] ))

        subsum += detects
        RetryNo+= 1
        if args.faultsim_mode:
            # << write TableB out >>
            print('* Saving detected fault points, pattern and expected into as tableB', args.tableB, args.patternB)
            np.save(args.tableB,   np.asarray(fault_injection_tableB))
            np.save(args.patternB, np.asarray([ i[-1] for i in fault_injection_tableP ]))
        else:
            print('* Not Saving detected fault points, pattern and expected at reproduct mode')
            break

        # << increase batch size for next simulation and reporting >>
        if args.inputName is None and var.batch < args.bmax:
            var.batch = update_batch( Try=RetryNo, batch=var.batch, bmax=args.bmax, Try2max=args.bmax2)
            DetHistory= np.asarray([0]*var.batch)   # renew for new batch size
            patNames.extend( var.batch )
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

        print('* Unique {} Random Patterns to Detect'.format(len(patSerrialNos)))

        # << creating new additional test patterns and re-run normal system >>
        print('* Creating New {} Test pattern'.format(var.batch))
        Test_Patterns = GenRndPatFloat32(
            var.batch, X=args.randmax, pos_only=args.pos_only, u8b=args.upper8bit, onehot=args.onehot
        )
        if args.gpu >= 0: Test_Patterns = cupy.asarray(Test_Patterns) # load data  to GPU
        print('* Generating Expected value of normal system')
        var.n = -1  # For normal system inference
        BeforeSMax, AfterSMax = forward.infer(Test_Patterns)
    else: break

    # << Limitation under retrymax >>
    if RetryNo > args.retrymax:
        print('Stop simulation {}th at over --retrymax {}'.format(RetryNo, args.retrymax))
        break

    # << Stop simulation at 1 time if reproduction mode >>
    if not args.faultsim_mode:
        print('Stop simulation at --reproduct {}'.format((not args.faultsim_mode)))
        break

if var.faultN>0:
    print('* Summary for Detected fault points det/all/%={}/{}/{:.3f}%'.format(
        subsum,var.faultN,100.*subsum/var.faultN)
    )
printargs(args)
Tstamp.click('Faultsim elaplsed time')
print('* End of Flow')

