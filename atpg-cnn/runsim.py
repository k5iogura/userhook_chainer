import warnings
warnings.simplefilter("ignore")
from pdb import set_trace
import os,sys,argparse
assert sys.version_info.major >= 3, 'Use over python3 version but now in {}'.format(sys.version_info)

import numpy as np

import forward
from   userfunc_var import *
from   userfunc import __f2i_union
from   random import seed, choice
from   utils import PatNames, timestamp

# for sharing Class variables
var = VAR()

# << Arguments >>
args = argparse.ArgumentParser()
args.add_argument('-g','--gpu',          type=int,default=-1)
args.add_argument('-tB','--tableB',   type=str,  default='dt_tableB.npy')
args.add_argument('-pB','--patternB', type=str,  default='dt_patternB.npy')
args.add_argument('--seed',           type=int,  default=2222222222)
args = args.parse_args()

# Check files
assert os.path.exists(args.patternB) and os.path.exists(args.tableB),'No inputs, run atpgrnd.py with python3 to make'

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
print(args)

# << Generate fault list >>
seed(args.seed)
repro_tableB   = args.tableB
repro_patternB = args.patternB
var.init(repro=repro_tableB)

# << Calculator fault difference function >>
# Notice: Can not use xor operator for float32 type
#
def faultDiff(A,B):
    assert len(A.reshape(-1))==len(B.reshape(-1)),'Mismatch length btn A and B'
    diff = [ __f2i_union(I).uint==__f2i_union(J).uint for I,J in zip(A.reshape(-1),B.reshape(-1)) ]
    return np.asarray(diff).reshape(A.shape)

Test_Patterns = np.load(args.patternB, allow_pickle=True)
print('* Reproduction : Load Test Pattern from', args.patternB,end=' ')
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
patSerrialNos = set()
DetHistory    = np.asarray([0]*var.batch)
detects = 0
Tstamp = timestamp('start')
for var.n, spec in enumerate(var.faultpat):

    # spec: [0]detect_flag [1]layer [2]node [3]bit [4]sa01
    (detect_flag_idx, layer_idx, node_idx, bit_idx, sa01_idx) = (0, 1, 2, 3, 4)
    if spec[detect_flag_idx]: continue    # skip already detected fault list

    # For fault system inference
    beforeSMax, afterSMax = forward.infer(Test_Patterns)

    # Calculate fault differencial function
    diffB = faultDiff(BeforeSMax.data, beforeSMax.data)
    diff  = ~diffB  # True : propagated fault / False : disappearance fault
                    # diff.shape : ( batch, output_nodes )

    # Choice test pattern to detect fault point
    if diff.any():  # case detected
                                    # <diff>    dim-0:pattern          / dim01:fault point
        detPtNo, detColm, DetHistory = uniquetest(diff,DetHistory)
        detects += 1
        var.faultpat[var.n][detect_flag_idx]=True
        SerrialNo = detPtNo
        new_flg = '*' if not SerrialNo in patSerrialNos else ' '
        patSerrialNos.add(SerrialNo)
        print('> detect faultNo={:6d} detPtNo={:6d}{} detects={:6d} spec={}'.format(
            var.n, SerrialNo, new_flg, detects, spec[layer_idx:]))
        assert SerrialNo in var.detpatterns[var.n],'pattern-{} not found in expected patterns {}'.format(
            SerrialNo, var.detpatterns[var.n])

if var.faultN>0:
    print('* Summary for Detected fault points det/all/%={}/{}/{:.3f}%'.format(
        detects,var.faultN,100.*detects/var.faultN)
    )
Tstamp.click('Faultsim elaplsed time')
print('* End of Flow')

