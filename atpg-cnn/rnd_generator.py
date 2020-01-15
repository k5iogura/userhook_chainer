
from pdb import set_trace
import sys, os
assert sys.version_info.major >= 3, 'Use over python3 version but now in {}'.format(sys.version_info)

import numpy as np

from   userfunc_var import *
from   userfunc import __f2i_union
from   random import seed, random, randint

# for sharing Class variables
var = VAR()

# << random number generators for int32 and float32 >>
# python random function generates 53bit float random number by Mersenne twister
def RxX(X, pos_only, u8b):

    rndV   = X*random()     # generate a value at random

    rndV = __f2i_union(rndV)
    if randint(1,100)<=u8b: # make upper 8bit at random
        rndV.uint = rndV.uint | np.uint32(randint(0,0x0f)<<27)

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
        oneHot      = 1.111111111   # 0b111111100011100011100011100100
        randpat[oh][ randint(0,pow(img_hw,2)*img_ch-1) ] = oneHot
    return np.asarray(randpat, dtype=np.float32).reshape(-1, img_hw, img_hw, img_ch)

