import numpy as np
from math import *
import sys,os
import PIL as p


def _wh_sqrt(n):
    x=1
    for i in range(n-1,2,-1):
        if (sqrt(n)%i)==0:
           x = i
           break
    return x

def wh_div(n):
    x=_wh_sqrt(n)
    if x!=1:return x,x
    x=1
    y=1
    for i in range(2,int(n/5)+1):
        h = int(n//i)
        if h!=1:
            x=i
            y=h
    return x,y

def erase_1(param):
    if param.ndim>=2 and param.shape[0]==1:
        param = param.reshape(param.shape[1:])
    return param

paramfile = sys.argv[-1]
param = np.load('dnn_params/lx1_Linear_in.npy')
param = np.load('dnn_params/lx1_Linear_b.npy')
param = np.load('dnn_params/lx1_Linear_W.npy')
param = np.load('dnn_params/lx1_Linear_out.npy')

#param : NWHC
param = np.zeros((2048,22,22)) #for test
param = np.zeros((512,512,22)) #for test
param = erase_1(param)

if param.ndim == 1:
    w,h=wh_div(param.shape[-1])
    buff = np.zeros((w*h),dtype=np.float32)
    buff = param.reshape(-1)[:w*h]
    param= buff.reshape((1,1,w,h))
elif param.ndim == 2:
    w,h=wh_div(param.shape[-1])
    buff = np.zeros((param.shape[0]*w*h),dtype=np.float32)
    buff = param.reshape(-1)[:param.shape[0]*w*h]
    param= buff.reshape((1,param.shape[0],w,h))
elif param.ndim == 3:
    w,h=param.shape[:2]
    param = param.reshape((1,param.shape[-1],w,h))
else:
    assert param.ndim <= 4 , 'not support'

#param : NCWH 4-dimension allways
print("Visualization as NCWH",param.shape)
iW,iH = param.shape[2:]
gW,gH = wh_div(param.shape[1])
pW,pH = gW*iW, gH*iH

print("* canvas WH",pW,pH)
print("* snipet WH",iW,iH)
print("* grid   WH",gW,gH)
print("* resize WH",640,480)

for gy in range(gH):
    for gx in range(gW):
        print("%8d%8d%8d%8d"%(gx,gy,gx*iW,gy*iH))
        pass



