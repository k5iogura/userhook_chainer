import numpy as np
import bitstring
from pdb import *
from chainer.functions.connection.linear import linear
from chainer.functions import convolution_2d

from userfunc_var import VAR, GenFP
var = VAR()

# Change fault bit in float32
import ctypes
class __f2i_union(ctypes.Union):
    _fields_=[('float',ctypes.c_float),('uint',ctypes.c_uint)]

def bitChange(v,bit,sa01):
    f2i_union = __f2i_union(v)
    normal    = __f2i_union(v)
    normal1s= bin(normal.uint).count("1")
    if sa01 == 1:
        faultValue = f2i_union.uint | (0x01<<bit)
        fault1s = bin(faultValue).count("1")
        if (normal1s) != (fault1s) and (normal1s) != (fault1s - 1):
            print("***** sa1>>>internal error bit operation {}-{}".format(normal1s,fault1s))
    elif sa01 == 0:
        faultValue = f2i_union.uint & ~(0x01<<bit)
        fault1s = bin(faultValue).count("1")
        if (normal1s) != (fault1s) and (normal1s) != (fault1s + 1):
            print("***** sa0>>>internal error bit operation {}-{}".format(normal1s,fault1s))
    else:
        assert False, "sa01 is out of value sa01={}".format(sa01)
    f2i_union.uint = faultValue
    return f2i_union.float, f2i_union.uint

##############################################################
# This file was automatically generated by UserHook system.
# Copy this file as ./userfunc.py
# And edit it.
# You can get results to expect. Lucks, K.Ogura.
##############################################################
#import numpy as np
def conv1_Convolution2D(_in,_out):
    _name='conv1_Convolution2D'
    # _in.args[0].shape : batch, channels, in_size, in_size          (1024, 1, 28, 28)
    # _out.shape        : batch, channels, go_size, go_size, go_size (1024, 32, 24, 24)
    batch, in_channels, in_size, in_size = _in.args[0].shape
    batch, go_channels, go_size, go_size = _out.data.shape

    if var.n<0: # No fault injection for Normal System case check only
        print(_name,'in/out =', _in.args[0].shape, _out.data.shape)
        assert _in.args[0].dtype == np.float32, 'Unsupport input type {}'.format(_out.dtype)
        assert _out.dtype        == np.float32, 'Unsupport out   type {}'.format(_out.dtype)
        return

    detect_flag, layer, node, bit, sa01 = var.faultpat[var.n]

    if layer == 0:
        # Update _in with sa01
        g = _in.args[0].copy()
        for i in range(batch):
            normal = g[ i ][ 0 ][ node//in_size ][ node%in_size ]
            v_float, v_uint = bitChange(normal, bit, sa01)
            g[ i ][ 0 ][ node//in_size ][ node%in_size ] = np.float32(v_float)
        if 0:
            print("{:8d} faultpattern={}".format(var.n, var.faultpat[var.n]))
            print(' '*8, np.max(_in.args[0]), '=>', np.max(g), np.min(_in.args[0]), '=>', np.min(g))
        _in.args[0].data = g

        # Get Link Convolution_2D attrigutes
        x, w, b      = _in.args[0], _in.link.__dict__['W'], _in.link.__dict__['b'],
        in_channels  = _in.link.in_channels
        out_channels = _in.link.out_channels
        ksize        = _in.link.ksize
        stride       = _in.link.stride
        pad          = _in.link.pad
        # nobias       = _in.link.nobias    # not attribute of the class
        dilate       = _in.link.dilate
        groups       = _in.link.groups

        # Calculate Convolution Layer after fault insertion
        this_conv2d_out = convolution_2d(
            x, w, b, stride=stride, pad=pad, dilate=dilate, groups=groups
        )
        this_conv2d_out = this_conv2d_out.reshape(_out.data.shape)
        _out.data = this_conv2d_out.data

def bn1_BatchNormalization(_in,_out):
    _name='bn1_BatchNormalization'
    pass

def conv2_Convolution2D(_in,_out):
    _name='conv2_Convolution2D'
    # _in.args[0].shape : batch, channels, in_size, in_size          (1024, 32, 12, 12)
    # _out.shape        : batch, channels, go_size, go_size, go_size (1024, 64, 8, 8)
    batch, in_channels, in_size, in_size = _in.args[0].shape
    batch, go_channels, go_size, go_size = _out.data.shape

    if var.n<0: # No fault injection for Normal System case check only
        print(_name,'in/out =', _in.args[0].data.shape, _out.data.shape)
        assert _in.args[0].dtype == np.float32, 'Unsupport input type {}'.format(_out.dtype)
        assert _out.dtype        == np.float32, 'Unsupport out   type {}'.format(_out.dtype)
        return

    detect_flag, layer, node, bit, sa01 = var.faultpat[var.n]

    if layer == 1:
        # Update _in with sa01
        g = _in.args[0].data.copy()
        for i in range(batch):
            normal = g[ i ][ node//(in_size*in_size) ][ node//in_size ][ node%in_size ]
            v_float, v_uint = bitChange(normal, bit, sa01)
            g[ i ][ node//(in_size*in_size) ][ node//in_size ][ node%in_size ] = np.float32(v_float)
        if 0:
            print("{:8d} faultpattern={}".format(var.n, var.faultpat[var.n]))
            print(' '*8, np.max(_in.args[0]), '=>', np.max(g), np.min(_in.args[0]), '=>', np.min(g))
        _in.args[0].data = g

        # Get Link Convolution_2D attrigutes
        x, w, b      = _in.args[0], _in.link.__dict__['W'], _in.link.__dict__['b'],
        in_channels  = _in.link.in_channels
        out_channels = _in.link.out_channels
        ksize        = _in.link.ksize
        stride       = _in.link.stride
        pad          = _in.link.pad
        # nobias       = _in.link.nobias    # not attribute of the class
        dilate       = _in.link.dilate
        groups       = _in.link.groups

        # Calculate Convolution Layer after fault insertion
        this_conv2d_out = convolution_2d(
            x, w, b, stride=stride, pad=pad, dilate=dilate, groups=groups
        )
        this_conv2d_out = this_conv2d_out.reshape(_out.data.shape)
        _out.data = this_conv2d_out.data

def bn2_BatchNormalization(_in,_out):
    _name='bn2_BatchNormalization'
    pass

def l1_Linear(_in,_out):
    _name='l1_Linear'
    if var.n<0: # No fault injection for Normal System case
        print(_name,'in/out           =', _in.args[0].data.shape, _out.data.shape)
        assert _in.args[0].dtype == np.float32, 'Unsupport input type {}'.format(_out.dtype)
        assert _out.dtype        == np.float32, 'Unsupport out   type {}'.format(_out.dtype)
        return

def l2_Linear(_in,_out):
    _name='l2_Linear'
    if var.n<0: # No fault injection for Normal System case
        print(_name,'in/out           =', _in.args[0].data.shape, _out.data.shape)
        assert _in.args[0].dtype == np.float32, 'Unsupport input type {}'.format(_out.dtype)
        assert _out.dtype        == np.float32, 'Unsupport out   type {}'.format(_out.dtype)
        return

### import numpy as np
### import bitstring
### from pdb import *
### from chainer.functions.connection.linear import linear
### 
### from userfunc_var import VAR, GenFP
### var = VAR()
### 
### # Change fault bit in float32
### import ctypes
### class __f2i_union(ctypes.Union):
###     _fields_=[('float',ctypes.c_float),('uint',ctypes.c_uint)]
### 
### def bitChange(v,bit,sa01):
###     f2i_union = __f2i_union(v)
###     normal    = __f2i_union(v)
###     normal1s= bin(normal.uint).count("1")
###     if sa01 == 1:
###         faultValue = f2i_union.uint | (0x01<<bit)
###         fault1s = bin(faultValue).count("1")
###         if (normal1s) != (fault1s) and (normal1s) != (fault1s - 1):
###             print("***** sa1>>>internal error bit operation {}-{}".format(normal1s,fault1s))
###     elif sa01 == 0:
###         faultValue = f2i_union.uint & ~(0x01<<bit)
###         fault1s = bin(faultValue).count("1")
###         if (normal1s) != (fault1s) and (normal1s) != (fault1s + 1):
###             print("***** sa0>>>internal error bit operation {}-{}".format(normal1s,fault1s))
###     else:
###         assert False, "sa01 is out of value sa01={}".format(sa01)
###     f2i_union.uint = faultValue
###     return f2i_union.float, f2i_union.uint
### 
### # layer-0/1 hook
### # Insert a fault in both _in and _out
### def lx1_Linear(_in,_out):
###     # _in.args[0].shape : batch, in_size, in_size, channels
###     # _out.shape        : batch, go_size
###     batch, in_size, in_size, channels = _in.args[0].shape
###     batch, go_size                    = _out.data.shape
### 
###     if var.n<0: # No fault injection for Normal System case check only
###         assert _in.args[0].dtype == np.float32, 'Unsupport input type {}'.format(_out.dtype)
###         assert _out.dtype        == np.float32, 'Unsupport out   type {}'.format(_out.dtype)
###         return
### 
###     detect_flag, layer, node, bit, sa01 = var.faultpat[var.n]
### 
###     if layer == 0:
###         # Update _in with sa01
###         g = _in.args[0].copy()
###         for i in range(batch):
###             normal = g[ i ][ node//in_size ][ node%in_size ][0]
###             v_float, v_uint = bitChange(normal, bit, sa01)
###             g[ i ][ node//in_size ][ node%in_size ][0] = np.float32(v_float)
###         if 0:
###             print("{:8d} faultpattern={}".format(var.n, var.faultpat[var.n]))
###             print(' '*8, np.max(_in.args[0]), '=>', np.max(g), np.min(_in.args[0]), '=>', np.min(g))
###         _in.args[0].data = g
### 
###         # Calculate Linear Layer after fault insertion
###         this_linear_out = linear(_in.args[0], _in.link.__dict__['W'], _in.link.__dict__['b'])
###         this_linear_out = this_linear_out.reshape(_out.data.shape)
###         _out.data = this_linear_out.data
### 
###     if layer == 10:
###         g = _out.data.copy()
###         for i in range(batch):
###             normal = g[ i ][ node//go_size ]
###             v_float, v_uint = bitChange(normal, bit, sa01)
###             g[ i ][ node//go_size ] = np.float32(v_float)
###         if 0:
###             print("{:8d} faultpattern={}".format(var.n, var.faultpat[var.n]))
###             print(' '*8, np.max(_out.data), '=>', np.max(g), np.min(_out.data), '=>', np.min(g))
###         _out.data = g
### 
### 
### # layer-2 hook
### def ly2_Linear(_in,_out):
###     if var.n<0: # No fault injection for Normal System case
###         assert _in.args[0].dtype == np.float32, 'Unsupport input type {}'.format(_out.dtype)
###         assert _out.dtype        == np.float32, 'Unsupport out   type {}'.format(_out.dtype)
###         return
###     batch, in_size = _in.args[0].shape
###     batch, go_size = _out.data.shape
###     detect_flag, layer, node, bit, sa01 = var.faultpat[var.n]
###     if layer == 1:
###         # Update _in with sa01
###         g = _in.args[0].data.copy()
###         for i in range(batch):
###             normal = g[ i ][ node%in_size ]
###             v_float, v_uint = bitChange(normal, bit, sa01)
###             g[ i ][ node%in_size ] = np.float32(v_float)
###         if 0:
###             print("{:8d} faultpattern={}".format(var.n, var.faultpat[var.n]))
###             print(' '*8, np.max(_in.args[0]), '=>', np.max(g), np.min(_in.args[0]), '=>', np.min(g))
###         _in.args[0].data = g
### 
###         # Calculate Linear Layer after fault insertion
###         this_linear_out = linear(_in.args[0], _in.link.__dict__['W'], _in.link.__dict__['b'])
###         this_linear_out = this_linear_out.reshape(_out.data.shape)
###         _out.data = this_linear_out.data
### 
###     if layer == 20:
###         g = _out.data.copy()
###         for i in range(batch):
###             normal = g[ i ][ node//go_size ]
###             v_float, v_uint = bitChange(normal, bit, sa01)
###             g[ i ][ node//go_size ] = np.float32(v_float)
###         if 0:
###             print("{:8d} faultpattern={}".format(var.n, var.faultpat[var.n]))
###             print(' '*8, np.max(_out.data), '=>', np.max(g), np.min(_out.data), '=>', np.min(g))
###         _out.data = g
### 
### 
### # layer-3 hook
### def lz3_Linear(_in,_out):
###     if var.n<0: # No fault injection for Normal System case
###         assert _in.args[0].dtype == np.float32, 'Unsupport input type {}'.format(_out.dtype)
###         assert _out.dtype        == np.float32, 'Unsupport out   type {}'.format(_out.dtype)
###         return
###     batch, in_size = _in.args[0].shape
###     batch, go_size = _out.data.shape
###     detect_flag, layer, node, bit, sa01 = var.faultpat[var.n]
###     if layer == 2:
###         # Update _in with sa01
###         g = _in.args[0].data.copy()
###         for i in range(batch):
###             normal = g[ i ][ node%in_size ]
###             v_float, v_uint = bitChange(normal, bit, sa01)
###             g[ i ][ node%in_size ] = np.float32(v_float)
###         if 0:
###             print("{:8d} faultpattern={}".format(var.n, var.faultpat[var.n]))
###             print(' '*8, np.max(_in.args[0]), '=>', np.max(g), np.min(_in.args[0]), '=>', np.min(g))
###         _in.args[0].data = g
### 
###         # Calculate Linear Layer after fault insertion
###         this_linear_out = linear(_in.args[0], _in.link.__dict__['W'], _in.link.__dict__['b'])
###         this_linear_out = this_linear_out.reshape(_out.data.shape)
###         _out.data = this_linear_out.data
### 
###     if layer == 3:
###         g = _out.data.copy()
###         for i in range(batch):
###             normal = g[ i ][ node//go_size ]
###             v_float, v_uint = bitChange(normal, bit, sa01)
###             g[ i ][ node//go_size ] = np.float32(v_float)
###         if 0:
###             print("{:8d} faultpattern={}".format(var.n, var.faultpat[var.n]))
###             print(' '*8, np.max(_out.data), '=>', np.max(g), np.min(_out.data), '=>', np.min(g))
###         _out.data = g
### 
### 
