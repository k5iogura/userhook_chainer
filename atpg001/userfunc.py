import numpy as np
import bitstring
from pdb import *
from chainer.functions.connection.linear import linear

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

# layer-0/1 hook
# Insert a fault in both _in and _out
def lx1_Linear(_in,_out):
    # _in.args[0].shape : batch, in_size, in_size, channels
    # _out.shape        : batch, go_size
    batch, in_size, in_size, channels = _in.args[0].shape
    batch, go_size                    = _out.data.shape

    if var.n<0: # No fault injection for Normal System case check only
        assert _in.args[0].dtype == np.float32, 'Unsupport input type {}'.format(_out.dtype)
        assert _out.dtype        == np.float32, 'Unsupport out   type {}'.format(_out.dtype)
        return

    detect_flag, layer, node, bit, sa01 = var.faultpat[var.n]

    if layer == 0:
        # Update _in with sa01
        g = _in.args[0].copy()
        for i in range(batch):
            normal = g[ i ][ node//in_size ][ node%in_size ][0]
            v_float, v_uint = bitChange(normal, bit, sa01)
            g[ i ][ node//in_size ][ node%in_size ][0] = np.float32(v_float)
        if 0:
            print("{:8d} faultpattern={}".format(var.n, var.faultpat[var.n]))
            print(' '*8, np.max(_in.args[0]), '=>', np.max(g), np.min(_in.args[0]), '=>', np.min(g))
        _in.args[0].data = g

        # Calculate Linear Layer after fault insertion
        this_linear_out = linear(_in.args[0], _in.link.__dict__['W'], _in.link.__dict__['b'])
        this_linear_out = this_linear_out.reshape(_out.data.shape)
        _out.data = this_linear_out.data

# layer-2 hook
def ly2_Linear(_in,_out):
    if var.n<0: # No fault injection for Normal System case
        assert _in.args[0].dtype == np.float32, 'Unsupport input type {}'.format(_out.dtype)
        assert _out.dtype        == np.float32, 'Unsupport out   type {}'.format(_out.dtype)
        return
    batch, in_size = _in.args[0].shape
    batch, go_size = _out.data.shape
    detect_flag, layer, node, bit, sa01 = var.faultpat[var.n]
    if layer == 1:
        # Update _in with sa01
        g = _in.args[0].data.copy()
        for i in range(batch):
            normal = g[ i ][ node%in_size ]
            v_float, v_uint = bitChange(normal, bit, sa01)
            g[ i ][ node%in_size ] = np.float32(v_float)
        if 0:
            print("{:8d} faultpattern={}".format(var.n, var.faultpat[var.n]))
            print(' '*8, np.max(_in.args[0]), '=>', np.max(g), np.min(_in.args[0]), '=>', np.min(g))
        _in.args[0].data = g

        # Calculate Linear Layer after fault insertion
        this_linear_out = linear(_in.args[0], _in.link.__dict__['W'], _in.link.__dict__['b'])
        this_linear_out = this_linear_out.reshape(_out.data.shape)
        _out.data = this_linear_out.data

# layer-3 hook
def lz3_Linear(_in,_out):
    if var.n<0: # No fault injection for Normal System case
        assert _in.args[0].dtype == np.float32, 'Unsupport input type {}'.format(_out.dtype)
        assert _out.dtype        == np.float32, 'Unsupport out   type {}'.format(_out.dtype)
        return
    batch, in_size = _in.args[0].shape
    batch, go_size = _out.data.shape
    detect_flag, layer, node, bit, sa01 = var.faultpat[var.n]
    if layer == 2:
        # Update _in with sa01
        g = _in.args[0].data.copy()
        for i in range(batch):
            normal = g[ i ][ node%in_size ]
            v_float, v_uint = bitChange(normal, bit, sa01)
            g[ i ][ node%in_size ] = np.float32(v_float)
        if 0:
            print("{:8d} faultpattern={}".format(var.n, var.faultpat[var.n]))
            print(' '*8, np.max(_in.args[0]), '=>', np.max(g), np.min(_in.args[0]), '=>', np.min(g))
        _in.args[0].data = g

        # Calculate Linear Layer after fault insertion
        this_linear_out = linear(_in.args[0], _in.link.__dict__['W'], _in.link.__dict__['b'])
        this_linear_out = this_linear_out.reshape(_out.data.shape)
        _out.data = this_linear_out.data

    if layer == 3:
        g = _out.data.copy()
        for i in range(batch):
            normal = g[ i ][ node//go_size ]
            v_float, v_uint = bitChange(normal, bit, sa01)
            g[ i ][ node//go_size ] = np.float32(v_float)
        if 0:
            print("{:8d} faultpattern={}".format(var.n, var.faultpat[var.n]))
            print(' '*8, np.max(_out.data), '=>', np.max(g), np.min(_out.data), '=>', np.min(g))
        _out.data = g


