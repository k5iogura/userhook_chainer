import numpy as np
import bitstring
from pdb import *
from chainer.functions.connection.linear import linear
from block_error import func

from userfunc_var import VAR, GenFP
var = VAR()

# Change fault bit in float32
import ctypes
class __f2i_union(ctypes.Union):
    _fields_=[('float',ctypes.c_float),('uint',ctypes.c_uint)]

def bitChange(v,bit,sa01):
    f2i_union = __f2i_union(v)
    t01 = 0 if ( f2i_union.uint & (0x01<<bit) )==0 else 1
    if sa01 == 1:
        f2i_union.uint = f2i_union.uint | (0x01<<bit)
    else:
        f2i_union.uint = f2i_union.uint & ~(0x01<<bit)
    return f2i_union.float, f2i_union.uint

# layer-0 hook
# Insert a fault in both _in and _out
def lx1_Linear(_in,_out):
    # _in.args[0].shape : batch, in_size, in_size, channels
    # _out.shape        : batch, go_size
    batch, in_size, in_size, channels = _in.args[0].shape

    if var.n<0: # No fault injection for Normal System case check only
        assert _in.args[0].dtype == np.float32, 'Unsupport input type {}'.format(_out.dtype)
        assert _out.dtype        == np.float32, 'Unsupport out   type {}'.format(_out.dtype)
        return

    detect_flag, layer, node, bit, sa01 = var.faultpat[var.n]
    if layer != 0: return # I'm not in this layer fault injection

    # Update _in with sa01
    g = _in.args[0].copy()
    for i in range(batch):
        normal = g[ i ][ node//in_size ][ node%in_size ][0]
        v_float, v_uint = bitChange(normal, bit, sa01)
        g[ i ][ node//in_size ][ node%in_size ][0] = np.float32(v_float)
        #set_trace()
    if 0:
        print("{:8d} faultpattern={}".format(var.n, var.faultpat[var.n]))
        print(' '*8, np.max(_in.args[0]), '=>', np.max(g), np.min(_in.args[0]), '=>', np.min(g))
    _in.args[0].data = g

    # Calculate Linear Layer after fault insertion
    this_linear_out = linear(_in.args[0], _in.link.__dict__['W'], _in.link.__dict__['b'])
    this_linear_out = this_linear_out.reshape(_out.data.shape)
    _out.data = this_linear_out.data

# layer-1 hook
def ly2_Linear(_in,_out):
    if var.n<0: # No fault injection for Normal System case
        assert _in.args[0].dtype == np.float32, 'Unsupport input type {}'.format(_out.dtype)
        assert _out.dtype        == np.float32, 'Unsupport out   type {}'.format(_out.dtype)
        return

# layer-2 hook
def lz3_Linear(_in,_out):
    if var.n<0: # No fault injection for Normal System case
        assert _in.args[0].dtype == np.float32, 'Unsupport input type {}'.format(_out.dtype)
        assert _out.dtype        == np.float32, 'Unsupport out   type {}'.format(_out.dtype)
        return

