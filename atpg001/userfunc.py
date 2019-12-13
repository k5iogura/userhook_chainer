import numpy as np
import bitstring
from pdb import *
from chainer.functions.connection.linear import linear
from block_error import func

from userfunc_var import VAR
var = VAR()

def lx1_Linear(_in,_out):
    if var.n<0: return  # No fault injection for Normal System case
    num = 1
    print("fault spec:", var.n, var.faultpat[var.n] )
    detect_flag, layer, node, bit, sa = var.faultpat[var.n]
    if layer != 0: return # I'm not in this layer fault injection
#    x=d[n][0]
#    y=d[n][1]
#    g = func(_in.args[0], x, y, num, 1)
#    _in.args[0].data = g

#    my_linear_out = linear(_in.args[0], _in.link.__dict__['W'], _in.link.__dict__['b'])

#    my_linear_out = my_linear_out.reshape(_out.data.shape)
#    _out.data = my_linear_out.data
#    set_trace()

def ly2_Linear(_in,_out):
    set_trace()
    pass


def lz3_Linear(_in,_out):
    set_trace()
    pass
