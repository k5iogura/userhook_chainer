import numpy as np
import bitstring
from pdb import *
from chainer.functions.connection.linear import linear
from block_error import func
from mylist_block import mylist

from userfunc_var import VAR
var = VAR()

def ly2_Linear(_in,_out):
    if var.n<0: return  # No fault injection for Normal System case
    num = 1

    # fault injection for 50 hidden layer
    Nimages, Nunits = _in.args[0].data.shape
    for b in range(Nimages):
        _in.args[0].data[b][var.n] = 1.

    my_linear_out = linear(_in.args[0], _in.link.__dict__['W'], _in.link.__dict__['b'])

    my_linear_out = my_linear_out.reshape(_out.data.shape)
    _out.data = my_linear_out.data

def lx1_Linear(_in,_out):
    pass


def lz3_Linear(_in,_out):
    pass
