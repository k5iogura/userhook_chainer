##############################################################
# This file was automatically generated by UserHook system.
# Copy this file as ./userfunc.py
# And edit it.
# You can get results to expect. Lucks, K.Ogura.
##############################################################
import numpy as np
from pdb import *
from chainer.functions.connection.linear import linear
def lx1_Linear(_in,_out):
    pass

def ly2_Linear(_in,_out):
    pass

def lz3_Linear(_in,_out):
    _out.data[0] = np.zeros(_out.data[0].shape)
    #print("_in",_in.args[0].shape)
    #_in.args = np.zeros(_in.args[0].shape,dtype=np.float32)
    print(_out.data[0].shape)
    my_linear_out = linear(_in.args[0], _in.link.__dict__['W'], _in.link.__dict__['b'])
    my_linear_out = my_linear_out.reshape(_out.data[0].shape)
    print(my_linear_out.shape)
    print(my_linear_out)
    set_trace()
    _out.data[0] = my_linear_out.data
    print(_out)
    pass

