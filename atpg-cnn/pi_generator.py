from pdb import set_trace
import numpy as np
from userfunc_var import VAR
from random import random

# for sharing Class variables
var = VAR()

# Set PI layer name
# if var.pi is None then nothing to do about PI
#
# For MNIST FC Task
var.pi = 'lx1_Linear'           # layer = 0
var.pi = 'ly2_Linear'           # layer = 1
var.pi = 'lz3_Linear'           # layer = 2
# For MNIST CNN Task
var.pi = 'conv1_Convolution2D'  # layer = 0
var.pi = 'conv2_Convolution2D'  # layer = 1
var.pi = 'l1_Linear'            # layer = 2
var.pi = 'l2_Linear'            # layer = 3
#
var.pi = None

# pi_generator()
# User Test Pattern Genenator for var.pi layer
# Set up sending data to chainer hook
def pi_generator(data):
    shape = data.shape
    ones = np.ones(shape,dtype=np.float32)
#    zeros= np.zeros(shape,dtype=np.float32)
#    rndV = np.asarray([random() for i in range(np.prod(data.shape))],dtype=np.float32).reshape(shape)
    return ones

