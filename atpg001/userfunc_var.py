import numpy as np
from pdb import set_trace

def GenFP(net_spec = ( 28*28, 50, 50, 10 ),bits = 32, sa01 = (1,0)):
    layers   = len(net_spec)

    faultpat = []
    print('* Falut Point Summaries')
    print('  layer nodes bits sa01')
    for layer,nodes in enumerate(net_spec):
        print('  {:5d}{:6d}{:5d}   {}'.format(layer,nodes,bits,sa01))
        for node in range(nodes):
            for bit in range(bits):
                for sa in sa01:
                    # faultpat: [0]detect_flag [1]layer [2]node [3]bit [4]sa01
                    faultpat.append([False, layer, node, bit, sa])
    return faultpat

class VAR:
    __os= 1  # 1:linux 0:Windows
    __n = 0

    @property
    def n(self):return VAR.__n
    @n.setter
    def n(self,val):VAR.__n=val

    @property
    def os(self):return VAR.__os
    @os.setter
    def os(self,val):VAR.__os=val

    batch    = 1024
    net_spec = (28*28,50,50,10)
    bit_spec = 32
    sa01     = (0,1)
    faultN   = np.sum(net_spec) * bit_spec * len(sa01)
    faultpat = GenFP(net_spec, bit_spec, sa01)
