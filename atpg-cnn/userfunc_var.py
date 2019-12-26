import numpy as np
from pdb import set_trace

def GenFP(net_spec = (28*28, 14*14*32, 7*7*64, 300, 10), bits = 32, sa01 = (1,0)):
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

    def init(self, Batch=1024, Net_spec=(28*28, 14*14*32, 7*7*64, 300, 10), Bit_spec=32, Sa01=(0,1), target=None):
        VAR.batch    = Batch
        VAR.net_spec = Net_spec
        VAR.bit_spec = Bit_spec
        VAR.sa01     = Sa01
        if target is None:
            VAR.faultN   = np.sum(VAR.net_spec) * VAR.bit_spec * len(VAR.sa01)
            VAR.faultpat = GenFP(VAR.net_spec, VAR.bit_spec, VAR.sa01)
        else:
            targetFaults = np.load(target)
            print('Loading Specified Falut List from {} total {} faults'.format(target, len(targetFaults)))
            VAR.faultN   = len(targetFaults)
            targetFaultsList = [ i for i in targetFaults.tolist() ]
            detect_flag_idx  = 0
            for i in targetFaultsList: i.insert( detect_flag_idx, False )
            VAR.faultpat = np.asarray(targetFaultsList)
            print(VAR.faultpat)

