import os, sys
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
    __n = 0
  #  faultpat = None
  #  faultN   = 1

    __enable_user_generator = False
    __rnd_generator_option  = {
        'batch':1024, 'img_hw':28, 'img_ch':1,
        'X':255., 'pos_only':False, 'u8b':0, 'onehot':0
    }
    __pi = None
    __po = None
    __PIpat = None
    __POpat = None

    @property
    def n(self):return VAR.__n
    @n.setter
    def n(self,val):VAR.__n=val

    @property
    def faultN(self):return VAR.faultN
    @faultN.setter
    def faultN(self,val):VAR.faultN=val

    @property
    def faultpat(self):return VAR.faultpat
    @faultpat.setter
    def faultpat(self,val):VAR.faultpat=val

    @property
    def enable_user_generator(self):return VAR.__enable_user_generator
    @enable_user_generator.setter
    def enable_user_generator(self,val):VAR.__enable_user_generator=val

    @property
    def rnd_generator_option(self):return VAR.__rnd_generator_option
    @rnd_generator_option.setter
    def rnd_generator_option(self,val):VAR.__rnd_generator_option=val

    @property
    def pi(self):return VAR.__pi
    @pi.setter
    def pi(self,val):VAR.__pi=val

    @property
    def po(self):return VAR.__po
    @po.setter
    def po(self,val):VAR.__po=val

    @property
    def PIpat(self):return VAR.__PIpat
    @PIpat.setter
    def PIpat(self,val):VAR.__PIpat=val

    @property
    def POpat(self):return VAR.__POpat
    @POpat.setter
    def POpat(self,val):VAR.__POpat=val

    def init(self, Batch=1024, Net_spec=(28*28, 14*14*32, 7*7*64, 300, 10), Bit_spec=32, Sa01=(0,1), target=None, repro=None):
        VAR.batch    = Batch
        VAR.net_spec = Net_spec
        VAR.bit_spec = Bit_spec
        VAR.sa01     = Sa01
        if repro is not None:
            B = np.load(repro, allow_pickle=True)
            faultpat = []
            for b in B:
                L = b[0].tolist() if type(b[0]) is np.ndarray else b[0]
                L.insert(0, False)
                faultpat.append(L)
            VAR.faultpat = np.asarray(faultpat)
            VAR.faultN   = len(VAR.faultpat)
            print(VAR.faultN,'faults\n',VAR.faultpat)
            return

        if target is None:
            VAR.faultN   = np.sum(VAR.net_spec) * VAR.bit_spec * len(VAR.sa01)
            VAR.faultpat = GenFP(VAR.net_spec, VAR.bit_spec, VAR.sa01)
            return

        if target is not None:
            targetFaults = np.load(target, allow_pickle=True)
            print('Loading Specified Falut List from {} total {} faults'.format(target, len(targetFaults)))
            VAR.faultN   = len(targetFaults)
            targetFaultsList = [ i for i in targetFaults.tolist() ]
            detect_flag_idx  = 0
            for i in targetFaultsList: i.insert( detect_flag_idx, False )
            VAR.faultpat = np.asarray(targetFaultsList)
            print(VAR.faultpat)
            return

