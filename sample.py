import chainer
import chainer.computational_graph as c
from chainer import serializers
from userhook import UserHook
import chainer.functions as F
#from chainer.function_hooks import TimerHook as fTH
#from Timer import TimerHook as fTH
from train import *
from pdb import *

model=NeuralNet(50,10)
serializers.load_npz('mnist.npz',model)

#x = np.zeros((1,28,28,1), dtype=np.float32)
_, test = chainer.datasets.get_mnist()
txs, tts = test._datasets
x = txs[0].reshape((1,28,28,1))

#fth = fTH()
hook = UserHook()
with chainer.using_config('train',False):
    with hook:
        p = model(x)

ans = F.argmax(F.softmax(p))
idx = int(ans.data)
print("ans=",idx,p,p.shape)
#set_trace()
#print(len(p))
#print(p.shape)
#print(p,type(p))
#print(type(p.data),p.data[0].shape)
for obj in p:
    if isinstance(obj,chainer.variable.Variable):
#        obj=obj.node
#        print(obj.name,obj.label,obj.rank)
       #print(obj.debug_print())
        pass

#g = c.build_computational_graph([p])
#with open('a.dot','w') as o:
#    o.write(g.dump())
hook.print_report()
#fth.print_report()
