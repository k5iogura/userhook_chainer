from distutils.dir_util import copy_tree
from shutil import copy2

#import subprocess
#from replace_sample import replace_sample
# from replace_userfunc import replace_userfunc

import sample                           # append
tsx, tts = sample.txs, sample.tts       # append
data_N   = len(tts)                     # append
from userfunc_var import VAR            # append
var = VAR()                             # append

#from distutils.dir_util import copy_tree
#import shutil
from mylist_block import mylist

d=mylist()
#print(d)
for k in range(0,3):              # k:fault pattern No.

    var.n = k
#    if(k!=0):
#        replace_userfunc(k)

    print("mylist=",d[k])   
    for i in range(0,2):       # i:number of feeding image
#    data_P = 10000              # data_P:amount of images a inference
#    data_P = 20
        
#        if(i!=0):
#            replace_sample(i)
#    print("input picture no=" ,i)
#        subprocess.run("python3 sample.py",shell =True)
        sample.infer(i)
        print('sample done')
#        rename =  "list" + str(k)+"_no-"+str(i)
        rename =  "list%d_no-%d"%(k, i)
#    rename2 = "cp -fr dnn_params "+ rename
#    subprocess.run(rename2,shell =True)
        copy_tree("dnn_params", rename)
#    subprocess.run("cp sample_origin.py sample.py",shell =True)
        
#subprocess.run("cp userfunc_origin.py userfunc.py",shell =True)
#subprocess.run("python3 heatmap.py",shell =True)
