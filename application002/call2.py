from distutils.dir_util import copy_tree
from shutil import rmtree
import os,sys,argparse

#import subprocess
#from replace_sample import replace_sample
# from replace_userfunc import replace_userfunc

import sample2
from userfunc_var import VAR
var = VAR()

args = argparse.ArgumentParser()
args.add_argument('-i','--images',type=int,default=20)
args.add_argument('-f','--faults',type=int,default=784, dest='faults')
args.add_argument('-F','--Faults',type=int,nargs='+')
args = args.parse_args()

faultNo_list = args.Faults if args.Faults is not None else range(args.faults)
data_P       = args.images
print("Run on Faults as",faultNo_list)

from mylist_block import mylist

d=mylist()
#print(d)
#for k in range(784):              # k:fault pattern No.
for k in faultNo_list:

    var.n = k
#    if(k!=0):
#        replace_userfunc(k)

    print("mylist=",d[k],k)   
#    for i in range(0,2):       # i:number of feeding image
        
#        if(i!=0):
#            replace_sample(i)
#    print("input picture no=" ,i)
#        subprocess.run("python3 sample.py",shell =True)
    sample2.infer(data_P)
    print('sample done')
#        rename =  "list" + str(k)+"_no-"+str(i)
    rename =  "list%d"%(k)
#    rename2 = "cp -fr dnn_params "+ rename
#    subprocess.run(rename2,shell =True)
    rmtree(rename, ignore_errors=True)
    copy_tree("dnn_params", rename)
#    subprocess.run("cp sample_origin.py sample.py",shell =True)
        
#subprocess.run("cp userfunc_origin.py userfunc.py",shell =True)
#subprocess.run("python3 heatmap.py",shell =True)
