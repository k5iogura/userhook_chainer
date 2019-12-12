from distutils.dir_util import copy_tree
from shutil import rmtree
import os,sys,argparse

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
for k in faultNo_list:

    var.n = k

    print("mylist=",d[k],k)   
    sample2.infer(data_P)
    print('sample done')
    rename =  "list%d"%(k)
    rmtree(rename, ignore_errors=True)
    copy_tree("dnn_params", rename)
