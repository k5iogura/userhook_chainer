import subprocess
from replace_sample import replace_sample
from replace_userfunc import replace_userfunc

#from distutils.dir_util import copy_tree
#import shutil
from mylist_block import mylist

#パラメータの変更(壊す場所の変更)＠userfunc.py

d=mylist()
#print(d)
for k in range(3):

    if(k!=0):
        replace_userfunc(k)
    print(d[k])   
    for i in range(0,2):
        
        if(i!=0):
            replace_sample(i)
        print("input picture no=" ,i)
        subprocess.run("python sample.py",shell =True)
        print('sample計算完了')
        rename =  "list" + str(k)+"_no"+str(i)
        #print(rename)
        rename2 = "xcopy dnn_params "+ rename+ "/C/I/Y/Q > nul"
        subprocess.run(rename2,shell =True)
    subprocess.run("copy sample_origin.py sample.py/Y > nul",shell =True)
        
subprocess.run("copy userfunc_origin.py userfunc.py/Y > nul",shell =True)
subprocess.run("python heatmap.py",shell =True)
