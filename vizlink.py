import numpy as np
from math import *
import sys,os
from PIL import Image


def wh_sqrt(n):
    x=int(sqrt(n))
    if x*(x+1) > n:
        return x,x
    else:
        return x,x+1

def _wh_sqrt(n):
    x=1
    for i in range(n-1,2,-1):
        if (sqrt(n)%i)==0:
           x = i
           break
    return x

def wh_div(n):
    return wh_sqrt(n)
    x=_wh_sqrt(n)
    if x!=1:return x,x
    x=1
    y=1
    for i in range(2,int(n/5)+1):
        h = int(n//i)
        if h!=1:
            x=i
            y=h
    return x,y

def erase_1(param):
    if param.ndim>=2 and param.shape[0]==1:
        param = param.reshape(param.shape[1:])
    return param

paramfile = sys.argv[-1]
param = np.load('dnn_params/lx1_Linear_out.npy')
param = np.load('dnn_params/lx1_Linear_b.npy')
param = np.load('dnn_params/lx1_Linear_in.npy')
param = np.load('dnn_params/lx1_Linear_W.npy')

#param : NWHC
param = np.ones((64,64,401)) #for test
#param = np.zeros((21,11)) #for test
param = erase_1(param)

print("Original Shape",param.shape)
if param.ndim == 1:
    # W
    w,h=wh_div(param.shape[-1])
    buff = np.zeros((w*h),dtype=np.float32)
    buff = param.reshape(-1)[:w*h]
    param= buff.reshape((1,w,h))

elif param.ndim == 2:
    # CW
    w,h=wh_div(param.shape[-1])
    buff = np.zeros((param.shape[0],w,h),dtype=np.float32)
    for n in range(param.shape[0]):
        for x in range(w):
            for y in range(h):
                buff[n][x][y] = param[n][w*y+x]
    param = buff

elif param.ndim == 3:
    # WHC -> CWH
    w,h=param.shape[:2]
    param = param.transpose((2,0,1)) 

else:
    assert param.ndim <= 4 , 'not support'

#param : NCWH 4-dimension allways
print("Visualization as NCWH",param.shape)

def make_canvas(param,canvas=[640,640],snipet=32):

    sW,sH = snipet, snipet
    gW,gH = int(canvas[0]/snipet), int(canvas[1]/snipet)
    cW,cH = gW*snipet, gH*snipet
    Nimg  = param.shape[0]

    print("* canvas WH",cW,cH)
    print("* snipet WH",sW,sH)
    print("* grid   WH",gW,gH)
    print("* Images   ",Nimg)

    Canvas = Image.new('L',(cW,cH))
    image_cnt = next_canvas = 0
    for gy in range(gH):
        for gx in range(gW):
            #print("%8d%8d%8d%8d"%(gx,gy,gx*sW,gy*sH))
            param = 255.*(param/np.max(param))
            snipet_img = Image.fromarray(param[image_cnt].astype(np.uint8))
            snipet_img = snipet_img.resize((sW,sH))
            Canvas.paste(snipet_img,(gx*sW,gy*sH))
            image_cnt += 1
            if image_cnt == Nimg:
                image_cnt=0
                break
            elif image_cnt >= gW*gH:
                next_canvas=1
                break
        if image_cnt==0 or next_canvas==1:break
    return Canvas, next_canvas

canvas=[640,640]
snipet= 32
gW,gH = int(canvas[0]/snipet), int(canvas[1]/snipet)

next_canvas=1
done_canvas=0
img_form="PNG"
while next_canvas==1:
    Canvas,next_canvas = make_canvas(
        param[done_canvas*gW*gH:],
        canvas=canvas,
        snipet=snipet
    )
    done_canvas+=1
    img_file = "c_%d.%s"%(done_canvas,img_form.lower())
    Canvas.save(img_file,img_form)
    print("** saved",img_file,Canvas.size,next_canvas)

