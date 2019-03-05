#! /usr/bin/env python3
import numpy as np
from math import *
import sys,os,argparse
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
            param_max=np.max(param)
            if np.max(param)==0:param_max=1.0
            param = 255.*(param/param_max)
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

def vizlink(param,noext_path,IMG_FORM='PNG'):
    #param : NWHC
    param = erase_1(param)

    if param.ndim == 1:
        # W -> CWH
        w,h=wh_div(param.shape[-1])
        buff = np.zeros((w*h),dtype=np.float32)
        buff = param.reshape(-1)[:w*h]
        param= buff.reshape((1,w,h))

    elif param.ndim == 2:
        # CW -> CWH
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

    #param : CWH 3-dimension allways
    print("Visualization as NCWH",param.shape)

    canvas=[640,640]
    snipet= 32
    gW,gH = int(canvas[0]/snipet), int(canvas[1]/snipet)

    img_form=IMG_FORM.lower()

    next_canvas=1
    done_canvas=0
    while next_canvas==1:
        Canvas,next_canvas = make_canvas(
            param[done_canvas*gW*gH:],
            canvas=canvas,
            snipet=snipet
        )
        outfile = noext_path+'_'+str(done_canvas)+'.'+img_form
        Canvas.save(outfile, IMG_FORM)
        done_canvas+=1
        print("** saved", outfile, Canvas.size)
    return Canvas, done_canvas

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='visualizer')
    parser.add_argument('paramfile', nargs='+', type=str, help="npy files")
    parser.add_argument('-d','--dir', type=str, help="image saved dir")
    args = parser.parse_args()

    IMG_FORM = 'PNG'
    img_form = IMG_FORM.lower()
    done_canvas = 0
    for filename in args.paramfile:
        if os.path.exists(filename):
            noext_path    = os.path.splitext(filename)[0]
            outfile       = noext_path+'.'+img_form
            filename_base = os.path.basename(noext_path)
            if args.dir is not None:
                noext_path = os.path.join(args.dir,filename_base)
                outfile    = noext_path+'.'+img_form
                os.makedirs(args.dir,exist_ok=True)
            param = np.load(filename)
            print(filename,"Input Parameter Shape",param.shape)
            Canvas, done_canvas =vizlink(param,noext_path)
        else:
            print(filename,"Not found, skiped")

