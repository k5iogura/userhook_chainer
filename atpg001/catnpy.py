#! /usr/bin/env python3
import os,sys,argparse
import numpy as np

args = argparse.ArgumentParser()
args.add_argument('file',type=str)
args.add_argument('-n',  type=int, default=None)
args = args.parse_args()

npy = np.load(args.file)
for ( No,i ) in enumerate(npy):
    if args.n is not None and args.n==No:
        print(i)
        break
    else:
        print(i)

