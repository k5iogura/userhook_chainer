#! /usr/bin/env python3
import os,sys,argparse
import numpy as np

args = argparse.ArgumentParser()
args.add_argument('file',type=str)
args = args.parse_args()

npy = np.load(args.file)
for i in npy:
    print(i)

