"""
    Parse the json and make some output.
"""
import os
import os.path as op
import sys
import pandas as pd
import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt
import shlex

thispath = op.abspath(op.dirname(__file__))
os.chdir(thispath)
toppath = op.dirname(thispath)
pypath = op.join(toppath, "runtools")
binpath = op.join(thispath, "bin")
resultpath = op.join(toppath, "results")
rawresultpath = op.join(thispath, "rslts") 

sys.path.append(pypath)
import result_help as rh
import main_help as mh

def jmerge(pth, prob):
    mys = os.listdir(pth)
    thesef = [op.join(pth, k) for k in mys if prob in k and k.startswith("s")]
    dic = dict()
    for t in thesef:
        mdb = rh.readj(t)
        dic.update(mdb)
        print(dic.keys())
    return dic
bat=False

if bat:
    kj = "Euler"
    sp = (2, 2)
else:
    kj = "Heat"
    sp = (1, 1)

# f = op.join(thispath, "solutes.json")

dm = jmerge(rawresultpath, kj)
print(dm.keys())
meta = {}
if "meta" in dm.keys():
    meta[kj] = dm.pop("meta")
    
jdf = rh.Solved(dm)
fg, axi = plt.subplots(sp[0], sp[1])
jdf.metaparse(meta)
jdf.plotResult(fg, axi)
jdf.savePlot(fg, resultpath)
    
                