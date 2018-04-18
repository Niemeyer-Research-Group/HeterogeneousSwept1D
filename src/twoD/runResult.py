"""
    Parse the json and make show results of numerical solver.
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
spath = op.dirname(thispath)
toppath = op.dirname(spath)
pypath = op.join(toppath, "runtools")

sys.path.append(pypath)
import result_help as rh
from main_help import *

prob=["Euler", "Heat"]

def justPlot(eq, shw=True, save=False):
    kj = prob[eq]
    dm, met = rh.jmerge(trspath, kj)

    meta = {}
    if "meta" in dm.keys():
        meta[kj] = dm.pop("meta")
    
    if not eq:
        pj = (2, 2)
    else:
        pj = (1, 1)

    jdf = rh.Solved(dm)
    fg, axi = plt.subplots(pj[0], pj[1])
    jdf.metaparse(meta)
    jdf.plotResult(fg, axi)

    if shw==True:
        plt.show()
    
    if save==True:
        jdf.savePlot(fg, resultpath, shw=True)

    dff = jdf.ddf

    return dff
    
if __name__ == "__main__":

    rn = False

    if len(sys.argv) <2:
        pch = 0
        
    else:
        pch = int(sys.argv[1]) 
        sch = " " + sys.argv[2] + " " 
        extra = " " + " ".join(sys.argv[3:]) + " "
        rn = True

    kj = prob[pch]

    if rn:
        ex = op.join(tbinpath, kj.lower())
        km = [k for k in os.listdir(ttestpath) if kj.lower() in k]
        eqf = op.join(ttestpath, km[0])
        cout = runMPICUDA(ex, 8, sch, eqf + " ", outdir=trspath, eqopt=extra)

    print(cout)
    dfp = justPlot(pch)

