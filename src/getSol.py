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

pch = 0

thispath = op.abspath(op.dirname(__file__))
os.chdir(thispath)
toppath = op.dirname(thispath)
pypath = op.join(toppath, "runtools")
binpath = op.join(thispath, "bin")
resultpath = op.join(toppath, "results")
rawresultpath = op.join(thispath, "rslts") 

sys.path.append(pypath)
import result_help as rh

prob=["Euler", "Heat", "Const"]

if not pch:
    sp = (2, 2)
else:
    sp = (1, 1)

kj = prob[pch]

dm, lemmesee = rh.jmerge(rawresultpath, kj)
print(dm.keys())
meta = {}
if "meta" in dm.keys():
    meta[kj] = dm.pop("meta")
    
jdf = rh.Solved(dm)
fg, axi = plt.subplots(sp[0], sp[1])
jdf.metaparse(meta)
jdf.plotResult(fg, axi)
jdf.savePlot(fg, resultpath, shw=True)
dff = jdf.ddf
ddfk = list(dff.keys())
dsam = dff[ddfk[0]]
dsk = dsam.columns.values.tolist()
                
