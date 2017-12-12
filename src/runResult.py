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
toppath = op.dirname(thispath)
pypath = op.join(toppath, "runtools")
binpath = op.join(thispath, "bin")
resultpath = op.join(toppath, "results")
rawresultpath = op.join(thispath, "rslts")
eqfpath = op.join(thispath, "tests")

sys.path.append(pypath)
import result_help as rh
import main_help as mh

prob=["Euler", "Heat", "Const"]

if len(sys.argv) <2:
    pch = 0
    rn = False
else:
    pch = int(sys.argv[1])
    sch = " " + sys.argv[2] + " "
    extra = " " + " ".join(sys.argv[3:]) + " "
    rn = True

if not pch:
    sp = (2, 2)
else:
    sp = (1, 1)

kj = prob[pch]

if rn:
    ex = op.join(binpath, kj.lower())
    km = [k for k in os.listdir(eqfpath) if kj.lower() in k]
    eqf = op.join(eqfpath, km[0])
    mh.runMPICUDA(ex, 8, sch, eqf + " ", outdir=rawresultpath, eqopt=extra)

dm, lemmesee = rh.jmerge(rawresultpath, kj)

meta = {}
if "meta" in dm.keys():
    print(dm["meta"])
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
