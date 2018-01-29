'''
    Run experiment.
'''
#%%
import os
import os.path as op
import sys
import pandas as pd
import numpy as np
import subprocess as sp
import shlex

thispath = op.abspath(op.dirname(__file__))
os.chdir(thispath)
spath = op.dirname(thispath)
toppath = op.dirname(spath)
pypath = op.join(toppath, "runtools")

sys.path.append(pypath)

from main_help import *
import result_help as rh
import timing_help as th

ext = ".json"

eq = ["heat", "euler"]


#%%
nproc = 8
mpiarg = "" #"--bind-to socket
tstrng = os.listdir(otestpath)
schemes = ["S", "C"]
schD = {schemes[0]: "Classic", schemes[1]: "Swept"}

#%%
#Say iterate over gpuA at one size and tpb
gpus = [k/1.5 for k in range(1, 15)] #GPU Affinity
nX = [2**k for k in range(12,21)] #Num spatial pts (Grid Size)
tpb = [2**k for k in range(6,10)]
ac = 1

#%%
for p in eq:
    prog = op.join(binpath, p)
    prog += " "
    eqs = [k for k in tstrng if p.upper() in k.upper()]
    eqspec = op.join(otestpath, eqs[0])
    eqspec += " "
    for sc in schemes:
        timeTitle = "t"+p.title()+sc+".csv"
        eqn = p + " " + schD[sc]
        print(timeTitle + eqn)
        if not ac:
            ac += 1
            break

        sc += " "

        for t in tpb:
            for n in nX:
                xl = int(n/10000) + 1
                for g in gpus:
                    exargs =  " freq 200 gpuA {:.4f} nX {:d} tpb {:d} lx {:d}".format(g, n, t, xl)
                    runMPICUDA(prog, nproc, sc, eqspec, mpiopt=mpiarg, eqopt=exargs, outdir=orspath)

        tfile = op.join(orspath, timeTitle)
        res = th.Perform(tfile) #Not exactly right.
        res.plotdict(eqn, plotpath=resultpath)
