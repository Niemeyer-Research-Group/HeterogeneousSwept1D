'''
    Run experiment.
'''

import os
import os.path as op
import sys
import pandas as pd
import numpy as np
import subprocess as sp
import shlex

thispath = op.abspath(op.dirname(__file__))
os.chdir(thispath)
toppath = op.dirname(thispath)
pypath = op.join(toppath, "runtools")
binpath = op.join(thispath, "bin")
resultpath = op.join(thispath, "results")

sys.path.append(pypath)
import result_help as rh
import main_help as mh

makr = "./allone.sh"
prog = op.join(binpath, "euler")

mpiarg = "--bind-to-socket "
eqspec = op.join(thispath, "sod.json")
schemes = ["C ", "S "]

if op.isfile(prog):

    sgrp = os.stat(prog)
    spp = sgrp.st_mtime 
    prer = os.listdir(thispath)
    suff = [".h", "cpp", ".cu", ".sh"]
    prereq = [k for k in prer for n in suff if n in k]
    print(prereq)

    ftim = []
    for p in prereq:
        fg = os.stat(p)
        ftim.append(fg.st_mtime)

    ts = sum([fo > spp for fo in ftim])
    if ts:
        sp.call(makr)   

else:
    sp.call(makr) # The maker

#Say iterate over gpuA at one size and tpb
gpus = [k/2.0 for k in range(11)]
prog += " "

for g in gpus:
    exargs = "gpuA {:.4f} ".format(g) 
    mh.runMPICUDA(prog, 1, schemes[0], eqspec, mpiopt=mpiarg, eqopt=exargs)


