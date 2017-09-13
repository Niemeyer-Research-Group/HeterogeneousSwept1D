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

makr = "nvcc solmain.cu jsoncpp.cpp -o ./bin/euler -gencode arch=compute_35,code=sm_35 -O3 -restrict -std=c++11 -I/usr/include/mpi -lmpi -Xcompiler -fopenmp -lm -w --ptxas-options=-v"
makr = shlex.split(makr)
prog = op.join(binpath, "euler")

mpiarg = "--bind-to socket "
eqspec = op.join(thispath, "sod.json")
schemes = ["C ", "S "]

if op.isfile(prog):

    sgrp = os.stat(prog)
    spp = sgrp.st_mtime
    prer = os.listdir(thispath)
    suff = [".h", "cpp", ".cu", ".sh"]
    prereq = [k for k in prer for n in suff if n in k]

    ftim = []
    for p in prereq:
        fg = os.stat(p)
        ftim.append(fg.st_mtime)

    ts = sum([fo > spp for fo in ftim])
    if ts:
        print("Making executable...")
        proc = sp.Popen(makr)
        sp.Popen.wait(proc)

else:
    print("Making executable...")
    proc = sp.Popen(makr)
    sp.Popen.wait(proc)


#Say iterate over gpuA at one size and tpb
gpus = [k/2.0 for k in range(1, 11)]
prog += " "
eqspec += " "
nX = [2**k for k in range(11,21,2)]

for n in nX:
    for g in gpus:
        exargs = "dt 1e-6 gpuA {:.4f} nX {:d}".format(g, n)
        mh.runMPICUDA(prog, 1, schemes[0], eqspec, mpiopt=mpiarg, eqopt=exargs, timefile="timer.json ")


