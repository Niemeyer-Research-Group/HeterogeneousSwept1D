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
rawresultpath = op.join(thispath, "rslts") 

sys.path.append(pypath)

from main_help import *
import result_help as rh
import timing_help as th

eq = "euler"

#makr = "nvcc solmain.cu jsoncpp.cpp -o ./bin/euler -gencode arch=compute_35,code=sm_35 -O3 -restrict -std=c++11 -I/usr/include/mpi -lmpi -Xcompiler -fopenmp -lm -w --ptxas-options=-v"
#makr = shlex.split(makr)
prog = op.join(binpath, eq)

nproc = 8
mpiarg = "" #"--bind-to socket "
eqspec = op.join(thispath, "sod.json")
schemes = ["C ", "S "]
schD = {schemes[0]: "Classic", schemes[1]: "Swept"}

#if op.isfile(prog):
#
#    sgrp = os.stat(prog)
#    spp = sgrp.st_mtime
#    prer = os.listdir(thispath)
#    suff = [".h", "cpp", ".cu", ".sh"]
#    prereq = [k for k in prer for n in suff if n in k]
#
#    ftim = []
#    for p in prereq:
#        fg = os.stat(p)
#        ftim.append(fg.st_mtime)
#
#    ts = sum([fo > spp for fo in ftim])
#    if ts:
#        print("Making executable...")
#        proc = sp.Popen(makr)
#        sp.Popen.wait(proc)
#
#else:
#    print("Making executable...")
#    proc = sp.Popen(makr)
#    sp.Popen.wait(proc)


#Say iterate over gpuA at one size and tpb
gpus = [k/2.0 for k in range(1, 11)] #GPU Affinity
prog += " "
eqspec += " "
nX = [2**k for k in range(11,21,2)] #Num spatial pts (Grid Size)
#tpb 

for s in schemes:
    for n in nX:
        for g in gpus:
            exargs = "dt 1e-6 gpuA {:.4f} nX {:d}".format(g, n)
            runMPICUDA(prog, nproc, s, eqspec, mpiopt=mpiarg, eqopt=exargs, outdir=rawresultpath)

    rd = os.listdir(rawresultpath)
    
    for f in rd:
        if f.startswith("t") and s in f:
            tfile = op.join(rawresultpath, f)
            break
    
    res = th.Perform(tfile)
    eqn = eq + " " + schD[s]
    res.plotframe(resultpath, eqn)
    
    