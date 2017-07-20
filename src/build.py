import os
import sys
import os.path as op
import subprocess as sp
import shlex 

# Preliminaries:  Path Stuff  This will work I swear.
projpath = op.abspath(op.dirname(__file__))
os.chdir(projpath)
outpath = op.join(projpath, 'bin')
mainpath = op.join(projpath, 'decomposition')
utilpath = op.join(projpath, 'utilities')
eqpath = op.join(projpath, 'equations')
dummyfile = op.join(eqpath, 'dummyheader.h')

eqdirs = next(os.walk(eqpath))[1]
apath = [op.join(eqpath, x) for x in eqdirs]
xeq = [op.join(outpath, x) for x in apath]

comp = "nvcc -o "

compflags =  " -gencode arch=compute_35,code=sm_35 -std=c++11 -restrict -O3"
includes = " -I" + utilpath + " -I" + eqpath + " -I/usr/include/mpi"
libs = " -Xcompiler -fopenmp -lm -lmpi"
prereq = " solver.cu"


for x, a in zip(xeq, apath):
    hfile = "#include <"
    for f in os.listdir(a):
        if ".h" in f:
            hfile += op.join(op.basename(a), f) + ">"
    
    with open(dummyfile, 'w') as ff:
        ff.write(hfile)

    outx = op.join(outpath, x)
    compStr = comp + x + prereq + libs + includes + compflags
    print(compStr)
    compSplit = shlex.split(compStr)
    proc = sp.Popen(compSplit)
    sp.Popen.wait(proc)