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
twopath = op.dirname(thispath)
spath = op.dirname(twopath)
toppath = op.dirname(spath)
pypath = op.join(toppath, "runtools")
testResult = op.join(thispath, "testResult")
testBin = op.join(thispath, "bin")
mainBin = op.join(spath, "bin")
utilBin = op.join(mainBin, "util")
utilInc = op.join(spath, "utilities")
os.mkdir(testResult)
os.mkdir(testBin)

sys.path.append(pypath)
import result_help as rh
from main_help import *

def runstring(toRun):
    compstr = shlex.split(toRun)
    proc = sp.Popen(compstr)
    sp.Popen.wait(proc)
    return 

# COMPILATION
if __name__ == "__main__":

    testobj = op.join(testBin, "waveTest.o")

    CUDAFLAGS       =" -gencode arch=compute_35,code=sm_35 -restrict         --ptxas-options=-v -I" + utilInc   
    CFLAGS          =" -O3 --std=c++11 -w "
    LIBFLAGS        =" -lm -lmpi "

    compileit = "nvcc testeq.cpp -o " + testobj + CFLAGS + CUDAFLAGS + LIBFLAGS

    runstring(toRun)

    utilObj = [op.join(utilBin, k) for k in os.listdir(utilBin)]
    execf = op.join(testBin, "waveTest")

    linkit = "nvcc " + utilObj + testobj + " -o " + execf + LIBFLAGS

    runstring(toRun)

    runTest = "mpirun -np 8 " + execf + " I waveTest.json " + testResult
    runstring(runTest)







