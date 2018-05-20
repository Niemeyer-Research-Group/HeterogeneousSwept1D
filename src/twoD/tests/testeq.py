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

sys.path.append(pypath)
# import result_help as rh
# from main_help import *

def runstring(toRun):
    compstr = shlex.split(toRun)
    proc = sp.Popen(compstr)
    rc = sp.Popen.wait(proc)
    if rc == 1:
        sys.exit(1)
    return True

# COMPILATION
if __name__ == "__main__": 
    compileq = 1
    try:
        compileq = int(sys.argv[1])
    except:
        pass

    os.makedirs(testResult, exist_ok=True)
    os.makedirs(testBin, exist_ok=True)

    testobj = op.join(testBin, "waveTest.o")

    CUDAFLAGS       =" -gencode arch=compute_35,code=sm_35 -DNOS -restrict --ptxas-options=-v -I" + utilInc
    CFLAGS          =" --std=c++14 -w -g -I/usr/include/python2.7 "
    LIBFLAGS        =" -lm -lpython2.7 -lmpi "

    compileit = "nvcc -c testeq.cu -o " + testobj + CFLAGS + CUDAFLAGS + LIBFLAGS
    utilObj = [op.join(utilBin, k) for k in os.listdir(utilBin)]
    execf = op.join(testBin, "waveTest")
    utilObj = " ".join(utilObj)
    linkit = "nvcc " + utilObj + " " + testobj + " -o " + execf + LIBFLAGS

    if compileq:
        print (compileit)

        runstring(compileit)

        print("   ---------------")
        print("Compiled")

        runstring(linkit)

        print("   ---------------")
        print("Linked")

    runTest = "mpirun -np 6 -report-pid - " + execf + " S waveTest.json " + testResult + " gpuA 0.0 tf 0.1 dt 0.0005 gridSize 65336 "
    print(runTest)
    runstring(runTest)
    print("   ---------------")







